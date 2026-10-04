"""
core/explainer.py  -  Phase 1.4：可解释迭代器

差异化核心（文献验证的真实空白 #2）：
  几乎所有 agent 式 AutoML 系统都缺乏系统性机制来评估每个流水线阶段
  的中间决策——大多数只报告最终指标，不评估中间过程的有效性。

本模块做的事：
  不只告诉用户"F1 = 0.72"，而是解释：
    - 为什么现在是 0.72（根本原因诊断）
    - 用比喻让非专家理解（类比教练给选手分析比赛录像）
    - 下一步应该做什么（具体可操作的建议）
    - 系统自动决策：继续训练 / 收集更多数据 / 调整超参 / 停止

这是让"零门槛"真正可用的关键——用户看到的不是机器的数字，
而是一个懂他们任务的顾问给出的分析报告。
"""

import json
import re
from typing import List

from core.llm_client import LLMClient

from config import (
    TaskSpec, DataReport, EpochResult,
    IterationExplanation, NextAction
)


# ── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个 AI 训练顾问，负责向完全不懂机器学习的业务人员解释模型训练进展。

你的分析风格：
- 用通俗语言，完全禁止技术术语（不说"过拟合"，说"模型死记硬背了训练数据"）
- 用具体比喻（类比老师教学生、教练指导球队等）
- 每条建议必须具体可操作，不能说"改善数据质量"（太泛），要说"给'支付问题'这个类别再补充 20 条例子"
- 做出明确的下一步决策，不能含糊

输出 JSON 格式（只输出 JSON，不要其他内容）：
{
  "diagnosis": "一句话描述当前模型状态（给非技术人员看）",
  "root_cause": "根本原因分析，必须用比喻（50字以内）",
  "recommendation": "1-2条具体可操作建议（用换行分隔）",
  "next_action": "continue_training" | "collect_more_data" | "adjust_hyperparams" | "stop_success" | "stop_plateau",
  "confidence": 0到1的数字,
  "hyperparam_delta": {}
}

next_action 选择标准：
- stop_success：指标 > 0.85，或用户任务要求已满足
- stop_plateau：连续 3 轮指标变化 < 0.005，且指标 < 0.70
- collect_more_data：某个类别 F1 明显低（< 0.50），或有数据警告
- adjust_hyperparams：训练损失下降但验证指标不上升（轻微过拟合），或某类别持续被误判
- continue_training：一般情况，还有提升空间

当 next_action = "adjust_hyperparams" 时，hyperparam_delta 必须给出具体改动（否则留 {}）。

【所有后端通用】
- "class_boost": {"标签名": 倍数}，对表现弱的类别加权（倍数建议 1.2~1.5，最大 3.0）
  sklearn 后端体现为超额过采样，神经网络后端体现为加权采样，语义一致
- "loss_fn": 换损失函数，按下面这张"错误模式 → 损失函数"对照表选（只对神经网络后端生效）。
  ⚠ 这张表**修订过一次**，是实测结果，不是教科书直觉——改之前请先看 core/loss_factory.py
    里记的那组数字（中文情感数据 9:1 不平衡，600 训练 / 1500 均衡测试，5 种子）：
    · 类别不均衡（某类样本远少于其它类、且该类 F1 明显低）→ **"weighted_ce"**
      实测这是唯一稳定有效的：+0.21 macro-F1（在模型尚未过拟合的训练量下）。
      不要选 "focal"——单独的 focal 只按"难易"重加权，不按"类别频次"重加权，
      实测对不平衡几乎没用（+0.004，在噪声内）。
    · 已经用了 weighted_ce 但少数类仍然被大量漏判 → "focal" + 类别权重
      （给 "focal_gamma" 0.5~5.0，默认 2.0；系统会自动带上类别权重）
    · 标签噪声（数据质量分低、且有明确的错标证据）→ "label_smoothing"
      可同时给 "label_smoothing"（0.0~0.4，默认 0.1）。
      ⚠ 只在**类别大致均衡**时用：实测在不平衡数据上它反而更差（-0.015）。
    · 没有明确证据指向上面任何一种 → 不要给 loss_fn，保持默认交叉熵
  注意：换损失函数是有代价的（要重新训练一轮），只有在诊断出上面某种明确的
  错误模式时才给，不要因为"想试试"就换。

【仅 sklearn 后端】
- "C": 数字，logreg/svm 的正则强度（越小正则越强，用于缓解过拟合）
- "alpha": 数字，sgd 的正则强度（越大正则越强）
- "switch_model": "logreg" | "svm" | "sgd"，切换模型族（当前模型持续欠拟合或某类别系统性表现差时）
"""


# ── Explainer ─────────────────────────────────────────────────────────────────

class Explainer:
    """
    为每轮训练结果生成人类可读的解释和决策建议。

    用法：
        explainer = Explainer(client)
        explanation = explainer.explain(latest_epoch, data_report, task_spec, history)
    """

    ACTION_MAP = {
        "continue_training":  NextAction.CONTINUE_TRAINING,
        "collect_more_data":  NextAction.COLLECT_MORE_DATA,
        "adjust_hyperparams": NextAction.ADJUST_HYPERPARAMS,
        "stop_success":       NextAction.STOP_SUCCESS,
        "stop_plateau":       NextAction.STOP_PLATEAU,
    }

    def __init__(self, client: LLMClient):
        self.client = client

    def explain(
        self,
        current:   EpochResult,
        report:    DataReport,
        spec:      TaskSpec,
        history:   List[EpochResult],
        noise_note: str = "",
    ) -> IterationExplanation:
        """
        生成当前迭代的可读解释。

        Args:
            current: 最新一轮的训练结果
            report:  数据质量报告（来自 Phase 1.2）
            spec:    任务规格（来自 Phase 1.1）
            history: 之前所有轮次的结果

        Returns:
            IterationExplanation
        """
        prompt = self._build_prompt(current, report, spec, history, noise_note)

        try:
            raw = self.client.complete(system=SYSTEM_PROMPT, user=prompt, max_tokens=1000)
            return self._parse_response(raw, current)

        except Exception as e:
            # LLM 调用失败时，用规则 fallback
            return self._rule_based_fallback(current, history, report)

    # ── 私有方法 ─────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        current:  EpochResult,
        report:   DataReport,
        spec:     TaskSpec,
        history:  List[EpochResult],
        noise_note: str = "",
    ) -> str:
        """构建发给 LLM 的上下文 prompt"""

        # 趋势分析
        recent_metrics = [round(e.val_metric, 4) for e in history[-5:]]
        is_plateau = (
            len(recent_metrics) >= 3
            and max(recent_metrics[-3:]) - min(recent_metrics[-3:]) < 0.005
        )
        trend_note = "（过去3轮几乎无变化——可能已到瓶颈）" if is_plateau else ""

        # 最弱的类别
        weak_classes = [
            f"'{cls}' (F1={score:.2f})"
            for cls, score in sorted(current.per_class_metrics.items(), key=lambda x: x[1])
            if score < 0.60
        ]

        # 指标可信度提示：验证集小的时候，指标差几个点很可能只是抽样波动。
        # 不告诉 LLM 这件事的话，它会把噪声当成真实趋势，给出"继续这个方向"之类
        # 的错误建议，进而让整个自动调整循环追着噪声跑（见 core/robust_eval.py）
        noise_block = f"""
=== 指标可信度（重要）===
{noise_note}
注意：如果指标变化幅度小于上面的噪声水平，请**不要**把它当成真实的提升或下降，
也不要据此建议"继续当前方向"。这种情况下更应该建议补充数据（collect_more_data），
因为样本量不足才是根本问题。
""" if noise_note else ""

        return f"""请分析以下机器学习训练情况并给出诊断：

任务描述：{spec.raw_description}
业务领域：{spec.domain}
目标指标：{spec.evaluation_metric}

=== 数据情况 ===
总样本数：{report.total_samples}（其中增强 {report.augmented_count} 条）
数据质量评分：{report.quality_score:.2f}/1.00
数据警告：{', '.join(report.warnings) if report.warnings else '无'}
边界样本数：{len(report.boundary_samples)}（标注可能有争议的样本）

=== 当前轮次（第 {current.epoch} 轮）===
{spec.evaluation_metric}：{current.val_metric:.4f} {trend_note}
训练损失：{current.train_loss:.4f}
验证损失：{current.val_loss:.4f}

各类别 F1：
{json.dumps(current.per_class_metrics, ensure_ascii=False)}

主要混淆情况：
{chr(10).join(current.confusion_highlights) if current.confusion_highlights else '暂无明显混淆对'}

表现较弱的类别：{', '.join(weak_classes) if weak_classes else '暂无明显弱类别'}

=== 历史趋势（过去几轮的指标）===
{recent_metrics}
{noise_block}
请基于以上信息，给出你的诊断和建议。记住：用非技术人员能理解的语言。"""

    def _parse_response(self, raw: str, current: EpochResult) -> IterationExplanation:
        text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                return self._rule_based_fallback(current, [], None)

        delta = parsed.get("hyperparam_delta", {})
        if not isinstance(delta, dict):
            delta = {}

        return IterationExplanation(
            diagnosis      = parsed.get("diagnosis", ""),
            root_cause     = parsed.get("root_cause", ""),
            recommendation = parsed.get("recommendation", ""),
            next_action    = self.ACTION_MAP.get(
                parsed.get("next_action", "continue_training"),
                NextAction.CONTINUE_TRAINING
            ),
            confidence     = float(parsed.get("confidence", 0.5)),
            hyperparam_delta = delta,
        )

    def _rule_based_fallback(
        self,
        current:  EpochResult,
        history:  List[EpochResult],
        report:   DataReport | None,
    ) -> IterationExplanation:
        """当 LLM 不可用时的规则回退，保证闭环不中断"""
        m = current.val_metric

        # 瓶颈检测
        if len(history) >= 3:
            recent = [e.val_metric for e in history[-3:]]
            if max(recent) - min(recent) < 0.005 and m < 0.70:
                return IterationExplanation(
                    diagnosis      = f"模型在 {m:.2%} 附近停滞，连续多轮没有提升",
                    root_cause     = "就像学生做了很多相似题目后遇到了知识瓶颈，需要引入新类型的练习材料",
                    recommendation = "建议收集更多样化的训练样本，尤其是表现较弱的类别",
                    next_action    = NextAction.COLLECT_MORE_DATA,
                    confidence     = 0.6,
                )

        if m >= 0.85:
            return IterationExplanation(
                diagnosis      = f"模型表现优秀，{current.metric_name} 已达 {m:.2%}",
                root_cause     = "训练数据的质量和数量已经足够支撑这个任务",
                recommendation = "可以停止训练，准备部署",
                next_action    = NextAction.STOP_SUCCESS,
                confidence     = 0.85,
            )

        return IterationExplanation(
            diagnosis      = f"模型仍在学习中，{current.metric_name} = {m:.2%}",
            root_cause     = "训练处于正常进展阶段，还有提升空间",
            recommendation = "继续训练，观察下一轮的变化",
            next_action    = NextAction.CONTINUE_TRAINING,
            confidence     = 0.5,
        )
