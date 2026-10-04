"""
core/vlm_cls_explainer.py  -  VLM 图像分类的可解释迭代器（core/explainer.py 的图像版本）

和 core/llm_ft_explainer.py 是同一类情况——不强行复用 core/explainer.py（它的
explain() 需要一个 DataReport，里面 boundary_samples/augmented_count 这些字段
是文本增强流水线特有的概念，硬塞图像分类数据进去既别扭又不诚实）。这里用同一个
IterationExplanation/NextAction 外形（事件契约不变，前端渲染逻辑不用改），
val_metric 是真实的加权 F1（有界 0-1，越高越好），next_action 用完整的 5 值
词表——图像分类本质上和文本分类一样是"有标注数据、有验证集"的监督学习，
COLLECT_MORE_DATA/ADJUST_HYPERPARAMS 都是有意义的选项。
"""

from typing import List

from core.llm_client import LLMClient
from core.json_extract import extract_json

from config import TaskSpec, EpochResult, IterationExplanation, NextAction


SYSTEM_PROMPT = """你是一个图像分类训练顾问，负责向完全不懂机器学习的业务人员解释模型的训练进展。

你的分析风格：
- 用通俗语言，完全禁止技术术语（不说"loss"，说"错误程度"；不说"过拟合"，说"死记硬背了训练图片但认不出新图片"）
- 用具体比喻（类比学生看图识物、师傅带徒弟辨认样品等）
- 每条建议必须具体可操作
- 做出明确的下一步决策，不能含糊

我们用 F1 分数衡量进展，取值在 0 到 1 之间，越接近 1 越好。

输出 JSON 格式（只输出 JSON，不要其他内容）：
{
  "diagnosis": "一句话描述当前模型的表现（给非技术人员看）",
  "root_cause": "根本原因分析，必须用比喻（50字以内）",
  "recommendation": "1-2条具体可操作建议（用换行分隔）",
  "next_action": "continue_training" | "collect_more_data" | "adjust_hyperparams" | "stop_success" | "stop_plateau",
  "confidence": 0到1的数字,
  "hyperparam_delta": {}
}

next_action 选择标准：
- stop_success：F1 已经达到较高水平（比如 0.85 以上）且趋势稳定
- stop_plateau：连续几轮 F1 几乎不再变化，且水平不算高，陷入瓶颈
- collect_more_data：F1 偏低，且怀疑是每个类别的图片数量太少、覆盖不够全面导致的
- adjust_hyperparams：F1 停滞但怀疑是训练轮数不合适导致的，给出
  hyperparam_delta（可选键：num_epochs 整数）
- continue_training：F1 还在提升，或轮次还不够多
"""


class VlmClsExplainer:
    ACTION_MAP = {
        "continue_training":  NextAction.CONTINUE_TRAINING,
        "collect_more_data":  NextAction.COLLECT_MORE_DATA,
        "adjust_hyperparams": NextAction.ADJUST_HYPERPARAMS,
        "stop_success":       NextAction.STOP_SUCCESS,
        "stop_plateau":       NextAction.STOP_PLATEAU,
    }

    def __init__(self, client: LLMClient):
        self.client = client

    def explain(self, current: EpochResult, spec: TaskSpec, history: List[EpochResult]) -> IterationExplanation:
        prompt = self._build_prompt(current, spec, history)
        try:
            raw = self.client.complete(system=SYSTEM_PROMPT, user=prompt, max_tokens=1000)
            parsed = extract_json(raw)
            delta = parsed.get("hyperparam_delta") or {}
            return IterationExplanation(
                diagnosis      = parsed.get("diagnosis", ""),
                root_cause     = parsed.get("root_cause", ""),
                recommendation = parsed.get("recommendation", ""),
                next_action    = self.ACTION_MAP.get(
                    parsed.get("next_action", "continue_training"), NextAction.CONTINUE_TRAINING),
                confidence     = float(parsed.get("confidence", 0.5)),
                hyperparam_delta = delta if isinstance(delta, dict) else {},
            )
        except Exception:
            return self._rule_based_fallback(current, history)

    def _build_prompt(self, current: EpochResult, spec: TaskSpec, history: List[EpochResult]) -> str:
        recent = [round(e.val_metric, 4) for e in history[-5:]]
        return f"""请分析以下图像分类训练情况并给出诊断：

任务描述：{spec.raw_description}

=== 当前轮次（第 {current.epoch} 轮）===
F1 分数：{current.val_metric:.4f}
验证损失：{current.val_loss:.4f}

=== 历史趋势（过去几轮的 F1 分数）===
{recent}

请基于以上信息，给出你的诊断和建议。"""

    def _rule_based_fallback(self, current: EpochResult, history: List[EpochResult]) -> IterationExplanation:
        if current.val_metric >= 0.85:
            return IterationExplanation(
                diagnosis      = f"模型的 F1 分数已经达到 {current.val_metric:.2f}，表现不错",
                root_cause     = "就像学生已经能熟练认出这几类样品了",
                recommendation = "可以先试试实际效果是否满意；满意的话就到这里",
                next_action    = NextAction.STOP_SUCCESS,
                confidence     = 0.6,
                hyperparam_delta = {},
            )
        if len(history) >= 3:
            recent = [e.val_metric for e in history[-3:]]
            if max(recent) - min(recent) < 0.02:
                return IterationExplanation(
                    diagnosis      = f"最近几轮 F1 分数几乎不再变化（约 {current.val_metric:.2f}）",
                    root_cause     = "就像学生反复看同一批图片但认不出新图片，可能是图片数量或角度太单一",
                    recommendation = "补充更多不同角度/光线条件下的图片，每个类别尽量多样",
                    next_action    = NextAction.STOP_PLATEAU,
                    confidence     = 0.55,
                    hyperparam_delta = {},
                )
        return IterationExplanation(
            diagnosis      = f"模型仍在学习中，当前 F1 分数 {current.val_metric:.2f}",
            root_cause     = "训练处于正常进展阶段，还有提升空间",
            recommendation = "继续训练，观察下一轮的变化",
            next_action    = NextAction.CONTINUE_TRAINING,
            confidence     = 0.5,
            hyperparam_delta = {},
        )
