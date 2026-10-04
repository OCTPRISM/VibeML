"""
core/vlm_gen_explainer.py  -  生成式 VLM 的可解释迭代器（core/vlm_cls_explainer.py 的生成版本）

跟 core/vlm_cls_explainer.py 同样的理由不复用 core/explainer.py（DataReport 里的字段
是文本增强流水线特有的概念）。这里 val_metric 是 ROUGE-L F1（有界 0-1，越高越好，
跟分类的 F1 是同一类"有界指标"，只是含义换成了"生成的文字和参考答案有多像"），
next_action 用完整 5 值词表——有参考答案、有验证集，本质上还是监督学习场景。
"""

from typing import List

from core.llm_client import LLMClient
from core.json_extract import extract_json

from config import TaskSpec, EpochResult, IterationExplanation, NextAction


SYSTEM_PROMPT = """你是一个"看图说话/视觉问答"模型的训练顾问，负责向完全不懂机器学习的业务人员解释模型的训练进展。

你的分析风格：
- 用通俗语言，完全禁止技术术语（不说"loss"，说"错误程度"；不说"过拟合"，说"死记硬背了训练时看过的图片和答案，但换一张新图片就说不出像样的话"）
- 用具体比喻（类比学生看图写作文、学徒描述看到的东西等）
- 每条建议必须具体可操作
- 做出明确的下一步决策，不能含糊

我们用 ROUGE-L 分数衡量"模型生成的文字"和"参考答案"有多相似，取值在 0 到 1 之间，越接近 1 越好。

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
- stop_success：ROUGE-L 已经达到较高水平（比如 0.6 以上，生成式任务比分类任务天然更难做到接近满分）且趋势稳定
- stop_plateau：连续几轮 ROUGE-L 几乎不再变化，且水平不算高，陷入瓶颈
- collect_more_data：ROUGE-L 偏低，且怀疑是参考答案的图片/问题/答案多样性不够
- adjust_hyperparams：ROUGE-L 停滞但怀疑是训练轮数不合适导致的，给出
  hyperparam_delta（可选键：num_epochs 整数）
- continue_training：ROUGE-L 还在提升，或轮次还不够多
"""


class VlmGenExplainer:
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
        return f"""请分析以下生成式 VLM 训练情况并给出诊断：

任务描述：{spec.raw_description}

=== 当前轮次（第 {current.epoch} 轮）===
ROUGE-L 分数：{current.val_metric:.4f}
训练损失：{current.train_loss:.4f}

=== 历史趋势（过去几轮的 ROUGE-L 分数）===
{recent}

请基于以上信息，给出你的诊断和建议。"""

    def _rule_based_fallback(self, current: EpochResult, history: List[EpochResult]) -> IterationExplanation:
        if current.val_metric >= 0.6:
            return IterationExplanation(
                diagnosis      = f"模型生成的文字和参考答案的相似度（ROUGE-L）已经达到 {current.val_metric:.2f}，表现不错",
                root_cause     = "就像学生已经能看图写出和标准答案很接近的句子了",
                recommendation = "可以先试试实际生成效果是否满意；满意的话就到这里",
                next_action    = NextAction.STOP_SUCCESS,
                confidence     = 0.55,
                hyperparam_delta = {},
            )
        if len(history) >= 3:
            recent = [e.val_metric for e in history[-3:]]
            if max(recent) - min(recent) < 0.02:
                return IterationExplanation(
                    diagnosis      = f"最近几轮 ROUGE-L 分数几乎不再变化（约 {current.val_metric:.2f}）",
                    root_cause     = "就像学生反复看同一批图片和答案却写不出更贴切的句子，可能是样本的问法/答案风格太单一",
                    recommendation = "补充更多不同风格的图片、问题和参考答案，让模型见识更多样的表达方式",
                    next_action    = NextAction.STOP_PLATEAU,
                    confidence     = 0.5,
                    hyperparam_delta = {},
                )
        return IterationExplanation(
            diagnosis      = f"模型仍在学习中，当前 ROUGE-L 分数 {current.val_metric:.2f}",
            root_cause     = "训练处于正常进展阶段，还有提升空间",
            recommendation = "继续训练，观察下一轮的变化",
            next_action    = NextAction.CONTINUE_TRAINING,
            confidence     = 0.5,
            hyperparam_delta = {},
        )
