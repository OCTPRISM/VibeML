"""
core/llm_ft_explainer.py  -  LLM 指令微调的可解释迭代器（core/explainer.py 的 SFT 版本）

和分类任务的 core/explainer.py 用同一个 IterationExplanation/NextAction 外形
（事件契约不变，前端渲染逻辑不用改），内容围绕 core/llm_ft_trainer.py 算出来的
sft_score（validation loss 的单调递减重新标度，越接近 1 越好，见该文件里的
诚实说明）：

- 和 core/rl_explainer.py 不同——sft_score 是有界（0,1]、越高越好的数字，
  不像 RL 的 reward 那样量纲因环境而异，所以这里可以像 core/explainer.py 一样
  用相对稳定的数值推理（"score 有没有在提升"“有没有停滞"），不需要 RL 那种
  只能看"趋势不看绝对值"的保守处理。
- next_action 词表沿用分类任务的完整 5 个值（不像 RL 收窄到 3 个）——指令微调
  本质上仍然是"有标注数据、有验证集"的监督学习，COLLECT_MORE_DATA（多要几条
  instruction/input/output 例子）和 ADJUST_HYPERPARAMS 都是有意义的选项。
- hyperparam_delta 换成 SFT 专属键（learning_rate / lora_rank / num_epochs），
  不复用 sklearn 的 C/alpha/class_boost——core/llm_ft_pipeline.py 自己的调参
  分支负责解释这些键。
"""

from typing import List

from core.llm_client import LLMClient
from core.json_extract import extract_json

from config import TaskSpec, EpochResult, IterationExplanation, NextAction


SYSTEM_PROMPT = """你是一个 LLM 指令微调训练顾问，负责向完全不懂机器学习的业务人员解释模型的训练进展。

你的分析风格：
- 用通俗语言，完全禁止技术术语（不说"loss"，说"错误程度"；不说"过拟合"，说"死记硬背了训练例子但学不会举一反三"）
- 用具体比喻（类比学生练习、师傅带徒弟等）
- 每条建议必须具体可操作
- 做出明确的下一步决策，不能含糊

我们用一个叫"训练得分"的指标衡量进展，取值在 0 到 1 之间，越接近 1 越好。

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
- stop_success：训练得分已经达到较高水平（比如 0.85 以上）且趋势稳定
- stop_plateau：连续几轮得分几乎不再变化，且水平不算高，陷入瓶颈
- collect_more_data：得分偏低，且怀疑是例子数量太少、覆盖不够全面导致的
- adjust_hyperparams：得分停滞但怀疑是学习率/训练轮数不合适导致的，给出
  hyperparam_delta（可选键：learning_rate 数字、lora_rank 整数、num_epochs 整数）
- continue_training：得分还在提升，或轮次还不够多
"""


class LLMFTExplainer:
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
        return f"""请分析以下 LLM 指令微调训练情况并给出诊断：

任务描述：{spec.raw_description}

=== 当前轮次（第 {current.epoch} 轮）===
训练得分：{current.val_metric:.4f}
验证损失：{current.val_loss:.4f}

=== 历史趋势（过去几轮的训练得分）===
{recent}

请基于以上信息，给出你的诊断和建议。"""

    def _rule_based_fallback(self, current: EpochResult, history: List[EpochResult]) -> IterationExplanation:
        """LLM 不可用时的规则回退——sft_score 有界且越高越好，可以用相对稳定的
        数值阈值判断，不需要像 RL 那样完全回避"好坏"判断。"""
        if current.val_metric >= 0.85:
            return IterationExplanation(
                diagnosis      = f"模型训练得分已经达到 {current.val_metric:.2f}，表现不错",
                root_cause     = "就像学生已经熟练掌握了这类题目的做法",
                recommendation = "可以先试试实际效果是否满意；满意的话就到这里",
                next_action    = NextAction.STOP_SUCCESS,
                confidence     = 0.6,
                hyperparam_delta = {},
            )
        if len(history) >= 3:
            recent = [e.val_metric for e in history[-3:]]
            if max(recent) - min(recent) < 0.02:
                return IterationExplanation(
                    diagnosis      = f"最近几轮训练得分几乎不再变化（约 {current.val_metric:.2f}）",
                    root_cause     = "就像学生反复做同样的练习但成绩不再提高，可能是练习方式需要调整",
                    recommendation = "可以尝试调整学习率，或者补充更多不同角度的例子",
                    next_action    = NextAction.STOP_PLATEAU,
                    confidence     = 0.55,
                    hyperparam_delta = {},
                )
        return IterationExplanation(
            diagnosis      = f"模型仍在学习中，当前训练得分 {current.val_metric:.2f}",
            root_cause     = "训练处于正常进展阶段，还有提升空间",
            recommendation = "继续训练，观察下一轮的变化",
            next_action    = NextAction.CONTINUE_TRAINING,
            confidence     = 0.5,
            hyperparam_delta = {},
        )
