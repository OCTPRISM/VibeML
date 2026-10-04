"""
core/rl_explainer.py  -  强化学习的可解释迭代器（core/explainer.py 的 RL 版本）

和分类任务的 core/explainer.py 用同一个 IterationExplanation/NextAction 外形
（保持事件契约不变，前端渲染逻辑不用改），但内容围绕 reward 而不是 F1/accuracy：

  - reward 不是天然 0-1 有界、"越接近 1 越好"的指标——不同环境的 reward 量纲完全不同
    （有的环境 reward 永远是负数，有的动辄几百），所以这里不能照搬 explainer.py 里
    "指标 > 0.85 就算成功" 这种绝对阈值，只能看"趋势"（reward 有没有在稳定提升 / 有没有停滞）。
  - next_action 词表收窄为 continue_training / stop_success / stop_plateau——
    RL 场景没有"追加数据"（在线交互式生成经验，不是从数据集采样），也不引入 sklearn 那套
    C/alpha/switch_model/class_boost 调参语义，hyperparam_delta 恒为空字典。
  - LLM 不可用时的规则兜底刻意保守：只能可靠判断"停滞"（最近几段 reward 几乎不变），
    不能凭空判断"成功"——reward 数值本身的好坏含义没有通用先验，判断"是否成功"
    这件事本身就应该主要交给 LLM（结合任务描述里对"好"的自然语言定义），规则兜底
    只负责保证闭环不会卡死，不负责冒充"我知道这个 reward 算不算好"。
"""

from typing import List

from core.llm_client import LLMClient
from core.json_extract import extract_json

from config import TaskSpec, EpochResult, IterationExplanation, NextAction


SYSTEM_PROMPT = """你是一个强化学习训练顾问，负责向完全不懂机器学习的业务人员解释智能体的训练进展。

你的分析风格：
- 用通俗语言，完全禁止技术术语（不说"reward"，说"得分"；不说"episode"，说"一轮尝试"）
- 用具体比喻（类比训练宠物、教练指导运动员等）
- 每条建议必须具体可操作
- 做出明确的下一步决策，不能含糊

注意：不同任务的"得分"量纲完全不同（有的任务得分永远是负数，有的动辄几百），
不能用固定的数字阈值判断好坏，要结合"得分随训练轮次的变化趋势"和任务描述本身
对"做得好"的自然语言定义来判断。

输出 JSON 格式（只输出 JSON，不要其他内容）：
{
  "diagnosis": "一句话描述当前智能体的表现（给非技术人员看）",
  "root_cause": "根本原因分析，必须用比喻（50字以内）",
  "recommendation": "1-2条具体可操作建议（用换行分隔）",
  "next_action": "continue_training" | "stop_success" | "stop_plateau",
  "confidence": 0到1的数字
}

next_action 选择标准：
- stop_success：得分趋势已经稳定在一个较高水平，且结合任务描述判断已经学会了期望行为
- stop_plateau：连续几轮得分几乎不再变化，但结合任务描述判断还没学到期望行为（陷入瓶颈）
- continue_training：得分还在提升，或轮次还不够多，继续训练有意义
"""


class RLExplainer:
    ACTION_MAP = {
        "continue_training": NextAction.CONTINUE_TRAINING,
        "stop_success":      NextAction.STOP_SUCCESS,
        "stop_plateau":       NextAction.STOP_PLATEAU,
    }

    def __init__(self, client: LLMClient):
        self.client = client

    def explain(
        self,
        current: EpochResult,
        spec:    TaskSpec,
        history: List[EpochResult],
    ) -> IterationExplanation:
        prompt = self._build_prompt(current, spec, history)
        try:
            raw = self.client.complete(system=SYSTEM_PROMPT, user=prompt, max_tokens=1000)
            parsed = extract_json(raw)
            return IterationExplanation(
                diagnosis      = parsed.get("diagnosis", ""),
                root_cause     = parsed.get("root_cause", ""),
                recommendation = parsed.get("recommendation", ""),
                next_action    = self.ACTION_MAP.get(
                    parsed.get("next_action", "continue_training"), NextAction.CONTINUE_TRAINING),
                confidence     = float(parsed.get("confidence", 0.5)),
                hyperparam_delta = {},
            )
        except Exception:
            return self._rule_based_fallback(current, history)

    def _build_prompt(self, current: EpochResult, spec: TaskSpec, history: List[EpochResult]) -> str:
        recent = [round(e.val_metric, 4) for e in history[-5:]]
        return f"""请分析以下强化学习训练情况并给出诊断：

任务描述：{spec.raw_description}

=== 当前轮次（第 {current.epoch} 段训练）===
平均得分：{current.val_metric:.4f}
得分标准差：{current.confusion_highlights[0] if current.confusion_highlights else '未知'}

=== 历史趋势（过去几段的平均得分）===
{recent}

请基于以上信息，给出你的诊断和建议。记住：用非技术人员能理解的语言，
并结合任务描述本身对"做得好"的定义来判断，而不是套用固定数字阈值。"""

    def _rule_based_fallback(self, current: EpochResult, history: List[EpochResult]) -> IterationExplanation:
        """LLM 不可用时的规则回退，只负责保证闭环不中断，不负责判断"reward 好不好"这件事"""
        if len(history) >= 3:
            recent = [e.val_metric for e in history[-3:]]
            spread = max(recent) - min(recent)
            baseline = max(abs(v) for v in recent) or 1.0
            if spread / baseline < 0.02:
                return IterationExplanation(
                    diagnosis      = f"最近几段训练得分几乎不再变化（约 {current.val_metric:.2f}）",
                    root_cause     = "就像运动员的成绩连续几周没有进步，可能遇到了当前方法的天花板",
                    recommendation = "可以先停下来看看效果是否满意；如果不满意，考虑换个训练方式重新来一轮",
                    next_action    = NextAction.STOP_PLATEAU,
                    confidence     = 0.55,
                )
        return IterationExplanation(
            diagnosis      = f"智能体仍在学习中，当前平均得分 {current.val_metric:.2f}",
            root_cause     = "训练处于正常进展阶段，还有提升空间",
            recommendation = "继续训练，观察下一段的变化",
            next_action    = NextAction.CONTINUE_TRAINING,
            confidence     = 0.5,
        )
