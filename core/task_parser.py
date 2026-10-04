"""
core/task_parser.py  -  Phase 1.1：会话解析器

差异化核心：
  用户用完全自然的语言描述任务（"帮我识别客服工单是什么类型的问题"），
  系统自动推断：任务类型、标签体系、评估指标、领域特征。
  如果描述不够清晰，自动追问。全程不暴露任何技术术语。
"""

import json
import re
from typing import Optional, Tuple
from core.llm_client import LLMClient
from config import TaskSpec, TaskType


# ── Prompt ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个 ML 任务解析专家。用户会用自然语言描述他们想解决的问题。
你的工作是将这段描述解析为结构化的机器学习任务定义。

输出必须是合法的 JSON，包含以下字段（不要输出任何其他内容）：
{
  "task_type": "classification" | "multi_label" | "ner" | "generation" | "regression" | "rl" | "llm_finetune" | "image_classification" | "vlm_generative",
  "domain": "领域描述，如：客服、医疗、法律、电商",
  "label_schema": ["标签1", "标签2"],
  "input_field": "输入内容的类型描述，如：工单文本、用户评论、合同段落",
  "output_description": "期望输出的一句话描述",
  "evaluation_metric": "f1" | "accuracy" | "mae",
  "language": "zh" | "en" | "mixed",
  "constraints": {},
  "needs_clarification": false,
  "clarification_question": ""
}

解析规则：
- label_schema：如描述中有明确标签列表则直接用；否则根据领域常识合理推断（2-6个）
- evaluation_metric：多分类优先选 f1；二分类选 accuracy；回归选 mae
- task_type = "rl"：用户描述的是"训练一个智能体/agent 在某种环境里通过试错学习行为策略"
  （比如走迷宫、玩游戏、控制机械臂、调度资源），而不是"给一批已标注数据分类"——
  这种任务没有 label_schema 概念，此时 label_schema 留空数组即可
- task_type = "llm_finetune"：用户明确提到"微调一个模型"/"训练一个能生成…的模型"/
  "教模型学会…的写法"，想让模型学会按照一批例子里的模式生成自由文本输出（不是从
  固定标签集合里选一个），且这批例子天然是"指令/输入/输出"这种问答式或者
  转换式的三元组（比如翻译、改写文风、按格式生成文案），而不是分类标签——
  这种任务也没有 label_schema 概念，label_schema 留空数组即可，和普通 "generation"
  的区别在于用户明确提到"微调"这个诉求（而不是让系统自动帮忙判断该用什么后端）
- task_type = "image_classification"：用户描述的是"给图片分类"（比如识别图片里
  的物体类别、判断图片质量好坏、区分产品外观是否合格），输入是图片而不是文本，
  输出是从固定标签集合里选一个——label_schema 按分类任务同样规则推断
- task_type = "vlm_generative"：用户描述的是"看图生成文字"（比如给图片生成一句
  描述/看图说话、根据图片回答问题/视觉问答），输入是图片+一个问题或指令，输出是
  自由文本而不是固定标签——这种任务也没有 label_schema 概念，留空数组即可，
  且这种任务需要参考答案来打分，如果用户描述里完全没提供任何图文对应的例子
  或者提到"没有标准答案"，仍然按这个 task_type 分类，具体样本收集在后面阶段处理
- needs_clarification：仅当连任务类型都无法判断时才为 true
- clarification_question：追问要具体，一句话，帮助用户理解系统需要什么信息
- 不要在 JSON 外添加任何解释文字
"""


# ── Parser ───────────────────────────────────────────────────────────────────

class TaskParser:
    """
    将用户的自然语言任务描述转化为结构化 TaskSpec。

    用法：
        parser = TaskParser(client)
        spec = parser.parse("帮我把客服工单按问题类型分类")
    """

    TASK_TYPE_MAP = {
        "classification": TaskType.CLASSIFICATION,
        "multi_label":    TaskType.MULTI_LABEL,
        "ner":            TaskType.NER,
        "generation":     TaskType.GENERATION,
        "regression":     TaskType.REGRESSION,
        "rl":             TaskType.RL,
        "llm_finetune":   TaskType.LLM_FINETUNE,
        "image_classification": TaskType.IMAGE_CLASSIFICATION,
        "vlm_generative":        TaskType.VLM_GENERATIVE,
    }

    def __init__(self, client: LLMClient):
        self.client = client

    def parse_once(
        self,
        user_input: str,
        raw_description: Optional[str] = None,
    ) -> Tuple[TaskSpec, bool, str]:
        """
        单轮解析：只调一次 LLM，不追问、不阻塞等待输入。

        供逐轮驱动的调用方（比如网页版的多轮对话编排器）使用——每轮由调用方
        自己决定要不要追问、怎么把用户的回答拼回去，这里只负责"这一轮 LLM
        怎么看这段输入"。

        Args:
            user_input:      这一轮实际喂给 LLM 的文本（多轮追问时可能是
                             "原始描述 + 补充信息" 拼接后的文本）
            raw_description: TaskSpec.raw_description 要保留的"最初的原始描述"，
                             不传则用 user_input 本身（单轮场景下两者相同）

        Returns:
            (TaskSpec, needs_clarification, clarification_question)
        """
        parsed = self._call_llm(user_input)
        spec = self._build_spec(parsed, raw_description if raw_description is not None else user_input)
        needs_clarification = parsed.get("needs_clarification", False)
        question = parsed.get("clarification_question", "请补充更多信息")
        return spec, needs_clarification, question

    def parse(
        self,
        user_input: str,
        max_clarification_rounds: int = 2,
        interactive: bool = True,
    ) -> TaskSpec:
        """
        解析用户描述。如果描述不清晰且 interactive=True，会自动追问（命令行 input()）。

        Args:
            user_input:              用户的自然语言任务描述
            max_clarification_rounds: 最多追问几轮
            interactive:             是否允许命令行追问（关闭则跳过追问）

        Returns:
            TaskSpec
        """
        current_input = user_input
        spec: Optional[TaskSpec] = None

        for round_num in range(max_clarification_rounds + 1):
            spec, needs_clarification, question = self.parse_once(current_input, raw_description=user_input)

            if not needs_clarification or round_num >= max_clarification_rounds or not interactive:
                return spec

            # 还需要追问
            print(f"\n🤔 {question}")
            answer = input("你的回答：").strip()

            if not answer:
                # 用户跳过追问，强制解析
                return spec

            current_input = f"原始描述：{user_input}\n补充信息：{answer}"

        return spec

    # ── 私有方法 ─────────────────────────────────────────────────────────────

    def _call_llm(self, user_input: str) -> dict:
        raw = self.client.complete(system=SYSTEM_PROMPT, user=user_input, max_tokens=1000)
        return self._extract_json(raw)

    def _extract_json(self, text: str) -> dict:
        """从可能包含 markdown 代码块的响应中提取 JSON"""
        # 去掉 ```json ... ``` 或 ``` ... ```
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试找到第一个 {...} 块
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"无法解析 JSON 响应：{text[:200]}")

    def _build_spec(self, parsed: dict, raw: str) -> TaskSpec:
        return TaskSpec(
            task_type=self.TASK_TYPE_MAP.get(
                parsed.get("task_type", "classification"),
                TaskType.CLASSIFICATION,
            ),
            domain=parsed.get("domain", "通用"),
            label_schema=parsed.get("label_schema", []),
            input_field=parsed.get("input_field", "文本"),
            output_description=parsed.get("output_description", ""),
            evaluation_metric=parsed.get("evaluation_metric", "f1"),
            language=parsed.get("language", "zh"),
            constraints=parsed.get("constraints", {}),
            raw_description=raw,
        )
