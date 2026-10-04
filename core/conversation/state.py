"""
core/conversation/state.py  -  对话式交互的状态模型

一次对话按固定的阶段顺序推进：
  clarifying_task → preparing_data → choosing_model → configuring_training
  → training → reporting

但不是死板的线性状态机——"一次性给全信息直接跳过没问题的阶段"和"训练完之后
回到 reporting 播报结果"这些流转逻辑都在 orchestrator.py 里，这里只是数据结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from config import TaskSpec
from core.llm_client import AgentMessage


class ConversationStage(str, Enum):
    CLARIFYING_TASK      = "clarifying_task"
    PREPARING_DATA       = "preparing_data"
    CHOOSING_MODEL       = "choosing_model"
    CONFIGURING_TRAINING = "configuring_training"
    TRAINING             = "training"
    REPORTING            = "reporting"   # 训练结果播报 + 部署播报（用户原话"甚至部署"），不引入新状态机


STAGE_ORDER = [
    ConversationStage.CLARIFYING_TASK,
    ConversationStage.PREPARING_DATA,
    ConversationStage.CHOOSING_MODEL,
    ConversationStage.CONFIGURING_TRAINING,
    ConversationStage.TRAINING,
    ConversationStage.REPORTING,
]

MAX_CLARIFICATION_ROUNDS = 2   # 每个阶段最多追问几轮，超过就用默认值强制往下走（镜像 TaskParser.parse 的设计）


@dataclass
class ConversationMessage:
    role:       str   # "assistant" | "user" | "training"
    type:       str   # "question" | "text" | "stage_changed" | "training_event" | "done"
    payload:    Any   # 文本，或者训练事件原始 dict（和 core/pipeline.py 发的一模一样）
    ui_hint:    Optional[str] = None   # "data_picker" | "backend_picker" | "training_config" | "dataset_recommendation"，告诉前端渲染哪张结构化卡片
    data:       Optional[Dict[str, Any]] = None   # ui_hint 对应卡片需要的结构化数据（比如推荐的数据集预览），payload 只放给用户看的文本
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ConversationState:
    conversation_id: str
    stage:    ConversationStage = ConversationStage.CLARIFYING_TASK
    messages: List[ConversationMessage] = field(default_factory=list)

    # 这个会话是网页发起的还是外部调用方拿 API token 发起的——网页会话列表据此
    # 显示"API"标记，且网页对 API 会话只读展示（真正推进只能通过 API），
    # 见 api/routes/conversations.py::create_conversation
    origin:     Literal["web", "api"] = "web"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # 编排模式：整个对话级别的配置，不属于任何单一阶段，创建对话时一次性定好
    # （见 api/routes/conversations.py::create_conversation）——"workflow" 就是上面
    # STAGE_ORDER 这套固定状态机（前端叫"传统方式"）；"multi_agent" 走
    # core/agent/agent_orchestrator.py，任务澄清仍然复用 clarify_task 拿到
    # task_spec 之后，才真正交给 Agent 循环驱动 prepare_data 往后的全部决策——
    # 训练/评估/部署/回滚本身永远走同一套确定性 pipeline，Agent 只是换了个
    # 更自主的方式决定"什么时候调、传什么参数"，见实施计划里的关键设计决策
    orchestration_mode: Literal["workflow", "multi_agent"] = "workflow"
    # 用户在前端"⚙️ 配置"里填的 MCP 服务器地址（Phase 3 才真正生效，这里先收着）
    mcp_server_urls: Optional[List[str]] = None
    # 用户勾选启用的内置 Skill id 列表（Phase 4 才真正生效，这里先收着）
    enabled_skills: Optional[List[str]] = None
    # Multi-Agent 循环因为需要用户确认（比如真实拉取外部数据集）而"挂起"时，
    # 记录挂起原因，下一条消息到达时 _handle_multi_agent 据此判断这条消息是不是
    # 对挂起动作的回应——终止-恢复模型，不是同步阻塞等用户点击（这里没有长驻进程
    # 可以同步等，ConversationOrchestrator.handle_message 每条消息调用一次就返回）
    agent_pending_action: Optional[str] = None
    # Agent 自己维护的、provider 无关的对话历史（core/llm_client.py::AgentMessage），
    # 跨消息续跑用；和上面的 self.messages（给用户看的完整对话回放）是两回事——
    # messages 是"用户能看到的一切"，agent_transcript 是"喂给 LLM 的原始工具调用
    # 历史"，两者字段形状完全不同，不能混用
    agent_transcript: Optional[List[AgentMessage]] = None

    # clarifying_task
    raw_description: str = ""
    task_spec: Optional[TaskSpec] = None

    # preparing_data（dataset_ref / examples / env_description / instruction_examples /
    # image_examples / vlm_examples 六者互不冲突——分类任务只用 dataset_ref 或 examples
    # 二选一，RL 任务只用 env_description，LLM_FINETUNE 任务只用 instruction_examples，
    # IMAGE_CLASSIFICATION 只用 image_examples，VLM_GENERATIVE 只用 vlm_examples）
    dataset_ref: Optional[str] = None
    examples:    Optional[List[Dict]] = None
    env_description: Optional[str] = None
    # LLM_FINETUNE 任务专用：[{"instruction": ..., "input": ..., "output": ...}, ...]
    instruction_examples: Optional[List[Dict]] = None
    # IMAGE_CLASSIFICATION 任务专用：[{"image_path": ..., "label": ...}, ...]——
    # image_path 是 POST /api/datasets/upload-images 存盘后的服务器本地路径，
    # 跟文本分类的 {"text":..., "label":...} 是同一种"引用而非内联大二进制"思路
    image_examples: Optional[List[Dict]] = None
    # VLM_GENERATIVE 任务专用：[{"image_path": ..., "prompt": ..., "reference_answer": ...}, ...]
    vlm_examples: Optional[List[Dict]] = None
    # core/dataset_recommender.py 自动推荐结果（用户没指定数据集时）：
    # {"candidates": [{"platform","ref","rationale","columns","sample_rows",
    #                  "suggested_text_col","suggested_label_col"}, ...]}
    recommended_dataset: Optional[Dict[str, Any]] = None

    # choosing_model
    model_backend: str = "sklearn"
    # LLM_FINETUNE 任务专用：用户/LLM 提供并经 core/llm_ft_selector.py 校验过的底座模型 ID
    base_model_id: Optional[str] = None

    # configuring_training
    max_iterations: int = 3
    target_metric:  float = 0.80
    enable_phase2:  bool = True
    llm_provider:   str = "ollama"
    llm_model:      Optional[str] = None
    api_key:        Optional[str] = None
    llm_base_url:   Optional[str] = None   # 只对 ollama（可选覆盖默认地址）/ openai_compatible（必填）有意义——openai_compatible 不等于本地服务，同一协议也用于 OpenAI/Azure OpenAI 等商业 API，本地还是商业只取决于这里的地址
    user_id:        Optional[Any] = None   # 登录用户 id；只在 llm_provider == "system_managed" 时用于配额计量

    # training
    task_id: Optional[str] = None

    # 每个阶段各自的追问轮次计数（key 用 ConversationStage.value）
    clarification_rounds: Dict[str, int] = field(default_factory=dict)

    # 单个对话内的运行时记忆（core/conversation/memory.py 维护）：持续更新的精简
    # 事实摘要，用于让重试循环里的 LLM 调用"记得"之前试过什么、为什么不满意，
    # 而不是重放全部原始消息；过长时会被压缩重写，不会无限增长
    memory_summary: str = ""

    def add_message(self, role: str, type_: str, payload: Any, ui_hint: Optional[str] = None,
                   data: Optional[Dict[str, Any]] = None) -> ConversationMessage:
        msg = ConversationMessage(role=role, type=type_, payload=payload, ui_hint=ui_hint, data=data)
        self.messages.append(msg)
        return msg
