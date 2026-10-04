"""
core/agent/events.py  -  Agent 工具执行时产出的、给前端渲染用的事件。

跟 core/pipeline.py 发的 training_event 是同一个思路：每个事件都是一份可以
原样存进 ConversationState.messages、原样回放给前端的记录——前端的 Plan 面板
靠重放这些 agent_event 消息重建"计划/当前步骤/正在做什么"，不需要额外维护
一份"当前计划"的服务端状态（跟 web/app.js::reduceEvent 靠重放 training_event
重建训练可视化状态是同一个模式）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AgentEvent:
    kind: str          # "plan_created" | "plan_step_update" | "tool_call" | "tool_result" |
                        # "confirmation_required" | "subagent_spawned" | "subagent_done" | "final" |
                        # "training_snapshot"（submit_training 补的那条特殊事件，见下）|
                        # "resource_usage"（每轮 LLM 调用的耗时/token，供前端资源面板显示）
    text: str           # 给用户看的一句话（前端聊天气泡/活动流直接显示这个）
    data: Dict[str, Any] = field(default_factory=dict)   # 结构化数据，供 Plan 面板渲染

    # 默认渲染成 role="assistant", type_="agent_event" 的 ConversationMessage；
    # submit_training 需要补一条跟 workflow 模式一模一样的 role="training",
    # type_="training_event" 消息（前端 reduceEvent 靠它拿到 task_id），用这两个
    # 字段覆盖默认形状——AgentOrchestrator._emit() 是唯一调用 state.add_message()
    # 的地方，工具函数自己不直接调用，否则新增的消息不会被推进 handle_message()
    # 返回值里、不会实时推给前端 WS
    role: str = "assistant"
    type_: str = "agent_event"
    payload_override: Optional[Any] = None
    # request_dataset_confirmation 需要复用现成的 dataset_recommendation 卡片
    # 契约（type_="question", ui_hint="dataset_recommendation"）——这样即使
    # Phase 1a 还没做任何前端改动，真实浏览器打开也已经能正确弹出这张卡片，
    # 不需要等 Phase 1b
    ui_hint: Optional[str] = None
