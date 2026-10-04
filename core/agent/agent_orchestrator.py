"""
core/agent/agent_orchestrator.py  -  Multi-Agent 模式的主循环。

一次 handle_message 调用最多对应一次 AgentOrchestrator.run()——不是长驻协程；
每次调用都从 state.agent_transcript 恢复上一次的工具调用历史继续跑（终止-恢复
模型，不是同步阻塞等用户点击确认，原因见实施计划"关键设计决策 #2"：
ConversationOrchestrator.handle_message 每条消息调用一次就返回，没有长驻进程
可以同步 await 一个未来的用户点击）。
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Callable, List, Optional

from core.agent.events import AgentEvent
from core.agent.mcp_client import McpToolset
from core.agent.skills import build_skills_appendix
from core.agent.tools import DATASET_CONFIRMATION_PREFIX, build_agent_tools
from core.conversation.state import ConversationMessage, ConversationState
from core.llm_client import AgentMessage, LLMClient, ToolResult

MAX_AGENT_TURNS = 15
AGENT_TIMEOUT_SECONDS = 300

SYSTEM_PROMPT_TEMPLATE = """你是一个自动化机器学习（AutoML）助手，负责把用户的任务描述变成一个训练好的模型。

任务信息：
{task_summary}

工作方式：
1. 先调用 set_plan 制定一份执行计划（数据准备/模型选择/训练/查看结果几个大步骤）。
2. 每开始/完成一个步骤都调用 update_step_status，这样用户能实时看到进度。
3. 需要数据集时，先用 search_datasets/preview_dataset 研究候选，选定后必须调用
   request_dataset_confirmation 请求用户确认——不能跳过这一步直接训练，用户的
   数据没经过确认之前你不能假装已经在用它。
4. 需要真实提交训练时用 submit_training；判断训练是否成功只能依据
   check_training_progress 返回的真实结果，绝不能自己猜测或编造指标。
5. 如果需要比较多个候选（比如好几个候选数据集）或者交叉核实某个判断，可以调用
   spawn_subagent 派生一个短生命周期的只读子 Agent 去做这段调研，它跑完会把
   结论直接返回给你——只在真的需要额外调研时用，简单任务不要用这个。
6. 全部完成（或者确认没有更多可以做的事）后调用 finish_run 做总结。

每一步都要用工具真正执行，不要只用文字描述"我会做什么"而不调用对应的工具。"""


def _build_task_summary(state: ConversationState) -> str:
    spec = state.task_spec
    if not spec:
        return "（任务类型尚未确定）"
    lines = [
        f"任务类型：{spec.task_type.value}",
        f"领域：{spec.domain}",
        f"输入：{spec.input_field}",
        f"期望输出：{spec.output_description}",
        f"评估指标：{spec.evaluation_metric}",
    ]
    if spec.label_schema:
        lines.append(f"标签体系：{spec.label_schema}")
    if state.dataset_ref:
        lines.append(f"已选定数据集：{state.dataset_ref}")
    if state.examples:
        lines.append(f"已有手动样本 {len(state.examples)} 条")
    return "\n".join(lines)


class AgentOrchestrator:
    async def run(
        self, state: ConversationState, user_text: str, structured: Optional[dict],
        client: LLMClient, start_training_relay: Callable[[str], None],
        push_live: Optional[Callable[[ConversationMessage], None]] = None,
        deterministic_fallback: Optional[Callable[[], Any]] = None,
    ) -> List[ConversationMessage]:
        """push_live：由调用方（core/conversation/orchestrator.py::_handle_multi_agent）
        传入，通常是 self.conv_store.push_live 的一层包装。这个方法本身现在总是被
        调用方用 asyncio.create_task(...) 当后台任务甩出去跑（而不是同步 await 到底）——
        单轮循环最多 15 次工具调用，真同步等完再返回的话，用户的 WS 在这期间什么都
        收不到，直到全部跑完才一次性收到一大堆消息，跟"实时看到 Agent 正在做什么"
        这个需求直接矛盾。跟 _start_training/_relay_training_events 是同一个模式：
        每产出一条消息就立刻 push_live，不等整个方法 return。"""
        new_messages: List[ConversationMessage] = []

        def _emit(events: List[AgentEvent]) -> None:
            # 这是唯一调用 state.add_message() 的地方——工具函数自己不直接调用，
            # 保证每一条新消息都立刻通过 push_live 实时推给前端 WS（不是等
            # 整个 run() 返回后才一次性推），同时也累积进返回值供非 WS 场景使用
            for ev in events:
                payload = ev.payload_override if ev.payload_override is not None else ev.text
                data = {"kind": ev.kind, **ev.data} if ev.type_ == "agent_event" else (ev.data or None)
                msg = state.add_message(
                    role=ev.role, type_=ev.type_, payload=payload, ui_hint=ev.ui_hint, data=data,
                )
                new_messages.append(msg)
                if push_live:
                    push_live(msg)

        transcript: List[AgentMessage] = list(state.agent_transcript or [])

        # 恢复挂起动作：上一轮 request_dataset_confirmation 之后，这条消息是不是
        # 用户对确认请求的回应
        if state.agent_pending_action and state.agent_pending_action.startswith(DATASET_CONFIRMATION_PREFIX):
            pending_call_id = state.agent_pending_action[len(DATASET_CONFIRMATION_PREFIX):]
            if structured and structured.get("type") == "data_selected":
                if structured.get("dataset_ref"):
                    state.dataset_ref = structured["dataset_ref"]
                elif structured.get("examples"):
                    state.examples = structured["examples"]
                result_text = f"用户已确认使用数据集：{structured.get('dataset_ref') or '（手动样本）'}"
                transcript.append(AgentMessage(role="tool", content=[
                    ToolResult(tool_call_id=pending_call_id, content=result_text)]))
                state.agent_pending_action = None
                _emit([AgentEvent(kind="tool_result", text="✅ 数据集已确认，继续执行")])
            else:
                # 用户发了别的消息而不是确认——保持挂起状态，把这句话原样当成
                # 新的一轮用户输入喂给 Agent，让它自己决定怎么回应（不强行打断）
                transcript.append(AgentMessage(
                    role="user", content=user_text or "（还没有确认数据集，用户发了其他消息）"))
        elif user_text.strip():
            transcript.append(AgentMessage(role="user", content=user_text))

        if not transcript:
            # 任务澄清刚结束、Agent 第一次接管时 user_text 是空的——大多数 provider
            # 要求第一条消息是 role=user，这里补一句启动语而不是发一个空列表过去
            transcript.append(AgentMessage(role="user", content="请根据以上任务信息开始执行。"))

        tools, dispatch = build_agent_tools(state, client, start_training_relay)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(task_summary=_build_task_summary(state))
        if state.enabled_skills:
            system_prompt += build_skills_appendix(state.enabled_skills)

        # MCP 工具集的连接/断开跟这次 run() 调用同生命周期——见模块顶部注释：
        # 不能跨 pause/resume 保活一个连接，每次 resume 都重新连接同样的 URL。
        mcp_toolset = McpToolset()
        if state.mcp_server_urls:
            await mcp_toolset.connect(state.mcp_server_urls)
            tools = tools + mcp_toolset.tools
            dispatch = {**dispatch, **mcp_toolset.dispatch}
            _emit([AgentEvent(kind="tool_result", text=mcp_toolset.summary_text())])

        try:
            try:
                start_time = time.monotonic()
                # 累计资源用量——每条 resource_usage 事件都带上累计值，这样前端
                # 不用自己攒状态就能显示"这次任务到目前为止一共花了多少"
                cumulative_tokens = 0
                cumulative_llm_ms = 0
                # 工具调用总数——用来判断"这个模型到底会不会用工具"。
                # 前端已经去掉了"传统方式/Multi-Agent"开关，统一走 Agent 模式，
                # 但本地小模型的工具调用能力参差不齐（README 里一直记着这条）。
                # 一次工具都没调用成功 = 这个模型撑不起 Agent 模式，必须退回
                # 确定性流程，否则用户只会收到一句闲聊、训练根本没发生
                total_tool_calls = 0
                for _turn in range(MAX_AGENT_TURNS):
                    if time.monotonic() - start_time > AGENT_TIMEOUT_SECONDS:
                        _emit([AgentEvent(kind="final",
                                           text="这一轮花的时间有点长，先停在这里——要继续的话直接回复我。")])
                        break

                    # complete_with_tools 本身没有 try/except——这个方法现在总是被
                    # asyncio.create_task(...) 当后台任务甩出去跑，异常在这里不catch
                    # 的话会变成 asyncio 的"Task exception was never retrieved"警告，
                    # 用户在前端什么都看不到，只会一直转圈；最外层这个 try 就是补这个洞
                    # 埋点：每一轮 LLM 调用的耗时和 token 用量。前端的"Agent 在做什么/
                    # 占了多少资源/干了多久"面板全靠这条事件——不埋的话用户只能看到
                    # 一堆工具调用在滚，完全不知道代价。client.last_usage 是 Phase 3
                    # 配额工作时就有的现成机制（每个 provider 的 complete_with_tools
                    # 内部都会写），这里直接读，不重复实现一套 token 计数
                    _turn_started = time.monotonic()
                    result = await client.complete_with_tools(transcript, tools, system=system_prompt, max_tokens=2000)
                    _turn_ms = int((time.monotonic() - _turn_started) * 1000)
                    _usage = getattr(client, "last_usage", None) or {}
                    cumulative_tokens += int(_usage.get("total_tokens", 0) or 0)
                    cumulative_llm_ms += _turn_ms
                    _emit([AgentEvent(
                        kind="resource_usage",
                        text=f"第 {_turn + 1} 轮思考：{_turn_ms / 1000:.1f}s"
                             + (f"，{_usage.get('total_tokens')} tokens" if _usage.get("total_tokens") else ""),
                        data={
                            "turn": _turn + 1,
                            "duration_ms": _turn_ms,
                            "prompt_tokens": _usage.get("prompt_tokens"),
                            "completion_tokens": _usage.get("completion_tokens"),
                            "total_tokens": _usage.get("total_tokens"),
                            "cumulative_tokens": cumulative_tokens,
                            "cumulative_llm_ms": cumulative_llm_ms,
                            "elapsed_ms": int((time.monotonic() - start_time) * 1000),
                        })])

                    if result.stop_reason != "tool_use":
                        # 模型选择直接用文本回应而不调用任何工具——当成这一轮的最终回复，
                        # 不强行继续循环空转
                        if result.text.strip():
                            transcript.append(AgentMessage(role="assistant", content=result.text))
                            _emit([AgentEvent(kind="final", text=result.text.strip())])
                        break

                    total_tool_calls += len(result.tool_calls or [])
                    transcript.append(AgentMessage(role="assistant", content=result.tool_calls))

                    paused = False
                    tool_results: List[ToolResult] = []
                    tool_calls = result.tool_calls
                    idx = 0
                    while idx < len(tool_calls):
                        call = tool_calls[idx]

                        if call.name == "spawn_subagent":
                            # 把这一轮里连续出现的 spawn_subagent 调用聚成一批，用
                            # asyncio.gather 真正并发跑——这是"需要并行研究时动态创建
                            # 子 Agent"里"并行"两个字真正落地的地方。之前的实现是严格
                            # 顺序的 for 循环，即使 LLM 在同一轮里一次性请求了好几个
                            # spawn_subagent（模型自己觉得"这些可以一起做"），也会被这
                            # 个循环逐个 await 完才轮到下一个，"并发"只是代码结构上支持
                            # （子 Agent 之间互不依赖），从未真正同时执行过。只批量并发
                            # spawn_subagent 这一种工具——它是只读、无副作用、不会产出
                            # confirmation_required/finish_run 这类需要中断主循环的事件，
                            # 其它工具（submit_training 等有副作用的、或可能改变控制流的）
                            # 依然严格按原有顺序逐个执行，不引入交叉执行的不确定性
                            batch = []
                            while idx < len(tool_calls) and tool_calls[idx].name == "spawn_subagent":
                                batch.append(tool_calls[idx])
                                idx += 1
                            fns = [dispatch.get(c.name) for c in batch]
                            outcomes = await asyncio.gather(
                                *(fn(c.input) for fn, c in zip(fns, batch)),
                                return_exceptions=True,
                            )
                            for c, outcome in zip(batch, outcomes):
                                if isinstance(outcome, BaseException):
                                    tool_results.append(ToolResult(tool_call_id=c.id,
                                                                    content=f"工具执行出错：{outcome}", is_error=True))
                                    _emit([AgentEvent(kind="tool_result", text=f"⚠️ {c.name} 执行出错：{outcome}")])
                                    continue
                                content, events = outcome
                                tool_results.append(ToolResult(tool_call_id=c.id, content=content))
                                _emit(events)
                            continue

                        fn = dispatch.get(call.name)
                        if fn is None:
                            tool_results.append(ToolResult(tool_call_id=call.id,
                                                            content=f"未知工具：{call.name}", is_error=True))
                            idx += 1
                            continue
                        _tool_started = time.monotonic()
                        try:
                            outcome = fn(call.input)
                            if inspect.isawaitable(outcome):
                                outcome = await outcome
                            content, events = outcome
                            # 工具耗时挂在它自己产出的第一条事件的 data 上——训练提交、
                            # 数据集拉取这类工具可能跑很久，用户需要看得到是哪一步在耗时，
                            # 而不是只看到"Agent 卡住了"
                            _tool_ms = int((time.monotonic() - _tool_started) * 1000)
                            if events:
                                events[0].data = {**(events[0].data or {}),
                                                  "tool_name": call.name, "duration_ms": _tool_ms}
                        except Exception as e:
                            tool_results.append(ToolResult(tool_call_id=call.id,
                                                            content=f"工具执行出错：{e}", is_error=True))
                            _emit([AgentEvent(kind="tool_result", text=f"⚠️ {call.name} 执行出错：{e}")])
                            idx += 1
                            continue

                        if events and events[0].kind == "confirmation_required":
                            # 终止本轮循环，等用户点确认后从上面的 agent_pending_action
                            # 分支恢复——不同步阻塞等待
                            state.agent_pending_action = f"{DATASET_CONFIRMATION_PREFIX}{call.id}"
                            _emit(events)
                            paused = True
                            break

                        tool_results.append(ToolResult(tool_call_id=call.id, content=content))
                        _emit(events)

                        if call.name == "finish_run":
                            paused = True  # 复用同一个跳出标记：这一轮循环到此为止
                            break

                        idx += 1

                    if paused:
                        break

                    transcript.append(AgentMessage(role="tool", content=tool_results))
                else:
                    _emit([AgentEvent(kind="final",
                                       text=f"达到本轮工具调用上限（{MAX_AGENT_TURNS} 次），先停在这里——要继续的话直接回复我。")])
            except Exception as e:
                _emit([AgentEvent(kind="final", text=f"执行过程中出错了：{e}。可以再试一次，或者换个方式描述任务。")])
        finally:
            await mcp_toolset.aclose()

        state.agent_transcript = transcript

        # ── 内部降级兜底 ──────────────────────────────────────────────────
        # 前端已经去掉"传统方式/Multi-Agent"的用户可见开关（统一走 Agent），
        # 但确定性状态机没有删——它降级成了这里的兜底：整轮下来一次工具都没调用
        # 成功，说明这个模型撑不起 Agent 模式（本地小模型的 function calling
        # 能力参差不齐，这是 README 里一直记着的已知限制）。这种情况下如果就这么
        # 结束，用户只会收到一句闲聊、训练根本没发生，而且完全不知道为什么。
        #
        # 只在"从头到尾零工具调用"时触发，不用"某几轮没调用"做判据——模型中途
        # 用一句文字收尾是完全正常的行为，不该因此把它踢回确定性流程。
        if total_tool_calls == 0 and deterministic_fallback is not None:
            _emit([AgentEvent(
                kind="fallback_to_workflow",
                text="当前模型没有成功发起任何工具调用（本地小模型常见），"
                     "已自动切换到确定性流程继续执行——功能不受影响，"
                     "只是不会有 Agent 的自主决策过程。",
                data={"reason": "no_tool_calls"})])
            try:
                outcome = deterministic_fallback()
                if inspect.isawaitable(outcome):
                    await outcome
            except Exception as e:
                _emit([AgentEvent(kind="final", text=f"确定性流程也失败了：{e}")])

        return new_messages
