"""
core/agent/tools.py  -  Multi-Agent 模式的工具集。

每个工具都是对现有确定性能力的一层薄包装，不重新实现任何逻辑：
  set_plan / update_step_status   纯记账，产出 Plan 面板需要的事件
  search_datasets / preview_dataset  包 core/data_sources.py
  request_dataset_confirmation    不直接拉取——复用现成的 dataset_recommendation
                                   ui_hint + data_selected 结构化消息确认流程，
                                   保留"真实外部下载前必须显式确认"的原则
  design_architecture / select_backbone  包 core/arch_designer.py / core/backbone_selector.py
  submit_training                 包 api/worker.py::submit_training_job（同一条
                                   训练/沙盒/子进程隔离路径，Agent 没有旁路）
  check_training_progress         只读 task_store 里真实测出来的字段，从不让
                                   Agent 自己判定"是否成功"
  spawn_subagent                  派生 core/agent/subagent.py 里的短生命周期只读
                                   研究子 Agent，MAX_CONCURRENT_SUBAGENTS 硬顶
  finish_run                      结束本轮 Agent 循环

返回约定：每个工具函数签名 (state, tool_input: dict) -> Tuple[str, List[AgentEvent]]，
第一个返回值是喂给 LLM 的 tool_result 文本，第二个是需要转成 ConversationMessage
推给前端的事件列表。
"""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any, Callable, Dict, List, Tuple

from config import TaskType
from core.agent.events import AgentEvent
from core.agent.subagent import run_subagent
from core.arch_designer import ArchDesigner
from core.backbone_selector import BackboneSelector
from core.conversation.state import ConversationState
from core.data_sources import get_source
from core.llm_client import LLMClient, ToolDef

MAX_CONCURRENT_SUBAGENTS = 3

ToolFn = Callable[[Dict[str, Any]], Tuple[str, List[AgentEvent]]]

# 数据集确认："真实外部下载前必须显式确认"——request_dataset_confirmation 触发后
# 把这个前缀 + platform + ref 存进 state.agent_pending_action，AgentOrchestrator
# 据此判断下一条消息是不是对这次确认请求的回应
DATASET_CONFIRMATION_PREFIX = "dataset_confirmation:"


def build_agent_tools(
    state: ConversationState, client: LLMClient,
    start_training_relay: Callable[[str], None],
) -> Tuple[List[ToolDef], Dict[str, ToolFn]]:
    """start_training_relay(task_id)：由 core/conversation/orchestrator.py 传入，
    包一层 self._relay_training_events(state.conversation_id, task_id) 的
    asyncio.create_task(...) 调用——训练事件推送机制跟 workflow 模式完全复用，
    不在这里（core/agent/）重新实现一遍，training_event 消息还是走原有的
    reduceEvent/render() 路径显示在右侧结果区。"""
    tools: List[ToolDef] = []
    dispatch: Dict[str, ToolFn] = {}

    def _register(tool_def: ToolDef, fn: ToolFn) -> None:
        tools.append(tool_def)
        dispatch[tool_def.name] = fn

    # ── set_plan ─────────────────────────────────────────────────────────
    def _set_plan(inp: Dict[str, Any]) -> Tuple[str, List[AgentEvent]]:
        steps = inp.get("steps") or []
        event = AgentEvent(kind="plan_created", text=f"制定了执行计划（共 {len(steps)} 步）",
                            data={"steps": steps})
        return "计划已记录，可以开始按步骤执行了。", [event]

    _register(ToolDef(
        name="set_plan",
        description="制定或更新本次任务的执行计划——一份给用户看的步骤清单（比如\"搜索/确认数据集\"\""
                    "设计模型架构\"\"提交训练\"\"查看结果\"）。任务一开始调用一次，"
                    "计划需要重大调整时可以再调一次。",
        input_schema={
            "type": "object",
            "properties": {
                "steps": {"type": "array", "items": {"type": "string"},
                          "description": "按执行顺序排列的步骤描述，每条一句话，给用户看的"},
            },
            "required": ["steps"],
        },
    ), _set_plan)

    # ── update_step_status ───────────────────────────────────────────────
    def _update_step_status(inp: Dict[str, Any]) -> Tuple[str, List[AgentEvent]]:
        index = inp.get("index", 0)
        status = inp.get("status", "in_progress")
        note = inp.get("note", "")
        event = AgentEvent(kind="plan_step_update",
                            text=note or f"步骤 {index + 1}：{status}",
                            data={"index": index, "status": status, "note": note})
        return f"步骤 {index} 状态已更新为 {status}。", [event]

    _register(ToolDef(
        name="update_step_status",
        description="更新某一步计划的执行状态。每次开始执行一个新步骤、完成一个步骤、或者"
                    "一个步骤失败时都要调用一次，这样用户能实时看到进度。",
        input_schema={
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "步骤在 set_plan 的 steps 数组里的下标，从 0 开始"},
                "status": {"type": "string", "enum": ["in_progress", "done", "failed"]},
                "note": {"type": "string", "description": "给用户看的简短说明，比如具体做了什么/为什么失败"},
            },
            "required": ["index", "status"],
        },
    ), _update_step_status)

    # ── search_datasets ──────────────────────────────────────────────────
    # async——AgentOrchestrator.run() 的工具调度循环本来就用 inspect.isawaitable
    # 判断要不要 await（spawn_subagent 等工具早就是 async def 了），这里改成
    # async 不需要动调度逻辑。真正的 HF/魔搭网络请求用 asyncio.to_thread 丢进
    # 线程池——不然这一次工具调用会同步阻塞整个事件循环，拖慢同一进程里其它
    # 所有请求（真实复现过：一次真实 Multi-Agent 会话卡住期间，完全无关的
    # /api/tasks/queue/stats 健康检查也会跟着没有响应）。
    async def _search_datasets(inp: Dict[str, Any]) -> Tuple[str, List[AgentEvent]]:
        platform = inp.get("platform", "huggingface")
        query = inp.get("query", "")
        try:
            results = await asyncio.to_thread(get_source(platform).search, query)
        except Exception as e:
            return f"搜索失败：{e}", [AgentEvent(kind="tool_result", text=f"🔍 搜索 {platform} 失败：{e}")]
        lines = [f"- {d.ref}（下载量 {d.downloads}，点赞 {d.likes}）：{d.description}" for d in results[:10]]
        text = "\n".join(lines) if lines else "没有搜到匹配的数据集。"
        event = AgentEvent(kind="tool_result", text=f"🔍 在 {platform} 搜到 {len(results)} 个数据集",
                            data={"tool": "search_datasets", "platform": platform, "query": query,
                                  "results": [{"ref": d.ref, "description": d.description} for d in results[:10]]})
        return text, [event]

    _register(ToolDef(
        name="search_datasets",
        description="在 HuggingFace（platform=huggingface）或魔搭（platform=modelscope）"
                    "上按关键词搜索公开数据集。",
        input_schema={
            "type": "object",
            "properties": {
                "platform": {"type": "string", "enum": ["huggingface", "modelscope"]},
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["platform", "query"],
        },
    ), _search_datasets)

    # ── preview_dataset ──────────────────────────────────────────────────
    async def _preview_dataset(inp: Dict[str, Any]) -> Tuple[str, List[AgentEvent]]:
        platform = inp.get("platform", "huggingface")
        ref = inp.get("ref", "")
        try:
            preview = await asyncio.to_thread(get_source(platform).preview, ref)
        except Exception as e:
            return f"预览失败：{e}", [AgentEvent(kind="tool_result", text=f"👀 预览 {ref} 失败：{e}")]
        text = (f"列名：{preview.columns}；建议文本列：{preview.suggested_text_col}；"
                f"建议标签列：{preview.suggested_label_col}；样本行数：{len(preview.sample_rows)}")
        event = AgentEvent(kind="tool_result", text=f"👀 预览了数据集「{ref}」",
                            data={"tool": "preview_dataset", "platform": platform, "ref": ref,
                                  "columns": preview.columns, "sample_rows": preview.sample_rows[:5]})
        return text, [event]

    _register(ToolDef(
        name="preview_dataset",
        description="预览一个数据集的列名、样本行、自动建议的文本/标签列映射，不会真的下载全部数据。",
        input_schema={
            "type": "object",
            "properties": {
                "platform": {"type": "string", "enum": ["huggingface", "modelscope"]},
                "ref": {"type": "string", "description": "数据集 ID，比如 stanfordnlp/imdb"},
            },
            "required": ["platform", "ref"],
        },
    ), _preview_dataset)

    # ── request_dataset_confirmation ─────────────────────────────────────
    # 不直接拉取——这里只产出预览 + 一个"需要用户确认"事件，AgentOrchestrator
    # 收到这个事件后会终止本轮循环、把 state.agent_pending_action 设好，
    # 复用现成的 dataset_recommendation UI 卡片等用户点"用这个"
    async def _request_dataset_confirmation(inp: Dict[str, Any]) -> Tuple[str, List[AgentEvent]]:
        platform = inp.get("platform", "huggingface")
        ref = inp.get("ref", "")
        rationale = inp.get("rationale", "")
        try:
            preview = await asyncio.to_thread(get_source(platform).preview, ref)
        except Exception as e:
            return f"预览失败，无法请求确认：{e}", [
                AgentEvent(kind="tool_result", text=f"⚠️ 数据集「{ref}」预览失败：{e}")]
        state.recommended_dataset = {"candidates": [{
            "platform": platform, "ref": ref, "rationale": rationale,
            "columns": preview.columns, "sample_rows": preview.sample_rows,
            "suggested_text_col": preview.suggested_text_col,
            "suggested_label_col": preview.suggested_label_col,
        }]}
        # 复用现成的 dataset_recommendation 卡片契约（跟 workflow 模式下
        # core/conversation/stages.py::prepare_data 生成的问题消息形状完全一样），
        # 前端不需要为 Multi-Agent 模式的数据确认单独做一套 UI
        event = AgentEvent(
            kind="confirmation_required",
            text=f"我想用「{ref}」这个数据集（{rationale}）——真正拉取数据前需要你确认一下，"
                 f"下面是预览，选好之后我会接着往下做。",
            type_="question", ui_hint="dataset_recommendation", data=state.recommended_dataset,
        )
        return "", [event]  # tool_result 内容留空，因为这一步不会真正返回给 LLM——见 AgentOrchestrator

    _register(ToolDef(
        name="request_dataset_confirmation",
        description="真正使用某个外部数据集训练之前，必须先调用这个工具请求用户确认——"
                    "不能跳过这一步直接训练，用户看到预览后点确认，你才能继续。",
        input_schema={
            "type": "object",
            "properties": {
                "platform": {"type": "string", "enum": ["huggingface", "modelscope"]},
                "ref": {"type": "string"},
                "rationale": {"type": "string", "description": "为什么推荐这个数据集，给用户看的一句话理由"},
            },
            "required": ["platform", "ref", "rationale"],
        },
    ), _request_dataset_confirmation)

    # ── design_architecture ──────────────────────────────────────────────
    def _design_architecture(inp: Dict[str, Any]) -> Tuple[str, List[AgentEvent]]:
        if not state.task_spec:
            return "还没有 task_spec，无法设计架构。", []
        n_samples = inp.get("n_samples", 100)
        try:
            spec = ArchDesigner(client).design(state.task_spec, n_samples=n_samples)
        except Exception as e:
            return f"架构设计失败：{e}", [AgentEvent(kind="tool_result", text=f"⚠️ 架构设计失败：{e}")]
        text = f"设计了自定义分类头「{spec.class_name}」，理由：{spec.rationale}"
        event = AgentEvent(kind="tool_result", text=f"🧩 {text}",
                            data={"tool": "design_architecture", "class_name": spec.class_name})
        return text, [event]

    _register(ToolDef(
        name="design_architecture",
        description="让 LLM 设计一个自定义 PyTorch 分类头架构（custom_nn 后端专用），"
                    "生成的代码会经过沙盒静态校验，不是这里直接执行。",
        input_schema={
            "type": "object",
            "properties": {"n_samples": {"type": "integer", "description": "训练样本数量，用于决定架构复杂度"}},
            "required": ["n_samples"],
        },
    ), _design_architecture)

    # ── select_backbone ───────────────────────────────────────────────────
    def _select_backbone(inp: Dict[str, Any]) -> Tuple[str, List[AgentEvent]]:
        if not state.task_spec:
            return "还没有 task_spec，无法选择预训练模型。", []
        n_samples = inp.get("n_samples", 100)
        try:
            spec = BackboneSelector(client).select(state.task_spec, n_samples=n_samples)
        except Exception as e:
            return f"预训练模型选择失败：{e}", [AgentEvent(kind="tool_result", text=f"⚠️ 选型失败：{e}")]
        text = f"选择了预训练模型「{spec.model_id}」，理由：{spec.rationale}"
        event = AgentEvent(kind="tool_result", text=f"🧩 {text}",
                            data={"tool": "select_backbone", "model_id": spec.model_id})
        return text, [event]

    _register(ToolDef(
        name="select_backbone",
        description="从预训练模型里选一个适合当前任务微调（pretrained_nn 后端专用）。",
        input_schema={
            "type": "object",
            "properties": {"n_samples": {"type": "integer"}},
            "required": ["n_samples"],
        },
    ), _select_backbone)

    # ── submit_training ───────────────────────────────────────────────────
    # 唯一需要 await 的工具——其它工具都是纯同步函数，AgentOrchestrator.run()
    # 在拿到 dispatch 函数的返回值后会检测是不是协程，是的话才 await，
    # 这样不用把所有工具函数都强改成 async def
    async def _submit_training(inp: Dict[str, Any]) -> Tuple[str, List[AgentEvent]]:
        from api.worker import submit_training_job
        from api.store import task_store
        from api.dataset_store import dataset_store
        from api.models import TrainingSample

        if not state.task_spec:
            return "还没有 task_spec，无法提交训练。", []
        if not (state.dataset_ref or state.examples or state.env_description or state.instruction_examples):
            return "还没有训练数据（数据集/样本/环境描述），不能提交训练。", []

        model_backend = inp.get("model_backend", state.model_backend)
        max_iterations = inp.get("max_iterations", state.max_iterations)
        target_metric = inp.get("target_metric", state.target_metric)
        task_id = f"agent-{state.conversation_id}"
        task_store.create(task_id)

        # examples/dataset_ref 二选一——dataset_ref 只是 /api/datasets/fetch 缓存里的
        # 引用键，submit_training_job 要的是真正的样本列表，这里要跟
        # ConversationOrchestrator._start_training 一样做解析，不能原样传字符串过去
        examples = None
        if state.examples:
            examples = [TrainingSample(text=e["text"], label=e["label"]) for e in state.examples]
        elif state.dataset_ref:
            cached = dataset_store.get(state.dataset_ref)
            if not cached:
                return "数据集引用已过期，需要重新确认一次数据集。", [
                    AgentEvent(kind="tool_result", text="⚠️ 数据集引用已过期")]
            examples = [TrainingSample(text=e["text"], label=e["label"]) for e in cached]

        ok, reason = await submit_training_job(
            task_id=task_id, description=state.raw_description,
            examples=examples, api_key=state.api_key,
            max_iterations=max_iterations, target_metric=target_metric,
            enable_phase2=state.enable_phase2,
            llm_provider=state.llm_provider, llm_model=state.llm_model,
            llm_base_url=state.llm_base_url, model_backend=model_backend,
            task_type=state.task_spec.task_type.value, env_description=state.env_description,
            instruction_examples=state.instruction_examples, base_model_id=state.base_model_id,
            user_id=state.user_id,
        )
        if not ok:
            return f"提交训练失败：{reason}", [AgentEvent(kind="tool_result", text=f"⚠️ 提交训练失败：{reason}")]

        state.model_backend = model_backend
        state.task_id = task_id
        start_training_relay(task_id)
        # 前端 reduceEvent 靠 snapshot 事件里的 task_id 才知道当前 task_id（后续 /predict
        # 调用要用）——task_store.events 队列本身不包含这条，这里跟 _start_training
        # 一样手工补一条，训练可视化才能跟 workflow 模式完全同构。用 AgentEvent 的
        # role/type_ 覆盖而不是直接调 state.add_message()，见 AgentEvent 类注释
        snapshot_event = AgentEvent(
            kind="training_snapshot", text="", role="training", type_="training_event",
            payload_override={"type": "snapshot", "task_id": task_id, "status": "queued"},
        )
        tool_result_event = AgentEvent(
            kind="tool_result", text=f"🚀 已提交训练任务（后端：{model_backend}）",
            data={"tool": "submit_training", "task_id": task_id, "model_backend": model_backend},
        )
        return (f"训练任务已提交（task_id={task_id}），需要用 check_training_progress 轮询真实进度/结果。",
                [snapshot_event, tool_result_event])

    _register(ToolDef(
        name="submit_training",
        description="提交训练任务——数据集/样本已经确认好之后调用。训练本身是确定性的，"
                    "会真实跑完，不是由你判定是否成功，之后要用 check_training_progress 查真实结果。",
        input_schema={
            "type": "object",
            "properties": {
                "model_backend": {"type": "string", "enum": ["sklearn", "pretrained_nn", "custom_nn", "rl", "llm_finetune"]},
                "max_iterations": {"type": "integer"},
                "target_metric": {"type": "number"},
            },
            "required": ["model_backend"],
        },
    ), _submit_training)

    # ── check_training_progress ──────────────────────────────────────────
    def _check_training_progress(inp: Dict[str, Any]) -> Tuple[str, List[AgentEvent]]:
        from api.store import task_store
        if not state.task_id:
            return "还没有提交训练任务。", []
        record = task_store.get(state.task_id)
        if not record:
            return "找不到训练任务记录。", []
        payload = {"status": record.status.value, "error": record.error}
        if record.result:
            payload["result"] = record.result
        text = json.dumps(payload, ensure_ascii=False)
        event = AgentEvent(kind="tool_result", text=f"📊 训练状态：{record.status.value}",
                            data={"tool": "check_training_progress", **payload})
        return text, [event]

    _register(ToolDef(
        name="check_training_progress",
        description="查询当前训练任务的真实状态和指标（如果已完成）。这是唯一的真相来源——"
                    "不要凭自己的判断认定训练是否成功，一切以这里返回的真实字段为准。",
        input_schema={"type": "object", "properties": {}, "required": []},
    ), _check_training_progress)

    # ── spawn_subagent ────────────────────────────────────────────────────
    # active_subagents/subagent_seq 是这次 build_agent_tools() 调用（也就是这次
    # AgentOrchestrator.run() 执行）范围内的闭包状态——子 Agent 全部在被
    # await 完之后才返回，函数返回时计数必然已经清零，不会跨 resume 泄漏。
    active_subagents = {"n": 0}
    subagent_seq = {"n": 0}

    async def _spawn_subagent(inp: Dict[str, Any]) -> Tuple[str, List[AgentEvent]]:
        if active_subagents["n"] >= MAX_CONCURRENT_SUBAGENTS:
            return (
                f"当前已有 {MAX_CONCURRENT_SUBAGENTS} 个子 Agent 在运行，达到并发上限，"
                f"请先处理已有的调研结果，不要一次性派生太多。"
            ), [AgentEvent(kind="tool_result", text="⚠️ 子 Agent 并发上限已达到，本次派生被拒绝")]

        role = inp.get("role", "researcher")
        task_description = inp.get("task_description", "")
        subagent_seq["n"] += 1
        subagent_id = f"{state.conversation_id}:subagent:{subagent_seq['n']}"

        spawn_event = AgentEvent(
            kind="subagent_spawned",
            text=f"🧑‍💻 派生子 Agent「{role}」：{task_description}",
            data={"tool": "spawn_subagent", "subagent_id": subagent_id, "role": role,
                  "task_description": task_description},
        )

        active_subagents["n"] += 1
        # 配额审计要能区分主 Agent 和子 Agent 的调用（决策点 #3）——client 若是
        # QuotaGatedClient，task_id 是记账时写进 llm_usage_events 的字段；这里
        # 用 duck typing 读写它，不 import api.accounts（core/ 不依赖 api/）。
        # 现在 complete_with_tools() 是真正的 async 方法、多个 spawn_subagent
        # 调用会被 agent_orchestrator.py 用 asyncio.gather 真并发跑——如果直接
        # "改写共享 client 的 task_id、跑完再恢复"，两个并发的子 Agent 会互相
        # 踩对方的 task_id（A 设完还没跑完就被 B 的设置覆盖，A 的用量被错误记
        # 到 B 头上）。用 copy.copy 给每个子 Agent 一份自己的浅拷贝：inner/
        # db_session_factory 这些还是指向同一个底层对象（该共享的东西不受影响），
        # 只有 task_id/last_usage 这两个实例属性是各自独立的，不会跨副本互相污染。
        has_task_id_attr = hasattr(client, "task_id")
        scoped_client = copy.copy(client) if has_task_id_attr else client
        if has_task_id_attr:
            scoped_client.task_id = subagent_id
        try:
            result = await run_subagent(role, task_description, scoped_client)
        finally:
            active_subagents["n"] -= 1

        done_event = AgentEvent(
            kind="subagent_done",
            text=f"✅ 子 Agent「{role}」完成：{result['summary']}",
            data={"tool": "spawn_subagent", "subagent_id": subagent_id, "role": role,
                  "success": result["success"], "summary": result["summary"]},
        )
        return result["summary"], [spawn_event, done_event]

    _register(ToolDef(
        name="spawn_subagent",
        description="派生一个短生命周期的只读研究子 Agent，让它去做一段独立的调研"
                    "（比如比较几个候选数据集、交叉核实某个判断），完成后把结论带回来。"
                    "子 Agent 不能训练模型、不能修改任何状态，只能搜索/预览数据集，"
                    "也不能再派生下一层子 Agent。只在真的需要额外调研时用——"
                    "简单、信息已经足够的任务不要用这个，直接自己继续做。",
        input_schema={
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": "子 Agent 的角色，比如\"数据集调研员\""},
                "task_description": {"type": "string", "description": "交给子 Agent 的具体调研任务"},
            },
            "required": ["role", "task_description"],
        },
    ), _spawn_subagent)

    # ── finish_run ────────────────────────────────────────────────────────
    def _finish_run(inp: Dict[str, Any]) -> Tuple[str, List[AgentEvent]]:
        summary = inp.get("summary", "任务已完成。")
        return summary, [AgentEvent(kind="final", text=summary)]

    _register(ToolDef(
        name="finish_run",
        description="任务已经完成（或者已经没有更多可以做的事）时调用，结束本轮执行并向用户总结结果。",
        input_schema={
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "给用户看的总结"}},
            "required": ["summary"],
        },
    ), _finish_run)

    return tools, dispatch
