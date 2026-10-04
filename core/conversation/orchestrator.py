"""
core/conversation/orchestrator.py  -  对话编排器

驱动一次对话从"用户随便说一句话"走到"训练完成播报结果"：
  1. 每条用户消息先过一遍"意图跳转"检查（改主意换后端之类）
  2. 从当前阶段开始，按 STAGE_ORDER 顺序调用对应 handler；
     satisfied=True 就写回 state、推进到下一阶段继续检查——这样"一次性说清楚
     任务描述"的用户会被直接带到下一个真正需要澄清的地方，不会被无意义地重复追问
  3. 到达 TRAINING 阶段时，直接调用现成的 api.worker.submit_training_job（不改
     它一行代码），后台订阅这个 task 的事件队列，把每个原始 pipeline 事件转发进
     对话消息流——前端拿到后照样喂给现成的 reduceEvent()，训练可视化完全复用
  4. 训练结束（finished/error）自动切到 REPORTING 阶段，用结果生成一条播报消息；
     用户回复"满意"就结束，回复"想调整"就退回 CHOOSING_MODEL 重新走一轮
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from typing import Any, Dict, List, Optional

from api.accounts.llm_provisioning import build_client_for_request
from . import memory
from .state import ConversationState, ConversationStage, STAGE_ORDER, ConversationMessage
from .stages import STAGE_HANDLERS

_SATISFIED_KEYWORDS = ("满意", "挺好", "可以了", "行了", "够了", "不用了", "没问题", "很好", "ok", "OK")
# "不满意"/"不行" 这类否定说法本身会作为子串命中上面的正向关键词（"不满意" 包含"满意"），
# 必须先排除，否则用户明确表示不满意时会被误判成满意
_UNSATISFIED_KEYWORDS = ("不满意", "不行", "不可以", "不够", "不ok", "不OK", "不没问题")
_ADJUST_INTENT_KEYWORDS = ("换成", "改用", "换个", "改成", "重新")
_BACKEND_NAMES = {
    "sklearn":   "sklearn",
    "预训练":     "pretrained_nn",
    "pretrained_nn": "pretrained_nn",
    "自定义网络":  "custom_nn",
    "custom_nn": "custom_nn",
}


def _looks_like_satisfied(text: str) -> bool:
    if not text or any(k in text for k in _UNSATISFIED_KEYWORDS):
        return False
    return any(k in text for k in _SATISFIED_KEYWORDS)


def _maybe_apply_intent_override(state: ConversationState, text: str) -> None:
    """轻量"改主意"检测——命中就直接改字段，不做复杂的阶段跳转，避免状态机风险。
    训练进行中的情况由 handle_message 在调用这里之前就已经拦掉了。"""
    if not text or state.stage == ConversationStage.CLARIFYING_TASK:
        return
    if not any(k in text for k in _ADJUST_INTENT_KEYWORDS):
        return
    for name, backend in _BACKEND_NAMES.items():
        if name in text:
            state.model_backend = backend
            return


def _compose_effective_text(text: str, attachments: Optional[List[dict]]) -> str:
    """把附件（文档提取出的文本/粘贴的长文本）拼进"真正喂给阶段处理器/LLM 的文本"里，
    但绝不能直接改写 text 本身——text 原样存进 state.messages 给用户自己的聊天气泡
    显示（附件另外以 data.attachments 的形式挂在同一条消息上，前端渲染成卡片，
    不是把一大段提取出来的文档内容糊在气泡文字里）。只有 extracted_text 非空的附件
    （文档/粘贴文本）才拼进来——单纯的参考图片没有可拼接的文本内容，
    parse_once/prepare_data 等阶段处理器该怎么处理图片附件是另一回事，不在这里管。"""
    if not attachments:
        return text
    parts = [text] if text and text.strip() else []
    for a in attachments:
        extracted = a.get("extracted_text")
        if extracted:
            parts.append(f"[附件：{a.get('filename', '未命名文件')}]\n{extracted}")
    return "\n\n".join(parts) if parts else text


def _describe_structured(structured: dict) -> str:
    t = structured.get("type")
    if t == "data_selected":
        if structured.get("dataset_ref"):
            return f"（已选择数据集：{structured['dataset_ref']}）"
        if structured.get("image_examples") is not None:
            examples = structured.get("image_examples") or []
            labels = sorted(set(e.get("label") for e in examples if e.get("label")))
            return f"（已上传 {len(examples)} 张图片，覆盖 {len(labels)} 个类别）"
        if structured.get("vlm_examples") is not None:
            return f"（已上传 {len(structured.get('vlm_examples') or [])} 条图文样本）"
        return f"（已提供 {len(structured.get('examples') or [])} 条手动样本）"
    if t == "backend_selected":
        return f"（已选择训练后端：{structured.get('model_backend')}）"
    if t == "training_config":
        return "（已设置训练参数）"
    return "（已提交）"


def _build_report_text(event: Optional[Dict[str, Any]]) -> str:
    if not event or event.get("type") == "error":
        reason = (event or {}).get("message", "未知错误")
        return f"训练失败了：{reason}。要不要换个方式重试？"
    metric_name = event.get("metric_name", "指标")
    best = event.get("best_metric", 0) or 0
    deploy_path = event.get("deploy_path")
    # episode_reward_mean（强化学习）不是 0-1 有界的比例，不能按百分比格式化——
    # 不同指标只有分类/回归的 f1/accuracy 这类才适合 :.1%，RL 的 reward 量纲因环境而异
    metric_str = f"{best:.1%}" if metric_name != "episode_reward_mean" else f"{best:.2f}"
    lines = [f"训练完成！最佳 {metric_name} = {metric_str}。"]
    if deploy_path:
        lines.append(f"已自动导出部署包：{deploy_path}")
    lines.append("对这个结果满意吗？满意的话就到这里；不满意可以告诉我想调整什么"
                 "（比如换个训练后端、加训练轮数），我们重新来一轮。")
    return "\n".join(lines)


class ConversationOrchestrator:
    def __init__(self, conv_store):
        self.conv_store = conv_store

    async def handle_message(
        self, conversation_id: str, kind: str,
        text: str = "", structured: Optional[dict] = None,
        attachments: Optional[List[dict]] = None,
    ) -> List[ConversationMessage]:
        """对外入口：跑完 _handle_message_inner 之后，把这一轮新增的所有消息（用户回显、
        追问、阶段切换……）都推到实时队列——这样前端只需要盯着一条 WS 就能拿到全部更新，
        不用再区分"这条消息是 REST 响应带回来的"还是"那条是训练事件异步推来的"。
        attachments：聊天输入框"添加文件"上传的附件（文档已在 api/routes/attachments.py
        提取出文本，图片没有）——真正喂给阶段处理器的是 text+附件文本拼起来的
        effective_text，用户自己聊天气泡里显示的仍然是原始 text，见
        _compose_effective_text 的注释。"""
        state = self.conv_store.get(conversation_id)
        if state is None:
            raise ValueError(f"会话不存在：{conversation_id}")

        effective_text = _compose_effective_text(text, attachments)
        new_messages = await self._handle_message_inner(
            state, kind, text, structured, effective_text=effective_text, attachments=attachments)
        for msg in new_messages:
            self.conv_store.push_live(conversation_id, msg)
        return new_messages

    async def _handle_message_inner(
        self, state: ConversationState, kind: str,
        text: str = "", structured: Optional[dict] = None,
        effective_text: Optional[str] = None, attachments: Optional[List[dict]] = None,
    ) -> List[ConversationMessage]:
        start_idx = len(state.messages)
        if effective_text is None:
            effective_text = text

        if kind == "text" and (text.strip() or attachments):
            # 附件本身没有配文字也要显示——用户可能就是单纯拖了个文件进来，
            # 不代表这一轮"什么都没说"
            state.add_message(role="user", type_="text", payload=text,
                              data={"attachments": attachments} if attachments else None)
        elif kind == "structured" and structured:
            state.add_message(role="user", type_="text", payload=_describe_structured(structured))

        # 训练进行中，拒绝修改配置
        if state.stage == ConversationStage.TRAINING and state.task_id:
            state.add_message(role="assistant", type_="text",
                              payload="训练正在进行中，暂时不能修改配置——等这一轮跑完再说吧。")
            return state.messages[start_idx:]

        # client 构造本身只是个轻量对象（不发请求），提前建好这样 REPORTING 分支也能用上，
        # 供 _handle_reporting 里 choose_model 生成"结合上一轮结果"的针对性建议
        client = build_client_for_request(
            state.llm_provider, state.api_key, state.llm_model, state.llm_base_url,
            user_id=state.user_id, conversation_id=state.conversation_id,
        )

        # Multi-Agent 模式完全不走下面的 STAGE_ORDER 循环/REPORTING——state.stage
        # 在这个模式下始终停在初始值，"训练中/播报"这些概念由 Agent 自己的
        # finish_run 工具和 check_training_progress 工具承担，不复用这套状态机
        if state.orchestration_mode == "multi_agent":
            await self._handle_multi_agent(state, effective_text, structured, client)
            return state.messages[start_idx:]

        if state.stage == ConversationStage.REPORTING:
            self._handle_reporting(state, text, client)
            return state.messages[start_idx:]

        _maybe_apply_intent_override(state, text)

        first_pass = True
        while state.stage not in (ConversationStage.TRAINING, ConversationStage.REPORTING):
            handler = STAGE_HANDLERS[state.stage]
            cur_text = effective_text if first_pass else ""
            cur_structured = structured if first_pass else None
            first_pass = False

            # prepare_data 是 async def（内部可能真实调用 HF/魔搭网络请求），
            # 其它几个 handler 保持同步——这里统一用 isawaitable 判断要不要
            # await，不强行把所有 handler 都改成 async
            outcome = handler(state, cur_text, cur_structured, client)
            result = await outcome if inspect.isawaitable(outcome) else outcome

            if not result.satisfied:
                extra_data = state.recommended_dataset if result.ui_hint == "dataset_recommendation" else None
                state.add_message(role="assistant", type_="question",
                                  payload=result.question, ui_hint=result.ui_hint, data=extra_data)
                return state.messages[start_idx:]

            for k, v in result.updates.items():
                setattr(state, k, v)

            idx = STAGE_ORDER.index(state.stage)
            state.stage = STAGE_ORDER[idx + 1]
            if state.stage not in (ConversationStage.TRAINING, ConversationStage.REPORTING):
                state.add_message(role="assistant", type_="stage_changed", payload=state.stage.value)

        if state.stage == ConversationStage.TRAINING and not state.task_id:
            await self._start_training(state)

        return state.messages[start_idx:]

    # ── Multi-Agent 模式入口 ──────────────────────────────────────────────────

    async def _handle_multi_agent(
        self, state: ConversationState, text: str, structured: Optional[dict], client,
    ) -> None:
        """任务澄清仍然复用现成的 clarify_task 一次性问答拿到 task_spec——这一步
        本来就是单轮结构化提取，没有"自主决策"的必要；拿到 task_spec 之后才真正
        交给 AgentOrchestrator 驱动 prepare_data 往后的全部决策。

        AgentOrchestrator.run() 用 asyncio.create_task(...) 甩到后台跑，不在这里
        await 到底——单轮循环最多 15 次工具调用，真同步等完再返回的话，这次
        handle_message 调用要等一大段时间才返回，期间用户的 WS 上什么都收不到，
        直到全部跑完才一次性推一大堆消息过去，跟"实时看到 Agent 正在做什么"这个
        需求直接矛盾。跟 _start_training 甩出 _relay_training_events 是同一个模式：
        这个方法自己只做"快"的那部分（澄清任务的一次性问答/回显），返回后台任务
        自己产出的消息通过 push_live 实时推送，不经过 handle_message() 外层那个
        "整批一次性推送"的循环，避免同一条消息被推两次。"""
        from core.agent.agent_orchestrator import AgentOrchestrator

        def _start_relay(task_id: str) -> None:
            asyncio.create_task(self._relay_training_events(state.conversation_id, task_id))

        def _push_live(msg: ConversationMessage) -> None:
            self.conv_store.push_live(state.conversation_id, msg)

        async def _deterministic_fallback() -> None:
            """Agent 一次工具都没调用成功时的兜底：改跑确定性状态机。

            前端已经去掉了"传统方式/Multi-Agent"的用户可见开关，统一走 Agent；
            但确定性流程并没有删掉，而是降级成这里的内部兜底——本地小模型的
            function calling 能力参差不齐，没有这条路的话，这批用户会直接
            拿不到训练结果（Agent 只回一句闲聊就结束）。

            走的是跟 workflow 模式一模一样的 STAGE_ORDER 循环，不是另写一套：
            把 stage 推进到 PREPARING_DATA 之后交给 _handle_message_inner
            的既有逻辑，产出的消息同样通过 push_live 实时推给前端。"""
            # ⚠ 必须先把 orchestration_mode 切走，否则 _handle_message_inner 会
            # 再次命中 `if state.orchestration_mode == "multi_agent"` 分支，
            # 又调回 _handle_multi_agent —— 无限递归。
            #
            # 而且这个切换是**粘性的**（不切回去）：既然已经确认这个模型发不出
            # 合法的工具调用，就没必要在之后每条消息上再花十几次 LLM 调用去
            # 重新发现同一件事。用户换模型后重开会话即可回到 Agent 模式。
            state.orchestration_mode = "workflow"
            if state.stage == ConversationStage.CLARIFYING_TASK and state.task_spec is not None:
                # task_spec 已经在上面 clarify_task 拿到了，直接进入下一阶段，
                # 否则确定性循环会把同一个澄清问题再问一遍
                state.stage = ConversationStage.PREPARING_DATA
            before = len(state.messages)
            # kind 故意不传 "text"/"structured"：用户那条消息在最外层
            # _handle_message_inner 早就已经加进 state.messages 了，这里再传
            # "text" 会让同一句话在聊天记录里出现两遍。effective_text 仍然
            # 默认取 text，阶段处理器拿到的输入不受影响。
            await self._handle_message_inner(state, "fallback_resume", text, structured)
            for msg in state.messages[before:]:
                self.conv_store.push_live(state.conversation_id, msg)

        if state.task_spec is None:
            result = STAGE_HANDLERS[ConversationStage.CLARIFYING_TASK](state, text, structured, client)
            if not result.satisfied:
                state.add_message(role="assistant", type_="question",
                                  payload=result.question, ui_hint=result.ui_hint)
                return
            for k, v in result.updates.items():
                setattr(state, k, v)
            state.add_message(role="assistant", type_="stage_changed", payload="multi_agent_start")
            asyncio.create_task(AgentOrchestrator().run(
                state, "", None, client, start_training_relay=_start_relay, push_live=_push_live,
                deterministic_fallback=_deterministic_fallback))
            return

        asyncio.create_task(AgentOrchestrator().run(
            state, text, structured, client, start_training_relay=_start_relay, push_live=_push_live,
            deterministic_fallback=_deterministic_fallback))

    # ── 训练启动 + 事件转发 ──────────────────────────────────────────────────

    async def _start_training(self, state: ConversationState) -> None:
        from api.models import TrainingSample
        from api.worker import submit_training_job
        from api.store import task_store
        from api.dataset_store import dataset_store
        from config import TaskType

        task_id = str(uuid.uuid4())
        task_store.create(task_id)

        if state.task_spec.task_type == TaskType.RL:
            # RL 没有 examples/dataset_ref，用 prepare_data 阶段收集的 env_description
            ok, reason = await submit_training_job(
                task_id=task_id, description=state.task_spec.raw_description,
                api_key=state.api_key, max_iterations=state.max_iterations,
                source_id=state.conversation_id, llm_provider=state.llm_provider,
                llm_model=state.llm_model, llm_base_url=state.llm_base_url,
                task_type="rl", env_description=state.env_description, user_id=state.user_id,
            )
        elif state.task_spec.task_type == TaskType.LLM_FINETUNE:
            # LLM 微调没有 examples/dataset_ref，用 prepare_data 阶段收集的
            # instruction_examples + choose_model 阶段校验过的 base_model_id
            ok, reason = await submit_training_job(
                task_id=task_id, description=state.task_spec.raw_description,
                api_key=state.api_key, max_iterations=state.max_iterations,
                source_id=state.conversation_id, llm_provider=state.llm_provider,
                llm_model=state.llm_model, llm_base_url=state.llm_base_url,
                task_type="llm_finetune", instruction_examples=state.instruction_examples,
                base_model_id=state.base_model_id, user_id=state.user_id,
            )
        elif state.task_spec.task_type == TaskType.IMAGE_CLASSIFICATION:
            # 图像分类没有 examples/dataset_ref，用 prepare_data 阶段收集的
            # image_examples（choose_model 阶段已经把固定的 CLIP 编码器 id
            # 写进了 state.base_model_id，跟 LLM_FINETUNE 复用同一个字段）
            ok, reason = await submit_training_job(
                task_id=task_id, description=state.task_spec.raw_description,
                api_key=state.api_key, max_iterations=state.max_iterations,
                source_id=state.conversation_id, llm_provider=state.llm_provider,
                llm_model=state.llm_model, llm_base_url=state.llm_base_url,
                task_type="image_classification", image_examples=state.image_examples,
                base_model_id=state.base_model_id, user_id=state.user_id,
            )
        elif state.task_spec.task_type == TaskType.VLM_GENERATIVE:
            # 生成式 VLM 没有 examples/dataset_ref，用 prepare_data 阶段收集的
            # vlm_examples + choose_model 阶段写入的默认底座模型 id
            ok, reason = await submit_training_job(
                task_id=task_id, description=state.task_spec.raw_description,
                api_key=state.api_key, max_iterations=state.max_iterations,
                source_id=state.conversation_id, llm_provider=state.llm_provider,
                llm_model=state.llm_model, llm_base_url=state.llm_base_url,
                task_type="vlm_generative", vlm_examples=state.vlm_examples,
                base_model_id=state.base_model_id, user_id=state.user_id,
            )
        else:
            if state.examples:
                examples = [TrainingSample(text=e["text"], label=e["label"]) for e in state.examples]
            else:
                cached = dataset_store.get(state.dataset_ref)
                if not cached:
                    state.add_message(role="assistant", type_="text",
                                      payload="数据集引用已过期，请重新提供训练数据。")
                    state.stage = ConversationStage.PREPARING_DATA
                    return
                examples = [TrainingSample(text=e["text"], label=e["label"]) for e in cached]

            ok, reason = await submit_training_job(
                task_id=task_id, description=state.task_spec.raw_description, examples=examples,
                api_key=state.api_key, max_iterations=state.max_iterations, target_metric=state.target_metric,
                enable_phase2=state.enable_phase2, source_id=state.conversation_id,
                llm_provider=state.llm_provider, llm_model=state.llm_model, llm_base_url=state.llm_base_url,
                model_backend=state.model_backend, user_id=state.user_id,
            )
        if not ok:
            state.add_message(role="assistant", type_="text", payload=f"提交训练失败：{reason}")
            state.stage = ConversationStage.CONFIGURING_TRAINING
            return

        state.task_id = task_id
        # 前端 reduceEvent 靠 "snapshot" 事件里的 task_id 才知道当前 task_id（用于后续 /predict 调用）——
        # task_store.events 队列本身不包含这条，tasks.py 的 WS 是在连接时手工拼出来的，这里同样手工补一条
        state.add_message(role="training", type_="training_event",
                          payload={"type": "snapshot", "task_id": task_id, "status": "queued"})
        state.add_message(role="assistant", type_="text", payload="好的，已经开始训练，实时进展会显示在下面。")
        asyncio.create_task(self._relay_training_events(state.conversation_id, task_id))

    async def _relay_training_events(self, conversation_id: str, task_id: str) -> None:
        from api.store import task_store

        record = task_store.get(task_id)
        if not record:
            return

        final_event: Optional[Dict[str, Any]] = None
        while True:
            try:
                event = await asyncio.wait_for(record.events.get(), timeout=130.0)
            except asyncio.TimeoutError:
                continue
            state = self.conv_store.get(conversation_id)
            if state is None:
                return
            msg = state.add_message(role="training", type_="training_event", payload=event)
            self.conv_store.push_live(conversation_id, msg)
            if event.get("type") in ("finished", "error"):
                final_event = event
                break

        state = self.conv_store.get(conversation_id)
        if state is None:
            return

        # 记忆：把这一轮训练的结果记进精简摘要，供后面万一要重试时给出针对性建议；
        # 摘要/消息列表过长时顺带压缩/裁剪——这三步都不影响下面的播报逻辑，失败也不报错
        client = build_client_for_request(
            state.llm_provider, state.api_key, state.llm_model, state.llm_base_url,
            user_id=state.user_id, conversation_id=state.conversation_id,
        )
        if final_event and final_event.get("type") == "error":
            memory.record_fact(state, f"训练失败：backend={state.model_backend}，原因：{final_event.get('message', '未知错误')}")
        elif final_event:
            memory.record_fact(
                state, f"训练完成：backend={state.model_backend}，"
                       f"{final_event.get('metric_name')}={final_event.get('best_metric')}")
        memory.trim_messages(state)
        memory.maybe_compress(state, client)

        state.stage = ConversationStage.REPORTING
        msg = state.add_message(role="assistant", type_="text", payload=_build_report_text(final_event))
        self.conv_store.push_live(conversation_id, msg)

    # ── reporting 阶段：满意就结束，不满意就回退重新配置 ─────────────────────

    def _handle_reporting(self, state: ConversationState, text: str, client) -> None:
        if _looks_like_satisfied(text):
            state.add_message(role="assistant", type_="text",
                              payload="好的，这次就到这里，随时可以开始新的任务。")
            return
        memory.record_fact(state, f"用户反馈（不满意）：{text}")
        state.stage = ConversationStage.CHOOSING_MODEL
        state.task_id = None
        state.add_message(role="assistant", type_="text", payload="好，那我们调整一下再跑一次。")
        # 立即把 choose_model 的开场问题问出来，而不是等用户再发一条消息才触发；
        # 传真实 client（不再是 None）——choose_model 内部会用 state.memory_summary
        # 生成一句结合上一轮结果的针对性建议
        result = STAGE_HANDLERS[ConversationStage.CHOOSING_MODEL](state, "", None, client)
        if not result.satisfied:
            state.add_message(role="assistant", type_="question",
                              payload=result.question, ui_hint=result.ui_hint)
