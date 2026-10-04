"""api/routes/conversations.py  -  对话式交互端点（多轮追问 → 数据准备 → 模型选择 → 训练 → 播报）"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from api.models import (
    ConversationCreatedResponse, CreateConversationRequest,
    PostMessageRequest, ConversationMessageModel,
)
from api.conversation_store import conversation_store
from api.accounts.deps import get_current_user, get_current_user_or_api_token
from api.accounts.models_db import User
from api.accounts.schemas import ConversationSummaryItem
from api.accounts.llm_provisioning import LoginRequiredError
from api.accounts.quota_gated_client import QuotaExceededError
from core.llm_client import SystemManagedNotConfiguredError
from core.conversation.orchestrator import ConversationOrchestrator
from core.conversation.state import ConversationMessage

router = APIRouter(prefix="/api/conversations", tags=["Conversations"])
orchestrator = ConversationOrchestrator(conversation_store)


def _serialize_message(m: ConversationMessage) -> dict:
    return {
        "role": m.role, "type": m.type, "payload": m.payload,
        "ui_hint": m.ui_hint, "data": m.data, "created_at": m.created_at,
    }


@router.post("", response_model=ConversationCreatedResponse, status_code=201)
async def create_conversation(req: CreateConversationRequest,
                              user_and_token=Depends(get_current_user_or_api_token)):
    # 未登录也能建会话（BYOK/本地 provider 不需要账号）——只有 llm_provider == "system_managed"
    # 才真的要求登录，那个校验发生在 api/accounts/llm_provisioning.py::build_client_for_request 里，
    # 这里不重复拦截，保持"匿名也能用本地/自带 Key"的现有体验不变。
    user, api_token = user_and_token
    conversation_id = str(uuid.uuid4())
    state = conversation_store.create(conversation_id)
    if api_token is not None:
        # API token 发起的会话——provider 配置完全用 token 自己预先配置好的那一份，
        # 忽略请求体里可能带的同名字段（不能既让调用方传、又让 token 覆盖，
        # 两边不一致时到底信谁会很含糊，所以直接不采信请求体这几个字段）
        state.llm_provider = api_token.llm_provider
        state.llm_model = api_token.llm_model
        state.api_key = api_token.llm_api_key
        state.llm_base_url = api_token.llm_base_url
        state.origin = "api"
    else:
        state.llm_provider = req.llm_provider
        state.llm_model = req.llm_model
        state.api_key = req.api_key
        state.llm_base_url = req.llm_base_url
        state.origin = "web"
    state.user_id = user.id if user else None
    if user is not None:
        conversation_store.index_for_user(conversation_id, str(user.id))
    state.orchestration_mode = req.orchestration_mode
    state.mcp_server_urls = req.mcp_server_urls
    state.enabled_skills = req.enabled_skills
    return ConversationCreatedResponse(
        conversation_id=conversation_id, ws_url=f"/api/conversations/{conversation_id}/stream")


@router.get("", response_model=List[ConversationSummaryItem])
async def list_my_conversations(user: User = Depends(get_current_user)):
    """登录浏览器用户查看自己名下的全部会话（网页发起的 + 通过 API token 发起的），
    前端据 origin 字段渲染"API"标记——只有这个端点要求强制登录（JWT），因为这是
    "查看我自己的东西"，不是训练/对话本身那种可以匿名用的操作。"""
    states = conversation_store.list_for_user(str(user.id))
    states.sort(key=lambda s: s.created_at, reverse=True)
    return [
        ConversationSummaryItem(
            conversation_id=s.conversation_id, origin=s.origin, stage=s.stage.value,
            task_type=s.task_spec.task_type.value if s.task_spec else None,
            created_at=s.created_at,
        )
        for s in states
    ]


@router.post("/{conversation_id}/messages", response_model=List[ConversationMessageModel])
async def post_message(conversation_id: str, req: PostMessageRequest):
    if conversation_store.get(conversation_id) is None:
        raise HTTPException(404, f"会话不存在：{conversation_id}")
    try:
        new_messages = await orchestrator.handle_message(
            conversation_id, kind=req.kind, text=req.text or "", structured=req.structured,
            attachments=req.attachments)
    except LoginRequiredError as e:
        raise HTTPException(401, str(e))
    except QuotaExceededError as e:
        raise HTTPException(429, str(e))
    except SystemManagedNotConfiguredError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(404, str(e))
    return [_serialize_message(m) for m in new_messages]


@router.get("/{conversation_id}/messages", response_model=List[ConversationMessageModel])
async def get_messages(conversation_id: str):
    state = conversation_store.get(conversation_id)
    if state is None:
        raise HTTPException(404, f"会话不存在：{conversation_id}")
    return [_serialize_message(m) for m in state.messages]


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str):
    # 前端"历史会话"列表本身存在 localStorage 里，删除主要是那边的操作——这个
    # 端点只是同时把服务端内存里的会话真正释放掉，不留孤儿状态占内存（不删的话
    # 反正也会被 MAX_CONVERSATIONS 的 LRU 淘汰，但用户主动点删除时应该立刻生效）
    if not conversation_store.delete(conversation_id):
        raise HTTPException(404, f"会话不存在：{conversation_id}")


@router.post("/{conversation_id}/fork", response_model=ConversationCreatedResponse, status_code=201)
async def fork_conversation(conversation_id: str):
    new_id = str(uuid.uuid4())
    forked = conversation_store.fork(conversation_id, new_id)
    if forked is None:
        raise HTTPException(404, f"会话不存在：{conversation_id}")
    return ConversationCreatedResponse(
        conversation_id=new_id, ws_url=f"/api/conversations/{new_id}/stream")


@router.websocket("/{conversation_id}/stream")
async def stream_conversation(ws: WebSocket, conversation_id: str):
    state = conversation_store.get(conversation_id)
    if state is None:
        await ws.close(code=1008, reason="Not found")
        return

    await ws.accept()
    # 上一个连到同一个 conversation_id 的 WS（比如客户端网络抖动重连、或者一个连接建立后
    # 还没来得及消费就被关掉）可能在共享队列里留下了消息——这些消息在被 push_live 之前
    # 已经写进了 state.messages（见 orchestrator.py::handle_message），下面这份快照已经
    # 完整包含它们了，所以这里连接建立时先把队列里的陈旧积压清空，否则会在 snapshot 之后
    # 把同样的消息当成"新消息"再推一遍，前端出现重复气泡
    conversation_store.drain_live(conversation_id)

    # 注意：外层 WS 帧用 "frame" 字段区分 snapshot/heartbeat/message，不能用 "type"——
    # ConversationMessage 自己也有一个 "type" 字段（question/text/stage_changed/training_event），
    # 两者同名会在 dict 展开时互相覆盖
    await ws.send_text(json.dumps({
        "frame": "snapshot",
        "messages": [_serialize_message(m) for m in state.messages],
    }, default=str))

    try:
        while True:
            try:
                msg = await conversation_store.next_live(conversation_id, timeout=120.0)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"frame": "heartbeat"}))
                continue
            await ws.send_text(json.dumps({"frame": "message", "message": _serialize_message(msg)}, default=str))
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
