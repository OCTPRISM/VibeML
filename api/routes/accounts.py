"""
api/routes/accounts.py  -  system_managed 的远程代理端点。

本机版桌面应用没有（也不该有）系统自己的商业 API Key——真正的 Anthropic 调用
只能发生在持有这把 key 的服务器上。这个端点就是那个服务器侧入口：本机版的
api/accounts/remote_client.py::RemoteSystemManagedClient 把 system/user/max_tokens
POST 到这里，服务器用已经验证过的 QuotaGatedClient（配额检查 + token 用量落库，
和网络版浏览器直连的路径完全一致，不是另开一套逻辑）跑真正的调用，把结果文本
原样返回。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.accounts.deps import get_current_user
from api.accounts.llm_provisioning import build_client_for_request
from api.accounts.models_db import User
from api.accounts.quota_gated_client import QuotaExceededError
from core.llm_client import SystemManagedNotConfiguredError

router = APIRouter(prefix="/api/accounts", tags=["Accounts"])


class RemoteCompleteRequest(BaseModel):
    system: str
    user: str
    max_tokens: int = 1000
    model: str | None = None
    task_id: str | None = None
    conversation_id: str | None = None


class RemoteCompleteResponse(BaseModel):
    text: str


@router.post("/complete", response_model=RemoteCompleteResponse)
async def system_managed_complete(req: RemoteCompleteRequest, user: User = Depends(get_current_user)):
    try:
        client = build_client_for_request(
            "system_managed", None, req.model, None,
            user_id=user.id, task_id=req.task_id, conversation_id=req.conversation_id,
        )
        text = client.complete(req.system, req.user, req.max_tokens)
    except QuotaExceededError as e:
        raise HTTPException(429, str(e))
    except SystemManagedNotConfiguredError as e:
        raise HTTPException(503, str(e))
    return RemoteCompleteResponse(text=text)
