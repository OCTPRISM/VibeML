"""
api/accounts/remote_client.py  -  本机版桌面应用访问远程账号/配额服务器用的瘦客户端。

本机版没有（也不该有）系统自己的商业 API Key——真正的调用只能发生在持有这把
key 的远程服务器上。这里只打两类端点：
  1. /api/auth/*                     登录/刷新（复用和网络版完全一样的账号系统）
  2. /api/accounts/complete           实际的 system_managed 调用代理

不是完整的 /api/tasks 那一整套——本机版不往远程提交训练任务，训练本身
（core/pipeline.py/core/rl_pipeline.py）继续在本机跑，只有"用系统托管的 Key
问一句"这一步才需要联网。

RemoteSystemManagedClient 实现的是和 core/llm_client.py::LLMClient 一样的
接口（complete(system, user, max_tokens) -> str），这样本机版的调用方
（core/pipeline.py 等）完全不需要区分"这是本地 client 还是远程代理 client"，
复用现有的 provider 分发逻辑即可。
"""

from __future__ import annotations

from typing import Optional

import httpx

from core.llm_client import LLMClient


class RemoteAuthError(Exception):
    """远程账号服务器登录/刷新失败（凭据错误、网络不通等）。"""


class RemoteAccountClient:
    """登录/刷新——本机版启动时用这个换到 access token，供 RemoteSystemManagedClient 使用。"""

    def __init__(self, base_url: str, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def login(self, email: str, password: str) -> str:
        try:
            resp = self._client.post("/api/auth/login", json={"email": email, "password": password})
        except httpx.HTTPError as e:
            raise RemoteAuthError(f"无法连接账号服务器：{e}")
        if not resp.is_success:
            raise RemoteAuthError(resp.json().get("detail", f"HTTP {resp.status_code}"))
        return resp.json()["access_token"]

    def refresh(self, refresh_cookie_jar: httpx.Cookies) -> str:
        try:
            resp = self._client.post("/api/auth/refresh", cookies=refresh_cookie_jar)
        except httpx.HTTPError as e:
            raise RemoteAuthError(f"无法连接账号服务器：{e}")
        if not resp.is_success:
            raise RemoteAuthError(resp.json().get("detail", f"HTTP {resp.status_code}"))
        return resp.json()["access_token"]

    def close(self) -> None:
        self._client.close()


class RemoteSystemManagedClient(LLMClient):
    """system_managed 在本机版下的实现——不直接持有商业 API Key，而是把请求转发到
    远程账号服务器的 /api/accounts/complete，由服务器代为调用真正的 Anthropic API
    并做配额计量。access_token 由 RemoteAccountClient.login()/refresh() 提供，
    过期时调用方需要自己重新登录/刷新（这里不做自动重试，保持这个类单一职责）。"""

    def __init__(self, base_url: str, access_token: str, model: Optional[str] = None,
                task_id: Optional[str] = None, conversation_id: Optional[str] = None,
                timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self.model = model
        self.task_id = task_id
        self.conversation_id = conversation_id
        self.timeout = timeout

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/accounts/complete",
            headers={"Authorization": f"Bearer {self.access_token}"},
            json={
                "system": system, "user": user, "max_tokens": max_tokens,
                "model": self.model, "task_id": self.task_id, "conversation_id": self.conversation_id,
            },
            timeout=self.timeout,
        )
        if not resp.is_success:
            detail = resp.json().get("detail", f"HTTP {resp.status_code}")
            raise RuntimeError(f"远程 system_managed 调用失败：{detail}")
        return resp.json()["text"]
