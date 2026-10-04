"""
api/accounts/llm_provisioning.py  -  给调用方（core/pipeline.py 等）用的 client 构建入口。

调用方不再直接调 core/llm_client.py::build_llm_client()，而是调这里的
build_client_for_request()——provider 不是 "system_managed" 时原样透传；
是的话，先拿到内层 AnthropicClient，再包一层 QuotaGatedClient 返回。
这样账号/DB 相关的知识全部封在 api/accounts/ 里，core/llm_client.py 保持
零账号依赖（唯一例外是读 api/settings.py 拿系统 key 本身，这是纯配置读取，
不是账号/DB 耦合）。
"""

from __future__ import annotations

from typing import Optional

from core.llm_client import LLMClient, build_llm_client
from api.accounts.db import SessionLocal
from api.accounts.quota_gated_client import QuotaGatedClient


class LoginRequiredError(Exception):
    """system_managed 需要登录身份才能计量配额——单独的异常类型，不用 ValueError，
    避免和调用方（比如 api/routes/conversations.py）已有的"会话不存在"式
    ValueError 处理混在一起被错误地映射成 404（这本质是 401，需要登录）。"""


def build_client_for_request(
    provider: str,
    api_key: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    user_id=None,
    task_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> LLMClient:
    inner = build_llm_client(provider, api_key, model, base_url)
    if provider != "system_managed":
        # BYOK（anthropic/ollama/openai_compatible 且自带 api_key）完全不经过配额网关，
        # 不写 llm_usage_events，不受配额限制——用户自己的钱自己的责任。
        return inner
    if user_id is None:
        raise LoginRequiredError("使用系统托管的商业 API 需要先登录；也可以选本地/自建服务或填自己的 API Key，不需要登录。")
    return QuotaGatedClient(inner, user_id, SessionLocal, task_id=task_id, conversation_id=conversation_id)
