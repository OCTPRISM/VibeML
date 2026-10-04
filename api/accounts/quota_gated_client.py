"""
api/accounts/quota_gated_client.py  -  system_managed 调用的配额网关 + token 用量落库。

按用户确认的"每次 LLM 调用计数，且要有 token 消耗清单供审计"设计：
一次调用 = 一行 llm_usage_events，这张表既是配额计数依据，也直接就是审计清单本体。

不放在 core/llm_client.py 里——那个模块要保持零账号/DB 依赖，纯 LLM 传输抽象。
这里包一层内层 client（通常是 build_llm_client("system_managed", ...) 产出的
AnthropicClient），账号/DB 相关的知识全部封在这一层。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from core.llm_client import AgentMessage, CompletionResult, LLMClient, ToolDef
from api.accounts.models_db import LLMUsageEvent
from api.settings import settings


class QuotaExceededError(Exception):
    """本月配额已用尽。"""


def current_period() -> str:
    now = datetime.now(timezone.utc)
    return f"{now.year:04d}-{now.month:02d}"


def calls_used_this_period(db: Session, user_id) -> int:
    return (
        db.query(LLMUsageEvent)
        .filter(LLMUsageEvent.user_id == user_id, LLMUsageEvent.period == current_period())
        .count()
    )


class QuotaGatedClient(LLMClient):
    """包一个内层 LLMClient，每次 complete() 调用前查配额、调用后写用量明细。

    调用前就已经超额：直接抛异常，不调用内层 client（省下一次本来就要超额的
    真实调用成本）。
    调用成功：读 inner.last_usage 写一行 llm_usage_events，计入配额。
    调用失败（网络错误/服务商报错）：不写入、不计入配额——没拿到真实响应就没有
    真实产生的可审计成本。一次 LLM 调用是秒级原子操作，不需要"预留-提交/回滚"
    这种给"一次训练可能跑几分钟"设计的多步事务。
    """

    def __init__(self, inner: LLMClient, user_id, db_session_factory,
                task_id: Optional[str] = None, conversation_id: Optional[str] = None):
        self.inner = inner
        self.user_id = user_id
        self.db_session_factory = db_session_factory
        self.task_id = task_id
        self.conversation_id = conversation_id

    def _check_quota(self, db: Session) -> None:
        used = calls_used_this_period(db, self.user_id)
        if used >= settings.quota_free_monthly_calls:
            raise QuotaExceededError(
                f"本月免费额度已用完（{used}/{settings.quota_free_monthly_calls} 次），"
                f"可以填自己的 API Key 继续使用，或等下月配额重置。"
            )

    def _log_usage(self, db: Session, usage: Dict[str, int]) -> None:
        db.add(LLMUsageEvent(
            user_id=self.user_id,
            period=current_period(),
            provider="system_managed",
            model=getattr(self.inner, "model", "unknown"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            task_id=self.task_id,
            conversation_id=self.conversation_id,
        ))
        db.commit()

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        db = self.db_session_factory()
        try:
            self._check_quota(db)
            text = self.inner.complete(system, user, max_tokens)
            usage = self.inner.last_usage or {}
            self._log_usage(db, usage)
            self.last_usage = usage
            return text
        finally:
            db.close()

    async def complete_with_tools(
        self, messages: List[AgentMessage], tools: List[ToolDef],
        system: str = "", max_tokens: int = 1000,
    ) -> CompletionResult:
        """Multi-Agent 模式的工具调用同样要过配额闸——不能因为换了个方法名就绕过
        计费/审计，否则 system_managed 用户能免费/不受限地跑多轮工具调用循环。
        db 查询/写入本身还是同步 SQLAlchemy（配额检查是毫秒级的本地操作，没必要
        为了这一层引入 async db session），真正的 IO 等待——LLM 网络请求——
        在 await self.inner.complete_with_tools(...) 这一行是真异步的。"""
        db = self.db_session_factory()
        try:
            self._check_quota(db)
            result = await self.inner.complete_with_tools(messages, tools, system, max_tokens)
            usage = self.inner.last_usage or {}
            self._log_usage(db, usage)
            self.last_usage = usage
            return result
        finally:
            db.close()
