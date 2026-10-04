"""
api/routes/providers.py  -  探测本地/自建或商业 LLM 服务实际可用的模型列表。

前端不再让用户盲打模型名称——填好地址/Key 之后点一下，这里帮忙探测出真实
可用的模型，前端渲染成下拉框。四种探测路径：
  self_hosted         先试 Ollama 原生 /api/tags，不行再试标准 OpenAI 协议
                      的 /v1/models——探测前不知道自建服务具体是哪种协议，
                      探测本身就是在回答这个问题。
  commercial_anthropic 用 anthropic SDK 自带的 client.models.list()，用户自己的 Key。
  commercial_openai    标准 OpenAI 协议的 /v1/models（OpenAI 官方、Azure OpenAI
                      等商业 API 都实现了这个端点），用户自己的 Key。
  system_managed      系统托管场景——不用用户填的 Key（用户根本没有 Key），
                      用服务器自己配置的 ANTHROPIC_SYSTEM_API_KEY 探测，
                      要求登录（和 system_managed 本身的要求一致），避免匿名
                      用户拿这个白嫖一次系统 Key 的只读探测。

这里只做只读探测（GET 请求/SDK 只读调用），不涉及任何训练/计费逻辑。
"""

from __future__ import annotations

from typing import List, Optional

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.accounts.deps import get_current_user_optional
from api.accounts.models_db import User
from api.settings import settings

router = APIRouter(prefix="/api/llm-providers", tags=["LLM Providers"])

_PROBE_TIMEOUT = 8.0


class ProbeRequest(BaseModel):
    mode: str   # "self_hosted" | "commercial_anthropic" | "commercial_openai" | "system_managed"
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class ProbeResponse(BaseModel):
    ok: bool
    detected_protocol: Optional[str] = None   # "ollama" | "openai_compatible"（只有 self_hosted 会填）
    models: List[str] = []
    error: Optional[str] = None


def _try_ollama_tags(root: str) -> Optional[List[str]]:
    try:
        resp = httpx.get(f"{root}/api/tags", timeout=_PROBE_TIMEOUT)
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", []) if m.get("name")]
        return names or None
    except Exception:
        return None


def _try_openai_models(base_url: str, api_key: Optional[str]) -> Optional[List[str]]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    base = base_url.rstrip("/")
    # 用户可能填了带 /v1 的地址，也可能没带——两种都试一遍，不强行要求特定写法
    candidates = [base] if base.endswith("/v1") else [f"{base}/v1", base]
    for candidate in candidates:
        try:
            resp = httpx.get(f"{candidate}/models", headers=headers, timeout=_PROBE_TIMEOUT)
            resp.raise_for_status()
            ids = [m["id"] for m in resp.json().get("data", []) if m.get("id")]
            if ids:
                return ids
        except Exception:
            continue
    return None


@router.post("/probe", response_model=ProbeResponse)
async def probe(req: ProbeRequest, user: Optional[User] = Depends(get_current_user_optional)) -> ProbeResponse:
    if req.mode == "system_managed":
        if not user:
            return ProbeResponse(ok=False, error="需要先登录")
        if not settings.anthropic_system_api_key:
            return ProbeResponse(ok=False, error="系统托管的商业 API 尚未配置，暂时无法列出型号")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.anthropic_system_api_key)
            page = client.models.list(limit=100)
            models = [m.id for m in page.data]
            return ProbeResponse(ok=True, models=models)
        except Exception as e:
            return ProbeResponse(ok=False, error=f"获取型号列表失败：{e}")

    if req.mode == "self_hosted":
        if not req.base_url:
            return ProbeResponse(ok=False, error="请先填写服务地址")
        root = req.base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        models = _try_ollama_tags(root)
        if models:
            return ProbeResponse(ok=True, detected_protocol="ollama", models=sorted(models))
        models = _try_openai_models(req.base_url, req.api_key)
        if models:
            return ProbeResponse(ok=True, detected_protocol="openai_compatible", models=sorted(models))
        return ProbeResponse(ok=False, error="连接不上这个地址，或者它不是 Ollama/OpenAI 兼容协议的服务，检查地址是否正确")

    if req.mode == "commercial_anthropic":
        if not req.api_key:
            return ProbeResponse(ok=False, error="请先填写 API Key")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=req.api_key)
            page = client.models.list(limit=100)
            models = [m.id for m in page.data]
            return ProbeResponse(ok=True, models=models)
        except Exception as e:
            return ProbeResponse(ok=False, error=f"Key 校验失败：{e}")

    if req.mode == "commercial_openai":
        if not req.api_key:
            return ProbeResponse(ok=False, error="请先填写 API Key")
        base_url = req.base_url or "https://api.openai.com/v1"
        models = _try_openai_models(base_url, req.api_key)
        if models:
            return ProbeResponse(ok=True, detected_protocol="openai_compatible", models=sorted(models))
        return ProbeResponse(ok=False, error="Key 或服务地址校验失败，检查是否正确")

    return ProbeResponse(ok=False, error=f"未知探测模式：{req.mode}")
