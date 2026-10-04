"""
api/routes/api_tokens.py  -  用户自己创建/撤销长期 API token 的端点。

每个 token 自带一份独立的 provider 配置（provider/model/key/base_url）——
不是挂在 User 上的共享默认配置，一个 token 可以配 Ollama、另一个配自带的
Anthropic Key，互不影响（见 api/accounts/models_db.py::ApiToken 的注释）。

创建/列表/撤销这几个操作本身逻辑简单，不像 api/routes/auth.py 那样拆出
api/accounts/service.py 的必要——直接在路由里做，跟 api/routes/providers.py
这类薄路由风格一致。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.accounts.db import get_db
from api.accounts.deps import get_current_user
from api.accounts.models_db import ApiToken, User
from api.accounts.schemas import ApiTokenCreatedResponse, ApiTokenItem, CreateApiTokenRequest
from api.accounts.security import generate_api_token_raw, hash_api_token

router = APIRouter(prefix="/api/auth/api-tokens", tags=["API Tokens"])

_TOKEN_PREFIX_DISPLAY_LEN = 12   # 列表页展示用，够辨认又不足以重建出完整 token


def _serialize(token: ApiToken) -> ApiTokenItem:
    return ApiTokenItem(
        id=str(token.id), name=token.name, token_prefix=token.token_prefix,
        llm_provider=token.llm_provider, llm_model=token.llm_model,
        llm_base_url=token.llm_base_url, created_at=token.created_at.isoformat(),
        last_used_at=token.last_used_at.isoformat() if token.last_used_at else None,
        revoked=token.revoked_at is not None,
    )


@router.post("", response_model=ApiTokenCreatedResponse, status_code=201)
def create_api_token(
    req: CreateApiTokenRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    raw_token = generate_api_token_raw()
    row = ApiToken(
        user_id=user.id, name=req.name,
        token_prefix=raw_token[:_TOKEN_PREFIX_DISPLAY_LEN],
        token_hash=hash_api_token(raw_token),
        llm_provider=req.llm_provider, llm_model=req.llm_model,
        llm_api_key=req.llm_api_key, llm_base_url=req.llm_base_url,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return ApiTokenCreatedResponse(id=str(row.id), raw_token=raw_token)


@router.get("", response_model=list[ApiTokenItem])
def list_api_tokens(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(ApiToken).filter_by(user_id=user.id).order_by(ApiToken.created_at.desc()).all()
    return [_serialize(r) for r in rows]


@router.delete("/{token_id}", status_code=204)
def revoke_api_token(
    token_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        tid = uuid.UUID(token_id)
    except ValueError:
        raise HTTPException(404, "token 不存在")
    row = db.query(ApiToken).filter_by(id=tid, user_id=user.id).one_or_none()
    if row is None:
        raise HTTPException(404, "token 不存在")
    if row.revoked_at is None:
        row.revoked_at = datetime.utcnow()
        db.commit()
