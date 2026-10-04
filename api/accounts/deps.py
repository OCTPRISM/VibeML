"""api/accounts/deps.py  -  FastAPI 鉴权依赖（仓库第一个 Depends()-based 鉴权守卫）。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, Tuple

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from api.accounts.db import get_db
from api.accounts.models_db import ApiToken, User
from api.accounts.security import decode_access_token, hash_api_token, is_api_token_format

_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "需要登录")
    user_id_str = decode_access_token(credentials.credentials)
    if user_id_str is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "登录状态无效或已过期")
    user = db.query(User).filter_by(id=uuid.UUID(user_id_str)).one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "账号不存在或已被禁用")
    return user


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """未登录返回 None 而不是 401——给"匿名也能用，登录了体验更好"的端点用
    （目前 tasks/conversations 是否要求强制登录还是产品决策，暂时保留这条可选路径）。"""
    if credentials is None:
        return None
    user_id_str = decode_access_token(credentials.credentials)
    if user_id_str is None:
        return None
    return db.query(User).filter_by(id=uuid.UUID(user_id_str)).one_or_none()


def get_current_api_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> ApiToken:
    """外部 API 调用方用长期 token 鉴权（区别于浏览器登录的短效 JWT）。"""
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "需要提供 API token")
    token_hash = hash_api_token(credentials.credentials)
    token = db.query(ApiToken).filter_by(token_hash=token_hash).one_or_none()
    if token is None or token.revoked_at is not None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API token 无效或已撤销")
    token.last_used_at = datetime.utcnow()
    db.commit()
    return token


def get_current_user_or_api_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> Tuple[Optional[User], Optional[ApiToken]]:
    """三种情况通吃：匿名（未带凭据）→ (None, None)；带的是 API token（vibe_sk_ 前缀）→
    (token.user, token)；否则当 JWT 解（浏览器登录场景）→ (user, None)。给
    POST /api/conversations 用——这样同一个端点、同一个 Authorization header，
    既服务浏览器前端也服务外部 API 调用方，不需要两套端点。"""
    if credentials is None:
        return None, None
    raw = credentials.credentials
    if is_api_token_format(raw):
        token_hash = hash_api_token(raw)
        token = db.query(ApiToken).filter_by(token_hash=token_hash).one_or_none()
        if token is None or token.revoked_at is not None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API token 无效或已撤销")
        token.last_used_at = datetime.utcnow()
        db.commit()
        return token.user, token
    user_id_str = decode_access_token(raw)
    if user_id_str is None:
        return None, None
    user = db.query(User).filter_by(id=uuid.UUID(user_id_str)).one_or_none()
    return user, None
