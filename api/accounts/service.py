"""
api/accounts/service.py  -  账号业务逻辑（注册/登录/token 发放与轮换/改密码）。

让 api/routes/auth.py 保持薄——照抄 api/worker.py 让 api/routes/tasks.py
保持薄的既有分层习惯。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from api.accounts.models_db import RefreshToken, User
from api.accounts.security import (
    create_access_token,
    generate_refresh_token_raw,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from api.settings import settings


class AccountError(Exception):
    """账号域的业务错误（邮箱已注册/密码不对等），路由层统一映射成 4xx。"""


def register_user(db: Session, email: str, password: str, display_name: Optional[str]) -> User:
    existing = db.query(User).filter_by(email=email.lower()).one_or_none()
    if existing is not None:
        raise AccountError("该邮箱已注册")
    user = User(
        email=email.lower(),
        password_hash=hash_password(password),
        display_name=display_name or email.split("@")[0],
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User:
    user = db.query(User).filter_by(email=email.lower()).one_or_none()
    if user is None or not user.password_hash or not verify_password(password, user.password_hash):
        raise AccountError("邮箱或密码不正确")
    if not user.is_active:
        raise AccountError("账号已被禁用")
    return user


def issue_token_pair(db: Session, user: User) -> Tuple[str, str]:
    """返回 (access_token, raw_refresh_token)——raw_refresh_token 只在这一刻明文存在，
    数据库里只存它的哈希（RefreshToken.token_hash）。"""
    access_token = create_access_token(user.id)
    raw_refresh = generate_refresh_token_raw()
    row = RefreshToken(
        user_id=user.id,
        token_hash=hash_refresh_token(raw_refresh),
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_token_days),
    )
    db.add(row)
    db.commit()
    return access_token, raw_refresh


def rotate_refresh_token(db: Session, raw_refresh_token: str) -> Tuple[User, str, str]:
    """用一次撤销一次，换发新的一对（access + refresh）。"""
    token_hash = hash_refresh_token(raw_refresh_token)
    row = db.query(RefreshToken).filter_by(token_hash=token_hash).one_or_none()
    now = datetime.now(timezone.utc)
    if row is None or row.revoked_at is not None or row.expires_at.replace(tzinfo=timezone.utc) < now:
        raise AccountError("登录状态已失效，请重新登录")
    row.revoked_at = now
    db.add(row)
    user = db.query(User).filter_by(id=row.user_id).one()
    access_token, new_raw_refresh = issue_token_pair(db, user)
    return user, access_token, new_raw_refresh


def revoke_refresh_token(db: Session, raw_refresh_token: str) -> None:
    token_hash = hash_refresh_token(raw_refresh_token)
    row = db.query(RefreshToken).filter_by(token_hash=token_hash).one_or_none()
    if row is not None and row.revoked_at is None:
        row.revoked_at = datetime.now(timezone.utc)
        db.add(row)
        db.commit()


def revoke_all_refresh_tokens(db: Session, user_id: uuid.UUID) -> None:
    now = datetime.now(timezone.utc)
    db.query(RefreshToken).filter(
        RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None)
    ).update({"revoked_at": now})
    db.commit()


def change_password(db: Session, user: User, current_password: str, new_password: str) -> None:
    if not user.password_hash or not verify_password(current_password, user.password_hash):
        raise AccountError("当前密码不正确")
    user.password_hash = hash_password(new_password)
    db.add(user)
    db.commit()
    # 改密码后旧的 refresh token 全部失效，防止旧凭据（比如泄露的浏览器会话）继续可用
    revoke_all_refresh_tokens(db, user.id)


def update_profile(db: Session, user: User, display_name: Optional[str]) -> User:
    if display_name is not None:
        user.display_name = display_name
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
