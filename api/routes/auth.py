"""api/routes/auth.py  -  账号端点：注册/登录/刷新/登出/个人信息/改密码。"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from sqlalchemy.orm import Session

from api.accounts import service
from api.accounts.db import get_db
from api.accounts.deps import get_current_user
from api.accounts.models_db import LLMUsageEvent, User
from api.accounts.quota_gated_client import current_period
from api.accounts.schemas import (
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UpdateProfileRequest,
    UsageEventItem,
    UsageSummaryResponse,
    UserProfileResponse,
)
from api.settings import settings

router = APIRouter(prefix="/api/auth", tags=["Auth"])

_REFRESH_COOKIE_NAME = "refresh_token"
_REFRESH_COOKIE_PATH = "/api/auth"   # 只在需要刷新/登出的端点上携带，缩小暴露面

# 跟现有 core/data_sources.py::UPLOAD_DIR（./data_cache/uploads）同一套约定，
# 单独开一个子目录，不跟数据集上传混在一起
AVATAR_DIR = Path("./data_cache/avatars")
AVATAR_DIR.mkdir(parents=True, exist_ok=True)
_MAX_AVATAR_BYTES = 2 * 1024 * 1024   # 2MB，头像不需要很大
_ALLOWED_AVATAR_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def set_refresh_cookie(response: Response, raw_refresh_token: str) -> None:
    response.set_cookie(
        key=_REFRESH_COOKIE_NAME,
        value=raw_refresh_token,
        httponly=True,
        secure=False,   # 本地开发是 http；生产部署到 https 域名后要改成 True
        samesite="lax",
        max_age=settings.jwt_refresh_token_days * 24 * 3600,
        path=_REFRESH_COOKIE_PATH,
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(_REFRESH_COOKIE_NAME, path=_REFRESH_COOKIE_PATH)


def _to_profile_response(user: User) -> UserProfileResponse:
    return UserProfileResponse(
        id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        avatar_url=user.avatar_url,
        email_verified=user.email_verified,
        has_password=user.password_hash is not None,
        oauth_providers=[oi.provider for oi in user.oauth_identities],
    )


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(req: RegisterRequest, response: Response, db: Session = Depends(get_db)):
    try:
        user = service.register_user(db, req.email, req.password, req.display_name)
    except service.AccountError as e:
        raise HTTPException(409, str(e))
    access_token, raw_refresh = service.issue_token_pair(db, user)
    set_refresh_cookie(response, raw_refresh)
    return TokenResponse(access_token=access_token)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, response: Response, db: Session = Depends(get_db)):
    try:
        user = service.authenticate_user(db, req.email, req.password)
    except service.AccountError as e:
        raise HTTPException(401, str(e))
    access_token, raw_refresh = service.issue_token_pair(db, user)
    set_refresh_cookie(response, raw_refresh)
    if settings.is_desktop_build:
        # 本机版：每次真正联网登录成功都算一次"验证"，供离线宽限期计时用（见
        # api/accounts/offline_grace.py）——网络版没有这个概念，不写这个文件
        from api.accounts.offline_grace import record_verification
        record_verification()
    return TokenResponse(access_token=access_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: Request, response: Response, db: Session = Depends(get_db)):
    raw_refresh = request.cookies.get(_REFRESH_COOKIE_NAME)
    if not raw_refresh:
        raise HTTPException(401, "没有找到登录状态，请重新登录")
    try:
        _, access_token, new_raw_refresh = service.rotate_refresh_token(db, raw_refresh)
    except service.AccountError as e:
        _clear_refresh_cookie(response)
        raise HTTPException(401, str(e))
    set_refresh_cookie(response, new_raw_refresh)
    if settings.is_desktop_build:
        from api.accounts.offline_grace import record_verification
        record_verification()
    return TokenResponse(access_token=access_token)


@router.post("/logout", status_code=204)
async def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    raw_refresh = request.cookies.get(_REFRESH_COOKIE_NAME)
    if raw_refresh:
        service.revoke_refresh_token(db, raw_refresh)
    _clear_refresh_cookie(response)


@router.get("/me", response_model=UserProfileResponse)
async def me(user: User = Depends(get_current_user)):
    return _to_profile_response(user)


@router.patch("/me", response_model=UserProfileResponse)
async def update_me(req: UpdateProfileRequest, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    updated = service.update_profile(db, user, req.display_name)
    return _to_profile_response(updated)


@router.post("/me/avatar", response_model=UserProfileResponse)
async def upload_avatar(file: UploadFile = File(...), user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_AVATAR_SUFFIXES:
        raise HTTPException(400, f"不支持的图片类型：{suffix or '(无后缀)'}"
                                 f"（支持 {', '.join(_ALLOWED_AVATAR_SUFFIXES)}）")
    content = await file.read()
    if len(content) > _MAX_AVATAR_BYTES:
        raise HTTPException(413, f"图片超过 {_MAX_AVATAR_BYTES // (1024*1024)}MB 上限")
    filename = f"{uuid.uuid4().hex}{suffix}"
    (AVATAR_DIR / filename).write_bytes(content)
    user.avatar_url = f"/avatars/{filename}"
    db.add(user)
    db.commit()
    db.refresh(user)
    return _to_profile_response(user)


@router.post("/change-password", status_code=204)
async def change_password(req: ChangePasswordRequest, user: User = Depends(get_current_user),
                          db: Session = Depends(get_db)):
    try:
        service.change_password(db, user, req.current_password, req.new_password)
    except service.AccountError as e:
        raise HTTPException(400, str(e))


@router.get("/me/usage", response_model=UsageSummaryResponse)
async def my_usage(period: str | None = None, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    """system_managed 调用的按月用量清单——同一张 llm_usage_events 表既是配额计数依据，
    也直接就是这里返回的 token 消耗明细本体（供用户/财务审计）。"""
    target_period = period or current_period()
    rows = (
        db.query(LLMUsageEvent)
        .filter(LLMUsageEvent.user_id == user.id, LLMUsageEvent.period == target_period)
        .order_by(LLMUsageEvent.created_at.desc())
        .all()
    )
    return UsageSummaryResponse(
        period=target_period,
        calls_used=len(rows),
        calls_limit=settings.quota_free_monthly_calls,
        total_tokens=sum(r.total_tokens for r in rows),
        events=[
            UsageEventItem(
                id=str(r.id), model=r.model, prompt_tokens=r.prompt_tokens,
                completion_tokens=r.completion_tokens, total_tokens=r.total_tokens,
                cost_estimate_usd=r.cost_estimate_usd, task_id=r.task_id,
                conversation_id=r.conversation_id, created_at=r.created_at.isoformat(),
            )
            for r in rows
        ],
    )
