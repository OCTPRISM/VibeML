"""
api/accounts/oauth_google.py  -  Google OAuth（首期唯一接入的第三方登录）。

流程：
  GET  /api/auth/google/login     -> 302 到 Google 授权页（带 CSRF state，存在
                                      HttpOnly cookie 里，回调时比对）
  GET  /api/auth/google/callback  -> 用 code 换 token，验证 ID token，upsert 账号，
                                      302 回前端时带一个一次性 exchange code（不是
                                      直接把 JWT 放 URL 里，避免出现在浏览器历史/
                                      referrer 里）
  POST /api/auth/oauth/exchange   -> 前端用 exchange code 换真正的 token pair
  POST /api/auth/google/confirm-link -> 邮箱冲突时，要求输入已有账号密码确认关联

`provider` 字段用字符串而不是每家单开一张表——这是给以后加 GitHub/微信等
留的扩展点，本文件目前只实现 "google" 这一个。
"""

from __future__ import annotations

import secrets
import time
import uuid
from typing import Any, Dict, Tuple

from authlib.integrations.httpx_client import OAuth2Client
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.accounts.db import get_db
from api.accounts.models_db import OAuthIdentity, User
from api.accounts.schemas import TokenResponse
from api.accounts.security import verify_password
from api.accounts.service import issue_token_pair
from api.routes.auth import set_refresh_cookie
from api.settings import settings

router = APIRouter(prefix="/api/auth", tags=["Auth"])

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

_OAUTH_STATE_COOKIE = "oauth_state"
_EXCHANGE_CODE_TTL_SECONDS = 60

# 一次性 exchange code / 待确认关联 的临时存储——都是秒级 TTL 的短命数据，
# 用进程内存字典足够，不需要为这个专门起一张数据库表（照抄本仓库其它地方
# "轻量场景用内存态，不是所有状态都要落库" 的一贯做法）。
_pending_exchanges: Dict[str, Tuple[uuid.UUID, float]] = {}
_pending_links: Dict[str, Tuple[str, str, str, float]] = {}   # code -> (existing_user_id, google_sub, google_email, expires)


def _require_google_configured() -> None:
    if not settings.google_oauth_client_id or not settings.google_oauth_client_secret:
        raise HTTPException(503, "Google 登录尚未配置（缺少 GOOGLE_OAUTH_CLIENT_ID/SECRET），"
                                 "请联系管理员配置后再试，或使用邮箱密码登录。")


def _cleanup_expired(store: dict) -> None:
    now = time.time()
    expired = [k for k, v in store.items() if v[-1] < now]
    for k in expired:
        store.pop(k, None)


@router.get("/google/login")
async def google_login():
    """真正的 302 跳转到 Google 授权页——前端只需要 window.location.href 指到这个端点，
    不需要先 fetch 再自己跳转。"""
    _require_google_configured()
    state = secrets.token_urlsafe(24)
    client = OAuth2Client(settings.google_oauth_client_id, settings.google_oauth_client_secret,
                          redirect_uri=settings.google_oauth_redirect_uri,
                          scope="openid email profile")
    uri, _ = client.create_authorization_url(_GOOGLE_AUTH_URL, state=state)
    resp = RedirectResponse(uri, status_code=302)
    resp.set_cookie(_OAUTH_STATE_COOKIE, state, httponly=True, samesite="lax",
                    max_age=600, path="/api/auth")
    return resp


@router.get("/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_db)):
    """Google 回调也是浏览器整页导航（不是前端 fetch 能拦下来的 XHR），所以这里必须
    真正 302 回前端页面，用查询参数带着 exchange_code / link_code，让 index.html
    加载时自己读参数、调对应的 POST 接口完成登录——不能像 REST 接口那样直接返回 JSON，
    浏览器会把 JSON 当成页面内容原样显示，没法回到 SPA 里。"""
    _require_google_configured()
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    cookie_state = request.cookies.get(_OAUTH_STATE_COOKIE)
    if not code or not state or not cookie_state or state != cookie_state:
        raise HTTPException(400, "OAuth state 校验失败，请重新发起登录（可能是跨站请求伪造或链接已过期）")

    client = OAuth2Client(settings.google_oauth_client_id, settings.google_oauth_client_secret,
                          redirect_uri=settings.google_oauth_redirect_uri)
    token = client.fetch_token(_GOOGLE_TOKEN_URL, code=code)
    userinfo_resp = client.get(_GOOGLE_USERINFO_URL, token=token)
    userinfo_resp.raise_for_status()
    userinfo: Dict[str, Any] = userinfo_resp.json()

    google_sub = userinfo["sub"]
    google_email = userinfo["email"]
    google_email_verified = bool(userinfo.get("email_verified"))

    identity = db.query(OAuthIdentity).filter_by(provider="google", provider_user_id=google_sub).one_or_none()
    if identity is not None:
        # 已经关联过，直接登录
        user = db.query(User).filter_by(id=identity.user_id).one()
        resp = _redirect_with_exchange_code(user.id)
    else:
        existing_user = db.query(User).filter_by(email=google_email.lower()).one_or_none()
        if existing_user is not None:
            # 邮箱冲突：不做静默自动合并，要求用现有账号密码确认关联
            _cleanup_expired(_pending_links)
            link_code = secrets.token_urlsafe(24)
            _pending_links[link_code] = (str(existing_user.id), google_sub, google_email,
                                         time.time() + _EXCHANGE_CODE_TTL_SECONDS)
            resp = RedirectResponse(f"/?oauth_link={link_code}&email={google_email}", status_code=302)
        else:
            # 全新账号：Google 邮箱本身经过 Google 验证过，可信度高于我们自己未验证的注册邮箱
            new_user = User(email=google_email.lower(),
                            display_name=userinfo.get("name") or google_email.split("@")[0],
                            avatar_url=userinfo.get("picture"), email_verified=google_email_verified)
            db.add(new_user)
            db.flush()
            db.add(OAuthIdentity(user_id=new_user.id, provider="google",
                                provider_user_id=google_sub, provider_email=google_email))
            db.commit()
            resp = _redirect_with_exchange_code(new_user.id)

    resp.delete_cookie(_OAUTH_STATE_COOKIE, path="/api/auth")
    return resp


def _redirect_with_exchange_code(user_id: uuid.UUID) -> RedirectResponse:
    _cleanup_expired(_pending_exchanges)
    code = secrets.token_urlsafe(24)
    _pending_exchanges[code] = (user_id, time.time() + _EXCHANGE_CODE_TTL_SECONDS)
    return RedirectResponse(f"/?oauth_exchange={code}", status_code=302)


class ExchangeRequest(BaseModel):
    code: str


@router.post("/oauth/exchange", response_model=TokenResponse)
async def oauth_exchange(req: ExchangeRequest, response: Response, db: Session = Depends(get_db)):
    _cleanup_expired(_pending_exchanges)
    entry = _pending_exchanges.pop(req.code, None)
    if entry is None:
        raise HTTPException(400, "登录链接已失效，请重新登录")
    user_id, _ = entry
    user = db.query(User).filter_by(id=user_id).one_or_none()
    if user is None:
        raise HTTPException(400, "账号不存在")
    access_token, raw_refresh = issue_token_pair(db, user)
    set_refresh_cookie(response, raw_refresh)
    return TokenResponse(access_token=access_token)


class ConfirmLinkRequest(BaseModel):
    link_code: str
    password: str


@router.post("/google/confirm-link", response_model=TokenResponse)
async def confirm_link(req: ConfirmLinkRequest, response: Response, db: Session = Depends(get_db)):
    _cleanup_expired(_pending_links)
    entry = _pending_links.pop(req.link_code, None)
    if entry is None:
        raise HTTPException(400, "关联请求已失效，请重新发起 Google 登录")
    existing_user_id, google_sub, google_email, _ = entry
    user = db.query(User).filter_by(id=uuid.UUID(existing_user_id)).one_or_none()
    if user is None or not user.password_hash or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "密码不正确，无法关联账号")
    db.add(OAuthIdentity(user_id=user.id, provider="google",
                        provider_user_id=google_sub, provider_email=google_email))
    db.commit()
    access_token, raw_refresh = issue_token_pair(db, user)
    set_refresh_cookie(response, raw_refresh)
    return TokenResponse(access_token=access_token)
