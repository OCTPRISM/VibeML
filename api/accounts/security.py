"""
api/accounts/security.py  -  密码哈希 + JWT 编解码。

密码用 argon2（OWASP 现推荐，没有 bcrypt 72 字节截断的坑）；
JWT 用 HS256 对称签名（本机版桌面应用始终联网向远程服务器校验，不需要
客户端本地验证 token，HS256 比 RS256 更简单，够用）。
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from api.settings import settings

_hasher = PasswordHasher()


def hash_password(raw_password: str) -> str:
    return _hasher.hash(raw_password)


def verify_password(raw_password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, raw_password)
    except VerifyMismatchError:
        return False
    except Exception:
        # argon2 对畸形哈希（比如老数据/损坏数据）会抛别的异常类型——按"验证失败"处理，
        # 不让一条脏数据变成 500
        return False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(user_id: uuid.UUID) -> str:
    payload: Dict[str, Any] = {
        "sub": str(user_id),
        "type": "access",
        "iat": _now(),
        "exp": _now() + timedelta(minutes=settings.jwt_access_token_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> Optional[str]:
    """返回 user_id 字符串；token 无效/过期/类型不对都返回 None（不抛异常给调用方判断分支）。"""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    if payload.get("type") != "access":
        return None
    return payload.get("sub")


def generate_refresh_token_raw() -> str:
    """生成一个随机不透明字符串（不是 JWT）——数据库只存它的哈希，泄露数据库不等于泄露可用 token。"""
    return secrets.token_urlsafe(48)


def hash_refresh_token(raw_token: str) -> str:
    # 这里不需要 argon2 的慢哈希特性（refresh token 本身已经是高熵随机值，不是用户
    # 可能重复使用的弱密码）——用一个快速、确定性的哈希即可，argon2 每次哈希结果不同
    # (自带随机 salt) 反而没法用来做等值查询。
    import hashlib
    return hashlib.sha256(raw_token.encode()).hexdigest()


_API_TOKEN_PREFIX = "vibe_sk_"


def generate_api_token_raw() -> str:
    """生成长期有效的 API token（跟 refresh token 同一个"随机不透明字符串+只存哈希"
    的模式，见上面 generate_refresh_token_raw）。前缀是刻意加的——鉴权依赖靠这个
    前缀就能一眼区分"这是 API token 不是 JWT"，不用先尝试 JWT 解码失败了再回退。"""
    return _API_TOKEN_PREFIX + secrets.token_urlsafe(32)


def is_api_token_format(raw: str) -> bool:
    return raw.startswith(_API_TOKEN_PREFIX)


def hash_api_token(raw_token: str) -> str:
    # 跟 hash_refresh_token 同样的理由：高熵 token 不需要 argon2 的慢哈希，
    # 需要的是能做等值查询的确定性哈希。
    import hashlib
    return hashlib.sha256(raw_token.encode()).hexdigest()
