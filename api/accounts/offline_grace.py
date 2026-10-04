"""
api/accounts/offline_grace.py  -  本机版离线宽限期（Phase 6）。

诚实的设计边界（这不是防篡改机制，只是提高随手作弊的成本）：本地缓存用打包
进二进制里的固定密钥做 HMAC 签名，挡得住"直接编辑本地文件改时间戳"这种最
省事的绕过方式，但挡不住有动机的用户逆向已打包的二进制、附调试器、或者同时
篡改系统时钟让联网校验也失败——这些更极端的绕过没有完整解决方案（本质上是
"给用户完全掌控的设备做 DRM"这个无解问题的一个实例）。真正权威的用量/登录
记录永远是账号服务器自己的数据库；这里的本地缓存只用来回答"离线的时候还能
不能继续用本机版"，不用作任何最终结算依据。

只在 settings.is_desktop_build 为 True 时才会被调用——网络版部署没有这个概念
（网络版本来就要求实时联网访问自己的账号数据库，不存在"离线"这回事）。
"""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from api.settings import settings

# 固定嵌入二进制里的密钥，纯混淆用途，不是真正的密码学秘密——见上面模块docstring
# 的诚实说明。用一个和 JWT_SECRET 完全不同的常量，避免服务器密钥意外泄露进
# 打包的桌面二进制里。
_LOCAL_CACHE_HMAC_KEY = b"vibe-ml-studio-desktop-offline-cache-v1"

_CACHE_DIR = Path.home() / ".vibe_ml_studio"
_CACHE_PATH = _CACHE_DIR / "offline_cache.sqlite3"


class RestrictedModeError(Exception):
    """离线超过宽限期——本机版拒绝新的训练提交，但已训练好的模型预测不受影响。"""


def _get_conn() -> sqlite3.Connection:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_CACHE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS verification (id INTEGER PRIMARY KEY CHECK (id = 1), "
        "verified_at_iso TEXT NOT NULL, signature TEXT NOT NULL)"
    )
    return conn


def _sign(verified_at_iso: str) -> str:
    return hmac.new(_LOCAL_CACHE_HMAC_KEY, verified_at_iso.encode(), hashlib.sha256).hexdigest()


def record_verification(now: Optional[datetime] = None) -> None:
    """每次成功的远程登录/刷新/配额校验之后调用——把"验证成功"这个事实签名后存到本地。"""
    now = now or datetime.now(timezone.utc)
    verified_at_iso = now.isoformat()
    signature = _sign(verified_at_iso)
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO verification (id, verified_at_iso, signature) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET verified_at_iso = excluded.verified_at_iso, "
            "signature = excluded.signature",
            (verified_at_iso, signature),
        )
        conn.commit()
    finally:
        conn.close()


def time_since_last_verification() -> Optional[timedelta]:
    """返回距上次验证过去了多久；从没验证过，或者本地记录被篡改（签名对不上）都返回 None
    ——按"未验证"处理（宁可错杀，不让被改过的本地文件冒充"刚验证过"）。"""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT verified_at_iso, signature FROM verification WHERE id = 1").fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    verified_at_iso, signature = row
    if not hmac.compare_digest(signature, _sign(verified_at_iso)):
        return None
    try:
        verified_at = datetime.fromisoformat(verified_at_iso)
    except ValueError:
        return None
    return datetime.now(timezone.utc) - verified_at


def is_within_grace_period() -> bool:
    elapsed = time_since_last_verification()
    if elapsed is None:
        return False
    return elapsed <= timedelta(days=settings.offline_grace_days)


def enforce_grace_period() -> None:
    """挂在训练提交路径上的守卫——只在 is_desktop_build 时生效，网络版直接跳过。"""
    if not settings.is_desktop_build:
        return
    if not is_within_grace_period():
        raise RestrictedModeError(
            f"本机版已经超过 {settings.offline_grace_days} 天没有联网校验，"
            f"暂时进入受限模式：不能开始新的训练，但已有模型的预测仍可正常使用。"
            f"联网后会自动恢复。"
        )
