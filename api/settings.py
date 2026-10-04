"""
api/settings.py  -  账号体系/配额代理相关的应用配置

不叫 config.py，因为仓库根目录的 config.py 已经是 ML 领域 dataclass
（TaskSpec/TrainingConfig 等），和这里的"应用运行时配置"是完全不同的概念，
混在一起会造成"改 config.py 到底改的是哪个"的混淆。

本机开发起一个 PostgreSQL 最简单的方式（不用装 native Postgres）：
    docker run -d --name vibeml_pg -e POSTGRES_USER=vibeml \\
        -e POSTGRES_PASSWORD=vibeml_dev_pw -e POSTGRES_DB=vibeml \\
        -p 55432:5432 postgres:16-alpine
对应的 DATABASE_URL：
    postgresql+psycopg://vibeml:vibeml_dev_pw@localhost:55432/vibeml
"""

from __future__ import annotations

from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 数据库
    database_url: str = "postgresql+psycopg://vibeml:vibeml_dev_pw@localhost:55432/vibeml"

    # JWT（access token 短效 + refresh token 存 refresh_tokens 表，见 api/accounts/models_db.py）
    jwt_secret: str = "dev-only-insecure-secret-change-me"   # 生产环境必须通过环境变量覆盖
    jwt_access_token_minutes: int = 15
    jwt_refresh_token_days: int = 30

    # 系统托管的商业 LLM Key（"system_managed" provider 用，见 core/llm_client.py）
    anthropic_system_api_key: Optional[str] = None
    system_managed_default_model: str = "claude-sonnet-4-6"

    # Google OAuth（首期只接这一家，见 api/accounts/oauth_google.py）
    google_oauth_client_id: Optional[str] = None
    google_oauth_client_secret: Optional[str] = None
    google_oauth_redirect_uri: str = "http://localhost:8000/api/auth/google/callback"

    # 配额（按每次 LLM 调用计数，不是按训练任务计数——见实施计划 Part B）
    quota_free_monthly_calls: int = 200

    # 本机版离线宽限期（Phase 6）
    offline_grace_days: int = 7
    # 是否是本机版桌面构建——由 desktop_packaging/desktop_app.py 在启动时设置
    # 环境变量打开，网络版部署不设这个，永远是 False（不受离线宽限期限制，
    # 因为网络版本来就要求实时联网访问自己的账号数据库）
    is_desktop_build: bool = False

    # CORS：不用 "*"，配合 allow_credentials=True 显式列出允许的 origin
    cors_allowed_origins: List[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]


settings = Settings()
