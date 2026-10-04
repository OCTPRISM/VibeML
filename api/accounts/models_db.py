"""
api/accounts/models_db.py  -  账号/鉴权/配额的 SQLAlchemy ORM 模型。

命名故意叫 models_db.py 而不是 models.py——api/models.py 已经是一批 pydantic
请求/响应模型，两个同名不同义的 "models" 模块放在一起会互相遮蔽，容易出错。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.accounts.db import Base


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(unique=True, index=True)
    # 纯 OAuth 账号没有密码，可空
    password_hash: Mapped[Optional[str]] = mapped_column(default=None)
    display_name: Mapped[str] = mapped_column(default="")
    avatar_url: Mapped[Optional[str]] = mapped_column(default=None)
    is_active: Mapped[bool] = mapped_column(default=True)
    # 密码重置/邮箱验证依赖邮件发送，本阶段（Phase 2b）未接入，先留字段
    email_verified: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)

    oauth_identities: Mapped[list["OAuthIdentity"]] = relationship(back_populates="user")
    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(back_populates="user")
    usage_events: Mapped[list["LLMUsageEvent"]] = relationship(back_populates="user")
    api_tokens: Mapped[list["ApiToken"]] = relationship(back_populates="user")
    compute_profiles: Mapped[list["ComputeResourceProfile"]] = relationship(back_populates="user")


class OAuthIdentity(Base):
    """账号关联表：一个 user 可以有 0/1/多个第三方登录身份，
    provider 用字符串而不是每家单开一张表——这是给以后加 GitHub/微信等留的扩展点。"""
    __tablename__ = "oauth_identities"
    __table_args__ = (UniqueConstraint("provider", "provider_user_id", name="uq_oauth_provider_identity"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    provider: Mapped[str] = mapped_column()   # "google"（首期只有这一个）
    provider_user_id: Mapped[str] = mapped_column()
    provider_email: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="oauth_identities")


class RefreshToken(Base):
    """存哈希不存明文；改密码/检测到泄露时把该用户全部未撤销的行 revoked_at 置位。"""
    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column()
    revoked_at: Mapped[Optional[datetime]] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="refresh_tokens")


class LLMUsageEvent(Base):
    """一行 = 一次真实计费的 LLM 调用（system_managed 专用；BYOK 不写这张表）。
    既是配额计数的依据，也直接就是用户要的"token 消耗清单"本体，不需要单独的审计表。
    task_id/conversation_id 只是字符串引用，不建外键——避免账号库和内存态任务库耦合。"""
    __tablename__ = "llm_usage_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    period: Mapped[str] = mapped_column(index=True)   # "2026-07"，写入时算好，避免查询期处理时区边界
    provider: Mapped[str] = mapped_column(default="system_managed")
    model: Mapped[str] = mapped_column()
    prompt_tokens: Mapped[int] = mapped_column(default=0)
    completion_tokens: Mapped[int] = mapped_column(default=0)
    total_tokens: Mapped[int] = mapped_column(default=0)
    cost_estimate_usd: Mapped[Optional[float]] = mapped_column(default=None)
    task_id: Mapped[Optional[str]] = mapped_column(default=None)
    conversation_id: Mapped[Optional[str]] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, index=True)

    user: Mapped["User"] = relationship(back_populates="usage_events")


Index("ix_llm_usage_events_user_period", LLMUsageEvent.user_id, LLMUsageEvent.period)


class ApiToken(Base):
    """长期有效的 API 调用凭据（区别于浏览器登录用的短效 JWT），给外部脚本/系统用。
    跟 RefreshToken 一样只存哈希，token_prefix 单独明文存一小段（创建时那个原始
    token 的前 12 位），列表页用来让用户认出"哪个是哪个"、撤销前确认，不需要（也
    不能）反解出完整 token。

    llm_provider/model/api_key/base_url 这四个字段是这个 token 自己的——每个
    token 各带一份独立的 provider 配置（不是挂在 User 上的共享默认配置），
    这样一个 token 可以配 Ollama、另一个配自带的 Anthropic Key，互不影响。
    llm_api_key 是明文存储：这里必须能反解出明文才能真正拿去发起 LLM 调用，
    不能像密码那样单向哈希——这是真实的数据泄露风险，但和现有 ANTHROPIC_SYSTEM_API_KEY
    等配置一样，本项目目前没有字段级加密机制，这次不新引入不一致的存法。"""
    __tablename__ = "api_tokens"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column()
    token_prefix: Mapped[str] = mapped_column()
    token_hash: Mapped[str] = mapped_column(unique=True, index=True)
    llm_provider: Mapped[str] = mapped_column()
    llm_model: Mapped[Optional[str]] = mapped_column(default=None)
    llm_api_key: Mapped[Optional[str]] = mapped_column(default=None)
    llm_base_url: Mapped[Optional[str]] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(default=None)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(default=None)

    user: Mapped["User"] = relationship(back_populates="api_tokens")


class ComputeResourceProfile(Base):
    """用户配置的外部计算资源（Slurm / Kubernetes），用来把训练任务真正提交到
    集群上跑，而不是在 API 服务器这台机器的线程池里跑（见 api/worker.py::_run_job
    里按 compute_profile_id 分发的那个分支）。

    凭据（kubeconfig 内容 / SSH 私钥）跟 ApiToken.llm_api_key 一样是**明文存储**，
    理由也一样：必须能反解出原文才能真正拿去连集群，不能像密码那样单向哈希。
    但要清醒地认识到这两者的风险等级不同——泄露一个 LLM Key 损失的是额度，
    泄露 kubeconfig / SSH 私钥损失的是整个计算集群的访问权。本项目目前没有字段级
    加密机制，这次沿用同一套存法保持一致，生产部署强烈建议改接密钥管理服务。

    backend 用字符串而不是每种调度器单开一张表——两种后端的字段差异很大，
    但都稀疏可空，一张表 + 一个判别字段比两张表更容易查询/展示（这跟
    OAuthIdentity.provider 是同一个取舍）。"""
    __tablename__ = "compute_resource_profiles"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column()
    backend: Mapped[str] = mapped_column()   # "kubernetes" | "slurm"

    # ── Kubernetes 专用 ──
    k8s_kubeconfig: Mapped[Optional[str]] = mapped_column(default=None)   # kubeconfig 全文，明文
    k8s_namespace: Mapped[Optional[str]] = mapped_column(default=None)
    k8s_image: Mapped[Optional[str]] = mapped_column(default=None)        # 远程执行用的镜像
    k8s_cpu: Mapped[Optional[str]] = mapped_column(default=None)          # "2" / "500m"
    k8s_memory: Mapped[Optional[str]] = mapped_column(default=None)       # "4Gi"
    k8s_gpu: Mapped[Optional[int]] = mapped_column(default=None)          # nvidia.com/gpu 数量，None=不要 GPU

    # ── Slurm 专用 ──
    slurm_host: Mapped[Optional[str]] = mapped_column(default=None)       # 登录节点
    slurm_port: Mapped[Optional[int]] = mapped_column(default=None)
    slurm_username: Mapped[Optional[str]] = mapped_column(default=None)
    slurm_ssh_private_key: Mapped[Optional[str]] = mapped_column(default=None)   # PEM 全文，明文
    slurm_partition: Mapped[Optional[str]] = mapped_column(default=None)
    slurm_workdir: Mapped[Optional[str]] = mapped_column(default=None)    # 远端放 job spec / sbatch 脚本的目录
    slurm_time_limit: Mapped[Optional[str]] = mapped_column(default=None) # "02:00:00"

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(default=None)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(default=None)

    user: Mapped["User"] = relationship(back_populates="compute_profiles")
