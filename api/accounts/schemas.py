"""api/accounts/schemas.py  -  账号域的 pydantic 请求/响应模型。

风格照抄 api/models.py，但物理上分开（账号域 vs ML 域），避免两个领域的
数据形状互相污染。
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    display_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)


class UpdateProfileRequest(BaseModel):
    display_name: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserProfileResponse(BaseModel):
    id: str
    email: str
    display_name: str
    avatar_url: Optional[str] = None
    email_verified: bool
    has_password: bool          # 前端据此决定要不要显示"修改密码"（纯 OAuth 账号可能没设密码）
    oauth_providers: List[str]  # 已关联的第三方登录，比如 ["google"]


class UsageEventItem(BaseModel):
    id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_estimate_usd: Optional[float] = None
    task_id: Optional[str] = None
    conversation_id: Optional[str] = None
    created_at: str


class UsageSummaryResponse(BaseModel):
    period: str
    calls_used: int
    calls_limit: int
    total_tokens: int
    events: List[UsageEventItem]


class CreateApiTokenRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    llm_provider: str   # "anthropic" / "ollama" / "openai_compatible" / "system_managed"
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None


class ApiTokenCreatedResponse(BaseModel):
    id: str
    raw_token: str   # 只在创建这一刻返回一次，之后无法再次查看


class ApiTokenItem(BaseModel):
    id: str
    name: str
    token_prefix: str
    llm_provider: str
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    created_at: str
    last_used_at: Optional[str] = None
    revoked: bool


class ConversationSummaryItem(BaseModel):
    conversation_id: str
    origin: str            # "web" | "api"
    stage: str
    task_type: Optional[str] = None
    created_at: str


class CreateComputeProfileRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    backend: Literal["kubernetes", "slurm"]
    # Kubernetes
    k8s_kubeconfig: Optional[str] = None
    k8s_namespace: Optional[str] = None
    k8s_image: Optional[str] = None
    k8s_cpu: Optional[str] = None
    k8s_memory: Optional[str] = None
    k8s_gpu: Optional[int] = None
    # Slurm
    slurm_host: Optional[str] = None
    slurm_port: Optional[int] = None
    slurm_username: Optional[str] = None
    slurm_ssh_private_key: Optional[str] = None
    slurm_partition: Optional[str] = None
    slurm_workdir: Optional[str] = None
    slurm_time_limit: Optional[str] = None


class ComputeProfileItem(BaseModel):
    """列表用——kubeconfig / SSH 私钥这两个真正的凭据字段绝不回传，
    只回传一个"配没配"的布尔值，跟 ApiTokenItem 不回传 llm_api_key 是同一个原则。"""
    id: str
    name: str
    backend: str
    k8s_namespace: Optional[str] = None
    k8s_image: Optional[str] = None
    k8s_cpu: Optional[str] = None
    k8s_memory: Optional[str] = None
    k8s_gpu: Optional[int] = None
    has_kubeconfig: bool = False
    slurm_host: Optional[str] = None
    slurm_port: Optional[int] = None
    slurm_username: Optional[str] = None
    slurm_partition: Optional[str] = None
    slurm_workdir: Optional[str] = None
    slurm_time_limit: Optional[str] = None
    has_ssh_key: bool = False
    created_at: str
    last_used_at: Optional[str] = None
    revoked: bool


class ComputeProfileTestResult(BaseModel):
    ok: bool
    message: str                       # 成功/失败都给一句人能看懂的话
    detail: Optional[str] = None       # 识别到的集群版本/用户名之类的佐证信息
