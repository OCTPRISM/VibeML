from __future__ import annotations
from core.version import __version__
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator

class TaskStatus(str, Enum):
    PENDING = "pending"; RUNNING = "running"
    COMPLETED = "completed"; FAILED = "failed"; QUEUED = "queued"

class TrainingSample(BaseModel):
    text: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)

class CreateTaskRequest(BaseModel):
    description: str = Field(..., min_length=5)
    examples: Optional[List[TrainingSample]] = Field(None, min_length=5)
    dataset_ref: Optional[str] = None   # 二选一：手动样本 或 已拉取的数据集引用（见 /api/datasets/fetch）
    api_key: Optional[str] = None
    max_iterations: int = Field(3, ge=1, le=10)
    target_metric: float = Field(0.80, ge=0.0, le=1.0)
    enable_phase2: bool = True
    llm_provider: Literal["anthropic", "ollama", "openai_compatible", "system_managed"] = "anthropic"
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None   # 只对 ollama（可选覆盖默认地址）/ openai_compatible（必填）有意义——openai_compatible 不等于本地服务，同一协议也用于 OpenAI/Azure OpenAI 等商业 API，本地还是商业只取决于这里的地址
    model_backend: Literal["sklearn", "pretrained_nn", "custom_nn"] = "sklearn"

    @field_validator("examples")
    @classmethod
    def at_least_two_labels(cls, v):
        if v is not None and len(set(s.label for s in v)) < 2:
            raise ValueError("至少需要 2 个不同标签")
        return v

    @model_validator(mode="after")
    def examples_xor_dataset_ref(self):
        if self.examples is None and self.dataset_ref is None:
            raise ValueError("必须提供 examples 或 dataset_ref 其中之一")
        return self

class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)

class FeedbackRequest(BaseModel):
    production_metric: float = Field(..., ge=0.0, le=1.0)
    drift_threshold: float = Field(0.05, ge=0.0, le=1.0)

class EpochSummary(BaseModel):
    iteration: int; epoch: int; val_metric: float; train_loss: float

class TaskResult(BaseModel):
    best_metric: float; metric_name: str; labels: List[str]
    domain: str; n_samples: int; epoch_history: List[EpochSummary]
    deploy_path: Optional[str] = None
    feedback_baseline: Optional[Dict[str, Any]] = None

class TaskCreatedResponse(BaseModel):
    task_id: str; status: TaskStatus; created_at: datetime
    ws_url: str; queue_position: Optional[int] = None

class TaskStatusResponse(BaseModel):
    task_id: str; status: TaskStatus; created_at: datetime
    started_at: Optional[datetime] = None; ended_at: Optional[datetime] = None
    progress: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[TaskResult] = None; error: Optional[str] = None

class PredictResponse(BaseModel):
    predictions: List[str]
    confidences: Optional[List[Dict[str, float]]] = None

class QueueStatsResponse(BaseModel):
    queued: int; running: int; completed: int; failed: int; max_concurrent: int

class HealthResponse(BaseModel):
    status: str = "ok"; version: str = __version__; phase: str = "stable"

# ── 数据源接入 ─────────────────────────────────────────────────────────────────

DataPlatform = Literal["local_upload", "local_path", "huggingface", "modelscope"]

class DatasetSearchResult(BaseModel):
    platform: str; ref: str; description: str = ""
    downloads: int = 0; likes: int = 0; tags: List[str] = Field(default_factory=list)

class DatasetUploadResponse(BaseModel):
    ref: str; filename: str; size_bytes: int

class ImageUploadItem(BaseModel):
    ref: str; filename: str; size_bytes: int

class ImageUploadResponse(BaseModel):
    # ref 是服务器本地路径（core/data_sources.py::UPLOAD_DIR 下），IMAGE_CLASSIFICATION/
    # VLM_GENERATIVE 任务把它塞进 image_path 字段——跟文本样本"引用而非内联"是同一个思路，
    # 图片本身不会经过 JSON 消息流
    images: List[ImageUploadItem]

class AttachmentUploadResponse(BaseModel):
    """聊天输入框"添加文件"上传的单个附件——跟 ImageUploadItem 是两回事：那个是
    训练样本图片（走 image_path，不回传浏览器），这个是给对话本身补充上下文用的
    普通附件（文档已经在这里提取好文本，图片会给一个可预览的 URL）。"""
    attachment_id: str
    filename: str
    kind: Literal["document", "image"]
    size_bytes: int
    extracted_text: Optional[str] = None    # 仅文档；提取失败时为 None
    extraction_error: Optional[str] = None   # 仅文档提取失败时有值，附件本身依然保留
    url: Optional[str] = None                # 仅图片；可直接用于 <img src>（.eps 除外，浏览器原生不认）

class SkippedArchiveEntry(BaseModel):
    archive: str      # 来自哪个压缩包（多个压缩包一起上传时用来定位）
    filename: str     # 压缩包内的成员名；整个压缩包都打不开时是 "(整个文件)"
    reason: str

class ArchiveUploadResponse(BaseModel):
    """一次可以传多个压缩包（zip/7z/tar系列/单文件gz-bz2/rar），每个压缩包解压出的
    每个成员都按 AttachmentUploadResponse 的方式处理（文档提取文本/图片给预览 URL）——
    跟直接上传单个文件是同一套处理逻辑，只是文件的来源变成了"从压缩包里解压"。"""
    items: List[AttachmentUploadResponse]
    skipped: List[SkippedArchiveEntry]   # 因为不安全/超限/格式不支持被跳过的成员

class DatasetPreviewRequest(BaseModel):
    platform: DataPlatform
    ref: str
    split: Optional[str] = "train"
    config: Optional[str] = None

class DatasetPreviewResponseModel(BaseModel):
    ref: str
    total_available: Optional[int] = None
    sample_rows: List[Dict[str, Any]]
    columns: List[str]
    suggested_text_col: Optional[str] = None
    suggested_label_col: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

class DatasetFetchRequest(BaseModel):
    platform: DataPlatform
    ref: str
    text_col: str
    label_col: str
    max_samples: int = Field(500, ge=5, le=5000)
    split: Optional[str] = "train"
    config: Optional[str] = None

class DatasetFetchResponse(BaseModel):
    dataset_ref: str; n_examples: int; labels: List[str]

# ── 对话式交互 ─────────────────────────────────────────────────────────────────

class CreateConversationRequest(BaseModel):
    llm_provider: Literal["anthropic", "ollama", "openai_compatible", "system_managed"] = "ollama"
    llm_model: Optional[str] = None
    api_key: Optional[str] = None
    llm_base_url: Optional[str] = None   # 只对 ollama（可选覆盖默认地址）/ openai_compatible（必填）有意义——openai_compatible 不等于本地服务，同一协议也用于 OpenAI/Azure OpenAI 等商业 API，本地还是商业只取决于这里的地址
    # "workflow" 是现有的固定 STAGE_ORDER 状态机（前端叫"传统方式"）；"multi_agent"
    # 走 core/agent/agent_orchestrator.py。mcp_server_urls/enabled_skills 这一期
    # 先收下存进 ConversationState，Phase 3/4 才真正生效
    orchestration_mode: Literal["workflow", "multi_agent"] = "workflow"
    mcp_server_urls: Optional[List[str]] = None
    enabled_skills: Optional[List[str]] = None

class ConversationCreatedResponse(BaseModel):
    conversation_id: str; ws_url: str

class PostMessageRequest(BaseModel):
    kind: Literal["text", "structured"] = "text"
    text: Optional[str] = None
    structured: Optional[Dict[str, Any]] = None
    # 聊天输入框"添加文件"上传的附件——每项 {filename, kind, extracted_text?, url?}，
    # 由 POST /api/attachments 上传时算好（文档已经提取出文本，图片没有 extracted_text）。
    # 只是原样透传给 ConversationOrchestrator.handle_message，这里不做任何加工。
    attachments: Optional[List[Dict[str, Any]]] = None

class ConversationMessageModel(BaseModel):
    role: str; type: str; payload: Any
    ui_hint: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    created_at: str
