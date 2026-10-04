"""
config.py - 所有共享数据结构

设计原则：用 dataclass 把每个阶段的输入/输出显式化，
让每个模块的契约清晰可见，方便后续升级替换。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    MULTI_LABEL   = "multi_label"
    NER           = "ner"
    GENERATION    = "generation"
    REGRESSION    = "regression"
    RL            = "rl"             # 强化学习：LLM 生成环境 + stable-baselines3 训练
    LLM_FINETUNE  = "llm_finetune"    # 任意 HF causal LM + LoRA 指令微调
    IMAGE_CLASSIFICATION = "image_classification"   # CLIP 类视觉编码器 + 分类头
    VLM_GENERATIVE       = "vlm_generative"         # 有参考答案的看图说话/VQA（BLEU/ROUGE-L 打分）


class NextAction(str, Enum):
    CONTINUE_TRAINING  = "continue_training"
    COLLECT_MORE_DATA  = "collect_more_data"
    ADJUST_HYPERPARAMS = "adjust_hyperparams"
    STOP_SUCCESS       = "stop_success"
    STOP_PLATEAU       = "stop_plateau"


class ModelBackend(str, Enum):
    """训练后端：sklearn（默认，快）/ 预训练 backbone 微调 / LLM 生成自定义分类头 / 强化学习 /
    LLM 指令微调 / VLM 图像分类 / VLM 生成式。RL/LLM_FINETUNE/VLM_IMAGE_CLS/VLM_GENERATIVE
    都不是和前三者并列的"文本分类三选一"，而是分别由对应的 TaskType 唯一决定的执行
    路径——对话阶段 choose_model 遇到这几种任务都会跳过三选一卡片，直接写入对应值。"""
    SKLEARN        = "sklearn"
    PRETRAINED_NN  = "pretrained_nn"
    CUSTOM_NN      = "custom_nn"
    RL             = "rl"
    LLM_FINETUNE   = "llm_finetune"
    VLM_IMAGE_CLS  = "vlm_image_cls"     # CLIP 类视觉编码器（冻结）+ 可训练分类头
    VLM_GENERATIVE = "vlm_generative"    # AutoModelForImageTextToText 端到端微调


@dataclass
class TaskSpec:
    """Phase 1.1 输出：结构化任务定义（由自然语言解析而来）"""
    task_type:          TaskType
    domain:             str
    label_schema:       List[str]
    input_field:        str
    output_description: str
    evaluation_metric:  str           # f1 / accuracy / mae ...
    language:           str = "zh"
    constraints:        Dict[str, Any] = field(default_factory=dict)
    raw_description:    str = ""      # 保留原始用户描述，供后续模块参考


@dataclass
class TrainingConfig:
    """Phase 1.3 自动选择的训练配置"""
    model_name:      str
    use_lora:        bool
    lora_rank:       int
    lora_alpha:      int
    learning_rate:   float
    num_epochs:      int
    batch_size:      int
    max_length:      int
    use_cpu_fallback: bool
    model_key:       str = "logreg_light"   # 内部路由键
    hyperparam_overrides: Dict[str, Any] = field(default_factory=dict)  # LLM/规则给出的调参增量（C / alpha）
    backend:         ModelBackend = ModelBackend.SKLEARN


@dataclass
class GeneratedArchSpec:
    """core/arch_designer.py 输出：LLM 生成的自定义分类头"""
    class_name:     str            # 生成代码里的 nn.Module 类名
    source_code:    str            # 完整源码（需经 core/nn_sandbox.py 校验后才能执行）
    loss_fn_name:   str            # "cross_entropy" | "nll_loss" | ...
    rationale:      str            # 给用户看的设计理由（通俗语言）


@dataclass
class BackboneSpec:
    """core/backbone_selector.py 输出：选中的预训练模型 + 微调方式"""
    model_id:       str            # 白名单里的 HF 模型 ID
    use_lora:       bool
    rationale:      str


@dataclass
class GeneratedEnvSpec:
    """core/rl_env_designer.py 输出：LLM 生成的强化学习环境（需经 core/rl_sandbox.py 校验后才能执行）"""
    class_name:            str    # 生成代码里继承 gymnasium.Env 的类名
    source_code:           str    # 完整源码
    action_space_desc:     str    # 给用户看的动作空间说明（通俗语言）
    observation_space_desc: str   # 给用户看的观测空间说明（通俗语言）
    reward_rationale:      str    # 奖励函数设计理由（通俗语言）


@dataclass
class DataReport:
    """Phase 1.2 输出：数据质量诊断报告"""
    total_samples:      int
    label_distribution: Dict[str, int]
    augmented_count:    int
    boundary_samples:   List[Dict]    # 边界样本列表（可能存在标注争议）
    quality_score:      float         # 0-1，综合质量评分
    warnings:           List[str]


@dataclass
class EpochResult:
    """单轮训练结果（Phase 1.3 输出）"""
    epoch:              int
    train_loss:         float
    val_loss:           float
    val_metric:         float
    metric_name:        str
    per_class_metrics:  Dict[str, float]   # 每个标签的 F1
    confusion_highlights: List[str]         # 最容易混淆的标签对


@dataclass
class IterationExplanation:
    """Phase 1.4 输出：一轮迭代的可读解释"""
    diagnosis:      str           # 一句话状态描述
    root_cause:     str           # 根本原因（用比喻/通俗语言）
    recommendation: str           # 具体可操作建议
    next_action:    NextAction
    confidence:     float         # 0-1，解释的置信度
    hyperparam_delta: Dict[str, Any] = field(default_factory=dict)
    # next_action == adjust_hyperparams 时，LLM 给出的具体改动：
    #   {"C": 0.5} / {"alpha": 0.01} / {"switch_model": "svm"} / {"class_boost": {"标签": 1.3}}


@dataclass
class LoopState:
    """Phase 1.5 全局状态，贯穿整个闭环"""
    iteration:      int
    task_spec:      TaskSpec
    data_report:    DataReport
    epoch_results:  List[EpochResult]
    explanations:   List[IterationExplanation]
    best_metric:    float
    plateau_count:  int
    final_model:    Any = None    # 最终 Trainer 实例


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 数据结构
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AugmentReport:
    """Phase 2.1 输出：特征空间增强报告"""
    strategy:              str            # "smote" / "mixup" / "none"
    original_train_count:  int
    augmented_train_count: int
    per_class_before:      Dict[str, int]
    per_class_after:       Dict[str, int]
    balance_ratio_before:  float          # min/max 类别比
    balance_ratio_after:   float


@dataclass
class FlywheelReport:
    """Phase 2.1 输出：数据飞轮过滤报告"""
    total_checked:         int
    high_confidence:       int            # 保留样本数
    low_confidence:        int            # 过滤/标记样本数
    avg_confidence:        float
    threshold_used:        float
    flagged_samples:       List[Dict]     # 低置信度样本（供用户复查）


@dataclass
class IterationNode:
    """Phase 2.2 迭代树节点"""
    node_id:        str
    parent_id:      Optional[str]
    iteration:      int
    action:         str            # "initial" / "collect_data" / "adjust_hyperparams" / ...
    metric_before:  float
    metric_after:   float
    delta:          float
    explanation:    str            # 本轮改动的因果解释
    counterfactual: str            # "如果不做此改动，预期结果是…"


@dataclass
class DeploymentPackage:
    """Phase 2.3 输出：部署包描述"""
    export_format:       str      # "onnx" / "joblib"
    model_path:          str
    inference_script:    str
    package_dir:         str
    model_size_kb:       float
    labels:              List[str]
    preprocessing_info:  Dict[str, Any]
    usage_example:       str


# ─────────────────────────────────────────────────────────────────────────────
# 数据源接入（本地上传 / 本地路径 / HuggingFace Hub / 魔搭 ModelScope）
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetSummary:
    """数据集搜索结果里的一条摘要"""
    platform:     str            # "huggingface" / "modelscope"
    ref:          str            # 数据集 ID（如 "stanfordnlp/imdb"）
    description:  str = ""
    downloads:    int = 0
    likes:        int = 0
    tags:         List[str] = field(default_factory=list)


@dataclass
class RecommendedDataset:
    """core/dataset_recommender.py 输出：用户没指定数据集时，根据任务自动推荐的候选"""
    platform:  str            # "huggingface" / "modelscope"
    ref:       str
    rationale: str            # 给用户看的推荐理由（通俗语言）


@dataclass
class DatasetPreview:
    """拉取/预览数据集后的结果：抽样几条 + 自动列映射建议"""
    ref:              str
    total_available:  Optional[int]     # 数据集总行数（部分来源无法提前得知，为 None）
    sample_rows:      List[Dict[str, Any]]   # 原始行（未转换前），最多几十条
    columns:          List[str]
    suggested_text_col:  Optional[str]
    suggested_label_col: Optional[str]
    warnings:         List[str] = field(default_factory=list)
