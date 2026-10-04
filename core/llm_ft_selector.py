"""
core/llm_ft_selector.py  -  Phase 2：LLM 微调底座模型校验

不设白名单（用户已确认）——允许任意 HuggingFace causal LM ID，但必须先在
真正下载权重之前，用 HfApi().model_info(expand=["safetensors"]) 拿到参数量，
算清楚这个模型会不会拖垮本机，拒绝时给出具体理由（"这个模型大约需要多少 GB，
这台机器可用大约多少 GB"），不做无根据的猜测。

trust_remote_code=False 这道安全底线不在这里强制——这个模块只做元数据校验，
不加载模型；真正的加载点在 core/llm_ft_trainer.py，硬编码在那一处，独立于
这里的"不设白名单"决定之外，防止之后有人在某个调用点悄悄改掉。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import psutil
from huggingface_hub import HfApi, hf_hub_download

MAX_PARAMS_LLM_FT = 3_000_000_000
# 理由：LoRA 微调仍需整个冻结基座常驻内存，bf16 下约 2 字节/参数，3B 参数≈6GB
# 仅权重部分，加上 LoRA 参数的优化器状态、几百 token 长度的激活值、以及
# OS/Python/PyTorch 本身开销，3B 模型一次真实训练大概率需要 12-20GB——这个上限
# 比现有 pretrained_nn 白名单的 66M-134M 级 BERT 模型宽松得多，但仍是一个
# 不会拖垮机器的实际上限。


class ModelRejectedError(Exception):
    """模型体积超限 / 无法确认体积 / Hub 上找不到——统一异常，携带给用户看的具体理由。"""


@dataclass
class ValidatedModel:
    model_id: str
    param_count: int
    param_count_source: str   # "safetensors" | "config_estimate"


def _effective_cap() -> int:
    """机器可用内存越小，有效上限越严格——不是所有机器都能扛住 3B 模型，
    内存小的机器拿到更严格的上限而不是同一个静态数字。"""
    available_bytes = psutil.virtual_memory().available
    dynamic_cap = int(available_bytes * 0.3 / 2)
    return min(MAX_PARAMS_LLM_FT, dynamic_cap)


def _estimate_params_from_config(model_id: str) -> Optional[int]:
    """safetensors 元数据拿不到时，下载真实 config.json（不是 model_info() 返回的
    摘要 config，那个通常不含 hidden_size 等维度字段）粗略估算参数量。
    只覆盖标准 transformer 解码器最常见的字段命名，估不出来就返回 None，
    不做没有根据的猜测。"""
    try:
        path = hf_hub_download(model_id, "config.json")
        import json
        cfg = json.load(open(path))
    except Exception:
        return None

    hidden_size = cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model")
    num_layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers")
    vocab_size = cfg.get("vocab_size")
    if not (hidden_size and num_layers and vocab_size):
        return None
    intermediate_size = cfg.get("intermediate_size") or hidden_size * 4

    # 粗略估算：每层 attention(4h²) + mlp(2h·i)，加上输入/输出 embedding（不假设权重共享，
    # 偏保守往大了估，避免真实体积比估算值大导致预检通过但实际装不下）
    per_layer = 4 * hidden_size ** 2 + 2 * hidden_size * intermediate_size
    embed = vocab_size * hidden_size
    return int(num_layers * per_layer + embed * 2)


def validate_model_id(model_id: str) -> ValidatedModel:
    """在下载任何权重之前调用。抛 ModelRejectedError 时消息可以直接展示给用户。"""
    api = HfApi()
    try:
        info = api.model_info(model_id, expand=["safetensors"])
    except Exception as e:
        raise ModelRejectedError(f"在 HuggingFace Hub 上找不到模型「{model_id}」：{e}")

    param_count: Optional[int] = None
    source = ""
    if info.safetensors and info.safetensors.total:
        param_count = info.safetensors.total
        source = "safetensors"
    else:
        estimated = _estimate_params_from_config(model_id)
        if estimated:
            param_count = estimated
            source = "config_estimate"

    if param_count is None:
        raise ModelRejectedError(
            f"无法确认模型「{model_id}」的参数量（既没有 safetensors 元数据，也无法从"
            f"配置文件估算），换一个模型，或者先把这个模型转换成 safetensors 格式再试。"
        )

    cap = _effective_cap()
    if param_count > cap:
        available_gb = psutil.virtual_memory().available / 1e9
        raise ModelRejectedError(
            f"模型「{model_id}」大约有 {param_count / 1e9:.2f}B 参数，"
            f"这台机器现在可用内存大约 {available_gb:.1f}GB，装不下——换一个更小的模型试试。"
        )

    return ValidatedModel(model_id=model_id, param_count=param_count, param_count_source=source)
