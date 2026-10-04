"""
core/llm_ft_trainer.py  -  Phase 2：任意 HF causal LM 的 LoRA 指令微调执行

复用 core/subprocess_runner.py 的隔离机制（和 core/nn_trainer.py 一样：
独立子进程 + 墙钟超时 + terminate/kill 兜底），只训练 LoRA adapter（不做全量
微调），子进程只把 adapter 的 state_dict（get_peft_model_state_dict，只有
LoRA 部分，不是整个冻结基座）送回父进程，训练完子进程退出，占用的内存/MPS
上下文一起释放。

trust_remote_code=False 硬编码在这一处（子进程内加载模型的唯一入口）——独立于
"不设白名单"这个决定之外的硬性安全底线，不管 core/llm_ft_selector.py 校验过
的模型 ID 是什么，这里都不打开这个开关；如果某个模型的加载器要求它，捕获
对应异常，返回清晰理由而不是自动打开。

Alpaca 风格的 instruction/input/output 三元组拼接成 prompt + completion，
对 prompt 部分的 token 做 loss mask（label=-100），只让completion部分参与
损失计算——标准 SFT 做法，不是本项目独创。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from config import EpochResult
from core.subprocess_runner import SubprocessTrainingError, run_in_subprocess

SFT_TIMEOUT_SECONDS = 2400     # 40 分钟——序列比分类任务长得多（512 token vs 128），
                               # 3B 级模型在 MPS 上无量化前向/反向比 66-134M 级 backbone 慢很多，
                               # 这是起始估计值，需要真实计时验证是否够用
MAX_TRAIN_STEPS = 500          # 硬步数上限，避免大数据集在一个 epoch 内悄悄耗光整个超时预算
DEFAULT_LORA_RANK = 8
DEFAULT_MAX_LENGTH = 512


class LLMFTTrainingError(SubprocessTrainingError):
    """子进程 SFT 训练失败（超时 / OOM / trust_remote_code 被拒 / 运行时异常）"""


class LLMFTTrainer:
    """接口与 core/trainer.py::Trainer 对齐（predict），但没有 predict_proba——
    生成式输出没有"类别概率"这个概念，api/routes/tasks.py::predict 对
    predict_proba 的调用本来就包了 try/except，AttributeError 会被吞掉、
    confidences 恒为 None，不需要在这里硬造一个假的。"""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._adapter_state: Optional[Dict[str, Any]] = None
        self._fitted = False

    def train_with_eval(self, instruction_examples: List[Dict], num_epochs: int = 3,
                        batch_size: int = 4, learning_rate: float = 2e-4,
                        lora_rank: int = DEFAULT_LORA_RANK,
                        max_length: int = DEFAULT_MAX_LENGTH) -> List[EpochResult]:
        train_ex, val_ex = self._split(instruction_examples)
        payload = {
            "model_id": self.model_id,
            "train_examples": train_ex,
            "val_examples": val_ex,
            "num_epochs": min(num_epochs, 10),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "max_length": max_length,
            "max_train_steps": MAX_TRAIN_STEPS,
        }
        result = run_in_subprocess(payload, _train_llm_ft_in_child, timeout=SFT_TIMEOUT_SECONDS,
                                   error_cls=LLMFTTrainingError)
        self._adapter_state = result["adapter_state_dict"]
        self._fitted = True
        return [EpochResult(**e) for e in result["epochs"]]

    def _reconstruct_peft_model(self):
        """把子进程送回的 adapter numpy state_dict 重建成真正的 PeftModel——
        predict() 和 save_adapter() 都需要这一步，抽成一个方法避免写两遍。"""
        if not self._fitted:
            raise RuntimeError("模型尚未训练")
        import torch
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from peft import TaskType as PeftTaskType
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=False)
        lora_cfg = LoraConfig(task_type=PeftTaskType.CAUSAL_LM, r=DEFAULT_LORA_RANK,
                              lora_alpha=DEFAULT_LORA_RANK * 2, lora_dropout=0.1)
        model = get_peft_model(base_model, lora_cfg)
        adapter_state = {k: torch.from_numpy(v) for k, v in self._adapter_state.items()}
        set_peft_model_state_dict(model, adapter_state)
        return model

    def predict(self, prompts: List[str]) -> List[str]:
        import torch
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = self._reconstruct_peft_model()
        model.eval()

        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model.to(device)

        outputs = []
        for prompt_text in prompts:
            full_prompt = _format_prompt(prompt_text, "")
            enc = tokenizer(full_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen_ids = model.generate(**enc, max_new_tokens=200, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id)
            completion = tokenizer.decode(gen_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            outputs.append(completion.strip())
        return outputs

    def save_adapter(self, output_dir: str) -> None:
        """存成标准 peft adapter 目录（adapter_config.json + adapter_model.safetensors），
        任何人 pip install peft transformers 之后都能用 PeftModel.from_pretrained(base, dir)
        原样加载，不依赖本项目自己的代码。"""
        model = self._reconstruct_peft_model()
        model.save_pretrained(output_dir)

    @staticmethod
    def _split(examples: List[Dict]):
        if len(examples) < 5:
            return examples, examples   # 样本太少时验证集直接复用训练集，不强行切分出空集
        split_idx = max(1, int(len(examples) * 0.8))
        return examples[:split_idx], examples[split_idx:]


def _format_prompt(instruction: str, input_text: str) -> str:
    """Alpaca 风格模板——instruction 必填，input 可选（很多公开指令数据集的 input 是空串）。"""
    if input_text and input_text.strip():
        return (f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n")
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def _train_llm_ft_in_child(payload: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from peft import TaskType as PeftTaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = payload["model_id"]
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        # trust_remote_code=False 硬编码——不管 model_id 是什么，这里永远不打开这个开关。
        # 如果某个模型的加载器要求它，下面 except 分支给出清晰理由，而不是自动允许执行
        # 仓库自带代码。
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=False)
    except Exception as e:
        msg = str(e)
        if "trust_remote_code" in msg or "custom code" in msg.lower():
            return {"error": f"模型「{model_id}」需要执行仓库自带代码才能加载，这里不允许，换一个模型。"}
        return {"error": f"加载模型「{model_id}」失败：{msg}"}

    lora_cfg = LoraConfig(task_type=PeftTaskType.CAUSAL_LM, r=payload["lora_rank"],
                          lora_alpha=payload["lora_rank"] * 2, lora_dropout=0.1)
    model = get_peft_model(model, lora_cfg)

    max_length = payload["max_length"]

    def _encode_example(ex: Dict) -> Dict[str, List[int]]:
        prompt = _format_prompt(ex["instruction"], ex.get("input", ""))
        completion = ex["output"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + completion_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]
        return {"input_ids": input_ids, "labels": labels}

    train_encoded = [_encode_example(e) for e in payload["train_examples"]]
    val_encoded = [_encode_example(e) for e in payload["val_examples"]]

    def _collate(batch: List[Dict[str, List[int]]]):
        max_len = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, labels, attn_mask = [], [], []
        for b in batch:
            pad_n = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad_n)
            labels.append(b["labels"] + [-100] * pad_n)
            attn_mask.append([1] * len(b["input_ids"]) + [0] * pad_n)
        return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn_mask))

    batch_size = payload["batch_size"]
    num_epochs = payload["num_epochs"]
    max_train_steps = payload["max_train_steps"]
    epochs_out: List[Dict] = []

    for attempt in range(2):    # OOM 时降 batch size 重试一次，和 core/nn_trainer.py 一致
        try:
            model.to(device)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=payload["learning_rate"])
            total_steps = 0
            epochs_out = []
            for epoch in range(num_epochs):
                model.train()
                total_loss, n_batches = 0.0, 0
                for start in range(0, len(train_encoded), batch_size):
                    if total_steps >= max_train_steps:
                        break
                    batch = train_encoded[start:start + batch_size]
                    input_ids, labels, attn_mask = _collate(batch)
                    input_ids, labels, attn_mask = input_ids.to(device), labels.to(device), attn_mask.to(device)
                    optimizer.zero_grad()
                    out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                    out.loss.backward()
                    optimizer.step()
                    total_loss += out.loss.item()
                    n_batches += 1
                    total_steps += 1
                train_loss = total_loss / max(n_batches, 1)

                model.eval()
                val_loss_total, val_batches = 0.0, 0
                with torch.no_grad():
                    for start in range(0, len(val_encoded), batch_size):
                        batch = val_encoded[start:start + batch_size]
                        if not batch:
                            continue
                        input_ids, labels, attn_mask = _collate(batch)
                        input_ids, labels, attn_mask = input_ids.to(device), labels.to(device), attn_mask.to(device)
                        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                        val_loss_total += out.loss.item()
                        val_batches += 1
                val_loss = val_loss_total / max(val_batches, 1)
                # val_metric 需要是"越高越好、大致有界"的数字才能复用现有迭代循环的
                # 停止条件数学（core/pipeline.py 假设 val_metric 越高越好）——validation loss
                # 本身是"越低越好、无界"，这里做一个单调递减变换压到 (0,1] 区间：
                # loss 越小，sft_score 越接近 1；不是伪造数据，是对真实测得的 val_loss 的
                # 诚实重新标度，需要在报告/论文里如实说明这是变换后的分数，不是准确率
                sft_score = 1.0 / (1.0 + val_loss)

                epochs_out.append({
                    "epoch": epoch + 1,
                    "train_loss": round(float(train_loss), 4),
                    "val_loss": round(float(val_loss), 4),
                    "val_metric": round(float(sft_score), 4),
                    "metric_name": "sft_score",
                    "per_class_metrics": {},
                    "confusion_highlights": [],
                })
                if total_steps >= max_train_steps:
                    break
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if attempt == 0 and ("out of memory" in msg or "mps backend" in msg):
                batch_size = max(1, batch_size // 2)
                continue
            return {"error": f"训练时出错：{e}"}

    adapter_state = get_peft_model_state_dict(model)
    adapter_state_np = {k: v.detach().cpu().numpy() for k, v in adapter_state.items()}
    return {"epochs": epochs_out, "adapter_state_dict": adapter_state_np}
