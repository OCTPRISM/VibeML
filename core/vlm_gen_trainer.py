"""
core/vlm_gen_trainer.py  -  生成式 VLM 训练执行（看图说话 / 有参考答案的视觉问答）

设计要点（跟 core/vlm_cls_trainer.py 的分工不同——那边编码器整个冻结，这里
整个 seq2seq 模型都参与训练）：
  - 默认底座 yuanzhoulvpi/vit-gpt2-image-chinese-captioning（ViT 编码器 + GPT2
    解码器，~242M 参数，跟 core/llm_ft_trainer.py 要应对的十亿级 causal LM
    完全不是一个量级）全量微调（不是 LoRA）在这台机器上完全跑得动。
    **这不是本项目最初选的 Salesforce/blip-image-captioning-base**——BLIP 的
    默认分词器是纯英文 BERT WordPiece 词表，真实训练验证过：喂中文参考答案
    进去，绝大部分汉字被分成 [UNK]，ROUGE-L 全程卡在 0，模型架构上就学不会
    中文（不是训练代码的 bug——同一套代码换成英文参考答案，ROUGE-L 第一轮
    就到 1.0）。换成这个 ViT-GPT2 中文模型是因为它的分词器（中文 BERT 词表）
    对中英文都能正确切分/还原，且这台机器没装 torchvision、Qwen2-VL 一类
    更大的多语言模型的 Processor 在类型检查层面硬性要求一个真正的
    torchvision-backed video processor（哪怕根本不处理视频），装不上匹配这台
    机器 torch 版本的 torchvision，只能放弃更大的模型。
  - 这个模型的 AutoProcessor.from_pretrained 不会返回打包好的组合 Processor
    （模型仓库没有注册 processor_config.json），直接返回裸 tokenizer——所以
    这里统一自己分别加载 ViTImageProcessor（图片）+ AutoTokenizer（文字），
    不依赖某个模型是否打包了组合 Processor。刻意不用 AutoImageProcessor——
    它在类级别就无条件要求 torchvision 没有 PIL 兜底；具体的 ViTImageProcessor
    在检测不到 torchvision 时会自动退回纯 PIL 实现（跟 core/vlm_cls_trainer.py
    用 CLIPImageProcessor 而不是 AutoImageProcessor 是同一个理由）。
  - VisionEncoderDecoderModel（这个模型的架构）的前向传播用 pixel_values +
    decoder_input_ids + labels（不是 BLIP 那种 input_ids），生成用
    model.generate(pixel_values=..., decoder_input_ids=<可选前缀>)。
  - 数据分两种情况，训练目标构造不同：
      * 无 prompt（纯看图说话）：参考答案本身既当 decoder 输入又当 label
        （只传 labels，模型内部的 GPT2 decoder 自己处理好错位预测）。
      * 有 prompt（视觉问答）：拼成 "问题：{prompt}\n回答：" 作为条件前缀，
        手动构造 decoder_input_ids=前缀+答案、labels 中前缀部分设成 -100
        屏蔽掉——标准的指令微调 loss mask 做法，跟 core/llm_ft_trainer.py
        对 instruction 部分做 loss mask 是同一个思路。
  - 指标用 ROUGE-L F1（有参考答案的生成任务里最基本、最省事的有界相似度
    指标，0-1 越高越好），不是 core/nn_trainer.py 那种模拟出来的东西——
    真实在验证集上生成文本、跟参考答案算真实 ROUGE-L。
  - 中文场景下 rouge_score 默认按空格分词（对无空格的中文几乎不起作用），
    这里统一按字符切分再算 ROUGE——对中英文都适用的简单折中，不是文献级别
    的多语言评测方案，但作为训练时的有界反馈信号完全够用。
  - 复用 core/subprocess_runner.py 做子进程隔离，跟其它训练器一致。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from config import TaskSpec, EpochResult
from core.subprocess_runner import run_in_subprocess, SubprocessTrainingError

VLM_GEN_TIMEOUT_SECONDS = 1800   # 全量微调一个 ~242M 的 seq2seq 模型，比冻结编码器慢得多
MAX_EPOCHS = 10
PROMPT_TEMPLATE = "问题：{prompt}\n回答："


class VlmGenTrainingError(SubprocessTrainingError):
    """子进程训练失败（图片读取失败/超时/运行时异常），携带简短原因供 pipeline.py 决定降级"""


class _CharTokenizer:
    """rouge_score 的默认分词器用的是纯 ASCII 字母数字正则——真实测试过：喂纯中文
    句子进去，两个完全相同的句子也会算出 fmeasure=0（默认分词器把所有汉字都当成
    "非单词"字符过滤掉了，两边都变成空 token 列表，0/0 按惯例算 0）。这不是靠
    "手动按字符加空格再传给 score()" 能绕过的——问题出在 rouge_score 内部的分词器，
    不是输入字符串的格式，所以直接把自定义的按字符切分分词器传给 RougeScorer
    （对中英文都按单个字符切分，是一个简单、能用但不是文献级别的多语言方案）。"""
    def tokenize(self, text: str):
        return [c for c in text if not c.isspace()]


def _load_processors(model_id: str) -> Tuple[Any, Any]:
    """返回 (image_processor, tokenizer)。不用 AutoImageProcessor（类级别硬依赖
    torchvision，没有 PIL 兜底）——ViTImageProcessor 在检测不到 torchvision 时
    会自动退回纯 PIL 实现，这是刻意选择，不是随手换的 API。"""
    from transformers import ViTImageProcessor, AutoTokenizer
    image_processor = ViTImageProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return image_processor, tokenizer


class VlmGenTrainer:
    """接口与 core/vlm_cls_trainer.py::VlmClsTrainer 对齐（train_with_eval/predict），
    这样 core/vlm_gen_pipeline.py 能用同样的方式消费它。"""

    def __init__(self, task_spec: TaskSpec):
        self.task_spec = task_spec
        self.model_id: Optional[str] = None
        self._model_state: Optional[Dict[str, Any]] = None
        self._fitted = False

    def train_with_eval(
        self, vlm_examples: List[Dict], model_id: str, num_epochs: int = 5,
    ) -> List[EpochResult]:
        train_examples, val_examples = self._split(vlm_examples)

        payload = {
            "model_id": model_id,
            "train_examples": train_examples,
            "val_examples": val_examples,
            "num_epochs": min(num_epochs, MAX_EPOCHS),
        }
        result = run_in_subprocess(payload, _vlm_gen_train_entrypoint,
                                   timeout=VLM_GEN_TIMEOUT_SECONDS, error_cls=VlmGenTrainingError)

        self._model_state = result["model_state_dict"]
        self.model_id = model_id
        self._fitted = True
        return [EpochResult(**e) for e in result["epochs"]]

    def predict(self, examples: List[Dict]) -> List[str]:
        """examples: [{"image_path": str, "prompt": str}]，用于部署后的真实推理验证。"""
        if not self._fitted:
            raise RuntimeError("模型尚未训练")
        import torch
        from transformers import AutoModelForImageTextToText

        image_processor, tokenizer = _load_processors(self.model_id)
        model = AutoModelForImageTextToText.from_pretrained(self.model_id)
        model.load_state_dict({k: torch.from_numpy(v) for k, v in self._model_state.items()})
        model.eval()

        outputs = []
        with torch.no_grad():
            for ex in examples:
                image = _load_image(ex["image_path"])
                pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
                prompt = ex.get("prompt") or ""
                text = _generate(model, tokenizer, pixel_values, prompt)
                outputs.append(text)
        return outputs

    @staticmethod
    def _split(examples: List[Dict]):
        try:
            return train_test_split(examples, test_size=0.25, random_state=42)
        except ValueError:
            # 样本太少没法切分验证集时，训练/验证用同一份（跟 nn_trainer.py 小样本兜底一致）
            return examples, examples


# ── 子进程执行（隔离机制本身在 core/subprocess_runner.py，这里只做真实训练）───────

def _load_image(path: str):
    from pathlib import Path
    from PIL import Image
    from core.data_sources import UPLOAD_DIR
    full_path = (UPLOAD_DIR / path).resolve() if not Path(path).is_absolute() else Path(path)
    return Image.open(full_path).convert("RGB")


def _generate(model, tokenizer, pixel_values, prompt: str) -> str:
    """有 prompt 时把它当解码前缀喂给 generate（decoder_input_ids），生成结果里
    再把前缀原样去掉，只留真正生成的部分——跟训练时"只在答案部分算 loss"
    是同一套"prompt 是条件、不是要生成的内容"的处理逻辑。"""
    if prompt:
        prefix = PROMPT_TEMPLATE.format(prompt=prompt)
        prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids
        out_ids = model.generate(pixel_values=pixel_values, decoder_input_ids=prefix_ids, max_new_tokens=64)
        full_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        prefix_text = tokenizer.decode(prefix_ids[0], skip_special_tokens=True)
        return full_text[len(prefix_text):].strip() if full_text.startswith(prefix_text) else full_text.strip()
    out_ids = model.generate(pixel_values=pixel_values, max_new_tokens=64)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


def _build_training_targets(tokenizer, example: Dict):
    """按有无 prompt 构造 (decoder_input_ids, labels)——有 prompt 时屏蔽掉前缀部分的 loss。"""
    import torch

    reference = example["reference_answer"]
    prompt = example.get("prompt") or ""

    if not prompt:
        # 纯看图说话：只传 labels，模型（GPT2 解码器）内部自己处理好"预测下一个 token"
        # 的错位对齐，不需要手动构造 decoder_input_ids
        labels = tokenizer(reference, return_tensors="pt", truncation=True, max_length=64).input_ids
        return None, labels

    prefix = PROMPT_TEMPLATE.format(prompt=prompt)
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    full_ids = tokenizer(prefix + reference, truncation=True, max_length=96).input_ids
    decoder_input_ids = torch.tensor([full_ids])
    labels = decoder_input_ids.clone()
    labels[0, :len(prefix_ids)] = -100
    return decoder_input_ids, labels


def _rouge_l(hypotheses: List[str], references: List[str]) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=_CharTokenizer())
    scores = [scorer.score(ref, hyp)["rougeL"].fmeasure
              for hyp, ref in zip(hypotheses, references)]
    return float(np.mean(scores)) if scores else 0.0


def _vlm_gen_train_entrypoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForImageTextToText

    model_id = payload["model_id"]
    train_examples = payload["train_examples"]
    val_examples = payload["val_examples"]
    num_epochs = payload["num_epochs"]

    try:
        image_processor, tokenizer = _load_processors(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id)
    except Exception as e:
        return {"error": f"模型加载失败：{e}"}

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    try:
        train_images = [_load_image(e["image_path"]) for e in train_examples]
        val_images = [_load_image(e["image_path"]) for e in val_examples]
    except Exception as e:
        return {"error": f"图片读取失败：{e}"}

    epochs_out: List[Dict] = []
    batch_size = 4

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(train_examples), batch_size):
            batch_ex = train_examples[start:start + batch_size]
            batch_img = train_images[start:start + batch_size]
            optimizer.zero_grad()
            batch_loss = 0.0
            # 一次只吃一张图对应一段文本，没有把整个 batch 一次性 collate 成张量
            # （不同长度 prompt 的 label 屏蔽长度不一致，硬 pad 成同一个 batch tensor
            # 增加不必要的复杂度）——直接逐样本前反向传播、梯度自然累加，等价于
            # 这个小 batch 的平均梯度更新，训练数据量本来就很小
            for ex, img in zip(batch_ex, batch_img):
                decoder_input_ids, labels = _build_training_targets(tokenizer, ex)
                pixel_values = image_processor(images=img, return_tensors="pt").pixel_values.to(device)
                kwargs = {"pixel_values": pixel_values, "labels": labels.to(device)}
                if decoder_input_ids is not None:
                    kwargs["decoder_input_ids"] = decoder_input_ids.to(device)
                out = model(**kwargs)
                loss = out.loss / len(batch_ex)
                loss.backward()
                batch_loss += loss.item()
            optimizer.step()
            total_loss += batch_loss
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        hyps, refs = [], []
        with torch.no_grad():
            for ex, img in zip(val_examples, val_images):
                pixel_values = image_processor(images=img, return_tensors="pt").pixel_values.to(device)
                prompt = ex.get("prompt") or ""
                text = _generate(model, tokenizer, pixel_values, prompt)
                hyps.append(text)
                refs.append(ex["reference_answer"])
        val_metric = _rouge_l(hyps, refs)

        epochs_out.append({
            "epoch": epoch + 1,
            "train_loss": round(float(train_loss), 4),
            "val_loss": round(float(train_loss), 4),  # 没有独立算验证集 loss（生成需要 decode），train_loss 作为近似趋势参考
            "val_metric": round(float(val_metric), 4),
            "metric_name": "rouge_l",
            "per_class_metrics": {},
            "confusion_highlights": [],
        })

    model_state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    return {"epochs": epochs_out, "model_state_dict": model_state}
