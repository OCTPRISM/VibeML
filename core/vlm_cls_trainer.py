"""
core/vlm_cls_trainer.py  -  VLM 图像分类训练执行（冻结 CLIP 视觉编码器 + 可训练分类头）

设计要点（跟 core/nn_trainer.py::NNTrainer 的 pretrained_nn 路径是同一个思路，
只是 backbone 换成视觉编码器、输入换成图片）：
  - 视觉编码器（CLIPVisionModelWithProjection）整个冻结，不参与反向传播——只是
    抽特征；真正训练的只有它上面一个很小的分类头（Linear-ReLU-Dropout-Linear）。
    这不是"custom_nn 那种 LLM 生成代码"，是这里自己写死的固定结构，不需要
    AST 沙盒校验（没有任意代码执行面）。
  - 复用 core/subprocess_runner.py 的子进程隔离——图片解码 + 编码器前向传播
    仍然可能吃掉不少内存/时间，跟其它训练器一样用墙钟超时兜底。
  - payload 里只传图片的服务器本地路径（字符串，天然可 pickle），不传 PIL
    Image/tensor——子进程自己用 PIL 读图 + CLIPImageProcessor 预处理，这样
    payload 体积跟图片数量无关，不会因为传大量像素数据拖慢子进程启动。
  - 用 transformers 的 CLIPImageProcessor（不是 AutoImageProcessor "fast" 版本）——
    这台机器没装 torchvision，AutoImageProcessor 的默认后端会因此直接报错；
    CLIPImageProcessor 在检测不到 torchvision 时会自动退回纯 PIL 实现，这是
    刻意选择而不是随手换的 API。
  - 因为编码器冻结、只训练一个小分类头，"epoch"是真实的梯度下降轮次（不是
    core/trainer.py 那种 sklearn 场景下模拟出来的东西）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from config import TaskSpec, EpochResult
from core.subprocess_runner import run_in_subprocess, SubprocessTrainingError

VLM_CLS_TIMEOUT_SECONDS = 600     # 冻结编码器只做前向传播，比真正微调整个模型快得多
MAX_EPOCHS = 20
EMBED_DIM  = 512    # openai/clip-vit-base-patch32 的 image_embeds 维度


class VlmClsTrainingError(SubprocessTrainingError):
    """子进程训练失败（图片读取失败/超时/运行时异常），携带简短原因供 pipeline.py 决定降级"""


class VlmClsTrainer:
    """接口与 core/nn_trainer.py::NNTrainer 对齐（predict/predict_proba），
    这样 core/vlm_cls_pipeline.py 能用和分类任务一致的方式消费它。"""

    def __init__(self, task_spec: TaskSpec):
        self.task_spec = task_spec
        self.label_encoder = LabelEncoder()
        self.encoder_id: Optional[str] = None
        self._head_state: Optional[Dict[str, Any]] = None
        self._fitted = False

    def train_with_eval(
        self, image_examples: List[Dict], encoder_id: str, num_epochs: int = 8,
    ) -> List[EpochResult]:
        paths, y = self._encode_labels(image_examples)
        X_tr_paths, X_val_paths, y_tr, y_val = self._split(paths, y)

        payload = {
            "encoder_id": encoder_id,
            "train_paths": X_tr_paths, "y_train": y_tr.astype("int64").tolist(),
            "val_paths": X_val_paths, "y_val": y_val.astype("int64").tolist(),
            "num_classes": len(self.label_encoder.classes_),
            "num_epochs": min(num_epochs, MAX_EPOCHS),
        }
        result = run_in_subprocess(payload, _vlm_cls_train_entrypoint,
                                   timeout=VLM_CLS_TIMEOUT_SECONDS, error_cls=VlmClsTrainingError)

        self._head_state = result["head_state_dict"]
        self.encoder_id = encoder_id
        self._fitted = True
        return [EpochResult(**e) for e in result["epochs"]]

    def predict(self, image_paths: List[str]) -> List[str]:
        probs = self._predict_proba_raw(image_paths)
        idx = probs.argmax(axis=1)
        return self.label_encoder.inverse_transform(idx).tolist()

    def predict_proba(self, image_paths: List[str]) -> List[Dict[str, float]]:
        probs = self._predict_proba_raw(image_paths)
        classes = list(self.label_encoder.classes_)
        return [dict(zip(classes, p.tolist())) for p in probs]

    def _predict_proba_raw(self, image_paths: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("模型尚未训练")
        import torch
        embeds = _extract_embeddings(self.encoder_id, image_paths)
        head = _build_head(EMBED_DIM, len(self.label_encoder.classes_))
        head.load_state_dict({k: torch.from_numpy(v) for k, v in self._head_state.items()})
        head.eval()
        with torch.no_grad():
            logits = head(torch.from_numpy(embeds))
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def _encode_labels(self, image_examples: List[Dict]):
        paths  = [e["image_path"] for e in image_examples]
        labels = [e["label"] for e in image_examples]
        self.label_encoder.fit(sorted(set(labels)))
        return paths, self.label_encoder.transform(labels)

    @staticmethod
    def _split(paths, y):
        try:
            return train_test_split(paths, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            return train_test_split(paths, y, test_size=0.2, random_state=42)


# ── 子进程执行（隔离机制本身在 core/subprocess_runner.py，这里只做真实训练）───────

def _load_image(path: str):
    from pathlib import Path
    from PIL import Image
    from core.data_sources import UPLOAD_DIR
    full_path = (UPLOAD_DIR / path).resolve() if not Path(path).is_absolute() else Path(path)
    return Image.open(full_path).convert("RGB")


def _extract_embeddings(encoder_id: str, image_paths: List[str]) -> np.ndarray:
    """冻结编码器抽特征——训练和预测都调这个，保证特征提取逻辑只有一份。"""
    import torch
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    processor = CLIPImageProcessor.from_pretrained(encoder_id)
    encoder = CLIPVisionModelWithProjection.from_pretrained(encoder_id)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    images = [_load_image(p) for p in image_paths]
    embeds = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            out = encoder(**inputs)
            embeds.append(out.image_embeds.numpy())
    return np.concatenate(embeds, axis=0).astype("float32")


def _build_head(embed_dim: int, num_classes: int):
    """固定结构的小分类头——不是生成代码，不需要沙盒校验。"""
    import torch.nn as nn
    hidden = max(32, embed_dim // 4)
    return nn.Sequential(
        nn.Linear(embed_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, num_classes),
    )


def _vlm_cls_train_entrypoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    encoder_id = payload["encoder_id"]
    try:
        X_train = _extract_embeddings(encoder_id, payload["train_paths"])
        X_val   = _extract_embeddings(encoder_id, payload["val_paths"])
    except Exception as e:
        return {"error": f"图片编码失败：{e}"}

    y_train = np.array(payload["y_train"], dtype="int64")
    y_val   = np.array(payload["y_val"], dtype="int64")
    num_classes = payload["num_classes"]
    num_epochs = payload["num_epochs"]

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    head = _build_head(X_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t   = torch.from_numpy(X_val).to(device)
    y_val_t   = torch.from_numpy(y_val).to(device)

    batch_size = 16
    n = X_train_t.shape[0]
    epochs_out: List[Dict] = []

    for epoch in range(num_epochs):
        head.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / max(n, 1)

        head.eval()
        with torch.no_grad():
            val_logits = head(X_val_t)
            val_loss = F.cross_entropy(val_logits, y_val_t).item()
            preds = val_logits.argmax(dim=1).cpu().numpy()
        val_metric = f1_score(y_val, preds, average="weighted", zero_division=0)

        epochs_out.append({
            "epoch": epoch + 1,
            "train_loss": round(float(train_loss), 4),
            "val_loss": round(float(val_loss), 4),
            "val_metric": round(float(val_metric), 4),
            "metric_name": "f1",
            "per_class_metrics": {},
            "confusion_highlights": [],
        })

    head_state = {k: v.detach().cpu().numpy() for k, v in head.state_dict().items()}
    return {"epochs": epochs_out, "head_state_dict": head_state}
