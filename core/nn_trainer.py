"""
core/nn_trainer.py  -  真实神经网络训练执行（custom_nn 与 pretrained_nn 共用的
子进程执行 / 超时 / OOM / 资源上限逻辑）

设计要点（core/nn_sandbox.py 校验"代码写得对不对"之后，这里管"跑起来安不安全"）：
  - 每次训练都在独立子进程（multiprocessing, spawn）里跑，父进程用
    proc.join(timeout=...) + terminate()/kill() 兜底卡死的生成代码——
    线程杀不掉，进程才有真正的 terminate 能力，这是选子进程而不是线程/纯 exec 的原因。
  - 子进程内部：参数量上限检查（dry-run 实例化后立即检查，不合格直接失败，不浪费时间训练）、
    epoch 硬上限、MPS/CPU OOM 时降 batch size 重试一次。
  - 训练完成后，子进程只把"训练好的权重（state_dict，CPU tensor）+ 每轮指标"送回父进程；
    父进程在自己的进程里重建模型对象用于后续 predict()。子进程随后退出，
    占用的内存/GPU 上下文一起释放，不会一直占着资源。
  - 任何失败（静态校验不过 / 超时 / OOM / 运行时异常）都抛出明确异常，
    由 core/pipeline.py 的降级链决定重试/换路径，绝不静默吞掉。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from config import TaskSpec, TrainingConfig, EpochResult, GeneratedArchSpec, BackboneSpec, ModelBackend
from core.nn_sandbox import validate as sandbox_validate, extract_class_name
from core.subprocess_runner import run_in_subprocess, SubprocessTrainingError


def _per_class_f1(y_true, y_pred, label_names: List[str]) -> Dict[str, float]:
    """每个类别各自的 F1。只统计验证集里真实出现过的类别——对没出现的类别
    报 0.0 会让 Explainer 误以为"这个类别完全学不会"，其实只是没被抽到验证集。"""
    try:
        from sklearn.metrics import f1_score as _f1
        import numpy as _np
        y_true = _np.asarray(y_true); y_pred = _np.asarray(y_pred)
        present = sorted(set(y_true.tolist()))
        if not present:
            return {}
        scores = _f1(y_true, y_pred, labels=present, average=None, zero_division=0)
        out = {}
        for idx, sc in zip(present, scores):
            name = label_names[idx] if idx < len(label_names) else str(idx)
            out[name] = round(float(sc), 4)
        return out
    except Exception:
        return {}


def _confusion_pairs(y_true, y_pred, label_names: List[str], top_k: int = 3) -> List[str]:
    """最容易混淆的几对标签，形状跟 core/trainer.py::_confusion_highlights 对齐，
    这样 Explainer 的 prompt 不用区分后端。"""
    try:
        import numpy as _np
        from collections import Counter
        y_true = _np.asarray(y_true); y_pred = _np.asarray(y_pred)
        def nm(i):
            return label_names[i] if i < len(label_names) else str(i)
        pairs = Counter((int(t), int(p)) for t, p in zip(y_true, y_pred) if t != p)
        return [f"'{nm(t)}' 被误判为 '{nm(p)}'（{n} 次）"
                for (t, p), n in pairs.most_common(top_k)]
    except Exception:
        return []


def _sample_weights_for(y, label_names: List[str], class_boost: Dict[str, float]):
    """把 {标签名: 倍数} 翻译成逐样本采样权重。没有 boost 时返回 None，
    让下游走原来的均匀打散路径，不引入任何行为变化。"""
    if not class_boost:
        return None
    from core.loss_factory import build_sample_weights
    return build_sample_weights(y, label_names, class_boost).tolist()

MAX_PARAMS_CUSTOM     = 20_000_000    # custom_nn 分类头参数量上限
MAX_PARAMS_PRETRAINED = 300_000_000   # 预训练 backbone（含 head）参数量上限，仅做兜底提示
TRAIN_TIMEOUT_SECONDS = 300           # custom_nn（TF-IDF 小分类头）子进程超时
# pretrained_nn 实测：即使模型权重已缓存本地、LoRA + 4 epoch + batch 16，在这台机器上
# 真实 BERT-base 微调（CPU/MPS 真实反向传播）也经常超过 300 秒——这是真实训练耗时，
# 不是卡死，所以给它单独一个更宽松的预算，而不是缩短 epoch 到不真实的程度
PRETRAINED_TIMEOUT_SECONDS = 900
TFIDF_MAX_FEATURES    = 3000          # custom_nn 走 TF-IDF，控制特征维度避免生成网络第一层过大


class ArchValidationError(Exception):
    """生成代码没通过 core/nn_sandbox.py 静态校验"""
    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class NNTrainingError(SubprocessTrainingError):
    """子进程训练失败（超时 / OOM / 运行时异常），携带简短原因供 pipeline.py 决定降级"""


class NNTrainer:
    """
    真实神经网络训练器，接口与 core/trainer.py::Trainer 对齐（predict / predict_proba），
    以便 core/pipeline.py 在 sklearn / pretrained_nn / custom_nn 三种后端间无缝切换。
    """

    def __init__(self, task_spec: TaskSpec):
        self.task_spec      = task_spec
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.label_encoder  = LabelEncoder()
        self.config: Optional[TrainingConfig] = None
        self._fitted         = False
        self._backend        = None
        self._model_state    = None    # 训练完成后子进程送回的 state_dict（CPU tensor）
        self._arch_source    = None
        self._class_name     = None
        self._model_id       = None    # pretrained_nn 用

    # ── 主入口 ──────────────────────────────────────────────────────────────

    def train_with_eval(
        self, data: List[Dict], config: TrainingConfig,
        arch_spec: Optional[GeneratedArchSpec] = None,
        backbone_spec: Optional[BackboneSpec] = None,
    ) -> List[EpochResult]:
        self.config = config
        if config.backend == ModelBackend.CUSTOM_NN:
            return self._train_custom_nn(data, arch_spec)
        elif config.backend == ModelBackend.PRETRAINED_NN:
            return self._train_pretrained(data, config, backbone_spec)
        raise ValueError(f"NNTrainer 不支持 backend={config.backend}")

    # ── 模式二：custom_nn（LLM 生成分类头 + TF-IDF 特征） ───────────────────

    def _train_custom_nn(self, data: List[Dict], arch_spec: Optional[GeneratedArchSpec]) -> List[EpochResult]:
        if arch_spec is None:
            raise ValueError("custom_nn 模式需要 arch_spec")

        check = sandbox_validate(arch_spec.source_code)
        if not check.ok:
            raise ArchValidationError(check.errors)
        class_name = extract_class_name(arch_spec.source_code) or arch_spec.class_name

        texts, y = self._encode_labels(data)
        X_tr_raw, X_val_raw, y_tr, y_val = self._split(texts, y)

        self.vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2),
                                          sublinear_tf=True, min_df=1, strip_accents="unicode")
        X_tr  = self.vectorizer.fit_transform(X_tr_raw).toarray().astype("float32")
        X_val = self.vectorizer.transform(X_val_raw).toarray().astype("float32")

        payload = {
            "mode": "custom_nn",
            "source_code": arch_spec.source_code,
            "class_name": class_name,
            "loss_fn_name": self.config.hyperparam_overrides.get(
                "loss_fn", arch_spec.loss_fn_name),
            # 损失函数的可调参数 + class_boost 转出来的逐样本采样权重，
            # 都来自 Explainer 的诊断（见 core/loss_factory.py）。这是"取消
            # class_boost 只对 sklearn 生效"这个限制的落地点
            "loss_params": dict(self.config.hyperparam_overrides or {}),
            "sample_weights": _sample_weights_for(
                y_tr, list(self.label_encoder.classes_),
                (self.config.hyperparam_overrides or {}).get("class_boost") or {}),
            "X_train": X_tr, "y_train": y_tr.astype("int64"),
            "X_val": X_val, "y_val": y_val.astype("int64"),
            "input_dim": X_tr.shape[1], "num_classes": len(self.label_encoder.classes_),
            "label_names": [str(c) for c in self.label_encoder.classes_],
            "num_epochs": self.config.num_epochs,
            "batch_size": self.config.batch_size,
            "max_params": MAX_PARAMS_CUSTOM,
        }
        result = run_in_subprocess(payload, _dispatch_entrypoint, timeout=TRAIN_TIMEOUT_SECONDS,
                                   error_cls=NNTrainingError)

        self._model_state = result["state_dict"]
        self._arch_source = arch_spec.source_code
        self._class_name  = class_name
        self._backend     = "custom_nn"
        self._fitted      = True
        return [EpochResult(**e) for e in result["epochs"]]

    # ── 模式一：pretrained_nn（HF backbone + 可选 LoRA） ────────────────────

    def _train_pretrained(self, data: List[Dict], config: TrainingConfig,
                          backbone_spec: Optional[BackboneSpec]) -> List[EpochResult]:
        if backbone_spec is None:
            raise ValueError("pretrained_nn 模式需要 backbone_spec")

        texts, y = self._encode_labels(data)
        X_tr_raw, X_val_raw, y_tr, y_val = self._split(texts, y)

        payload = {
            "mode": "pretrained_nn",
            "model_id": backbone_spec.model_id,
            "use_lora": backbone_spec.use_lora,
            "X_train": list(X_tr_raw), "y_train": y_tr.astype("int64").tolist(),
            "X_val": list(X_val_raw), "y_val": y_val.astype("int64").tolist(),
            "num_classes": len(self.label_encoder.classes_),
            "num_epochs": min(config.num_epochs, 4),     # 预训练模型收敛快，epoch 上限更保守
            "batch_size": min(config.batch_size, 16),
            "max_params": MAX_PARAMS_PRETRAINED,
        }
        result = run_in_subprocess(payload, _dispatch_entrypoint, timeout=PRETRAINED_TIMEOUT_SECONDS,
                                   error_cls=NNTrainingError)

        self._model_state = result["state_dict"]
        self._model_id    = backbone_spec.model_id
        self._backend     = "pretrained_nn"
        self._fitted      = True
        return [EpochResult(**e) for e in result["epochs"]]

    # ── 预测（用送回的 state_dict 在父进程里重建模型）────────────────────────

    def predict(self, texts: List[str]) -> List[str]:
        probs = self._predict_proba_raw(texts)
        idx = probs.argmax(axis=1)
        return self.label_encoder.inverse_transform(idx).tolist()

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        probs = self._predict_proba_raw(texts)
        classes = list(self.label_encoder.classes_)
        return [dict(zip(classes, p.tolist())) for p in probs]

    def _predict_proba_raw(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("模型尚未训练")
        import torch
        # 子进程送回的是 numpy（见 _train_*_in_child 里的转换原因），这里转回 tensor 才能 load_state_dict
        state_dict = {k: torch.from_numpy(v) for k, v in self._model_state.items()}
        if self._backend == "custom_nn":
            X = self.vectorizer.transform(texts).toarray().astype("float32")
            model = _instantiate_custom_model(self._arch_source, self._class_name,
                                              X.shape[1], len(self.label_encoder.classes_))
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(X))
                probs = torch.softmax(logits, dim=1).numpy()
            return probs
        elif self._backend == "pretrained_nn":
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            model = AutoModelForSequenceClassification.from_pretrained(
                self._model_id, num_labels=len(self.label_encoder.classes_))
            model.load_state_dict(state_dict)
            model.eval()
            enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=1).numpy()
            return probs
        raise RuntimeError(f"未知 backend：{self._backend}")

    # ── 私有辅助 ─────────────────────────────────────────────────────────────

    def _encode_labels(self, data: List[Dict]):
        texts  = [d["text"] for d in data]
        labels = [d["label"] for d in data]
        known_labels = self.task_spec.label_schema if self.task_spec.label_schema else sorted(set(labels))
        all_labels = sorted(set(list(known_labels) + list(set(labels))))
        self.label_encoder.fit(all_labels)
        return texts, self.label_encoder.transform(labels)

    @staticmethod
    def _split(texts, y):
        try:
            return train_test_split(texts, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            return train_test_split(texts, y, test_size=0.2, random_state=42)


# ── 子进程执行（隔离机制本身在 core/subprocess_runner.py，这里只做 mode 分发）─────

def _dispatch_entrypoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """子进程内按 payload["mode"] 路由到对应训练函数——必须是模块级函数（spawn 要求可 pickle）"""
    if payload["mode"] == "custom_nn":
        return _train_custom_nn_in_child(payload)
    elif payload["mode"] == "pretrained_nn":
        return _train_pretrained_in_child(payload)
    return {"error": f"未知 mode：{payload['mode']}"}


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """替换掉的 __import__：只允许 import torch / torch.nn / torch.nn.functional。
    AST 门禁（core/nn_sandbox.py）已经在执行前拦过一遍非法 import，这里是第二层防线——
    exec() 环境本身也不给"能 import 任意模块"的能力，而不是仅仅"事先没被发现"。"""
    from core.nn_sandbox import ALLOWED_IMPORT_MODULES
    if name not in ALLOWED_IMPORT_MODULES:
        raise ImportError(f"import '{name}' 不被允许")
    import importlib
    module = importlib.import_module(name)
    if not fromlist:
        # "import a.b.c as x" 场景：真正的 __import__ 语义是返回顶层包 a，
        # 调用方（编译出的字节码）自己沿 a.b.c 做属性访问来绑定名字
        return importlib.import_module(name.split(".")[0])
    return module


def _safe_exec_globals() -> Dict[str, Any]:
    """restricted exec 用的最小 builtins 集合（AST 门禁是主要防线，这里是 belt-and-suspenders）"""
    import builtins
    safe_names = ("range", "len", "enumerate", "int", "float", "str", "bool", "list", "dict",
                  "tuple", "set", "super", "print", "min", "max", "sum", "abs", "isinstance",
                  "True", "False", "None", "ValueError", "TypeError", "Exception", "zip")
    safe_builtins = {name: getattr(builtins, name) for name in safe_names if hasattr(builtins, name)}
    safe_builtins["__import__"] = _restricted_import
    safe_builtins["__build_class__"] = builtins.__build_class__   # class 语句本身依赖它，不是可疑用法
    # __build_class__ 内部会读取模块级 __name__ 来设置 cls.__module__，这是常规 global 查找，不是 builtin
    return {"__builtins__": safe_builtins, "__name__": "generated_arch"}


def _instantiate_custom_model(source_code: str, class_name: str, input_dim: int, num_classes: int):
    exec_globals = _safe_exec_globals()
    exec(compile(source_code, "<generated_arch>", "exec"), exec_globals)
    cls = exec_globals[class_name]
    return cls(input_dim, num_classes)


def _train_custom_nn_in_child(payload: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    X_train = torch.from_numpy(payload["X_train"])
    y_train = torch.from_numpy(payload["y_train"])
    X_val   = torch.from_numpy(payload["X_val"])
    y_val   = torch.from_numpy(payload["y_val"])

    model = _instantiate_custom_model(payload["source_code"], payload["class_name"],
                                      payload["input_dim"], payload["num_classes"])
    n_params = sum(p.numel() for p in model.parameters())
    if n_params > payload["max_params"]:
        return {"error": f"生成的网络参数量过大（{n_params:,} > 上限 {payload['max_params']:,}），已拒绝训练"}

    # 损失函数由 core/loss_factory.py 按名字构建——之前这里是写死的两项字典，
    # 现在支持 focal / weighted_ce / label_smoothing，由 Explainer 按诊断出的
    # 错误模式选（类别不均衡→focal，标签噪声/过拟合→label_smoothing）
    from core.loss_factory import build_loss_fn, compute_class_weights
    _num_classes = int(payload["num_classes"])
    _class_weights = compute_class_weights(payload["y_train"], _num_classes)
    loss_fn = build_loss_fn(payload["loss_fn_name"],
                            payload.get("loss_params") or {}, _class_weights)

    batch_size = payload["batch_size"]
    num_epochs = min(payload["num_epochs"], 20)
    epochs_out: List[Dict] = []

    for attempt in range(2):    # OOM 时降 batch size 重试一次
        try:
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            n = X_train.shape[0]
            # class_boost 落地：有采样权重时用带放回的加权采样代替均匀打散，
            # 等价于 torch.utils.data.WeightedRandomSampler，但不必为了它引入
            # 一整套 Dataset/DataLoader（这里的数据本来就是一整块内存里的张量）
            _sw = payload.get("sample_weights")
            _sw_t = None
            if _sw is not None and len(_sw) == n and float(np.std(_sw)) > 1e-9:
                _sw_t = torch.tensor(np.asarray(_sw, dtype="float64"), dtype=torch.double)

            for epoch in range(num_epochs):
                model.train()
                perm = (torch.multinomial(_sw_t, n, replacement=True)
                        if _sw_t is not None else torch.randperm(n))
                total_loss = 0.0
                for start in range(0, n, batch_size):
                    idx = perm[start:start + batch_size]
                    xb, yb = X_train[idx].to(device), y_train[idx].to(device)
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(idx)
                train_loss = total_loss / max(n, 1)

                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val.to(device))
                    val_loss = loss_fn(val_logits, y_val.to(device)).item()
                    preds = val_logits.argmax(dim=1).cpu().numpy()
                val_metric = f1_score(payload["y_val"], preds, average="weighted", zero_division=0)

                # per_class_metrics 以前是硬编码的空字典，导致 NN 后端下
                # core/explainer.py 永远看不到"哪个类别弱"——而那正是它诊断
                # 类别不均衡、进而决定要不要换 focal loss 的唯一依据。空着的话
                # "按诊断选损失函数"这条闭环等于没有输入信号，必须真实算出来。
                epochs_out.append({
                    "epoch": epoch + 1,
                    "train_loss": round(float(train_loss), 4),
                    "val_loss": round(float(val_loss), 4),
                    "val_metric": round(float(val_metric), 4),
                    "metric_name": "f1",
                    "per_class_metrics": _per_class_f1(
                        payload["y_val"], preds, payload.get("label_names") or []),
                    "confusion_highlights": _confusion_pairs(
                        payload["y_val"], preds, payload.get("label_names") or []),
                })
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if attempt == 0 and ("out of memory" in msg or "mps backend" in msg):
                batch_size = max(4, batch_size // 2)
                epochs_out = []
                model = _instantiate_custom_model(payload["source_code"], payload["class_name"],
                                                  payload["input_dim"], payload["num_classes"])
                continue
            return {"error": f"训练时出错：{e}"}

    # 转成 numpy 再入队：torch.Tensor 走共享内存 pickling，子进程刚退出时后端文件可能已被回收，
    # 导致父进程反序列化报 "No such file or directory"；numpy 数组走标准 pickle，没有这个问题。
    state_dict = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    return {"epochs": epochs_out, "state_dict": state_dict}


def _train_pretrained_in_child(payload: Dict[str, Any]) -> Dict[str, Any]:
    import random
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model_id = payload["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=payload["num_classes"])

    n_params = sum(p.numel() for p in model.parameters())
    if n_params > payload["max_params"]:
        return {"error": f"预训练模型参数量过大（{n_params:,} > 上限 {payload['max_params']:,}）"}

    if payload.get("use_lora"):
        try:
            from peft import LoraConfig, get_peft_model, TaskType as PeftTaskType
            lora_cfg = LoraConfig(task_type=PeftTaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1)
            model = get_peft_model(model, lora_cfg)
        except Exception:
            pass    # LoRA 装配失败不影响全量微调兜底

    X_train, y_train = payload["X_train"], payload["y_train"]
    X_val, y_val = payload["X_val"], payload["y_val"]
    batch_size = payload["batch_size"]
    num_epochs = min(payload["num_epochs"], 4)
    epochs_out: List[Dict] = []

    for attempt in range(2):
        try:
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            n = len(X_train)
            for epoch in range(num_epochs):
                model.train()
                perm = list(range(n))
                random.shuffle(perm)
                total_loss = 0.0
                for start in range(0, n, batch_size):
                    batch_idx = perm[start:start + batch_size]
                    batch_texts = [X_train[i] for i in batch_idx]
                    batch_labels = torch.tensor([y_train[i] for i in batch_idx], device=device)
                    enc = tokenizer(batch_texts, padding=True, truncation=True,
                                    max_length=128, return_tensors="pt").to(device)
                    optimizer.zero_grad()
                    out = model(**enc, labels=batch_labels)
                    out.loss.backward()
                    optimizer.step()
                    total_loss += out.loss.item() * len(batch_idx)
                train_loss = total_loss / max(n, 1)

                model.eval()
                with torch.no_grad():
                    enc_val = tokenizer(X_val, padding=True, truncation=True,
                                        max_length=128, return_tensors="pt").to(device)
                    labels_val = torch.tensor(y_val, device=device)
                    out_val = model(**enc_val, labels=labels_val)
                    val_loss = out_val.loss.item()
                    preds = out_val.logits.argmax(dim=1).cpu().numpy()
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
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if attempt == 0 and ("out of memory" in msg or "mps backend" in msg):
                batch_size = max(2, batch_size // 2)
                epochs_out = []
                continue
            return {"error": f"训练时出错：{e}"}

    # 转成 numpy 再入队：torch.Tensor 走共享内存 pickling，子进程刚退出时后端文件可能已被回收，
    # 导致父进程反序列化报 "No such file or directory"；numpy 数组走标准 pickle，没有这个问题。
    state_dict = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    return {"epochs": epochs_out, "state_dict": state_dict}
