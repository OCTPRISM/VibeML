"""
core/augmentor.py  -  Phase 2.1：特征空间增强器 + 数据飞轮

差异化核心（Phase 2 路线图 §2.4）：
  现有 AutoML 工具缺乏系统性小数据策略。本模块在 TF-IDF 特征空间做
  两件 Phase 1 做不到的事：

  1. SMOTE 过采样（在特征向量空间插值生成合成样本）
     ── Phase 1 的 LLM 增强是文本层面；SMOTE 是特征层面
     ── 无需调用 API，纯本地运行，适合隐私敏感场景
     ── 对极小类别（< 5 条）有显著效果

  2. 数据飞轮（训练完成后用模型置信度反向过滤训练集）
     ── 找出模型"不确定"的训练样本 → 可能是噪声/错标
     ── 下一轮训练自动剔除，形成自我净化的闭环
     ── 这是竞品系统普遍缺失的机制

依赖：sklearn / scipy（已在 requirements.txt 中）
无需 imbalanced-learn，SMOTE 纯 numpy+scipy 实现。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Tuple

from sklearn.neighbors import NearestNeighbors

from config import AugmentReport, FlywheelReport

if TYPE_CHECKING:
    from core.trainer import Trainer


class Augmentor:
    """
    特征空间增强器。

    使用方式（在 trainer.train_with_eval 内部调用）：

        augmentor = Augmentor(strategy="smote")
        X_aug, y_aug = augmentor.apply(X_train_sparse, y_train)

    数据飞轮（训练完成后调用）：

        high_conf, report = augmentor.flywheel(data, trainer, threshold=0.65)
    """

    STRATEGIES = ("smote", "mixup", "none")

    # 数据飞轮的两道硬护栏——没有它们，flywheel 会把训练集打崩（见 flywheel() 的注释）。
    # 单轮最多删掉 20%：即使模型系统性地学偏了某个类别，也不会一次把那个类别删光。
    MAX_REMOVE_FRACTION = 0.20
    # 无论如何保留 60%：多轮迭代累积下来也不会指数级塌缩
    MIN_RETAIN_FRACTION = 0.60

    def __init__(self, strategy: str = "smote", random_state: int = 42):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy 必须是 {self.STRATEGIES} 之一")
        self.strategy     = strategy
        self.rng          = np.random.default_rng(random_state)
        self.last_report: AugmentReport | None = None
        self.pending_class_boost: Dict[str, float] = {}   # {标签: 倍数}，由自动调参闭环设置

    # ── 主入口 ───────────────────────────────────────────────────────────────

    def apply(
        self,
        X: sp.spmatrix,
        y: np.ndarray,
        label_encoder=None,
    ) -> Tuple[sp.spmatrix, np.ndarray, AugmentReport]:
        """
        对训练集做特征空间增强。

        Args:
            X:             TF-IDF 稀疏矩阵 (n_samples, n_features)
            y:             整数标签数组
            label_encoder: sklearn LabelEncoder，用于报告中显示类名

        Returns:
            (X_augmented, y_augmented, AugmentReport)
        """
        counts_before = dict(Counter(y.tolist()))

        # 把 {标签名: 倍数} 翻译成 {编码后的类别索引: 倍数}，供 _smote 使用
        boost_by_class: Dict[int, float] = {}
        if self.pending_class_boost and label_encoder is not None:
            known = set(label_encoder.classes_)
            for label_name, factor in self.pending_class_boost.items():
                if label_name in known:
                    boost_by_class[int(label_encoder.transform([label_name])[0])] = factor

        if self.strategy == "smote":
            X_out, y_out = self._smote(X, y, boost_by_class=boost_by_class)
        elif self.strategy == "mixup":
            X_out, y_out = self._mixup(X, y)
        else:
            X_out, y_out = X, y

        counts_after = dict(Counter(y_out.tolist()))

        # 把整数 key 转为类名（如果有 label_encoder）
        def decode(d):
            if label_encoder is None:
                return {str(k): v for k, v in d.items()}
            return {label_encoder.inverse_transform([k])[0]: v for k, v in d.items()}

        vals_before = list(counts_before.values())
        vals_after  = list(counts_after.values())

        report = AugmentReport(
            strategy              = self.strategy,
            original_train_count  = int(X.shape[0]),
            augmented_train_count = int(X_out.shape[0]),
            per_class_before      = decode(counts_before),
            per_class_after       = decode(counts_after),
            balance_ratio_before  = round(min(vals_before) / max(vals_before), 3) if vals_before else 0.0,
            balance_ratio_after   = round(min(vals_after)  / max(vals_after),  3) if vals_after  else 0.0,
        )
        self.last_report = report
        return X_out, y_out, report

    # ── 数据飞轮 ─────────────────────────────────────────────────────────────

    def flywheel(
        self,
        data: List[Dict],
        trainer: "Trainer",
        threshold: float | None = None,
    ) -> Tuple[List[Dict], FlywheelReport]:
        """
        用训练好的模型给训练数据找出**可能标错**的样本并剔除。

        ⚠ 这个方法之前的实现会把数据集打崩（真实复现：190 条 → 30 条），
        原因有三个，都已修正，改之前请先读懂为什么：

        1. **绝对阈值 0.65 不随类别数变化**。2 类问题随机基线是 0.5，0.65 尚可；
           5 类问题随机基线只有 0.2，此时一个校准良好的模型在困难样本上给出
           0.4 的置信度完全正常，却会被当成"噪声"删掉。现在阈值按类别数自适应。

        2. **"模型不确定" ≠ "样本标错了"**。决策边界附近的不确定样本恰恰是
           信息量最大的（主动学习的核心洞察就是优先要这些）。把它们全删掉，
           等于系统性地删掉最有价值的训练信号。现在只有"模型高置信度地预测成了
           **另一个**标签"才算真正指向标错，单纯的低置信度样本一律保留。

        3. **自我确认的正反馈**。打分用的模型就是在这批数据上训出来的，
           它只保留自己已经学会的、删掉自己没学会的——每轮迭代都让数据集
           朝"模型已有的偏见"收缩。现在有删除比例硬上限兜住这个循环。

        Args:
            data:      原始训练样本列表 [{"text": ..., "label": ...}]
            trainer:   训练完成的 Trainer 实例
            threshold: 置信度阈值；None 表示按类别数自适应（推荐）

        Returns:
            (clean_data, FlywheelReport)
        """
        if not hasattr(trainer.model, "predict_proba") or trainer.vectorizer is None:
            # 不支持概率输出（如 LinearSVC 未校准），跳过飞轮
            return data, FlywheelReport(
                total_checked    = len(data),
                high_confidence  = len(data),
                low_confidence   = 0,
                avg_confidence   = 1.0,
                threshold_used   = threshold,
                flagged_samples  = [],
            )

        texts  = [d["text"] for d in data]
        X      = trainer.vectorizer.transform(texts)

        # ⚠ 必须用**样本外**（out-of-fold）预测，不能用 trainer.model 直接打分。
        # trainer.model 是在这批数据上训出来的，它早就把错标样本背下来了——
        # 实测：故意混进 10 条明确标错的样本，in-sample 打分一条都抓不出来（0/10），
        # 因为模型对这些错标样本同样给出高置信度的"正确"预测。
        # cross_val_predict 保证每条样本的预测来自一个没见过它的模型，
        # 这是 confident learning 这类错标检测方法的基本前提。
        y_true = trainer.label_encoder.transform([d.get("label", "") for d in data])
        n_classes = max(int(len(trainer.label_encoder.classes_)), 2)
        try:
            from sklearn.base import clone
            from sklearn.model_selection import cross_val_predict
            # 折数不能超过最小类别样本数
            min_class = int(np.min(np.bincount(y_true))) if len(y_true) else 0
            folds = max(2, min(5, min_class))
            if min_class < 2:
                raise ValueError("类别样本太少，无法做样本外预测")
            probas = cross_val_predict(clone(trainer.model), X, y_true,
                                       cv=folds, method="predict_proba")
        except Exception:
            # 样本太少/模型不支持时，退回 in-sample 打分。这种情况下检测能力很弱
            # （见上面的说明），但下面的删除上限护栏保证它至少不会造成破坏
            probas = trainer.model.predict_proba(X)

        conf   = probas.max(axis=1)               # (n,)

        # 自适应阈值：随机猜的基线是 1/n_classes，"高置信度"应该是明显高于随机，
        # 而不是一个跟类别数无关的固定数字
        if threshold is None:
            random_baseline = 1.0 / n_classes
            threshold = min(0.65, random_baseline + (1.0 - random_baseline) * 0.45)

        # 只把"模型高置信度地预测成了另一个标签"当作可能标错的证据。
        # 单纯的低置信度（边界样本）一律保留——见上面第 2 条说明。
        candidates = []      # (badness, idx, pred_label, conf)
        for i, (sample, c, proba_row) in enumerate(zip(data, conf, probas)):
            pred_idx   = int(np.argmax(proba_row))
            pred_label = trainer.label_encoder.inverse_transform([pred_idx])[0]
            true_label = sample.get("label", "")
            if pred_label != true_label and c >= threshold:
                # 模型越自信地说"这条不是这个标签"，越可能真的标错了
                candidates.append((float(c), i, pred_label, float(c)))

        # 删除比例硬上限：无论有多少"疑似标错"，单轮最多删掉这么多。
        # 没有这个上限的话，模型只要系统性地学偏了一个类别，就会把那个类别
        # 的样本成批删光——数据集越删越偏，正反馈直接把训练集打崩
        max_removable = int(len(data) * self.MAX_REMOVE_FRACTION)
        floor = max(int(len(data) * self.MIN_RETAIN_FRACTION), n_classes * 4)
        max_removable = max(0, min(max_removable, len(data) - floor))

        candidates.sort(reverse=True)            # 最可疑的排前面
        remove_idx = {c[1] for c in candidates[:max_removable]}

        clean_data = [d for i, d in enumerate(data) if i not in remove_idx]
        flagged = [{
            "text":              data[i].get("text", "")[:80],
            "true_label":        data[i].get("label", "?"),
            "model_prediction":  pred_label,
            "confidence":        round(c_val, 3),
            "possible_mislabel": True,
        } for (_, i, pred_label, c_val) in candidates[:max_removable]]

        report = FlywheelReport(
            total_checked   = len(data),
            high_confidence = len(clean_data),
            low_confidence  = len(flagged),
            avg_confidence  = round(float(conf.mean()), 3),
            threshold_used  = round(float(threshold), 3),
            flagged_samples = flagged,
        )
        return clean_data, report

    # ── SMOTE（纯 numpy/scipy 实现）─────────────────────────────────────────

    def _smote(
        self,
        X:  sp.spmatrix,
        y:  np.ndarray,
        k:  int = 5,
        boost_by_class: Dict[int, float] | None = None,
    ) -> Tuple[sp.spmatrix, np.ndarray]:
        """
        SMOTE 在 TF-IDF 稀疏矩阵上的实现。

        算法：
          对每个少数类样本，找 k 个同类最近邻；
          在样本和随机邻居之间线性插值生成合成样本：
            synthetic = x_i + α * (x_neighbor - x_i),  α ∈ [0,1]

        目标：把所有类别的样本数对齐到最大类别的数量；若 boost_by_class 中
        指定了某个类别，该类别再额外过采样到 max * 倍数（用于自动调参闭环
        对表现弱的类别做超额补偿，而不只是追平最大类）。
        """
        boost_by_class = boost_by_class or {}
        classes, counts = np.unique(y, return_counts=True)
        base_target     = int(counts.max())          # 对齐到最大类别数

        X_parts = [X]
        y_parts = [y]

        for cls, count in zip(classes, counts):
            target = int(base_target * boost_by_class.get(int(cls), 1.0))
            needed = target - count
            if needed <= 0:
                continue

            mask  = (y == cls)
            X_cls = X[mask]
            n_cls = X_cls.shape[0]

            if n_cls < 2:
                # 只有 1 个样本，直接复制（退化情况）
                repeats = np.tile(X_cls, (needed, 1)) if not sp.issparse(X_cls) \
                          else sp.vstack([X_cls] * needed)
                X_parts.append(repeats)
                y_parts.append(np.full(needed, cls))
                continue

            k_actual = min(k, n_cls - 1)
            nbrs     = NearestNeighbors(
                n_neighbors = k_actual + 1,
                metric      = "cosine",
                algorithm   = "brute",
            ).fit(X_cls)
            _, indices = nbrs.kneighbors(X_cls)   # indices[:,0] 是自身

            synthetics = []
            for i in range(needed):
                src_idx  = i % n_cls
                nbr_col  = self.rng.integers(1, k_actual + 1)
                nbr_idx  = indices[src_idx, nbr_col]
                alpha    = self.rng.random()

                src = X_cls[src_idx]
                nbr = X_cls[nbr_idx]
                syn = src + alpha * (nbr - src)
                synthetics.append(syn)

            X_syn = sp.vstack(synthetics) if sp.issparse(X) else np.vstack(synthetics)
            X_parts.append(X_syn)
            y_parts.append(np.full(needed, cls))

        X_out = sp.vstack(X_parts) if sp.issparse(X) else np.vstack(X_parts)
        y_out = np.concatenate(y_parts)

        # 打乱顺序
        perm  = self.rng.permutation(len(y_out))
        return X_out[perm], y_out[perm]

    # ── Mixup（标签平滑插值）────────────────────────────────────────────────

    def _mixup(
        self,
        X:     sp.spmatrix,
        y:     np.ndarray,
        alpha: float = 0.2,
        n_aug: int   = None,
    ) -> Tuple[sp.spmatrix, np.ndarray]:
        """
        特征空间 Mixup：对同类样本对做凸组合，生成软边界样本。
        适合边界不清晰的细粒度分类任务。

        生成数量默认为原始训练集的 30%。
        """
        n      = X.shape[0]
        n_aug  = n_aug or max(10, int(n * 0.30))

        synthetics_X = []
        synthetics_y = []

        classes = np.unique(y)
        per_cls = n_aug // len(classes)

        for cls in classes:
            mask  = (y == cls)
            X_cls = X[mask]
            n_cls = X_cls.shape[0]
            if n_cls < 2:
                continue

            for _ in range(per_cls):
                i, j  = self.rng.choice(n_cls, size=2, replace=False)
                lam    = self.rng.beta(alpha, alpha)
                syn    = lam * X_cls[i] + (1 - lam) * X_cls[j]
                synthetics_X.append(syn)
                synthetics_y.append(cls)   # 同类 mixup，标签不变

        if not synthetics_X:
            return X, y

        X_syn = sp.vstack(synthetics_X) if sp.issparse(X) else np.vstack(synthetics_X)
        y_syn = np.array(synthetics_y)

        X_out = sp.vstack([X, X_syn]) if sp.issparse(X) else np.vstack([X, X_syn])
        y_out = np.concatenate([y, y_syn])
        return X_out, y_out
