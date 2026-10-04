"""
core/drift_detector.py  -  Phase 4：持续学习 · 数据漂移检测

路线图 Phase 4：持续学习——部署后模型自动适应数据漂移。

核心能力：
  1. 特征漂移检测（KS 检验，文本 TF-IDF 分布）
  2. 标签漂移检测（类别分布变化）
  3. 性能漂移检测（监控指标下降）
  4. 自动给出重训建议 + 样本收集优先级

与 pipeline.check_feedback() 的区别：
  - check_feedback：基于单一聚合指标判断
  - DriftDetector：多维度统计检验，给出更细粒度诊断

依赖：sklearn / scipy（已包含在 requirements.txt）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp


@dataclass
class DriftReport:
    """漂移检测结果"""
    overall_drift_score:    float           # 0-1，越高越严重
    is_drifted:             bool
    feature_drift:          Dict[str, float]  # 维度 → KS 统计量
    label_drift:            Dict[str, float]  # 标签 → 分布变化
    performance_drift:      Optional[float]   # 性能下降量（若有）
    recommendation:         str
    priority_labels:        List[str]         # 最需要补充样本的类别
    retrain_urgency:        str               # "immediate" / "soon" / "monitor"


class DriftDetector:
    """
    生产数据漂移检测器。

    用法：
        # 训练完成后注册参考分布
        detector = DriftDetector()
        detector.fit_reference(train_texts, train_labels, trainer.vectorizer)

        # 定期对生产数据做检测
        report = detector.detect(production_texts, production_labels)
        if report.is_drifted:
            trigger_retraining()
    """

    def __init__(
        self,
        feature_drift_threshold: float = 0.15,  # KS 统计量阈值
        label_drift_threshold:   float = 0.10,  # 标签分布变化阈值
        n_components:            int   = 50,    # 降维后维度（加速检验）
    ):
        self.feature_drift_threshold = feature_drift_threshold
        self.label_drift_threshold   = label_drift_threshold
        self.n_components            = n_components

        self._ref_features:    Optional[np.ndarray] = None
        self._ref_label_dist:  Dict[str, float]     = {}
        self._vectorizer       = None
        self._svd              = None
        self._fitted           = False

    def fit_reference(
        self,
        texts:      List[str],
        labels:     List[str],
        vectorizer,
    ):
        """
        注册参考分布（训练集）。

        Args:
            texts:      训练文本列表
            labels:     训练标签列表
            vectorizer: 已拟合的 TF-IDF 向量化器
        """
        from sklearn.decomposition import TruncatedSVD

        self._vectorizer = vectorizer
        X = vectorizer.transform(texts)

        # 使用 SVD 降维加速 KS 检验
        n_comp = min(self.n_components, X.shape[1] - 1, X.shape[0] - 1)
        if n_comp > 0:
            self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
            self._ref_features = self._svd.fit_transform(X)
        else:
            self._ref_features = X.toarray() if sp.issparse(X) else X

        # 标签分布
        total = len(labels)
        from collections import Counter
        counts = Counter(labels)
        self._ref_label_dist = {k: v / total for k, v in counts.items()}
        self._fitted = True

    def detect(
        self,
        texts:              List[str],
        labels:             Optional[List[str]] = None,
        performance_metric: Optional[float]     = None,
    ) -> DriftReport:
        """
        检测生产数据相对于训练集的漂移程度。

        Args:
            texts:              生产数据文本列表
            labels:             生产数据标签（可选）
            performance_metric: 当前生产指标（可选，用于性能漂移检测）

        Returns:
            DriftReport
        """
        if not self._fitted:
            raise RuntimeError("请先调用 fit_reference() 注册参考分布")

        # ── 1. 特征漂移（KS 检验） ────────────────────────────────────────────
        X_prod = self._vectorizer.transform(texts)
        if self._svd is not None:
            X_prod_reduced = self._svd.transform(X_prod)
        else:
            X_prod_reduced = X_prod.toarray() if sp.issparse(X_prod) else X_prod

        feature_drift = self._ks_test(self._ref_features, X_prod_reduced)
        avg_feature_drift = float(np.mean(list(feature_drift.values())))

        # ── 2. 标签漂移 ──────────────────────────────────────────────────────
        label_drift: Dict[str, float] = {}
        priority_labels: List[str]    = []
        if labels:
            from collections import Counter
            total = len(labels)
            prod_dist = {k: v / total for k, v in Counter(labels).items()}
            for lbl, ref_pct in self._ref_label_dist.items():
                prod_pct = prod_dist.get(lbl, 0.0)
                change   = abs(prod_pct - ref_pct)
                label_drift[lbl] = round(change, 4)
            # 最需要补充的类别（生产中占比下降的）
            priority_labels = sorted(
                [l for l in label_drift if label_drift[l] > self.label_drift_threshold],
                key=lambda l: label_drift[l], reverse=True
            )[:3]

        avg_label_drift = float(np.mean(list(label_drift.values()))) if label_drift else 0.0

        # ── 3. 综合漂移分数 ───────────────────────────────────────────────────
        overall = min(1.0, (avg_feature_drift / self.feature_drift_threshold * 0.6 +
                            avg_label_drift   / self.label_drift_threshold   * 0.4))
        is_drifted = overall > 0.5

        # ── 4. 紧迫度和建议 ──────────────────────────────────────────────────
        if overall > 0.8:
            urgency = "immediate"
            rec = "严重漂移：立即重新训练，优先收集新数据"
        elif overall > 0.5:
            urgency = "soon"
            rec = f"显著漂移：建议在 1–2 周内重新训练"
            if priority_labels:
                rec += f"，重点补充标签 {priority_labels} 的样本"
        elif is_drifted:
            urgency = "soon"
            rec = "轻度漂移：持续监控，收集边界样本"
        else:
            urgency = "monitor"
            rec = "数据分布稳定，继续监控"

        return DriftReport(
            overall_drift_score  = round(overall, 4),
            is_drifted           = is_drifted,
            feature_drift        = {f"dim_{i}": round(v, 4)
                                    for i, v in enumerate(feature_drift.values()) if i < 10},
            label_drift          = label_drift,
            performance_drift    = performance_metric,
            recommendation       = rec,
            priority_labels      = priority_labels,
            retrain_urgency      = urgency,
        )

    # ── 私有方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _ks_test(
        ref:  np.ndarray,
        prod: np.ndarray,
    ) -> Dict[str, float]:
        """对每个降维维度做 KS 检验，返回统计量字典"""
        from scipy.stats import ks_2samp
        n_dims = min(ref.shape[1], prod.shape[1])
        result = {}
        for i in range(n_dims):
            stat, _ = ks_2samp(ref[:, i], prod[:, i])
            result[f"dim_{i}"] = round(float(stat), 4)
        return result
