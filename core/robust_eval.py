"""
core/robust_eval.py  -  可信的指标估计 + 噪声门禁

为什么需要这个模块（这是整套"自动调整"能力的地基）：

系统面向的是 10~200 条/类的低资源场景。一次 20% 切分下来，验证集常常只有
二三十条——这种规模上，val_metric 差 5 个点完全可能只是切分抖动。原来
core/pipeline.py 里是一行裸的 `if latest.val_metric > best_metric`，
等于把噪声当成真实提升接受下来。

一旦让 Agent 基于这个信号去决定"要不要换损失函数/重构网络结构"，它会
**追着噪声改架构**，而且因为验证集同样小，改完看起来"提升了"——其实是过拟合
到那二十几条验证样本上。之前 eval/runner.py 踩过同型的坑（重复 3 次方差恒为 0）。

本模块提供两件事：
  1. estimate_noise() —— 给出"即使模型一点没变，重新抽一次验证集也会有的波动"
  2. is_real_improvement() —— 提升必须超过这个波动才算数，否则判为无效改动

对"为什么不总是做交叉验证"的诚实说明：sklearn 后端上 CV 很便宜（百来条样本
毫秒级），但 NN 后端单次训练可能几十分钟，5 折就是 5 倍——那不现实。所以
NN 后端走"二项分布标准误"这个零成本的解析下界，见 binomial_standard_error()。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

# 样本量低于这个数时，sklearn 后端自动切到交叉验证（单次切分的验证集太小，不可信）
CV_SAMPLE_THRESHOLD = 500
DEFAULT_CV_FOLDS = 5

# 判定"真提升"要求超过几倍噪声标准差。
# 取 1.0 而不是更严格的 2.0：验证集只有二三十条时，2σ 门槛意味着任何改动都
# 永远通不过，整个自动调整循环会退化成"什么都不做"。1σ 大致对应"更可能是
# 真效应而不是巧合"，是在"别追噪声"和"别把循环卡死"之间的折中——这个取舍
# 本身要在事件里如实暴露给用户，不能假装是确定性结论。
SIGNIFICANCE_SIGMA = 1.0


@dataclass
class RobustMetric:
    """一次评估的可信度描述。mean 是点估计，std 是"这个点估计有多不可靠"。"""
    mean: float
    std: float
    metric_name: str = "accuracy"
    n_val_samples: int = 0
    n_folds: int = 1                      # 1 = 单次切分；>1 = 交叉验证
    per_fold: List[float] = field(default_factory=list)
    noise_source: str = "binomial"        # "cv" | "binomial" | "cv+binomial"

    @property
    def is_cv(self) -> bool:
        return self.n_folds > 1

    def summary(self) -> str:
        band = f"±{self.std:.1%}" if self.std else ""
        how = f"{self.n_folds} 折交叉验证" if self.is_cv else f"{self.n_val_samples} 条验证样本"
        return f"{self.mean:.1%} {band}（{how}）"


def binomial_standard_error(p: float, n: int) -> float:
    """准确率/F1 这类比例型指标，在 n 条验证样本上的标准误：sqrt(p(1-p)/n)。

    含义："即使模型完全没变，换一批同样大小的验证集重新测一次，指标也会有
    这么大的自然波动"。这是噪声的**下界**——真实波动只会更大（还叠加了
    训练随机性、切分随机性）。

    这个值在小验证集上大得惊人，而这正是重点：n=20、p=0.7 时标准误就有 10.2%，
    也就是说在 20 条样本上你**根本分不清 70% 和 75%**。与其让 Agent 基于这种
    差异去重构网络，不如如实告诉它"这个差异不可信"。
    """
    if n <= 0:
        return 0.5      # 没有验证样本，等于完全不知道
    p = min(max(p, 0.0), 1.0)
    # p(1-p) 在 p=0 或 1 时为 0，会给出"零噪声"的错觉——用 Wilson 式的下界兜住，
    # 避免 100% 准确率被当成"完全可信"（小样本上 20/20 全对是很容易碰巧发生的）
    variance = max(p * (1.0 - p), 0.25 / max(n, 1))
    return math.sqrt(variance / n)


def estimate_noise(metric: float, n_val_samples: int,
                   cv_std: Optional[float] = None, n_folds: int = 1) -> RobustMetric:
    """把一次评估结果包装成带噪声估计的 RobustMetric。

    cv_std 非空（做过交叉验证）时，取 CV 标准差和二项标准误里**大的那个**——
    两者度量的是不同来源的波动（CV std 含训练随机性，二项标准误含抽样随机性），
    取大值是保守但诚实的做法，不会低估不确定性。
    """
    binom = binomial_standard_error(metric, n_val_samples)
    if cv_std is not None and n_folds > 1:
        std = max(cv_std, binom)
        source = "cv+binomial" if cv_std < binom else "cv"
    else:
        std = binom
        source = "binomial"
    return RobustMetric(mean=metric, std=std, n_val_samples=n_val_samples,
                        n_folds=n_folds, noise_source=source)


def is_real_improvement(new: RobustMetric, baseline: float,
                        sigma: float = SIGNIFICANCE_SIGMA) -> tuple[bool, str]:
    """改动带来的提升是否超过噪声。返回 (是否真提升, 给人看的一句话解释)。

    这是防止"Agent 追着噪声改架构"的核心门禁：任何被判为 False 的改动，
    调用方都应该**不采纳**（不更新 best_metric、不把它当成"这个方向有效"的证据）。
    """
    delta = new.mean - baseline
    threshold = sigma * new.std

    if delta <= 0:
        return False, (f"指标从 {baseline:.1%} 变为 {new.mean:.1%}，没有提升")
    if delta < threshold:
        return False, (
            f"指标从 {baseline:.1%} 升到 {new.mean:.1%}（+{delta:.1%}），"
            f"但这个差异小于噪声水平 ±{new.std:.1%}——在{new.n_val_samples}条验证样本上"
            f"无法区分它是真实提升还是抽样波动，判定为无效改动"
        )
    return True, (
        f"指标从 {baseline:.1%} 升到 {new.mean:.1%}（+{delta:.1%}），"
        f"超过噪声水平 ±{new.std:.1%}，判定为真实提升"
    )


def should_use_cv(n_samples: int, backend_is_cheap: bool) -> bool:
    """要不要走交叉验证。backend_is_cheap 由调用方判断（sklearn=True，NN=False）——
    NN 后端单次训练可能几十分钟，5 折不现实，只能靠解析噪声下界。"""
    return backend_is_cheap and n_samples < CV_SAMPLE_THRESHOLD


def cross_val_metric(estimator_factory, X, y, metric_name: str = "accuracy",
                     n_folds: int = DEFAULT_CV_FOLDS) -> RobustMetric:
    """对 sklearn 估计器做分层交叉验证，返回 mean±std。

    estimator_factory: 无参可调用，每折都新建一个全新未训练的估计器
                       （不能复用同一个实例，否则第二折是在第一折的基础上继续训练）
    X: 原始文本列表（向量化在每折内部做，避免验证集信息通过 TF-IDF 词表泄漏到训练集）
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    # 折数不能超过最小类别的样本数，否则 StratifiedKFold 直接报错
    min_class_count = int(np.min(np.bincount(y))) if len(y) else 0
    folds = max(2, min(n_folds, min_class_count))
    if min_class_count < 2:
        # 有类别只有 1 个样本，分层交叉验证做不了——退回单次评估，
        # 噪声完全由二项标准误兜底，如实标注 n_folds=1
        return RobustMetric(mean=0.0, std=0.5, metric_name=metric_name,
                            n_val_samples=0, n_folds=1, noise_source="binomial")

    scores: List[float] = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    X_arr = np.asarray(X, dtype=object)
    for train_idx, val_idx in skf.split(X_arr, y):
        est = estimator_factory()
        est.fit(X_arr[train_idx].tolist(), y[train_idx])
        pred = est.predict(X_arr[val_idx].tolist())
        if metric_name == "accuracy":
            scores.append(float(accuracy_score(y[val_idx], pred)))
        else:
            scores.append(float(f1_score(y[val_idx], pred, average="macro", zero_division=0)))

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    avg_val_n = len(y) // folds
    binom = binomial_standard_error(mean, avg_val_n)
    return RobustMetric(
        mean=round(mean, 4), std=round(max(std, binom), 4), metric_name=metric_name,
        n_val_samples=avg_val_n, n_folds=folds, per_fold=[round(s, 4) for s in scores],
        noise_source="cv" if std >= binom else "cv+binomial",
    )
