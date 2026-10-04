"""
core/loss_factory.py  -  按"诊断出的错误模式"选损失函数

之前 core/nn_trainer.py 里损失函数是写死的两项白名单：
    {"cross_entropy": F.cross_entropy, "nll_loss": F.nll_loss}
而且只在生成架构那一刻由 LLM 选一次，之后再不回头看——训练完发现某个类别
被系统性误判，也没有任何机制去换一个更合适的损失函数。

这个模块把"错误模式 → 损失函数"变成一张可查的映射表，由 Explainer 的诊断驱动：

    类别不均衡            → weighted_ce（不是 focal，见下）
    少数类仍大量漏判      → focal + 类别权重
    标签噪声（且类别均衡）→ label_smoothing
    一般情况              → cross_entropy

## ⚠ 这张表被实测修正过：单独的 focal 治不了类别不平衡

最初按常见直觉写的是"严重不平衡 → focal"。实测推翻了它。

实验：中文情感数据（sepidmnorozy/Chinese_sentiment），训练集 600 条做成
9:1 不平衡（多数类 540 / 少数类 60），测试集 1500 条均衡，指标 macro-F1，
5 个种子取平均，1σ 噪声下界 ≈ 0.011。同一个 MLP 只换损失函数：

    损失函数                60 步     300 步    1000 步
    cross_entropy          0.3333    0.3812    0.3841
    focal γ=2              0.3333    0.3865    0.3881    ← 和 CE 没区别
    focal γ=2 + 类别权重    0.5123    0.4173    0.4066
    weighted_ce            0.5446    0.4349    0.4104    ← 最好
    label_smoothing        0.3333    0.3931    0.3686    ← 反而更差

（0.3333 = macro-F1 在二分类均衡测试集上恒定预测多数类的值，即模型完全塌掉）

两条结论：

1. **focal 单独用几乎没有效果**（1000 步时 +0.004，在噪声内）。原因不难理解：
   focal 是按"预测得有多准"来重加权（难样本权重高），**不是**按"这个类有多少
   样本"来重加权。类别不平衡的问题是后者，focal 治的是前者。要让 focal 对
   不平衡起作用，必须带上 alpha（类别权重）——加上之后确实有效（+0.023）。
2. **label_smoothing 在不平衡数据上是负收益**（-0.015，超过噪声）。它把目标
   分布抹平，在多数类本来就压倒少数类的情况下等于帮倒忙。所以它只在类别
   大致均衡、且确实怀疑标签噪声时才该用。

另外注意训练步数这一列：weighted_ce 在 60 步时领先 CE 达 **+0.21**，到 1000 步
只剩 +0.026——因为步数一多所有配置都开始过拟合这 600 条数据。所以"选对损失
函数"的收益大小强烈依赖于是否在合适的训练量上比较，报告单一数字会误导。

**刻意不做对比学习类损失（SupCon / Triplet）**，这不是遗漏：
  1. 本项目 NN 后端的输入是 **TF-IDF 稠密向量**（见 core/nn_trainer.py），
     不是端到端学出来的 embedding——表征本身是冻住的，对比损失只能作用在
     后面那个小分类头上，收益远低于它在真正 encoder 上的效果。
  2. SupCon 要求同一 batch 内每类 ≥2 个样本。本项目面向 10~200 条/类的低资源
     场景，batch_size=16，类别一多就有大量 batch 凑不出正样本对，损失会退化
     成 0 或 NaN。
  3. 它不是"换个损失函数"，而是换训练协议（投影头 + 两阶段：对比预训练 →
     线性探针），塞进当前单阶段循环里改动面远超一个函数。
  "易混淆类别"这一类错误模式，用 focal loss 调大 gamma 覆盖大部分收益。

所有实现都是纯 PyTorch 函数式写法，不引入新依赖。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# 错误模式 → 推荐损失函数。Explainer 在 prompt 里看到的就是这张表的语义，
# 这里是它给出的 loss_fn 值的合法取值域（收到表外的值一律退回 cross_entropy）
SUPPORTED_LOSSES = ("cross_entropy", "nll_loss", "focal", "weighted_ce", "label_smoothing")

# 参数的合法区间——LLM 给出的数值必须夹到这个范围内再用，
# 否则一个离谱的 gamma=50 会让梯度直接消失，训练静默失败
PARAM_RANGES = {
    "focal_gamma":      (0.5, 5.0),
    "label_smoothing":  (0.0, 0.4),
}


def clamp_param(name: str, value: Any, default: float) -> float:
    lo, hi = PARAM_RANGES.get(name, (float("-inf"), float("inf")))
    try:
        return float(min(max(float(value), lo), hi))
    except (TypeError, ValueError):
        return default


def compute_class_weights(y, num_classes: int):
    """按类别频率的倒数算权重（归一化到均值 1），用于 weighted_ce。
    样本数为 0 的类别权重给 1.0，避免除零和无穷大权重。"""
    import numpy as np
    import torch

    counts = np.bincount(np.asarray(y), minlength=num_classes).astype("float64")
    weights = np.ones(num_classes, dtype="float64")
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].sum() / (counts[nonzero] * nonzero.sum())
    return torch.tensor(weights, dtype=torch.float32)


def build_loss_fn(name: str, params: Optional[Dict[str, Any]] = None,
                  class_weights=None):
    """返回一个 (logits, target) -> scalar loss 的可调用对象。

    name 不在 SUPPORTED_LOSSES 里时静默退回 cross_entropy——损失函数选型是
    LLM 给的建议，收到没见过的名字属于正常情况，不该让训练直接崩掉。
    """
    import torch
    import torch.nn.functional as F

    params = params or {}

    if name == "focal":
        # Focal Loss（Lin et al. 2017）：把权重压向难分样本。
        # 类别不均衡时比加权 CE 更进一步——不只按类别频率加权，
        # 而是按"这条样本当前有多难"动态加权
        gamma = clamp_param("focal_gamma", params.get("focal_gamma", 2.0), 2.0)

        def focal(logits, target):
            logp = F.log_softmax(logits, dim=-1)
            logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
            pt = logpt.exp()
            loss = -((1.0 - pt) ** gamma) * logpt
            if class_weights is not None:
                loss = loss * class_weights.to(logits.device)[target]
            return loss.mean()
        return focal

    if name == "weighted_ce":
        w = class_weights

        def weighted(logits, target):
            return F.cross_entropy(
                logits, target,
                weight=w.to(logits.device) if w is not None else None)
        return weighted

    if name == "label_smoothing":
        # 标签平滑：缓解"模型对错标样本也给出 100% 置信度"这种过拟合，
        # 正是标签噪声场景（LLM 增强出来的数据难免有脏样本）需要的
        eps = clamp_param("label_smoothing", params.get("label_smoothing", 0.1), 0.1)

        def smoothed(logits, target):
            return F.cross_entropy(logits, target, label_smoothing=eps)
        return smoothed

    if name == "nll_loss":
        return F.nll_loss

    return F.cross_entropy


def build_sample_weights(y, label_names: List[str], class_boost: Dict[str, float]):
    """把 Explainer 给的 {标签名: 倍数} 翻译成逐样本权重，供
    torch.utils.data.WeightedRandomSampler 使用。

    这是"取消 sklearn 后端独享 class_boost"的落地点：sklearn 那边靠 SMOTE
    过采样实现类别加权，PyTorch 这边靠带权重的采样器实现同一个语义，
    两条路径行为对齐，Explainer 不需要知道当前是哪个后端。
    """
    import numpy as np

    y = np.asarray(y)
    weights = np.ones(len(y), dtype="float64")
    if not class_boost:
        return weights
    name_to_idx = {name: i for i, name in enumerate(label_names)}
    for label_name, factor in class_boost.items():
        idx = name_to_idx.get(label_name)
        if idx is None:
            continue
        try:
            f = float(factor)
        except (TypeError, ValueError):
            continue
        # 倍数夹在合理范围：过大的倍数会让采样几乎只剩这一个类别，
        # 反而把别的类别学没了
        f = min(max(f, 0.5), 3.0)
        weights[y == idx] = f
    return weights
