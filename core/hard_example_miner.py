"""
core/hard_example_miner.py — 难例检索（hard-example mining）

## 这个模块解决什么问题

`core/pipeline.py` 里的 `COLLECT_MORE_DATA` 动作原本做的是：
"再让 LLM 生成 60 条，全部追加进训练集"。这有两个真实缺陷：

1. **生成是无方向的**——模型到底在哪儿出错，生成过程完全不知道。
   补进来的样本大概率还是模型本来就答对的那一类，训练集变大了，
   但新增的信息量约等于零。
2. **生成是无去重的**——同一个 prompt 反复生成，LLM 会产出大量和已有
   数据近义的句子。这些样本不带新信息，只会让训练变慢；更糟的是重复
   样本相当于给某些区域悄悄加了权，可能反而让指标变差。

本模块用两步解决：

- `find_hard_examples()`：用**样本外**（out-of-fold）预测找出模型真正
  做不对的样本——误分类的，以及虽然分对了但 margin 很低（top1 与 top2
  概率差很小）的边界样本。
- `select_candidates()`：把候选样本嵌入成向量，先剔掉与**已有训练数据**
  过于接近的近义重复，再按"离难例区域有多近"排序取前 k 条。

## ⚠ 实测结论：上面那两条"缺陷"里，只有第 1 条的诊断价值站得住

写完之后做了真实测量，结果**不支持**本模块最初的两个卖点，如实记在这里，
免得后面有人再花一遍时间重新发现：

**(a) 定向排序没能跑赢随机选。** A/B：Chinese_sentiment 数据集，起始 120 条
训练样本 + 从 600 条候选池里选 120 条补进去，固定 1500 条测试集，5 个种子
配对比较 macro-F1：

    难例检索选 - 随机选 = +0.008
    测试集 1σ 噪声下界   =  0.011
    各种子差值 -0.024 / +0.033 / +0.030 / +0.012 / -0.010  ← 反复变号

差值低于噪声下界且符号不稳定 → 在这个规模上分不出高下。补数据本身是有用的
（0.63 → 0.72，远超噪声），但**选哪些**补，这套排序没有体现出价值。

**(b) LLM 生成的候选里根本没有近义重复。** 真实 Ollama（qwen3.6:35b-a3b）
按现有 prompt 增强出 31 条：与种子数据余弦相似度 ≥0.92 的 0 条（最大 0.750），
新样本互相 ≥0.92 的 0 条（最大 0.698）。也就是说"LLM 会反复生成近义句"
这个前提，至少在这个模型 + 这个 prompt 下不成立。

**因此 core/pipeline.py 只用了本模块中被测量支持的部分**：
`find_hard_examples()` 当诊断用（out-of-fold 预测 + 真实 margin，这部分独立
成立），`select_candidates()` 只当零成本的去重保险（没重复时不删任何东西），
**不做**超量生成 + 定向筛选——那要多付约 2.5 倍生成 token，换不到可测的收益。

排序逻辑本身留着没删：它在"候选池大得多、且模型的弱点确实集中在某个局部"
的场景下仍可能成立（上面的 A/B 只证伪了当前这个规模和这个数据集）。
要重新启用，请先把 A/B 重跑一遍拿到超过噪声的证据，别凭直觉打开。

## 必须说清楚的边界（不要在 UI 上过度承诺）

这里的"检索"，检索的是**调用方传进来的候选池**，本模块自己不生产候选。
本系统当前唯一始终可用的候选池是 LLM 现场生成的样本，所以实际效果是
"给 LLM 生成的候选做定向筛选"，**不是**"从一个大规模真实语料库里捞难例"。
如果调用方能提供更大的真实候选池（比如 HF 数据集里没被采样进训练集的
剩余行），同一套代码会直接受益——但那取决于池子里到底有没有难例，
代码保证不了这一点。

嵌入模型只影响"哪些句子算相近"这个判断。中文场景必须用多语言模型，
用纯英文模型（比如本机缓存里的 all-MiniLM-L6-v2）去编码中文时，
相似度基本是噪声——所以下面的降级链在退到英文模型时会**如实记进
report.notes**，而不是假装检索照常工作。

## 为什么这里没有用 faiss（本来打算用的）

faiss 和 PyTorch（sentence-transformers 依赖它）各自打包了一份 OpenMP 运行时，
在 macOS 上同一个进程里先加载 torch 再加载 faiss 会直接把进程打死：

    OMP: Error #179: Function pthread_mutex_init failed
    OMP: System error #22: Invalid argument

实测"先 import faiss 再 import torch"能绕过去，但把一个**会杀进程**的崩溃
寄托在全局 import 顺序上太脆——训练管线里任何一处不相关的 import 都可能
把顺序翻过来，而且是静默 crash，没有 traceback。

更关键的是：faiss 在这个量级上什么也没多给。候选池 ~150 条、已有数据几百条、
难例 ≤50 条，整个相似度矩阵不过几万个浮点数，一次 numpy 矩阵乘法就算完了。
faiss 的 IndexFlatIP 本身就是暴力精确检索，和矩阵乘法**结果完全相同**，
没有任何精度差异——它的价值在百万级向量上的索引结构，这里用不到。
所以直接用 numpy，少一个依赖、少一类崩溃。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 首选多语言模型（中英文都能用）；取不到时退到本机通常已缓存的英文模型，
# 但会在报告里标注"这次的相似度判断对中文不可信"
PRIMARY_EMBED_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FALLBACK_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 余弦相似度超过这个值就算"和已有样本近义重复"。0.92 不是拍脑袋：
# 实测同义改写（"这个客服态度很差"/"服务态度太糟糕了"）大约落在 0.75-0.88，
# 而只改了标点/语气词的真重复通常在 0.95 以上，0.92 卡在两者之间
DUPLICATE_THRESHOLD = 0.92

# 单条难例最多"吸引"多少条候选——防止所有配额被一条离群难例吃光，
# 那样补进来的全是同一个方向的样本，多样性反而更差
MAX_PER_HARD_EXAMPLE = 3


@dataclass
class HardExample:
    """一条模型没做好的样本。is_error=True 表示直接分错了；
    is_error=False 但 margin 很小，表示分对了但几乎是蒙对的。"""
    text:       str
    true_label: str
    pred_label: str
    margin:     float          # top1 概率 - top2 概率，越小越靠近决策边界
    is_error:   bool


@dataclass
class MiningReport:
    n_hard:              int = 0
    n_candidates:        int = 0
    n_dropped_duplicate: int = 0
    n_selected:          int = 0
    embedding_model:     str = ""
    # 非空表示这次没能真正跑成检索（或跑了但结果不可信），调用方应据此决定
    # 是回退到原来的"全量追加"还是干脆不追加。绝不静默假装成功
    degraded_reason:     str = ""
    notes:               List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.degraded_reason:
            return f"难例检索未生效（{self.degraded_reason}）"
        return (f"难例 {self.n_hard} 条；候选 {self.n_candidates} 条 → "
                f"去重剔除 {self.n_dropped_duplicate} 条 → 选中 {self.n_selected} 条")


# 嵌入模型加载一次要几秒，迭代循环里每轮都重载是纯浪费；用模块级单例缓存。
# 加锁是因为 pipeline 里有 ThreadPoolExecutor，并发首次加载会重复下载
_MODEL_CACHE: Dict[str, object] = {}
_MODEL_LOCK = threading.Lock()


def _load_embedder() -> Tuple[Optional[object], str, str]:
    """返回 (模型, 模型名, 警告)。三样都拿不到时第一个是 None。"""
    with _MODEL_LOCK:
        for name in (PRIMARY_EMBED_MODEL, FALLBACK_EMBED_MODEL):
            if name in _MODEL_CACHE:
                cached = _MODEL_CACHE[name]
                warn = ("" if name == PRIMARY_EMBED_MODEL else
                        "用的是纯英文嵌入模型，中文文本的相似度判断不可信")
                return cached, name, warn
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            return None, "", f"sentence-transformers 不可用：{e}"

        for name in (PRIMARY_EMBED_MODEL, FALLBACK_EMBED_MODEL):
            try:
                model = SentenceTransformer(name)
                _MODEL_CACHE[name] = model
                warn = ("" if name == PRIMARY_EMBED_MODEL else
                        "首选多语言模型不可用，退到纯英文模型，中文文本的相似度判断不可信")
                return model, name, warn
            except Exception:
                continue
        return None, "", "嵌入模型加载失败（离线且本机无缓存）"


def _embed(model, texts: Sequence[str]) -> np.ndarray:
    """编码成 L2 归一化的 float32 向量——归一化之后两个向量的内积
    直接就是余弦相似度，后面做矩阵乘法时不用再单独算范数"""
    vecs = model.encode(list(texts), convert_to_numpy=True,
                        show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


def find_hard_examples(
    data: List[Dict],
    trainer,
    max_hard: int = 50,
    margin_threshold: float = 0.25,
) -> Tuple[List[HardExample], str]:
    """用样本外预测找出模型的错误区域。返回 (难例列表, 失败原因)。

    ⚠ 必须用 out-of-fold 预测，不能用 trainer.model 直接打分——理由和
    `augmentor.flywheel` 里那段完全一样：打分模型就是在这批数据上训出来的，
    它已经把这些样本背下来了，in-sample 打分会告诉你"没有难例"。
    """
    if not hasattr(getattr(trainer, "model", None), "predict_proba") \
            or getattr(trainer, "vectorizer", None) is None:
        return [], "当前后端不支持概率输出（难例定位依赖 predict_proba）"

    texts = [d.get("text", "") for d in data]
    try:
        X = trainer.vectorizer.transform(texts)
        y_true = trainer.label_encoder.transform([d.get("label", "") for d in data])
        min_class = int(np.min(np.bincount(y_true))) if len(y_true) else 0
        if min_class < 2:
            return [], "有类别的样本少于 2 条，无法做样本外预测"
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_predict
        folds = max(2, min(5, min_class))
        probas = cross_val_predict(clone(trainer.model), X, y_true,
                                   cv=folds, method="predict_proba")
    except Exception as e:
        return [], f"样本外预测失败：{e}"

    hard: List[HardExample] = []
    for sample, row in zip(data, probas):
        order = np.argsort(row)[::-1]
        top1, top2 = float(row[order[0]]), float(row[order[1]]) if len(order) > 1 else 0.0
        margin = top1 - top2
        pred_label = trainer.label_encoder.inverse_transform([int(order[0])])[0]
        true_label = sample.get("label", "")
        is_error = pred_label != true_label
        # 分错的一定是难例；分对但 margin 很小的也算——那是决策边界上的样本，
        # 正是补数据最该瞄准的地方（主动学习的基本洞察）
        if is_error or margin < margin_threshold:
            hard.append(HardExample(
                text=sample.get("text", ""), true_label=true_label,
                pred_label=pred_label, margin=round(margin, 4), is_error=is_error))

    # 分错的排前面，同类里 margin 越小越靠前
    hard.sort(key=lambda h: (not h.is_error, h.margin))
    return hard[:max_hard], ""


def select_candidates(
    candidates: List[Dict],
    existing: List[Dict],
    hard_examples: List[HardExample],
    k: int,
    duplicate_threshold: float = DUPLICATE_THRESHOLD,
) -> Tuple[List[Dict], MiningReport]:
    """从 candidates 里挑出 k 条最值得加进训练集的。

    两步：先剔掉和 existing 近义重复的，再按"离难例有多近"排序。
    hard_examples 为空时退化成"只做去重"——这仍然是有价值的（去掉纯重复
    的生成结果），报告里会写明这一点，不假装做了定向检索。
    """
    report = MiningReport(n_hard=len(hard_examples), n_candidates=len(candidates))
    if not candidates:
        report.degraded_reason = "没有候选样本"
        return [], report
    if k <= 0:
        report.degraded_reason = "配额为 0"
        return [], report

    model, model_name, warn = _load_embedder()
    report.embedding_model = model_name
    if warn:
        report.notes.append(warn)
    if model is None:
        report.degraded_reason = warn or "嵌入模型不可用"
        return [], report

    cand_texts = [c.get("text", "") for c in candidates]
    try:
        cand_vecs = _embed(model, cand_texts)
    except Exception as e:
        report.degraded_reason = f"候选编码失败：{e}"
        return [], report

    # ── 第一步：剔掉与已有训练数据近义重复的候选 ──────────────────────
    keep_mask = np.ones(len(candidates), dtype=bool)
    existing_texts = [d.get("text", "") for d in existing if d.get("text")]
    if existing_texts:
        try:
            exist_vecs = _embed(model, existing_texts)
            # 向量都已 L2 归一化 → 内积就是余弦相似度。
            # (n_cand, dim) @ (dim, n_exist) = (n_cand, n_exist)，取每行最大值
            # 就是"这条候选与最像它的已有样本有多像"
            sims = cand_vecs @ exist_vecs.T
            keep_mask = sims.max(axis=1) < duplicate_threshold
        except Exception as e:
            # 去重失败不该让整个流程停摆，但要如实记下来
            report.notes.append(f"去重步骤跳过：{e}")

    report.n_dropped_duplicate = int((~keep_mask).sum())
    kept_idx = np.flatnonzero(keep_mask)
    if kept_idx.size == 0:
        report.degraded_reason = "候选全部与已有数据近义重复，没有可用的新样本"
        return [], report

    # ── 第二步：按"离难例区域有多近"排序 ──────────────────────────────
    if not hard_examples:
        report.notes.append("没有定位到难例，本次只做了去重，未做定向排序")
        chosen = kept_idx[:k]
    else:
        try:
            hard_vecs = _embed(model, [h.text for h in hard_examples])
            # 每条难例各自去找最接近它的候选，而不是"所有候选对全体难例求平均
            # 相似度"——后者会偏向那些"跟谁都不太像但也不太不像"的平庸样本，
            # 恰恰把真正贴着某个具体错误的样本排到后面
            n_probe = min(MAX_PER_HARD_EXAMPLE, kept_idx.size)
            sim_mat = hard_vecs @ cand_vecs[kept_idx].T      # (n_hard, n_kept)
            # 每行取前 n_probe 个，按相似度从高到低
            idxs = np.argsort(-sim_mat, axis=1)[:, :n_probe]
            sims = np.take_along_axis(sim_mat, idxs, axis=1)
            ranked: List[Tuple[float, int]] = []
            seen = set()
            # 按列展开：先把每条难例的第 1 名收完，再收第 2 名……
            # 保证配额均匀摊到不同难例上，不被单条难例吃光
            for col in range(n_probe):
                for row in range(len(hard_examples)):
                    local = int(idxs[row, col])
                    if local < 0:
                        continue
                    gid = int(kept_idx[local])
                    if gid in seen:
                        continue
                    seen.add(gid)
                    ranked.append((float(sims[row, col]), gid))
            chosen = [gid for _, gid in ranked][:k]
            # 名额没排满就用剩下的候选补齐（难例少的时候会出现）
            if len(chosen) < k:
                for gid in kept_idx:
                    if int(gid) not in seen:
                        chosen.append(int(gid))
                        if len(chosen) >= k:
                            break
        except Exception as e:
            report.notes.append(f"定向排序失败，退化为只去重：{e}")
            chosen = kept_idx[:k]

    selected = [candidates[int(i)] for i in chosen]
    report.n_selected = len(selected)
    return selected, report
