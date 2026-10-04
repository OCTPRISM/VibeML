"""
eval/datasets.py  -  Phase 3：基准数据集加载器

为论文实验提供标准化的小数据集接口。
所有数据集均可离线获取，无需外部 API。

支持的数据集：
  - 20newsgroups_4cls   20 Newsgroups（4 类）
  - 20newsgroups_2cls   20 Newsgroups（二分类）
  - customer_tickets    项目内置客服工单（5 类）

小数据实验规格（论文 Table 1 行）：
  每类样本数：10 / 20 / 50 / 100 / 200
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


# 论文实验用的每类样本数档位
SAMPLE_SIZES = [10, 20, 50, 100, 200]


@dataclass
class BenchmarkDataset:
    name:        str
    description: str
    train_texts: List[str]
    train_labels: List[str]
    test_texts:  List[str]
    test_labels: List[str]
    label_names: List[str]

    @property
    def n_classes(self) -> int:
        return len(self.label_names)

    @property
    def n_train(self) -> int:
        return len(self.train_texts)

    @property
    def n_test(self) -> int:
        return len(self.test_texts)

    def subsample(self, n_per_class: int, seed: int = 42) -> "BenchmarkDataset":
        """
        按每类 n_per_class 条对训练集做分层子采样。
        测试集保持完整不变（保证评估公平性）。
        """
        rng  = np.random.default_rng(seed)
        sel_texts, sel_labels = [], []

        for label in self.label_names:
            idx = [i for i, l in enumerate(self.train_labels) if l == label]
            n   = min(n_per_class, len(idx))
            if n == 0:
                continue
            chosen = rng.choice(idx, size=n, replace=False).tolist()
            sel_texts.extend([self.train_texts[i] for i in chosen])
            sel_labels.extend([self.train_labels[i] for i in chosen])

        return BenchmarkDataset(
            name         = f"{self.name}_n{n_per_class}",
            description  = f"{self.description} (每类 {n_per_class} 条)",
            train_texts  = sel_texts,
            train_labels = sel_labels,
            test_texts   = self.test_texts,
            test_labels  = self.test_labels,
            label_names  = self.label_names,
        )


# ── 数据集加载函数 ────────────────────────────────────────────────────────────

def load_20newsgroups(n_classes: int = 4) -> BenchmarkDataset:
    """
    加载 20 Newsgroups 数据集（去除 headers/footers/quotes）。

    Args:
        n_classes: 使用前 N 个类别（2 或 4）

    Returns:
        BenchmarkDataset
    """
    from sklearn.datasets import fetch_20newsgroups

    CATS_4 = [
        "alt.atheism",
        "comp.graphics",
        "sci.med",
        "soc.religion.christian",
    ]
    cats = CATS_4[:n_classes]
    remove = ("headers", "footers", "quotes")

    train = fetch_20newsgroups(subset="train", categories=cats, remove=remove)
    test  = fetch_20newsgroups(subset="test",  categories=cats, remove=remove)

    label_names = [c.split(".")[-1] for c in cats]   # 简化标签名
    train_labels = [label_names[y] for y in train.target]
    test_labels  = [label_names[y] for y in test.target]

    return BenchmarkDataset(
        name         = f"20newsgroups_{n_classes}cls",
        description  = f"20 Newsgroups ({n_classes} 类)",
        train_texts  = train.data,
        train_labels = train_labels,
        test_texts   = test.data,
        test_labels  = test_labels,
        label_names  = label_names,
    )


def load_customer_tickets(jsonl_path: str | None = None) -> BenchmarkDataset:
    """
    加载项目内置客服工单数据集（5 类，共 30 条）。
    由于样本量小，训练/测试各用一半。
    """
    if jsonl_path is None:
        jsonl_path = Path(__file__).parent.parent / "examples" / "data" / "customer_tickets.jsonl"

    examples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    rng = np.random.default_rng(42)
    rng.shuffle(examples)

    split    = len(examples) // 2
    train    = examples[:split]
    test     = examples[split:]
    labels   = sorted(set(e["label"] for e in examples))

    return BenchmarkDataset(
        name         = "customer_tickets",
        description  = "客服工单分类（5 类，30 条）",
        train_texts  = [e["text"] for e in train],
        train_labels = [e["label"] for e in train],
        test_texts   = [e["text"] for e in test],
        test_labels  = [e["label"] for e in test],
        label_names  = labels,
    )


def load_chinese_sentiment(max_test: int = 500) -> BenchmarkDataset:
    """
    加载真实 HuggingFace 中文情感数据集 sepidmnorozy/Chinese_sentiment（二分类，
    正式 train/test 切分，非本项目自造）——覆盖系统本身的双语（中/英）定位，
    20 Newsgroups/customer_tickets 都只测了英文/内置小样本，这个数据集补上
    "真实第三方中文数据集"这一环。

    测试集用固定 seed 子采样到 max_test 条，控制评估阶段的向量化/预测耗时
    （原始测试集 4896 条，对每个 (system, size, repeat) 组合都跑一遍没有必要）。
    """
    from datasets import load_dataset

    train = load_dataset("sepidmnorozy/Chinese_sentiment", split="train")
    test  = load_dataset("sepidmnorozy/Chinese_sentiment", split="test")

    label_map = {0: "负面", 1: "正面"}
    train_texts  = list(train["text"])
    train_labels = [label_map[l] for l in train["label"]]
    test_texts   = list(test["text"])
    test_labels  = [label_map[l] for l in test["label"]]

    if len(test_texts) > max_test:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(test_texts), size=max_test, replace=False)
        test_texts  = [test_texts[i]  for i in idx]
        test_labels = [test_labels[i] for i in idx]

    return BenchmarkDataset(
        name         = "chinese_sentiment",
        description  = "真实 HF 数据集 sepidmnorozy/Chinese_sentiment（中文情感二分类）",
        train_texts  = train_texts,
        train_labels = train_labels,
        test_texts   = test_texts,
        test_labels  = test_labels,
        label_names  = sorted(set(label_map.values())),
    )


# ── 一次性加载全部基准数据集 ─────────────────────────────────────────────────

def load_all() -> Dict[str, BenchmarkDataset]:
    """
    加载所有可用的基准数据集。
    如果某个数据集加载失败（如网络问题），跳过并警告。
    """
    datasets = {}

    loaders = {
        "20newsgroups_4cls": lambda: load_20newsgroups(4),
        "20newsgroups_2cls": lambda: load_20newsgroups(2),
        "customer_tickets":  lambda: load_customer_tickets(),
        "chinese_sentiment": lambda: load_chinese_sentiment(),
    }

    for name, loader in loaders.items():
        try:
            datasets[name] = loader()
            print(f"  ✅ {name}: train={datasets[name].n_train}, test={datasets[name].n_test}")
        except Exception as e:
            print(f"  ⚠️  {name} 加载失败（已跳过）：{e}")

    return datasets
