"""
eval/baselines.py  -  Phase 3：基准对比系统

论文 Table 1 的对比基线，均无需 API Key，可完全离线运行。

基线系统：
  LogReg        TF-IDF + Logistic Regression（经典基线）
  SVM           TF-IDF + Linear SVM（强基线）
  AutoGluon     AutoGluon TabularPredictor（如已安装）
  AutoSklearn   auto-sklearn（如已安装）

我们的系统（Ours）：
  Ours-Base     仅 Trainer（无增强）
  Ours-SMOTE    Trainer + SMOTE 过采样
  Ours-Full     完整 Pipeline（含 LLM 增强，需 API Key）
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@dataclass
class EvalResult:
    system_name:   str
    dataset_name:  str
    n_per_class:   int
    f1_weighted:   float
    accuracy:      float
    train_time_s:  float
    n_train:       int
    n_test:        int
    extra:         dict = field(default_factory=dict)   # 额外指标


# ── 抽象基类 ──────────────────────────────────────────────────────────────────

class BaseSystem(ABC):
    """所有对比系统的统一接口"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fit_predict(
        self,
        train_texts:  List[str],
        train_labels: List[str],
        test_texts:   List[str],
        test_labels:  List[str],
    ) -> EvalResult: ...


# ── sklearn 经典基线 ──────────────────────────────────────────────────────────

class SklearnBaseline(BaseSystem):
    """
    通用 sklearn Pipeline 基线。
    TF-IDF 向量化 + 指定分类器。
    """

    def __init__(self, clf_name: str, clf):
        self._name = f"TF-IDF + {clf_name}"
        self._clf  = clf

    @property
    def name(self) -> str:
        return self._name

    def fit_predict(
        self,
        train_texts:  List[str],
        train_labels: List[str],
        test_texts:   List[str],
        test_labels:  List[str],
    ) -> EvalResult:

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features = 15000,
                ngram_range  = (1, 2),
                sublinear_tf = True,
                min_df       = 1,
            )),
            ("clf", self._clf),
        ])

        t0 = time.time()
        pipeline.fit(train_texts, train_labels)
        train_time = time.time() - t0

        y_pred = pipeline.predict(test_texts)
        f1  = f1_score(test_labels, y_pred, average="weighted", zero_division=0)
        acc = accuracy_score(test_labels, y_pred)

        return EvalResult(
            system_name  = self.name,
            dataset_name = "",          # 由 runner 填充
            n_per_class  = 0,
            f1_weighted  = round(float(f1),  4),
            accuracy     = round(float(acc), 4),
            train_time_s = round(train_time, 3),
            n_train      = len(train_texts),
            n_test       = len(test_texts),
        )


# ── 我们的系统（Ours-Base & Ours-SMOTE）─────────────────────────────────────

class OursBaseSystem(BaseSystem):
    """
    我们的训练引擎（不含 LLM 组件）。
    使用 Phase 1.3 的 Trainer + 自动模型选择。
    无需 API Key，适合快速基准测试。
    """

    def __init__(self, use_smote: bool = False):
        self._use_smote = use_smote

    @property
    def name(self) -> str:
        return "Ours-SMOTE" if self._use_smote else "Ours-Base"

    def fit_predict(
        self,
        train_texts:  List[str],
        train_labels: List[str],
        test_texts:   List[str],
        test_labels:  List[str],
    ) -> EvalResult:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from config import TaskSpec, TaskType
        from core.trainer  import Trainer
        from core.augmentor import Augmentor

        # 构造最小 TaskSpec（不需要 LLM 解析）
        label_names = sorted(set(train_labels))
        task_spec   = TaskSpec(
            task_type          = TaskType.CLASSIFICATION,
            domain             = "benchmark",
            label_schema       = label_names,
            input_field        = "text",
            output_description = "label",
            evaluation_metric  = "f1",
            raw_description    = "benchmark evaluation",
        )

        train_data = [
            {"text": t, "label": l}
            for t, l in zip(train_texts, train_labels)
        ]

        augmentor = Augmentor(strategy="smote") if self._use_smote else None

        t0      = time.time()
        trainer = Trainer(task_spec)
        config  = trainer.auto_select(len(train_data))
        trainer.train_with_eval(train_data, config, augmentor=augmentor)
        train_time = time.time() - t0

        y_pred = trainer.predict(test_texts)
        f1  = f1_score(test_labels, y_pred, average="weighted", zero_division=0)
        acc = accuracy_score(test_labels, y_pred)

        extra = {}
        if self._use_smote and trainer.augment_report:
            r = trainer.augment_report
            extra["smote_balance_before"] = r.balance_ratio_before
            extra["smote_balance_after"]  = r.balance_ratio_after
            extra["smote_added"]          = r.augmented_train_count - r.original_train_count

        return EvalResult(
            system_name  = self.name,
            dataset_name = "",
            n_per_class  = 0,
            f1_weighted  = round(float(f1),  4),
            accuracy     = round(float(acc), 4),
            train_time_s = round(train_time, 3),
            n_train      = len(train_data),
            n_test       = len(test_texts),
            extra        = extra,
        )


# ── AutoGluon（可选，需另行安装）─────────────────────────────────────────────

class AutoGluonSystem(BaseSystem):
    """
    AutoGluon TextPredictor 基线（需 pip install autogluon.text）。
    如未安装则自动退出（runner 会跳过）。
    """

    @property
    def name(self) -> str:
        return "AutoGluon"

    def fit_predict(
        self,
        train_texts:  List[str],
        train_labels: List[str],
        test_texts:   List[str],
        test_labels:  List[str],
    ) -> EvalResult:
        try:
            import pandas as pd
            from autogluon.text import TextPredictor
        except ImportError:
            raise ImportError("AutoGluon 未安装：pip install autogluon.text")

        train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
        test_df  = pd.DataFrame({"text": test_texts,  "label": test_labels})

        t0        = time.time()
        predictor = TextPredictor(label="label", verbosity=0)
        predictor.fit(train_df, time_limit=120)
        train_time = time.time() - t0

        y_pred = predictor.predict(test_df)
        f1  = f1_score(test_labels, y_pred, average="weighted", zero_division=0)
        acc = accuracy_score(test_labels, y_pred)

        return EvalResult(
            system_name  = self.name,
            dataset_name = "",
            n_per_class  = 0,
            f1_weighted  = round(float(f1),  4),
            accuracy     = round(float(acc), 4),
            train_time_s = round(train_time, 3),
            n_train      = len(train_texts),
            n_test       = len(test_texts),
        )


# ── 系统注册表 ────────────────────────────────────────────────────────────────

def get_all_systems(include_autogluon: bool = False) -> List[BaseSystem]:
    """返回所有可用的评估系统列表"""
    systems = [
        SklearnBaseline("LogReg", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        SklearnBaseline("SVM",    LinearSVC(C=0.5, max_iter=2000, random_state=42)),
        OursBaseSystem(use_smote=False),
        OursBaseSystem(use_smote=True),
    ]
    if include_autogluon:
        systems.append(AutoGluonSystem())
    return systems
