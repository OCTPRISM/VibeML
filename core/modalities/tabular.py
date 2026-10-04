"""
core/modalities/tabular.py  -  Phase 4：表格数据模态

路线图 Phase 4：支持更多模态（图像、时序、多模态）。
本模块实现最高优先级：表格/CSV 数据分类。

差异化：
  现有 AutoML 工具（AutoGluon、Auto-sklearn）要求用户理解特征工程。
  本模块通过对话确认目标列，自动检测特征类型，选择预处理策略，
  完全无需用户了解 one-hot / normalization 等概念。

支持：
  - 二分类 / 多分类
  - 数值特征（自动标准化）+ 类别特征（自动 One-Hot）
  - 缺失值自动填充
  - 与现有 Explainer / IterationTree 完全兼容
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TabularDataReport:
    n_rows:           int
    n_features:       int
    num_features:     List[str]
    cat_features:     List[str]
    missing_pct:      Dict[str, float]   # 列名 → 缺失率
    class_distribution: Dict[str, int]
    warnings:         List[str]


class TabularTrainer:
    """
    表格数据训练器。与 core/trainer.py 接口对齐（predict / predict_proba）。

    用法：
        trainer = TabularTrainer(task_spec)
        report  = trainer.fit(df, target_col="label")
        preds   = trainer.predict(new_df)
    """

    def __init__(self, task_spec):
        self.task_spec     = task_spec
        self.pipeline      = None
        self.label_encoder = None
        self._fitted       = False
        self._feature_cols: List[str] = []

    # ── 主接口 ───────────────────────────────────────────────────────────────

    def fit(self, df, target_col: str) -> TabularDataReport:
        """
        训练模型。

        Args:
            df:         pandas DataFrame
            target_col: 目标列名（分类标签列）

        Returns:
            TabularDataReport
        """
        import pandas as pd
        from sklearn.compose         import ColumnTransformer
        from sklearn.impute          import SimpleImputer
        from sklearn.linear_model    import LogisticRegression
        from sklearn.pipeline        import Pipeline
        from sklearn.preprocessing   import LabelEncoder, OneHotEncoder, StandardScaler

        # ── 特征检测 ─────────────────────────────────────────────────────────
        feat_df   = df.drop(columns=[target_col])
        num_cols  = feat_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols  = feat_df.select_dtypes(include=["object", "category"]).columns.tolist()
        self._feature_cols = num_cols + cat_cols

        missing_pct = {
            col: round(float(df[col].isna().mean()), 3)
            for col in self._feature_cols
            if df[col].isna().any()
        }

        # ── 质量警告 ─────────────────────────────────────────────────────────
        warnings: List[str] = []
        for col, pct in missing_pct.items():
            if pct > 0.3:
                warnings.append(f"'{col}' 缺失率 {pct:.0%}，建议确认是否有效")
        if len(cat_cols) > 10:
            warnings.append(f"类别特征较多（{len(cat_cols)} 列），高基数列会显著增大模型体积")

        # ── 预处理流水线 ─────────────────────────────────────────────────────
        num_transformer = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
        ])
        cat_transformer = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        steps = []
        if num_cols: steps.append(("num", num_transformer, num_cols))
        if cat_cols: steps.append(("cat", cat_transformer, cat_cols))
        preprocessor = ColumnTransformer(steps, remainder="drop")

        # ── 标签编码 ─────────────────────────────────────────────────────────
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df[target_col].astype(str))

        n_classes  = len(self.label_encoder.classes_)
        data_size  = len(df)
        C          = 1.0 if data_size < 200 else 0.5

        # ── 选择分类器 ───────────────────────────────────────────────────────
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42)

        self.pipeline = Pipeline([
            ("prep", preprocessor),
            ("clf",  clf),
        ])
        self.pipeline.fit(feat_df, y)
        self._fitted = True

        # ── 报告 ─────────────────────────────────────────────────────────────
        from collections import Counter
        return TabularDataReport(
            n_rows             = len(df),
            n_features         = len(self._feature_cols),
            num_features       = num_cols,
            cat_features       = cat_cols,
            missing_pct        = missing_pct,
            class_distribution = dict(Counter(df[target_col].astype(str))),
            warnings           = warnings,
        )

    def predict(self, df) -> List[str]:
        """预测标签列表"""
        if not self._fitted:
            raise RuntimeError("请先调用 fit()")
        feat_df = df[self._feature_cols] if self._feature_cols else df
        y_enc   = self.pipeline.predict(feat_df)
        return self.label_encoder.inverse_transform(y_enc).tolist()

    def predict_proba(self, df) -> List[Dict[str, float]]:
        """返回每个样本的类别概率字典"""
        if not self._fitted:
            raise RuntimeError("请先调用 fit()")
        feat_df = df[self._feature_cols] if self._feature_cols else df
        probas  = self.pipeline.predict_proba(feat_df)
        classes = list(self.label_encoder.classes_)
        return [dict(zip(classes, p.tolist())) for p in probas]

    # ── 静态工具 ─────────────────────────────────────────────────────────────

    @staticmethod
    def from_csv(
        csv_path:   str,
        target_col: str,
        task_spec=None,
    ) -> Tuple["TabularTrainer", "TabularDataReport"]:
        """
        一行加载 CSV 并训练。

        用法：
            trainer, report = TabularTrainer.from_csv("data.csv", "label")
        """
        import pandas as pd
        from config import TaskSpec, TaskType

        df = pd.read_csv(csv_path)
        if target_col not in df.columns:
            raise ValueError(f"目标列 '{target_col}' 不在 CSV 中，可用列：{list(df.columns)}")

        if task_spec is None:
            labels = sorted(df[target_col].dropna().astype(str).unique().tolist())
            task_spec = TaskSpec(
                task_type=TaskType.CLASSIFICATION,
                domain="tabular",
                label_schema=labels,
                input_field="features",
                output_description=f"预测 {target_col}",
                evaluation_metric="f1",
                raw_description=f"表格分类任务，目标列：{target_col}",
            )

        trainer = TabularTrainer(task_spec)
        report  = trainer.fit(df, target_col)
        return trainer, report

    @staticmethod
    def detect_target_column(df) -> Optional[str]:
        """
        启发式检测目标列：
        优先选择名为 label/target/class/y 的列，
        其次选择低基数（≤20 个唯一值）的字符串列。
        """
        common_names = {"label", "target", "class", "y", "output", "category"}
        for col in df.columns:
            if col.lower() in common_names:
                return col
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].nunique() <= 20:
                return col
        return None
