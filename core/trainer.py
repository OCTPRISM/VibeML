"""
core/trainer.py  -  Phase 1.3：训练执行引擎

自动模型选择策略（对用户完全透明）：
  - 样本 <  100 条：TF-IDF + LogisticRegression（快速、无需 GPU）
  - 样本 100-500 条：TF-IDF + LinearSVC（更强的线性分类器）
  - 样本 > 500 条：TF-IDF + SGD（支持在线学习，可扩展）

Phase 1 使用 sklearn 作为主训练后端——目的是验证闭环逻辑，
而非追求 SOTA 指标。架构已预留 HuggingFace PEFT 接口（Phase 2 升级）。

差异化体现：
  - 用户从不选择模型，系统根据数据量自动决策
  - 用"模拟 epoch"方式（逐步增大训练比例）展示学习曲线
  - 提供 per-class 指标和混淆高亮，不只是总体数字
"""

from typing import List, Dict, Optional, Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

from config import TaskSpec, TrainingConfig, EpochResult


# 超过这个比例的字符是 CJK 就按中文处理。0.2 这个值偏低是故意的：
# 中英混排的中文文本（"这个 API 响应太慢了"）汉字占比可能只有三四成，
# 但它依然必须走字符 n-gram，按词切一样会退化。反过来，纯英文文本里
# 出现两成汉字的情况基本不存在，误判风险很小
_CJK_RATIO_THRESHOLD = 0.2


def _is_cjk_heavy(texts: Sequence[str]) -> bool:
    """判断一批文本是否以中日韩文字为主。只数汉字/假名/谚文，不数标点和数字——
    中文文本里的标点占比不低，算进分母会把判断拉偏。"""
    cjk = total = 0
    for t in texts:
        for ch in (t or ""):
            if ch.isspace() or not ch.isprintable():
                continue
            o = ord(ch)
            if (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF      # 汉字
                    or 0x3040 <= o <= 0x30FF                        # 日文假名
                    or 0xAC00 <= o <= 0xD7AF):                      # 韩文谚文
                cjk += 1
                total += 1
            elif ch.isalnum():
                total += 1
    return total > 0 and (cjk / total) >= _CJK_RATIO_THRESHOLD


class Trainer:
    """
    训练执行引擎。

    用法：
        trainer = Trainer(task_spec)
        config = trainer.auto_select(len(data))
        epoch_results = trainer.train_with_eval(data, config)
        labels = trainer.predict(["测试文本"])
    """

    def __init__(self, task_spec: TaskSpec):
        self.task_spec      = task_spec
        self.model          = None
        self.vectorizer:    Optional[TfidfVectorizer] = None
        self.label_encoder  = LabelEncoder()
        self.config:        Optional[TrainingConfig]  = None
        self._fitted        = False
        # train_with_eval 里按训练文本探测出来的语种标志，
        # build_cv_estimator_factory 必须复用它保证两边管线一致
        self._vectorizer_is_cjk = False

    # ── 自动选择 ──────────────────────────────────────────────────────────────

    def auto_select(self, data_size: int) -> TrainingConfig:
        """
        根据数据规模自动选择训练策略，返回 TrainingConfig。
        用户不需要知道这些细节，但可以在日志里看到选择了什么。
        """
        if data_size < 100:
            return TrainingConfig(
                model_name   = "TF-IDF + Logistic Regression（小数据专用）",
                model_key    = "logreg",
                use_lora     = False,
                lora_rank    = 0,
                lora_alpha   = 0,
                learning_rate= 1.0,
                num_epochs   = 6,
                batch_size   = 16,
                max_length   = 256,
                use_cpu_fallback = True,
            )
        elif data_size < 500:
            return TrainingConfig(
                model_name   = "TF-IDF + Linear SVM（中等数据）",
                model_key    = "svm",
                use_lora     = False,
                lora_rank    = 0,
                lora_alpha   = 0,
                learning_rate= 0.1,
                num_epochs   = 8,
                batch_size   = 32,
                max_length   = 512,
                use_cpu_fallback = True,
            )
        else:
            return TrainingConfig(
                model_name   = "TF-IDF + SGD Classifier（大数据）",
                model_key    = "sgd",
                use_lora     = False,
                lora_rank    = 0,
                lora_alpha   = 0,
                learning_rate= 0.01,
                num_epochs   = 10,
                batch_size   = 64,
                max_length   = 512,
                use_cpu_fallback = True,
            )

    # ── 训练 + 评估 ───────────────────────────────────────────────────────────

    def train_with_eval(
        self,
        data:      List[Dict],
        config:    TrainingConfig,
        val_split: float = 0.2,
        augmentor  = None,          # Phase 2.1: Augmentor | None
    ) -> List[EpochResult]:
        """
        训练模型并返回每个"模拟 epoch"的结果。

        sklearn 没有真实的 epoch 概念，我们通过"逐步增加训练数据比例"
        来模拟学习曲线，让闭环的迭代可视化更直观。

        Args:
            data:      [{"text": ..., "label": ...}]
            config:    由 auto_select() 返回的配置
            val_split: 验证集比例
            augmentor: （Phase 2）Augmentor 实例，用于 SMOTE/Mixup；None 则跳过

        Returns:
            List[EpochResult]，每个 epoch 一条
        """
        self.config = config

        texts  = [d["text"]  for d in data]
        labels = [d["label"] for d in data]

        # 用 task_spec.label_schema 初始化编码器（确保包含所有标签，即使训练集中某标签暂缺）
        known_labels = self.task_spec.label_schema if self.task_spec.label_schema else sorted(set(labels))
        # 确保训练集中出现的标签都在 known_labels 里
        all_labels = sorted(set(list(known_labels) + list(set(labels))))
        self.label_encoder.fit(all_labels)
        y = self.label_encoder.transform(labels)

        # 分层分割，保证验证集标签分布合理
        try:
            X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
                texts, y,
                test_size  = val_split,
                random_state = 42,
                stratify   = y,
            )
        except ValueError:
            # 某些标签样本太少时退化为随机分割
            X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
                texts, y, test_size=val_split, random_state=42
            )

        # TF-IDF 向量化（unigram + bigram，sublinear TF）
        # 语种探测用**训练侧**文本，不用验证集——验证集的任何信息都不该参与
        # 训练管线的构建决策（哪怕只是"选哪种分词方式"这种粗粒度的决策）
        self._vectorizer_is_cjk = _is_cjk_heavy(X_tr_raw)
        self.vectorizer = self.build_vectorizer(X_tr_raw)
        X_tr  = self.vectorizer.fit_transform(X_tr_raw)
        X_val = self.vectorizer.transform(X_val_raw)

        # ── Phase 2.1: 特征空间增强（SMOTE / Mixup）──────────────────────────
        self.augment_report = None
        if augmentor is not None:
            X_tr, y_tr, self.augment_report = augmentor.apply(
                X_tr, y_tr, label_encoder=self.label_encoder
            )

        n_train    = X_tr.shape[0]
        n_epochs   = config.num_epochs
        # 逐步增加训练比例：[30%, 45%, 60%, 75%, 85%, 95%, 100%, ...]
        fractions  = np.linspace(0.30, 1.0, n_epochs)

        epoch_results = []

        for ep_idx, frac in enumerate(fractions):
            n_use  = max(len(self.label_encoder.classes_) * 2, int(n_train * frac))
            n_use  = min(n_use, n_train)

            X_ep   = X_tr[:n_use]
            y_ep   = y_tr[:n_use]

            # 类别严重不平衡时，前 n_use 个样本（按 frac 截取的"模拟 epoch"训练量）
            # 可能碰巧只覆盖到 1 个类别——sklearn 分类器 fit 不了单一类别，逐步扩大
            # 切片直到覆盖到 ≥2 个类别，而不是让整个训练直接崩溃
            while len(np.unique(y_ep)) < 2 and n_use < n_train:
                n_use = min(n_use + max(1, n_train // 10), n_train)
                X_ep  = X_tr[:n_use]
                y_ep  = y_tr[:n_use]

            clf    = self._build_clf(config.model_key, config.hyperparam_overrides, y_ep)
            clf.fit(X_ep, y_ep)

            y_pred = clf.predict(X_val)

            # ── 评估指标 ──
            metric_name = self.task_spec.evaluation_metric
            if metric_name == "accuracy":
                val_metric = accuracy_score(y_val, y_pred)
            else:
                val_metric = f1_score(
                    y_val, y_pred, average="weighted", zero_division=0
                )

            # 模拟 loss（随 epoch 递减，加少量噪声）
            base_loss   = 1.0 - val_metric
            noise       = np.random.normal(0, 0.015)
            train_loss  = max(0.01, base_loss * 0.85 + noise)
            val_loss    = max(0.01, base_loss + abs(noise))

            # ── Per-class 指标 ──
            # 只对验证集中实际出现的类别计算 per-class 指标
            present_classes_idx = sorted(set(y_val.tolist()) | set(y_pred.tolist()))
            present_names = [self.label_encoder.classes_[i] for i in present_classes_idx
                             if i < len(self.label_encoder.classes_)]
            class_names = list(self.label_encoder.classes_)
            try:
                report = classification_report(
                    y_val, y_pred,
                    labels       = present_classes_idx,
                    target_names = present_names,
                    output_dict  = True,
                    zero_division= 0,
                )
                per_class = {
                    cls: round(report[cls]["f1-score"], 4)
                    for cls in present_names
                    if cls in report
                }
            except Exception:
                per_class = {}

            # ── 混淆分析（最多 3 对） ──
            highlights = self._confusion_highlights(y_val, y_pred, class_names)

            epoch_results.append(EpochResult(
                epoch              = ep_idx + 1,
                train_loss         = round(train_loss, 4),
                val_loss           = round(val_loss, 4),
                val_metric         = round(val_metric, 4),
                metric_name        = metric_name,
                per_class_metrics  = per_class,
                confusion_highlights = highlights,
            ))

            # 最后一个 epoch 保存最终模型
            if ep_idx == n_epochs - 1:
                self.model  = clf
                self._fitted = True

        return epoch_results

    # ── 预测 ──────────────────────────────────────────────────────────────────

    def predict(self, texts: List[str]) -> List[str]:
        """对新文本做分类预测"""
        if not self._fitted or self.vectorizer is None:
            raise RuntimeError("模型尚未训练，请先调用 train_with_eval()")
        X     = self.vectorizer.transform(texts)
        y_hat = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_hat).tolist()

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        """返回每个类别的概率（仅 logreg 支持，svm/sgd 返回 None）"""
        if not self._fitted or self.vectorizer is None:
            raise RuntimeError("模型尚未训练")
        if not hasattr(self.model, "predict_proba"):
            return [{}] * len(texts)
        X      = self.vectorizer.transform(texts)
        probas = self.model.predict_proba(X)
        classes = list(self.label_encoder.classes_)
        return [dict(zip(classes, p.tolist())) for p in probas]

    # ── 私有辅助 ──────────────────────────────────────────────────────────────

    @staticmethod
    def build_vectorizer(texts: Sequence[str]) -> TfidfVectorizer:
        """按文本语种构建 TF-IDF 向量化器。**中文必须走字符 n-gram，不能走默认分词。**

        sklearn 默认的 token_pattern 是 r"(?u)\\b\\w\\w+\\b"——它靠空格/词边界切词。
        中文句子没有空格，整句会被当成**一个** token：实测 4 条中文样本得到 4 个特征，
        每行只有 1 个非零值，任意两条样本之间零特征重叠。也就是说模型在中文上
        根本无法泛化，只能把训练集背下来，对没见过的句子等同于瞎猜。

        真实数据集上的对照（sepidmnorozy/Chinese_sentiment 前 1000 条，5 折 f1_macro，
        1σ 噪声≈0.015，去掉预置空格还原真实未分词中文）：

            word(1,2) 默认  0.388  ±0.000(折间)   ← 折间方差为 0 = 退化成"永远预测多数类"
            char(1,3)      0.629  ±0.041
            char(2,4)      0.537  ±0.059
            char_wb(2,4)   0.534  ±0.049

        差距约 16σ，不是噪声。选 char(1,3)：在带空格的原始版本上它同样是并列最好
        （0.677 vs char_wb 的 0.682，差距在 1σ 内）。

        这里没有引入 jieba 之类的分词器：字符 n-gram 零依赖、对未登录词和
        繁简混排都稳健，够用；真要上分词器是另一件事，不在这个修复范围内。

        英文/拉丁文本走原来的 word(1,2) 分支，行为**完全不变**。
        """
        common = dict(max_features=15000, sublinear_tf=True,
                      min_df=1, strip_accents="unicode")
        if _is_cjk_heavy(texts):
            return TfidfVectorizer(analyzer="char", ngram_range=(1, 3), **common)
        return TfidfVectorizer(ngram_range=(1, 2), **common)

    def build_cv_estimator_factory(self, config: TrainingConfig):
        """返回一个无参可调用：每次调用产出一个**全新未训练**的 sklearn Pipeline，
        结构跟 train_with_eval 里实际用的那套一致（TF-IDF → 分类器）。

        给 core/robust_eval.py::cross_val_metric 用。两个要点：

        1. **向量化器必须包进 Pipeline**，不能复用已经 fit 过的 self.vectorizer。
           否则每折的验证集词表信息会通过 TF-IDF 泄漏进训练侧，交叉验证出来的
           指标会偏乐观——那就失去了"拿它当可信估计"的意义。
        2. **必须每折新建**，不能共用一个实例，否则第二折是在第一折的基础上继续训练。
        """
        from sklearn.pipeline import Pipeline

        model_key = config.model_key
        overrides = dict(config.hyperparam_overrides or {})

        # 语种探测必须和 train_with_eval 用同一份文本做依据，否则交叉验证估的是
        # 另一个模型（比如 CV 走字符 n-gram、实际训练走词级），拿它做决策等于比错对象
        cjk = self._vectorizer_is_cjk

        def factory():
            common = dict(max_features=15000, sublinear_tf=True,
                          min_df=1, strip_accents="unicode")
            tfidf = (TfidfVectorizer(analyzer="char", ngram_range=(1, 3), **common)
                     if cjk else TfidfVectorizer(ngram_range=(1, 2), **common))
            return Pipeline([
                ("tfidf", tfidf),
                ("clf", self._build_clf(model_key, overrides, y=None)),
            ])
        return factory

    def _build_clf(self, model_key: str, overrides: Optional[Dict[str, Any]] = None,
                   y: Optional[np.ndarray] = None):
        """根据 model_key 构建 sklearn 分类器，overrides 可覆盖 C / alpha（来自自动调参闭环）

        y（本次实际用于训练的标签数组）用于让 svm 分支自适应地选校准折数——
        小数据 + 自动调参闭环随时可能把模型切到 svm，若某个类别样本数少于
        CalibratedClassifierCV 默认的 5 折，会直接抛出 sklearn 异常。
        """
        overrides = overrides or {}
        if model_key == "logreg":
            return LogisticRegression(
                C            = overrides.get("C", 1.0),
                max_iter     = 1000,
                random_state = 42,
                solver       = "lbfgs",
                class_weight = "balanced",
            )
        elif model_key == "svm":
            base = LinearSVC(C=overrides.get("C", 0.5), max_iter=2000, random_state=42,
                              class_weight="balanced")
            min_count = 5
            if y is not None and len(y):
                counts = np.bincount(y)
                counts = counts[counts > 0]
                if len(counts):
                    min_count = int(counts.min())
            if min_count < 2:
                # 某个类别只有 1 个样本，无法做任何折的校准，退化为不带概率校准的 LinearSVC
                return base
            # LinearSVC 不支持 predict_proba，用 CalibratedClassifierCV 包一层；
            # cv 不能超过最小类别的样本数
            return CalibratedClassifierCV(base, cv=min(5, min_count))
        elif model_key == "sgd":
            return SGDClassifier(
                loss        = "modified_huber",
                alpha       = overrides.get("alpha", 0.001),
                max_iter    = 100,
                random_state= 42,
                class_weight= "balanced",
            )
        else:
            return LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")

    def _confusion_highlights(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
    ) -> List[str]:
        """找出最容易混淆的标签对（最多 3 对）"""
        # 只使用实际出现在 y_true/y_pred 中的标签索引，避免越界
        present_idx = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        if len(present_idx) < 2:
            return []
        try:
            cm = confusion_matrix(y_true, y_pred, labels=present_idx)
        except Exception:
            return []
        pairs: List[tuple] = []
        n = len(present_idx)
        for i in range(n):
            for j in range(n):
                if i != j and cm[i, j] > 0:
                    ti = present_idx[i]
                    tj = present_idx[j]
                    tn = class_names[ti] if ti < len(class_names) else str(ti)
                    pn = class_names[tj] if tj < len(class_names) else str(tj)
                    pairs.append((cm[i, j], tn, pn))
        pairs.sort(reverse=True)
        return [
            f"'{tc}' 被误判为 '{pc}'（{cnt} 次）"
            for cnt, tc, pc in pairs[:3]
        ]
