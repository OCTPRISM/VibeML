"""
core/deployer.py  -  Phase 2.3：部署导出器

路线图 Milestone 2 验收点：
  "用户上传 100 条样本，系统完整走完 数据诊断→训练→迭代→导出 ONNX 部署包"

本模块做三件事：
  1. 导出模型（ONNX 优先，joblib 备用）
  2. 生成推理脚本（用户可直接在生产环境使用）
  3. 打包完整部署包（模型 + 脚本 + README）

ONNX 导出链：
  sklearn Pipeline(TF-IDF → LogReg/SVM) → skl2onnx → .onnx
  如果 skl2onnx 未安装，自动降级为 joblib 导出（.pkl）

设计原则：
  - 部署包必须独立可运行，不依赖 automl_agent 项目代码
  - 推理脚本包含完整的预处理逻辑，边缘设备可以直接使用
"""

from __future__ import annotations

import os
import json
import shutil
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any

from config import DeploymentPackage, TaskSpec

if TYPE_CHECKING:
    from core.trainer import Trainer


class Deployer:
    """
    部署导出器。

    用法：
        deployer = Deployer()
        package = deployer.export(trainer, task_spec, output_dir="./deploy")
        print(f"模型已保存到：{package.model_path}")
        print(package.usage_example)
    """

    def export(
        self,
        trainer:    "Trainer",
        task_spec:  TaskSpec,
        output_dir: str = "./deploy",
    ) -> DeploymentPackage:
        """
        完整部署导出流程：
          1. 导出模型文件（ONNX or joblib）
          2. 生成独立推理脚本
          3. 写入部署说明
          4. 返回 DeploymentPackage

        Args:
            trainer:    训练完成的 Trainer 实例
            task_spec:  任务规格
            output_dir: 输出目录

        Returns:
            DeploymentPackage
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Step 1: 导出模型
        model_path, export_format = self._export_model(trainer, out)

        # Step 2: INT8 量化（ONNX 格式可选）
        if export_format == "onnx":
            model_path = self._quantize_int8(model_path, out)

        # Step 3: 生成推理脚本
        script_path = self._write_inference_script(
            trainer, task_spec, out, export_format
        )

        # Step 4: 写部署说明
        self._write_deploy_readme(trainer, task_spec, out, export_format)

        # Step 4: 收集元信息
        model_size_kb = os.path.getsize(model_path) / 1024
        preproc_info  = self._get_preproc_info(trainer)
        labels        = list(trainer.label_encoder.classes_)
        usage         = self._build_usage_example(task_spec, labels, export_format)

        return DeploymentPackage(
            export_format      = export_format,
            model_path         = str(model_path),
            inference_script   = str(script_path),
            package_dir        = str(out),
            model_size_kb      = round(model_size_kb, 2),
            labels             = labels,
            preprocessing_info = preproc_info,
            usage_example      = usage,
        )

    # ── 模型导出 ─────────────────────────────────────────────────────────────

    def _export_model(
        self, trainer: "Trainer", out: Path
    ):
        """ONNX 优先，joblib 降级"""
        # 尝试 ONNX
        try:
            path = self._export_onnx(trainer, out)
            return path, "onnx"
        except Exception:
            pass

        # 降级：joblib
        path = self._export_joblib(trainer, out)
        return path, "joblib"

    def _export_onnx(self, trainer: "Trainer", out: Path) -> Path:
        """
        将 sklearn Pipeline 导出为 ONNX。

        需要：pip install skl2onnx
        支持导出：TF-IDF + LogisticRegression / LinearSVC / SGD
        """
        from sklearn.pipeline import Pipeline
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import StringTensorType

        if trainer.vectorizer is None or trainer.model is None:
            raise RuntimeError("模型尚未训练")

        # 构建 sklearn Pipeline（整合向量化器和分类器）
        pipeline = Pipeline(steps=[
            ("tfidf", trainer.vectorizer),
            ("clf",   trainer.model),
        ])

        # ONNX 转换（输入：字符串数组）
        initial_type = [("text_input", StringTensorType([None, 1]))]
        onnx_model   = convert_sklearn(pipeline, initial_types=initial_type)

        path = out / "model.onnx"
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        return path

    def _export_joblib(self, trainer: "Trainer", out: Path) -> Path:
        """
        用 joblib 序列化完整 Pipeline（含 TF-IDF + 分类器 + LabelEncoder）。
        体积比 ONNX 稍大，但无需 onnxruntime 即可运行。
        """
        import joblib
        from sklearn.pipeline import Pipeline

        if trainer.vectorizer is None or trainer.model is None:
            raise RuntimeError("模型尚未训练")

        bundle = {
            "pipeline": Pipeline(steps=[
                ("tfidf", trainer.vectorizer),
                ("clf",   trainer.model),
            ]),
            "label_encoder": trainer.label_encoder,
            "task_spec": {
                "domain":         trainer.task_spec.domain,
                "label_schema":   trainer.task_spec.label_schema,
                "evaluation_metric": trainer.task_spec.evaluation_metric,
            },
        }
        path = out / "model.pkl"
        joblib.dump(bundle, path)
        return path

    def _quantize_int8(self, onnx_path: Path, out: Path) -> Path:
        """
        对 ONNX 模型做 INT8 静态量化（体积减半，推理加速 2–4×）。

        路线图 Phase 2 要求：INT8/FP16 自动压缩。
        需要 onnxruntime，若未安装则跳过量化直接返回原路径。

        量化前：model.onnx
        量化后：model_int8.onnx（原文件保留作对比）
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantized_path = out / "model_int8.onnx"
            quantize_dynamic(
                str(onnx_path),
                str(quantized_path),
                weight_type=QuantType.QInt8,
            )
            orig_kb = onnx_path.stat().st_size / 1024
            quant_kb = quantized_path.stat().st_size / 1024
            print(f"  INT8 量化：{orig_kb:.1f} KB → {quant_kb:.1f} KB "
                  f"（压缩 {100*(1-quant_kb/orig_kb):.0f}%）")
            return quantized_path
        except ImportError:
            # onnxruntime 未安装，跳过量化
            return onnx_path
        except Exception as e:
            # 量化失败（某些模型结构不支持），退回原模型
            print(f"  ⚠  INT8 量化跳过（{e}），使用 FP32 ONNX")
            return onnx_path

    def _write_inference_script(
        self,
        trainer:       "Trainer",
        task_spec:     TaskSpec,
        out:           Path,
        export_format: str,
    ) -> Path:
        """生成独立推理脚本，不依赖 automl_agent 项目"""

        labels_repr = repr(list(trainer.label_encoder.classes_))

        if export_format == "onnx":
            script = textwrap.dedent(f'''\
                #!/usr/bin/env python3
                """
                自动生成的推理脚本（ONNX 格式）
                任务：{task_spec.raw_description}
                标签：{labels_repr}

                依赖：pip install onnxruntime numpy
                """
                import numpy as np
                import onnxruntime as ort
                from pathlib import Path

                # 加载模型
                MODEL_PATH = Path(__file__).parent / "model.onnx"
                session    = ort.InferenceSession(str(MODEL_PATH))
                LABELS     = {labels_repr}


                def predict(texts: list[str]) -> list[str]:
                    """
                    对文本列表做分类预测。

                    Args:
                        texts: 输入文本列表，每条是一个字符串

                    Returns:
                        预测标签列表
                    """
                    input_data = np.array(texts, dtype=object).reshape(-1, 1)
                    outputs    = session.run(None, {{"text_input": input_data}})
                    label_indices = outputs[0]              # int 数组
                    return [LABELS[i] for i in label_indices]


                if __name__ == "__main__":
                    test_texts = [
                        "你的测试文本1",
                        "你的测试文本2",
                    ]
                    results = predict(test_texts)
                    for text, label in zip(test_texts, results):
                        print(f"{{label:15s}} ← {{text}}")
            ''')
        else:
            script = textwrap.dedent(f'''\
                #!/usr/bin/env python3
                """
                自动生成的推理脚本（joblib 格式）
                任务：{task_spec.raw_description}
                标签：{labels_repr}

                依赖：pip install scikit-learn joblib
                """
                import joblib
                from pathlib import Path

                # 加载模型包
                MODEL_PATH = Path(__file__).parent / "model.pkl"
                bundle     = joblib.load(MODEL_PATH)
                pipeline   = bundle["pipeline"]
                le         = bundle["label_encoder"]


                def predict(texts: list[str]) -> list[str]:
                    """
                    对文本列表做分类预测。

                    Args:
                        texts: 输入文本列表

                    Returns:
                        预测标签列表
                    """
                    y_enc = pipeline.predict(texts)
                    return le.inverse_transform(y_enc).tolist()


                if __name__ == "__main__":
                    test_texts = [
                        "你的测试文本1",
                        "你的测试文本2",
                    ]
                    results = predict(test_texts)
                    for text, label in zip(test_texts, results):
                        print(f"{{label:15s}} ← {{text}}")
            ''')

        path = out / "inference.py"
        path.write_text(script, encoding="utf-8")
        return path

    def _write_deploy_readme(
        self,
        trainer:       "Trainer",
        task_spec:     TaskSpec,
        out:           Path,
        export_format: str,
    ):
        """生成部署包说明"""
        labels   = list(trainer.label_encoder.classes_)
        dep      = "pip install onnxruntime numpy" if export_format == "onnx" \
                   else "pip install scikit-learn joblib"
        model_f  = "model.onnx" if export_format == "onnx" else "model.pkl"

        readme = textwrap.dedent(f"""\
            # 部署包说明

            任务：{task_spec.raw_description}
            领域：{task_spec.domain}
            标签：{labels}
            格式：{export_format.upper()}

            ## 快速使用

            ```bash
            {dep}
            python inference.py
            ```

            ## 文件说明

            | 文件 | 说明 |
            |------|------|
            | {model_f} | 训练好的分类模型 |
            | inference.py | 推理脚本（可直接集成到生产代码）|
            | README.md | 本文件 |

            ## 集成示例

            ```python
            from inference import predict
            results = predict(["用户输入文本"])
            print(results)  # ['预测标签']
            ```

            ## 模型信息

            - 训练指标：{task_spec.evaluation_metric}
            - 训练语言：{task_spec.language}
            - 格式：{export_format.upper()}
        """)

        (out / "README.md").write_text(readme, encoding="utf-8")

    # ── 辅助方法 ─────────────────────────────────────────────────────────────

    def _get_preproc_info(self, trainer: "Trainer") -> Dict[str, Any]:
        if trainer.vectorizer is None:
            return {}
        return {
            "type":          "TF-IDF",
            "max_features":  trainer.vectorizer.max_features,
            "ngram_range":   list(trainer.vectorizer.ngram_range),
            "vocab_size":    len(trainer.vectorizer.vocabulary_) if hasattr(trainer.vectorizer, "vocabulary_") else 0,
        }

    def _build_usage_example(
        self, task_spec: TaskSpec, labels: list, fmt: str
    ) -> str:
        dep = "onnxruntime" if fmt == "onnx" else "scikit-learn joblib"
        return (
            f"# 安装依赖: pip install {dep}\n"
            f"# 在 deploy/ 目录下运行：\n"
            f"from inference import predict\n"
            f"print(predict(['示例文本']))  # 输出: ['{labels[0] if labels else '标签'}']\n"
        )
