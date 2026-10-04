"""
core/vlm_cls_deployer.py  -  VLM 图像分类部署导出器（core/deployer.py 的图像版本）

只导出训练出来的分类头（几十 KB 级别），不导出整个 CLIP 编码器——推理脚本运行时
从 HuggingFace 重新加载冻结的编码器（跟训练时用的是同一个 encoder_id，结果完全
一致），这样部署包体积不会因为带着一个 ~90M 参数的编码器而膨胀到几百 MB。
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from config import DeploymentPackage, TaskSpec

if TYPE_CHECKING:
    from core.vlm_cls_trainer import VlmClsTrainer


def _oneline(text: str) -> str:
    """跟 core/rl_deployer.py::_oneline 同样的坑——多行文本直接嵌进
    textwrap.dedent() 的 f-string 模板会破坏公共缩进计算，生成脚本 SyntaxError。"""
    return " ".join(text.split())


class VlmClsDeployer:
    """
    用法：
        deployer = VlmClsDeployer()
        package = deployer.export(trainer, task_spec, output_dir="./deploy")
    """

    def export(self, trainer: "VlmClsTrainer", task_spec: TaskSpec, output_dir: str = "./deploy") -> DeploymentPackage:
        if trainer._head_state is None:
            raise RuntimeError("模型尚未训练")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        import torch
        model_path = out / "head_state_dict.pt"
        torch.save({k: torch.from_numpy(v) for k, v in trainer._head_state.items()}, model_path)

        # LabelEncoder.classes_ 是 numpy.str_（str 的子类，json.dumps 能处理，但
        # 显式转成普通 str 更干净，避免下游任何地方对类型做严格 isinstance(str) 检查时出岔子
        labels = [str(x) for x in trainer.label_encoder.classes_]
        script_path = self._write_inference_script(trainer, task_spec, labels, out)
        self._write_deploy_readme(trainer, task_spec, labels, out)

        model_size_kb = os.path.getsize(model_path) / 1024
        usage = (
            f"# 安装依赖: pip install torch transformers pillow\n"
            f"# 在 deploy/ 目录下运行：\n"
            f"from inference import predict\n"
            f"print(predict(['photo1.jpg', 'photo2.jpg']))  # 输出: ['类别A', '类别B']\n"
        )

        return DeploymentPackage(
            export_format       = "torch_head_state_dict",
            model_path           = str(model_path),
            inference_script     = str(script_path),
            package_dir          = str(out),
            model_size_kb        = round(model_size_kb, 2),
            labels                = labels,
            preprocessing_info   = {"encoder_id": trainer.encoder_id, "embed_dim": 512},
            usage_example         = usage,
        )

    def _write_inference_script(self, trainer: "VlmClsTrainer", task_spec: TaskSpec,
                                labels: list, out: Path) -> Path:
        encoder_id = trainer.encoder_id
        script = textwrap.dedent(f'''\
            #!/usr/bin/env python3
            """
            自动生成的推理脚本（VLM 图像分类：冻结 CLIP 编码器 + 训练好的分类头）
            任务：{_oneline(task_spec.raw_description)}
            类别：{labels}

            依赖：pip install torch transformers pillow
            """
            import torch
            import torch.nn as nn
            from pathlib import Path
            from PIL import Image
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

            ENCODER_ID = "{encoder_id}"
            LABELS = {labels!r}
            HEAD_PATH = Path(__file__).parent / "head_state_dict.pt"

            _processor = CLIPImageProcessor.from_pretrained(ENCODER_ID)
            _encoder = CLIPVisionModelWithProjection.from_pretrained(ENCODER_ID)
            _encoder.eval()

            def _build_head(embed_dim, num_classes):
                hidden = max(32, embed_dim // 4)
                return nn.Sequential(
                    nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(hidden, num_classes),
                )

            _head = _build_head(512, len(LABELS))
            _head.load_state_dict(torch.load(HEAD_PATH, map_location="cpu"))
            _head.eval()


            def predict(image_paths: list) -> list[str]:
                """给一批图片路径，返回预测的类别标签。"""
                images = [Image.open(p).convert("RGB") for p in image_paths]
                inputs = _processor(images=images, return_tensors="pt")
                with torch.no_grad():
                    embeds = _encoder(**inputs).image_embeds
                    logits = _head(embeds)
                    idx = logits.argmax(dim=1).tolist()
                return [LABELS[i] for i in idx]


            if __name__ == "__main__":
                import sys
                if len(sys.argv) > 1:
                    print(predict(sys.argv[1:]))
                else:
                    print("用法：python inference.py 图片1.jpg 图片2.jpg ...")
        ''')
        path = out / "inference.py"
        path.write_text(script, encoding="utf-8")
        return path

    def _write_deploy_readme(self, trainer: "VlmClsTrainer", task_spec: TaskSpec, labels: list, out: Path):
        readme = textwrap.dedent(f"""\
            # 部署包说明（VLM 图像分类）

            任务：{_oneline(task_spec.raw_description)}
            视觉编码器：{trainer.encoder_id}（冻结，推理时从 HuggingFace 重新加载）
            类别：{labels}

            ## 快速使用

            ```bash
            pip install torch transformers pillow
            python inference.py photo1.jpg photo2.jpg
            ```

            ## 文件说明

            | 文件 | 说明 |
            |------|------|
            | head_state_dict.pt | 训练好的分类头权重（编码器本身不打包，运行时重新加载） |
            | inference.py | 推理脚本（可直接集成到生产代码） |
            | README.md | 本文件 |

            ## 集成示例

            ```python
            from inference import predict
            labels = predict(["photo1.jpg", "photo2.jpg"])
            print(labels)  # ['类别A', '类别B']
            ```
        """)
        (out / "README.md").write_text(readme, encoding="utf-8")
