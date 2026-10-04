"""
core/vlm_gen_deployer.py  -  生成式 VLM 部署导出器（core/vlm_cls_deployer.py 的生成版本）

跟 core/vlm_cls_deployer.py 只导出一个小分类头不同——这里整个 seq2seq 模型
（编码器+解码器）都参与了训练，导出的是完整的 state_dict（~242M 参数，
fp32 下大约 1GB），推理脚本运行时用同一个 model_id 重新加载网络结构
（config/tokenizer 这些不参与训练、不需要重复保存），再灌入训练好的权重。

生成的推理脚本用 ViTImageProcessor + AutoTokenizer 分开处理图片/文字、
model.generate(pixel_values=..., decoder_input_ids=<可选前缀>)——这是
core/vlm_gen_trainer.py 里 VisionEncoderDecoderModel 实际的调用方式，
跟 BLIP 那种"组合 Processor 一次性处理图文"的调用方式不一样，见
core/vlm_gen_trainer.py 顶部注释里换模型的完整原因。
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from config import DeploymentPackage, TaskSpec

if TYPE_CHECKING:
    from core.vlm_gen_trainer import VlmGenTrainer


def _oneline(text: str) -> str:
    """跟 core/rl_deployer.py::_oneline 同样的坑——多行文本直接嵌进
    textwrap.dedent() 的 f-string 模板会破坏公共缩进计算，生成脚本 SyntaxError。"""
    return " ".join(text.split())


class VlmGenDeployer:
    """
    用法：
        deployer = VlmGenDeployer()
        package = deployer.export(trainer, task_spec, output_dir="./deploy")
    """

    def export(self, trainer: "VlmGenTrainer", task_spec: TaskSpec, output_dir: str = "./deploy") -> DeploymentPackage:
        if trainer._model_state is None:
            raise RuntimeError("模型尚未训练")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        import torch
        model_path = out / "model_state_dict.pt"
        torch.save({k: torch.from_numpy(v) for k, v in trainer._model_state.items()}, model_path)

        script_path = self._write_inference_script(trainer, task_spec, out)
        self._write_deploy_readme(trainer, task_spec, out)

        model_size_kb = os.path.getsize(model_path) / 1024
        usage = (
            f"# 安装依赖: pip install torch transformers pillow\n"
            f"# 在 deploy/ 目录下运行：\n"
            f"from inference import predict\n"
            f"print(predict([{{'image_path': 'photo1.jpg', 'prompt': ''}}]))  # 输出: ['一段看图说话的文字']\n"
        )

        return DeploymentPackage(
            export_format       = "torch_full_state_dict",
            model_path           = str(model_path),
            inference_script     = str(script_path),
            package_dir          = str(out),
            model_size_kb        = round(model_size_kb, 2),
            labels                = [],   # 生成式任务没有固定标签集合，字段留空跟其它 Deployer 保持同一个 dataclass 形状
            preprocessing_info   = {"model_id": trainer.model_id},
            usage_example         = usage,
        )

    def _write_inference_script(self, trainer: "VlmGenTrainer", task_spec: TaskSpec, out: Path) -> Path:
        model_id = trainer.model_id
        script = textwrap.dedent(f'''\
            #!/usr/bin/env python3
            """
            自动生成的推理脚本（生成式 VLM：看图说话 / 有参考答案的视觉问答）
            任务：{_oneline(task_spec.raw_description)}

            依赖：pip install torch transformers pillow
            """
            import torch
            from pathlib import Path
            from PIL import Image
            from transformers import AutoModelForImageTextToText, ViTImageProcessor, AutoTokenizer

            MODEL_ID = "{model_id}"
            PROMPT_TEMPLATE = "问题：{{prompt}}\\n回答："
            MODEL_PATH = Path(__file__).parent / "model_state_dict.pt"

            # ViTImageProcessor 检测不到 torchvision 时会自动退回纯 PIL 实现，
            # 不需要安装 torchvision（AutoImageProcessor 会，所以这里不用它）
            _image_processor = ViTImageProcessor.from_pretrained(MODEL_ID)
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            _model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)
            _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            _model.eval()


            def predict(examples: list) -> list[str]:
                """examples: [{{"image_path": str, "prompt": str}}, ...]，prompt 留空表示纯看图说话。"""
                outputs = []
                with torch.no_grad():
                    for ex in examples:
                        image = Image.open(ex["image_path"]).convert("RGB")
                        pixel_values = _image_processor(images=image, return_tensors="pt").pixel_values
                        prompt = ex.get("prompt") or ""
                        if prompt:
                            prefix = PROMPT_TEMPLATE.format(prompt=prompt)
                            prefix_ids = _tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids
                            out_ids = _model.generate(pixel_values=pixel_values, decoder_input_ids=prefix_ids, max_new_tokens=64)
                            full_text = _tokenizer.decode(out_ids[0], skip_special_tokens=True)
                            prefix_text = _tokenizer.decode(prefix_ids[0], skip_special_tokens=True)
                            text = full_text[len(prefix_text):].strip() if full_text.startswith(prefix_text) else full_text.strip()
                        else:
                            out_ids = _model.generate(pixel_values=pixel_values, max_new_tokens=64)
                            text = _tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
                        outputs.append(text)
                return outputs


            if __name__ == "__main__":
                import sys
                if len(sys.argv) > 1:
                    print(predict([{{"image_path": p, "prompt": ""}} for p in sys.argv[1:]]))
                else:
                    print("用法：python inference.py 图片1.jpg 图片2.jpg ...")
        ''')
        path = out / "inference.py"
        path.write_text(script, encoding="utf-8")
        return path

    def _write_deploy_readme(self, trainer: "VlmGenTrainer", task_spec: TaskSpec, out: Path):
        readme = textwrap.dedent(f"""\
            # 部署包说明（生成式 VLM：看图说话 / 视觉问答）

            任务：{_oneline(task_spec.raw_description)}
            底座模型：{trainer.model_id}（编码器+解码器整体参与了训练，权重完整导出）

            ## 快速使用

            ```bash
            pip install torch transformers pillow
            python inference.py photo1.jpg photo2.jpg
            ```

            ## 文件说明

            | 文件 | 说明 |
            |------|------|
            | model_state_dict.pt | 训练好的完整模型权重 |
            | inference.py | 推理脚本（可直接集成到生产代码，支持看图说话/视觉问答两种用法） |
            | README.md | 本文件 |

            ## 集成示例

            ```python
            from inference import predict
            texts = predict([{{"image_path": "photo1.jpg", "prompt": ""}}])
            print(texts)  # ['一段看图说话的文字']
            ```
        """)
        (out / "README.md").write_text(readme, encoding="utf-8")
