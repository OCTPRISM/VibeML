"""
core/llm_ft_deployer.py  -  LLM 指令微调部署导出器（core/deployer.py 的 SFT 版本）

只保存 LoRA adapter（core/llm_ft_trainer.py::save_adapter，几十 MB，standard
peft 格式），不做全量合并权重导出——3B 级模型全量 fp16/fp32 保存的磁盘占用
风险不小，全量合并导出作为之后可选的功能，本次不做。

部署包内容：
  adapter/       - peft 标准 adapter 目录（adapter_config.json + adapter_model.safetensors）
  inference.py   - 独立推理脚本（加载基座 + adapter，predict(prompts) -> completions）
  README.md      - 部署说明
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from config import DeploymentPackage, TaskSpec

if TYPE_CHECKING:
    from core.llm_ft_trainer import LLMFTTrainer


def _oneline(text: str) -> str:
    """和 core/rl_deployer.py 一样的坑——task_spec.raw_description 可能是多行，
    直接嵌进 textwrap.dedent() 的 f-string 模板会破坏缩进计算，塞进模板前压成单行。"""
    return " ".join(text.split())


def _dir_size_kb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1024


class LLMFTDeployer:
    """
    用法：
        deployer = LLMFTDeployer()
        package = deployer.export(trainer, task_spec, model_id, output_dir="./deploy")
    """

    def export(
        self,
        trainer:    "LLMFTTrainer",
        task_spec:  TaskSpec,
        model_id:   str,
        output_dir: str = "./deploy",
    ) -> DeploymentPackage:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        adapter_dir = out / "adapter"
        trainer.save_adapter(str(adapter_dir))

        script_path = self._write_inference_script(task_spec, model_id, out)
        self._write_deploy_readme(task_spec, model_id, out)

        model_size_kb = _dir_size_kb(adapter_dir)
        usage = (
            f"# 安装依赖: pip install transformers peft torch\n"
            f"# 在 deploy/ 目录下运行：\n"
            f"from inference import predict\n"
            f"print(predict(['你的指令']))\n"
        )

        return DeploymentPackage(
            export_format      = "lora_adapter",
            model_path          = str(adapter_dir),
            inference_script    = str(script_path),
            package_dir         = str(out),
            model_size_kb       = round(model_size_kb, 2),
            labels              = [],   # 生成式任务没有分类标签概念
            preprocessing_info  = {"base_model_id": model_id},
            usage_example       = usage,
        )

    def _write_inference_script(self, task_spec: TaskSpec, model_id: str, out: Path) -> Path:
        script = textwrap.dedent(f'''\
            #!/usr/bin/env python3
            """
            自动生成的推理脚本（LoRA 指令微调 adapter）
            任务：{_oneline(task_spec.raw_description)}
            底座模型：{model_id}

            依赖：pip install transformers peft torch
            """
            import torch
            from pathlib import Path
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            BASE_MODEL_ID = "{model_id}"
            ADAPTER_DIR = Path(__file__).parent / "adapter"

            _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            _base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, trust_remote_code=False)
            _model = PeftModel.from_pretrained(_base_model, str(ADAPTER_DIR))
            _model.eval()
            _device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            _model.to(_device)


            def _format_prompt(instruction: str, input_text: str = "") -> str:
                if input_text and input_text.strip():
                    return f"### Instruction:\\n{{instruction}}\\n\\n### Input:\\n{{input_text}}\\n\\n### Response:\\n"
                return f"### Instruction:\\n{{instruction}}\\n\\n### Response:\\n"


            def predict(instructions: list, inputs: list = None) -> list[str]:
                """
                给一批指令（可选配对输入），返回模型生成的回复。

                Args:
                    instructions: 指令文本列表
                    inputs: 每条指令对应的输入文本（可选，不提供则视为无输入）

                Returns:
                    生成的回复文本列表
                """
                inputs = inputs or [""] * len(instructions)
                outputs = []
                for instr, inp in zip(instructions, inputs):
                    prompt = _format_prompt(instr, inp)
                    enc = _tokenizer(prompt, return_tensors="pt").to(_device)
                    with torch.no_grad():
                        gen_ids = _model.generate(**enc, max_new_tokens=200, do_sample=False,
                                                  pad_token_id=_tokenizer.pad_token_id)
                    text = _tokenizer.decode(gen_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    outputs.append(text.strip())
                return outputs


            if __name__ == "__main__":
                print(predict(["示例指令"]))
        ''')
        path = out / "inference.py"
        path.write_text(script, encoding="utf-8")
        return path

    def _write_deploy_readme(self, task_spec: TaskSpec, model_id: str, out: Path):
        readme = textwrap.dedent(f"""\
            # 部署包说明（LLM 指令微调）

            任务：{_oneline(task_spec.raw_description)}
            底座模型：{model_id}
            微调方式：LoRA（仅保存 adapter，不含基座权重）

            ## 快速使用

            ```bash
            pip install transformers peft torch
            python inference.py
            ```

            首次运行会从 HuggingFace Hub 下载底座模型「{model_id}」的权重
            （adapter 本身只有几十 MB，不含基座）。

            ## 文件说明

            | 文件/目录 | 说明 |
            |------|------|
            | adapter/ | LoRA adapter（peft 标准格式，`adapter_config.json` + `adapter_model.safetensors`） |
            | inference.py | 推理脚本（可直接集成到生产代码） |
            | README.md | 本文件 |

            ## 集成示例

            ```python
            from inference import predict
            results = predict(["指令1", "指令2"], inputs=["输入1", "输入2"])
            print(results)
            ```
        """)
        (out / "README.md").write_text(readme, encoding="utf-8")
