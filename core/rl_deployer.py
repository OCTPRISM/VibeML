"""
core/rl_deployer.py  -  强化学习部署导出器（core/deployer.py 的 RL 版本）

和 sklearn 的 core/deployer.py 不同：不做 ONNX/joblib 转换——stable-baselines3
自带的 `.zip` 序列化格式已经是可移植格式（包含策略网络权重 + 超参数 + 归一化统计量），
直接用它，不额外造一层转换逻辑增加出错面。

部署包内容：
  policy.zip     - SB3 policy（RLTrainer._policy_bytes 原样落盘）
  inference.py   - 独立推理脚本（加载 policy，predict(obs) -> action）
  README.md      - 部署说明（含动作/观测空间的通俗解释）
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from config import DeploymentPackage, TaskSpec, GeneratedEnvSpec

if TYPE_CHECKING:
    from core.rl_trainer import RLTrainer


def _oneline(text: str) -> str:
    """task_spec.raw_description 可能是多行（RL 的 prepare_data 阶段会把任务描述
    和环境补充说明拼成两行存回 raw_description）——直接嵌进 textwrap.dedent() 的
    f-string 模板会导致某一行顶格、破坏 dedent 用来找"公共缩进"的计算，生成的脚本
    文件整体缩进错乱，SyntaxError。塞进模板前压成单行，从根上避免这个坑。"""
    return " ".join(text.split())


class RLDeployer:
    """
    用法：
        deployer = RLDeployer()
        package = deployer.export(trainer, task_spec, env_spec, output_dir="./deploy")
    """

    def export(
        self,
        trainer:    "RLTrainer",
        task_spec:  TaskSpec,
        env_spec:   GeneratedEnvSpec,
        output_dir: str = "./deploy",
    ) -> DeploymentPackage:
        if trainer._policy_bytes is None:
            raise RuntimeError("模型尚未训练")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        model_path = out / "policy.zip"
        model_path.write_bytes(trainer._policy_bytes)

        script_path = self._write_inference_script(trainer, task_spec, env_spec, out)
        self._write_deploy_readme(trainer, task_spec, env_spec, out)

        model_size_kb = os.path.getsize(model_path) / 1024
        usage = (
            f"# 安装依赖: pip install stable-baselines3 gymnasium\n"
            f"# 在 deploy/ 目录下运行：\n"
            f"from inference import predict\n"
            f"print(predict([观测值]))  # 输出: [动作]\n"
        )

        return DeploymentPackage(
            export_format      = trainer._policy_algo.lower(),   # "ppo" | "dqn"
            model_path          = str(model_path),
            inference_script    = str(script_path),
            package_dir         = str(out),
            model_size_kb       = round(model_size_kb, 2),
            labels              = [],   # RL 没有分类标签概念
            preprocessing_info  = {
                "algo": trainer._policy_algo,
                "action_space": env_spec.action_space_desc,
                "observation_space": env_spec.observation_space_desc,
            },
            usage_example       = usage,
        )

    def _write_inference_script(
        self, trainer: "RLTrainer", task_spec: TaskSpec, env_spec: GeneratedEnvSpec, out: Path,
    ) -> Path:
        algo = trainer._policy_algo
        script = textwrap.dedent(f'''\
            #!/usr/bin/env python3
            """
            自动生成的推理脚本（强化学习策略，{algo}）
            任务：{_oneline(task_spec.raw_description)}
            动作空间：{_oneline(env_spec.action_space_desc)}
            观测空间：{_oneline(env_spec.observation_space_desc)}

            依赖：pip install stable-baselines3 gymnasium
            """
            import numpy as np
            from pathlib import Path
            from stable_baselines3 import {algo}

            MODEL_PATH = Path(__file__).parent / "policy.zip"
            model = {algo}.load(str(MODEL_PATH))


            def predict(observations: list) -> list[int]:
                """
                给一批观测，返回策略选择的动作。

                Args:
                    observations: 观测值列表，每个元素的形状需要和训练时的观测空间一致

                Returns:
                    动作列表
                """
                actions = []
                for obs in observations:
                    action, _ = model.predict(np.array(obs), deterministic=True)
                    actions.append(int(action))
                return actions


            if __name__ == "__main__":
                # 替换成和观测空间形状一致的示例观测
                test_obs = [[0.0]]
                print(predict(test_obs))
        ''')
        path = out / "inference.py"
        path.write_text(script, encoding="utf-8")
        return path

    def _write_deploy_readme(
        self, trainer: "RLTrainer", task_spec: TaskSpec, env_spec: GeneratedEnvSpec, out: Path,
    ):
        readme = textwrap.dedent(f"""\
            # 部署包说明（强化学习）

            任务：{_oneline(task_spec.raw_description)}
            算法：{trainer._policy_algo}
            动作空间：{_oneline(env_spec.action_space_desc)}
            观测空间：{_oneline(env_spec.observation_space_desc)}
            奖励设计：{_oneline(env_spec.reward_rationale)}

            ## 快速使用

            ```bash
            pip install stable-baselines3 gymnasium
            python inference.py
            ```

            ## 文件说明

            | 文件 | 说明 |
            |------|------|
            | policy.zip | 训练好的策略（stable-baselines3 自带序列化格式） |
            | inference.py | 推理脚本（可直接集成到生产代码） |
            | README.md | 本文件 |

            ## 集成示例

            ```python
            from inference import predict
            actions = predict([观测值1, 观测值2])
            print(actions)  # [动作1, 动作2]
            ```
        """)
        (out / "README.md").write_text(readme, encoding="utf-8")
