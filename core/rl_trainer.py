"""
core/rl_trainer.py  -  强化学习训练执行（LLM 生成环境 + stable-baselines3 训练）

设计要点（core/rl_sandbox.py 校验"环境代码写得对不对"之后，这里管"跑起来安不安全"）：
  - 复用 core/subprocess_runner.py 的隔离机制：训练在独立子进程里跑，父进程用墙钟
    超时兜底卡死的生成代码——不管子进程内部在跑什么，到点就会被外部杀掉。
  - 子进程内部：先用受限 exec()（只允许 import gymnasium/numpy/math/random，第二层
    防线，belt-and-suspenders）实例化校验过的环境类，无条件套一层
    gymnasium.wrappers.TimeLimit(env, max_episode_steps=...)——不管生成代码有没有
    正确设置 terminated，都不会让一个 episode 无限跑下去；再跑一次
    gymnasium.utils.env_checker.check_env 做免费的正确性预检；然后用
    stable-baselines3 的 DQN（离散动作）或 PPO（其他动作空间）做真正的训练——
    训练算法是可信库代码，只有环境是 LLM 生成的，把最高风险的代码执行面限制到最小。
  - 把 MAX_TOTAL_TIMESTEPS 切成几段，每段训练完跑几个 eval episode 算平均 reward，
    包成一个 EpochResult——这样 core/rl_pipeline.py 能直接复用现有事件流/前端可视化
    （学习曲线、迭代日志），不需要为 RL 单独做一套图表。
  - 训练完成后，子进程把 SB3 policy 存成 zip（SB3 自带序列化格式），读成 bytes 送回
    父进程；父进程 predict() 时重新反序列化——和 NNTrainer 的"送回 state_dict，
    predict 时重建模型"是同一个模式。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import TaskSpec, GeneratedEnvSpec, EpochResult
from core.rl_sandbox import validate as sandbox_validate, extract_class_name, ALLOWED_IMPORT_MODULES
from core.subprocess_runner import run_in_subprocess, SubprocessTrainingError

RL_TIMEOUT_SECONDS  = 1200      # 环境代码任意导致吞吐量无法预先估计，超时是主要防线而非权宜之计
MAX_EPISODE_STEPS   = 1000      # TimeLimit 包装：单个 episode 的步数硬上限
MAX_TOTAL_TIMESTEPS = 200_000   # 单次训练调用的默认总步数上限，需要更多训练走对话式"继续训练"多轮迭代
EVAL_EPISODES       = 5         # 每个训练分段结束后跑几个 episode 算平均 reward
N_SEGMENTS          = 5         # 把 total_timesteps 切成几段，每段结束当一次"epoch"上报


class RLValidationError(Exception):
    """生成的环境代码没通过 core/rl_sandbox.py 静态校验，或 gymnasium check_env 不通过"""
    def __init__(self, errors):
        errors = errors if isinstance(errors, list) else [str(errors)]
        super().__init__("; ".join(errors))
        self.errors = errors


class RLTrainingError(SubprocessTrainingError):
    """子进程训练失败（超时 / 运行时异常），携带简短原因供 pipeline.py 决定降级"""


class RLTrainer:
    """强化学习训练器，接口风格和 core/trainer.py::Trainer / core/nn_trainer.py::NNTrainer
    对齐（train_with_eval 返回 EpochResult 列表），但 predict() 输出的是动作而不是
    标签/置信度——RL 场景没有"标签"这个概念，调用方需要按 backend 类型分别处理。"""

    def __init__(self, task_spec: TaskSpec):
        self.task_spec = task_spec
        self._policy_bytes: Optional[bytes] = None
        self._policy_algo: Optional[str] = None    # "PPO" | "DQN"
        self._fitted = False

    def train_with_eval(
        self, env_spec: GeneratedEnvSpec,
        total_timesteps: int = MAX_TOTAL_TIMESTEPS,
    ) -> List[EpochResult]:
        check = sandbox_validate(env_spec.source_code)
        if not check.ok:
            raise RLValidationError(check.errors)
        class_name = extract_class_name(env_spec.source_code) or env_spec.class_name

        payload = {
            "source_code": env_spec.source_code,
            "class_name": class_name,
            "total_timesteps": min(total_timesteps, MAX_TOTAL_TIMESTEPS),
            "n_segments": N_SEGMENTS,
            "eval_episodes": EVAL_EPISODES,
            "max_episode_steps": MAX_EPISODE_STEPS,
        }
        result = run_in_subprocess(payload, _rl_train_entrypoint, timeout=RL_TIMEOUT_SECONDS,
                                   error_cls=RLTrainingError)

        self._policy_bytes = result["policy_bytes"]
        self._policy_algo  = result["algo"]
        self._fitted = True
        return [EpochResult(**e) for e in result["epochs"]]

    def predict(self, observations: List[List[float]]) -> List[int]:
        """给一批观测，返回策略选择的动作"""
        if not self._fitted:
            raise RuntimeError("模型尚未训练")
        import numpy as np
        model = self._load_policy()
        actions = []
        for obs in observations:
            action, _ = model.predict(np.array(obs), deterministic=True)
            actions.append(int(action))
        return actions

    def _load_policy(self):
        from stable_baselines3 import PPO, DQN
        cls = {"PPO": PPO, "DQN": DQN}[self._policy_algo]
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(self._policy_bytes)
            path = f.name
        try:
            return cls.load(path)
        finally:
            Path(path).unlink(missing_ok=True)


# ── 子进程执行入口（必须是模块级函数，spawn 要求可 pickle）───────────────────────

def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """替换掉的 __import__：只允许 core/rl_sandbox.py 白名单里的模块。
    AST 门禁已经在执行前拦过一遍非法 import，这里是第二层防线——
    exec() 环境本身也不给"能 import 任意模块"的能力，而不是仅仅"事先没被发现"。"""
    if name not in ALLOWED_IMPORT_MODULES:
        raise ImportError(f"import '{name}' 不被允许")
    import importlib
    module = importlib.import_module(name)
    if not fromlist:
        return importlib.import_module(name.split(".")[0])
    return module


def _safe_exec_globals() -> Dict[str, Any]:
    """受限 exec 用的最小 builtins 集合（AST 门禁是主要防线，这里是 belt-and-suspenders）"""
    import builtins
    safe_names = ("range", "len", "enumerate", "int", "float", "str", "bool", "list", "dict",
                  "tuple", "set", "super", "print", "min", "max", "sum", "abs", "isinstance",
                  "True", "False", "None", "ValueError", "TypeError", "Exception", "zip")
    safe_builtins = {name: getattr(builtins, name) for name in safe_names if hasattr(builtins, name)}
    safe_builtins["__import__"] = _restricted_import
    safe_builtins["__build_class__"] = builtins.__build_class__
    return {"__builtins__": safe_builtins, "__name__": "generated_env"}


def _instantiate_env_class(source_code: str, class_name: str):
    exec_globals = _safe_exec_globals()
    exec(compile(source_code, "<generated_env>", "exec"), exec_globals)
    return exec_globals[class_name]


def _rl_train_entrypoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    from gymnasium.wrappers import TimeLimit
    from gymnasium.utils.env_checker import check_env
    from gymnasium import spaces

    env_cls = _instantiate_env_class(payload["source_code"], payload["class_name"])
    try:
        raw_env = env_cls()
    except Exception as e:
        return {"error": f"环境初始化失败：{e}"}

    try:
        check_env(raw_env, skip_render_check=True)
    except Exception as e:
        return {"error": f"环境未通过 gymnasium 正确性校验：{e}"}

    env = TimeLimit(raw_env, max_episode_steps=payload["max_episode_steps"])

    algo_name = "DQN" if isinstance(env.action_space, spaces.Discrete) else "PPO"
    from stable_baselines3 import PPO, DQN
    ModelCls = {"DQN": DQN, "PPO": PPO}[algo_name]

    try:
        model = ModelCls("MlpPolicy", env, verbose=0)
    except Exception as e:
        return {"error": f"无法为该动作/观测空间创建 {algo_name} 模型：{e}"}

    n_segments = payload["n_segments"]
    steps_per_segment = max(1, payload["total_timesteps"] // n_segments)
    epochs_out: List[Dict] = []

    for seg in range(n_segments):
        try:
            model.learn(total_timesteps=steps_per_segment, reset_num_timesteps=False)
        except Exception as e:
            if seg == 0:
                return {"error": f"训练时出错：{e}"}
            break  # 已经跑完至少一段，后面再出错就提前收尾，返回已有结果而不是整体失败

        rewards = []
        for _ in range(payload["eval_episodes"]):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += float(reward)
                done = terminated or truncated
            rewards.append(ep_reward)

        mean_reward = sum(rewards) / len(rewards)
        var = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        epochs_out.append({
            "epoch":               seg + 1,
            "train_loss":          0.0,     # RL 没有传统意义上的 train_loss，占位保持事件契约不变
            "val_loss":            0.0,
            "val_metric":          round(mean_reward, 4),
            "metric_name":         "episode_reward_mean",
            "per_class_metrics":   {},
            "confusion_highlights": [f"reward_std={round(var ** 0.5, 4)}"],
        })

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        tmp_path = f.name
    model.save(tmp_path)
    policy_bytes = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)

    return {"epochs": epochs_out, "policy_bytes": policy_bytes, "algo": algo_name}
