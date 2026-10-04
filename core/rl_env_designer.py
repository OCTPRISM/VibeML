"""
core/rl_env_designer.py  -  强化学习：LLM 根据任务描述生成 Gym 风格环境

契约（写死在 prompt 里，core/rl_trainer.py 按这个契约实例化生成的类）：
  - 生成的类可以叫任意名字，但必须继承 gymnasium.Env
  - __init__(self)：无必填参数，内部设置 self.action_space / self.observation_space
  - reset(self, seed=None, options=None) -> (obs, info)
  - step(self, action) -> (obs, reward, terminated, truncated, info)
  - 只允许 import gymnasium / gymnasium.spaces / numpy / math / random，
    刻意不允许 import torch —— 策略网络由 stable-baselines3 提供，训练算法是
    可信库代码，只有环境本身是 LLM 生成，把最高风险的代码执行面限制到最小

生成的代码在真正执行前必须经过 core/rl_sandbox.py 的静态校验，本模块自己不执行任何代码。
子进程隔离 + TimeLimit 步数上限（core/rl_trainer.py）才是"卡死/失控代码"的真正防线，
这里的 prompt 约束只是尽量让第一次生成就是对的，减少重试次数。
"""

from __future__ import annotations

from core.llm_client import LLMClient
from core.json_extract import extract_json
from config import TaskSpec, GeneratedEnvSpec

SYSTEM_PROMPT = """你是一个强化学习环境设计师，负责根据任务描述，用 gymnasium 库设计一个自定义环境。

严格约束（违反任何一条都会被自动拒绝，不会被执行）：
1. 只能 `import gymnasium as gym`（或 `import gymnasium`）、`from gymnasium import spaces`、
   `import numpy as np`、`import math`、`import random`，不允许任何其他 import（尤其不允许 import torch，
   策略网络由训练框架提供，环境代码不需要它）
2. 必须恰好定义 1 个类，继承 gym.Env（或 gymnasium.Env）
3. `__init__(self)`：不能有其他必填参数；必须在这里设置 `self.action_space`（用 `spaces.Discrete`
   或 `spaces.Box` 等）和 `self.observation_space`（同样用 spaces.* 定义）
4. `reset(self, seed=None, options=None)`：返回 `(observation, info_dict)`，observation 的形状/类型
   必须和 self.observation_space 一致
5. `step(self, action)`：返回 `(observation, reward, terminated, truncated, info_dict)`，
   reward 是 float，terminated/truncated 是 bool——一定要有明确的终止条件（不能设计成永远不终止的环境）
6. 不允许使用装饰器；不允许出现 os/sys/subprocess/eval/exec/open 等危险内容
7. 环境的物理/规则逻辑要清晰简单，避免任何可能无限循环的写法（比如 while True 不带明确 break 条件）

输出 JSON（只输出 JSON，不要其他内容；code 字段里的换行请用 \\n 转义）：
{
  "code": "完整的 Python 源码字符串",
  "class_name": "类名，必须和 code 里定义的类名一致",
  "action_space_desc": "一句话说明动作空间是什么（给非技术人员看）",
  "observation_space_desc": "一句话说明观测空间是什么（给非技术人员看）",
  "reward_rationale": "一句话说明奖励函数是怎么设计的、为什么这样设计能引导智能体学到期望行为"
}
"""


class RLEnvDesigner:
    def __init__(self, client: LLMClient):
        self.client = client

    def design(self, task_spec: TaskSpec, repair_hint: str = "") -> GeneratedEnvSpec:
        prompt = f"任务：{task_spec.raw_description}\n领域：{task_spec.domain}\n"
        if repair_hint:
            prompt += f"\n上一次生成的代码执行失败，错误信息：\n{repair_hint}\n请修复这个问题，重新生成。"

        raw = self.client.complete(system=SYSTEM_PROMPT, user=prompt, max_tokens=1800)
        parsed = extract_json(raw)
        return GeneratedEnvSpec(
            class_name=parsed.get("class_name", ""),
            source_code=parsed.get("code", ""),
            action_space_desc=parsed.get("action_space_desc", ""),
            observation_space_desc=parsed.get("observation_space_desc", ""),
            reward_rationale=parsed.get("reward_rationale", ""),
        )
