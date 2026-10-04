"""
core/rl_sandbox.py  -  LLM 生成的强化学习环境代码：执行前的静态安全门禁

纯 AST 静态分析，不 import gymnasium，可独立单测、无副作用。
这里只是编译期过滤明显危险/不合规的生成代码，不是运行时沙箱——
真正的执行隔离（多进程 + 超时）在 core/rl_trainer.py（通过 core/subprocess_runner.py）。

和 core/nn_sandbox.py 的关系：结构契约本质不同（"一个 nn.Module 分类头"
vs "一个 gymnasium.Env 环境"），所以是独立文件而不是改造 nn_sandbox.py；
但危险名字/dunder/禁止装饰器这套通用规则原样复用，不重复定义第二遍。

规则：
  - 只允许 import gymnasium / gymnasium.spaces / numpy / math / random（及其子路径）；
    刻意不允许 import torch —— 策略网络由 stable-baselines3 提供，环境代码没有
    合法理由要用到 torch，缩小攻击面
  - 危险名字/dunder/禁止装饰器：复用 core/nn_sandbox.py 的 BANNED_NAMES/BANNED_DUNDERS
  - 必须恰好一个顶层类定义，继承 gymnasium.Env / gym.Env / Env，
    且定义了 __init__、reset、step，并在类体内某处出现
    self.action_space = ... 和 self.observation_space = ... 赋值
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional

from core.nn_sandbox import BANNED_NAMES, BANNED_DUNDERS

ALLOWED_IMPORT_MODULES = {"gymnasium", "gymnasium.spaces", "numpy", "math", "random", "typing"}
# typing 是运行时才发现必须放行的例外：gymnasium.Env 本身继承自 typing.Generic，
# 子类化它会在类创建阶段触发 typing 内部逻辑的隐式 import，和生成代码有没有显式
# `import typing` 无关——typing 模块本身没有文件/网络/进程访问能力，放行它不扩大风险面。

_ENV_BASE_NAMES = {"Env", "gym.Env", "gymnasium.Env"}


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)


def _is_allowed_import(module: str) -> bool:
    return module in ALLOWED_IMPORT_MODULES


def validate(source_code: str) -> ValidationResult:
    """静态校验 LLM 生成的环境代码，返回 (是否通过, 错误列表)。不执行任何代码。"""
    errors: List[str] = []
    try:
        tree = ast.parse(source_code, mode="exec")
    except SyntaxError as e:
        return ValidationResult(ok=False, errors=[f"语法错误：{e}"])

    class_defs: List[ast.ClassDef] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_allowed_import(alias.name):
                    errors.append(f"不允许 import '{alias.name}'（只允许 {', '.join(sorted(ALLOWED_IMPORT_MODULES))}）")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if not _is_allowed_import(mod):
                errors.append(f"不允许 from '{mod}' import ...（只允许 {', '.join(sorted(ALLOWED_IMPORT_MODULES))}）")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                errors.append(f"禁止使用名字 '{node.id}'")
            elif node.id in BANNED_DUNDERS:
                errors.append(f"禁止访问 '{node.id}'")
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_NAMES:
                errors.append(f"禁止访问属性 '.{node.attr}'")
            elif node.attr in BANNED_DUNDERS:
                errors.append(f"禁止访问属性 '.{node.attr}'")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.decorator_list:
                errors.append(f"'{node.name}' 不允许使用装饰器")
            if isinstance(node, ast.ClassDef):
                class_defs.append(node)

    if len(class_defs) != 1:
        errors.append(f"必须恰好定义 1 个类（当前检测到 {len(class_defs)} 个）")
    else:
        cls = class_defs[0]
        base_names = [
            b.id if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else "")
            for b in cls.bases
        ]
        if not any(b in _ENV_BASE_NAMES for b in base_names):
            errors.append(f"类 '{cls.name}' 必须继承 gymnasium.Env")

        method_names = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if "__init__" not in method_names:
            errors.append(f"类 '{cls.name}' 必须定义 __init__")
        if "reset" not in method_names:
            errors.append(f"类 '{cls.name}' 必须定义 reset")
        if "step" not in method_names:
            errors.append(f"类 '{cls.name}' 必须定义 step")

        assigned_attrs = {
            n.attr for assign in ast.walk(cls)
            if isinstance(assign, ast.Assign)
            for n in assign.targets
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"
        }
        if "action_space" not in assigned_attrs:
            errors.append(f"类 '{cls.name}' 必须设置 self.action_space")
        if "observation_space" not in assigned_attrs:
            errors.append(f"类 '{cls.name}' 必须设置 self.observation_space")

    return ValidationResult(ok=(len(errors) == 0), errors=errors)


def extract_class_name(source_code: str) -> Optional[str]:
    """校验通过后调用，取出唯一的类名（供 rl_trainer 实例化用）"""
    tree = ast.parse(source_code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return node.name
    return None
