"""
core/nn_sandbox.py  -  LLM 生成的 PyTorch 代码：执行前的静态安全门禁

纯 AST 静态分析，不 import torch，可独立单测、无副作用。
这里只是编译期过滤明显危险/不合规的生成代码，不是运行时沙箱——
真正的执行隔离（多进程 + 超时 + 资源上限）在 core/nn_trainer.py 里。

规则：
  - 只允许 import torch / torch.nn / torch.nn.functional（及其子路径），其他一律拒绝
  - 禁止直接引用危险模块名/内建函数：os / sys / subprocess / socket / shutil /
    eval / exec / compile / __import__ / open / input / globals / locals /
    getattr / setattr / delattr / vars / importlib / ctypes / multiprocessing / threading
  - 禁止访问经典沙箱逃逸用到的特定 dunder（__class__ / __bases__ / __subclasses__ /
    __globals__ / __code__ / __mro__ / __reduce__ / ...），但**不**禁止 __init__ 这类
    正常定义/调用 nn.Module 子类必须用到的 dunder（否则连 super().__init__() 都写不出来）
  - 禁止任何函数/类使用装饰器
  - 必须恰好一个顶层类定义，继承 nn.Module / Module，且定义了 __init__ 和 forward
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional

ALLOWED_IMPORT_MODULES = {"torch", "torch.nn", "torch.nn.functional"}

BANNED_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
    "ctypes", "multiprocessing", "threading", "requests", "urllib",
    "eval", "exec", "compile", "__import__", "open", "input",
    "globals", "locals", "getattr", "setattr", "delattr", "vars",
}

# 经典沙箱逃逸链条会用到的特定 dunder；不是"所有 dunder"，__init__/__call__/__len__ 等正常方法不受影响
BANNED_DUNDERS = {
    "__class__", "__bases__", "__base__", "__subclasses__", "__mro__",
    "__globals__", "__code__", "__closure__", "__func__", "__self__",
    "__dict__", "__reduce__", "__reduce_ex__", "__getattribute__",
    "__setattr__", "__delattr__", "__builtins__", "__loader__", "__spec__",
}


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)


def _is_allowed_import(module: str) -> bool:
    return module in ALLOWED_IMPORT_MODULES


def validate(source_code: str) -> ValidationResult:
    """静态校验生成代码，返回 (是否通过, 错误列表)。不执行任何代码。"""
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
        if not any(b in ("Module", "nn.Module") or b == "Module" for b in base_names):
            errors.append(f"类 '{cls.name}' 必须继承 nn.Module")
        method_names = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if "__init__" not in method_names:
            errors.append(f"类 '{cls.name}' 必须定义 __init__")
        if "forward" not in method_names:
            errors.append(f"类 '{cls.name}' 必须定义 forward")

    return ValidationResult(ok=(len(errors) == 0), errors=errors)


def extract_class_name(source_code: str) -> Optional[str]:
    """校验通过后调用，取出唯一的类名（供 nn_trainer 实例化用）"""
    tree = ast.parse(source_code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return node.name
    return None
