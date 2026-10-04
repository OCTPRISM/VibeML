"""
core/json_extract.py  -  从 LLM 文本响应里稳健解析 JSON（共用工具）

这个"去 markdown 围栏 + json.loads + 正则兜底找 {...}/[...]" 的模式在
task_parser.py / data_engine.py / explainer.py / arch_designer.py /
backbone_selector.py 里几乎逐字重复了五遍——这里收成一个函数，新代码
（core/conversation/ 下的 stage handler）统一用它，不再复制第六遍。

不回头改造上面那五个旧调用点：它们各自的实现已经过充分测试，重构它们
不是这次改动的目的，属于没有必要承担的额外风险。
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    """从可能包含 markdown 代码块/多余文字的 LLM 响应里提取 JSON（对象或数组）"""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"无法解析 JSON 响应：{cleaned[:200]}")
