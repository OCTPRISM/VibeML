"""
core/conversation/memory.py  -  单个对话内的运行时记忆管理 + token 压缩

背景：core/llm_client.py::LLMClient.complete() 每次调用都是无状态单轮调用，
不会自动携带之前的对话历史；ConversationState.messages 本身也没有任何长度
上限。长对话（尤其是"训练完成→用户不满意→回到 choose_model 重新配置"这种
循环，每轮都会把该次训练的全部 training_event 原样追加进去）会无限增长，
而且重试时系统不记得之前试过什么、为什么失败。

这里做两件事：
  1. "记忆"——用一份持续更新的精简文本摘要（state.memory_summary）记录关键
     事实（任务描述、每轮训练结果、用户不满意的具体原因），而不是让后续的
     LLM 调用去重放全部原始消息。
  2. "token 压缩"——摘要本身如果因为重试轮次太多而变长，就调一次 LLM 把它
     压缩重写；同时把 state.messages 里已经"翻篇"的训练中间过程事件（比如
     早前几轮的 epoch_done）裁掉，只留关键节点，防止对话记录随轮次无限增长。

只做单个对话内的记忆，不做跨对话/跨用户的持久记忆——现有系统没有用户身份
体系，那是范围之外的另一块工作。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.json_extract import extract_json

if TYPE_CHECKING:
    from core.llm_client import LLMClient
    from .state import ConversationState

MAX_SUMMARY_CHARS = 1200          # memory_summary 超过这个长度就触发一次 LLM 压缩重写
MAX_MESSAGES_BEFORE_TRIM = 60     # state.messages 超过这个条数就裁剪旧的训练中间过程事件

# 训练事件里值得永久保留的关键节点——中间过程（epoch_done 等）翻篇后可以裁掉，
# 这些是"这一轮训练最终发生了什么"的关键信息，裁掉会丢失可读性
_KEEP_TRAINING_EVENT_TYPES = {
    "task_parsed", "arch_designed", "nn_codegen_fallback",
    "iteration_done", "deploy_done", "finished", "error",
}

_COMPRESS_SYSTEM_PROMPT = """你是一个对话记忆压缩器。下面是一段 AutoML 训练助手在对话过程中
积累的关键事实记录（任务描述、已做的决定、每轮训练结果、用户反馈）。请把它压缩重写成更精简
的版本，用短句罗列，保留：最初的任务目标、目前为止试过的方法和结果、用户明确表达过的不满意
原因。去掉重复和冗余措辞。只输出压缩后的文本，不要输出任何解释或 JSON。"""

_SUGGESTION_SYSTEM_PROMPT = """你是一个 AI 训练顾问。用户对上一轮训练结果不满意，要求调整后重试。
下面是到目前为止的对话记忆摘要。请用一句话给出一个具体、有针对性的建议（结合摘要里提到的
上一轮用了什么方法、结果如何、用户不满意的具体原因），帮用户决定这一轮该怎么调整。
用通俗语言，不用技术术语。

输出 JSON（只输出 JSON）：{"suggestion": "一句话建议"}"""


def record_fact(state: "ConversationState", fact: str) -> None:
    """规则拼接：把一条新事实追加到 memory_summary 末尾（换行分隔）"""
    if not fact:
        return
    state.memory_summary = f"{state.memory_summary}\n{fact}".strip() if state.memory_summary else fact


def maybe_compress(state: "ConversationState", client: "LLMClient") -> None:
    """memory_summary 过长时调一次 LLM 压缩重写；LLM 调用失败就保留原文，不让这个增强影响主流程"""
    if len(state.memory_summary) <= MAX_SUMMARY_CHARS:
        return
    try:
        compressed = client.complete(
            system=_COMPRESS_SYSTEM_PROMPT, user=state.memory_summary, max_tokens=500,
        )
        if compressed.strip():
            state.memory_summary = compressed.strip()
    except Exception:
        pass   # 压缩失败不影响对话继续，只是这次没压缩成功，摘要保持原样


def trim_messages(state: "ConversationState") -> None:
    """state.messages 超过上限时，把已经翻篇的训练中间过程事件裁掉，
    只留关键节点和全部用户/assistant 文本消息——裁掉的只是对话回放列表，
    不影响 api/store.py::TaskRecord.event_log（训练可视化回放走那个，独立存储）"""
    if len(state.messages) <= MAX_MESSAGES_BEFORE_TRIM:
        return
    kept = []
    for m in state.messages:
        if m.type != "training_event":
            kept.append(m)
            continue
        payload = m.payload if isinstance(m.payload, dict) else {}
        if payload.get("type") in _KEEP_TRAINING_EVENT_TYPES:
            kept.append(m)
    state.messages = kept


def suggest_retry_hint(state: "ConversationState", client: "LLMClient") -> str:
    """基于 memory_summary 生成一句针对性建议，供 choose_model 重试时接在写死问句前面。
    memory_summary 为空或 LLM 调用失败时返回空字符串，调用方回退到原来的写死问句。"""
    if not state.memory_summary:
        return ""
    try:
        raw = client.complete(system=_SUGGESTION_SYSTEM_PROMPT, user=state.memory_summary, max_tokens=200)
        parsed = extract_json(raw)
        return (parsed.get("suggestion") or "").strip()
    except Exception:
        return ""
