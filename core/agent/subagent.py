"""
core/agent/subagent.py  -  短生命周期子 Agent。

主 Agent 遇到"需要并行研究、交叉审核"的场景（比如要比较好几个候选数据集）时，
通过 tools.py::spawn_subagent 动态派生一个这里定义的子 Agent。子 Agent 是一个
独立的、比主循环短得多的 complete_with_tools() 循环，工具集被硬编码收窄成只读
研究类（search_datasets/preview_dataset），不包含 submit_training 等任何有副
作用的工具——子 Agent 不能训练模型、不能修改 ConversationState、也不能再派生
下一层子 Agent（工具集里根本没有 spawn_subagent，结构上就杜绝了无限递归）。

跑完（或者达到 max_turns 上限）后返回一个结构化摘要 dict，由调用方
（core/agent/tools.py::_spawn_subagent）包成一个 tool_result 喂回主 Agent的
transcript——子 Agent 本身不知道、也不需要知道主 Agent 的 plan/transcript。

client.complete_with_tools() 现在是真正的 async 方法（core/llm_client.py 三个
provider 实现都用各自的异步 HTTP 客户端），所以多个 run_subagent() 被
asyncio.gather 摆在一起时，等待网络响应期间是真的交替执行、不是"同一时刻只有
一个在跑网络请求"的假并发——这是从最初"同步阻塞、只是代码结构支持"的限制
里专门修的，不是顺带发生的。.complete()（非工具调用版本，其余十几个调用点用
的那个）仍然是同步的，没有一并改，因为改这个是影响全仓库的更大改动，超出这
次要解决的具体问题范围。
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, List, Tuple

from core.data_sources import get_source
from core.llm_client import AgentMessage, LLMClient, ToolDef, ToolResult

MAX_SUBAGENT_TURNS = 8

SUBAGENT_SYSTEM_PROMPT_TEMPLATE = """你是一个短生命周期的研究子 Agent，角色是「{role}」。

主 Agent 交给你的调研任务：
{task_description}

你只能用只读工具做调研（搜索/预览数据集），不能提交训练、不能修改任何状态，
也没有派生下一层子 Agent 的能力。完成调研后直接用一段简洁的文字总结你的发现
和结论（这段总结是给主 Agent 读的，不是给最终用户看的，不需要客套），不需要
再调用任何工具。"""


def _build_subagent_tools() -> Tuple[List[ToolDef], Dict[str, Any]]:
    """独立声明子 Agent 自己的只读工具子集，不复用
    core/agent/tools.py::build_agent_tools（那边是主 Agent 的全集，混了
    submit_training 等有副作用的工具），避免不小心把危险工具带进来。"""
    tools: List[ToolDef] = []
    dispatch: Dict[str, Any] = {}

    # async + asyncio.to_thread：这两个工具会发起真实的 HF/魔搭网络请求，
    # 同步调用会阻塞整个事件循环（拖慢同一进程里其它所有请求，真实复现过）；
    # run_subagent() 的调度循环用 inspect.isawaitable 判断要不要 await，
    # 跟 core/agent/tools.py 主 Agent 工具集是同一个模式
    async def _search_datasets(inp: Dict[str, Any]) -> str:
        platform = inp.get("platform", "huggingface")
        query = inp.get("query", "")
        try:
            results = await asyncio.to_thread(get_source(platform).search, query)
        except Exception as e:
            return f"搜索失败：{e}"
        lines = [f"- {d.ref}（下载量 {d.downloads}，点赞 {d.likes}）：{d.description}" for d in results[:10]]
        return "\n".join(lines) if lines else "没有搜到匹配的数据集。"

    tools.append(ToolDef(
        name="search_datasets",
        description="在 HuggingFace（platform=huggingface）或魔搭（platform=modelscope）"
                    "上按关键词搜索公开数据集。",
        input_schema={
            "type": "object",
            "properties": {
                "platform": {"type": "string", "enum": ["huggingface", "modelscope"]},
                "query": {"type": "string"},
            },
            "required": ["platform", "query"],
        },
    ))
    dispatch["search_datasets"] = _search_datasets

    async def _preview_dataset(inp: Dict[str, Any]) -> str:
        platform = inp.get("platform", "huggingface")
        ref = inp.get("ref", "")
        try:
            preview = await asyncio.to_thread(get_source(platform).preview, ref)
        except Exception as e:
            return f"预览失败：{e}"
        return (f"列名：{preview.columns}；建议文本列：{preview.suggested_text_col}；"
                f"建议标签列：{preview.suggested_label_col}；样本行数：{len(preview.sample_rows)}；"
                f"样例：{preview.sample_rows[:2]}")

    tools.append(ToolDef(
        name="preview_dataset",
        description="预览一个数据集的列名、样本行、自动建议的文本/标签列映射，不会真的下载全部数据。",
        input_schema={
            "type": "object",
            "properties": {
                "platform": {"type": "string", "enum": ["huggingface", "modelscope"]},
                "ref": {"type": "string"},
            },
            "required": ["platform", "ref"],
        },
    ))
    dispatch["preview_dataset"] = _preview_dataset

    return tools, dispatch


async def run_subagent(
    role: str, task_description: str, client: LLMClient, max_turns: int = MAX_SUBAGENT_TURNS,
) -> Dict[str, Any]:
    """返回 {"success": bool, "summary": str}。不产出 ConversationMessage——子
    Agent 本身对前端不可见，只有"派生了/完成了"这两个事件（由调用方产出）可见。"""
    tools, dispatch = _build_subagent_tools()
    system_prompt = SUBAGENT_SYSTEM_PROMPT_TEMPLATE.format(role=role, task_description=task_description)
    transcript: List[AgentMessage] = [AgentMessage(role="user", content="请开始调研并给出总结。")]

    for _turn in range(max_turns):
        try:
            result = await client.complete_with_tools(transcript, tools, system=system_prompt, max_tokens=1000)
        except Exception as e:
            return {"success": False, "summary": f"子 Agent 执行出错：{e}"}

        if result.stop_reason != "tool_use":
            return {"success": True, "summary": result.text.strip() or "（子 Agent 没有给出具体总结）"}

        transcript.append(AgentMessage(role="assistant", content=result.tool_calls))
        tool_results: List[ToolResult] = []
        for call in result.tool_calls:
            fn = dispatch.get(call.name)
            if fn is None:
                tool_results.append(ToolResult(tool_call_id=call.id,
                                                content=f"未知工具：{call.name}", is_error=True))
                continue
            try:
                outcome = fn(call.input)
                content = await outcome if inspect.isawaitable(outcome) else outcome
            except Exception as e:
                content = f"工具执行出错：{e}"
            tool_results.append(ToolResult(tool_call_id=call.id, content=content))
        transcript.append(AgentMessage(role="tool", content=tool_results))

    return {"success": False,
            "summary": f"子 Agent 达到最大轮次上限（{max_turns} 次）仍未给出总结，调研可能不完整。"}
