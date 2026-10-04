"""
core/agent/mcp_client.py  -  真·MCP 接入：把用户在前端配置的 MCP 服务器地址
（state.mcp_server_urls）转换成可以直接合并进主 Agent 工具集的 ToolDef/dispatch。

**关键实现约束（实测踩过的坑，不是理论顾虑）**：mcp SDK 的
streamablehttp_client/ClientSession 内部用 anyio TaskGroup 管理读写协程，这类
资源的"建立连接 - 使用 - 关闭"必须全部发生在同一个 asyncio Task 里——一旦
`__aenter__` 在一个 task 里执行、后续的工具调用或 `__aexit__` 在另一个 task 里
（哪怕只是被另一个协程 await 调用），anyio 就会报
"Attempted to exit cancel scope in a different task than it was entered in"
之类的错误，而且一旦某个连接失败，还会连累同一个 AsyncExitStack 里其它本来
连接成功的 session 在关闭时一起炸掉。这里用"每个 MCP 服务器一个专属后台
task + asyncio.Queue 请求/响应"的模式解决：整个连接生命周期
（streamablehttp_client + ClientSession 的两层 async with）从头到尾都在
_McpConnection._run() 这一个 task 里跑完，工具调用通过队列转发进去，不把
ClientSession 对象直接传给其它 task 调用。

生命周期：McpToolset 由 AgentOrchestrator.run() 在每次执行开头创建、连接，这
次 run() 期间（可能有多轮工具调用）全程复用同一批连接，run() 结束前（不管
正常结束、暂停等确认还是异常）统一 aclose()。这跟 run() 本身"一次 handle_message
调用对应一次有界执行"的生命周期完全对齐——不需要、也没办法跨 pause/resume 保
活一个 MCP 连接，因为 resume 是全新的一次 asyncio 任务，没有长驻进程能一直
攥着上次的连接对象不放。每次 resume 都会重新连接同样的 URL，这是可以接受的
代价（重连一次的开销远小于维护跨请求存活的连接池）。

每个 URL 独立失败隔离——一个 MCP 服务器连不上/超时，只影响它自己那部分工具
不可用，不会让整个 Agent 运行失败或者拖慢其它服务器的连接。

已知限制：不做鉴权/凭据管理，只支持无认证的 MCP 服务器，或者 URL 里自带 token
的公开/自持服务器（例如 https://host/mcp?token=xxx）；真正的凭据管理（类似
OAuth 流程）超出本期范围。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool as McpTool

from core.agent.events import AgentEvent
from core.llm_client import ToolDef

logger = logging.getLogger(__name__)

MCP_CONNECT_TIMEOUT_SECONDS = 15


def _describe_connect_error(e: BaseException) -> str:
    """anyio 的 TaskGroup 把内部协程的真实失败原因包进
    ExceptionGroup("unhandled errors in a TaskGroup", [...])，直接 str() 那个
    外层包装没有任何实际信息量（"unhandled errors in a TaskGroup (1
    sub-exception)"）——这里往里挖到最深的那个叶子异常，把它的类型名 + 消息
    作为真正对用户有意义的错误原因。用 duck typing（.exceptions 属性）而不是
    `isinstance(e, BaseExceptionGroup)`——那个类型 Python 3.11+ 才是内置的，
    这个仓库要求 3.10+（见 requirements.txt 顶部注释），3.10 上 anyio 走的是
    exceptiongroup 这个 backport 包，实例同样有 .exceptions 属性，duck typing
    两边都兼容。"""
    current = e
    while getattr(current, "exceptions", None):
        current = current.exceptions[0]
    return f"{type(current).__name__}: {current}"


class _McpConnection:
    """一个 MCP 服务器的连接，整个生命周期（连接、工具调用、关闭）都跑在
    self._task 这一个专属后台 task 里，外部通过 asyncio.Queue 请求/响应，
    从不直接触碰 ClientSession 对象——避免见模块顶部注释里的跨 task 问题。"""

    def __init__(self, url: str) -> None:
        self.url = url
        self.ok = False
        self.error: Optional[str] = None
        self.tools: List[McpTool] = []
        self._request_q: "asyncio.Queue[Optional[tuple]]" = asyncio.Queue()
        self._ready = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())
        await self._ready.wait()

    async def _run(self) -> None:
        try:
            async with streamablehttp_client(self.url, timeout=MCP_CONNECT_TIMEOUT_SECONDS) as (
                read_stream, write_stream, _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    self.tools = listed.tools
                    self.ok = True
                    self._ready.set()
                    while True:
                        item = await self._request_q.get()
                        if item is None:
                            break
                        real_name, arguments, fut = item
                        try:
                            result = await session.call_tool(real_name, arguments or {})
                            if not fut.cancelled():
                                fut.set_result(result)
                        except BaseException as e:  # noqa: BLE001 — 见模块顶部注释
                            if not fut.cancelled():
                                fut.set_exception(e)
        except BaseException as e:
            self.ok = False
            self.error = _describe_connect_error(e)
        finally:
            self._ready.set()  # 连接失败也要放行 start()，不让它挂死等待

    async def call_tool(self, real_name: str, arguments: Dict[str, Any]) -> Any:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._request_q.put((real_name, arguments, fut))
        return await fut

    async def close(self) -> None:
        if self._task is None or self._task.done():
            return
        await self._request_q.put(None)
        try:
            await asyncio.wait_for(self._task, timeout=MCP_CONNECT_TIMEOUT_SECONDS)
        except Exception as e:
            logger.warning("关闭 MCP 连接 %s 时出错（忽略，不影响其它连接）：%s", self.url, e)


class McpToolset:
    """持有一批已连接的 MCP 服务器（每个一个 _McpConnection），直到 aclose()。"""

    def __init__(self) -> None:
        self._connections: List[_McpConnection] = []
        self.tools: List[ToolDef] = []
        self.dispatch: Dict[str, Callable] = {}
        self.connect_results: List[Dict[str, Any]] = []  # 给前端反馈用

    async def connect(self, urls: List[str]) -> None:
        for raw_url in urls:
            url = (raw_url or "").strip()
            if not url:
                continue
            conn = _McpConnection(url)
            await conn.start()
            if not conn.ok:
                logger.warning("MCP 服务器连接失败：%s（%s）", url, conn.error)
                self.connect_results.append({"url": url, "ok": False, "error": conn.error, "n_tools": 0})
                continue

            self._connections.append(conn)
            server_idx = len(self._connections)
            for t in conn.tools:
                # 前缀避免不同 MCP 服务器碰巧暴露同名工具时互相覆盖
                tool_name = f"mcp_{server_idx}_{t.name}"
                self.tools.append(ToolDef(
                    name=tool_name,
                    description=f"[MCP:{url}] {t.description or t.name}",
                    input_schema=t.inputSchema or {"type": "object", "properties": {}},
                ))
                self.dispatch[tool_name] = self._make_caller(conn, t.name)
            self.connect_results.append({"url": url, "ok": True, "error": None, "n_tools": len(conn.tools)})

    def _make_caller(self, conn: _McpConnection, real_name: str) -> Callable:
        # 返回形状跟 core/agent/tools.py 里原生工具的 (content, events) 约定对齐，
        # 而不是裸字符串——这样 AgentOrchestrator 的调度循环不需要为 MCP 工具
        # 特殊分叉处理，MCP 工具调用也能像原生工具一样在活动流里可见。
        async def _call(inp: Dict[str, Any]) -> tuple:
            result = await conn.call_tool(real_name, inp or {})
            parts = []
            for block in result.content:
                text = getattr(block, "text", None)
                parts.append(text if text is not None else str(block))
            text = "\n".join(parts) if parts else "（无返回内容）"
            if result.isError:
                raise RuntimeError(text)
            event = AgentEvent(kind="tool_result", text=f"🔌 MCP 工具「{real_name}」返回：{text[:200]}")
            return text, [event]
        return _call

    def summary_text(self) -> str:
        ok = [r for r in self.connect_results if r["ok"]]
        failed = [r for r in self.connect_results if not r["ok"]]
        total_tools = sum(r["n_tools"] for r in ok)
        lines = [f"🔌 MCP：成功接入 {len(ok)} 个服务器（共 {total_tools} 个工具）"
                 + (f"，{len(failed)} 个连接失败" if failed else "")]
        for r in failed:
            lines.append(f"  - {r['url']} 连接失败：{r['error']}")
        return "\n".join(lines)

    async def aclose(self) -> None:
        for conn in self._connections:
            await conn.close()
