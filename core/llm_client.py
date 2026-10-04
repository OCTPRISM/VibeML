"""
core/llm_client.py  -  统一 LLM 调用抽象

在此之前，core/ 下 5 个文件各自 `import anthropic` 并直接调用
`anthropic.Anthropic(...).messages.create(...)`，模型名字符串重复了 6 次，
完全没有 provider 抽象，无法接入本地模型。

这里提供一个最小的 LLMClient 接口：
    client.complete(system, user, max_tokens) -> str

三个实现：
  AnthropicClient        包一层现有的 anthropic SDK 调用（行为与之前完全一致）
  OllamaClient           调本地 Ollama 的 /api/chat，强制 format="json"
                         （本项目所有 LLM 调用点的 prompt 都要求 JSON 输出，
                         强制 json 模式比正则兜底更可靠，Ollama 原生 API 支持这个参数）
  OpenAICompatibleClient 调任意实现了 OpenAI /v1/chat/completions 协议的本地/自建
                         服务（vLLM 的 OpenAI 兼容 server、LM Studio 等）——不强制
                         response_format（不是所有后端都支持），复用各调用点已有的
                         core/json_extract.py 正则兜底解析
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union


class SystemManagedNotConfiguredError(Exception):
    """system_managed 需要服务器配置 ANTHROPIC_SYSTEM_API_KEY——单独的异常类型
    （不是 ValueError），因为这是服务器侧配置缺失（该映射成 503），跟"用户没填
    必填字段"（400）或"没登录"（401）性质不同，不能被这几种情况共用的错误处理
    分支顺手接住、映射错状态码。"""


# ── 工具调用抽象（Multi-Agent 模式专用，.complete() 完全不受影响）──────────────
#
# 三个 provider 的工具调用 wire format 互不相同（Anthropic 原生 tool_use/tool_result
# content block；Ollama 原生 /api/chat 的 tool_calls 没有 id 字段、arguments 是
# object；OpenAI 协议的 tool_calls 有 id、arguments 是 JSON 字符串）。AgentMessage
# 是这三者的公共交集——core/agent/ 下的 Agent 循环只操作 AgentMessage，从不知道
# 自己在跟哪个 provider 说话；每个 complete_with_tools() 实现自己负责把 AgentMessage
# 列表翻译成自家 wire format、再把响应翻译回 CompletionResult。这样 AgentMessage
# 列表可以直接存进 ConversationState 里跨消息续跑，不需要关心当时用的是哪个 provider。

@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class AgentMessage:
    role: Literal["user", "assistant", "tool"]
    content: Union[str, List[ToolCall], List[ToolResult]]


@dataclass
class CompletionResult:
    stop_reason: Literal["tool_use", "end_turn"]
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)


class LLMClient:
    """LLM 调用统一接口。

    last_usage：上一次 complete()/complete_with_tools() 调用的 token 用量
    （{"prompt_tokens","completion_tokens","total_tokens"}），成功调用后由各子类
    实现填充，默认 None（比如调用失败时）。这是为账号体系的按次计量/token 审计
    清单（api/accounts/quota_gated_client.py）新增的，complete() 本身的签名/
    返回值不变——各 provider 的响应里本来就带这些数字，之前只是没读。
    """

    last_usage: Optional[Dict[str, int]] = None

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        raise NotImplementedError

    async def complete_with_tools(
        self,
        messages: List[AgentMessage],
        tools: List[ToolDef],
        system: str = "",
        max_tokens: int = 1000,
    ) -> CompletionResult:
        """Multi-Agent 模式专用：单轮工具调用请求。messages 是到目前为止的完整
        对话历史（Agent 自己维护，不是 core/conversation/ 那套 ConversationState.messages）。
        不支持工具调用的实现应当 raise NotImplementedError，而不是静默退化。

        真正的 async（不是同步方法套一层 async def 的假异步）——底层用各 provider
        的异步 HTTP 客户端（httpx.AsyncClient / anthropic.AsyncAnthropic），这样
        core/agent/subagent.py 用 asyncio.gather 摆在一起的多个子 Agent 在等待
        网络响应期间真的能交替执行，不是"同一时刻只有一个在跑网络请求"的假并发。
        （.complete() 本身保持同步不变——它的十几个调用点都是普通同步函数，
        全部改成 async 是一次影响全仓库的改动，超出这次要解决的具体问题范围，
        这个事实性的限制已经如实记录，不在这里顺手做掉。）"""
        raise NotImplementedError


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        # complete_with_tools() 是真正的 async 方法，不能借用上面那个同步 client——
        # 这里单独建一个官方异步客户端，两者共享同一个 api_key，互不影响
        self._async_client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self.last_usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        return response.content[0].text.strip()

    async def complete_with_tools(
        self, messages: List[AgentMessage], tools: List[ToolDef],
        system: str = "", max_tokens: int = 1000,
    ) -> CompletionResult:
        anthropic_messages = []
        for m in messages:
            if m.role == "user" and isinstance(m.content, str):
                anthropic_messages.append({"role": "user", "content": m.content})
            elif m.role == "assistant" and isinstance(m.content, str):
                anthropic_messages.append({"role": "assistant", "content": m.content})
            elif m.role == "assistant":
                # List[ToolCall] —— 上一轮 Claude 自己发起的工具调用，原样回放
                anthropic_messages.append({"role": "assistant", "content": [
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                    for tc in m.content
                ]})
            elif m.role == "tool":
                # List[ToolResult] —— Anthropic 协议里 tool_result 是放在 user 消息里的
                anthropic_messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tr.tool_call_id,
                     "content": tr.content, "is_error": tr.is_error}
                    for tr in m.content
                ]})

        response = await self._async_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=anthropic_messages,
            tools=[{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                   for t in tools],
        )
        self.last_usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        text_parts, tool_calls = [], []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))
        stop_reason = "tool_use" if response.stop_reason == "tool_use" else "end_turn"
        return CompletionResult(stop_reason=stop_reason, text="".join(text_parts).strip(),
                                 tool_calls=tool_calls)


class OllamaClient(LLMClient):
    def __init__(self, model: str = "qwen3.6:35b-a3b",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        import httpx
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "format": "json",
                "think": False,  # qwen3.6 是思考模型，不关闭的话推理过程会占满 num_predict 导致 content 为空
                "options": {"num_predict": max_tokens},
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return data["message"]["content"].strip()

    async def complete_with_tools(
        self, messages: List[AgentMessage], tools: List[ToolDef],
        system: str = "", max_tokens: int = 1000,
    ) -> CompletionResult:
        import httpx

        # Ollama 原生 /api/chat 的 tool_calls 没有 id 字段、arguments 是 object
        # 不是 JSON 字符串——跟标准 OpenAI 协议（OpenAICompatibleClient 那边）不一样，
        # 这里按 Ollama 自己的格式构造/解析，不能直接照抄 OpenAI 的写法
        chat_messages = [{"role": "system", "content": system}]
        for m in messages:
            if isinstance(m.content, str):
                chat_messages.append({"role": m.role, "content": m.content})
            elif m.role == "assistant":
                chat_messages.append({"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": tc.name, "arguments": tc.input}} for tc in m.content
                ]})
            elif m.role == "tool":
                for tr in m.content:
                    chat_messages.append({"role": "tool", "content": tr.content})

        async with httpx.AsyncClient() as http_client:
            resp = await http_client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": chat_messages,
                    "stream": False,
                    "think": False,
                    "options": {"num_predict": max_tokens},
                    "tools": [{"type": "function", "function": {
                        "name": t.name, "description": t.description, "parameters": t.input_schema,
                    }} for t in tools],
                },
                timeout=120,
            )
        resp.raise_for_status()
        data = resp.json()
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        self.last_usage = {
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        message = data.get("message", {})
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = []
        for i, raw in enumerate(raw_tool_calls):
            fn = raw.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                # 不是所有本地模型/Ollama 版本都严格遵守"arguments 是 object"，
                # 遇到字符串形式的也兼容解析，避免因为个别模型行为差异直接崩掉
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            tool_calls.append(ToolCall(id=f"call_{i}", name=fn.get("name", ""), input=args))
        stop_reason = "tool_use" if tool_calls else "end_turn"
        return CompletionResult(stop_reason=stop_reason,
                                 text=(message.get("content") or "").strip(),
                                 tool_calls=tool_calls)


class OpenAICompatibleClient(LLMClient):
    """任意实现了 OpenAI /v1/chat/completions 协议的本地或自建服务
    （vLLM 的 OpenAI 兼容 server、LM Studio、text-generation-webui 等）。
    base_url 由用户显式提供（不像 Ollama 有约定俗成的默认地址，本地/自建服务
    地址因人而异，没有合理默认值可以兜底）。"""

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        import httpx
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        self.last_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        return data["choices"][0]["message"]["content"].strip()

    async def complete_with_tools(
        self, messages: List[AgentMessage], tools: List[ToolDef],
        system: str = "", max_tokens: int = 1000,
    ) -> CompletionResult:
        import httpx

        # 标准 OpenAI 协议：tool_calls 有 id、arguments 是 JSON 字符串——
        # 跟 Ollama 原生 /api/chat 格式不同，两边各自实现，不共用一份翻译逻辑
        chat_messages = [{"role": "system", "content": system}]
        for m in messages:
            if isinstance(m.content, str):
                chat_messages.append({"role": m.role, "content": m.content})
            elif m.role == "assistant":
                chat_messages.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.input)}}
                    for tc in m.content
                ]})
            elif m.role == "tool":
                for tr in m.content:
                    chat_messages.append({
                        "role": "tool", "tool_call_id": tr.tool_call_id, "content": tr.content,
                    })

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": chat_messages,
                    "max_tokens": max_tokens,
                    "tools": [{"type": "function", "function": {
                        "name": t.name, "description": t.description, "parameters": t.input_schema,
                    }} for t in tools],
                },
                timeout=120,
            )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        self.last_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        message = data["choices"][0]["message"]
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = []
        for raw in raw_tool_calls:
            fn = raw.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=raw.get("id", ""), name=fn.get("name", ""), input=args))
        stop_reason = "tool_use" if tool_calls else "end_turn"
        return CompletionResult(stop_reason=stop_reason,
                                 text=(message.get("content") or "").strip(),
                                 tool_calls=tool_calls)


def build_llm_client(
    provider: str = "anthropic",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMClient:
    """根据 provider 构建对应的 LLMClient。
    base_url 只对 ollama / openai_compatible 有意义——Ollama 有默认地址可以兜底，
    openai_compatible 没有通用默认值，必须显式提供。openai_compatible 不等于"本地
    服务"：它是 OpenAI 的 /v1/chat/completions 协议本身，vLLM/LM Studio 等自建服务
    说这个协议，OpenAI 官方、Azure OpenAI 等商业 API 也是同一套协议，本地还是
    商业只取决于这里传入的 base_url，这个函数不对此做任何假设。"""
    if provider == "ollama":
        return OllamaClient(
            model=model or os.environ.get("OLLAMA_MODEL", "qwen3.6:35b-a3b"),
            base_url=base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    if provider == "openai_compatible":
        if not base_url:
            raise ValueError("使用本地/自建 OpenAI 兼容服务需要提供服务地址"
                             "（比如 vLLM 的 http://localhost:8000/v1）")
        if not model:
            raise ValueError("使用本地/自建 OpenAI 兼容服务需要提供模型名称"
                             "（服务启动时注册的模型名，比如 vLLM 的 --served-model-name）")
        return OpenAICompatibleClient(base_url=base_url, model=model, api_key=api_key)
    if provider == "system_managed":
        # 系统托管的商业 Key（配额网关见 api/accounts/quota_gated_client.py）——走的是
        # 一模一样的 Anthropic 协议，复用现成的 AnthropicClient，不需要新 client 类。
        # 这里只负责"用哪个 key"，完全不知道用户/配额这些概念（那是 api/accounts/ 的事）。
        from api.settings import settings as _settings
        if not _settings.anthropic_system_api_key:
            # 没配置系统 key 时必须在这里就报清楚的错，而不是让请求打到 Anthropic SDK
            # 内部去抛一个"authentication method"相关的原始报错——那种报错信息对用户
            # 没有意义，也分不清是我们没配置还是他们自己的 Key 有问题
            raise SystemManagedNotConfiguredError(
                "系统托管的商业 API 尚未配置（缺少 ANTHROPIC_SYSTEM_API_KEY），"
                "请联系管理员配置后再试，或选择本地/自建服务或填自己的 API Key。")
        return AnthropicClient(
            api_key=_settings.anthropic_system_api_key,
            model=model or _settings.system_managed_default_model,
        )
    return AnthropicClient(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
        model=model or "claude-sonnet-4-6",
    )
