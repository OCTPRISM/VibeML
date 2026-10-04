"""api/conversation_store.py  -  内存会话存储（结构照抄 api/store.py::TaskStore）"""
from __future__ import annotations

import asyncio
import copy
from collections import OrderedDict
from typing import Dict, List, Optional

from core.conversation.state import ConversationState, ConversationMessage

MAX_CONVERSATIONS = 100


class ConversationStore:
    def __init__(self):
        self._conversations: "OrderedDict[str, ConversationState]" = OrderedDict()
        self._queues: "OrderedDict[str, asyncio.Queue]" = OrderedDict()
        # 会话按用户的反查索引——只用来支持"我的会话列表"这个查询，不是主存储
        # （主存储永远是 _conversations，按 conversation_id 查）。user_id 在会话
        # 创建时未必已知（路由层先 create() 拿到 state，再单独设置 state.user_id），
        # 所以这里不在 create()/fork() 里自动建索引，调用方设置完 user_id 后
        # 显式调 index_for_user()。
        self._by_user: Dict[str, List[str]] = {}
        # 跟 api/store.py::TaskStore 同样的原因绑定真正在跑的主 loop——push_live()
        # 目前的调用点都在 asyncio.create_task() 里所以本来是安全的，但和
        # TaskStore 一样的 try/except RuntimeError 兜底模式本身是脆弱的（子线程里
        # asyncio.get_event_loop() 必定抛错，会静默退化成跨线程直接 put_nowait），
        # 统一绑定 loop 消除这个隐患，即使以后有新的调用点从线程池里调这里也不会中招
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def create(self, conversation_id: str) -> ConversationState:
        if len(self._conversations) >= MAX_CONVERSATIONS:
            oldest_id, _ = self._conversations.popitem(last=False)
            self._queues.pop(oldest_id, None)
            self._deindex(oldest_id)
        state = ConversationState(conversation_id=conversation_id)
        self._conversations[conversation_id] = state
        self._queues[conversation_id] = asyncio.Queue()
        return state

    def get(self, conversation_id: str) -> Optional[ConversationState]:
        return self._conversations.get(conversation_id)

    def delete(self, conversation_id: str) -> bool:
        existed = conversation_id in self._conversations
        self._conversations.pop(conversation_id, None)
        self._queues.pop(conversation_id, None)
        self._deindex(conversation_id)
        return existed

    def index_for_user(self, conversation_id: str, user_id: str) -> None:
        """路由层设置完 state.user_id 之后调用——create()/fork() 两条创建路径都要调，
        这样"我的会话列表"（GET /api/conversations）才能同时看到网页和 API 发起的会话。"""
        ids = self._by_user.setdefault(user_id, [])
        if conversation_id not in ids:
            ids.append(conversation_id)

    def _deindex(self, conversation_id: str) -> None:
        """LRU 淘汰/主动删除时，同步从 _by_user 摘掉，避免野指针（列表里存在
        但 _conversations 里已经找不到的 conversation_id）。不知道属于哪个
        user_id 也没关系——遍历一遍所有 user 的列表，摘掉就行（列表数量级很小，
        不值得为这个反向维护 conversation_id -> user_id 的第二份索引）。"""
        for ids in self._by_user.values():
            if conversation_id in ids:
                ids.remove(conversation_id)

    def list_for_user(self, user_id: str) -> List[ConversationState]:
        ids = self._by_user.get(user_id, [])
        return [self._conversations[cid] for cid in ids if cid in self._conversations]

    def fork(self, source_id: str, new_id: str) -> Optional[ConversationState]:
        """深拷贝 source_id 当前的完整状态到 new_id，两个会话之后各自独立推进，
        互不影响——ConversationState 全是原始类型/列表/字典/嵌套 dataclass
        （没有锁、文件句柄等不可深拷贝的对象），copy.deepcopy 直接可用，不需要
        手写逐字段拷贝。task_id 也一并拷贝：如果源会话训练已经跑完，分支直接
        显示同样的（只读）结果；如果还在训练中，分支会看到同一个真实后台任务
        的实时进度——这就是同一个任务本身，分叉的是"对话历史"，不是"训练任务"，
        分支里之后如果重新提交训练会拿到全新的 task_id，不会和原会话冲突。"""
        source = self._conversations.get(source_id)
        if source is None:
            return None
        forked = copy.deepcopy(source)
        forked.conversation_id = new_id
        if len(self._conversations) >= MAX_CONVERSATIONS:
            oldest_id, _ = self._conversations.popitem(last=False)
            self._queues.pop(oldest_id, None)
            self._deindex(oldest_id)
        self._conversations[new_id] = forked
        self._queues[new_id] = asyncio.Queue()
        # forked.user_id 是深拷贝带过来的（跟源会话同一个所有者），直接建索引，
        # 不用等路由层再显式调一次 index_for_user
        if forked.user_id:
            self.index_for_user(new_id, str(forked.user_id))
        return forked

    def push_live(self, conversation_id: str, message: ConversationMessage) -> None:
        queue = self._queues.get(conversation_id)
        if queue is None:
            return
        if self._loop is not None:
            self._loop.call_soon_threadsafe(queue.put_nowait, message)
        else:
            queue.put_nowait(message)

    async def next_live(self, conversation_id: str, timeout: float) -> Optional[ConversationMessage]:
        queue = self._queues.get(conversation_id)
        if queue is None:
            return None
        return await asyncio.wait_for(queue.get(), timeout=timeout)

    def drain_live(self, conversation_id: str) -> None:
        """清空积压——调用方必须保证队列里的每条消息都已经在 state.messages 里有了
        （push_live 的所有调用点都是先 append 到 state.messages 再 push，这个前提恒成立），
        否则会丢消息。用于新 WS 连接接管一个可能有陈旧积压的旧队列时，避免快照之后重复推送。"""
        queue = self._queues.get(conversation_id)
        if queue is None:
            return
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break


conversation_store = ConversationStore()
