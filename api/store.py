"""
api/store.py  -  内存任务存储（生产换 Redis）
"""
from __future__ import annotations
import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from api.models import TaskStatus

MAX_TASKS = 100
MAX_EVENT_LOG = 2000

@dataclass
class TaskRecord:
    task_id:    str
    status:     TaskStatus       = TaskStatus.PENDING
    created_at: datetime         = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime]  = None
    ended_at:   Optional[datetime]  = None
    progress:   Dict[str, Any]   = field(default_factory=dict)
    result:     Optional[Dict]   = None
    error:      Optional[str]    = None
    model:      Any              = None
    events:     asyncio.Queue    = field(default_factory=asyncio.Queue)
    event_log:  List[dict]       = field(default_factory=list)  # 全量事件历史，供前端重放会话

class TaskStore:
    def __init__(self):
        self._tasks: OrderedDict[str, TaskRecord] = OrderedDict()
        # push_event() 在真实训练时是从 asyncio.to_thread() 甩出去的子线程里调用的
        # （on_event 回调在 _train_sync 等同步流水线代码内部被直接调用）——子线程里
        # asyncio.get_event_loop() 在 Python 3.10+ 会抛 RuntimeError（"没有当前事件循环"），
        # 之前的 try/except 兜底在这种情况下会退化成直接对 asyncio.Queue 做跨线程
        # put_nowait，这是不安全的（Queue 内部通过 Future.set_result 唤醒等待方，
        # 从非 loop 线程调用不保证被及时/正确处理，实测会导致 WS 端的
        # record.events.get() 偶发性地再也收不到后续事件，界面看起来像"训练卡住了"，
        # 其实后端训练本身在正常跑）。启动时绑定真正在跑的主 loop，之后统一走
        # call_soon_threadsafe，不再依赖那个总是会命中的 except 分支。
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def create(self, task_id: str) -> TaskRecord:
        if len(self._tasks) >= MAX_TASKS:
            self._tasks.popitem(last=False)
        record = TaskRecord(task_id=task_id)
        self._tasks[task_id] = record
        return record

    def get(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)

    def update(self, task_id: str, **kwargs) -> Optional[TaskRecord]:
        r = self.get(task_id)
        if r:
            for k, v in kwargs.items():
                if hasattr(r, k): setattr(r, k, v)
        return r

    def list_all(self) -> List[TaskRecord]:
        return list(self._tasks.values())

    def push_event(self, task_id: str, event: dict):
        r = self.get(task_id)
        if not r: return
        r.event_log.append(event)
        if len(r.event_log) > MAX_EVENT_LOG:
            del r.event_log[0]
        if self._loop is not None:
            self._loop.call_soon_threadsafe(r.events.put_nowait, event)
        else:
            # bind_loop() 还没被调用过（比如脱离 FastAPI app 直接跑脚本/测试）——
            # 这种情况下调用方本来就保证是在同一个线程里，直接 put 是安全的
            r.events.put_nowait(event)

    def update_progress(self, task_id: str, event: dict):
        r = self.get(task_id)
        if not r: return
        t = event.get("type", "")
        if t == "task_parsed":
            r.progress.update({"labels": event.get("labels",[]), "domain": event.get("domain","")})
        elif t == "data_ready":
            r.progress.update({"n_samples": event.get("total_samples",0), "quality": event.get("quality_score",0)})
        elif t == "epoch_done":
            r.progress.update({"current_metric": event.get("val_metric",0),
                               "iteration": event.get("iteration",0), "epoch": event.get("epoch",0)})
        elif t == "finished":
            r.progress["best_metric"] = event.get("best_metric",0)

task_store = TaskStore()
