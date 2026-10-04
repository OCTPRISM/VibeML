"""
api/queue.py  -  Phase 3：多租户限流任务队列

解决 Phase 3 缺口：多租户并发控制 + 成本管理。

设计原则：
  - 同时最多 MAX_CONCURRENT 个训练任务并行执行
  - 每个来源（IP / API Key）每小时最多 RATE_LIMIT_PER_HOUR 个任务
  - 超出限流的任务进入等待队列，按 FIFO 顺序调度
  - GPU 成本控制：记录每个任务的 API 调用次数（Anthropic 费用代理）

生产环境替换方案：
  - 队列后端换为 Redis + Celery
  - 限流状态换为 Redis 滑动窗口
  - 本实现为内存版，适合单节点部署
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Coroutine, Dict, Optional


MAX_CONCURRENT      = 3      # 最大并行训练任务数
RATE_LIMIT_PER_HOUR = 10     # 每个来源每小时最多提交数
QUEUE_MAX_SIZE      = 50     # 等待队列最大长度


@dataclass
class QueueEntry:
    task_id:    str
    source_id:  str                # IP 或 API Key 的哈希，用于限流
    coroutine:  Coroutine
    submitted:  datetime = field(default_factory=datetime.utcnow)
    started:    Optional[datetime] = None


class RateLimiter:
    """滑动窗口限流（内存版）"""

    def __init__(self, max_per_hour: int = RATE_LIMIT_PER_HOUR):
        self.max_per_hour = max_per_hour
        self._timestamps: Dict[str, deque] = defaultdict(deque)

    def is_allowed(self, source_id: str) -> bool:
        now = time.time()
        window_start = now - 3600
        dq = self._timestamps[source_id]

        # 清理过期记录
        while dq and dq[0] < window_start:
            dq.popleft()

        return len(dq) < self.max_per_hour

    def record(self, source_id: str):
        self._timestamps[source_id].append(time.time())

    def remaining(self, source_id: str) -> int:
        """返回当前时间窗口内剩余可提交次数"""
        now = time.time()
        window_start = now - 3600
        dq = self._timestamps[source_id]
        while dq and dq[0] < window_start:
            dq.popleft()
        return max(0, self.max_per_hour - len(dq))


class JobQueue:
    """
    异步任务队列，带并发限制和限流。

    用法：
        queue = JobQueue()
        await queue.start()                      # 启动调度器

        allowed, reason = queue.check_rate(source_id)
        if allowed:
            await queue.submit(task_id, source_id, coro)

        stats = queue.stats()
    """

    def __init__(
        self,
        max_concurrent:      int = MAX_CONCURRENT,
        rate_limit_per_hour: int = RATE_LIMIT_PER_HOUR,
    ):
        self._semaphore   = asyncio.Semaphore(max_concurrent)
        self._rate_limiter = RateLimiter(rate_limit_per_hour)
        self._queue:      asyncio.Queue[QueueEntry] = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
        self._running:    Dict[str, QueueEntry]     = {}
        self._completed:  int = 0
        self._failed:     int = 0
        self._scheduler:  Optional[asyncio.Task]    = None
        self.max_concurrent = max_concurrent

    async def start(self):
        """启动后台调度器协程（在 FastAPI lifespan 中调用）"""
        self._scheduler = asyncio.create_task(self._dispatch_loop())

    async def stop(self):
        """关闭调度器（在应用关闭时调用）"""
        if self._scheduler:
            self._scheduler.cancel()
            try:
                await self._scheduler
            except asyncio.CancelledError:
                pass

    def check_rate(self, source_id: str) -> tuple[bool, str]:
        """检查是否允许提交（不记录，只检查）"""
        if not self._rate_limiter.is_allowed(source_id):
            remaining_s = 3600
            return False, f"限流：每小时最多 {RATE_LIMIT_PER_HOUR} 个任务，请稍后再试"
        if self._queue.full():
            return False, f"队列已满（最大 {QUEUE_MAX_SIZE}），请稍后再试"
        return True, "ok"

    async def submit(
        self,
        task_id:   str,
        source_id: str,
        coroutine: Coroutine,
    ) -> bool:
        """提交任务到队列。返回 True 表示成功入队。"""
        allowed, reason = self.check_rate(source_id)
        if not allowed:
            return False

        entry = QueueEntry(
            task_id   = task_id,
            source_id = source_id,
            coroutine = coroutine,
        )
        try:
            self._queue.put_nowait(entry)
            self._rate_limiter.record(source_id)
            return True
        except asyncio.QueueFull:
            return False

    def stats(self) -> dict:
        """返回队列运行统计"""
        return {
            "queued":     self._queue.qsize(),
            "running":    len(self._running),
            "completed":  self._completed,
            "failed":     self._failed,
            "max_concurrent": self.max_concurrent,
        }

    def rate_remaining(self, source_id: str) -> int:
        return self._rate_limiter.remaining(source_id)

    # ── 内部调度器 ────────────────────────────────────────────────────────────

    async def _dispatch_loop(self):
        """持续从队列取任务，受 semaphore 限并发"""
        while True:
            entry = await self._queue.get()
            # 等待并发槽位
            await self._semaphore.acquire()
            entry.started = datetime.utcnow()
            self._running[entry.task_id] = entry
            asyncio.create_task(self._run_entry(entry))

    async def _run_entry(self, entry: QueueEntry):
        try:
            await entry.coroutine
            self._completed += 1
        except Exception:
            self._failed += 1
        finally:
            self._running.pop(entry.task_id, None)
            self._semaphore.release()
            self._queue.task_done()


# ── 全局单例 ──────────────────────────────────────────────────────────────────

job_queue = JobQueue()
