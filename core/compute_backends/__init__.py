"""
core/compute_backends/  -  把训练任务提交到外部计算资源（Kubernetes / Slurm）

设计原则（和 core/subprocess_runner.py 的本机子进程隔离是平行关系，不是替换）：
  - 训练逻辑本身一行不改：远端跑的还是 core/*_pipeline.py 里那套同样的
    run_*_pipeline，只是执行位置从"API 服务器的线程池"换成"集群节点"。
  - 每个后端只负责三件事：submit（提交并返回作业标识）、poll_status（查状态）、
    cancel（取消）。进度事件怎么传回来是 remote_worker/run_remote_job.py 和
    api/routes/tasks.py 的 remote-events 回调端点的事，不在后端模块里。
  - 连接失败/认证失败/权限不足要给出可区分的、人能看懂的错误，不是笼统的"失败"。
"""
from __future__ import annotations

from typing import Protocol


class ComputeBackendError(Exception):
    """后端操作失败——message 直接面向用户展示，要写人话。"""


class ComputeBackend(Protocol):
    """两个后端实现共同遵守的形状。故意用 Protocol 而不是抽象基类：
    这两个实现之间没有可共享的实现细节（一个走 HTTP REST，一个走 SSH），
    继承一个空基类只是徒增一层，没有实际收益。"""

    def test_connection(self, profile) -> tuple[bool, str, str | None]:
        """返回 (ok, message, detail)。不抛异常——连不上本身就是这个方法要报告的结果。"""
        ...
