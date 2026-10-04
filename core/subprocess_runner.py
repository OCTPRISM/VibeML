"""
core/subprocess_runner.py  -  通用子进程隔离执行器

从 core/nn_trainer.py 抽出来的"跑起来安不安全"逻辑，让后续新增的训练器
（RL / LLM 微调 / VLM 等）都复用同一套安全机制，不用重复实现第二遍：
  - 每次训练都在独立子进程（multiprocessing, spawn）里跑，父进程用
    proc.join(timeout=...) + terminate()/kill() 兜底卡死的生成代码——
    线程杀不掉，进程才有真正的 terminate 能力，这是选子进程而不是线程/纯 exec 的原因。
    这个墙钟超时是最后一道防线：不管子进程内部在执行什么（哪怕是任意生成代码的死循环），
    到点就会被外部杀掉，不依赖被执行代码本身"配合"退出。
  - 子进程只把可 pickle 的结果（dict）送回父进程，任何异常都被子进程捕获成
    {"error": ...} 送回，不会带着未处理异常静默消失。
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Any, Callable, Dict


class SubprocessTrainingError(Exception):
    """子进程训练失败（超时 / OOM / 运行时异常），携带简短原因供调用方决定降级"""


def run_in_subprocess(
    payload: Dict[str, Any],
    entrypoint: Callable[[Dict[str, Any]], Dict[str, Any]],
    timeout: int,
    error_cls: type = SubprocessTrainingError,
) -> Dict[str, Any]:
    """
    在隔离子进程里跑 entrypoint(payload) -> dict，超时/子进程崩溃/内部异常统一转成 error_cls。

    entrypoint 必须是模块级函数（spawn 上下文要求参数可 pickle，不能是闭包/局部函数/lambda）。
    error_cls 让调用方（core/nn_trainer.py 等）保持自己原有的异常类型不变，
    只共用这一套子进程机制本身。
    """
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_child_entrypoint, args=(entrypoint, payload, result_queue))
    proc.start()

    # 关键：必须先从 queue 里取结果，再 join 进程——顺序不能反过来。如果子进程往
    # queue 里塞的对象（比如 RL 策略权重、大模型 state_dict）超过操作系统管道缓冲区
    # 大小，子进程内部负责把对象写进管道的 feeder 线程会阻塞在 write() 上，等父进程
    # 来读；但如果父进程先 proc.join() 等子进程退出，子进程又要等 feeder 线程写完
    # 才能真正退出——两边互相等对方，形成经典死锁。这是 multiprocessing 官方文档
    # 明确警告过的用法陷阱（"你必须在 join 之前处理完 queue 里的所有条目"），
    # 之前的实现顺序反了，在小 payload（分类头 state_dict 较小）时侥幸没触发，
    # 但 RL policy 的序列化体积更大，稳定触发死锁——本函数曾经的实现就是这个反序。
    try:
        result = result_queue.get(timeout=timeout)
    except Exception:
        # 到这里说明子进程还没在预算时间内产出结果——可能仍在训练（超时），
        # 也可能已经崩溃退出且什么都没塞进 queue，两种情况对调用方来说都是同一种失败
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join(2)
            raise error_cls(f"训练超时（超过 {timeout} 秒），已终止子进程")
        proc.join(2)
        raise error_cls(f"子进程异常退出（exitcode={proc.exitcode}），未返回任何结果")

    proc.join(5)   # 结果已经拿到，子进程正常情况下会很快自然退出，给个宽限期收尾
    if proc.is_alive():
        proc.terminate()
        proc.join(2)

    if result.get("error"):
        raise error_cls(result["error"])
    return result


def _child_entrypoint(entrypoint: Callable, payload: Dict[str, Any], result_queue) -> None:
    """独立子进程入口——任何异常都被捕获并作为 {"error": ...} 送回，不让子进程带着未处理异常静默消失"""
    try:
        result = entrypoint(payload)
    except Exception:
        result = {"error": traceback.format_exc(limit=6)}
    result_queue.put(result)
