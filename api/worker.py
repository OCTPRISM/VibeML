"""api/worker.py  -  异步后台训练 Worker（接入限流队列）"""
from __future__ import annotations
import asyncio, os
from datetime import datetime
from typing import List, Optional
from api.models import TaskStatus, TrainingSample
from api.store  import task_store
from api.queue  import job_queue

async def submit_training_job(
    task_id: str, description: str, examples: Optional[List[TrainingSample]] = None,
    api_key: str|None = None, max_iterations: int = 3, target_metric: float = 0.80,
    enable_phase2: bool = True, source_id: str = "default",
    llm_provider: str = "anthropic", llm_model: str|None = None, llm_base_url: str|None = None,
    model_backend: str = "sklearn",
    task_type: str = "classification", env_description: str|None = None,
    instruction_examples: Optional[List[dict]] = None, base_model_id: str|None = None,
    image_examples: Optional[List[dict]] = None, vlm_examples: Optional[List[dict]] = None,
    user_id=None,
) -> tuple[bool, str]:
    """提交到限流队列，返回 (成功, 原因)。task_type="rl" 时 examples 不使用，
    改用 env_description（RL 场景没有"训练样本"这个概念）；task_type="llm_finetune"
    时 examples 也不使用，改用 instruction_examples + base_model_id；
    task_type="image_classification" 用 image_examples（[{"image_path","label"}]）；
    task_type="vlm_generative" 用 vlm_examples（[{"image_path","prompt","reference_answer"}]，
    base_model_id 复用同一个字段传底座模型 id）。user_id 只在 llm_provider ==
    "system_managed" 时需要（配额计量用），其它 provider 传 None 即可。"""
    coro = _run_job(task_id, description, examples, api_key,
                    max_iterations, target_metric, enable_phase2,
                    llm_provider, llm_model, llm_base_url, model_backend, task_type, env_description,
                    instruction_examples, base_model_id, image_examples, vlm_examples, user_id)
    ok = await job_queue.submit(task_id, source_id, coro)
    if ok:
        task_store.update(task_id, status=TaskStatus.QUEUED)
    return ok, "ok" if ok else "队列已满或限流"

async def _run_job(task_id, description, examples, api_key,
                   max_iterations, target_metric, enable_phase2,
                   llm_provider="anthropic", llm_model=None, llm_base_url=None, model_backend="sklearn",
                   task_type="classification", env_description=None,
                   instruction_examples=None, base_model_id=None,
                   image_examples=None, vlm_examples=None, user_id=None):
    task_store.update(task_id, status=TaskStatus.RUNNING, started_at=datetime.utcnow())

    # 本机版离线宽限期（Phase 6）：只在 settings.is_desktop_build 时生效，网络版
    # 直接跳过（enforce_grace_period 内部自己判断）。挂在这里而不是只在前端隐藏
    # 按钮，是因为一个绕过 UI 直接打本地 API 的用户也应该被同样拦下——这道限制
    # 是本地 FastAPI 进程自己强制的，不是纯前端展示层的事。
    from api.accounts.offline_grace import RestrictedModeError, enforce_grace_period
    try:
        enforce_grace_period()
    except RestrictedModeError as e:
        task_store.update(task_id, status=TaskStatus.FAILED,
                         error=str(e), ended_at=datetime.utcnow())
        task_store.push_event(task_id, {"type": "error", "message": str(e)})
        return

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if llm_provider == "anthropic" and not key:
        task_store.update(task_id, status=TaskStatus.FAILED,
                         error="未提供 ANTHROPIC_API_KEY", ended_at=datetime.utcnow())
        task_store.push_event(task_id, {"type":"error","message":"未提供 API Key"})
        return
    def on_event(ev):
        task_store.update_progress(task_id, ev)
        task_store.push_event(task_id, ev)
    try:
        if task_type == "rl":
            result = await asyncio.to_thread(
                _train_rl_sync, key, description, env_description or "",
                max_iterations, f"./deploy/{task_id}", on_event, llm_provider, llm_model, llm_base_url,
                user_id, task_id)
        elif task_type == "llm_finetune":
            result = await asyncio.to_thread(
                _train_llm_ft_sync, key, description, instruction_examples or [], base_model_id,
                max_iterations, f"./deploy/{task_id}", on_event, llm_provider, llm_model, llm_base_url,
                user_id, task_id)
        elif task_type == "image_classification":
            result = await asyncio.to_thread(
                _train_vlm_cls_sync, key, description, image_examples or [], base_model_id,
                max_iterations, f"./deploy/{task_id}", on_event, llm_provider, llm_model, llm_base_url,
                user_id, task_id)
        elif task_type == "vlm_generative":
            result = await asyncio.to_thread(
                _train_vlm_gen_sync, key, description, vlm_examples or [], base_model_id,
                max_iterations, f"./deploy/{task_id}", on_event, llm_provider, llm_model, llm_base_url,
                user_id, task_id)
        else:
            examples_raw = [{"text": s.text, "label": s.label} for s in examples]
            result = await asyncio.to_thread(
                _train_sync, key, description, examples_raw,
                max_iterations, target_metric, enable_phase2,
                f"./deploy/{task_id}", on_event, llm_provider, llm_model, llm_base_url, model_backend,
                user_id, task_id)
    except Exception as e:
        result = {"status": "error", "message": str(e)}
    if result.get("status") == "error":
        task_store.update(task_id, status=TaskStatus.FAILED,
                         error=result.get("message"), ended_at=datetime.utcnow())
    else:
        trainer = result.pop("_trainer", None)
        task_store.update(task_id, status=TaskStatus.COMPLETED,
                         result=result, model=trainer, ended_at=datetime.utcnow())

def _train_sync(api_key, description, examples, max_iterations,
                target_metric, enable_phase2, deploy_dir, on_event,
                llm_provider="anthropic", llm_model=None, llm_base_url=None, model_backend="sklearn",
                user_id=None, task_id=None):
    from core.pipeline import run_pipeline
    return run_pipeline(api_key=api_key, description=description, examples=examples,
                       max_iterations=max_iterations, target_metric=target_metric,
                       enable_phase2=enable_phase2, deploy_dir=deploy_dir, on_event=on_event,
                       llm_provider=llm_provider, llm_model=llm_model, llm_base_url=llm_base_url,
                       model_backend=model_backend, user_id=user_id, task_id=task_id)

def _train_rl_sync(api_key, description, env_description, max_iterations,
                   deploy_dir, on_event, llm_provider="anthropic", llm_model=None, llm_base_url=None,
                   user_id=None, task_id=None):
    from core.rl_pipeline import run_rl_pipeline
    return run_rl_pipeline(api_key=api_key, description=description, env_description=env_description,
                          max_iterations=max_iterations, deploy_dir=deploy_dir, on_event=on_event,
                          llm_provider=llm_provider, llm_model=llm_model, llm_base_url=llm_base_url,
                          user_id=user_id, task_id=task_id)

def _train_llm_ft_sync(api_key, description, instruction_examples, base_model_id, max_iterations,
                       deploy_dir, on_event, llm_provider="anthropic", llm_model=None, llm_base_url=None,
                       user_id=None, task_id=None):
    from core.llm_ft_pipeline import run_llm_ft_pipeline
    return run_llm_ft_pipeline(api_key=api_key, description=description,
                              instruction_examples=instruction_examples, base_model_id=base_model_id,
                              max_iterations=max_iterations, deploy_dir=deploy_dir, on_event=on_event,
                              llm_provider=llm_provider, llm_model=llm_model, llm_base_url=llm_base_url,
                              user_id=user_id, task_id=task_id)

def _train_vlm_cls_sync(api_key, description, image_examples, encoder_id, max_iterations,
                        deploy_dir, on_event, llm_provider="anthropic", llm_model=None, llm_base_url=None,
                        user_id=None, task_id=None):
    from core.vlm_cls_pipeline import run_vlm_cls_pipeline
    return run_vlm_cls_pipeline(api_key=api_key, description=description,
                               image_examples=image_examples, encoder_id=encoder_id,
                               max_iterations=max_iterations, deploy_dir=deploy_dir, on_event=on_event,
                               llm_provider=llm_provider, llm_model=llm_model, llm_base_url=llm_base_url,
                               user_id=user_id, task_id=task_id)

def _train_vlm_gen_sync(api_key, description, vlm_examples, base_model_id, max_iterations,
                        deploy_dir, on_event, llm_provider="anthropic", llm_model=None, llm_base_url=None,
                        user_id=None, task_id=None):
    from core.vlm_gen_pipeline import run_vlm_gen_pipeline
    return run_vlm_gen_pipeline(api_key=api_key, description=description,
                               vlm_examples=vlm_examples, base_model_id=base_model_id,
                               max_iterations=max_iterations, deploy_dir=deploy_dir, on_event=on_event,
                               llm_provider=llm_provider, llm_model=llm_model, llm_base_url=llm_base_url,
                               user_id=user_id, task_id=task_id)
