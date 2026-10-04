"""api/routes/tasks.py  -  训练任务端点（含限流队列 + 部署反馈）"""
from __future__ import annotations
import asyncio, json, uuid
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from api.models import (CreateTaskRequest, PredictRequest, FeedbackRequest,
                        PredictResponse, TaskCreatedResponse, TaskStatus,
                        TaskStatusResponse, TaskResult, EpochSummary,
                        QueueStatsResponse, TrainingSample)
from api.store  import task_store
from api.queue  import job_queue
from api.worker import submit_training_job
from api.dataset_store import dataset_store

router = APIRouter(prefix="/api/tasks", tags=["Tasks"])

# ── REST ─────────────────────────────────────────────────────────────────────

@router.post("", response_model=TaskCreatedResponse, status_code=202)
async def create_task(req: CreateTaskRequest, request: Request):
    if req.examples is not None:
        examples = req.examples
    else:
        cached = dataset_store.get(req.dataset_ref)
        if not cached:
            raise HTTPException(404, f"数据集引用不存在或已过期：{req.dataset_ref}")
        examples = [TrainingSample(text=e["text"], label=e["label"]) for e in cached]

    task_id   = str(uuid.uuid4())
    source_id = request.client.host if request.client else "unknown"
    task_store.create(task_id)

    ok, reason = await submit_training_job(
        task_id=task_id, description=req.description, examples=examples,
        api_key=req.api_key, max_iterations=req.max_iterations,
        target_metric=req.target_metric, enable_phase2=req.enable_phase2,
        source_id=source_id, llm_provider=req.llm_provider, llm_model=req.llm_model,
        llm_base_url=req.llm_base_url, model_backend=req.model_backend)

    if not ok:
        task_store.update(task_id, status=TaskStatus.FAILED, error=reason)
        raise HTTPException(status_code=429, detail=reason)

    remaining = job_queue.rate_remaining(source_id)
    return TaskCreatedResponse(task_id=task_id, status=TaskStatus.QUEUED,
        created_at=datetime.utcnow(), ws_url=f"/api/tasks/{task_id}/stream",
        queue_position=job_queue.stats()["queued"])

@router.get("", response_model=List[TaskStatusResponse])
async def list_tasks():
    return [_to_response(r) for r in task_store.list_all()]

@router.get("/queue/stats", response_model=QueueStatsResponse, tags=["System"])
async def queue_stats():
    return QueueStatsResponse(**job_queue.stats())

@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task(task_id: str):
    r = task_store.get(task_id)
    if not r: raise HTTPException(404, f"Task '{task_id}' not found")
    return _to_response(r)

@router.get("/{task_id}/events")
async def get_task_events(task_id: str):
    """任务的全量事件历史，供前端刷新/切换会话后重放（reduceAll）恢复完整状态"""
    r = task_store.get(task_id)
    if not r: raise HTTPException(404, f"Task '{task_id}' not found")
    return r.event_log

@router.post("/{task_id}/predict", response_model=PredictResponse)
async def predict(task_id: str, req: PredictRequest):
    r = task_store.get(task_id)
    if not r: raise HTTPException(404, f"Task '{task_id}' not found")
    if r.status != TaskStatus.COMPLETED: raise HTTPException(409, f"Not completed ({r.status})")
    if r.model is None: raise HTTPException(410, "Model evicted from memory")
    preds = r.model.predict(req.texts)
    try: confs = r.model.predict_proba(req.texts)
    except Exception: confs = None
    return PredictResponse(predictions=preds, confidences=confs)

@router.post("/{task_id}/feedback")
async def deployment_feedback(task_id: str, req: FeedbackRequest):
    """Phase 2 部署后反馈：检查是否需要重新训练"""
    r = task_store.get(task_id)
    if not r: raise HTTPException(404, f"Task '{task_id}' not found")
    if r.status != TaskStatus.COMPLETED: raise HTTPException(409, "Task not completed")
    deploy_path = (r.result or {}).get("deploy_path", f"./deploy/{task_id}")
    from core.pipeline import check_feedback
    return check_feedback(deploy_path, req.production_metric, req.drift_threshold)

# ── WebSocket ─────────────────────────────────────────────────────────────────

@router.websocket("/{task_id}/stream")
async def stream_task(ws: WebSocket, task_id: str):
    r = task_store.get(task_id)
    if not r:
        await ws.close(code=1008, reason="Not found"); return
    await ws.accept()
    await ws.send_text(json.dumps({"type":"snapshot","task_id":task_id,
        "status":r.status.value,"progress":r.progress,"created_at":r.created_at.isoformat()}))
    if r.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
        await ws.send_text(json.dumps({"type":"finished" if r.status==TaskStatus.COMPLETED else "error",
            "result":r.result,"error":r.error}))
        await ws.close(); return
    try:
        while True:
            try:
                event = await asyncio.wait_for(r.events.get(), timeout=120.0)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type":"heartbeat"})); continue
            await ws.send_text(json.dumps(event, default=str))
            if event.get("type") in ("finished","error"): break
    except WebSocketDisconnect:
        pass
    finally:
        try: await ws.close()
        except Exception: pass

# ── helper ────────────────────────────────────────────────────────────────────

def _to_response(r) -> TaskStatusResponse:
    result = None
    if r.result and r.status == TaskStatus.COMPLETED:
        result = TaskResult(
            best_metric=r.result.get("best_metric",0),
            metric_name=r.result.get("metric_name","f1"),
            labels=r.result.get("labels",[]),
            domain=r.result.get("domain",""),
            n_samples=r.result.get("n_samples",0),
            epoch_history=[EpochSummary(**e) for e in r.result.get("epoch_history",[])],
            deploy_path=r.result.get("deploy_path"),
            feedback_baseline=r.result.get("feedback_baseline"),
        )
    return TaskStatusResponse(task_id=r.task_id, status=r.status, created_at=r.created_at,
        started_at=r.started_at, ended_at=r.ended_at, progress=r.progress,
        result=result, error=r.error)
