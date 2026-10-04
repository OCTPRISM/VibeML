"""
api/routes/exports.py  -  训练产物打包下载

两个出口，对应用户想做的两件不同的事：
  GET /api/tasks/{task_id}/export/model  → 权重+推理脚本+模型卡片（要部署/上传模型平台的）
  GET /api/tasks/{task_id}/export/code   → 代码（要归档/推 GitHub 的，不含权重）

鉴权沿用现状：task_id 是 UUID，拿到 UUID 就能访问，跟现有 /api/tasks/{id}/events
等端点的既有行为一致，这次不单独在这里发明一套新的权限模型。但**路径穿越防护
是必须的**（见 core/export_packager.py::resolve_deploy_dir）——这个端点拿
路径参数直接去拼文件系统路径，没有这层校验就是任意文件读取漏洞。
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from api.store import task_store
from core.export_packager import ExportError, build_code_package, build_model_package

router = APIRouter(prefix="/api/tasks", tags=["Exports"])


def _event_log_for(task_id: str) -> list:
    """训练事件历史用来生成 model_card / train.py 里的真实指标。任务可能已经被
    LRU 淘汰（api/store.py::MAX_TASKS）或者服务重启过——那种情况下 deploy 目录
    还在磁盘上，只是拿不到事件了，此时降级成"打包但不带指标"，比直接 404 有用。"""
    record = task_store.get(task_id)
    return list(record.event_log) if record else []


def _serve_zip(zip_path: Path, download_name: str) -> FileResponse:
    # BackgroundTask 保证响应发完之后再删临时文件——不能在 return 之前删（文件还没发出去），
    # 也不能不删（每次下载都在临时目录里留一份，累积下去会撑满磁盘）
    return FileResponse(
        path=str(zip_path), filename=download_name, media_type="application/zip",
        background=BackgroundTask(lambda: os.unlink(zip_path)),
    )


@router.get("/{task_id}/export/model")
def export_model_package(task_id: str):
    try:
        zip_path = build_model_package(task_id, _event_log_for(task_id))
    except ExportError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"打包失败：{e}")
    return _serve_zip(zip_path, f"model-{task_id[:8]}.zip")


@router.get("/{task_id}/export/code")
def export_code_package(task_id: str):
    try:
        zip_path = build_code_package(task_id, _event_log_for(task_id))
    except ExportError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"打包失败：{e}")
    return _serve_zip(zip_path, f"code-{task_id[:8]}.zip")
