"""
api/routes/compute_profiles.py  -  用户配置外部计算资源（Slurm / Kubernetes）的端点

结构照抄 api/routes/api_tokens.py（薄路由，逻辑简单不单独拆 service 层）。
跟它一个关键的共同点：列表接口**不回传凭据本身**（kubeconfig / SSH 私钥），
只回传"配没配"的布尔值——配置面板里要改凭据就重新填一遍，不提供"查看当前值"。

连通性测试（POST /{id}/test）是真的去连集群，不是格式校验：
K8s 打 API Server 读版本 + 检查 Job 权限，Slurm SSH 上去跑 whoami + sinfo。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.accounts.db import get_db
from api.accounts.deps import get_current_user
from api.accounts.models_db import ComputeResourceProfile, User
from api.accounts.schemas import (
    ComputeProfileItem, ComputeProfileTestResult, CreateComputeProfileRequest,
)

router = APIRouter(prefix="/api/compute-profiles", tags=["Compute Resources"])


def _serialize(p: ComputeResourceProfile) -> ComputeProfileItem:
    return ComputeProfileItem(
        id=str(p.id), name=p.name, backend=p.backend,
        k8s_namespace=p.k8s_namespace, k8s_image=p.k8s_image,
        k8s_cpu=p.k8s_cpu, k8s_memory=p.k8s_memory, k8s_gpu=p.k8s_gpu,
        has_kubeconfig=bool(p.k8s_kubeconfig),
        slurm_host=p.slurm_host, slurm_port=p.slurm_port,
        slurm_username=p.slurm_username, slurm_partition=p.slurm_partition,
        slurm_workdir=p.slurm_workdir, slurm_time_limit=p.slurm_time_limit,
        has_ssh_key=bool(p.slurm_ssh_private_key),
        created_at=p.created_at.isoformat(),
        last_used_at=p.last_used_at.isoformat() if p.last_used_at else None,
        revoked=p.revoked_at is not None,
    )


def _get_owned(profile_id: str, user: User, db: Session) -> ComputeResourceProfile:
    try:
        pid = uuid.UUID(profile_id)
    except ValueError:
        raise HTTPException(404, "计算资源配置不存在")
    row = db.query(ComputeResourceProfile).filter_by(id=pid, user_id=user.id).one_or_none()
    if row is None:
        raise HTTPException(404, "计算资源配置不存在")
    return row


@router.post("", response_model=ComputeProfileItem, status_code=201)
def create_compute_profile(
    req: CreateComputeProfileRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # 每种后端各自的必填项在这里挡住——等到真正提交训练任务时才发现缺字段
    # 就太晚了（那时候用户已经等了一段时间，而且错误信息离配置界面很远）
    if req.backend == "kubernetes":
        if not (req.k8s_kubeconfig or "").strip():
            raise HTTPException(400, "Kubernetes 配置需要提供 kubeconfig 内容")
        if not (req.k8s_image or "").strip():
            raise HTTPException(400, "Kubernetes 配置需要指定训练用的容器镜像")
    else:
        for field, label in (("slurm_host", "登录节点地址"),
                             ("slurm_username", "SSH 用户名"),
                             ("slurm_ssh_private_key", "SSH 私钥")):
            if not (getattr(req, field) or "").strip():
                raise HTTPException(400, f"Slurm 配置需要提供{label}")

    row = ComputeResourceProfile(
        user_id=user.id, name=req.name, backend=req.backend,
        k8s_kubeconfig=req.k8s_kubeconfig, k8s_namespace=req.k8s_namespace,
        k8s_image=req.k8s_image, k8s_cpu=req.k8s_cpu, k8s_memory=req.k8s_memory,
        k8s_gpu=req.k8s_gpu,
        slurm_host=req.slurm_host, slurm_port=req.slurm_port,
        slurm_username=req.slurm_username, slurm_ssh_private_key=req.slurm_ssh_private_key,
        slurm_partition=req.slurm_partition, slurm_workdir=req.slurm_workdir,
        slurm_time_limit=req.slurm_time_limit,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _serialize(row)


@router.get("", response_model=list[ComputeProfileItem])
def list_compute_profiles(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (db.query(ComputeResourceProfile)
              .filter_by(user_id=user.id)
              .order_by(ComputeResourceProfile.created_at.desc()).all())
    return [_serialize(r) for r in rows]


@router.delete("/{profile_id}", status_code=204)
def revoke_compute_profile(
    profile_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    row = _get_owned(profile_id, user, db)
    if row.revoked_at is None:
        row.revoked_at = datetime.utcnow()
        db.commit()


@router.post("/{profile_id}/test", response_model=ComputeProfileTestResult)
async def test_compute_profile(
    profile_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    import asyncio

    row = _get_owned(profile_id, user, db)
    if row.revoked_at is not None:
        return ComputeProfileTestResult(ok=False, message="这个配置已被撤销，无法测试连接")

    # K8s 的 HTTP 调用和 Slurm 的 SSH 都是同步阻塞的，丢到线程里避免堵住事件循环
    # （跟 api/worker.py 用 asyncio.to_thread 跑训练是同一个理由）
    if row.backend == "kubernetes":
        from core.compute_backends import k8s_backend as backend
    else:
        from core.compute_backends import slurm_backend as backend

    try:
        ok, message, detail = await asyncio.to_thread(backend.test_connection, row)
    except ImportError as e:
        # kubernetes / paramiko 没装的情况——给出可执行的修复建议而不是裸的 ImportError
        pkg = "kubernetes" if row.backend == "kubernetes" else "paramiko"
        return ComputeProfileTestResult(
            ok=False, message=f"服务器上缺少 {pkg} 依赖包，无法连接{row.backend}集群",
            detail=f"pip install {pkg}（{e}）")
    except Exception as e:
        return ComputeProfileTestResult(ok=False, message=f"测试连接时出错：{e}")

    if ok:
        row.last_used_at = datetime.utcnow()
        db.commit()
    return ComputeProfileTestResult(ok=ok, message=message, detail=detail)
