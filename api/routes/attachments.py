"""api/routes/attachments.py  -  聊天输入框"添加文件"附件上传

跟 api/routes/datasets.py 的图片上传（IMAGE_CLASSIFICATION/VLM_GENERATIVE 训练样本，
不回传浏览器）是两条完全独立的路径——这里的附件是给对话本身补充上下文用的：
文档（PDF/Word/Excel/Markdown/纯文本）提取出文本，随下一条消息一起喂给
ConversationOrchestrator（见 core/conversation/orchestrator.py::_compose_effective_text）；
图片（JPG/PNG/SVG/EPS）存盘后给一个可预览的 URL，只是聊天里的参考附件，不经过
任何训练样本流程——前端如果检测到当前正在展示图片分类/VLM 数据卡片，会改用
现成的 POST /api/datasets/upload-images，不会打到这个端点。

压缩包（POST .../archive）复用同一套"文档提取文本/图片给预览 URL"的处理逻辑
（_process_stored_file），只是文件来源从"用户直接上传"变成"从压缩包里安全解压出来"，
见 core/archive_extractor.py 的路径穿越/压缩炸弹/符号链接防护。
"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File

from api.models import AttachmentUploadResponse, ArchiveUploadResponse, SkippedArchiveEntry
from core.data_sources import UPLOAD_DIR
from core.attachment_extractor import (
    extract_text, TEXT_SUFFIXES, PDF_SUFFIXES, DOCX_SUFFIXES, XLSX_SUFFIXES,
    LEGACY_OFFICE_SUFFIXES,
)
from core.archive_extractor import extract_archive, matched_suffix

router = APIRouter(prefix="/api/attachments", tags=["Attachments"])

ATTACHMENT_DIR = UPLOAD_DIR / "attachments"
ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024   # 20MB——单个文档/图片共用一个上限，够绝大多数真实文件
# 压缩包总大小上限——跟单文件的 20MB 分开算，压缩包本身（解压前）允许更大，
# 支持一次选中多个压缩包一起上传，这个上限是"这次请求里所有压缩包加起来"的原始体积
MAX_ARCHIVE_REQUEST_TOTAL_BYTES = 200 * 1024 * 1024   # 200MB

DOCUMENT_SUFFIXES = TEXT_SUFFIXES + PDF_SUFFIXES + DOCX_SUFFIXES + XLSX_SUFFIXES + LEGACY_OFFICE_SUFFIXES
# 只列用户明确要求的四种图片格式（JPG/PNG/SVG/EPS）——不是 datasets.py 那条训练样本
# 上传路径用的 webp/bmp 白名单，两边各自维护，没必要强行统一
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".svg", ".eps")


def _process_stored_file(stored_path: Path, filename: str, suffix: str) -> AttachmentUploadResponse:
    """给定一个已经落盘在 ATTACHMENT_DIR 下的文件（直接上传或压缩包解压出来的），
    按后缀分流成文档提取文本 / 图片给预览 URL——单文件上传和压缩包解压出来的每个
    成员，最终都走这一个函数，行为完全一致。调用方保证 stored_path 已经在
    ATTACHMENT_DIR 内且文件名是 f"{attachment_id}{suffix}"。"""
    attachment_id = stored_path.stem
    content_size = stored_path.stat().st_size

    if suffix in IMAGE_SUFFIXES:
        content = stored_path.read_bytes()
        if suffix in (".jpg", ".jpeg", ".png"):
            import io
            from PIL import Image
            try:
                Image.open(io.BytesIO(content)).verify()
            except Exception:
                raise HTTPException(400, f"图片 {filename} 无法解析，可能已损坏或不是有效的图片文件")
        elif suffix == ".svg":
            import xml.etree.ElementTree as ET
            try:
                ET.fromstring(content)
            except ET.ParseError:
                raise HTTPException(400, f"SVG 文件 {filename} 不是合法的 XML，可能已损坏")
        return AttachmentUploadResponse(
            attachment_id=attachment_id, filename=filename, kind="image",
            size_bytes=content_size, url=f"/attachments/{stored_path.name}",
        )

    if suffix not in DOCUMENT_SUFFIXES:
        raise HTTPException(400, f"不支持的文件格式：{suffix or '(无后缀)'}"
                                 f"（支持 PDF/Word/Excel/Markdown/纯文本 和 {', '.join(IMAGE_SUFFIXES)} 图片）")

    extracted_text, extraction_error = extract_text(stored_path, suffix)
    return AttachmentUploadResponse(
        attachment_id=attachment_id, filename=filename, kind="document",
        size_bytes=content_size, extracted_text=extracted_text, extraction_error=extraction_error,
    )


@router.post("", response_model=AttachmentUploadResponse)
async def upload_attachment(file: UploadFile = File(...)):
    filename = file.filename or "未命名文件"
    suffix = Path(filename).suffix.lower()
    content = await file.read()
    if len(content) > MAX_ATTACHMENT_BYTES:
        raise HTTPException(413, f"文件超过 {MAX_ATTACHMENT_BYTES // (1024*1024)}MB 上限")
    if not content:
        raise HTTPException(400, "文件内容为空")

    attachment_id = uuid.uuid4().hex
    stored_path = ATTACHMENT_DIR / f"{attachment_id}{suffix}"
    stored_path.write_bytes(content)
    return _process_stored_file(stored_path, filename, suffix)


@router.post("/archive", response_model=ArchiveUploadResponse)
async def upload_archive(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "没有收到任何压缩包")

    # 逐个读取的同时累加总大小，一旦超限立刻拒绝——不需要等全部读完才发现超限，
    # 但为了给出准确的错误信息，还是把已读的内容留着（读取本身开销不大，
    # 压缩包上限本来就控制在 200MB）
    contents: List[tuple[UploadFile, bytes]] = []
    total_bytes = 0
    for f in files:
        data = await f.read()
        total_bytes += len(data)
        if total_bytes > MAX_ARCHIVE_REQUEST_TOTAL_BYTES:
            raise HTTPException(413, f"本次上传的压缩包总大小超过 {MAX_ARCHIVE_REQUEST_TOTAL_BYTES // (1024*1024)}MB 上限"
                                     f"（{'、'.join((ff.filename or '未命名') for ff in files)}）")
        contents.append((f, data))

    items: List[AttachmentUploadResponse] = []
    skipped: List[SkippedArchiveEntry] = []

    for f, data in contents:
        archive_filename = f.filename or "未命名压缩包"
        if matched_suffix(archive_filename) is None:
            skipped.append(SkippedArchiveEntry(archive=archive_filename, filename="(整个文件)",
                                               reason="不是支持的压缩包格式"))
            continue
        if not data:
            skipped.append(SkippedArchiveEntry(archive=archive_filename, filename="(整个文件)", reason="文件内容为空"))
            continue

        # 压缩包本身和解压出的临时文件都放在系统临时目录，处理完这一个压缩包就清掉——
        # 跟 ATTACHMENT_DIR 分开，避免半路失败时在正式附件目录里留下孤儿文件
        with tempfile.TemporaryDirectory(prefix="archive_upload_") as tmp:
            tmp_dir = Path(tmp)
            archive_path = tmp_dir / "archive"
            archive_path.write_bytes(data)
            extract_dir = tmp_dir / "extracted"

            extracted_paths, archive_skipped, error = extract_archive(archive_path, archive_filename, extract_dir)
            for name, reason in archive_skipped:
                skipped.append(SkippedArchiveEntry(archive=archive_filename, filename=name, reason=reason))
            if error:
                skipped.append(SkippedArchiveEntry(archive=archive_filename, filename="(整个文件)", reason=error))
                continue

            for extracted_path in extracted_paths:
                # extracted_path 内部是用 .resolve() 算出来的（core/archive_extractor.py
                # 的 zip-slip 防护要这么做）——macOS 上 tempfile 给的目录在 /var/folders/...，
                # 是指向 /private/var/folders/... 的符号链接，两边不 resolve() 成同一种
                # 表示形式的话 relative_to() 会报"不在子路径下"，这里统一按 resolve() 后比较
                member_name = str(extracted_path.relative_to(extract_dir.resolve()))
                suffix = Path(member_name).suffix.lower()
                if suffix not in IMAGE_SUFFIXES and suffix not in DOCUMENT_SUFFIXES:
                    skipped.append(SkippedArchiveEntry(archive=archive_filename, filename=member_name,
                                                       reason=f"不支持的文件格式：{suffix or '(无后缀)'}"))
                    continue
                attachment_id = uuid.uuid4().hex
                stored_path = ATTACHMENT_DIR / f"{attachment_id}{suffix}"
                shutil.move(str(extracted_path), str(stored_path))
                try:
                    items.append(_process_stored_file(stored_path, member_name, suffix))
                except HTTPException as e:
                    stored_path.unlink(missing_ok=True)
                    skipped.append(SkippedArchiveEntry(archive=archive_filename, filename=member_name,
                                                       reason=str(e.detail)))

    return ArchiveUploadResponse(items=items, skipped=skipped)
