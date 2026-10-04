"""api/routes/datasets.py  -  数据集接入端点：本地上传 / 本地路径 / HF Hub / 魔搭 搜索与拉取"""
from __future__ import annotations
import asyncio
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Query

from api.models import (DatasetSearchResult, DatasetUploadResponse,
                        DatasetPreviewRequest, DatasetPreviewResponseModel,
                        DatasetFetchRequest, DatasetFetchResponse,
                        ImageUploadItem, ImageUploadResponse)
from api.dataset_store import dataset_store
from core.data_sources import get_source, UPLOAD_DIR

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024   # 50MB
ALLOWED_SUFFIXES = (".csv", ".tsv", ".json", ".jsonl", ".ndjson")

# IMAGE_CLASSIFICATION / VLM_GENERATIVE 任务专用的图片上传——独立子目录，
# 不跟表格文件混在一起（避免后面按后缀扫描目录时误把图片当成表格文件处理）
IMAGE_UPLOAD_DIR = UPLOAD_DIR / "images"
IMAGE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_IMAGE_BYTES = 10 * 1024 * 1024    # 单张图片 10MB 上限
MAX_IMAGES_PER_UPLOAD = 200
ALLOWED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@router.get("/search", response_model=List[DatasetSearchResult])
async def search_datasets(platform: str = Query(...), query: str = Query(..., min_length=1),
                          page: int = Query(1, ge=1)):
    if platform not in ("huggingface", "modelscope"):
        raise HTTPException(400, "search 仅支持 huggingface / modelscope")
    try:
        # get_source(...).search() 是同步方法，真实发起 HF/魔搭网络请求——直接
        # 在这个 async handler 里同步调用会阻塞整个 uvicorn 事件循环，拖慢同一
        # 进程里其它所有并发请求（真实复现过：一次真实调用卡住期间，完全无关
        # 的 /api/tasks/queue/stats 健康检查也会跟着没有响应超过一分钟）。用
        # asyncio.to_thread 丢进线程池执行，不改 DataSource 本身的同步实现。
        results = await asyncio.to_thread(get_source(platform).search, query, page=page, page_size=10)
    except Exception as e:
        raise HTTPException(502, f"搜索失败：{e}")
    return [DatasetSearchResult(**r.__dict__) for r in results]


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"不支持的文件类型：{suffix or '(无后缀)'}（支持 {', '.join(ALLOWED_SUFFIXES)}）")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"文件超过 {MAX_UPLOAD_BYTES // (1024*1024)}MB 上限")
    ref = f"{uuid.uuid4().hex}{suffix}"
    (UPLOAD_DIR / ref).write_bytes(content)
    return DatasetUploadResponse(ref=ref, filename=file.filename or ref, size_bytes=len(content))


@router.post("/upload-images", response_model=ImageUploadResponse)
async def upload_images(files: List[UploadFile] = File(...)):
    """IMAGE_CLASSIFICATION / VLM_GENERATIVE 任务的图片上传——多文件一次性上传，
    每张图存到 IMAGE_UPLOAD_DIR 下、返回可引用的 ref 列表；前端拿到 ref 后再
    自己给每张图配标签（分类任务）或问题+参考答案（生成式任务），最终作为
    image_examples/vlm_examples 结构化数据发给 prepare_data 阶段——这里只管
    "把图片存下来"，不关心图片将来怎么被打标，跟文本文件上传端点职责一致。"""
    if len(files) > MAX_IMAGES_PER_UPLOAD:
        raise HTTPException(400, f"一次最多上传 {MAX_IMAGES_PER_UPLOAD} 张图片")
    if not files:
        raise HTTPException(400, "没有收到任何文件")

    from PIL import Image
    import io

    items: List[ImageUploadItem] = []
    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise HTTPException(400, f"不支持的图片格式：{suffix or '(无后缀)'}"
                                     f"（支持 {', '.join(ALLOWED_IMAGE_SUFFIXES)}）")
        content = await file.read()
        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"图片 {file.filename} 超过 {MAX_IMAGE_BYTES // (1024*1024)}MB 上限")
        try:
            # 落盘前先真的用 PIL 打开一次校验不是损坏/伪装成图片的文件——
            # 训练阶段才发现图片打不开会浪费一整轮迭代，这里提前挡掉
            Image.open(io.BytesIO(content)).verify()
        except Exception:
            raise HTTPException(400, f"图片 {file.filename} 无法解析，可能已损坏或不是有效的图片文件")

        ref = f"images/{uuid.uuid4().hex}{suffix}"
        (UPLOAD_DIR / ref).write_bytes(content)
        items.append(ImageUploadItem(ref=ref, filename=file.filename or ref, size_bytes=len(content)))

    return ImageUploadResponse(images=items)


@router.post("/preview", response_model=DatasetPreviewResponseModel)
async def preview_dataset(req: DatasetPreviewRequest):
    try:
        preview = await asyncio.to_thread(
            get_source(req.platform).preview, req.ref, split=req.split, config=req.config)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, f"预览失败：{e}")
    return DatasetPreviewResponseModel(**preview.__dict__)


@router.post("/fetch", response_model=DatasetFetchResponse)
async def fetch_dataset(req: DatasetFetchRequest):
    try:
        examples = await asyncio.to_thread(
            get_source(req.platform).fetch, req.ref, req.text_col, req.label_col,
            max_samples=req.max_samples, split=req.split, config=req.config)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, f"拉取失败：{e}")

    if len(examples) < 5:
        raise HTTPException(422, f"有效样本不足 5 条（实际 {len(examples)} 条），请检查列映射是否正确")
    labels = sorted(set(e["label"] for e in examples))
    if len(labels) < 2:
        raise HTTPException(422, "拉取到的数据只有 1 个标签类别，无法用于分类训练")

    dataset_ref = dataset_store.put(examples)
    return DatasetFetchResponse(dataset_ref=dataset_ref, n_examples=len(examples), labels=labels)
