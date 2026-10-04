"""
core/data_sources.py  -  数据源接入：本地上传 / 本地路径 / HuggingFace Hub / 魔搭 ModelScope

统一接口：
    source.search(query, page)  -> List[DatasetSummary]              # 仅 HF/魔搭支持
    source.preview(ref, **kw)   -> DatasetPreview                    # 抽样 + 自动列映射建议
    source.fetch(ref, text_col, label_col, max_samples, **kw) -> List[Dict]  # [{text,label}]

设计取舍：
  - 所有 fetch 都有 max_samples 硬上限，不会不受控地拉整个数据集/整份文件。
  - 哪列是文本、哪列是标签用启发式猜测（复用 core/modalities/tabular.py 的
    detect_target_column 思路），但只是"建议"，最终由用户在预览界面确认/修改。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import DatasetSummary, DatasetPreview

UPLOAD_DIR = Path("./data_cache/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

COMMON_LABEL_NAMES = {"label", "target", "class", "y", "output", "category", "labels"}
COMMON_TEXT_NAMES  = {"text", "sentence", "content", "review", "comment", "input", "sentence1", "document"}


# ── 列映射启发式 ────────────────────────────────────────────────────────────

def _detect_label_col(columns: List[str], sample_rows: List[Dict]) -> Optional[str]:
    for col in columns:
        if col.lower() in COMMON_LABEL_NAMES:
            return col
    # 退化：选取值集合较小的列（典型分类标签特征），唯一值数越少越像标签列
    best, best_unique = None, None
    for col in columns:
        vals = [r.get(col) for r in sample_rows if r.get(col) is not None]
        if not vals:
            continue
        n_unique = len(set(map(str, vals)))
        if n_unique <= max(2, len(sample_rows) // 3) and (best_unique is None or n_unique < best_unique):
            best, best_unique = col, n_unique
    return best


def _detect_text_col(columns: List[str], sample_rows: List[Dict], exclude: Optional[str]) -> Optional[str]:
    candidates = [c for c in columns if c != exclude]
    for col in candidates:
        if col.lower() in COMMON_TEXT_NAMES:
            return col
    # 退化：选平均字符串长度最长的列（自由文本通常比其他字段长得多）
    best, best_len = None, -1.0
    for col in candidates:
        vals = [str(r.get(col, "")) for r in sample_rows]
        if not vals:
            continue
        avg_len = sum(len(v) for v in vals) / len(vals)
        if avg_len > best_len:
            best, best_len = col, avg_len
    return best


def _sanitize_value(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def _sanitize_row(row: Dict) -> Dict:
    return {k: _sanitize_value(v) for k, v in row.items()}


def _rows_to_examples(rows: List[Dict], text_col: str, label_col: str) -> List[Dict]:
    out = []
    for r in rows:
        text, label = r.get(text_col), r.get(label_col)
        if text is None or label is None:
            continue
        text = str(text).strip()
        if not text:
            continue
        out.append({"text": text, "label": str(label)})
    return out


def _build_preview(ref: str, rows: List[Dict], total_available: Optional[int],
                    warnings: Optional[List[str]] = None) -> DatasetPreview:
    sample = [_sanitize_row(r) for r in rows[:30]]
    columns = list(sample[0].keys()) if sample else []
    label_col = _detect_label_col(columns, sample)
    text_col = _detect_text_col(columns, sample, exclude=label_col)
    if not sample:
        warnings = (warnings or []) + ["未能读到任何样本，无法给出列映射建议"]
    return DatasetPreview(
        ref=ref, total_available=total_available, sample_rows=sample, columns=columns,
        suggested_text_col=text_col, suggested_label_col=label_col, warnings=warnings or [],
    )


def _read_tabular_file(path: Path) -> List[Dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        import pandas as pd
        return pd.read_csv(path).to_dict(orient="records")
    if suffix == ".tsv":
        import pandas as pd
        return pd.read_csv(path, sep="\t").to_dict(orient="records")
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for v in data.values():          # 常见形态 {"data": [...]}
                if isinstance(v, list):
                    return v
            return [data]
        return data
    if suffix in (".jsonl", ".ndjson"):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise ValueError(f"不支持的文件格式：{suffix}（支持 csv/tsv/json/jsonl）")


# ── 数据源基类 ──────────────────────────────────────────────────────────────

class DataSource:
    platform = "base"

    def search(self, query: str, page: int = 1, page_size: int = 10) -> List[DatasetSummary]:
        raise NotImplementedError("该数据源不支持搜索")

    def preview(self, ref: str, **kwargs) -> DatasetPreview:
        raise NotImplementedError

    def fetch(self, ref: str, text_col: str, label_col: str, max_samples: int = 500, **kwargs) -> List[Dict]:
        raise NotImplementedError


class LocalUploadSource(DataSource):
    """已通过 POST /api/datasets/upload 存到 UPLOAD_DIR 的文件，ref = 文件名"""
    platform = "local_upload"

    def _load_rows(self, ref: str) -> List[Dict]:
        path = (UPLOAD_DIR / ref).resolve()
        if UPLOAD_DIR.resolve() not in path.parents:
            raise ValueError("非法文件引用")
        if not path.exists():
            raise FileNotFoundError(f"上传文件不存在或已过期：{ref}")
        return _read_tabular_file(path)

    def preview(self, ref: str, **kwargs) -> DatasetPreview:
        rows = self._load_rows(ref)
        return _build_preview(ref, rows, total_available=len(rows))

    def fetch(self, ref: str, text_col: str, label_col: str, max_samples: int = 500, **kwargs) -> List[Dict]:
        return _rows_to_examples(self._load_rows(ref)[:max_samples], text_col, label_col)


class LocalPathSource(DataSource):
    """服务器本地文件系统路径（本工具单用户本地运行，路径就是用户自己机器上的路径）"""
    platform = "local_path"

    def _load_rows(self, ref: str) -> List[Dict]:
        path = Path(ref).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"本地路径不存在：{ref}")
        return _read_tabular_file(path)

    def preview(self, ref: str, **kwargs) -> DatasetPreview:
        rows = self._load_rows(ref)
        return _build_preview(ref, rows, total_available=len(rows))

    def fetch(self, ref: str, text_col: str, label_col: str, max_samples: int = 500, **kwargs) -> List[Dict]:
        return _rows_to_examples(self._load_rows(ref)[:max_samples], text_col, label_col)


class HFHubSource(DataSource):
    """HuggingFace Hub 数据集：搜索走 huggingface_hub，拉取走 datasets（streaming，只取前 N 条）"""
    platform = "huggingface"

    def search(self, query: str, page: int = 1, page_size: int = 10) -> List[DatasetSummary]:
        from huggingface_hub import HfApi
        api = HfApi()
        results = list(api.list_datasets(search=query, limit=page_size * page))
        results = results[(page - 1) * page_size: page * page_size]
        return [
            DatasetSummary(
                platform="huggingface", ref=r.id,
                description=(getattr(r, "description", None) or "") or "",
                downloads=getattr(r, "downloads", 0) or 0,
                likes=getattr(r, "likes", 0) or 0,
                tags=list(getattr(r, "tags", []) or []),
            )
            for r in results
        ]

    def preview(self, ref: str, split: str = "train", config: Optional[str] = None, **kwargs) -> DatasetPreview:
        rows, total, warnings = self._sample(ref, split, config, limit=30)
        return _build_preview(ref, rows, total_available=total, warnings=warnings)

    def fetch(self, ref: str, text_col: str, label_col: str, max_samples: int = 500,
              split: str = "train", config: Optional[str] = None, **kwargs) -> List[Dict]:
        rows, _, _ = self._sample(ref, split, config, limit=max_samples)
        return _rows_to_examples(rows, text_col, label_col)

    def _sample(self, ref: str, split: str, config: Optional[str],
                limit: int) -> Tuple[List[Dict], Optional[int], List[str]]:
        import itertools
        from datasets import load_dataset, get_dataset_config_names
        try:
            ds = load_dataset(ref, name=config, split=split, streaming=True)
        except ValueError as e:
            if config is None:
                try:
                    names = get_dataset_config_names(ref)
                except Exception:
                    names = []
                if names:
                    raise ValueError(
                        f"数据集 '{ref}' 有多个子集，请指定 config（可选：{', '.join(names[:10])}）"
                    ) from e
            raise
        rows = list(itertools.islice(ds, limit))
        return rows, None, []


class ModelScopeSource(DataSource):
    """阿里 魔搭 ModelScope 数据集：搜索走 HubApi，拉取走 MsDataset"""
    platform = "modelscope"

    def search(self, query: str, page: int = 1, page_size: int = 10) -> List[DatasetSummary]:
        from modelscope.hub.api import HubApi
        api = HubApi()
        res = api.list_datasets(search=query, page_size=page_size, page_number=page)
        items = getattr(res, "items", res)
        return [
            DatasetSummary(
                platform="modelscope",
                ref=getattr(it, "id", None) or getattr(it, "name", ""),
                description=(getattr(it, "description", None) or getattr(it, "display_name", None) or ""),
                downloads=getattr(it, "downloads", 0) or 0,
                likes=getattr(it, "likes", 0) or 0,
                tags=list(getattr(it, "tags", []) or []),
            )
            for it in items
        ]

    def preview(self, ref: str, split: str = "train", **kwargs) -> DatasetPreview:
        rows, total, warnings = self._sample(ref, split, limit=30)
        return _build_preview(ref, rows, total_available=total, warnings=warnings)

    def fetch(self, ref: str, text_col: str, label_col: str, max_samples: int = 500,
              split: str = "train", **kwargs) -> List[Dict]:
        rows, _, _ = self._sample(ref, split, limit=max_samples)
        return _rows_to_examples(rows, text_col, label_col)

    def _sample(self, ref: str, split: str, limit: int) -> Tuple[List[Dict], Optional[int], List[str]]:
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load(ref, split=split)
        rows = []
        for i, row in enumerate(ds):
            if i >= limit:
                break
            rows.append(dict(row))
        return rows, None, []


_REGISTRY = {
    "local_upload": LocalUploadSource,
    "local_path":   LocalPathSource,
    "huggingface":  HFHubSource,
    "modelscope":   ModelScopeSource,
}


def get_source(platform: str) -> DataSource:
    if platform not in _REGISTRY:
        raise ValueError(f"未知数据源：{platform}（支持：{', '.join(_REGISTRY)}）")
    return _REGISTRY[platform]()
