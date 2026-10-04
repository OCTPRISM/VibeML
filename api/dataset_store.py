"""api/dataset_store.py  -  内存缓存：已拉取的数据集样本（dataset_ref -> examples）"""
from __future__ import annotations
import uuid
from collections import OrderedDict
from typing import Dict, List, Optional

MAX_CACHED = 50


class DatasetStore:
    def __init__(self):
        self._cache: "OrderedDict[str, List[Dict]]" = OrderedDict()

    def put(self, examples: List[Dict]) -> str:
        if len(self._cache) >= MAX_CACHED:
            self._cache.popitem(last=False)
        ref = str(uuid.uuid4())
        self._cache[ref] = examples
        return ref

    def get(self, ref: str) -> Optional[List[Dict]]:
        return self._cache.get(ref)


dataset_store = DatasetStore()
