"""
core/marketplace.py  -  Phase 4：模型市场（本地 Registry）

路线图 Phase 4：Marketplace——用户可共享/售卖训练好的小模型配置。

Phase 4 实现：JSON 格式的本地模型注册表。
  - 注册已训练的模型（含元数据、性能指标、任务描述）
  - 按领域 / 任务类型搜索可复用模型
  - 导出模型配置供他人复用（零样本迁移）
  - 追踪下载量和用户评分

生产升级路径：
  - 本地 JSON → S3 + DynamoDB（云端 Registry）
  - 模型权重上传 → Hugging Face Hub
  - 付费模型 → Stripe 集成
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


REGISTRY_FILE = "marketplace/registry.json"


class ModelMarketplace:
    """
    模型市场：注册、搜索、复用训练好的模型配置。

    用法：
        mp = ModelMarketplace()

        # 训练完成后注册
        entry_id = mp.register(
            trainer=trainer,
            task_spec=task_spec,
            metrics={"f1": 0.85},
            description="客服工单分类，5类，中文",
            tags=["客服", "中文", "电商"],
        )

        # 搜索可复用模型
        results = mp.search(domain="客服", language="zh")

        # 查看排行榜
        top = mp.top_models(n=5)
    """

    def __init__(self, registry_path: str = REGISTRY_FILE):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, dict] = self._load()

    # ── 注册 ──────────────────────────────────────────────────────────────────

    def register(
        self,
        task_spec,
        metrics:      Dict[str, float],
        description:  str,
        deploy_path:  Optional[str] = None,
        tags:         Optional[List[str]] = None,
        author:       str = "anonymous",
        is_public:    bool = True,
    ) -> str:
        """
        注册一个训练好的模型到市场。

        Args:
            task_spec:    训练任务规格
            metrics:      评估指标字典，如 {"f1": 0.85, "accuracy": 0.87}
            description:  用户可读的模型描述
            deploy_path:  部署包目录（含 model.pkl / model.onnx）
            tags:         标签列表（便于搜索）
            author:       作者名称
            is_public:    是否公开（False = 私有，仅自己可用）

        Returns:
            model_id（UUID）
        """
        model_id = str(uuid.uuid4())[:8]

        entry = {
            "model_id":     model_id,
            "description":  description,
            "author":       author,
            "is_public":    is_public,
            "registered_at": datetime.utcnow().isoformat(),
            "tags":         tags or [],
            "task": {
                "type":        task_spec.task_type.value,
                "domain":      task_spec.domain,
                "labels":      task_spec.label_schema,
                "metric":      task_spec.evaluation_metric,
                "language":    task_spec.language,
                "description": task_spec.raw_description,
            },
            "metrics":      metrics,
            "deploy_path":  deploy_path,
            "downloads":    0,
            "rating":       None,
            "rating_count": 0,
        }

        self._registry[model_id] = entry
        self._save()
        return model_id

    # ── 搜索 ──────────────────────────────────────────────────────────────────

    def search(
        self,
        domain:    Optional[str] = None,
        task_type: Optional[str] = None,
        language:  Optional[str] = None,
        min_f1:    float = 0.0,
        tags:      Optional[List[str]] = None,
        public_only: bool = True,
    ) -> List[dict]:
        """
        按条件搜索已注册模型。

        Args:
            domain:     领域关键词（模糊匹配）
            task_type:  任务类型（classification / ner / ...）
            language:   语言（zh / en / mixed）
            min_f1:     最低 F1 要求
            tags:       必须包含的标签列表
            public_only: 是否只返回公开模型

        Returns:
            匹配的模型条目列表（按 F1 降序）
        """
        results = []
        for entry in self._registry.values():
            if public_only and not entry.get("is_public", True):
                continue
            task = entry.get("task", {})

            if domain and domain.lower() not in task.get("domain", "").lower():
                continue
            if task_type and task.get("type") != task_type:
                continue
            if language and task.get("language") != language:
                continue

            f1 = entry.get("metrics", {}).get("f1", 0)
            if f1 < min_f1:
                continue

            if tags:
                entry_tags = set(t.lower() for t in entry.get("tags", []))
                if not all(t.lower() in entry_tags for t in tags):
                    continue

            results.append(entry)

        results.sort(key=lambda e: e.get("metrics", {}).get("f1", 0), reverse=True)
        return results

    # ── 排行榜 ────────────────────────────────────────────────────────────────

    def top_models(self, n: int = 10, metric: str = "f1") -> List[dict]:
        """返回指标最高的 N 个公开模型"""
        public = [e for e in self._registry.values() if e.get("is_public", True)]
        public.sort(key=lambda e: e.get("metrics", {}).get(metric, 0), reverse=True)
        return public[:n]

    def get(self, model_id: str) -> Optional[dict]:
        """获取指定模型的完整信息"""
        entry = self._registry.get(model_id)
        if entry:
            # 记录下载
            entry["downloads"] = entry.get("downloads", 0) + 1
            self._save()
        return entry

    def rate(self, model_id: str, score: float):
        """给模型评分（1-5 分）"""
        entry = self._registry.get(model_id)
        if not entry: return
        old_rating = entry.get("rating") or score
        old_count  = entry.get("rating_count", 0)
        new_count  = old_count + 1
        new_rating = (old_rating * old_count + score) / new_count
        entry["rating"]       = round(new_rating, 2)
        entry["rating_count"] = new_count
        self._save()

    def summary(self) -> dict:
        """市场汇总统计"""
        entries = list(self._registry.values())
        public  = [e for e in entries if e.get("is_public", True)]
        domains = list(set(e["task"]["domain"] for e in entries if "task" in e))
        return {
            "total_models":  len(entries),
            "public_models": len(public),
            "domains":       domains,
            "avg_f1":        round(float(
                np.mean([e.get("metrics", {}).get("f1", 0) for e in public])
            ), 4) if public else 0.0,
            "total_downloads": sum(e.get("downloads", 0) for e in entries),
        }

    # ── 私有方法 ──────────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, dict]:
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self):
        self.registry_path.write_text(
            json.dumps(self._registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


try:
    import numpy as np
except ImportError:
    import builtins
    class _np:
        @staticmethod
        def mean(lst): return sum(lst) / len(lst) if lst else 0
    np = _np()
