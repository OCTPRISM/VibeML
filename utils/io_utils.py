"""
utils/io_utils.py - 结果持久化工具

把 LoopState 保存为人类可读的 JSON，方便：
  - 查看完整的训练历程
  - 对比不同迭代的诊断结论
  - 为后续 Phase 2 (HuggingFace PEFT) 提供基线记录
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from config import LoopState


def save_state(state: LoopState, output_dir: str = "outputs") -> str:
    """
    将 LoopState 序列化为 JSON 并保存。

    Returns:
        保存路径
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"run_{ts}.json"

    data = {
        "timestamp":   ts,
        "task":        _serialize_task(state),
        "data_report": _serialize_report(state),
        "training":    _serialize_training(state),
        "summary": {
            "best_metric":  state.best_metric,
            "iterations":   state.iteration,
            "final_action": state.explanations[-1].next_action.value
                            if state.explanations else "none",
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return str(path)


def load_state_summary(path: str) -> Dict[str, Any]:
    """读取保存的运行记录（仅摘要，不恢复模型）"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict]:
    """加载 JSONL 格式训练数据"""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def save_jsonl(data: List[Dict], path: str):
    """保存 JSONL 格式数据"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── 私有序列化辅助 ────────────────────────────────────────────────────────────

def _serialize_task(state: LoopState) -> Dict:
    spec = state.task_spec
    return {
        "description":   spec.raw_description,
        "task_type":     spec.task_type.value,
        "domain":        spec.domain,
        "label_schema":  spec.label_schema,
        "metric":        spec.evaluation_metric,
        "language":      spec.language,
    }


def _serialize_report(state: LoopState) -> Dict:
    r = state.data_report
    return {
        "total_samples":      r.total_samples,
        "augmented_count":    r.augmented_count,
        "label_distribution": r.label_distribution,
        "quality_score":      r.quality_score,
        "warnings":           r.warnings,
        "boundary_count":     len(r.boundary_samples),
        "boundary_samples": [
            {
                "score":  b.get("boundary_score"),
                "reason": b.get("reason"),
                "text":   b.get("original", {}).get("text", "")[:80],
                "label":  b.get("original", {}).get("label", ""),
            }
            for b in r.boundary_samples
        ],
    }


def _serialize_training(state: LoopState) -> Dict:
    epochs = [
        {
            "epoch":       e.epoch,
            "val_metric":  e.val_metric,
            "train_loss":  e.train_loss,
            "val_loss":    e.val_loss,
            "per_class":   e.per_class_metrics,
            "confusion":   e.confusion_highlights,
        }
        for e in state.epoch_results
    ]

    explanations = [
        {
            "diagnosis":      ex.diagnosis,
            "root_cause":     ex.root_cause,
            "recommendation": ex.recommendation,
            "next_action":    ex.next_action.value,
            "confidence":     ex.confidence,
        }
        for ex in state.explanations
    ]

    return {
        "epoch_results":  epochs,
        "explanations":   explanations,
        "plateau_count":  state.plateau_count,
    }
