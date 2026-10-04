"""
eval/exp2_fallback_ladder.py  -  实验 2：三级降级链 vs. 纯 sklearn

真实调用 core/pipeline.py::run_pipeline（真实 Ollama qwen3.6:35b-a3b + 真实子进程训练，
不是 mock），对比 model_backend="custom_nn"（触发 custom_nn → pretrained_nn → sklearn
降级链）vs. model_backend="sklearn"（纯 sklearn 基线）。

方法论偏离说明（如实记录）：原计划用 customer_tickets 数据集（业务场景最贴近），但
该数据集总共只有 30 条（约 6 条/类），无法支撑计划里"每类 20 条"的取样量——用它会导致
两个条件都退化到近乎随机的 F1（Experiment 1 已经验证过这一点：customer_tickets 在
所有系统上都是 0.01-0.03 的地板值，对比不出降级链有没有价值）。改用 chinese_sentiment
（真实 HF 数据集，双语定位的另一半，池子够大）在 n=20/类 上取样，这是发现问题后的
诚实调整，不是为了让结果好看。

耗时警告：custom_nn 两次尝试失败后降级到 pretrained_nn，本次会话实测最坏情况单次运行
约 26 分钟；本实验 3 个种子 × 2 条件，custom_nn 条件预计 1-3 小时，建议后台运行。

运行：python -m eval.exp2_fallback_ladder
"""

from __future__ import annotations

import json
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from eval.datasets import load_chinese_sentiment

N_SEEDS = 3
N_PER_CLASS = 20
OUT_DIR = Path(__file__).parent.parent / "outputs" / "exp2_fallback_ladder"


def _subsample_examples(seed: int) -> List[Dict[str, str]]:
    ds = load_chinese_sentiment().subsample(N_PER_CLASS, seed=seed)
    return [{"text": t, "label": l} for t, l in zip(ds.train_texts, ds.train_labels)]


def _run_one(model_backend: str, seed: int) -> Dict[str, Any]:
    from core.pipeline import run_pipeline

    examples = _subsample_examples(seed)
    events: List[Dict[str, Any]] = []

    def on_event(ev):
        events.append(ev)

    with tempfile.TemporaryDirectory() as tmp_deploy:
        t0 = time.time()
        result = run_pipeline(
            api_key="", description="把中文评论按情感倾向分类为正面或负面",
            examples=examples, max_iterations=1, target_metric=0.80,
            enable_phase2=(model_backend == "sklearn"),  # 部署导出目前只支持 sklearn 后端，
            # custom_nn 条件下关掉避免部署步骤的额外耗时干扰计时（部署本身不是本实验的度量对象）
            deploy_dir=tmp_deploy, on_event=on_event,
            llm_provider="ollama", llm_model=None, model_backend=model_backend,
        )
        elapsed = time.time() - t0

    result.pop("_trainer", None)

    arch_events   = [e for e in events if e.get("type") == "arch_designed"]
    fallback_events = [e for e in events if e.get("type") == "nn_codegen_fallback"]
    final_backend = arch_events[-1]["mode"] if arch_events else "sklearn"
    codegen_first_try_ok = (model_backend == "custom_nn" and len(fallback_events) == 0
                            and final_backend == "custom_nn")

    return {
        "model_backend_requested": model_backend,
        "seed": seed,
        "n_examples": len(examples),
        "wall_clock_s": round(elapsed, 1),
        "status": result.get("status"),
        "best_metric": result.get("best_metric"),
        "metric_name": result.get("metric_name"),
        "final_backend_used": final_backend,
        "n_fallback_events": len(fallback_events),
        "fallback_reasons": [e.get("reason", "")[:200] for e in fallback_events],
        "custom_nn_first_try_success": codegen_first_try_ok,
        "error": result.get("message") if result.get("status") == "error" else None,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_runs: List[Dict[str, Any]] = []

    for seed in range(N_SEEDS):
        print(f"\n=== seed={seed}: model_backend=sklearn ===")
        r = _run_one("sklearn", seed)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        all_runs.append(r)
        (OUT_DIR / "raw_runs.json").write_text(json.dumps(all_runs, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n=== seed={seed}: model_backend=custom_nn (fallback ladder) ===")
        r = _run_one("custom_nn", seed)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        all_runs.append(r)
        (OUT_DIR / "raw_runs.json").write_text(json.dumps(all_runs, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总
    sklearn_runs = [r for r in all_runs if r["model_backend_requested"] == "sklearn"]
    ladder_runs  = [r for r in all_runs if r["model_backend_requested"] == "custom_nn"]

    def _avg(runs, key):
        vals = [r[key] for r in runs if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "n_seeds": N_SEEDS,
        "dataset": "chinese_sentiment (真实 HF 数据集, n=20/类, 替代原计划的 customer_tickets"
                   "——后者仅 30 条样本，无法支撑 n=20/类，详见模块 docstring)",
        "sklearn_baseline": {
            "mean_best_metric": _avg(sklearn_runs, "best_metric"),
            "mean_wall_clock_s": _avg(sklearn_runs, "wall_clock_s"),
            "runs": sklearn_runs,
        },
        "fallback_ladder": {
            "mean_best_metric": _avg(ladder_runs, "best_metric"),
            "mean_wall_clock_s": _avg(ladder_runs, "wall_clock_s"),
            "custom_nn_first_try_success_rate": round(
                sum(r["custom_nn_first_try_success"] for r in ladder_runs) / len(ladder_runs), 4
            ) if ladder_runs else None,
            "final_backend_distribution": {
                b: sum(1 for r in ladder_runs if r["final_backend_used"] == b)
                for b in set(r["final_backend_used"] for r in ladder_runs)
            } if ladder_runs else {},
            "runs": ladder_runs,
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 汇总已保存：{OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
