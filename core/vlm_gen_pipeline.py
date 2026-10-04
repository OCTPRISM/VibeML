"""
core/vlm_gen_pipeline.py  -  生成式 VLM 编排（core/vlm_cls_pipeline.py 的生成版本）

跟 core/vlm_cls_pipeline.py 同一套事件信封（step_start/task_parsed/model_selected/
epoch_done/iteration_done/deploy_done/feedback_check/finished/error），前端
web/app.js::reduceEvent 不用改一行代码。

和图像分类那条流水线的对应关系：
  - 没有"数据准备/增强/飞轮"——图文样本在对话阶段已经收集好（真实标注的
    问题/参考答案，不是猜的），这里直接用
  - task_spec.label_schema 这里不适用（生成式任务没有固定标签集合），
    task_parsed 事件里 labels 字段留空，跟 vlm_cls 那条线的字段形状保持
    一致但语义上如实反映"这个任务没有分类标签"
  - VlmGenExplainer 的 next_action 用完整 5 值词表（有参考答案、有验证集，
    跟图像分类是同一类"有界指标、可以持续迭代"的监督学习场景）
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from config import NextAction

EventCB = Callable[[Dict[str, Any]], None]


def _emit(cb: Optional[EventCB], event: Dict[str, Any]):
    if cb:
        try:
            cb(event)
        except Exception:
            pass


def run_vlm_gen_pipeline(
    api_key:        str,
    description:    str,
    vlm_examples:   List[Dict],
    base_model_id:  str,
    max_iterations: int = 3,
    deploy_dir:     str = "./deploy",
    on_event:       Optional[EventCB] = None,
    llm_provider:   str = "anthropic",
    llm_model:      Optional[str] = None,
    llm_base_url:   Optional[str] = None,
    user_id:        Optional[Any] = None,
    task_id:        Optional[str] = None,
) -> Dict[str, Any]:
    """
    完整生成式 VLM 流程（无终端 UI），事件契约和 core/pipeline.py::run_pipeline 一致。

    Returns 结果字典，含 best_metric（最佳 ROUGE-L）/ deploy_path。
    """
    from api.accounts.llm_provisioning import build_client_for_request
    from core.task_parser import TaskParser
    from core.vlm_gen_trainer import VlmGenTrainer, VlmGenTrainingError
    from core.vlm_gen_explainer import VlmGenExplainer
    from core.vlm_gen_deployer import VlmGenDeployer

    client    = build_client_for_request(llm_provider, api_key, llm_model, llm_base_url,
                                         user_id=user_id, task_id=task_id)
    parser    = TaskParser(client)
    explainer = VlmGenExplainer(client)

    try:
        # ── Step 1: 解析（生成式任务没有固定标签集合，label_schema 留空）──────
        _emit(on_event, {"type": "step_start", "step": "parsing", "message": "正在理解任务描述…"})
        task_spec = parser.parse(description, interactive=False)
        task_spec.label_schema = []
        _emit(on_event, {
            "type": "task_parsed", "task_type": task_spec.task_type.value,
            "domain": task_spec.domain, "labels": [], "metric": "rouge_l",
        })

        _emit(on_event, {"type": "model_selected",
                         "model_name": f"生成式 VLM 全量微调（{base_model_id}）", "n_epochs": 5})

        # ── Steps 2–5: 训练闭环 ─────────────────────────────────────────────
        best_metric   = float("-inf")
        final_trainer = None
        all_epochs: List[Dict] = []
        history = []

        for iteration in range(max(1, max_iterations)):
            _emit(on_event, {"type": "iteration_start", "iteration": iteration + 1, "n_data": len(vlm_examples)})
            trainer = VlmGenTrainer(task_spec)
            try:
                epoch_results = trainer.train_with_eval(vlm_examples, base_model_id)
            except VlmGenTrainingError as e:
                _emit(on_event, {"type": "nn_codegen_fallback", "stage": "vlm_gen_train",
                                 "attempt": iteration + 1, "reason": str(e)})
                if final_trainer is not None:
                    break
                raise

            for ep in epoch_results:
                _emit(on_event, {
                    "type": "epoch_done", "iteration": iteration + 1, "epoch": ep.epoch,
                    "val_metric": ep.val_metric, "train_loss": ep.train_loss, "val_loss": ep.val_loss,
                    "per_class": ep.per_class_metrics, "confusion": ep.confusion_highlights,
                })
                all_epochs.append({"iteration": iteration + 1, "epoch": ep.epoch,
                                   "val_metric": ep.val_metric, "train_loss": ep.train_loss})
                history.append(ep)

            latest = epoch_results[-1]
            explanation = explainer.explain(latest, task_spec, history[:-1])
            _emit(on_event, {
                "type": "iteration_done", "iteration": iteration + 1, "val_metric": latest.val_metric,
                "diagnosis": explanation.diagnosis, "root_cause": explanation.root_cause,
                "recommendation": explanation.recommendation, "next_action": explanation.next_action.value,
                "confidence": explanation.confidence,
            })

            if latest.val_metric > best_metric:
                best_metric   = latest.val_metric
                final_trainer = trainer

            if explanation.next_action in (NextAction.STOP_SUCCESS, NextAction.STOP_PLATEAU):
                break
            # COLLECT_MORE_DATA 在对话式流程里没有"回去多问一轮"的机制（图文样本已经
            # 一次性给定），如实继续训练，不假装能自动补数据

        if final_trainer is None:
            raise RuntimeError("生成式 VLM 训练未能产出可用模型")

        # ── Step 6: 部署导出 ─────────────────────────────────────────────────
        deploy_path = None
        _emit(on_event, {"type": "step_start", "step": "deploy", "message": "正在导出部署包…"})
        try:
            deployer = VlmGenDeployer()
            pkg = deployer.export(final_trainer, task_spec, deploy_dir)
            deploy_path = pkg.package_dir
            _emit(on_event, {
                "type": "deploy_done", "format": pkg.export_format, "model_path": pkg.model_path,
                "size_kb": pkg.model_size_kb, "labels": pkg.labels, "usage_example": pkg.usage_example,
            })
        except Exception as e:
            _emit(on_event, {"type": "deploy_done", "error": str(e)})

        feedback_baseline = {
            "metric_name":     "rouge_l",
            "baseline_metric": round(best_metric, 4),
            "recorded_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_train_samples": len(vlm_examples),
        }
        _emit(on_event, {"type": "feedback_check", "action": "baseline_recorded", **feedback_baseline})

        result = {
            "status":            "completed",
            "best_metric":       round(best_metric, 4),
            "metric_name":       "rouge_l",
            "labels":            [],
            "domain":            task_spec.domain,
            "n_samples":         len(vlm_examples),
            "epoch_history":     all_epochs,
            "deploy_path":       deploy_path,
            "feedback_baseline": feedback_baseline,
        }
        _emit(on_event, {"type": "finished", **result})
        result["_trainer"] = final_trainer
        return result

    except Exception as e:
        error = {"status": "error", "message": str(e)}
        _emit(on_event, {"type": "error", **error})
        return error
