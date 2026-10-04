"""
core/vlm_cls_pipeline.py  -  VLM 图像分类编排（core/pipeline.py 的图像版本）

跟 core/rl_pipeline.py/core/llm_ft_pipeline.py 一样，不往 core/pipeline.py 那个
350+ 行的文本分类主循环里加 if 分支——独立成一个模块，但发出和它完全一样的
事件信封（step_start/task_parsed/model_selected/epoch_done/iteration_done/
deploy_done/feedback_check/finished/error），前端 web/app.js::reduceEvent
不用改一行代码。

和文本分类主循环的对应关系：
  - 没有"数据准备/增强/飞轮"这些步骤——图片样本在对话阶段已经收集好了（真实
    标注，不是 LLM 猜的），这里直接用
  - 每个 outer iteration 都重新训练一次分类头（冻结编码器只做一次特征提取，
    重训头本身很快），跟踪 best_metric（最高 F1），保留表现最好的一次
  - VlmClsExplainer 的 next_action 用完整 5 值词表（有界指标、有验证集，
    跟 LLM 微调是同一类监督学习场景，不是 RL 那种只能看趋势的情况）
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


def run_vlm_cls_pipeline(
    api_key:        str,
    description:    str,
    image_examples: List[Dict],
    encoder_id:     str,
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
    完整 VLM 图像分类流程（无终端 UI），事件契约和 core/pipeline.py::run_pipeline 一致。

    Returns 结果字典，含 best_metric（最佳 F1）/ labels / deploy_path。
    """
    from api.accounts.llm_provisioning import build_client_for_request
    from core.task_parser import TaskParser
    from core.vlm_cls_trainer import VlmClsTrainer, VlmClsTrainingError
    from core.vlm_cls_explainer import VlmClsExplainer
    from core.vlm_cls_deployer import VlmClsDeployer

    client    = build_client_for_request(llm_provider, api_key, llm_model, llm_base_url,
                                         user_id=user_id, task_id=task_id)
    parser    = TaskParser(client)
    explainer = VlmClsExplainer(client)

    try:
        # ── Step 1: 解析（label_schema 用真实标注覆盖 LLM 的猜测，不是靠猜）──────
        _emit(on_event, {"type": "step_start", "step": "parsing", "message": "正在理解任务描述…"})
        task_spec = parser.parse(description, interactive=False)
        real_labels = sorted(set(e["label"] for e in image_examples))
        task_spec.label_schema = real_labels
        _emit(on_event, {
            "type": "task_parsed", "task_type": task_spec.task_type.value,
            "domain": task_spec.domain, "labels": real_labels, "metric": "f1",
        })

        _emit(on_event, {"type": "model_selected",
                         "model_name": f"冻结视觉编码器（{encoder_id}）+ 可训练分类头", "n_epochs": 8})

        # ── Steps 2–5: 训练闭环 ─────────────────────────────────────────────
        best_metric   = float("-inf")
        final_trainer = None
        all_epochs: List[Dict] = []
        history = []

        for iteration in range(max(1, max_iterations)):
            _emit(on_event, {"type": "iteration_start", "iteration": iteration + 1, "n_data": len(image_examples)})
            trainer = VlmClsTrainer(task_spec)
            try:
                epoch_results = trainer.train_with_eval(image_examples, encoder_id)
            except VlmClsTrainingError as e:
                _emit(on_event, {"type": "nn_codegen_fallback", "stage": "vlm_cls_train",
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
            # COLLECT_MORE_DATA 在对话式流程里没有"回去多问一轮"的机制（图片已经
            # 一次性给定），如实继续训练，不假装能自动补数据

        if final_trainer is None:
            raise RuntimeError("图像分类训练未能产出可用模型")

        # ── Step 6: 部署导出 ─────────────────────────────────────────────────
        deploy_path = None
        _emit(on_event, {"type": "step_start", "step": "deploy", "message": "正在导出部署包…"})
        try:
            deployer = VlmClsDeployer()
            pkg = deployer.export(final_trainer, task_spec, deploy_dir)
            deploy_path = pkg.package_dir
            _emit(on_event, {
                "type": "deploy_done", "format": pkg.export_format, "model_path": pkg.model_path,
                "size_kb": pkg.model_size_kb, "labels": pkg.labels, "usage_example": pkg.usage_example,
            })
        except Exception as e:
            _emit(on_event, {"type": "deploy_done", "error": str(e)})

        feedback_baseline = {
            "metric_name":     "f1",
            "baseline_metric": round(best_metric, 4),
            "recorded_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_train_samples": len(image_examples),
        }
        _emit(on_event, {"type": "feedback_check", "action": "baseline_recorded", **feedback_baseline})

        result = {
            "status":            "completed",
            "best_metric":       round(best_metric, 4),
            "metric_name":       "f1",
            "labels":            real_labels,
            "domain":            task_spec.domain,
            "n_samples":         len(image_examples),
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
