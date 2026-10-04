"""
core/llm_ft_pipeline.py  -  LLM 指令微调编排（core/pipeline.py 的 SFT 版本）

和 core/rl_pipeline.py 一样，不往 core/pipeline.py 那个 350+ 行的分类任务主循环
里加 if 分支——发出完全一样的事件信封（step_start/task_parsed/model_selected/
epoch_done/iteration_done/deploy_done/feedback_check/finished/error），
web/app.js::reduceEvent 不需要知道这是 LLM 微调任务。

base_model_id 已经在对话阶段（core/conversation/stages.py::choose_model）经过
core/llm_ft_selector.py 校验过，这里直接信任，不重复校验——和 run_pipeline
信任 examples 已经在 prepare_data 阶段收集好是同一个原则。

next_action == adjust_hyperparams 时，从 hyperparam_delta 里读 SFT 专属键
（learning_rate/lora_rank/num_epochs）应用到下一轮迭代——不复用 sklearn 的
C/alpha/class_boost 语义。next_action == collect_more_data 时，对话式场景下
没法在训练循环中途去问用户要更多例子，这里只是如实把这个信号报出去（前端能
看到"建议补充更多例子"的播报），不在本轮循环内自动做什么，交由用户在下一轮
对话里决定。
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


def run_llm_ft_pipeline(
    api_key:               str,
    description:           str,
    instruction_examples:  List[Dict],
    base_model_id:         str,
    max_iterations:        int = 1,
    deploy_dir:            str = "./deploy",
    on_event:              Optional[EventCB] = None,
    llm_provider:          str = "anthropic",
    llm_model:             Optional[str] = None,
    llm_base_url:          Optional[str] = None,
    user_id:               Optional[Any] = None,
    task_id:               Optional[str] = None,
) -> Dict[str, Any]:
    """
    完整 LLM 指令微调流程（无终端 UI），事件契约和 core/pipeline.py::run_pipeline 一致。

    Returns 结果字典，含 best_metric（sft_score）/ deploy_path。
    """
    from api.accounts.llm_provisioning import build_client_for_request
    from core.task_parser import TaskParser
    from core.llm_ft_trainer import LLMFTTrainer, LLMFTTrainingError
    from core.llm_ft_explainer import LLMFTExplainer
    from core.llm_ft_deployer import LLMFTDeployer

    client    = build_client_for_request(llm_provider, api_key, llm_model, llm_base_url,
                                         user_id=user_id, task_id=task_id)
    parser    = TaskParser(client)
    explainer = LLMFTExplainer(client)

    try:
        # ── Step 1: 解析 ──────────────────────────────────────────────────────
        _emit(on_event, {"type": "step_start", "step": "parsing", "message": "正在理解任务描述…"})
        task_spec = parser.parse(description, interactive=False)
        _emit(on_event, {
            "type": "task_parsed", "task_type": task_spec.task_type.value,
            "domain": task_spec.domain, "labels": [], "metric": "sft_score",
        })

        _emit(on_event, {"type": "model_selected", "model_name": base_model_id, "n_epochs": 0})

        # 训练超参——被 explainer 的 adjust_hyperparams 建议在每轮迭代之间调整，
        # 不是每轮都重新问 LLM 要固定值
        num_epochs    = 3
        batch_size    = 4
        learning_rate = 2e-4
        lora_rank     = 8

        # ── Steps 2-5: 训练闭环 ─────────────────────────────────────────────
        best_metric   = float("-inf")
        final_trainer = None
        all_epochs: List[Dict] = []
        history = []

        for iteration in range(max(1, max_iterations)):
            _emit(on_event, {"type": "iteration_start", "iteration": iteration + 1,
                             "n_data": len(instruction_examples)})
            trainer = LLMFTTrainer(base_model_id)
            try:
                epoch_results = trainer.train_with_eval(
                    instruction_examples, num_epochs=num_epochs, batch_size=batch_size,
                    learning_rate=learning_rate, lora_rank=lora_rank)
            except LLMFTTrainingError as e:
                _emit(on_event, {"type": "nn_codegen_fallback", "stage": "llm_ft_train",
                                 "attempt": iteration + 1, "reason": str(e)})
                if final_trainer is not None:
                    break   # 已经有一次成功的训练可用，这次失败就不再重试，直接用已有结果收尾
                continue

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

            if explanation.next_action == NextAction.ADJUST_HYPERPARAMS:
                delta = explanation.hyperparam_delta or {}
                if "learning_rate" in delta:
                    learning_rate = float(delta["learning_rate"])
                if "lora_rank" in delta:
                    lora_rank = int(delta["lora_rank"])
                if "num_epochs" in delta:
                    num_epochs = int(delta["num_epochs"])
            elif explanation.next_action in (NextAction.STOP_SUCCESS, NextAction.STOP_PLATEAU):
                break
            # COLLECT_MORE_DATA：如实报出信号即可，见模块 docstring

        if final_trainer is None:
            raise RuntimeError("LLM 指令微调训练连续失败，未能产出可用模型")

        # ── Step 6: 部署导出 ─────────────────────────────────────────────────
        deploy_path = None
        _emit(on_event, {"type": "step_start", "step": "deploy", "message": "正在导出部署包…"})
        try:
            deployer = LLMFTDeployer()
            pkg = deployer.export(final_trainer, task_spec, base_model_id, deploy_dir)
            deploy_path = pkg.package_dir
            _emit(on_event, {
                "type": "deploy_done", "format": pkg.export_format, "model_path": pkg.model_path,
                "size_kb": pkg.model_size_kb, "labels": pkg.labels, "usage_example": pkg.usage_example,
            })
        except Exception as e:
            _emit(on_event, {"type": "deploy_done", "error": str(e)})

        feedback_baseline = {
            "metric_name":     "sft_score",
            "baseline_metric": round(best_metric, 4),
            "recorded_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_train_samples": len(instruction_examples),
        }
        _emit(on_event, {"type": "feedback_check", "action": "baseline_recorded", **feedback_baseline})

        result = {
            "status":            "completed",
            "best_metric":       round(best_metric, 4),
            "metric_name":       "sft_score",
            "labels":            [],
            "domain":            task_spec.domain,
            "n_samples":         len(instruction_examples),
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
