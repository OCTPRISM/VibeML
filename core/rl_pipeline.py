"""
core/rl_pipeline.py  -  强化学习编排（core/pipeline.py 的 RL 版本）

不往 core/pipeline.py 那个已经 350+ 行的分类任务主循环里加 if 分支——RL 的整个
"训练闭环"形状和分类任务本质不同（没有数据准备/增强/飞轮，没有 sklearn 风格调参），
硬塞进去只会让两边都难读。这里独立成一个模块，但发出和 core/pipeline.py 完全一样的
事件信封（step_start/task_parsed/model_selected/epoch_done/iteration_done/
deploy_done/feedback_check/finished/error），前端 web/app.js::reduceEvent 不需要
知道这是 RL 任务，一行代码都不用改。

和分类任务主循环的对应关系：
  - "环境设计"对应 custom_nn 的"架构设计"（LLM 生成 + 沙盒校验 + 失败重试）
  - 每个 outer iteration 对应分类任务的每一轮训练：每次都是全新初始化的策略网络
    重新训练（不是断点续训），跟踪 best_metric（最高的平均 reward），保留表现最好
    的一次作为最终部署的策略——这和分类任务"每轮保留最优 trainer"是同一个模式，
    重新训练几次也有实际价值（SB3 训练本身有随机性，多试几次有机会拿到更好的策略）
  - RLExplainer 的 next_action 只有 continue_training/stop_success/stop_plateau，
    达到 max_iterations 或者 explainer 判定 stop 就结束
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


def run_rl_pipeline(
    api_key:         str,
    description:     str,
    env_description: str,
    max_iterations:  int = 1,
    deploy_dir:      str = "./deploy",
    on_event:        Optional[EventCB] = None,
    llm_provider:    str = "anthropic",
    llm_model:       Optional[str] = None,
    llm_base_url:    Optional[str] = None,
    user_id:         Optional[Any] = None,
    task_id:         Optional[str] = None,
) -> Dict[str, Any]:
    """
    完整强化学习流程（无终端 UI），事件契约和 core/pipeline.py::run_pipeline 一致。

    Returns 结果字典，含 best_metric（最佳平均 reward）/ deploy_path。
    """
    from api.accounts.llm_provisioning import build_client_for_request
    from core.task_parser import TaskParser
    from core.rl_env_designer import RLEnvDesigner
    from core.rl_sandbox import validate as sandbox_validate
    from core.rl_trainer import RLTrainer, RLValidationError, RLTrainingError
    from core.rl_explainer import RLExplainer
    from core.rl_deployer import RLDeployer

    client    = build_client_for_request(llm_provider, api_key, llm_model, llm_base_url,
                                         user_id=user_id, task_id=task_id)
    parser    = TaskParser(client)
    designer  = RLEnvDesigner(client)
    explainer = RLExplainer(client)

    try:
        # ── Step 1: 解析 ──────────────────────────────────────────────────────
        _emit(on_event, {"type": "step_start", "step": "parsing", "message": "正在理解任务描述…"})
        combined_desc = f"{description}\n补充的环境描述：{env_description}" if env_description else description
        task_spec = parser.parse(combined_desc, interactive=False)
        _emit(on_event, {
            "type": "task_parsed", "task_type": task_spec.task_type.value,
            "domain": task_spec.domain, "labels": [], "metric": "episode_reward_mean",
        })

        # ── Step 2: 环境设计（LLM 生成 + 沙盒校验 + 失败重试）──────────────────
        _emit(on_event, {"type": "step_start", "step": "env_design", "message": "正在设计强化学习环境…"})
        repair_hint = ""
        env_spec = None
        for attempt in range(2):    # 首次 + 1 次把错误喂回 LLM 的修复重试
            spec = designer.design(task_spec, repair_hint=repair_hint)
            _emit(on_event, {
                "type": "arch_designed", "mode": "rl_env",
                "class_name": spec.class_name, "code": spec.source_code,
                "action_space": spec.action_space_desc,
                "observation_space": spec.observation_space_desc,
                "rationale": spec.reward_rationale,
            })
            check = sandbox_validate(spec.source_code)
            if check.ok:
                env_spec = spec
                break
            _emit(on_event, {"type": "nn_codegen_fallback", "stage": "rl_env",
                             "attempt": attempt + 1, "reason": "; ".join(check.errors)})
            repair_hint = "; ".join(check.errors)

        if env_spec is None:
            raise RuntimeError("强化学习环境代码连续两次未通过安全校验，已放弃")

        _emit(on_event, {"type": "model_selected",
                         "model_name": "stable-baselines3（自动选择 PPO/DQN）", "n_epochs": 0})

        # ── Steps 3–6: 训练闭环 ─────────────────────────────────────────────
        best_metric   = float("-inf")
        final_trainer = None
        all_epochs: List[Dict] = []
        history = []

        for iteration in range(max(1, max_iterations)):
            _emit(on_event, {"type": "iteration_start", "iteration": iteration + 1, "n_data": 0})
            trainer = RLTrainer(task_spec)
            try:
                epoch_results = trainer.train_with_eval(env_spec)
            except (RLValidationError, RLTrainingError) as e:
                _emit(on_event, {"type": "nn_codegen_fallback", "stage": "rl_train",
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

            if explanation.next_action in (NextAction.STOP_SUCCESS, NextAction.STOP_PLATEAU):
                break

        if final_trainer is None:
            raise RuntimeError("强化学习训练连续失败，未能产出可用策略")

        # ── Step 7: 部署导出 ─────────────────────────────────────────────────
        deploy_path = None
        _emit(on_event, {"type": "step_start", "step": "deploy", "message": "正在导出部署包…"})
        try:
            deployer = RLDeployer()
            pkg = deployer.export(final_trainer, task_spec, env_spec, deploy_dir)
            deploy_path = pkg.package_dir
            _emit(on_event, {
                "type": "deploy_done", "format": pkg.export_format, "model_path": pkg.model_path,
                "size_kb": pkg.model_size_kb, "labels": pkg.labels, "usage_example": pkg.usage_example,
            })
        except Exception as e:
            _emit(on_event, {"type": "deploy_done", "error": str(e)})

        feedback_baseline = {
            "metric_name":     "episode_reward_mean",
            "baseline_metric": round(best_metric, 4),
            "recorded_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_train_samples": 0,
        }
        _emit(on_event, {"type": "feedback_check", "action": "baseline_recorded", **feedback_baseline})

        result = {
            "status":            "completed",
            "best_metric":       round(best_metric, 4),
            "metric_name":       "episode_reward_mean",
            "labels":            [],
            "domain":            task_spec.domain,
            "n_samples":         0,
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
