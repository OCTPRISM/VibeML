"""
core/pipeline.py  -  无 UI 事件驱动管道 + 部署后反馈闭环

Phase 2 缺口补全：
  - 部署后反馈闭环：生产指标 → 触发下一轮训练的决策逻辑

每次训练完成后，pipeline 记录 baseline_metric。
若后续调用 check_feedback() 时发现生产指标下降超过阈值，
自动返回 "retrain_recommended" 信号，外层调用者（API worker）
可据此触发新一轮 run_pipeline()。

事件类型（on_event callback 的 type 字段）：
  step_start / task_parsed / parallel_prep_done / data_ready / model_selected /
  arch_designed / nn_codegen_fallback / augment_done / epoch_done / iteration_done /
  hyperparams_adjusted / flywheel_done / deploy_done / feedback_check / finished / error
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import ModelBackend

EventCB = Callable[[Dict[str, Any]], None]

_MODEL_NAMES = {
    "logreg": "TF-IDF + Logistic Regression（自动调参切换）",
    "svm":    "TF-IDF + Linear SVM（自动调参切换）",
    "sgd":    "TF-IDF + SGD Classifier（自动调参切换）",
}


def _emit(cb: Optional[EventCB], event: Dict[str, Any]):
    if cb:
        try:
            cb(event)
        except Exception:
            pass


def _prewarm_design(model_backend, client, task_spec, n_samples):
    """
    在数据准备（下载/清洗/拆分/格式转换）跑的同时，先做一次架构设计尝试——这一步只
    依赖 task_spec + 原始样本量级，不需要等数据准备完成，因此可以和它并行跑（用
    ThreadPoolExecutor，见 run_pipeline）。只负责"设计 + 当场能做的静态校验"，不做
    真正训练（真正训练必须等最终数据齐了才能跑）。设计/校验失败就返回 None，交给
    _select_and_train_backend 的完整重试链在数据就绪后重新来一遍，不影响正确性，
    只是没吃到并行的时间红利而已。
    """
    try:
        if model_backend == "custom_nn":
            from core.arch_designer import ArchDesigner
            from core.nn_sandbox import validate as sandbox_validate
            spec = ArchDesigner(client).design(task_spec, n_samples)
            if not sandbox_validate(spec.source_code).ok:
                return None
            return {"kind": "custom_nn", "spec": spec}
        elif model_backend == "pretrained_nn":
            from core.backbone_selector import BackboneSelector
            bspec = BackboneSelector(client).select(task_spec, n_samples)
            return {"kind": "pretrained_nn", "spec": bspec}
    except Exception:
        return None
    return None


def _select_and_train_backend(model_backend, client, task_spec, data, on_event, augmentor, prewarmed=None):
    """
    根据 model_backend 选择训练后端，必要时走 custom_nn → pretrained_nn → sklearn 降级链
    （每一步失败都发 nn_codegen_fallback 事件，不静默吞掉）。

    prewarmed：run_pipeline 在数据准备并行阶段跑出来的第一次设计结果（见
    _prewarm_design），如果类型匹配就直接复用，省一次 LLM 调用；用不上（None，
    或者种类对不上）就退回原来"当场设计"的行为，正确性不受影响。

    返回 (trainer, config, epoch_results, arch_spec, backbone_spec)——epoch_results 是
    用这个后端跑出来的第一轮真实训练结果，调用方直接当作 iteration 1 使用，不重复训练。
    """
    from config import TrainingConfig
    from core.trainer import Trainer

    if model_backend == "custom_nn":
        from core.arch_designer import ArchDesigner
        from core.nn_trainer import NNTrainer, ArchValidationError, NNTrainingError

        designer = ArchDesigner(client)
        repair_hint = ""
        for attempt in range(2):   # 首次 + 1 次把错误喂回 LLM 的修复重试
            try:
                if attempt == 0 and prewarmed and prewarmed.get("kind") == "custom_nn":
                    spec = prewarmed["spec"]
                else:
                    spec = designer.design(task_spec, len(data), repair_hint=repair_hint)
                _emit(on_event, {"type": "arch_designed", "mode": "custom_nn",
                                 "class_name": spec.class_name, "code": spec.source_code,
                                 "loss_fn": spec.loss_fn_name, "rationale": spec.rationale})
                config = TrainingConfig(
                    model_name=f"LLM 自定义分类头（{spec.class_name}）", use_lora=False,
                    lora_rank=0, lora_alpha=0, learning_rate=1e-3, num_epochs=6, batch_size=16,
                    max_length=256, use_cpu_fallback=True, backend=ModelBackend.CUSTOM_NN)
                trainer = NNTrainer(task_spec)
                epoch_results = trainer.train_with_eval(data, config, arch_spec=spec)
                return trainer, config, epoch_results, spec, None
            except (ArchValidationError, NNTrainingError) as e:
                _emit(on_event, {"type": "nn_codegen_fallback", "stage": "custom_nn",
                                 "attempt": attempt + 1, "reason": str(e)})
                repair_hint = str(e)
        model_backend = "pretrained_nn"   # 两次都失败，降级

    if model_backend == "pretrained_nn":
        from core.backbone_selector import BackboneSelector
        from core.nn_trainer import NNTrainer

        try:
            if prewarmed and prewarmed.get("kind") == "pretrained_nn":
                bspec = prewarmed["spec"]
            else:
                bspec = BackboneSelector(client).select(task_spec, len(data))
            _emit(on_event, {"type": "arch_designed", "mode": "pretrained_nn",
                             "model_id": bspec.model_id, "use_lora": bspec.use_lora,
                             "rationale": bspec.rationale})
            config = TrainingConfig(
                model_name=f"预训练模型微调（{bspec.model_id}）", use_lora=bspec.use_lora,
                lora_rank=8, lora_alpha=16, learning_rate=2e-5, num_epochs=4, batch_size=16,
                max_length=128, use_cpu_fallback=True, backend=ModelBackend.PRETRAINED_NN)
            trainer = NNTrainer(task_spec)
            epoch_results = trainer.train_with_eval(data, config, backbone_spec=bspec)
            return trainer, config, epoch_results, None, bspec
        except Exception as e:
            _emit(on_event, {"type": "nn_codegen_fallback", "stage": "pretrained_nn",
                             "attempt": 1, "reason": str(e)})

    # sklearn：默认路径，也是两级降级链最终的兜底
    trainer = Trainer(task_spec)
    config  = trainer.auto_select(len(data))
    epoch_results = trainer.train_with_eval(data, config, augmentor=augmentor)
    return trainer, config, epoch_results, None, None


# ── 主训练管道 ────────────────────────────────────────────────────────────────

def run_pipeline(
    api_key:        str,
    description:    str,
    examples:       List[Dict],
    max_iterations: int   = 3,
    target_metric:  float = 0.80,
    enable_phase2:  bool  = True,
    deploy_dir:     str   = "./deploy",
    on_event:       Optional[EventCB] = None,
    llm_provider:   str   = "anthropic",
    llm_model:      Optional[str] = None,
    llm_base_url:   Optional[str] = None,
    model_backend:  str   = "sklearn",
    user_id:        Optional[Any] = None,   # provider == "system_managed" 时必填，配额计量用
    task_id:        Optional[str] = None,   # 用量明细里标注来源，纯审计用途，不影响训练逻辑
) -> Dict[str, Any]:
    """
    完整 AutoML 流程（无终端 UI）。

    Returns 结果字典，含 best_metric / labels / deploy_path /
    feedback_baseline（供后续 check_feedback 使用）。
    """
    from config import NextAction
    from api.accounts.llm_provisioning import build_client_for_request
    from core.loss_factory import SUPPORTED_LOSSES
    from core.robust_eval  import (
        cross_val_metric, estimate_noise, is_real_improvement, should_use_cv)
    from core.hard_example_miner import find_hard_examples, select_candidates
    from core.task_parser  import TaskParser
    from core.data_engine  import DataEngine
    from core.trainer      import Trainer
    from core.explainer    import Explainer
    from core.augmentor    import Augmentor
    from core.deployer     import Deployer

    client    = build_client_for_request(llm_provider, api_key, llm_model, llm_base_url,
                                         user_id=user_id, task_id=task_id)
    parser    = TaskParser(client)
    engine    = DataEngine(client)
    explainer = Explainer(client)

    try:
        # ── Step 1: 解析 ──────────────────────────────────────────────────────
        _emit(on_event, {"type": "step_start", "step": "parsing",
                         "message": "正在理解任务描述…"})
        task_spec = parser.parse(description, interactive=False)
        _emit(on_event, {
            "type":      "task_parsed",
            "task_type": task_spec.task_type.value,
            "domain":    task_spec.domain,
            "labels":    task_spec.label_schema,
            "metric":    task_spec.evaluation_metric,
        })

        # ── Step 2: 数据引导 + 模型架构设计 并行 ─────────────────────────────
        # 数据侧（下载/清洗/拆分/格式转换）和模型侧（网络结构/损失函数选型）除了都
        # 依赖 Step 1 的 task_spec 之外互不依赖，只在"真正训练"时才需要汇合——都是
        # "发请求等 LLM/下载"的阻塞调用，用线程池重叠两条链路的等待时间。
        _emit(on_event, {"type": "step_start", "step": "data",
                         "message": "正在分析和增强数据…"})
        t_prep_start = time.time()
        with ThreadPoolExecutor(max_workers=2) as ex:
            data_future = ex.submit(engine.bootstrap, examples, task_spec, verbose=False)
            design_future = (ex.submit(_prewarm_design, model_backend, client, task_spec, len(examples))
                             if model_backend != "sklearn" else None)
            data, report = data_future.result()
            t_data_done = time.time()
            prewarmed = design_future.result() if design_future else None
            t_design_done = time.time()

        if design_future:
            _emit(on_event, {"type": "parallel_prep_done",
                             "data_seconds":   round(t_data_done - t_prep_start, 2),
                             "design_seconds": round(t_design_done - t_prep_start, 2),
                             "design_reused":  prewarmed is not None})

        _emit(on_event, {
            "type":           "data_ready",
            "total_samples":  report.total_samples,
            "augmented":      report.augmented_count,
            "quality_score":  report.quality_score,
            "warnings":       report.warnings,
            "boundary_count": len(report.boundary_samples),
            "label_dist":     report.label_distribution,
        })

        # ── Phase 2 初始化 ───────────────────────────────────────────────────
        augmentor = Augmentor(strategy="smote") if enable_phase2 else None
        deployer  = Deployer() if enable_phase2 else None

        trainer, config, first_epoch_results, arch_spec, backbone_spec = _select_and_train_backend(
            model_backend, client, task_spec, data, on_event, augmentor, prewarmed=prewarmed)
        _emit(on_event, {"type": "model_selected",
                         "model_name": config.model_name,
                         "n_epochs":   config.num_epochs})

        best_metric  = 0.0
        final_model  = None
        current_data = data
        all_epochs:  List[Dict] = []
        prev_metric  = 0.0

        # ── Steps 3–6: 训练闭环 ─────────────────────────────────────────────
        for iteration in range(max_iterations):
            _emit(on_event, {"type": "iteration_start",
                             "iteration": iteration + 1,
                             "n_data":    len(current_data)})

            if iteration == 0:
                # 第一轮训练已经在 _select_and_train_backend 里跑过了（用于验证选中的
                # 后端真的能训练成功），这里直接复用结果，不重复训练一遍
                epoch_results = first_epoch_results
            elif config.backend == ModelBackend.SKLEARN:
                epoch_results = trainer.train_with_eval(current_data, config, augmentor=augmentor)
            elif config.backend == ModelBackend.CUSTOM_NN:
                epoch_results = trainer.train_with_eval(current_data, config, arch_spec=arch_spec)
            else:
                epoch_results = trainer.train_with_eval(current_data, config, backbone_spec=backbone_spec)

            if enable_phase2 and config.backend == ModelBackend.SKLEARN and trainer.augment_report:
                r = trainer.augment_report
                _emit(on_event, {
                    "type":           "augment_done",
                    "original":       r.original_train_count,
                    "augmented":      r.augmented_train_count,
                    "balance_before": r.balance_ratio_before,
                    "balance_after":  r.balance_ratio_after,
                })

            for ep in epoch_results:
                ev = {
                    "type":       "epoch_done",
                    "iteration":  iteration + 1,
                    "epoch":      ep.epoch,
                    "val_metric": ep.val_metric,
                    "train_loss": ep.train_loss,
                    "val_loss":   ep.val_loss,
                    "per_class":  ep.per_class_metrics,
                    "confusion":  ep.confusion_highlights,
                }
                _emit(on_event, ev)
                all_epochs.append({
                    "iteration":  iteration + 1,
                    "epoch":      ep.epoch,
                    "val_metric": ep.val_metric,
                    "train_loss": ep.train_loss,
                })

            latest = epoch_results[-1]

            # 噪声门禁：低资源场景下验证集常常只有二三十条，指标差几个点很可能只是
            # 抽样波动。这里先算出"即使模型没变也会有的波动"，再判断这一轮的变化
            # 是不是真的超过了它——没超过就不更新 best_metric，避免后续 Agent 把
            # 噪声当成"这个方向有效"的证据继续往下改（见 core/robust_eval.py 的说明）
            n_val = max(1, int(len(current_data) * 0.2))
            robust = None
            # sklearn 后端 + 小数据时走真正的交叉验证：单次切分在这个规模上
            # 太不稳定（同一个模型换个切分能差十几个点），5 折的 mean±std 才是
            # 能拿来做决策的估计。NN 后端不走这条路——单次训练几十分钟，5 折
            # 不现实，只能靠 estimate_noise 的解析噪声下界（见 robust_eval 的说明）
            if should_use_cv(len(current_data),
                             backend_is_cheap=(config.backend == ModelBackend.SKLEARN)):
                try:
                    cv = cross_val_metric(
                        trainer.build_cv_estimator_factory(config),
                        [d["text"] for d in current_data],
                        trainer.label_encoder.transform([d["label"] for d in current_data]),
                        metric_name=task_spec.evaluation_metric,
                    )
                    # cross_val_metric 在"有类别样本不足 2 条"时会退化成
                    # mean=0.0/n_folds=1 的哨兵值——那种情况下不能当成真实指标用，
                    # 否则会把一个 0.0 当成"模型崩了"报给用户
                    if cv.n_folds > 1:
                        robust = cv
                except Exception:
                    robust = None      # CV 本身出问题就退回单次切分的估计，不影响主流程
            if robust is None:
                robust = estimate_noise(latest.val_metric, n_val)

            # 用可信度更高的那个估计来做"要不要采纳这一轮"的判断：有 CV 时用 CV 均值，
            # 没有时才用单次切分的 val_metric。事件里两个都发出去，metric_summary
            # 会写明这次用的是哪一种，不让用户对着两个不一样的数字猜
            improved, verdict = is_real_improvement(robust, best_metric)

            # 噪声水平也要喂给 Explainer——否则它会把噪声级别的波动当成真实趋势，
            # 给出"继续当前方向"这种基于假信号的建议
            explanation = explainer.explain(
                latest, report, task_spec, [],
                noise_note=f"当前指标 {robust.summary()}；本轮判定：{verdict}")

            _emit(on_event, {
                "type":           "iteration_done",
                "iteration":      iteration + 1,
                "val_metric":     latest.val_metric,
                # 新增三个字段供前端/Agent 判断这个指标有多可信（reduceEvent 只读
                # 已知字段，新增字段不影响现有渲染契约）
                "metric_std":     round(robust.std, 4),
                "metric_summary": robust.summary(),
                "robust_metric":  round(robust.mean, 4),   # 做决策用的那个值（有 CV 时是 CV 均值）
                "eval_method":    "cv" if robust.is_cv else "holdout",
                "significant":    improved,
                "significance_note": verdict,
                "diagnosis":      explanation.diagnosis,
                "root_cause":     explanation.root_cause,
                "recommendation": explanation.recommendation,
                "next_action":    explanation.next_action.value,
                "confidence":     explanation.confidence,
            })

            # best_metric 必须存**跟下一轮比较时同一种口径**的值：这一轮用 CV 均值
            # 判断的，就要存 CV 均值，不能存单次切分的 val_metric——两种口径混着比
            # 等于拿苹果比橘子，CV 均值通常低于单次切分的乐观估计，混用会让
            # "本轮更好"的判断系统性偏向后出现的那一轮
            if improved:
                best_metric = robust.mean
                final_model = trainer
            elif final_model is None:
                # 第一轮就没通过门禁时也要留下一个可用的模型——否则后面部署导出
                # 拿不到 trainer，整个流程会以"训练成功但没有产物"这种更糟的方式结束
                final_model = trainer
                best_metric = robust.mean

            action = explanation.next_action
            if action == NextAction.STOP_SUCCESS:
                break
            elif action == NextAction.STOP_PLATEAU:
                break
            elif action == NextAction.COLLECT_MORE_DATA and iteration < max_iterations - 1:
                # ⚠ 这里必须传 current_data（本轮累积到的全部数据），不能传 examples。
                # 传 examples 的话，每次"补充数据"都是从最初那十几条原始样本重新增强，
                # 把前面几轮积累的数据全部丢掉——真实复现：第 1 轮 197 条，执行一次
                # collect_more_data 之后直接掉回 43 条，比不补充还少得多，而且严重
                # 破坏类别平衡（实测出现过增强后只剩单一类别、下一轮训练直接崩掉）。
                # "补充数据"这个动作的语义就是**在现有基础上增加**，不是推倒重来。
                #
                # ⚠ 这里**故意没有**做"超量生成 + 按难例定向筛选"。原本是打算做的
                # （core/hard_example_miner.py 完整实现了那套逻辑），但真实测量
                # 不支持它，为一个测不出收益的功能付 2.5 倍生成 token 是坏交易：
                #
                #   A/B（Chinese_sentiment，起始 120 条 + 从 600 条池选 120 条，
                #        固定 1500 条测试集，5 个种子配对比较）：
                #        难例检索选 - 随机选 = +0.008，1σ 噪声下界 0.011，
                #        各种子差值在 -0.024 ~ +0.033 之间反复变号 → 分不出高下。
                #   去重收益（真实 Ollama qwen3.6:35b-a3b 增强 31 条）：
                #        与种子数据相似度 ≥0.92 的：0 条（最大 0.750）；
                #        新样本互相 ≥0.92 的：0 条（最大 0.698）→ 根本没有重复可去。
                #
                # 所以保留的只有**测量支持的那部分**：
                #   1. 生成量维持原样（不多花钱）；
                #   2. 去重作为零成本保险——没有重复时它什么也不删，
                #      换个啰嗦的模型真生成重复了就能兜住；
                #   3. find_hard_examples 的产出当**诊断**用：把模型到底在哪些
                #      样本上出错发给前端和 Explainer，这部分是独立成立的
                #      （out-of-fold 预测，margin 真实可信），不依赖上面那个
                #      没被证实的排序假设。
                grown, _ = engine.augment(current_data, task_spec, additional=60)
                # _augment 内部是 `augmented = list(examples)` 之后再 append，
                # 所以前 len(current_data) 条一定还是原样，尾巴才是新生成的候选
                new_candidates = grown[len(current_data):]
                if new_candidates:
                    hard, hard_err = (find_hard_examples(current_data, final_model)
                                      if final_model is not None else ([], "还没有可用的模型"))
                    # k 给全量：这里只想让它去重，不想让它按未经证实的排序砍掉样本
                    picked, mining = select_candidates(
                        new_candidates, current_data, hard, k=len(new_candidates))
                    if hard_err:
                        mining.notes.append(f"难例定位未生效：{hard_err}")
                    # 去重真的删掉了东西才改写 grown；miner 整体降级时保持原样
                    if picked and mining.n_dropped_duplicate > 0:
                        grown = current_data + picked
                    _emit(on_event, {
                        "type":            "hard_mining_done",
                        "n_hard":          mining.n_hard,
                        "n_candidates":    mining.n_candidates,
                        "n_duplicate":     mining.n_dropped_duplicate,
                        "embedding_model": mining.embedding_model,
                        "degraded_reason": mining.degraded_reason,
                        "notes":           mining.notes,
                        # 让用户看得见模型到底在哪儿出错，而不是只看到一个数字
                        "hard_samples":    [{"text": h.text[:80], "true_label": h.true_label,
                                             "pred_label": h.pred_label, "margin": h.margin,
                                             "is_error": h.is_error} for h in hard[:5]],
                    })
                # 兜底：增强结果比原来还少、或者类别塌成一类时，保留原数据不动。
                # 增强是"锦上添花"的动作，绝不能让它把本来好好的训练集弄坏
                if len(grown) >= len(current_data) and len({d["label"] for d in grown}) >= 2:
                    current_data = grown
                # flywheel 跟下面 Step 7 的部署导出是同一类限制：core/augmentor.py::flywheel
                # 假设 trainer.model/trainer.vectorizer 是 sklearn 对象（用 predict_proba
                # 给样本打置信度分），NNTrainer 没有 .model 这个属性——不加这个 backend
                # 判断的话，pretrained_nn/custom_nn 后端只要触发一次"数据不够，收集更多
                # 数据"的迭代动作就会在这里崩：AttributeError: 'NNTrainer' object has no
                # attribute 'model'（真实复现过）。跟部署导出一样，是当前版本的已知范围
                # 限制，不是遗漏——NN 后端跳过飞轮过滤，直接用增强后的全量数据继续训练。
                if enable_phase2 and augmentor and final_model and config.backend == ModelBackend.SKLEARN:
                    clean, fw = augmentor.flywheel(current_data, final_model)
                    if fw.low_confidence > 0:
                        _emit(on_event, {
                            "type":           "flywheel_done",
                            "removed":        fw.low_confidence,
                            "remaining":      fw.high_confidence,
                            "avg_confidence": fw.avg_confidence,
                        })
                        # 飞轮过滤后样本太少会导致下一轮 train_test_split 失败
                        # （数据不够分层切分/验证集甚至可能为空），此时保留过滤前的数据。
                        # 下限从原来的 max(10, n_labels*2) 提高到"至少保留 60%"——10 条
                        # 这个绝对下限低到没有意义：真按它执行，一个 200 条的训练集被
                        # 削到 10 条也算"可用"，但那时候验证集只剩 2 条，任何指标都是噪声。
                        # 另外必须检查类别数：过滤后只剩单一类别的话，下一轮 sklearn 会直接
                        # 抛 "needs samples of at least 2 classes"（真实复现过）
                        min_viable = max(int(len(current_data) * 0.6),
                                         len(task_spec.label_schema) * 4)
                        if len(clean) >= min_viable and len({d["label"] for d in clean}) >= 2:
                            current_data = clean

            elif action == NextAction.ADJUST_HYPERPARAMS and iteration < max_iterations - 1:
                # 这个分支以前带着 `and config.backend == ModelBackend.SKLEARN` 的条件，
                # 意味着神经网络后端下整个"自动调参"闭环实际上什么都不做——转一圈回来
                # 还是同一个模型同一套超参。现在按参数各自的适用范围分别处理：
                #   class_boost / loss_fn  → 所有后端通用（NN 侧走加权采样 + 损失函数工厂）
                #   C / alpha / switch_model → 仍然只对 sklearn 有意义（见 trainer._build_clf）
                is_sklearn = config.backend == ModelBackend.SKLEARN
                delta = explanation.hyperparam_delta or {}
                if not delta and is_sklearn:
                    # 无 LLM 信号时的兜底规则：收紧当前模型的正则强度，缓解过拟合。
                    # 只对 sklearn 兜底——NN 侧没有等价的"通用安全默认调整"，
                    # 瞎改损失函数比不改更糟
                    if config.model_key in ("logreg", "svm"):
                        delta = {"C": round(config.hyperparam_overrides.get(
                            "C", 1.0 if config.model_key == "logreg" else 0.5) * 0.7, 4)}
                    else:
                        delta = {"alpha": round(config.hyperparam_overrides.get("alpha", 0.001) * 1.5, 5)}

                changes = {}

                # ── 所有后端通用的调整 ──
                if not is_sklearn and delta.get("loss_fn") in SUPPORTED_LOSSES:
                    config.hyperparam_overrides["loss_fn"] = delta["loss_fn"]
                    changes["loss_fn"] = delta["loss_fn"]
                    # 损失函数自己的参数跟着一起带过去（超出范围的值由
                    # core/loss_factory.py::clamp_param 夹住，这里不重复校验）
                    for k in ("focal_gamma", "label_smoothing"):
                        if k in delta:
                            config.hyperparam_overrides[k] = delta[k]
                            changes[k] = delta[k]

                # ── 以下仅 sklearn ──
                if is_sklearn and delta.get("switch_model") in ("logreg", "svm", "sgd"):
                    config.model_key  = delta["switch_model"]
                    config.model_name = _MODEL_NAMES.get(delta["switch_model"], config.model_name)
                    changes["switch_model"] = delta["switch_model"]
                if is_sklearn and "C" in delta:
                    config.hyperparam_overrides["C"] = delta["C"]
                    changes["C"] = delta["C"]
                if is_sklearn and "alpha" in delta:
                    config.hyperparam_overrides["alpha"] = delta["alpha"]
                    changes["alpha"] = delta["alpha"]
                if isinstance(delta.get("class_boost"), dict) and delta["class_boost"]:
                    if augmentor:
                        augmentor.pending_class_boost = delta["class_boost"]
                    changes["class_boost"] = delta["class_boost"]

                if changes:
                    _emit(on_event, {"type": "hyperparams_adjusted",
                                     "iteration": iteration + 1, "changes": changes})

        # ── Step 7: 部署导出 ─────────────────────────────────────────────────
        # 注：ONNX/joblib 导出目前只支持 sklearn 后端（core/deployer.py 假设
        # trainer.vectorizer/trainer.model 是 sklearn 对象）；NN 后端训练/预测都正常工作，
        # 只是暂不生成部署包——这是当前版本的已知范围限制，不是遗漏。
        deploy_path = None
        if enable_phase2 and deployer and final_model and config.backend == ModelBackend.SKLEARN:
            _emit(on_event, {"type": "step_start", "step": "deploy",
                             "message": "正在导出部署包（含 INT8 量化）…"})
            try:
                pkg = deployer.export(final_model, task_spec, deploy_dir)
                deploy_path = pkg.package_dir
                _emit(on_event, {
                    "type":          "deploy_done",
                    "format":        pkg.export_format,
                    "model_path":    pkg.model_path,
                    "size_kb":       pkg.model_size_kb,
                    "labels":        pkg.labels,
                    "usage_example": pkg.usage_example,
                })
            except Exception as e:
                _emit(on_event, {"type": "deploy_done", "error": str(e)})

        # ── 部署后反馈闭环：记录基线指标 ─────────────────────────────────────
        feedback_baseline = {
            "metric_name":     task_spec.evaluation_metric,
            "baseline_metric": round(best_metric, 4),
            "recorded_at":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_train_samples": report.total_samples,
        }
        if deploy_path:
            fb_path = Path(deploy_path) / "feedback_baseline.json"
            fb_path.write_text(
                json.dumps(feedback_baseline, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        _emit(on_event, {"type": "feedback_check",
                         "action": "baseline_recorded",
                         **feedback_baseline})

        result = {
            "status":            "completed",
            "best_metric":       round(best_metric, 4),
            "metric_name":       task_spec.evaluation_metric,
            "labels":            task_spec.label_schema,
            "domain":            task_spec.domain,
            "n_samples":         report.total_samples,
            "epoch_history":     all_epochs,
            "deploy_path":       deploy_path,
            "feedback_baseline": feedback_baseline,
        }
        _emit(on_event, {"type": "finished", **result})
        # "_trainer" 是内部专用键：训练好的 trainer 对象要传给 api/worker.py 存进
        # task_store 供 /predict 使用，但绝不能进 pydantic 响应模型或事件流，调用方
        # 用完必须 pop 掉，不能当成 result 的公开字段传播出去
        result["_trainer"] = final_model
        return result

    except Exception as e:
        error = {"status": "error", "message": str(e)}
        _emit(on_event, {"type": "error", **error})
        return error


# ── 部署后反馈检查 ────────────────────────────────────────────────────────────

def check_feedback(
    deploy_dir:      str,
    production_metric: float,
    drift_threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Phase 2 路线图：部署后反馈 → 触发下一轮训练。

    比较生产指标与训练基线，若下降超过 drift_threshold
    则返回 retrain_recommended=True。

    Args:
        deploy_dir:          部署包目录（含 feedback_baseline.json）
        production_metric:   当前生产环境的实测指标（由监控系统提供）
        drift_threshold:     允许的最大下降幅度（默认 5%）

    Returns:
        {
          "retrain_recommended": bool,
          "drift":               float,   # 正值 = 性能下降
          "reason":              str,
          "baseline_metric":     float,
          "production_metric":   float,
        }
    """
    fb_path = Path(deploy_dir) / "feedback_baseline.json"
    if not fb_path.exists():
        return {
            "retrain_recommended": False,
            "drift":               0.0,
            "reason":              "未找到基线文件，跳过反馈检查",
            "baseline_metric":     None,
            "production_metric":   production_metric,
        }

    baseline = json.loads(fb_path.read_text(encoding="utf-8"))
    baseline_val = baseline.get("baseline_metric", 1.0)
    drift = round(baseline_val - production_metric, 4)
    retrain = drift > drift_threshold

    reason = (
        f"性能下降 {drift:.4f}（超过阈值 {drift_threshold}），建议重新训练"
        if retrain else
        f"性能稳定（下降 {drift:.4f} < 阈值 {drift_threshold}）"
    )

    return {
        "retrain_recommended": retrain,
        "drift":               drift,
        "reason":              reason,
        "baseline_metric":     baseline_val,
        "production_metric":   production_metric,
    }
