"""
core/loop.py  -  Phase 1.5 + Phase 2 集成：闭环决策代理

完整流程（Phase 1 + Phase 2）：
  Step 1  会话解析          (task_parser.py)
  Step 2  数据引导          (data_engine.py)
  Step 3  自动训练 + SMOTE  (trainer.py + augmentor.py)   ← Phase 2.1
  Step 4  解释迭代          (explainer.py)
  Step 5  迭代树记录        (iteration_tree.py)            ← Phase 2.2
  Step 6  闭环决策 + 数据飞轮 (loop.py + augmentor.py)    ← Phase 2.1
  Step 7  ONNX 部署导出     (deployer.py)                  ← Phase 2.3

Phase 2 Milestone 验收：
  用户上传 100 条样本 → 数据诊断 → SMOTE 增强训练 → 可解释迭代
  → 飞轮过滤 → 导出 ONNX/joblib 部署包
"""

from typing import List, Dict, Optional
from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich         import box as rich_box

from config import TaskSpec, DataReport, LoopState, NextAction, EpochResult
from core.llm_client      import build_llm_client
from core.task_parser     import TaskParser
from core.data_engine     import DataEngine
from core.trainer         import Trainer
from core.explainer       import Explainer
from core.augmentor       import Augmentor       # Phase 2.1
from core.iteration_tree  import IterationTree   # Phase 2.2
from core.deployer        import Deployer        # Phase 2.3


console = Console()


class AutoMLLoop:
    """
    Phase 1 + Phase 2 完整闭环。

    用法：
        loop = AutoMLLoop(api_key="sk-ant-...")
        state = loop.run(
            user_description = "帮我把客服工单按问题类型分类",
            examples         = [{"text": "...", "label": "..."}, ...],
            enable_phase2    = True,   # 开启 SMOTE + 飞轮 + 迭代树 + ONNX 导出
        )
        predictions = state.final_model.predict(["新的测试文本"])
    """

    def __init__(self, api_key: str, llm_provider: str = "anthropic", llm_model: Optional[str] = None):
        self.client    = build_llm_client(llm_provider, api_key, llm_model)
        self.parser    = TaskParser(self.client)
        self.engine    = DataEngine(self.client)
        self.explainer = Explainer(self.client)

    def run(
        self,
        user_description: str,
        examples:         List[Dict],
        max_iterations:   int   = 3,
        target_metric:    float = 0.80,
        interactive:      bool  = True,
        enable_phase2:    bool  = True,   # Phase 2: SMOTE + 飞轮 + 迭代树 + ONNX
        deploy_dir:       str   = "./deploy",
    ) -> LoopState:
        """
        启动完整的 AutoML 闭环（Phase 1 + Phase 2）。

        Args:
            user_description: 自然语言任务描述
            examples:         标注样本 [{"text": ..., "label": ...}]
            max_iterations:   最多跑几轮闭环
            target_metric:    达到此指标后自动停止
            interactive:      是否允许命令行追问
            enable_phase2:    开启 SMOTE / 飞轮 / 迭代树 / ONNX 导出
            deploy_dir:       部署包输出目录

        Returns:
            LoopState，其中 final_model 是可用于预测的 Trainer 实例
        """

        self._print_header(user_description, len(examples))

        # ── Step 1: 解析任务 ─────────────────────────────────────────────────
        console.print("\n[bold cyan]▶ Step 1 / 7 · 会话解析[/bold cyan]")
        with console.status("[dim]正在理解你的任务描述...[/dim]"):
            task_spec = self.parser.parse(user_description, interactive=interactive)
        self._print_task_spec(task_spec)

        # ── Step 2: 数据引导 ─────────────────────────────────────────────────
        console.print("\n[bold cyan]▶ Step 2 / 7 · 数据引导[/bold cyan]")
        data, report = self.engine.bootstrap(examples, task_spec, verbose=True)
        self._print_data_report(report)

        # ── Phase 2 初始化 ───────────────────────────────────────────────────
        augmentor  = Augmentor(strategy="smote") if enable_phase2 else None
        iter_tree  = IterationTree(self.client, task_spec) if enable_phase2 else None
        deployer   = Deployer() if enable_phase2 else None

        if enable_phase2:
            console.print("\n[bold green]⚡ Phase 2 已启用：SMOTE + 数据飞轮 + 迭代树 + ONNX 导出[/bold green]")

        # ── Step 3–6: 训练闭环 ───────────────────────────────────────────────
        console.print("\n[bold cyan]▶ Steps 3–6 · 训练 → 增强 → 解释 → 决策（闭环）[/bold cyan]")

        trainer = Trainer(task_spec)
        config  = trainer.auto_select(len(data))
        console.print(f"  ▷ 自动选择模型：[yellow]{config.model_name}[/yellow]")

        state = LoopState(
            iteration    = 0,
            task_spec    = task_spec,
            data_report  = report,
            epoch_results  = [],
            explanations   = [],
            best_metric    = 0.0,
            plateau_count  = 0,
            final_model    = None,
        )

        current_data = data
        stop_reason  = "max_iterations"
        prev_metric  = 0.0

        for iteration in range(max_iterations):
            state.iteration = iteration + 1
            console.print(f"\n  [bold]── 迭代 {iteration + 1} / {max_iterations} ──[/bold]")

            # 训练（含 Phase 2.1 SMOTE）
            with console.status(f"  训练中（{config.num_epochs} epoch + SMOTE）..."):
                epoch_results = trainer.train_with_eval(
                    current_data, config, augmentor=augmentor
                )

            # 打印增强报告（Phase 2.1）
            if enable_phase2 and trainer.augment_report:
                r = trainer.augment_report
                console.print(
                    f"  ⚡ SMOTE：训练集 {r.original_train_count} → {r.augmented_train_count} 条  "
                    f"平衡度 {r.balance_ratio_before:.2f} → [green]{r.balance_ratio_after:.2f}[/green]"
                )

            # 打印学习曲线
            self._print_curve(epoch_results, task_spec.evaluation_metric)

            latest = epoch_results[-1]
            state.epoch_results.extend(epoch_results)

            # 解释
            with console.status("  生成迭代解释..."):
                explanation = self.explainer.explain(
                    current = latest,
                    report  = report,
                    spec    = task_spec,
                    history = state.epoch_results[:-len(epoch_results)],
                )
            state.explanations.append(explanation)

            # ── Phase 2.2: 迭代树节点 ────────────────────────────────────────
            if iter_tree is not None:
                action_label = explanation.next_action.value
                iter_tree.add_node(
                    iteration     = state.iteration,
                    action        = action_label if iteration > 0 else "initial",
                    metric_before = prev_metric,
                    metric_after  = latest.val_metric,
                    explanation   = explanation.diagnosis,
                )

            # 更新最佳模型
            if latest.val_metric > state.best_metric:
                state.best_metric = latest.val_metric
                state.final_model = trainer
                state.plateau_count = 0
            else:
                state.plateau_count += 1

            prev_metric = latest.val_metric

            # 打印解释
            self._print_explanation(explanation, latest, task_spec)

            # ── 决策 ─────────────────────────────────────────────────────────
            action = explanation.next_action

            if action == NextAction.STOP_SUCCESS:
                stop_reason = "success"
                console.print(Panel(
                    f"✅ 任务达成！\n{task_spec.evaluation_metric} = "
                    f"[bold green]{latest.val_metric:.2%}[/bold green]",
                    border_style="green",
                ))
                break

            elif action == NextAction.STOP_PLATEAU:
                stop_reason = "plateau"
                console.print(Panel(
                    f"⏹  已到瓶颈，停止训练。\n"
                    f"最佳 {task_spec.evaluation_metric} = {state.best_metric:.2%}\n"
                    f"建议：{explanation.recommendation}",
                    border_style="yellow",
                ))
                break

            elif action == NextAction.COLLECT_MORE_DATA:
                if iteration < max_iterations - 1:
                    console.print("  📥 决策：追加数据...")
                    current_data, added = self.engine.augment(
                        examples, task_spec, additional=60
                    )
                    console.print(f"     数据量 {len(data)} → {len(current_data)} 条（+{added}）")

                    # Phase 2.1 数据飞轮：过滤低置信度样本
                    if enable_phase2 and augmentor and state.final_model:
                        clean, fw_report = augmentor.flywheel(current_data, state.final_model)
                        if fw_report.low_confidence > 0:
                            console.print(
                                f"  🔄 数据飞轮：移除 {fw_report.low_confidence} 条低置信度样本"
                                f"（置信度 < {fw_report.threshold_used}）"
                            )
                            # 飞轮过滤后样本太少会导致下一轮 train_test_split 失败，此时保留过滤前的数据
                            min_viable = max(10, len(task_spec.label_schema) * 2)
                            if len(clean) >= min_viable:
                                current_data = clean

            elif action == NextAction.ADJUST_HYPERPARAMS:
                console.print("  ⚙️  决策：微调超参数...")
                config.num_epochs = min(config.num_epochs + 2, 15)

            else:
                console.print("  ▶ 决策：继续训练...")

        # ── Step 7: Phase 2.2 迭代树归因 + Phase 2.3 ONNX 导出 ───────────────
        if enable_phase2:
            # 迭代树汇总
            if iter_tree and iter_tree.nodes:
                console.print("\n[bold cyan]▶ Step 7a / 7 · 迭代树归因分析[/bold cyan]")
                with console.status("  生成贡献归因..."):
                    tree_summary = iter_tree.summarize()
                self._print_tree_summary(tree_summary)
                # 保存迭代树 JSON
                import json, os
                os.makedirs("outputs", exist_ok=True)
                with open("outputs/iteration_tree.json", "w", encoding="utf-8") as f:
                    json.dump(tree_summary, f, ensure_ascii=False, indent=2)
                console.print("  [dim]迭代树已保存：outputs/iteration_tree.json[/dim]")

            # ONNX / joblib 导出
            if deployer and state.final_model:
                console.print("\n[bold cyan]▶ Step 7b / 7 · 部署包导出[/bold cyan]")
                with console.status("  导出模型..."):
                    try:
                        pkg = deployer.export(state.final_model, task_spec, deploy_dir)
                        self._print_deploy_package(pkg)
                        state.deploy_package = pkg   # 附加到 state（动态属性）
                    except Exception as e:
                        console.print(f"  [yellow]⚠  导出失败（不影响模型使用）：{e}[/yellow]")

        # ── 最终报告 ─────────────────────────────────────────────────────────
        self._print_final_report(state, stop_reason)
        return state

    # ── 打印工具方法 ─────────────────────────────────────────────────────────

    def _print_header(self, description: str, n_examples: int):
        console.print(Panel(
            f"[bold]AutoML 闭环启动[/bold]\n\n"
            f"任务描述：{description}\n"
            f"初始样本：{n_examples} 条",
            title="🚀 Phase 1 原型 · 真实执行 + 小数据专项 + 可解释迭代",
            border_style="blue",
        ))

    def _print_task_spec(self, spec: TaskSpec):
        t = Table(box=rich_box.SIMPLE, show_header=False, padding=(0, 1))
        t.add_row("[dim]任务类型[/dim]", f"[cyan]{spec.task_type.value}[/cyan]")
        t.add_row("[dim]业务领域[/dim]", spec.domain)
        t.add_row("[dim]标签体系[/dim]", "  ".join(f"[white]{l}[/white]" for l in spec.label_schema))
        t.add_row("[dim]评估指标[/dim]", spec.evaluation_metric)
        t.add_row("[dim]数据语言[/dim]", spec.language)
        console.print(t)

    def _print_data_report(self, report: DataReport):
        console.print(
            f"  总样本：{report.total_samples} 条"
            f"（原始 {report.total_samples - report.augmented_count} + 增强 {report.augmented_count}）"
        )
        console.print(f"  质量评分：{report.quality_score:.2f} / 1.00")

        dist_str = "  ".join(
            f"{label}:{cnt}" for label, cnt in report.label_distribution.items()
        )
        console.print(f"  标签分布：{dist_str}")

        for w in report.warnings:
            console.print(f"  [yellow]⚠  {w}[/yellow]")

        if report.boundary_samples:
            console.print(
                f"  [red]🔴 {len(report.boundary_samples)} 个边界样本（建议人工复查）[/red]"
            )

    def _print_curve(self, results: List[EpochResult], metric_name: str):
        metrics    = [r.val_metric for r in results]
        bar_chars  = "▁▂▃▄▅▆▇█"
        lo, hi     = min(metrics), max(metrics)

        bars = ""
        for m in metrics:
            idx   = int((m - lo) / (hi - lo + 1e-9) * 7)
            bars += bar_chars[min(idx, 7)]

        final = metrics[-1]
        color = "green" if final >= 0.80 else ("yellow" if final >= 0.60 else "red")
        console.print(
            f"\n  {metric_name} 学习曲线：{bars}  "
            f"最终 [{color}]{final:.4f}[/{color}]"
        )

        # per-class 最弱项提示
        latest = results[-1]
        if latest.per_class_metrics:
            weakest = min(latest.per_class_metrics.items(), key=lambda x: x[1])
            if weakest[1] < 0.60:
                console.print(
                    f"  [yellow]△ 最弱类别：'{weakest[0]}'  F1 = {weakest[1]:.4f}[/yellow]"
                )

    def _print_explanation(
        self,
        exp:    "IterationExplanation",
        latest: EpochResult,
        spec:   TaskSpec,
    ):
        action_color = {
            NextAction.STOP_SUCCESS:       "green",
            NextAction.STOP_PLATEAU:       "yellow",
            NextAction.COLLECT_MORE_DATA:  "blue",
            NextAction.CONTINUE_TRAINING:  "cyan",
            NextAction.ADJUST_HYPERPARAMS: "magenta",
        }.get(exp.next_action, "white")

        action_label = {
            NextAction.STOP_SUCCESS:       "✅ 达成目标，停止",
            NextAction.STOP_PLATEAU:       "⏹  触达瓶颈，停止",
            NextAction.COLLECT_MORE_DATA:  "📥 追加训练数据",
            NextAction.CONTINUE_TRAINING:  "▶  继续训练",
            NextAction.ADJUST_HYPERPARAMS: "⚙️  调整超参数",
        }.get(exp.next_action, exp.next_action.value)

        body = (
            f"[bold]诊断：[/bold]{exp.diagnosis}\n\n"
            f"[bold]原因：[/bold]{exp.root_cause}\n\n"
            f"[bold]建议：[/bold]{exp.recommendation}\n\n"
            f"[bold]决策：[/bold][{action_color}]{action_label}[/{action_color}]"
            f"  [dim](置信度 {exp.confidence:.0%})[/dim]"
        )

        console.print(Panel(
            body,
            title=f"[dim]第 {latest.epoch} 轮解释  {spec.evaluation_metric} = {latest.val_metric:.4f}[/dim]",
            border_style=action_color,
        ))

    def _print_tree_summary(self, summary: dict):
        attr = summary.get("attribution", {})
        console.print(Panel(
            f"[bold]最大贡献改动：[/bold]{attr.get('top_contributor', '—')}"
            f"  (+{attr.get('top_contribution_delta', 0):.4f})\n\n"
            f"[bold]训练历程：[/bold]{attr.get('summary', '—')}\n\n"
            f"[bold]核心洞察：[/bold]{attr.get('key_insight', '—')}",
            title="🌳 迭代树归因",
            border_style="blue",
        ))

    def _print_deploy_package(self, pkg):
        size_str = f"{pkg.model_size_kb:.1f} KB"
        console.print(Panel(
            f"[bold]格式：[/bold]{pkg.export_format.upper()}\n"
            f"[bold]模型大小：[/bold]{size_str}\n"
            f"[bold]标签：[/bold]{', '.join(pkg.labels)}\n"
            f"[bold]模型路径：[/bold]{pkg.model_path}\n"
            f"[bold]推理脚本：[/bold]{pkg.inference_script}\n\n"
            f"[bold]快速使用：[/bold]\n[dim]{pkg.usage_example}[/dim]",
            title="📦 部署包",
            border_style="green",
        ))

    def _print_final_report(self, state: LoopState, stop_reason: str):
        reason_map = {
            "success":        "✅ 达成目标指标",
            "plateau":        "⏹  训练遇到瓶颈",
            "max_iterations": "🔄 已达最大迭代数",
        }
        reason_str = reason_map.get(stop_reason, stop_reason)
        history_str = " → ".join(
            f"{e.val_metric:.4f}"
            for e in state.epoch_results[::max(1, len(state.epoch_results) // 8)]
        )
        deploy_note = f"\n部署包：{state.deploy_package.package_dir}" \
                      if hasattr(state, "deploy_package") else ""

        console.print("\n")
        console.print(Panel(
            f"[bold]AutoML 完成（Phase 1 + Phase 2）[/bold]\n\n"
            f"停止原因：{reason_str}\n"
            f"完成迭代：{state.iteration} 轮\n"
            f"最佳 {state.task_spec.evaluation_metric}："
            f"[bold green]{state.best_metric:.2%}[/bold green]\n"
            f"训练样本：{state.data_report.total_samples} 条\n"
            f"指标历程：{history_str}{deploy_note}\n\n"
            f"[dim]state.final_model.predict([\"文本\"]) 可直接预测[/dim]",
            title="📊 训练报告",
            border_style="green",
        ))
