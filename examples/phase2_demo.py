#!/usr/bin/env python3
"""
examples/phase2_demo.py  -  Phase 2 完整演示

验收 Milestone 2：
  "用户上传 100 条样本，系统完整走完：
   数据诊断 → SMOTE 增强训练 → 可解释迭代 → 数据飞轮 → 导出 ONNX 部署包"

直接运行：
    cd automl_agent
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/phase2_demo.py

Phase 2 新增展示（相比 Phase 1 demo）：
  ⚡ SMOTE：TF-IDF 特征空间过采样，修正类别不平衡
  🔄 数据飞轮：训练后过滤低置信度样本，自我净化
  🌳 迭代树：每步改动的因果解释 + 反事实分析
  📦 部署包：导出 ONNX/joblib + 独立推理脚本
"""

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.rule    import Rule

console = Console()


# ── 配置 ─────────────────────────────────────────────────────────────────────

TASK = (
    "帮我把客服工单按问题类型分类，类别包括："
    "账号问题、支付问题、物流问题、产品质量、其他。"
    "这是一个电商平台的工单系统，每天几千条，需要自动分类。"
)

DATA_PATH = ROOT / "examples" / "data" / "customer_tickets.jsonl"

PREDICT_TEXTS = [
    "我的登录密码忘记了，怎么重置",
    "收到的手机屏幕有裂纹，明显是翻新机",
    "快递三天没有物流更新，是不是丢失了",
    "付款成功但订单一直显示待支付",
    "需要开一张电子发票，在哪里申请",
    "账号被封了，我什么都没做，怎么申诉",
]


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]❌ 请先 export ANTHROPIC_API_KEY=sk-ant-...[/red]")
        sys.exit(1)

    if not DATA_PATH.exists():
        console.print(f"[red]❌ 数据文件不存在：{DATA_PATH}[/red]")
        sys.exit(1)

    # 加载数据
    examples = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    console.print(Rule("[bold blue]AutoML Agent · Phase 2 Demo[/bold blue]"))
    console.print()
    console.print(f"[dim]Phase 2 新增：SMOTE + 数据飞轮 + 迭代树归因 + ONNX 导出[/dim]")
    console.print(f"[dim]样本数量：{len(examples)} 条（小数据专项验证）[/dim]")
    console.print()

    # ── 启动 Phase 2 完整闭环 ─────────────────────────────────────────────────
    from core.loop import AutoMLLoop

    loop  = AutoMLLoop(api_key=api_key)
    state = loop.run(
        user_description = TASK,
        examples         = examples,
        max_iterations   = 3,
        target_metric    = 0.78,
        interactive      = False,
        enable_phase2    = True,         # ← Phase 2 全部开启
        deploy_dir       = str(ROOT / "deploy"),
    )

    # ── 预测展示 ──────────────────────────────────────────────────────────────
    if state.final_model:
        console.print()
        console.print(Rule("[bold green]预测验证[/bold green]"))
        console.print()

        predictions = state.final_model.predict(PREDICT_TEXTS)
        probas      = state.final_model.predict_proba(PREDICT_TEXTS)

        from rich.table import Table
        from rich       import box as rich_box

        t = Table(box=rich_box.SIMPLE_HEAD)
        t.add_column("输入文本",  style="dim",  width=30)
        t.add_column("预测类别",  style="bold", width=14)
        t.add_column("置信度",    justify="right", width=8)

        for text, pred, prob in zip(PREDICT_TEXTS, predictions, probas):
            conf = f"{max(prob.values()):.1%}" if prob else "—"
            t.add_row(text[:28] + ("…" if len(text) > 28 else ""), pred, conf)

        console.print(t)

    # ── 迭代树 JSON ───────────────────────────────────────────────────────────
    tree_path = ROOT / "outputs" / "iteration_tree.json"
    if tree_path.exists():
        console.print()
        console.print(Rule("[bold blue]迭代树快照（outputs/iteration_tree.json）[/bold blue]"))
        with open(tree_path, encoding="utf-8") as f:
            tree = json.load(f)
        for node in tree.get("nodes", []):
            delta_str = f"+{node['delta']:.4f}" if node['delta'] >= 0 else f"{node['delta']:.4f}"
            color     = "green" if node['delta'] > 0 else ("yellow" if node['delta'] == 0 else "red")
            console.print(
                f"  迭代 {node['iteration']:02d}  [{color}]{delta_str}[/{color}]  "
                f"[dim]{node['action']:20s}[/dim]  {node['explanation'][:50]}"
            )
        attr = tree.get("attribution", {})
        if attr:
            console.print(f"\n  [bold]核心洞察：[/bold]{attr.get('key_insight', '—')}")

    # ── 部署包 ────────────────────────────────────────────────────────────────
    deploy_dir = ROOT / "deploy"
    if deploy_dir.exists():
        files = list(deploy_dir.iterdir())
        console.print()
        console.print(Rule("[bold green]部署包（./deploy/）[/bold green]"))
        for f in files:
            size = f.stat().st_size / 1024
            console.print(f"  {f.name:25s}  {size:7.1f} KB")

    # ── 保存训练记录 ──────────────────────────────────────────────────────────
    try:
        from utils.io_utils import save_state
        out = save_state(state, output_dir=str(ROOT / "outputs"))
        console.print(f"\n[dim]训练记录已保存：{out}[/dim]")
    except Exception as e:
        console.print(f"\n[dim]保存失败（不影响结果）：{e}[/dim]")


if __name__ == "__main__":
    main()
