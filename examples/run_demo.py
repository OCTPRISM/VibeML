#!/usr/bin/env python3
"""
examples/run_demo.py
====================
开箱即用的演示脚本——不需要任何命令行参数。

直接运行：
    cd automl_agent
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/run_demo.py

演示内容：
    30 条客服工单（故意很少）→ 完整 AutoML 闭环
    展示三个差异化能力：
      1. 零门槛会话 —— 自然语言描述任务
      2. 小数据专项 —— 30 条→自动增强→可训练
      3. 可解释迭代 —— 每轮给出文字化诊断
"""

import os
import sys
from pathlib import Path

# 把项目根目录加入 path（适应从 examples/ 子目录运行的场景）
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.rule    import Rule

console = Console()


# ── 配置 ─────────────────────────────────────────────────────────────────────

TASK_DESCRIPTION = (
    "帮我把客服工单按问题类型分类。"
    "类别包括：账号问题、支付问题、物流问题、产品质量、其他。"
    "我们是一个电商平台，每天有几千条工单，想用 AI 自动分类减少人工成本。"
)

DATA_PATH = ROOT / "examples" / "data" / "customer_tickets.jsonl"

# 分类完成后，用这几条文本做预测展示
PREDICT_TEXTS = [
    "我的登录密码忘记了，怎么重置",
    "买的手机收到是坏的，屏幕有裂纹",
    "快递三天没动静，是不是丢了",
    "付款成功但订单状态还是待支付",
    "想开一张电子发票",
]


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    # 检查 API Key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]❌ 请先设置 ANTHROPIC_API_KEY 环境变量[/red]")
        console.print("   export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # 检查数据文件
    if not DATA_PATH.exists():
        console.print(f"[red]❌ Demo 数据文件不存在：{DATA_PATH}[/red]")
        sys.exit(1)

    # 加载数据
    import json
    examples = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    console.print(Rule("[bold blue]AutoML Agent · Phase 1 Demo[/bold blue]"))
    console.print()
    console.print(f"[dim]数据文件：{DATA_PATH}[/dim]")
    console.print(f"[dim]样本数量：{len(examples)} 条（故意很少，演示小数据能力）[/dim]")
    console.print()

    # ── 启动闭环 ──────────────────────────────────────────────────────────────
    from core.loop import AutoMLLoop

    loop  = AutoMLLoop(api_key=api_key)
    state = loop.run(
        user_description = TASK_DESCRIPTION,
        examples         = examples,
        max_iterations   = 3,
        target_metric    = 0.80,
        interactive      = False,   # Demo 模式：不追问，全自动
    )

    # ── 预测展示 ──────────────────────────────────────────────────────────────
    if state.final_model:
        console.print()
        console.print(Rule("[bold green]预测展示[/bold green]"))
        console.print()

        predictions = state.final_model.predict(PREDICT_TEXTS)
        probas      = state.final_model.predict_proba(PREDICT_TEXTS)

        from rich.table import Table
        from rich       import box as rich_box

        t = Table(box=rich_box.SIMPLE_HEAD, show_lines=False)
        t.add_column("输入文本",  style="dim",    width=32)
        t.add_column("预测类别",  style="bold",   width=14)
        t.add_column("最高置信度", justify="right", width=10)

        for text, pred, prob in zip(PREDICT_TEXTS, predictions, probas):
            conf = f"{max(prob.values()):.1%}" if prob else "—"
            t.add_row(
                text[:30] + ("…" if len(text) > 30 else ""),
                pred,
                conf,
            )

        console.print(t)

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    try:
        from utils.io_utils import save_state
        out_path = save_state(state, output_dir=str(ROOT / "outputs"))
        console.print(f"\n[dim]训练记录已保存：{out_path}[/dim]")
    except Exception as e:
        console.print(f"\n[dim]保存失败（不影响结果）：{e}[/dim]")


if __name__ == "__main__":
    main()
