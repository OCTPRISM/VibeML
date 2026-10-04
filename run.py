#!/usr/bin/env python3
"""
run.py - AutoML Agent Phase 1 原型入口

用法：
  # Demo 模式（用内置客服工单数据）
  python run.py --demo

  # 指定任务和数据文件
  python run.py \\
    --task "帮我把客服工单按问题类型分类：账号、支付、物流、产品质量、其他" \\
    --data examples/data/customer_tickets.jsonl

  # 交互模式（会引导你描述任务）
  python run.py --data examples/data/customer_tickets.jsonl

环境要求：
  export ANTHROPIC_API_KEY=sk-ant-...
  pip install -r requirements.txt
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_jsonl(path: str):
    """加载 JSONL 格式的样本文件"""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                examples.append(obj)
            except json.JSONDecodeError as e:
                print(f"⚠️  第 {line_no} 行 JSON 解析失败，已跳过：{e}")
    return examples


def validate_examples(examples):
    """检查样本格式是否正确"""
    errors = []
    for i, e in enumerate(examples):
        if "text" not in e:
            errors.append(f"第 {i+1} 条缺少 'text' 字段")
        if "label" not in e:
            errors.append(f"第 {i+1} 条缺少 'label' 字段")
    if errors:
        for err in errors[:5]:
            print(f"❌ {err}")
        if len(errors) > 5:
            print(f"   ...（共 {len(errors)} 个问题）")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="AutoML Agent · 用自然语言训练专属 AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task",     type=str,  help="自然语言任务描述")
    parser.add_argument("--data",     type=str,  help="训练数据文件（JSONL）")
    parser.add_argument("--demo",     action="store_true", help="运行内置客服 demo")
    parser.add_argument("--api-key",  type=str,  help="Anthropic API Key")
    parser.add_argument("--max-iter", type=int,  default=3, help="最大闭环迭代次数（默认 3）")
    parser.add_argument("--target",   type=float,default=0.80, help="目标指标值（默认 0.80）")
    parser.add_argument("--no-augment", action="store_true", help="跳过数据增强（快速测试用）")
    parser.add_argument("--predict",  type=str,  nargs="+", help="训练后对指定文本预测")
    args = parser.parse_args()

    # ── API Key ──────────────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ 需要提供 Anthropic API Key")
        print()
        print("   方法 1：export ANTHROPIC_API_KEY=sk-ant-...")
        print("   方法 2：python run.py --api-key sk-ant-...")
        sys.exit(1)

    # ── 准备任务描述和数据 ───────────────────────────────────────────────────
    if args.demo:
        task = (
            "帮我把客服工单按问题类型分类，类别包括："
            "账号问题、支付问题、物流问题、产品质量、其他"
        )
        demo_path = Path(__file__).parent / "examples" / "data" / "customer_tickets.jsonl"
        if not demo_path.exists():
            print(f"❌ Demo 数据文件不存在：{demo_path}")
            sys.exit(1)
        examples = load_jsonl(str(demo_path))
        print(f"✅ Demo 模式：加载 {len(examples)} 条客服工单样本")

    elif args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"❌ 数据文件不存在：{args.data}")
            sys.exit(1)
        examples = load_jsonl(str(data_path))
        validate_examples(examples)
        print(f"✅ 加载 {len(examples)} 条样本：{args.data}")

        if args.task:
            task = args.task
        else:
            print("\n🤖 请描述你想解决的问题（例如：帮我把评论按情感分类，分为正面和负面）：")
            task = input("任务描述：").strip()
            if not task:
                print("❌ 任务描述不能为空")
                sys.exit(1)

    else:
        parser.print_help()
        print("\n提示：至少需要提供 --demo 或 --data 参数")
        sys.exit(1)

    # ── 启动闭环 ─────────────────────────────────────────────────────────────
    from core.loop import AutoMLLoop

    loop = AutoMLLoop(api_key=api_key)
    state = loop.run(
        user_description = task,
        examples         = examples,
        max_iterations   = args.max_iter,
        target_metric    = args.target,
        interactive      = True,
    )

    # ── 可选：预测 ───────────────────────────────────────────────────────────
    if args.predict and state.final_model:
        print("\n" + "─" * 60)
        print("📌 预测结果：")
        predictions = state.final_model.predict(args.predict)
        for text, pred in zip(args.predict, predictions):
            print(f"  {pred:12s}  ←  {text}")

    return state


if __name__ == "__main__":
    main()
