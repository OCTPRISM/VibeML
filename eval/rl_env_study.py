"""
eval/rl_env_study.py  -  实验 4：LLM 生成强化学习环境的成功率研究

手写一组自然语言 RL 任务描述（刻意不直接点名知名 Gym 环境，避免 LLM 单纯背诵记忆的
标准实现），真实跑 core/rl_pipeline.py::run_rl_pipeline（真实 Ollama + 真实
stable-baselines3 训练），记录：
  - 环境代码首次/二次沙盒通过率（core/rl_sandbox.py::validate()）
  - 训练是否完整跑完（未超时/未崩溃）
  - reward 从第一段训练到最后一段训练的提升幅度（epoch_history 首尾差）
  - 每条任务耗时

运行：python -m eval.rl_env_study
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

OUT_DIR = Path(__file__).parent.parent / "outputs" / "exp4_rl_env_study"

# 任务描述：覆盖导航、资源调度、简单博弈、序列决策等结构，不直接点名 CartPole/MountainCar
# 等知名 Gym 环境，测的是"LLM 能不能根据业务化的自然语言描述自己设计环境"而不是背诵
TASKS: List[Dict[str, str]] = [
    {"id": "corridor_walk", "domain": "导航",
     "description": "一个智能体在一条走廊上左右移动，走到走廊尽头的出口就算成功，走到另一头的陷阱就算失败。"},
    {"id": "grid_treasure", "domain": "导航",
     "description": "一个二维网格地图里有一个宝藏和几个陷阱，智能体每步能上下左右移动一格，找到宝藏得高分，踩到陷阱扣分并结束。"},
    {"id": "room_temperature", "domain": "控制",
     "description": "控制一个房间的空调，目标是把温度维持在舒适区间内，同时尽量少开空调以节省电费。"},
    {"id": "inventory_restock", "domain": "资源调度",
     "description": "一个仓库每天要决定进多少货，进多了要付仓储费，进少了会缺货损失订单，需求量每天随机波动，目标是让总成本最低。"},
    {"id": "ad_budget_allocation", "domain": "资源调度",
     "description": "每天有一笔广告预算，要决定投放在哪个渠道，不同渠道的转化效果不确定，目标是让总点击量最大化。"},
    {"id": "elevator_dispatch", "domain": "调度",
     "description": "一栋楼里有乘客在不同楼层等电梯，电梯每次要决定往上还是往下走，目标是让乘客平均等待时间最短。"},
    {"id": "guessing_game", "domain": "简单博弈",
     "description": "有一个隐藏的目标数字，智能体每次猜一个数，系统会告诉它猜大了还是猜小了，目标是尽快猜中。"},
    {"id": "balance_pole", "domain": "控制",
     "description": "一根杆子立在一个可以左右移动的底座上，底座需要通过左右移动让杆子尽量保持竖直不倒。"},
    {"id": "traffic_light", "domain": "调度",
     "description": "一个十字路口的红绿灯需要决定东西向还是南北向放行，目标是让所有方向排队的车辆总等待时间最短。"},
    {"id": "betting_game", "domain": "简单博弈",
     "description": "有一笔初始筹码，每一轮可以选择下注大小，赢的概率和赔率固定，目标是若干轮后筹码尽量多。"},
    {"id": "drone_altitude", "domain": "控制",
     "description": "一架无人机需要通过调整升力来保持在目标高度悬停，风力会随机让它偏离目标高度。"},
    {"id": "queue_routing", "domain": "调度",
     "description": "一个客服中心有多个窗口，新来的顾客需要被分配到某个窗口排队，各窗口当前排队长度不同，目标是让顾客平均等待时间最短。"},
    {"id": "irrigation_decision", "domain": "资源调度",
     "description": "一块农田需要根据土壤湿度决定要不要灌溉，灌溉多了浪费水，灌溉少了作物会枯萎，目标是用最少的水让作物保持健康。"},
    {"id": "obstacle_avoidance_2d", "domain": "导航",
     "description": "一个智能体在二维平面上移动，需要绕开随机出现的障碍物到达目标点，撞到障碍物要扣分。"},
    {"id": "resource_gathering", "domain": "简单博弈",
     "description": "一个智能体在地图上移动收集散落的资源点，每收集一个得分，地图有限定的步数上限。"},
    {"id": "reservoir_scheduling", "domain": "资源调度",
     "description": "一个水库需要每天决定放多少水，放太多下游可能干旱缺水，放太少水库可能溢出，目标是全年总体损失最小。"},
]


def _summarize_epochs(epoch_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not epoch_history:
        return {"first_reward": None, "last_reward": None, "improvement": None}
    first = epoch_history[0]["val_metric"]
    last  = epoch_history[-1]["val_metric"]
    return {"first_reward": first, "last_reward": last, "improvement": round(last - first, 4)}


def _run_one(task: Dict[str, str]) -> Dict[str, Any]:
    from core.rl_pipeline import run_rl_pipeline

    events: List[Dict[str, Any]] = []

    def on_event(ev):
        events.append(ev)

    t0 = time.time()
    result = run_rl_pipeline(
        api_key="", description=task["description"], env_description="",
        max_iterations=1, deploy_dir=f"/tmp/rl_env_study_deploy/{task['id']}",
        on_event=on_event, llm_provider="ollama", llm_model=None,
    )
    elapsed = time.time() - t0
    result.pop("_trainer", None)

    fallback_events = [e for e in events if e.get("type") == "nn_codegen_fallback" and e.get("stage") == "rl_env"]
    arch_events = [e for e in events if e.get("type") == "arch_designed" and e.get("mode") == "rl_env"]
    sandbox_passed_first_try = len(fallback_events) == 0 and len(arch_events) >= 1

    epoch_summary = _summarize_epochs(result.get("epoch_history", []))

    return {
        "task_id": task["id"],
        "domain": task["domain"],
        "description": task["description"],
        "wall_clock_s": round(elapsed, 1),
        "status": result.get("status"),
        "error": result.get("message") if result.get("status") == "error" else None,
        "sandbox_passed_first_try": sandbox_passed_first_try,
        "n_env_design_attempts": max(len(arch_events), 1),
        "env_class_name": arch_events[-1].get("class_name") if arch_events else None,
        "best_metric_episode_reward_mean": result.get("best_metric"),
        **epoch_summary,
        "deploy_path": result.get("deploy_path"),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_runs: List[Dict[str, Any]] = []

    for i, task in enumerate(TASKS):
        print(f"\n=== [{i+1}/{len(TASKS)}] {task['id']} ({task['domain']}) ===")
        try:
            r = _run_one(task)
        except Exception as e:
            r = {"task_id": task["id"], "domain": task["domain"], "status": "exception", "error": str(e)}
        print(json.dumps(r, ensure_ascii=False, indent=2))
        all_runs.append(r)
        (OUT_DIR / "raw_runs.json").write_text(json.dumps(all_runs, ensure_ascii=False, indent=2), encoding="utf-8")

    completed = [r for r in all_runs if r.get("status") == "completed"]
    sandbox_first_try = [r for r in all_runs if r.get("sandbox_passed_first_try")]
    improvements = [r["improvement"] for r in completed if r.get("improvement") is not None]
    positive_improvements = [x for x in improvements if x > 0]

    summary = {
        "n_tasks": len(TASKS),
        "n_completed": len(completed),
        "completion_rate": round(len(completed) / len(TASKS), 4),
        "sandbox_first_try_pass_rate": round(len(sandbox_first_try) / len(TASKS), 4),
        "mean_wall_clock_s": round(sum(r["wall_clock_s"] for r in all_runs if "wall_clock_s" in r) / len(all_runs), 1),
        "n_runs_with_positive_reward_improvement": len(positive_improvements),
        "n_runs_with_improvement_measured": len(improvements),
        "positive_improvement_rate_among_completed": round(len(positive_improvements) / len(improvements), 4) if improvements else None,
        "mean_improvement_among_completed": round(sum(improvements) / len(improvements), 4) if improvements else None,
        "failed_tasks": [{"task_id": r["task_id"], "error": r.get("error")} for r in all_runs if r.get("status") != "completed"],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 汇总已保存：{OUT_DIR / 'summary.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
