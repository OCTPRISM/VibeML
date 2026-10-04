"""
core/iteration_tree.py  -  Phase 2.2：迭代树 + 反事实分析

差异化核心（Phase 2 路线图 §2.4，对应文献验证空白 #2）：
  现有 AutoML 系统迭代过程完全黑盒——只报告最终指标，
  用户无法知道"哪步改动带来了多少提升"。

本模块实现三件事：
  1. 迭代树：把每次闭环迭代记录为树节点，追踪改动→效果的因果链
  2. 贡献归因：量化每次改动贡献了多少性能提升（简单差分 + LLM 解读）
  3. 反事实分析：LLM 回答"如果不做 X 改动，结果会怎样"

输出可序列化为 JSON，供前端迭代树可视化使用（Phase 3）。
"""

from __future__ import annotations

import json
import uuid
import re
from typing import List, Optional, Dict
from dataclasses import asdict

from core.llm_client import LLMClient

from config import IterationNode, TaskSpec


# ── Prompt ───────────────────────────────────────────────────────────────────

COUNTERFACTUAL_SYSTEM = """你是一个因果推理专家，负责分析机器学习训练中每次改动的影响。

用户会提供一棵迭代树，以及一个特定的节点（某次改动）。
你的任务是：
1. 解释这次改动为什么能（或不能）带来性能提升
2. 推断如果不做这次改动，结果会是什么

输出 JSON（只输出 JSON）：
{
  "causal_explanation": "这次改动带来提升的因果解释（用比喻，50字以内）",
  "counterfactual": "如果不做此改动，预期指标会在X附近，因为…（50字以内）",
  "confidence": 0到1
}"""

ATTRIBUTION_SYSTEM = """你是一个机器学习优化顾问，负责总结整个训练过程中哪次改动贡献最大。

给你一个完整的迭代记录，分析：
1. 哪次改动带来了最大的性能提升
2. 哪次改动是"锦上添花"，哪次是"雪中送炭"
3. 如果只能做一件事，应该做什么

输出 JSON（只输出 JSON）：
{
  "top_contributor": "改动描述",
  "top_contribution_delta": 0.05,
  "summary": "整个训练历程的一句话总结",
  "key_insight": "最重要的一条经验（用类比让非技术人员理解）"
}"""


# ── IterationTree ─────────────────────────────────────────────────────────────

class IterationTree:
    """
    迭代树：记录 AutoML 闭环的每次决策和效果。

    每个节点 = 一次迭代（包含改动类型、指标变化、解释、反事实）。
    节点之间形成线性链（Phase 2），未来可扩展为真正的树（Phase 3 多路探索）。

    用法：
        tree = IterationTree(client, task_spec)
        tree.add_node(iteration=1, action="initial", metric_before=0, metric_after=0.61, explanation="...")
        tree.add_node(iteration=2, action="collect_data", metric_before=0.61, metric_after=0.72, explanation="...")
        report = tree.summarize()
    """

    def __init__(self, client: LLMClient, task_spec: TaskSpec):
        self.client    = client
        self.task_spec = task_spec
        self.nodes:    List[IterationNode] = []

    # ── 添加节点 ─────────────────────────────────────────────────────────────

    def add_node(
        self,
        iteration:     int,
        action:        str,
        metric_before: float,
        metric_after:  float,
        explanation:   str,
    ) -> IterationNode:
        """
        添加一个迭代节点。自动计算 delta，异步生成反事实分析。

        Args:
            iteration:     迭代编号（1-based）
            action:        本次改动类型（"initial"/"collect_data"/"adjust_hyperparams"...）
            metric_before: 改动前的指标值
            metric_after:  改动后的指标值
            explanation:   本轮解释器给出的诊断（来自 explainer.py）

        Returns:
            IterationNode
        """
        node_id   = f"node_{iteration:02d}_{uuid.uuid4().hex[:6]}"
        parent_id = self.nodes[-1].node_id if self.nodes else None
        delta     = round(metric_after - metric_before, 4)

        # 反事实分析（LLM 调用）
        counterfactual = self._gen_counterfactual(
            action, metric_before, metric_after, delta, explanation
        )

        node = IterationNode(
            node_id        = node_id,
            parent_id      = parent_id,
            iteration      = iteration,
            action         = action,
            metric_before  = round(metric_before, 4),
            metric_after   = round(metric_after, 4),
            delta          = delta,
            explanation    = explanation,
            counterfactual = counterfactual,
        )
        self.nodes.append(node)
        return node

    # ── 汇总分析 ─────────────────────────────────────────────────────────────

    def summarize(self) -> Dict:
        """
        生成完整的迭代分析报告：
        - 每步改动的贡献量化
        - 最关键的改动
        - 全程关键洞察
        """
        if not self.nodes:
            return {"error": "没有迭代记录"}

        attribution = self._gen_attribution()

        return {
            "task":        self.task_spec.raw_description,
            "metric":      self.task_spec.evaluation_metric,
            "iterations":  len(self.nodes),
            "total_gain":  round(
                self.nodes[-1].metric_after - self.nodes[0].metric_before, 4
            ),
            "nodes": [self._node_to_dict(n) for n in self.nodes],
            "attribution": attribution,
        }

    def best_node(self) -> Optional[IterationNode]:
        """返回带来最大提升的节点"""
        if not self.nodes:
            return None
        return max(self.nodes, key=lambda n: n.delta)

    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON 字符串（供前端可视化）"""
        return json.dumps(self.summarize(), ensure_ascii=False, indent=indent)

    # ── 私有 LLM 调用 ─────────────────────────────────────────────────────────

    def _gen_counterfactual(
        self,
        action:        str,
        metric_before: float,
        metric_after:  float,
        delta:         float,
        explanation:   str,
    ) -> str:
        """生成反事实分析：不做此改动结果会怎样"""
        direction = "提升" if delta > 0.005 else ("下降" if delta < -0.005 else "几乎没有变化")

        prompt = (
            f"任务：{self.task_spec.raw_description}\n"
            f"本次改动：{action}\n"
            f"改动前指标：{metric_before:.4f}\n"
            f"改动后指标：{metric_after:.4f}（{direction} {abs(delta):.4f}）\n"
            f"诊断说明：{explanation}\n\n"
            f"请分析：为什么这次改动带来了这个效果？如果不做此改动，结果会是什么？"
        )

        try:
            resp   = self.client.complete(system=COUNTERFACTUAL_SYSTEM, user=prompt, max_tokens=400)
            raw    = self._extract_json(resp)
            parsed = json.loads(raw)
            return parsed.get("counterfactual", "（分析不可用）")

        except Exception:
            # 回退：基于规则生成简单反事实
            if delta > 0.01:
                return f"如果不做此改动，指标可能停留在 {metric_before:.4f} 附近，改善幅度会大幅减小"
            elif delta > 0:
                return f"此改动带来小幅提升；不做此改动指标变化不大，可能在 {metric_before:.4f} 附近"
            else:
                return f"此改动未带来明显提升；如果不做，结果与现在相近"

    def _gen_attribution(self) -> Dict:
        """用 LLM 归因哪步改动贡献最大"""
        if len(self.nodes) < 2:
            first = self.nodes[0] if self.nodes else None
            return {
                "top_contributor":       first.action if first else "none",
                "top_contribution_delta": first.delta if first else 0,
                "summary":               "只有一次迭代，无法做对比归因",
                "key_insight":           "增加迭代次数可以获得更多分析数据",
            }

        history = json.dumps(
            [self._node_to_dict(n) for n in self.nodes],
            ensure_ascii=False, indent=2
        )

        try:
            resp = self.client.complete(
                system=ATTRIBUTION_SYSTEM,
                user=f"任务：{self.task_spec.raw_description}\n\n迭代历史：\n{history}",
                max_tokens=600,
            )
            raw = self._extract_json(resp)
            return json.loads(raw)

        except Exception:
            best = self.best_node()
            return {
                "top_contributor":        best.action if best else "unknown",
                "top_contribution_delta": best.delta  if best else 0,
                "summary":                f"共 {len(self.nodes)} 轮迭代，累计提升 {self.nodes[-1].metric_after - self.nodes[0].metric_before:.4f}",
                "key_insight":            "数据增强和迭代训练是本次任务最有效的改进方向",
            }

    # ── 辅助方法 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _node_to_dict(node: IterationNode) -> Dict:
        return {
            "node_id":       node.node_id,
            "parent_id":     node.parent_id,
            "iteration":     node.iteration,
            "action":        node.action,
            "metric_before": node.metric_before,
            "metric_after":  node.metric_after,
            "delta":         node.delta,
            "explanation":   node.explanation[:120],
            "counterfactual": node.counterfactual,
        }

    @staticmethod
    def _extract_json(text: str) -> str:
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group() if match else text
