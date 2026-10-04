"""
core/data_engine.py  -  Phase 1.2：小数据引导器

差异化核心（文献验证的真实空白 #3）：
  现有 AutoML 工具对"训练上游"支持极弱——任务定义、数据构建、
  边界样本识别这些决策常常主导最终效果，但没人帮用户做。

本模块做三件事：
  1. 分析数据质量，发现类别不平衡、标签缺失等问题
  2. 识别边界样本（标注争议案例，帮用户重新审查）
  3. 使用 LLM 按标签均衡增强数据（10 条→200 条）
"""

import json
import re
from typing import List, Dict, Tuple
from collections import Counter

from core.llm_client import LLMClient
from config import TaskSpec, DataReport


# ── Prompts ──────────────────────────────────────────────────────────────────

AUGMENT_SYSTEM = """你是一个数据增强专家，负责生成高质量的训练样本。

要求：
1. 生成的样本必须真实、多样，覆盖该类别的不同表达方式
2. 保持领域语言风格（如客服工单的口语化、法律文本的正式化）
3. 不能简单复制示例，要有真实的变体
4. 长度分布要自然（不要全部很短或很长）

输出格式：JSON 数组，每个元素为 {"text": "...", "label": "..."}
只输出 JSON 数组，不要其他内容。"""

BOUNDARY_SYSTEM = """你是一个数据质量专家，负责识别训练数据中的"边界样本"。

边界样本是指：
- 描述模糊、可能属于多个类别的样本
- 标注存在争议的样本
- 极端或罕见表达的样本（可能导致模型偏差）

对每个样本输出：
{
  "index": 0,
  "boundary_score": 0.0到1.0（越高越模糊，0.5以上才输出）,
  "reason": "为什么这是边界样本（一句话）",
  "alternative_label": "可能的其他标签"
}

只输出 boundary_score > 0.5 的样本，格式为 JSON 数组。
如果没有边界样本，输出 []。
只输出 JSON，不要其他内容。"""


# ── DataEngine ───────────────────────────────────────────────────────────────

class DataEngine:
    """
    小数据引导器：分析、识别边界、增强。

    用法：
        engine = DataEngine(client)
        data, report = engine.bootstrap(examples, task_spec, target_total=200)
    """

    def __init__(self, client: LLMClient):
        self.client = client

    # ── 主入口 ──────────────────────────────────────────────────────────────

    def bootstrap(
        self,
        examples: List[Dict],
        task_spec: TaskSpec,
        target_total: int = 200,
        verbose: bool = True,
    ) -> Tuple[List[Dict], DataReport]:
        """
        完整数据引导流程。

        Args:
            examples:     原始标注样本 [{"text": ..., "label": ...}]
            task_spec:    任务规格
            target_total: 目标数据量（增强后）
            verbose:      是否打印进度

        Returns:
            (augmented_data, DataReport)
        """
        if verbose:
            print(f"  📊 分析 {len(examples)} 条原始样本...")

        # Step 1: 质量检查
        label_dist = Counter(e["label"] for e in examples)
        warnings = self._check_quality(examples, task_spec, label_dist)

        if verbose and warnings:
            for w in warnings:
                print(f"  ⚠️  {w}")

        # Step 2: 识别边界样本（最多分析前 30 条，避免 token 超限）
        if verbose:
            print("  🔍 识别边界样本...")
        boundary_samples = self._identify_boundary(examples[:30])

        if verbose and boundary_samples:
            print(f"  🔴 发现 {len(boundary_samples)} 个边界样本（可能需要重新审查）")

        # Step 3: 数据增强
        augmented = list(examples)
        augmented_count = 0

        if len(examples) < target_total:
            if verbose:
                print(f"  🔧 数据增强：{len(examples)} → {target_total} 条（按标签均衡）")
            augmented, augmented_count = self._augment(examples, task_spec, target_total, verbose)
        else:
            if verbose:
                print(f"  ✅ 样本量充足（{len(examples)} 条），跳过增强")

        # Step 4: 计算综合质量分
        quality_score = self._compute_quality(augmented, task_spec, warnings)

        report = DataReport(
            total_samples=len(augmented),
            label_distribution=dict(Counter(e["label"] for e in augmented)),
            augmented_count=augmented_count,
            boundary_samples=boundary_samples,
            quality_score=quality_score,
            warnings=warnings,
        )

        return augmented, report

    # ── 公开方法（供 loop.py 的中途增强调用）────────────────────────────────

    def augment(
        self,
        examples: List[Dict],
        task_spec: TaskSpec,
        additional: int = 50,
    ) -> Tuple[List[Dict], int]:
        """在已有数据基础上再增强 additional 条"""
        current_total = len(examples)
        return self._augment(examples, task_spec, current_total + additional, verbose=True)

    # ── 私有方法 ─────────────────────────────────────────────────────────────

    def _check_quality(
        self,
        examples: List[Dict],
        task_spec: TaskSpec,
        label_dist: Counter,
    ) -> List[str]:
        warnings = []

        # 检查标签覆盖
        missing = set(task_spec.label_schema) - set(label_dist.keys())
        if missing:
            warnings.append(f"以下标签在数据中无样本：{', '.join(missing)}")

        # 检查类别不平衡
        if len(label_dist) > 1:
            counts = list(label_dist.values())
            ratio = max(counts) / min(counts)
            if ratio > 5:
                dominant = label_dist.most_common(1)[0]
                least = label_dist.most_common()[-1]
                warnings.append(
                    f"类别严重不平衡（最多 '{dominant[0]}' {dominant[1]} 条，"
                    f"最少 '{least[0]}' {least[1]} 条，比例 {ratio:.1f}x）"
                )

        # 检查样本量
        if len(examples) < 10:
            warnings.append(f"样本量极少（{len(examples)} 条），训练效果可能受限，建议至少 20 条")

        # 检查 text 字段是否存在
        missing_text = sum(1 for e in examples if "text" not in e or not e["text"].strip())
        if missing_text > 0:
            warnings.append(f"{missing_text} 条样本缺少 text 字段或内容为空")

        return warnings

    def _identify_boundary(self, examples: List[Dict]) -> List[Dict]:
        if len(examples) < 3:
            return []

        examples_str = json.dumps(
            [{"index": i, **e} for i, e in enumerate(examples)],
            ensure_ascii=False, indent=2
        )

        try:
            raw = self.client.complete(
                system=BOUNDARY_SYSTEM,
                user=f"请分析以下训练样本，识别边界样本：\n\n{examples_str}",
                max_tokens=2000,
            )
            boundaries = self._coerce_list(self._parse_json(raw))

            # 附加原始样本内容，方便用户查看
            for b in boundaries:
                idx = b.get("index", -1)
                if 0 <= idx < len(examples):
                    b["original"] = examples[idx]

            return [b for b in boundaries if b.get("boundary_score", 0) > 0.5]

        except Exception as e:
            # 边界识别失败不影响主流程
            return []

    def _augment(
        self,
        examples: List[Dict],
        task_spec: TaskSpec,
        target: int,
        verbose: bool = True,
    ) -> Tuple[List[Dict], int]:
        """
        按标签均衡增强：每个标签独立增强到 target/num_labels 条。
        这样可以同时修正类别不平衡问题。
        """
        label_dist = Counter(e["label"] for e in examples)
        labels = task_spec.label_schema or list(label_dist.keys())
        per_label_target = max(5, target // max(len(labels), 1))

        augmented = list(examples)
        total_added = 0

        for label in labels:
            current_count = label_dist.get(label, 0)
            need = max(0, per_label_target - current_count)

            if need <= 0:
                continue

            # 取该标签的种子示例（最多 5 条）
            seeds = [e for e in examples if e.get("label") == label][:5]
            if not seeds:
                seeds = examples[:3]  # fallback：借用其他标签的样本作格式参考

            seeds_str = json.dumps(seeds, ensure_ascii=False, indent=2)

            try:
                raw = self.client.complete(
                    system=AUGMENT_SYSTEM,
                    user=(
                        f"任务描述：{task_spec.raw_description}\n"
                        f"领域：{task_spec.domain}\n"
                        f"当前目标标签：{label}\n\n"
                        f"以下是该标签的参考示例（请生成风格类似但内容不同的新样本）：\n{seeds_str}\n\n"
                        f"请生成 {need} 条新样本（所有样本的 label 必须为 \"{label}\"）："
                    ),
                    max_tokens=3000,
                )
                new_samples = self._coerce_list(self._parse_json(raw))

                # 确保 label 字段正确
                for s in new_samples:
                    s["label"] = label
                    s.setdefault("text", "")

                new_samples = [s for s in new_samples if s["text"].strip()]
                augmented.extend(new_samples)
                total_added += len(new_samples)

                if verbose:
                    print(f"    标签 '{label}'：+{len(new_samples)} 条")

            except Exception as e:
                if verbose:
                    print(f"    ⚠️  标签 '{label}' 增强失败：{e}")

        return augmented, total_added

    def _compute_quality(
        self,
        data: List[Dict],
        task_spec: TaskSpec,
        warnings: List[str],
    ) -> float:
        score = 1.0

        # 每个 warning 扣分
        score -= len(warnings) * 0.08

        # 类别平衡度加权
        if data:
            dist = Counter(e["label"] for e in data)
            counts = list(dist.values())
            if len(counts) > 1:
                balance = min(counts) / max(counts)
                score = score * 0.6 + balance * 0.4

        # 样本量奖励
        n = len(data)
        if n >= 200:
            score = min(1.0, score + 0.1)
        elif n < 50:
            score -= 0.1

        return round(max(0.0, min(1.0, score)), 3)

    @staticmethod
    def _parse_json(text: str):
        """稳健地解析 LLM 输出的 JSON"""
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 找第一个 [...] 或 {...}
            match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return []

    @staticmethod
    def _coerce_list(parsed) -> list:
        """
        期望得到 JSON 数组，但强制 JSON 模式的本地模型（如 Ollama qwen）不保证
        输出形状：常见的偏差有两种——
          1. 把数组包一层对象再输出，如 {"data": [...]}
          2. 干脆只生成一条，直接输出单个对象而不是数组，如 {"text": ..., "label": ...}
        这里做兜底：本来就是数组就直接用；是对象则优先取第一个数组类型的 value；
        如果对象本身没有任何数组字段，则视为"退化成单条"，包一层 [obj] 返回。
        """
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
            if parsed:
                return [parsed]
        return []
