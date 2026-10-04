"""
core/dataset_recommender.py  -  用户没有指定数据集时，根据任务需求自动推荐

不新增搜索能力——复用 core/data_sources.py 已有的 DataSource.search()（HF Hub /
魔搭），这里只新增"谁来生成搜索关键词、谁来从结果里挑出最匹配的"这一层：
  1. 用 LLM 根据任务描述生成搜索关键词（中英文各一个——HF 数据集大多用英文
     命名/描述，纯中文关键词命中率低，两个都试，英文优先）
  2. 用现有 DataSource.search() 拿候选列表
  3. 用 LLM 从候选（ref/description/downloads/likes/tags）里选出最匹配的 1-3 个，
     给出一句话理由

任何一步失败（LLM 调用出错、搜索无结果）都返回空列表，不抛异常——调用方
（core/conversation/stages.py::prepare_data）拿到空列表就回退到现有的手动
data_picker 流程，不会因为推荐失败而打断对话。
"""

from __future__ import annotations

from typing import List

from core.llm_client import LLMClient
from core.json_extract import extract_json
from core.data_sources import get_source
from config import TaskSpec, DatasetSummary, RecommendedDataset

_QUERY_SYSTEM_PROMPT = """你是一个数据集检索专家。根据任务描述，生成用于在 HuggingFace Hub
搜索公开数据集的关键词。HuggingFace Hub 的搜索是简单的文本匹配，不是语义搜索——关键词
越短、越通用越容易命中（比如 "sentiment classification" 比
"e-commerce customer service ticket sentiment classification" 更容易搜到结果），
不要把任务描述里的所有细节都塞进关键词。
HuggingFace 上的数据集大多用英文命名和描述，所以英文关键词通常更容易搜到结果；
但任务如果明确是中文场景，也生成一个中文关键词作为补充。
关键词要简短（1-3 个词，不要整句话）。

输出 JSON（只输出 JSON）：
{"query_en": "英文关键词（1-3个词）", "query_zh": "中文关键词（1-3个词，任务不涉及中文可以留空）"}
"""

_RANK_SYSTEM_PROMPT = """你是一个数据集推荐专家。下面是任务描述和一批候选数据集
（来自 HuggingFace Hub 搜索结果，含数据集 ID、描述、下载量、点赞数、标签）。
请选出最匹配这个任务的 1-3 个候选，按匹配程度从高到低排列，每个给一句话理由
（给非技术人员看，说清楚"为什么适合这个任务"，不用技术术语）。
只能从给出的候选 ref 里选，不要编造不存在的数据集 ID。

输出 JSON（只输出 JSON）：
{"picks": [{"ref": "候选里的某个 ref", "rationale": "一句话理由"}, ...]}
"""


class DatasetRecommender:
    def __init__(self, client: LLMClient):
        self.client = client

    def recommend(self, task_spec: TaskSpec, platform: str = "huggingface",
                  max_candidates: int = 10) -> List[RecommendedDataset]:
        try:
            source = get_source(platform)
        except Exception:
            return []

        candidates = self._search_candidates(task_spec, source, max_candidates)
        if not candidates:
            return []
        return self._rank(task_spec, candidates, platform)

    # ── 私有方法 ─────────────────────────────────────────────────────────────

    def _search_candidates(self, task_spec: TaskSpec, source, max_candidates: int) -> List[DatasetSummary]:
        queries = self._generate_queries(task_spec)
        if not queries:
            return []

        # HF Hub 的搜索是简单文本匹配，过长/过具体的关键词经常 0 命中（实测验证过）——
        # 每个 LLM 生成的关键词都再拆出一个"只取第一个词"的更宽泛兜底版本一起搜，
        # 不额外调 LLM，命中率更高
        broadened = [q.split()[0] for q in queries if q.strip() and " " in q]
        all_queries = queries + broadened

        seen_refs = set()
        candidates: List[DatasetSummary] = []
        for q in all_queries:
            if not q or len(candidates) >= max_candidates:
                continue
            try:
                results = source.search(q, page=1, page_size=max_candidates)
            except Exception:
                continue
            for r in results:
                if r.ref not in seen_refs:
                    seen_refs.add(r.ref)
                    candidates.append(r)

        if not candidates and task_spec.domain:
            # 所有关键词都 0 命中——最后退一步直接用领域词搜，聊胜于无
            try:
                for r in source.search(task_spec.domain, page=1, page_size=max_candidates):
                    if r.ref not in seen_refs:
                        seen_refs.add(r.ref)
                        candidates.append(r)
            except Exception:
                pass

        return candidates[:max_candidates]

    def _generate_queries(self, task_spec: TaskSpec) -> List[str]:
        prompt = (
            f"任务：{task_spec.raw_description}\n"
            f"领域：{task_spec.domain}\n"
            f"语言：{task_spec.language}\n"
            f"标签：{', '.join(task_spec.label_schema) if task_spec.label_schema else '未指定'}"
        )
        try:
            raw = self.client.complete(system=_QUERY_SYSTEM_PROMPT, user=prompt, max_tokens=150)
            parsed = extract_json(raw)
            queries = [parsed.get("query_en", ""), parsed.get("query_zh", "")]
            return [q.strip() for q in queries if q and q.strip()]
        except Exception:
            # 兜底：LLM 调用失败时直接用领域词当查询词，还能凑合搜一下，不至于完全放弃
            return [task_spec.domain] if task_spec.domain else []

    def _rank(self, task_spec: TaskSpec, candidates: List[DatasetSummary],
              platform: str) -> List[RecommendedDataset]:
        candidates_desc = "\n".join(
            f"- ref={c.ref} | 描述={c.description[:100]} | 下载={c.downloads} | "
            f"点赞={c.likes} | 标签={','.join(c.tags[:3])}"
            for c in candidates
        )
        prompt = f"任务：{task_spec.raw_description}\n\n候选数据集：\n{candidates_desc}"
        valid_refs = {c.ref for c in candidates}

        try:
            raw = self.client.complete(system=_RANK_SYSTEM_PROMPT, user=prompt, max_tokens=400)
            parsed = extract_json(raw)
            picks = parsed.get("picks", []) if isinstance(parsed, dict) else parsed
            results = [
                RecommendedDataset(platform=platform, ref=p["ref"], rationale=p.get("rationale", ""))
                for p in picks if isinstance(p, dict) and p.get("ref") in valid_refs
            ]
            if results:
                return results[:3]
        except Exception:
            pass

        # 兜底：LLM 排序失败时按下载量选最热门的一个，仍然给用户一个候选而不是直接放弃推荐
        top = max(candidates, key=lambda c: c.downloads)
        return [RecommendedDataset(platform=platform, ref=top.ref,
                                   rationale="下载量最高的候选（自动排序暂时不可用，按热度兜底选择）")]
