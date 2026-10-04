"""
core/backbone_selector.py  -  模式一：预训练 backbone 微调选型

只负责"选哪个模型、要不要 LoRA"，不负责实际训练——真正的训练/超时/资源上限
都在 core/nn_trainer.py（和 custom_nn 模式共用同一套执行与降级逻辑）。

白名单设计：只允许 LLM 从几个体积可控、CPU/MPS 都能跑得动的模型里选，
不接受 LLM 自由发挥的任意模型 ID（避免拉取不可控的大模型/私有仓库）。
"""

from __future__ import annotations

import json
import re

from core.llm_client import LLMClient
from config import TaskSpec, BackboneSpec

WHITELIST = {
    "distilbert-base-uncased": "英文，轻量（约 66M 参数），适合快速微调",
    "bert-base-chinese": "中文，BERT 基座（约 102M 参数）",
    "hfl/chinese-macbert-base": "中文，MacBERT 改进版（约 102M 参数），中文任务效果通常优于 bert-base-chinese",
    "distilbert-base-multilingual-cased": "多语言，轻量（约 134M 参数）",
}

SYSTEM_PROMPT = f"""你是一个模型选型专家，负责为文本分类任务选择合适的预训练模型。
只能从以下白名单中选择一个（不要编造其他模型 ID）：
{json.dumps(WHITELIST, ensure_ascii=False, indent=2)}

输出 JSON（只输出 JSON，不要其他内容）：
{{
  "model_id": "白名单中的一个 key",
  "use_lora": true 或 false（数据量少/想省显存时用 LoRA，数据量充足时可以选全量微调）,
  "rationale": "一句话说明为什么选这个模型（给非技术人员看，不用技术术语）"
}}
"""


class BackboneSelector:
    def __init__(self, client: LLMClient):
        self.client = client

    def select(self, task_spec: TaskSpec, n_samples: int) -> BackboneSpec:
        prompt = (
            f"任务：{task_spec.raw_description}\n"
            f"领域：{task_spec.domain}\n"
            f"语言：{task_spec.language}\n"
            f"样本数：{n_samples}"
        )
        try:
            raw = self.client.complete(system=SYSTEM_PROMPT, user=prompt, max_tokens=300)
            parsed = self._extract_json(raw)
            model_id = parsed.get("model_id", "")
            if model_id not in WHITELIST:
                return self._fallback(task_spec.language, reason="LLM 选择的模型不在白名单内，已按语言自动匹配")
            return BackboneSpec(
                model_id=model_id,
                use_lora=bool(parsed.get("use_lora", True)),
                rationale=parsed.get("rationale", "根据任务自动选择的预训练模型"),
            )
        except Exception:
            return self._fallback(task_spec.language, reason="模型选型的 LLM 调用失败，已按语言自动匹配兜底模型")

    def _fallback(self, language: str, reason: str) -> BackboneSpec:
        if language == "zh":
            model_id = "hfl/chinese-macbert-base"
        elif language == "en":
            model_id = "distilbert-base-uncased"
        else:
            model_id = "distilbert-base-multilingual-cased"
        return BackboneSpec(model_id=model_id, use_lora=True, rationale=reason)

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise
