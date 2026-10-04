"""
core/arch_designer.py  -  模式二：LLM 生成自定义分类头（架构设计）

契约（写死在 prompt 里，core/nn_trainer.py 按这个契约实例化/调用生成的类）：
  - 生成的类可以叫任意名字，但必须继承 nn.Module
  - __init__(self, input_dim: int, num_classes: int)
  - forward(self, x) -> 形状 (batch, num_classes) 的 logits（不做 softmax）
  - 输入 x 是已经做好 TF-IDF 向量化的稠密浮点张量——这是让"LLM 现场设计网络结构"
    从"不现实的幻想"变成"确实能跑、确实安全"的关键简化：LLM 只负责设计分类头，
    不负责从原始文本自己设计 tokenizer/embedding 层（那是模式一 backbone_selector 干的事）
  - 只允许 import torch / torch.nn / torch.nn.functional

生成的代码在真正执行前必须经过 core/nn_sandbox.py 的静态校验，本模块自己不执行任何代码。
"""

from __future__ import annotations

import json
import re

from core.llm_client import LLMClient
from config import TaskSpec, GeneratedArchSpec

SYSTEM_PROMPT = """你是一个 PyTorch 模型架构师，负责为一个已经完成 TF-IDF 向量化的文本分类任务设计一个分类头网络。

严格约束（违反任何一条都会被自动拒绝，不会被执行）：
1. 只能 `import torch`、`import torch.nn as nn`、`import torch.nn.functional as F`，不允许任何其他 import
2. 必须恰好定义 1 个类，继承 nn.Module
3. 构造函数签名必须是 `__init__(self, input_dim, num_classes)`（不能有其他必填参数）
4. 必须定义 `forward(self, x)`：x 是形状 (batch, input_dim) 的稠密浮点张量（已经是 TF-IDF 特征，
   不是原始文本，不需要 tokenizer/embedding），forward 必须返回形状 (batch, num_classes) 的 logits
   （不要在 forward 里做 softmax，损失函数会处理）
5. 不允许使用装饰器；不允许出现 os/sys/subprocess/eval/exec/open 等危险内容
6. 网络规模要克制：这是跑在本地 CPU/Apple GPU 上的小分类头，不是大模型，总参数量不应超过几百万

输出 JSON（只输出 JSON，不要其他内容；code 字段里的换行请用 \\n 转义）：
{
  "code": "完整的 Python 源码字符串",
  "class_name": "类名，必须和 code 里定义的类名一致",
  "loss_fn_name": "cross_entropy" 或 "nll_loss",
  "rationale": "一句话说明这个架构设计的考虑（给非技术人员看，不用术语，比如为什么用这么多层/为什么加 dropout）"
}
"""


class ArchDesigner:
    def __init__(self, client: LLMClient):
        self.client = client

    def design(self, task_spec: TaskSpec, n_samples: int, repair_hint: str = "") -> GeneratedArchSpec:
        """
        n_samples: 训练样本数量参考值（不要求是最终增强后的精确值——这一步设计
        本来就只依赖任务描述本身，允许和数据准备并行跑，n_samples 只是给 LLM
        一个"数据量级"的提示，不影响架构设计的正确性）。
        """
        prompt = (
            f"任务：{task_spec.raw_description}\n"
            f"领域：{task_spec.domain}\n"
            f"类别数：{len(task_spec.label_schema)}（{', '.join(task_spec.label_schema)}）\n"
            f"训练样本数量级：约 {n_samples} 条\n"
        )
        if repair_hint:
            prompt += f"\n上一次生成的代码执行失败，错误信息：\n{repair_hint}\n请修复这个问题，重新生成。"

        raw = self.client.complete(system=SYSTEM_PROMPT, user=prompt, max_tokens=1500)
        parsed = self._extract_json(raw)
        return GeneratedArchSpec(
            class_name=parsed.get("class_name", ""),
            source_code=parsed.get("code", ""),
            loss_fn_name=parsed.get("loss_fn_name", "cross_entropy"),
            rationale=parsed.get("rationale", ""),
        )

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
