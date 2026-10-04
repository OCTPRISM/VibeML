"""
core/agent/skills.py  -  固定内置 Skill 库。

不对接 Anthropic 官方 Skills API——那是 Anthropic 专有能力，跟本产品"多提供方
平权"（Anthropic/Ollama/自建/商业 OpenAI 兼容/系统托管全部平等支持）的定位冲突。
这里的"Skill"就是一段追加进 Agent 系统提示的说明文字，用户在前端勾选后通过
state.enabled_skills（一份 skill id 列表）传进来，AgentOrchestrator 按需拼接
进系统提示——不是运行时代码注入，没有任何执行面，只是让 LLM"读到更具体的指导"。
"""

from __future__ import annotations

from typing import Dict, List

SKILL_LIBRARY: Dict[str, str] = {
    "dataset_search_tips": """
【数据集搜索技巧】
- 中文任务优先中英文关键词都搜一遍——很多高质量数据集只有英文名/描述，纯中文
  关键词经常搜不到，但内容其实是通用/多语言的。
- 搜到多个结果时优先看下载量/点赞数高的，通常意味着更多人验证过质量。
- 数据集列名不一定叫 text/label，preview_dataset 返回的建议列名仅供参考，拿到
  预览后要自己确认列内容是否真的对应文本和标签，不要盲目相信自动建议。
- 候选数据集的类别体系和任务要求不完全一致时（比如标签数量、粒度不同），要
  明确指出差异并让用户确认这个差异是否可以接受，不要自己悄悄决定。
""".strip(),
    "arch_design_guidelines": """
【架构设计规范】
- 训练样本少于 50 条/类时，优先用 pretrained_nn（预训练模型微调）而不是
  custom_nn（从零设计的自定义结构）——样本太少时自定义结构容易过拟合，预训练
  权重自带的先验知识更可靠。
- 只有在任务明显偏离常规文本分类（比如需要特殊的输入处理逻辑）、且样本量足够
  支撑从零训练时，才考虑 custom_nn。
- 不确定选哪个后端时，优先选 pretrained_nn 作为稳妥的默认项，而不是为了"看起
  来更有技术含量"选 custom_nn。
""".strip(),
}


def build_skills_appendix(enabled_skills: List[str]) -> str:
    """把已启用的 skill 文本拼成一段可以直接追加到系统提示末尾的字符串；
    没有匹配到任何已知 skill id（或列表为空）时返回空字符串，调用方不需要
    额外判断。"""
    parts = [SKILL_LIBRARY[sid] for sid in enabled_skills if sid in SKILL_LIBRARY]
    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)
