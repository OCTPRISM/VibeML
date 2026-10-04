"""
core/conversation/stages.py  -  每个对话阶段的处理逻辑

每个 handler 签名一致：
    handler(state, user_text, structured, client) -> StageResult
（prepare_data 是 async def——它内部可能触发真实的 HF/魔搭网络请求做数据集
 推荐预览，用 asyncio.to_thread 避免阻塞事件循环；调用方
 core/conversation/orchestrator.py 用 inspect.isawaitable 统一处理，
 其它几个 handler 保持同步不受影响）

- user_text:   用户发的自由文本（结构化消息时通常为空字符串）
- structured:  前端发来的结构化数据（如 {"type": "data_selected", "dataset_ref": "..."}），
              自由文本消息时为 None
- 返回 StageResult：
    satisfied=True  → orchestrator 把 updates 写回 state，推进到下一阶段
    satisfied=False → question 作为一条新的助手消息返回给用户；ui_hint 告诉前端
                      要不要在这条问题下面渲染对应的结构化操作卡片（数据来源 tab /
                      训练后端单选 / 训练参数输入），这些卡片本身完全复用现有 UI，
                      不是这里生成的
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.llm_client import LLMClient
from core.task_parser import TaskParser
from config import TaskType

from api.settings import settings

from . import memory

# 轻量单机版只支持机器学习/深度学习任务（分类/多标签/实体识别/生成/回归，
# 对应 ModelBackend.SKLEARN/PRETRAINED_NN/CUSTOM_NN）——强化学习、LLM 微调、
# VLM 图像分类/生成式 本地资源开销（stable-baselines3 训练、整个基座模型/
# 视觉编码器常驻显存）超出桌面构建的预算，只在完整版（联网版）开放。
# 这不是"问不清楚就将就"的追问逻辑，是能力边界
DESKTOP_UNSUPPORTED_TASK_TYPES = (
    TaskType.RL, TaskType.LLM_FINETUNE, TaskType.IMAGE_CLASSIFICATION, TaskType.VLM_GENERATIVE,
)
from .state import ConversationState, ConversationStage, MAX_CLARIFICATION_ROUNDS


@dataclass
class StageResult:
    satisfied: bool
    question:  Optional[str] = None
    updates:   Dict[str, Any] = field(default_factory=dict)
    ui_hint:   Optional[str] = None


def _rounds_exceeded(state: ConversationState, stage: ConversationStage) -> bool:
    return state.clarification_rounds.get(stage.value, 0) >= MAX_CLARIFICATION_ROUNDS


def _bump_rounds(state: ConversationState, stage: ConversationStage) -> None:
    state.clarification_rounds[stage.value] = state.clarification_rounds.get(stage.value, 0) + 1


def _looks_like_default_choice(text: str) -> bool:
    keywords = ("你决定", "随便", "都行", "默认", "你来选", "无所谓", "都可以", "随意")
    return any(k in text for k in keywords)


def _parse_instruction_lines(text: str) -> list:
    """把用户粘贴的多行文本解析成 instruction/input/output 三元组列表——
    每行「指令 | 输入 | 输出」（没有输入写「指令 | | 输出」或「指令 | 输出」两段式）。
    格式不对/缺指令或输出的行直接跳过，不报错中断，最后看凑够了几条有效的。"""
    examples = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 2:
            instruction, input_text, output = parts[0], "", parts[1]
        elif len(parts) >= 3:
            instruction, input_text, output = parts[0], parts[1], parts[2]
        else:
            continue
        if instruction and output:
            examples.append({"instruction": instruction, "input": input_text, "output": output})
    return examples


_MIN_INSTRUCTION_EXAMPLES = 3
_DEFAULT_LLM_FT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# 已知参数量小、在普通机器上跑得动的模型，用户说"你决定"或反复给不出有效
# 模型 ID 时的兜底选择——不是白名单（用户已确认不设白名单），只是默认建议

_MIN_IMAGE_EXAMPLES = 6    # 至少覆盖 2 个类别、每类至少几张，太少没法有意义地切训练/验证集
_MIN_VLM_EXAMPLES   = 4    # 生成式 VLM 训练本来就慢，先用一个较小的下限，够跑通一轮迭代就行
_DEFAULT_VLM_CLS_ENCODER = "openai/clip-vit-base-patch32"
_DEFAULT_VLM_GEN_MODEL   = "yuanzhoulvpi/vit-gpt2-image-chinese-captioning"
# 之前默认用 Salesforce/blip-image-captioning-base，但真实训练验证过：它的分词器是纯
# 英文 BERT WordPiece 词表，中文参考答案里的汉字绝大部分被切成 [UNK]，ROUGE-L 全程卡在
# 0（同一套训练代码换成英文参考答案，ROUGE-L 第一轮就到 1.0，证明不是训练代码的 bug）。
# 这个系统的任务描述/UI 都是中文优先，静默用一个学不会中文的默认模型等于这个功能对
# 主要使用场景直接是坏的。换成这个 ViT 编码器 + GPT2 中文解码器的模型（真实验证过：
# 中文 BERT 词表对中英文都能正确切分/还原，未微调的底座就已经能生成通顺中文）——
# 本来想换更大的多语言模型（如 Qwen2-VL），但它的 Processor 在类型检查层面硬性
# 要求一个真正的 torchvision-backed 视频处理器（哪怕根本不处理视频），这台机器装不上
# 匹配现有 torch 版本的 torchvision，只能放弃，改用这个跟 BLIP 同量级（~242M）但能
# 处理中文的模型。


def _try_validate_llm_ft_model(model_id: str):
    """返回 (ValidatedModel, None) 或 (None, 错误原因字符串)——不抛异常，
    调用方（choose_model）自己决定拿到错误原因之后怎么问。"""
    from core.llm_ft_selector import validate_model_id, ModelRejectedError
    try:
        return validate_model_id(model_id), None
    except ModelRejectedError as e:
        return None, str(e)


async def _try_recommend_dataset(state: ConversationState, client: LLMClient) -> Optional[Dict[str, Any]]:
    """调 core/dataset_recommender.py 找候选 + 预览，任何一步失败都返回 None
    （调用方据此静默回退到手动 data_picker，不让推荐失败打断对话）。

    DatasetRecommender.recommend() 内部会调一次 LLM（同步阻塞调用），随后最多
    再对 3 个候选各发一次真实的 HF/魔搭网络请求做预览——这是一整串可能耗时
    数十秒的阻塞调用链，真实测过如果直接同步跑会整个卡住 uvicorn 的事件循环
    （对话线程之外，其它任何请求，包括完全无关的 /api/tasks/queue/stats 健康
    检查，都会跟着一起没有响应）。用 asyncio.to_thread 把这一整段丢到线程池
    执行，不阻塞事件循环——DatasetRecommender/get_source 本身保持同步实现，
    不需要为了这一个调用点把它们全部改写成 async。"""
    try:
        from core.dataset_recommender import DatasetRecommender
        from core.data_sources import get_source

        picks = await asyncio.to_thread(DatasetRecommender(client).recommend, state.task_spec)
        if not picks:
            return None

        candidates = []
        for pick in picks[:3]:
            try:
                preview = await asyncio.to_thread(get_source(pick.platform).preview, pick.ref)
            except Exception:
                continue
            candidates.append({
                "platform": pick.platform, "ref": pick.ref, "rationale": pick.rationale,
                "columns": preview.columns, "sample_rows": preview.sample_rows,
                "suggested_text_col": preview.suggested_text_col,
                "suggested_label_col": preview.suggested_label_col,
            })
        return {"candidates": candidates} if candidates else None
    except Exception:
        return None


# ── Stage 1: 任务澄清（复用 core/task_parser.py::TaskParser.parse_once）────────

def clarify_task(state: ConversationState, user_text: str, structured: Optional[dict],
                 client: LLMClient) -> StageResult:
    if not user_text.strip():
        return StageResult(satisfied=False,
                           question="先说说你想做什么任务吧？比如「帮我把客服工单按问题类型分类」。")

    parser = TaskParser(client)
    raw = state.raw_description or user_text
    combined = user_text if not state.raw_description else f"原始描述：{state.raw_description}\n补充信息：{user_text}"
    if state.memory_summary:
        # 用户在修正/补充一个已经跑过至少一轮的任务——把记忆摘要也喂给解析，
        # 避免"记得"的信息在重新解析这一步被无声丢掉
        combined = f"{combined}\n\n（此前的对话记忆：{state.memory_summary}）"
    spec, needs_clarification, question = parser.parse_once(combined, raw_description=raw)

    if not state.raw_description:
        state.raw_description = user_text

    if needs_clarification and not _rounds_exceeded(state, ConversationStage.CLARIFYING_TASK):
        _bump_rounds(state, ConversationStage.CLARIFYING_TASK)
        return StageResult(satisfied=False, question=question)

    # 桌面构建里强化学习/LLM 微调是能力边界，不是"追问几轮就将就通过"的东西——
    # 不受 _rounds_exceeded 的强制放行逻辑影响，用户不换任务描述就一直停在这里
    if settings.is_desktop_build and spec.task_type in DESKTOP_UNSUPPORTED_TASK_TYPES:
        return StageResult(satisfied=False, question=(
            "轻量单机版目前只支持机器学习和深度学习任务（比如文本分类、实体识别、"
            "文本生成、回归），强化学习和 LLM 微调需要用完整版（联网版）才能训练。"
            "要不换一个分类/生成类的任务描述试试？"
        ))

    return StageResult(satisfied=True, updates={"task_spec": spec})


# ── Stage 2: 数据准备（结构化卡片复用现有数据来源 UI，这里只负责收结果）──────────

async def prepare_data(state: ConversationState, user_text: str, structured: Optional[dict],
                       client: LLMClient) -> StageResult:
    if state.task_spec and state.task_spec.task_type == TaskType.RL:
        # RL 任务没有"上传训练数据"这一步——环境本身是根据任务描述生成的代码，
        # 这里只是多收集一点环境细节（规则/奖励设计），完全跳过数据来源选择器
        if user_text.strip() and _looks_like_default_choice(user_text):
            return StageResult(satisfied=True, updates={"env_description": state.raw_description})
        if user_text.strip() and len(user_text.strip()) >= 5:
            return StageResult(satisfied=True, updates={"env_description": user_text.strip()})
        if _rounds_exceeded(state, ConversationStage.PREPARING_DATA):
            return StageResult(satisfied=True, updates={"env_description": state.raw_description})
        _bump_rounds(state, ConversationStage.PREPARING_DATA)
        return StageResult(
            satisfied=False,
            question="能再详细描述一下这个强化学习环境吗？比如智能体能做哪些动作、"
                     "能观察到什么信息、什么情况算成功/失败、怎么给奖励——"
                     "描述得越具体，生成的环境就越准确。不确定的话直接说「你决定」，"
                     "我按最初的任务描述来设计。",
        )

    if state.task_spec and state.task_spec.task_type == TaskType.LLM_FINETUNE:
        # LLM 微调任务收集 instruction/input/output 三元组，不是 text/label——
        # 完全跳过现有的数据来源选择器（那是给分类任务的 dataset_ref/手动样本用的）
        if structured and structured.get("type") == "data_selected" and structured.get("instruction_examples"):
            return StageResult(satisfied=True, updates={"instruction_examples": structured["instruction_examples"]})
        if user_text.strip() and not _looks_like_default_choice(user_text):
            parsed = _parse_instruction_lines(user_text)
            if len(parsed) >= _MIN_INSTRUCTION_EXAMPLES:
                return StageResult(satisfied=True, updates={"instruction_examples": parsed})
        # 没有合理默认值可以强行凑数（不像 RL 可以退回原始任务描述当环境说明）——
        # 微调用的例子必须是真实数据，轮次超限也只能继续问，不编造假样本
        _bump_rounds(state, ConversationStage.PREPARING_DATA)
        return StageResult(
            satisfied=False,
            question=f"接下来需要微调用的例子。每行一条，用「指令 | 输入 | 输出」的格式"
                     f"（没有输入的话写「指令 | 输出」两段式也行），至少需要 {_MIN_INSTRUCTION_EXAMPLES} 条，比如：\n"
                     f"把下面这句话翻译成英文 | 你好，世界 | Hello, world\n"
                     f"把下面这句话翻译成英文 | 我喜欢猫 | I like cats",
        )

    if state.task_spec and state.task_spec.task_type == TaskType.IMAGE_CLASSIFICATION:
        # 图像分类任务收集 {image_path, label} 而不是 {text, label}——完全跳过
        # 现有的文本数据来源选择器，前端走新的图片上传卡片
        # （POST /api/datasets/upload-images 落盘后拿到 image_path 引用，
        # 图片本身不会内联进对话消息流，跟文本样本"引用而非内联大二进制"是同一个思路）
        if structured and structured.get("type") == "data_selected" and structured.get("image_examples"):
            examples = structured["image_examples"]
            labels = set(e.get("label") for e in examples if e.get("label"))
            if len(examples) >= _MIN_IMAGE_EXAMPLES and len(labels) >= 2:
                return StageResult(satisfied=True, updates={"image_examples": examples})
            return StageResult(
                satisfied=False,
                question=f"图片样本不够——至少需要 {_MIN_IMAGE_EXAMPLES} 张、覆盖至少 2 个不同标签，"
                         f"当前是 {len(examples)} 张、{len(labels)} 个标签，请再补充一些。",
                ui_hint="image_picker",
            )
        _bump_rounds(state, ConversationStage.PREPARING_DATA)
        return StageResult(
            satisfied=False,
            question=f"接下来需要训练图片。上传图片并给每张标注类别标签，"
                     f"至少 {_MIN_IMAGE_EXAMPLES} 张、覆盖至少 2 个类别。",
            ui_hint="image_picker",
        )

    if state.task_spec and state.task_spec.task_type == TaskType.VLM_GENERATIVE:
        # 生成式 VLM 收集 {image_path, prompt, reference_answer} 三元组——
        # 每张图配一个问题/指令 + 一个参考答案（有参考答案才能用 ROUGE-L 打分，
        # 完全开放式、无参考答案的生成评估不在这次范围内，见计划里的范围收紧说明）
        if structured and structured.get("type") == "data_selected" and structured.get("vlm_examples"):
            examples = structured["vlm_examples"]
            valid = [e for e in examples if e.get("image_path") and e.get("reference_answer")]
            if len(valid) >= _MIN_VLM_EXAMPLES:
                return StageResult(satisfied=True, updates={"vlm_examples": valid})
            return StageResult(
                satisfied=False,
                question=f"有效样本不够——每条都需要图片 + 参考答案，至少需要 {_MIN_VLM_EXAMPLES} 条，"
                         f"当前只有 {len(valid)} 条有效，请再补充一些。",
                ui_hint="vlm_picker",
            )
        _bump_rounds(state, ConversationStage.PREPARING_DATA)
        return StageResult(
            satisfied=False,
            question=f"接下来需要训练样本。每条样本包含一张图片、一个问题/指令、和一个参考答案"
                     f"（比如看图说话的参考描述，或者视觉问答的标准答案），至少需要 {_MIN_VLM_EXAMPLES} 条。",
            ui_hint="vlm_picker",
        )

    if structured and structured.get("type") == "data_selected":
        if structured.get("dataset_ref"):
            return StageResult(satisfied=True, updates={"dataset_ref": structured["dataset_ref"]})
        if structured.get("examples"):
            return StageResult(satisfied=True, updates={"examples": structured["examples"]})
        return StageResult(satisfied=False,
                           question="没有收到有效的数据集，请重新选择数据来源。",
                           ui_hint="data_picker")

    if user_text.strip() and _looks_like_default_choice(user_text) and state.task_spec:
        # 用户没有自己的数据——尝试根据任务描述自动推荐一个公开数据集，而不是
        # 只会反复追问"请选一种数据来源"。推荐失败（搜不到候选/LLM 调用出错/
        # 预览失败）时静默走到下面的兜底分支，回退到手动 data_picker，不报错
        recommended = await _try_recommend_dataset(state, client)
        if recommended:
            state.recommended_dataset = recommended
            top = recommended["candidates"][0]
            return StageResult(
                satisfied=False,
                question=f"你没有自己的数据，我根据任务描述找了一个合适的公开数据集：\n"
                         f"「{top['ref']}」——{top['rationale']}\n"
                         f"下面是预览，用这个还是自己选数据来源？",
                ui_hint="dataset_recommendation",
            )

    _bump_rounds(state, ConversationStage.PREPARING_DATA)
    return StageResult(
        satisfied=False,
        question="接下来需要训练数据。你可以手动粘贴几条「文本 | 标签」样本，也可以上传文件、"
                 "填服务器本地路径，或者搜索 HuggingFace / 魔搭上的公开数据集——"
                 "在下面选一种方式吧。",
        ui_hint="data_picker",
    )


# ── Stage 3: 选择训练后端（结构化卡片复用现有单选组）────────────────────────────

def choose_model(state: ConversationState, user_text: str, structured: Optional[dict],
                 client: LLMClient) -> StageResult:
    if state.task_spec and state.task_spec.task_type == TaskType.RL:
        # RL 任务的后端是由 task_type 唯一决定的（stable-baselines3），不是"三选一"，
        # 不展示后端选择卡片，直接写入并跳过这一问
        return StageResult(satisfied=True, updates={"model_backend": "rl"})

    if state.task_spec and state.task_spec.task_type == TaskType.LLM_FINETUNE:
        # 后端本身由 task_type 唯一决定（同 RL），但还需要额外问一句"用哪个底座模型"——
        # 不是三选一后端卡片，是一个需要校验的自由文本输入（任意 HF 模型 ID，不设白名单）
        candidate = None
        if user_text.strip() and not _looks_like_default_choice(user_text):
            candidate = user_text.strip()
        elif user_text.strip() and _looks_like_default_choice(user_text):
            candidate = _DEFAULT_LLM_FT_MODEL
        elif _rounds_exceeded(state, ConversationStage.CHOOSING_MODEL):
            candidate = _DEFAULT_LLM_FT_MODEL

        if candidate:
            validated, error = _try_validate_llm_ft_model(candidate)
            if validated:
                return StageResult(satisfied=True, updates={
                    "model_backend": "llm_finetune", "base_model_id": validated.model_id,
                })
            if candidate != _DEFAULT_LLM_FT_MODEL:
                # 用户自己给的模型不行——继续问，不用默认模型偷偷顶替用户的选择意图
                _bump_rounds(state, ConversationStage.CHOOSING_MODEL)
                return StageResult(satisfied=False,
                                   question=f"{error}\n\n换一个模型 ID 试试？（比如 {_DEFAULT_LLM_FT_MODEL}）")
            # 走到这里说明连兜底默认模型自己都校验失败了（比如网络问题）——如实告知，不假装成功
            return StageResult(satisfied=False,
                               question=f"连默认的小模型都校验失败了（{error}），可能是网络问题，"
                                        f"稍后再试，或者直接给一个你确定存在的模型 ID。")

        _bump_rounds(state, ConversationStage.CHOOSING_MODEL)
        return StageResult(
            satisfied=False,
            question=f"想用哪个底座模型做微调？可以是任意 HuggingFace 模型 ID（比如 "
                     f"{_DEFAULT_LLM_FT_MODEL}），模型不能太大——会先检查参数量和这台机器的"
                     f"内存是否够用。拿不定主意就说「你决定」，我用一个小模型先跑一版。",
        )

    if state.task_spec and state.task_spec.task_type == TaskType.IMAGE_CLASSIFICATION:
        # 后端由 task_type 唯一决定（CLIP 类视觉编码器冻结 + 可训练分类头），
        # 不是三选一——也不像 LLM_FINETUNE 那样问底座模型，直接用固定的编码器，
        # 用户没有选型需求（这是"分类头"而不是"整个模型"的微调）
        return StageResult(satisfied=True, updates={
            "model_backend": "vlm_image_cls", "base_model_id": _DEFAULT_VLM_CLS_ENCODER,
        })

    if state.task_spec and state.task_spec.task_type == TaskType.VLM_GENERATIVE:
        # 后端同样由 task_type 唯一决定；生成式 VLM 的底座模型不像 LLM_FINETUNE
        # 那样开放任意 HF ID 自由选择（范围收紧，见计划——这里先用一个验证过、
        # 已知能在本机跑得动的默认模型，不额外做一套模型校验/选型逻辑）
        return StageResult(satisfied=True, updates={
            "model_backend": "vlm_generative", "base_model_id": _DEFAULT_VLM_GEN_MODEL,
        })

    if structured and structured.get("type") == "backend_selected":
        backend = structured.get("model_backend")
        if backend not in ("sklearn", "pretrained_nn", "custom_nn"):
            return StageResult(satisfied=False, question="没有识别到有效的训练后端，请重新选一个。",
                               ui_hint="backend_picker")
        return StageResult(satisfied=True, updates={"model_backend": backend})

    if user_text.strip() and _looks_like_default_choice(user_text):
        return StageResult(satisfied=True, updates={"model_backend": "sklearn"})

    if _rounds_exceeded(state, ConversationStage.CHOOSING_MODEL):
        return StageResult(satisfied=True, updates={"model_backend": "sklearn"})

    _bump_rounds(state, ConversationStage.CHOOSING_MODEL)
    question = ("用哪种训练方式？sklearn 最快，几秒出结果；预训练模型微调和 LLM 自定义网络结构"
                "是真实神经网络训练，更慢但可能效果更好。拿不定主意就说「你决定」，我用 sklearn 先跑一版。")
    # 有记忆摘要说明这是重试（之前至少跑过一轮）——生成一句结合上一轮结果的针对性建议接在
    # 通用问句前面；memory_summary 为空或 LLM 调用失败时 suggest_retry_hint 直接返回空串，
    # 不影响这一步的健壮性
    hint = memory.suggest_retry_hint(state, client)
    if hint:
        question = f"{hint}\n\n{question}"
    return StageResult(satisfied=False, question=question, ui_hint="backend_picker")


# ── Stage 4: 训练配置（自由文本走轻量正则，不必为两个数字单开一次 LLM 调用）─────

_ITER_RE = re.compile(r"(\d+)\s*轮")
_PCT_RE  = re.compile(r"(\d{1,3})\s*%")
_DEC_RE  = re.compile(r"0\.\d+")


def _try_parse_training_config_text(text: str) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    m_iter = _ITER_RE.search(text)
    if m_iter:
        updates["max_iterations"] = min(10, max(1, int(m_iter.group(1))))
    m_pct = _PCT_RE.search(text)
    if m_pct:
        updates["target_metric"] = min(1.0, max(0.0, int(m_pct.group(1)) / 100))
    else:
        m_dec = _DEC_RE.search(text)
        if m_dec:
            updates["target_metric"] = float(m_dec.group(0))
    return updates


def configure_training(state: ConversationState, user_text: str, structured: Optional[dict],
                       client: LLMClient) -> StageResult:
    if structured and structured.get("type") == "training_config":
        updates: Dict[str, Any] = {}
        if "max_iterations" in structured:
            updates["max_iterations"] = int(structured["max_iterations"])
        if "target_metric" in structured:
            updates["target_metric"] = float(structured["target_metric"])
        return StageResult(satisfied=True, updates=updates)

    if user_text.strip():
        if _looks_like_default_choice(user_text):
            return StageResult(satisfied=True, updates={})
        parsed = _try_parse_training_config_text(user_text)
        if parsed:
            return StageResult(satisfied=True, updates=parsed)

    if _rounds_exceeded(state, ConversationStage.CONFIGURING_TRAINING):
        return StageResult(satisfied=True, updates={})

    _bump_rounds(state, ConversationStage.CONFIGURING_TRAINING)
    return StageResult(
        satisfied=False,
        question=f"最多训练几轮、目标指标是多少？不确定的话直接说「默认就行」"
                 f"（当前默认：最多 {state.max_iterations} 轮，目标 {state.target_metric:.0%}）。",
        ui_hint="training_config",
    )


STAGE_HANDLERS = {
    ConversationStage.CLARIFYING_TASK:      clarify_task,
    ConversationStage.PREPARING_DATA:       prepare_data,
    ConversationStage.CHOOSING_MODEL:       choose_model,
    ConversationStage.CONFIGURING_TRAINING: configure_training,
}
