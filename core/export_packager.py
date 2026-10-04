"""
core/export_packager.py  -  把训练产出打包成可下载的 zip

训练结束时 core/*_deployer.py 已经在 ./deploy/<task_id>/ 下产出了一个完整的、
独立可运行的部署包（模型文件 + inference.py + README）——这个模块**不重新生成
任何产物**，只是给已有的目录加两个"出口视图"：

  模型包（build_model_package）：权重 + 推理脚本 + README + 新生成的 model_card.md
      → 用户下载到本地，或者之后上传到 HuggingFace / 魔搭
  代码包（build_code_package）：inference.py + LLM 生成的架构/环境源码 +
      新生成的 train.py / requirements.txt / README
      → 用户下载到本地，或者之后推到 GitHub
      **不含权重文件**——代码仓库里塞二进制是反模式，README 里指向模型页即可

两个包都过滤 macOS 的 ._* AppleDouble 伴生文件：这个仓库挂在不支持原生
resource fork 的外部卷上，deploy/ 目录里真实存在这些文件，不过滤的话用户
下载到的包里会混进一堆看不懂的二进制垃圾。
"""
from __future__ import annotations

import json
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

DEPLOY_ROOT = Path("./deploy")

# 权重/模型二进制的后缀——代码包要排除这些
_MODEL_BINARY_SUFFIXES = (".pkl", ".onnx", ".joblib", ".bin", ".safetensors", ".pt", ".pth", ".ckpt")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


class ExportError(Exception):
    """打包失败——message 直接面向用户展示。"""


def _is_junk(path: Path) -> bool:
    """macOS AppleDouble 伴生文件（._foo）和 .DS_Store 一律不打进包里。"""
    return path.name.startswith("._") or path.name == ".DS_Store"


def resolve_deploy_dir(task_id: str) -> Path:
    """task_id 必须是 UUID 格式，且拼出来的路径 resolve() 后必须仍在 deploy/ 内——
    照抄 core/data_sources.py::LocalUploadSource._load_rows 已有的路径穿越防护写法。
    这个端点是拿 task_id 直接拼路径的，不做这层校验就等于开了任意文件读取。"""
    if not _UUID_RE.match(task_id or ""):
        raise ExportError("任务 ID 格式不合法")
    root = DEPLOY_ROOT.resolve()
    path = (DEPLOY_ROOT / task_id).resolve()
    if root not in path.parents:
        raise ExportError("非法的任务引用")
    if not path.is_dir():
        raise ExportError("这个任务还没有产出部署包——可能训练尚未完成，或者训练失败了")
    return path


def _find_event(event_log: List[dict], event_type: str) -> Optional[dict]:
    for ev in reversed(event_log or []):
        if ev.get("type") == event_type:
            return ev
    return None


def _build_model_card(task_id: str, deploy_dir: Path, event_log: List[dict]) -> str:
    """model_card.md 是 HuggingFace / 魔搭 都认的模型说明格式。内容全部来自
    已经真实产生过的事件（finished / task_parsed / data_ready），不编造指标。"""
    finished = _find_event(event_log, "finished") or {}
    parsed = _find_event(event_log, "task_parsed") or {}
    data_ready = _find_event(event_log, "data_ready") or {}

    metric_name = finished.get("metric_name", "指标")
    best = finished.get("best_metric")
    labels = finished.get("labels") or parsed.get("labels") or []

    lines = [
        "---",
        "library_name: sklearn",
        "tags:",
        "  - vibe-ml-studio",
        "  - automl",
        "---",
        "",
        f"# 模型卡片 · {task_id}",
        "",
        "本模型由 Vibe ML Studio 通过对话式 AutoML 流程自动训练产出。",
        "",
        "## 任务",
        "",
        f"- 任务类型：{parsed.get('task_type', '未知')}",
        f"- 领域：{parsed.get('domain', '未标注')}",
    ]
    if labels:
        lines.append(f"- 类别：{', '.join(str(x) for x in labels)}")
    lines += ["", "## 训练结果", ""]
    if best is not None:
        # RL 的 episode_reward_mean 不是 0-1 有界比例，不能按百分比格式化
        formatted = f"{best:.1%}" if metric_name != "episode_reward_mean" else f"{best:.2f}"
        lines.append(f"- {metric_name}：{formatted}")
    else:
        lines.append("- 未记录到最终指标")
    if data_ready:
        lines.append(f"- 训练样本量：{data_ready.get('total_samples', '未知')}"
                     f"（含增强 {data_ready.get('augmented', 0)} 条）")
    lines += [
        "",
        "## 使用方式",
        "",
        "包内的 `inference.py` 是独立可运行的推理脚本，不依赖 Vibe ML Studio 本身：",
        "",
        "```bash",
        "python inference.py",
        "```",
        "",
        "## 局限性",
        "",
        "- 这是在少量样本上自动训练出的模型，未经过系统性的分布外/对抗性评估。",
        "- 上面报告的指标来自训练时切分出的验证集，不代表在你自己的真实数据上的表现。",
        "- 投入生产前请用你自己的数据独立验证。",
        "",
    ]
    return "\n".join(lines)


def _build_requirements(event_log: List[dict]) -> str:
    """按这次训练实际用到的后端给依赖，不是把整个项目的 requirements 抄过去——
    一个 sklearn 模型的代码包不该要求用户装 torch。"""
    arch = _find_event(event_log, "arch_designed") or {}
    mode = arch.get("mode")
    base = ["scikit-learn>=1.3.0", "numpy>=1.24.0"]
    if mode in ("custom_nn", "pretrained_nn"):
        base += ["torch>=2.1.0", "transformers>=4.40.0"]
    return "\n".join(base) + "\n"


def _build_train_script(task_id: str, event_log: List[dict]) -> str:
    """一份"这次训练是怎么跑出来的"的可复现说明脚本。

    **刻意不内联训练样本数据**：如果用户当初是手动粘贴的样本，那些内容可能是
    敏感的业务数据，而代码包是要推到 GitHub 的——把数据写进去等于把它们公开。
    这里只写数据来源的描述，需要复现的人自己提供数据。"""
    parsed = _find_event(event_log, "task_parsed") or {}
    data_ready = _find_event(event_log, "data_ready") or {}
    arch = _find_event(event_log, "arch_designed") or {}
    finished = _find_event(event_log, "finished") or {}

    return f'''"""
train.py  -  复现这次训练的说明脚本（由 Vibe ML Studio 自动生成）

任务 ID：{task_id}
任务类型：{parsed.get("task_type", "未知")}
训练后端：{arch.get("mode", "sklearn")}
最终指标：{finished.get("metric_name", "指标")} = {finished.get("best_metric", "未记录")}

数据来源：{data_ready.get("total_samples", "未知")} 条样本
  （其中自动增强 {data_ready.get("augmented", 0)} 条）

注意：出于隐私考虑，这个脚本**不包含原始训练数据**——如果原始数据是手动粘贴的
业务样本，把它们写进一个可能被推到公开代码仓库的文件里是不合适的。要复现训练，
请把你自己的数据按下面 EXAMPLES 的格式填进去。
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 按 [{{"text": "...", "label": "..."}}, ...] 的格式填入你自己的数据
EXAMPLES = []

if not EXAMPLES:
    raise SystemExit("请先在 EXAMPLES 里填入训练数据（见文件顶部说明）")

texts = [e["text"] for e in EXAMPLES]
labels = [e["label"] for e in EXAMPLES]
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels)

model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
'''


def _write_zip(files: Dict[str, bytes], prefix: str) -> Path:
    """files: {包内相对路径: 内容}。返回临时 zip 路径，调用方负责用完删除。"""
    tmp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".zip", delete=False)
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, content in files.items():
            zf.writestr(arcname, content)
    return Path(tmp.name)


def build_model_package(task_id: str, event_log: Optional[List[dict]] = None) -> Path:
    deploy_dir = resolve_deploy_dir(task_id)
    files: Dict[str, bytes] = {}
    for path in sorted(deploy_dir.rglob("*")):
        if path.is_dir() or _is_junk(path):
            continue
        files[str(path.relative_to(deploy_dir))] = path.read_bytes()
    if not files:
        raise ExportError("部署目录是空的，没有可以打包的内容")
    files["model_card.md"] = _build_model_card(task_id, deploy_dir, event_log or []).encode("utf-8")
    return _write_zip(files, f"model-{task_id[:8]}-")


def build_code_package(task_id: str, event_log: Optional[List[dict]] = None) -> Path:
    deploy_dir = resolve_deploy_dir(task_id)
    event_log = event_log or []
    files: Dict[str, bytes] = {}

    # 部署目录里的代码/文本文件（排除权重二进制）
    for path in sorted(deploy_dir.rglob("*")):
        if path.is_dir() or _is_junk(path):
            continue
        if path.suffix.lower() in _MODEL_BINARY_SUFFIXES:
            continue
        files[str(path.relative_to(deploy_dir))] = path.read_bytes()

    # LLM 生成的模型架构 / RL 环境源码——训练时已经通过 arch_designed 事件
    # 存进 event_log 了，不用重新问一次 LLM
    arch = _find_event(event_log, "arch_designed") or {}
    source_code = arch.get("source_code") or arch.get("code")
    if source_code:
        name = "model_architecture.py" if arch.get("mode") != "rl" else "environment.py"
        files[name] = str(source_code).encode("utf-8")

    files["train.py"] = _build_train_script(task_id, event_log).encode("utf-8")
    files["requirements.txt"] = _build_requirements(event_log).encode("utf-8")
    files["README.md"] = (
        f"# 模型代码包 · {task_id}\n\n"
        "由 Vibe ML Studio 自动生成。这个包**只含代码，不含训练好的权重文件**——\n"
        "权重请从对应的模型包下载（或者从模型托管平台上的模型页获取）。\n\n"
        "## 文件说明\n\n"
        "- `inference.py`：独立可运行的推理脚本（需要配合权重文件使用）\n"
        + ("- `model_architecture.py`：LLM 生成的模型结构源码\n" if source_code and arch.get("mode") != "rl" else "")
        + ("- `environment.py`：LLM 生成的强化学习环境源码\n" if source_code and arch.get("mode") == "rl" else "")
        + "- `train.py`：复现训练的说明脚本（不含原始数据，见文件内说明）\n"
        "- `requirements.txt`：这次训练实际用到的依赖\n"
    ).encode("utf-8")
    return _write_zip(files, f"code-{task_id[:8]}-")
