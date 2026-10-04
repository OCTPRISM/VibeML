"""
core/archive_extractor.py  -  聊天附件"上传压缩包"的安全解压

支持 zip / 7z / tar（含 .tar.gz/.tgz/.tar.bz2/.tbz2）/ 单文件 gz/bz2 / rar。
每种格式一个函数，统一走 extract_archive() 总入口，返回：
  (extracted_paths, skipped[(name, reason)], error)
error 非 None 表示整个压缩包都打不开（格式不对/损坏/工具缺失）；
skipped 是压缩包内某些成员因为不安全/超限被跳过，但压缩包本身处理成功。

安全边界（针对"解压任意上传的压缩包"这件事本身就有风险，不是可选项）：
  - 路径穿越（zip slip）：每个成员解压后的目标路径必须仍在目标目录内，
    绝对路径/包含 .. 的成员直接跳过
  - 压缩炸弹：解压后总大小超过 MAX_EXTRACTED_TOTAL_BYTES 就不再继续解压
    剩余成员（已解压的部分保留，不是整体失败）
  - 符号链接：一律跳过，不跟随也不解压
  - 文件数量：超过 MAX_ARCHIVE_MEMBERS 就不再解压剩余成员
  - 不做递归解压——压缩包里如果还嵌套着压缩包，内层压缩包原样跳过
    （标注"暂不支持嵌套压缩包"），避免压缩包套压缩包无限展开
"""
from __future__ import annotations

import bz2
import gzip
import stat
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

MAX_ARCHIVE_MEMBERS = 200
MAX_EXTRACTED_TOTAL_BYTES = 100 * 1024 * 1024   # 100MB——解压后总大小上限，防压缩炸弹
_CHUNK = 1024 * 1024

ZIP_SUFFIXES = (".zip",)
SEVENZ_SUFFIXES = (".7z",)
TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
SINGLE_GZ_SUFFIXES = (".gz",)     # 不是 .tar.gz 的情况下，纯 gzip 压缩单个文件
SINGLE_BZ2_SUFFIXES = (".bz2",)   # 同理，纯 bzip2 压缩单个文件
RAR_SUFFIXES = (".rar",)

ARCHIVE_SUFFIXES = ZIP_SUFFIXES + SEVENZ_SUFFIXES + TAR_SUFFIXES + SINGLE_GZ_SUFFIXES + SINGLE_BZ2_SUFFIXES + RAR_SUFFIXES
_NESTED_ARCHIVE_HINT_SUFFIXES = ARCHIVE_SUFFIXES   # 用来识别"压缩包里的压缩包"，原样跳过不解压


def matched_suffix(filename: str) -> Optional[str]:
    """返回 filename 命中的归档后缀（可能是多段的，比如 .tar.gz），命中 TAR_SUFFIXES
    要按长度降序比较，否则 "x.tar.gz" 会先被 ".gz" 命中，错误当成单文件 gzip 处理。"""
    lower = filename.lower()
    for suffix in sorted(ARCHIVE_SUFFIXES, key=len, reverse=True):
        if lower.endswith(suffix):
            return suffix
    return None


def _safe_target(base_dir: Path, member_name: str) -> Optional[Path]:
    if not member_name or member_name.startswith("/") or member_name.startswith("\\"):
        return None
    base_resolved = base_dir.resolve()
    target = (base_dir / member_name).resolve()
    if target != base_resolved and base_resolved not in target.parents:
        return None
    return target


def _is_nested_archive(name: str) -> bool:
    return matched_suffix(name) is not None


class _Budget:
    """成员数量 + 解压后总字节数的共用预算，三种归档格式共用同一份判断逻辑。"""
    def __init__(self):
        self.count = 0
        self.total_bytes = 0

    def accepts(self, size: int) -> bool:
        if self.count >= MAX_ARCHIVE_MEMBERS:
            return False
        if self.total_bytes + size > MAX_EXTRACTED_TOTAL_BYTES:
            return False
        return True

    def register(self, size: int) -> None:
        self.count += 1
        self.total_bytes += size


def _extract_zip(archive_path: Path, out_dir: Path) -> Tuple[List[Path], List[Tuple[str, str]]]:
    extracted, skipped = [], []
    budget = _Budget()
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            unix_mode = info.external_attr >> 16
            if unix_mode and stat.S_ISLNK(unix_mode):
                skipped.append((info.filename, "跳过符号链接")); continue
            if _is_nested_archive(info.filename):
                skipped.append((info.filename, "暂不支持嵌套压缩包")); continue
            target = _safe_target(out_dir, info.filename)
            if target is None:
                skipped.append((info.filename, "文件路径不安全，已跳过")); continue
            if not budget.accepts(info.file_size):
                skipped.append((info.filename, "超过解压数量/总大小上限，已跳过")); continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            budget.register(info.file_size)
            extracted.append(target)
    return extracted, skipped


def _extract_tar(archive_path: Path, out_dir: Path) -> Tuple[List[Path], List[Tuple[str, str]]]:
    extracted, skipped = [], []
    budget = _Budget()
    with tarfile.open(archive_path, mode="r:*") as tf:
        for member in tf.getmembers():
            if member.isdir():
                continue
            if member.issym() or member.islnk():
                skipped.append((member.name, "跳过符号链接")); continue
            if not member.isfile():
                skipped.append((member.name, "跳过非普通文件（设备/管道等）")); continue
            if _is_nested_archive(member.name):
                skipped.append((member.name, "暂不支持嵌套压缩包")); continue
            target = _safe_target(out_dir, member.name)
            if target is None:
                skipped.append((member.name, "文件路径不安全，已跳过")); continue
            if not budget.accepts(member.size):
                skipped.append((member.name, "超过解压数量/总大小上限，已跳过")); continue
            src = tf.extractfile(member)
            if src is None:
                skipped.append((member.name, "无法读取该成员")); continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(src.read())
            budget.register(member.size)
            extracted.append(target)
    return extracted, skipped


def _extract_7z(archive_path: Path, out_dir: Path) -> Tuple[List[Path], List[Tuple[str, str]]]:
    import py7zr
    extracted, skipped = [], []
    budget = _Budget()
    targets = []
    with py7zr.SevenZipFile(archive_path, mode="r") as zf:
        for info in zf.list():
            if info.is_directory:
                continue
            if info.is_symlink:
                skipped.append((info.filename, "跳过符号链接")); continue
            if _is_nested_archive(info.filename):
                skipped.append((info.filename, "暂不支持嵌套压缩包")); continue
            if _safe_target(out_dir, info.filename) is None:
                skipped.append((info.filename, "文件路径不安全，已跳过")); continue
            if not budget.accepts(info.uncompressed):
                skipped.append((info.filename, "超过解压数量/总大小上限，已跳过")); continue
            budget.register(info.uncompressed)
            targets.append(info.filename)
    if targets:
        with py7zr.SevenZipFile(archive_path, mode="r") as zf:
            zf.extract(path=out_dir, targets=targets)
        # 跟 zip/tar 的抽取路径保持一致，统一走 _safe_target() 算出的 resolve() 后的路径——
        # 不然 out_dir 在 macOS 上是 /var/folders/... 这种指向 /private/var/... 的符号链接时，
        # 这里跟 zip/tar 分支返回的路径表示形式不一样，调用方用 relative_to() 会报错
        extracted = [t for name in targets if (t := _safe_target(out_dir, name)) is not None and t.is_file()]
    return extracted, skipped


def _extract_single_compressed(archive_path: Path, filename: str, out_dir: Path, opener) -> Tuple[List[Path], List[Tuple[str, str]]]:
    """.gz/.bz2 包的不是多文件归档，而是单个文件本身被压缩——流式读取、边读边数字节数，
    超过总大小上限就中止（不像 zip/tar 那样有元数据里现成的"解压后大小"可以先看一眼）。
    注意用 filename（原始上传文件名）而不是 archive_path.stem 算内层文件名——调用方
    （api/routes/attachments.py）落盘时用的是固定的临时文件名（比如 "archive"），
    不是用户上传时的真实文件名，从那个路径算 stem 会丢失真实的内层文件名。"""
    inner_name = Path(filename).stem   # "data.csv.gz" -> "data.csv"
    target = _safe_target(out_dir, inner_name) or (out_dir / "解压内容")
    total = 0
    with opener(archive_path) as src, open(target, "wb") as dst:
        while True:
            chunk = src.read(_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_EXTRACTED_TOTAL_BYTES:
                dst.close()
                target.unlink(missing_ok=True)
                return [], [(archive_path.name, "解压后内容超过大小上限，已中止")]
            dst.write(chunk)
    return [target], []


def _extract_rar(archive_path: Path, out_dir: Path) -> Tuple[List[Path], List[Tuple[str, str]]]:
    import rarfile
    extracted, skipped = [], []
    budget = _Budget()
    with rarfile.RarFile(archive_path) as rf:
        for info in rf.infolist():
            if info.is_dir():
                continue
            if info.filename and (info.filename.startswith("/") or ".." in info.filename.split("/")):
                skipped.append((info.filename, "文件路径不安全，已跳过")); continue
            if _is_nested_archive(info.filename):
                skipped.append((info.filename, "暂不支持嵌套压缩包")); continue
            target = _safe_target(out_dir, info.filename)
            if target is None:
                skipped.append((info.filename, "文件路径不安全，已跳过")); continue
            if not budget.accepts(info.file_size):
                skipped.append((info.filename, "超过解压数量/总大小上限，已跳过")); continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with rf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            budget.register(info.file_size)
            extracted.append(target)
    return extracted, skipped


def extract_archive(archive_path: Path, filename: str, out_dir: Path) -> Tuple[List[Path], List[Tuple[str, str]], Optional[str]]:
    """suffix 从 filename 重新计算（不是外部传入的单段后缀）——归档后缀可能是 .tar.gz
    这种多段形式，调用方（api/routes/attachments.py）不需要关心这个细节。"""
    suffix = matched_suffix(filename)
    if suffix is None:
        return [], [], f"不是支持的压缩包格式：{filename}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if suffix in ZIP_SUFFIXES:
            extracted, skipped = _extract_zip(archive_path, out_dir)
        elif suffix in TAR_SUFFIXES:
            extracted, skipped = _extract_tar(archive_path, out_dir)
        elif suffix in SEVENZ_SUFFIXES:
            extracted, skipped = _extract_7z(archive_path, out_dir)
        elif suffix in SINGLE_GZ_SUFFIXES:
            extracted, skipped = _extract_single_compressed(archive_path, filename, out_dir, gzip.open)
        elif suffix in SINGLE_BZ2_SUFFIXES:
            extracted, skipped = _extract_single_compressed(archive_path, filename, out_dir, bz2.open)
        elif suffix in RAR_SUFFIXES:
            extracted, skipped = _extract_rar(archive_path, out_dir)
        else:
            return [], [], f"不是支持的压缩包格式：{filename}"
    except Exception as e:
        if suffix in RAR_SUFFIXES:
            return [], [], ("无法解压这个 RAR 文件——本环境没有安装 unrar 工具，"
                            "只能处理不需要外部工具的部分 RAR 包；建议改用 zip/7z 格式重新打包后上传。"
                            f"（详细原因：{e}）")
        return [], [], f"解压失败：{e}"
    return extracted, skipped, None
