"""
core/attachment_extractor.py  -  聊天输入框"添加文件"附件的文本提取

只负责"从文件字节里挖出纯文本"这一件事，不管上传/存盘/HTTP——那些是
api/routes/attachments.py 的事。每个格式一个函数，互相独立，一个格式挂了
不影响其它格式；对外只暴露一个 extract_text() 总入口，返回 (text, error)，
提取失败时 text=None、error 是给用户看的一句话解释（附件依然会被保留，
只是喂不进对话文本里），不抛异常，因为"这个文档解析不了"不该打断整个上传流程。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

# 超过这个长度就截断——附件是喂给 LLM 补充上下文用的，不是要完整搬运一整本书；
# 单个附件动辄几万字会让对话历史迅速膨胀，也会占掉大部分 prompt 预算
MAX_EXTRACTED_CHARS = 20_000

TEXT_SUFFIXES = (".md", ".markdown", ".txt")
PDF_SUFFIXES = (".pdf",)
DOCX_SUFFIXES = (".docx",)
XLSX_SUFFIXES = (".xlsx",)
# .doc/.xls 是旧的二进制 Office 格式，python-docx/openpyxl 都不认——列在这里只是
# 为了在 _extract_by_suffix 里给出"这个旧格式暂不支持解析"的明确提示，而不是让
# 用户以为是上传失败
LEGACY_OFFICE_SUFFIXES = (".doc", ".xls")


def _truncate(text: str) -> str:
    text = text.strip()
    if len(text) > MAX_EXTRACTED_CHARS:
        return text[:MAX_EXTRACTED_CHARS] + f"\n…（内容过长，已截断，原文共 {len(text)} 字）"
    return text


def _extract_plain_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    import docx
    doc = docx.Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    # 表格里的文字 python-docx 不会算进 paragraphs，很多真实文档（需求表/参数表）
    # 恰恰是表格形式，漏掉表格等于漏掉大半有效信息
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _extract_xlsx(path: Path) -> str:
    import openpyxl
    wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    parts = []
    for sheet in wb.worksheets:
        parts.append(f"# 表：{sheet.title}")
        row_count = 0
        for row in sheet.iter_rows(values_only=True):
            if row_count >= 500:   # 单表最多摘 500 行，超大表格没必要整表塞进对话
                parts.append("…（该表剩余行已省略）")
                break
            cells = [str(c) for c in row if c is not None]
            if cells:
                parts.append(" | ".join(cells))
                row_count += 1
    return "\n".join(parts)


def extract_text(path: Path, suffix: str) -> Tuple[Optional[str], Optional[str]]:
    """suffix 传小写、带点（如 ".pdf"）。返回 (text, error)——正好一个是 None。"""
    suffix = suffix.lower()
    try:
        if suffix in TEXT_SUFFIXES:
            text = _extract_plain_text(path)
        elif suffix in PDF_SUFFIXES:
            text = _extract_pdf(path)
        elif suffix in DOCX_SUFFIXES:
            text = _extract_docx(path)
        elif suffix in XLSX_SUFFIXES:
            text = _extract_xlsx(path)
        elif suffix in LEGACY_OFFICE_SUFFIXES:
            return None, "这是旧版 Office 格式（.doc/.xls），暂不支持提取内容——文件本身已保留为附件，建议另存为 .docx/.xlsx 后重新上传"
        else:
            return None, f"不支持解析 {suffix} 格式的内容"
    except Exception as e:
        return None, f"解析失败：{e}"

    text = text.strip()
    if not text:
        return None, "没有从文件里提取到文字内容（可能是扫描件/纯图片 PDF，需要 OCR 才能读取，暂不支持）"
    return _truncate(text), None
