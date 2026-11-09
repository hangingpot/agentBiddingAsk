"""从 data/ 读取语料。新增文件不用改代码，重新 ingest 即可。"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
JSON_SUFFIXES = {".json"}
PDF_SUFFIXES = {".pdf"}
DOCX_SUFFIXES = {".docx"}
ZIP_SUFFIXES = {".zip"}
SUPPORTED = TEXT_SUFFIXES | JSON_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES | ZIP_SUFFIXES

MAX_TEXT_BYTES = 15 * 1024 * 1024
MAX_ZIP_MEMBER = 30 * 1024 * 1024
CHUNK_SIZE = 680
CHUNK_OVERLAP = 120

_QUESTION_RE = re.compile(r"^.{4,80}[？?]$")


@dataclass
class Chunk:
    id: str
    title: str
    text: str
    search_text: str
    source: str
    kind: str
    question: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> Chunk:
        return cls(
            id=data["id"],
            title=data["title"],
            text=data["text"],
            search_text=data.get("search_text") or data["text"],
            source=data.get("source", ""),
            kind=data.get("kind", "doc"),
            question=data.get("question", ""),
        )


def load_corpus(data_dir: Path) -> list[Chunk]:
    """遍历 data_dir，解析支持的文件并去重。"""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"语料目录不存在：{data_dir}")

    raw: list[Chunk] = []
    files = sorted(
        path for path in data_dir.rglob("*") if path.is_file() and not _ignored(path, data_dir)
    )
    if not files:
        raise FileNotFoundError(f"{data_dir} 下没有可索引的文件")

    for path in files:
        suffix = path.suffix.lower()
        try:
            source = path.relative_to(data_dir.parent).as_posix()
        except ValueError:
            source = path.name
        try:
            if suffix in ZIP_SUFFIXES:
                raw.extend(_load_zip(path, source))
            else:
                raw.extend(_load_file(path, source))
        except Exception:
            logger.exception("跳过无法解析的文件 %s", path)

    chunks = _dedupe(raw)
    if not chunks:
        raise RuntimeError(f"{data_dir} 中的文件没有解析出任何片段")
    for index, chunk in enumerate(chunks, start=1):
        prefix = "faq" if chunk.kind == "faq" else "doc"
        chunk.id = f"{prefix}-{index:04d}"
    return chunks


def fingerprint(data_dir: Path) -> str:
    """用路径、大小、修改时间判断语料是否变化。"""
    data_dir = Path(data_dir)
    rows: list[str] = []
    if data_dir.is_dir():
        for path in sorted(data_dir.rglob("*")):
            if not path.is_file() or _ignored(path, data_dir):
                continue
            stat = path.stat()
            rel = path.relative_to(data_dir).as_posix()
            rows.append(f"{rel}:{stat.st_size}:{stat.st_mtime_ns}")
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return digest


def _ignored(path: Path, data_dir: Path) -> bool:
    rel = path.relative_to(data_dir)
    if any(part.startswith(".") for part in rel.parts):
        return True
    if path.name.lower() in {"readme.md", "readme.txt"}:
        return True
    if path.suffix.lower() not in SUPPORTED and path.suffix.lower() != "":
        # 无后缀的文件不当作语料；有后缀但不支持的也跳过（图片等）。
        return path.suffix.lower() not in SUPPORTED
    return path.suffix.lower() not in SUPPORTED


def _load_file(path: Path, source: str) -> list[Chunk]:
    suffix = path.suffix.lower()
    if path.stat().st_size > MAX_TEXT_BYTES and suffix not in ZIP_SUFFIXES:
        logger.warning("文件超过 %s 字节，已跳过：%s", MAX_TEXT_BYTES, path)
        return []
    data = path.read_bytes()
    return _chunks_from_bytes(data, suffix, source)


def _load_zip(path: Path, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts:
                logger.warning("跳过 zip 内可疑路径 %s", name)
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in (TEXT_SUFFIXES | JSON_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES):
                continue
            if info.file_size > MAX_ZIP_MEMBER:
                logger.warning("跳过 zip 内过大文件 %s", name)
                continue
            data = archive.read(info)
            member_source = f"{source}!{name}"
            chunks.extend(_chunks_from_bytes(data, suffix, member_source))
    return chunks


def _chunks_from_bytes(data: bytes, suffix: str, source: str) -> list[Chunk]:
    if suffix in JSON_SUFFIXES:
        text = _decode(data)
        return _chunks_from_json(text, source)
    if suffix in TEXT_SUFFIXES:
        text = _decode(data)
        faq = _parse_faq_text(text)
        if faq:
            return [_faq_chunk(q, a, source) for q, a in faq]
        return _chunks_from_prose(text, source)
    if suffix in PDF_SUFFIXES:
        text = _read_pdf(data)
        return _chunks_from_prose(text, source) if text.strip() else []
    if suffix in DOCX_SUFFIXES:
        text = _read_docx(data)
        return _chunks_from_prose(text, source) if text.strip() else []
    return []


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _chunks_from_json(text: str, source: str) -> list[Chunk]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("JSON 无法解析，按普通文本切分：%s", source)
        return _chunks_from_prose(text, source)

    records = _json_records(payload)
    if records:
        return [_faq_chunk(q, a, source) for q, a in records]
    if isinstance(payload, list) and payload and all(isinstance(item, str) for item in payload):
        return _chunks_from_prose("\n\n".join(payload), source)
    return _chunks_from_prose(text, source)


def _json_records(payload: object) -> list[tuple[str, str]]:
    if isinstance(payload, dict):
        for key in ("items", "faq", "data", "records", "questions"):
            if key in payload:
                return _json_records(payload[key])
        pair = _pair_from_mapping(payload)
        return [pair] if pair else []
    if isinstance(payload, list):
        pairs: list[tuple[str, str]] = []
        for item in payload:
            if isinstance(item, dict):
                pair = _pair_from_mapping(item)
                if pair:
                    pairs.append(pair)
        return pairs
    return []


def _pair_from_mapping(item: dict) -> tuple[str, str] | None:
    keys = {str(key).lower(): value for key, value in item.items()}
    question = ""
    answer = ""
    for name in ("q", "question", "query", "title", "ask"):
        if keys.get(name):
            question = str(keys[name]).strip()
            break
    for name in ("a", "answer", "response", "content", "reply"):
        if keys.get(name):
            answer = str(keys[name]).strip()
            break
    if question and answer:
        return question, answer
    return None


def _parse_faq_text(text: str) -> list[tuple[str, str]] | None:
    """识别「问题行 + 答案行」的百问榜纯文本。不像问答集时返回 None。"""
    pairs: list[tuple[str, str]] = []
    current_q: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current_q, buffer
        answer = "\n".join(line for line in buffer if line).strip()
        if current_q and answer:
            pairs.append((current_q, answer))
        current_q = None
        buffer = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if current_q is not None:
                buffer.append("")
            continue
        if "\t" in line:
            left, right = line.split("\t", 1)
            left, right = left.strip(), right.strip()
            if _is_question(left):
                flush()
                current_q = left
                buffer = [right] if right else []
                continue
        if _is_question(line):
            flush()
            current_q = line
            buffer = []
            continue
        if current_q is not None:
            buffer.append(line)
    flush()

    if len(pairs) < 5:
        return None
    answer_chars = sum(len(answer) for _, answer in pairs)
    if answer_chars < 200:
        return None
    return pairs


def _is_question(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact or len(compact) > 80:
        return False
    return bool(_QUESTION_RE.match(compact))


def _faq_chunk(question: str, answer: str, source: str) -> Chunk:
    question = question.strip()
    answer = answer.strip()
    text = f"问题：{question}\n答案：{answer}"
    search_text = f"{question}\n{question}\n{answer}"
    return Chunk(
        id="",
        title=question,
        text=text,
        search_text=search_text,
        source=source,
        kind="faq",
        question=question,
    )


def _chunks_from_prose(text: str, source: str) -> list[Chunk]:
    body = text.strip()
    if not body:
        return []
    windows = _windows(body, CHUNK_SIZE, CHUNK_OVERLAP)
    name = source.rsplit("/", 1)[-1]
    chunks: list[Chunk] = []
    total = len(windows)
    for index, window in enumerate(windows, start=1):
        heading = window.strip().splitlines()[0][:40]
        title = f"{name} #{index}" if total > 1 else name
        if heading:
            title = f"{title} {heading}"
        chunks.append(
            Chunk(
                id="",
                title=title[:80],
                text=window.strip(),
                search_text=window.strip(),
                source=source,
                kind="doc",
            )
        )
    return chunks


def _windows(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    windows: list[str] = []
    start = 0
    while start < len(text):
        windows.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return windows


def _read_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages[:400]:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _read_docx(data: bytes) -> str:
    from docx import Document

    document = Document(io.BytesIO(data))
    parts = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    return "\n".join(parts)


def _question_key(question: str) -> str:
    text = re.sub(r"\s+", "", question).lower()
    return text.replace("？", "?").replace("（", "(").replace("）", ")")


def _dedupe(chunks: list[Chunk]) -> list[Chunk]:
    seen_questions: set[str] = set()
    seen_text: set[str] = set()
    kept: list[Chunk] = []
    for chunk in chunks:
        text_key = re.sub(r"\s+", "", chunk.text)
        if chunk.kind == "faq" and chunk.question:
            key = _question_key(chunk.question)
            if key in seen_questions:
                continue
            seen_questions.add(key)
        if text_key in seen_text:
            continue
        seen_text.add(text_key)
        kept.append(chunk)
    return kept
