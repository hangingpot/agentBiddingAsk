"""从 data/ 读取语料。新增文件不用改代码，重新 ingest 即可。"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".markdown"}
JSON_SUFFIXES = {".json"}
PDF_SUFFIXES = {".pdf"}
DOCX_SUFFIXES = {".docx"}
DOC_SUFFIXES = {".doc"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
ZIP_SUFFIXES = {".zip"}
SUPPORTED = (
    TEXT_SUFFIXES | JSON_SUFFIXES | PDF_SUFFIXES | DOCX_SUFFIXES | DOC_SUFFIXES | IMAGE_SUFFIXES | ZIP_SUFFIXES
)

MAX_TEXT_BYTES = 15 * 1024 * 1024
MAX_DOC_BYTES = 120 * 1024 * 1024
MAX_ZIP_MEMBER = 80 * 1024 * 1024
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
    images: list[str] = field(default_factory=list)

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
            images=list(data.get("images") or []),
        )


def load_corpus(
    data_dir: Path,
    extra_dirs: list[Path] | None = None,
    assets_dir: Path | None = None,
    ocr=None,
) -> list[Chunk]:
    """遍历语料目录。手册可以在 data/ 里，也可以在额外的 MANUALS_DIR。"""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"语料目录不存在：{data_dir}")
    roots = _corpus_roots(data_dir, extra_dirs)

    raw: list[Chunk] = []
    files: list[tuple[Path, Path]] = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not _ignored(path, root):
                files.append((path, root))
    if not files:
        raise FileNotFoundError(f"{data_dir} 下没有可索引的文件")

    for path, root in files:
        suffix = path.suffix.lower()
        try:
            source = path.resolve().relative_to(root.resolve()).as_posix()
            source = f"{root.name}/{source}"
        except ValueError:
            source = path.name
        try:
            if suffix in ZIP_SUFFIXES:
                raw.extend(_load_zip(path, source, assets_dir, ocr))
            else:
                raw.extend(_load_file(path, source, assets_dir, ocr))
        except Exception:
            logger.exception("跳过无法解析的文件 %s", path)

    chunks = _dedupe(raw)
    if not chunks:
        raise RuntimeError(f"{data_dir} 中的文件没有解析出任何片段")
    for index, chunk in enumerate(chunks, start=1):
        prefix = "faq" if chunk.kind == "faq" else "doc"
        chunk.id = f"{prefix}-{index:04d}"
    return chunks


def fingerprint(data_dir: Path, extra_dirs: list[Path] | None = None) -> str:
    """用路径、大小、修改时间判断语料是否变化。"""
    rows: list[str] = []
    for root in _corpus_roots(Path(data_dir), extra_dirs):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or _ignored(path, root):
                continue
            stat = path.stat()
            rel = path.relative_to(root).as_posix()
            rows.append(f"{root.resolve()}/{rel}:{stat.st_size}:{stat.st_mtime_ns}")
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return digest


def _corpus_roots(data_dir: Path, extra_dirs: list[Path] | None) -> list[Path]:
    roots = [data_dir]
    data_resolved = data_dir.resolve()
    for extra in extra_dirs or []:
        if extra is None:
            continue
        extra = Path(extra)
        if not extra.is_dir():
            continue
        resolved = extra.resolve()
        if resolved == data_resolved or _is_relative_to(resolved, data_resolved):
            continue
        if any(resolved == root.resolve() for root in roots):
            continue
        roots.append(extra)
    return roots


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


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


def _load_file(path: Path, source: str, assets_dir: Path | None, ocr) -> list[Chunk]:
    suffix = path.suffix.lower()
    limit = MAX_DOC_BYTES if suffix in PDF_SUFFIXES | DOCX_SUFFIXES | DOC_SUFFIXES | IMAGE_SUFFIXES else MAX_TEXT_BYTES
    if path.stat().st_size > limit:
        logger.warning("文件超过 %s 字节，已跳过：%s", limit, path)
        return []
    if suffix in PDF_SUFFIXES | DOCX_SUFFIXES | DOC_SUFFIXES | IMAGE_SUFFIXES:
        from bidding_qa.documents import parse_path

        text, images = parse_path(path, assets_dir, ocr=ocr)
        return _chunks_from_prose(text, source, images)
    data = path.read_bytes()
    return _chunks_from_bytes(data, suffix, source, assets_dir, ocr)


def _load_zip(path: Path, source: str, assets_dir: Path | None, ocr) -> list[Chunk]:
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
            chunks.extend(_chunks_from_bytes(data, suffix, member_source, assets_dir, ocr))
    return chunks


def _chunks_from_bytes(
    data: bytes,
    suffix: str,
    source: str,
    assets_dir: Path | None = None,
    ocr=None,
) -> list[Chunk]:
    if suffix in JSON_SUFFIXES:
        text = _decode(data)
        return _chunks_from_json(text, source)
    if suffix in TEXT_SUFFIXES:
        text = _decode(data)
        faq = _parse_faq_text(text)
        if faq:
            return [_faq_chunk(q, a, source) for q, a in faq]
        return _chunks_from_prose(text, source)
    if suffix in PDF_SUFFIXES | DOCX_SUFFIXES | DOC_SUFFIXES | IMAGE_SUFFIXES:
        from bidding_qa.documents import parse_bytes

        text, images = parse_bytes(data, suffix, assets_dir, ocr=ocr, source_name=source)
        return _chunks_from_prose(text, source, images)
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


def _chunks_from_prose(text: str, source: str, images: list[str] | None = None) -> list[Chunk]:
    body = text.strip()
    if not body:
        return []
    windows = _split_text(body)
    name = source.rsplit("/", 1)[-1]
    chunks: list[Chunk] = []
    total = len(windows)
    known_images = list(images or [])
    for index, window in enumerate(windows, start=1):
        window = window.strip()
        if not window:
            continue
        heading = window.splitlines()[0][:40]
        title = f"{name} #{index}" if total > 1 else name
        if heading:
            title = f"{title} {heading}"
        found = _image_refs(window)
        chunk_images = found or (known_images if total == 1 else [])
        chunks.append(
            Chunk(
                id="",
                title=title[:80],
                text=window,
                search_text=window,
                source=source,
                kind="doc",
                images=chunk_images,
            )
        )
    return chunks


_IMAGE_REF = re.compile(r"/assets/[A-Za-z0-9._-]+")


def _image_refs(text: str) -> list[str]:
    seen: list[str] = []
    for match in _IMAGE_REF.findall(text):
        if match not in seen:
            seen.append(match)
    return seen


def _split_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    parts = [part.strip() for part in splitter.split_text(text) if part.strip()]
    return parts or [text]


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
