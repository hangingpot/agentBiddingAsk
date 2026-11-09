"""从 docx、pdf、旧版 doc 和图片里抽出正文、表格和截图说明。

默认解析器不依赖 Docling。设置 PARSER_BACKEND=docling 且已安装 docling 时，先用它导出 Markdown。
截图文字默认走 RapidOCR；OCR_BACKEND=paddle 时改用 PaddleOCR。两者都未安装时只保留图片路径。
VISION_CAPTION=kimi 时，再用当前 Kimi 视觉模型给截图写一句说明，这一步才需要密钥。
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_OCR_ENGINE = None
_CAPTION_LEFT: int | None = None
_OCR_LEFT: int | None = None


def parse_path(path: Path, assets_dir: Path | None, ocr=None) -> tuple[str, list[str]]:
    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix == ".doc":
        converted = _doc_to_docx_bytes(path)
        if converted is None:
            return "", []
        data, suffix = converted, ".docx"
    return parse_bytes(data, suffix, assets_dir, ocr=ocr, source_name=path.name)


def parse_bytes(
    data: bytes,
    suffix: str,
    assets_dir: Path | None,
    ocr=None,
    source_name: str = "",
) -> tuple[str, list[str]]:
    suffix = suffix.lower()
    docling_text = _docling_markdown(data, suffix, source_name) if suffix in {".docx", ".pdf", ".doc"} else ""
    if suffix == ".docx":
        body, images = _parse_docx(data, assets_dir, ocr)
    elif suffix == ".pdf":
        body, images = _parse_pdf(data, assets_dir, ocr)
    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
        body, images = _parse_image(data, assets_dir, ocr, source_name or "image")
    else:
        body, images = "", []
    if docling_text and len(docling_text) > len(body):
        body = docling_text + ("\n\n" + body if images else "")
    return body.strip(), images


def _parse_docx(data: bytes, assets_dir: Path | None, ocr) -> tuple[str, list[str]]:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(io.BytesIO(data))
    parts: list[str] = []
    images: list[str] = []
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            notes, found = _images_from_element(child, document, assets_dir, ocr)
            images.extend(found)
            if text:
                parts.append(text)
            parts.extend(notes)
        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            rows = []
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                parts.append("\n".join(rows))
            notes, found = _images_from_element(child, document, assets_dir, ocr)
            images.extend(found)
            parts.extend(notes)
    return "\n\n".join(part for part in parts if part), _unique(images)


def _parse_pdf(data: bytes, assets_dir: Path | None, ocr) -> tuple[str, list[str]]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    images: list[str] = []
    for index, page in enumerate(reader.pages[:400], start=1):
        text = (page.extract_text() or "").strip()
        if text:
            parts.append(f"第{index}页\n{text}")
        page_images = getattr(page, "images", None) or []
        for image in page_images:
            blob = getattr(image, "data", None)
            if not blob:
                continue
            note, url = _store_image(blob, assets_dir, ocr, getattr(image, "name", "page"))
            if url:
                images.append(url)
            if note:
                parts.append(note)
    return "\n\n".join(parts), _unique(images)


def _parse_image(data: bytes, assets_dir: Path | None, ocr, name: str) -> tuple[str, list[str]]:
    note, url = _store_image(data, assets_dir, ocr, name)
    if not url and not note:
        return "", []
    return note or f"【截图】{url}", _unique([url] if url else [])


def _images_from_element(element, document, assets_dir, ocr) -> tuple[list[str], list[str]]:
    from docx.oxml.ns import qn

    notes: list[str] = []
    urls: list[str] = []
    for blip in element.xpath(".//a:blip"):
        embed = blip.get(qn("r:embed"))
        if not embed:
            continue
        part = document.part.related_parts.get(embed)
        blob = getattr(part, "blob", None) if part is not None else None
        if not blob:
            continue
        note, url = _store_image(blob, assets_dir, ocr, embed)
        if url:
            urls.append(url)
        if note:
            notes.append(note)
    return notes, urls


def _store_image(blob: bytes, assets_dir: Path | None, ocr, hint: str) -> tuple[str, str]:
    if len(blob) < 400 or _too_small(blob):
        return "", ""
    if assets_dir is None:
        return "", ""
    assets_dir.mkdir(parents=True, exist_ok=True)
    ext = _image_ext(blob)
    name = hashlib.sha1(blob).hexdigest()[:16] + ext
    path = assets_dir / name
    if not path.exists():
        path.write_bytes(blob)
    url = f"/assets/{name}"
    lines = [f"【截图】{url}"]
    ocr_text = _run_ocr(path, ocr)
    if ocr_text:
        lines.append(f"截图文字：{ocr_text}")
    caption = _run_caption(path)
    if caption:
        lines.append(f"截图说明：{caption}")
    return "\n".join(lines), url


def _too_small(blob: bytes) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False
    try:
        with Image.open(io.BytesIO(blob)) as image:
            width, height = image.size
    except Exception:
        return False
    return width < 80 or height < 80


def _image_ext(blob: bytes) -> str:
    if blob.startswith(b"\xff\xd8"):
        return ".jpg"
    if blob.startswith(b"GIF"):
        return ".gif"
    if blob.startswith(b"RIFF") and b"WEBP" in blob[:16]:
        return ".webp"
    return ".png"


def _run_ocr(path: Path, ocr) -> str:
    global _OCR_LEFT
    if _OCR_LEFT is None:
        _OCR_LEFT = _int_env("OCR_MAX_IMAGES", 800)
    if _OCR_LEFT <= 0:
        return ""
    runner = ocr or default_ocr
    try:
        text = runner(path) or ""
    except Exception:
        logger.exception("OCR 失败 %s", path)
        return ""
    text = " ".join(text.split())
    if text:
        _OCR_LEFT -= 1
    return text[:800]


def default_ocr(path: Path) -> str:
    backend = os.getenv("OCR_BACKEND", "rapidocr").strip().lower()
    if backend in {"", "none", "off"}:
        return ""
    if backend == "paddle":
        return _paddle_ocr(path)
    return _rapid_ocr(path)


def _rapid_ocr(path: Path) -> str:
    global _OCR_ENGINE
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        logger.info("未安装 rapidocr-onnxruntime，截图不做 OCR。可执行 pip install rapidocr-onnxruntime")
        return ""
    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()
    result, _elapsed = _OCR_ENGINE(str(path))
    if not result:
        return ""
    return "\n".join(str(item[1]) for item in result if len(item) > 1 and item[1])


def _paddle_ocr(path: Path) -> str:
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.info("未安装 paddleocr。可执行 pip install paddlepaddle paddleocr，或改用 OCR_BACKEND=rapidocr")
        return ""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        _OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    result = _OCR_ENGINE.ocr(str(path), cls=True) or []
    lines: list[str] = []
    for block in result:
        for item in block or []:
            if len(item) > 1 and item[1]:
                lines.append(str(item[1][0]))
    return "\n".join(lines)


def _run_caption(path: Path) -> str:
    global _CAPTION_LEFT
    mode = os.getenv("VISION_CAPTION", "off").strip().lower()
    if mode not in {"kimi", "on", "true"}:
        return ""
    if _CAPTION_LEFT is None:
        _CAPTION_LEFT = _int_env("VISION_MAX_IMAGES", 30)
    if _CAPTION_LEFT <= 0:
        return ""
    try:
        text = _kimi_caption(path)
    except Exception:
        logger.exception("视觉说明失败 %s", path)
        return ""
    if text:
        _CAPTION_LEFT -= 1
    return text[:400]


def _kimi_caption(path: Path) -> str:
    from langchain_core.messages import HumanMessage

    from bidding_qa.config import load_settings
    from bidding_qa.llm import build_chat_model, message_text

    settings = load_settings()
    if not settings.llm_configured:
        return ""
    model_name = os.getenv("VISION_MODEL", "").strip() or settings.model
    if model_name != settings.model:
        from dataclasses import replace

        settings = replace(settings, model=model_name)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower().lstrip(".") or "png"
    if suffix == "jpg":
        suffix = "jpeg"
    model = build_chat_model(settings)
    message = model.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "这是中石化易派客或电招平台操作截图。用一两句中文写出界面上的菜单、按钮、报错和关键数字。不要编造看不见的内容。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{suffix};base64,{encoded}"},
                    },
                ]
            )
        ]
    )
    return message_text(message)


def _docling_markdown(data: bytes, suffix: str, source_name: str) -> str:
    if os.getenv("PARSER_BACKEND", "builtin").strip().lower() != "docling":
        return ""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        logger.info("PARSER_BACKEND=docling 但未安装 docling，改用内置解析。pip install docling")
        return ""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / (source_name or f"manual{suffix}")
        path.write_bytes(data)
        converted = DocumentConverter().convert(str(path))
        document = getattr(converted, "document", None)
        if document is None:
            return ""
        export = getattr(document, "export_to_markdown", None)
        return export() if export else ""


def _doc_to_docx_bytes(path: Path) -> bytes | None:
    office = shutil.which("soffice") or shutil.which("libreoffice")
    if not office:
        logger.warning("跳过 %s：旧版 .doc 需要本机的 LibreOffice（soffice）先转成 docx。", path)
        return None
    with tempfile.TemporaryDirectory() as tmp:
        completed = subprocess.run(
            [office, "--headless", "--convert-to", "docx", "--outdir", tmp, str(path)],
            check=False,
            capture_output=True,
            timeout=180,
        )
        target = Path(tmp) / f"{path.stem}.docx"
        if completed.returncode != 0 or not target.is_file():
            logger.warning("LibreOffice 未能转换 %s", path)
            return None
        return target.read_bytes()


def _unique(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return seen


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default
