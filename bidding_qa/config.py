"""从环境变量读取运行配置。密钥只出现在进程环境或本地 .env 中。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()


def _path(name: str, default: str) -> Path:
    raw = os.getenv(name, default).strip() or default
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    reasoning_effort: str
    max_completion_tokens: int
    embedding_backend: str
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    data_dir: Path = PROJECT_ROOT / "data"
    manuals_dir: Path | None = None
    index_dir: Path = PROJECT_ROOT / "storage" / "index"
    top_k: int = 5
    history_turns: int = 6
    ocr_backend: str = "rapidocr"
    vision_caption: str = "off"
    parser_backend: str = "builtin"

    @property
    def llm_configured(self) -> bool:
        return bool(self.api_key)


def load_settings() -> Settings:
    _load_env()
    key = os.getenv("KIMI_API_KEY", "").strip() or os.getenv("MOONSHOT_API_KEY", "").strip()
    effort = os.getenv("KIMI_REASONING_EFFORT", "low").strip().lower() or "low"
    if effort not in {"low", "high", "max"}:
        effort = "low"
    try:
        max_tokens = int(os.getenv("KIMI_MAX_COMPLETION_TOKENS", "2048"))
    except ValueError:
        max_tokens = 2048
    try:
        top_k = max(1, int(os.getenv("TOP_K", "5")))
    except ValueError:
        top_k = 5
    try:
        history_turns = max(1, int(os.getenv("HISTORY_TURNS", "6")))
    except ValueError:
        history_turns = 6
    base_url = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1").strip()
    base_url = base_url.rstrip("/")
    return Settings(
        api_key=key,
        base_url=base_url or "https://api.moonshot.cn/v1",
        model=os.getenv("KIMI_MODEL", "kimi-k3.1-pro-highspeed").strip()
        or "kimi-k3.1-pro-highspeed",
        reasoning_effort=effort,
        max_completion_tokens=max(256, min(max_tokens, 8192)),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "fastembed").strip().lower() or "fastembed",
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5").strip()
        or "BAAI/bge-small-zh-v1.5",
        data_dir=_path("DATA_DIR", "data"),
        manuals_dir=_manuals_dir(),
        index_dir=_path("INDEX_DIR", "storage/index"),
        top_k=top_k,
        history_turns=history_turns,
        ocr_backend=os.getenv("OCR_BACKEND", "rapidocr").strip().lower() or "rapidocr",
        vision_caption=os.getenv("VISION_CAPTION", "off").strip().lower() or "off",
        parser_backend=os.getenv("PARSER_BACKEND", "builtin").strip().lower() or "builtin",
    )


def _manuals_dir() -> Path | None:
    raw = os.getenv("MANUALS_DIR", "").strip()
    if raw:
        return _path("MANUALS_DIR", raw)
    for candidate in (PROJECT_ROOT / "data" / "manuals", PROJECT_ROOT / "doc" / "6.最新操作手册"):
        if candidate.is_dir():
            return candidate
    return None
