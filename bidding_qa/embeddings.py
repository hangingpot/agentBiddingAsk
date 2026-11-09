"""本地嵌入。默认 FastEmbed 的 BGE-small-zh，不访问 Kimi。"""

from __future__ import annotations

import hashlib
import logging
import re

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
_CACHE: dict[tuple[str, str], Embeddings] = {}


def normalize_backend(name: str) -> str:
    value = (name or "fastembed").strip().lower()
    if value == "local":
        return "fastembed"
    return value


def uses_vector_store(backend: str) -> bool:
    return normalize_backend(backend) in {"fastembed", "huggingface", "hash"}


class HashEmbeddings(Embeddings):
    """字符二元组哈希向量。测试和不想下载模型时使用，同一进程内外都稳定。"""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        compact = re.sub(r"\s+", "", text.lower())
        if len(compact) < 2:
            compact = compact or "空"
        for index in range(len(compact) - 1):
            digest = hashlib.sha1(compact[index : index + 2].encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:2], "little") % self.dim
            vector[bucket] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]


def get_embeddings(backend: str, model: str = "") -> Embeddings:
    kind = normalize_backend(backend)
    model_name = (model or DEFAULT_MODEL).strip()
    key = (kind, model_name)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    if kind == "hash":
        embedder: Embeddings = HashEmbeddings()
    elif kind == "huggingface":
        embedder = _huggingface(model_name)
    elif kind == "fastembed":
        embedder = _fastembed(model_name)
    else:
        raise ValueError(f"不支持向量嵌入后端：{backend}")
    _CACHE[key] = embedder
    return embedder


def _fastembed(model_name: str) -> Embeddings:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise RuntimeError("未安装 fastembed。请执行 pip install fastembed") from exc

    class _Wrapper(Embeddings):
        def __init__(self):
            self._model = TextEmbedding(model_name=model_name)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [vector.tolist() for vector in self._model.embed(texts)]

        def embed_query(self, text: str) -> list[float]:
            return next(self._model.embed([text])).tolist()

    logger.info("加载本地嵌入模型 %s", model_name)
    return _Wrapper()


def _huggingface(model_name: str) -> Embeddings:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "未安装 sentence-transformers。请执行 pip install sentence-transformers langchain-huggingface"
        ) from exc
    logger.info("加载 HuggingFace 嵌入模型 %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
