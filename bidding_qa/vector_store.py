"""把同一批片段写入本地 Chroma，并在检索时与字面索引合并。"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from langchain_core.documents import Document

from bidding_qa.corpus import Chunk
from bidding_qa.embeddings import get_embeddings, normalize_backend, uses_vector_store

logger = logging.getLogger(__name__)

COLLECTION = "bidding_qa"


def chroma_dir(index_dir: Path) -> Path:
    return Path(index_dir) / "chroma"


def chroma_ready(index_dir: Path, backend: str, model: str) -> bool:
    meta_path = chroma_dir(index_dir) / "meta.json"
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    kind = normalize_backend(backend)
    return meta.get("backend") == kind and (meta.get("model") or "") == (model or "")


def build_chroma(chunks: list[Chunk], index_dir: Path, backend: str, model: str) -> bool:
    kind = normalize_backend(backend)
    if not uses_vector_store(kind):
        return False
    try:
        embeddings = get_embeddings(kind, model)
    except Exception:
        logger.exception("嵌入模型不可用，这次只保留字面索引")
        return False
    documents = [_to_document(chunk) for chunk in chunks if chunk.text.strip()]
    if not documents:
        return False
    path = chroma_dir(index_dir)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    from langchain_chroma import Chroma

    Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=str(path),
        collection_name=COLLECTION,
        collection_metadata={"hnsw:space": "cosine"},
    )
    (path / "meta.json").write_text(
        json.dumps({"backend": kind, "model": model or ""}, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Chroma 已写入 %s 条", len(documents))
    return True


def merge_vector_hits(
    documents: list[Document],
    queries: list[str],
    *,
    index_dir: Path,
    backend: str,
    model: str,
    k: int,
    allow_new: bool,
) -> list[Document]:
    store = _open(index_dir, backend, model)
    if store is None:
        return documents
    by_id = {str(doc.metadata.get("id")): doc for doc in documents}
    threshold = 0.50 if allow_new else 0.72
    for query in queries:
        try:
            pairs = store.similarity_search_with_relevance_scores(query, k=k)
        except Exception:
            logger.exception("向量检索失败，继续使用字面结果")
            return documents
        for found, relevance in pairs:
            doc_id = str(found.metadata.get("id") or "")
            if not doc_id:
                continue
            score = float(relevance)
            if doc_id in by_id:
                current = by_id[doc_id]
                current.metadata["score"] = max(float(current.metadata.get("score") or 0), score)
                continue
            if score < threshold:
                continue
            by_id[doc_id] = _from_chroma(found, score)
    merged = sorted(by_id.values(), key=lambda item: float(item.metadata.get("score") or 0), reverse=True)
    return merged[:8]


def _open(index_dir: Path, backend: str, model: str):
    path = chroma_dir(index_dir)
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    kind = normalize_backend(backend)
    if meta.get("backend") != kind or (meta.get("model") or "") != (model or ""):
        logger.warning("已有向量库的嵌入模型与当前配置不一致，忽略 Chroma。请重新 ingest。")
        return None
    try:
        embeddings = get_embeddings(kind, model)
        from langchain_chroma import Chroma

        return Chroma(
            persist_directory=str(path),
            embedding_function=embeddings,
            collection_name=COLLECTION,
        )
    except Exception:
        logger.exception("无法打开 Chroma")
        return None


def _to_document(chunk: Chunk) -> Document:
    return Document(
        page_content=chunk.text[:1800],
        metadata={
            "id": chunk.id,
            "title": chunk.title[:200],
            "source": chunk.source[:300],
            "kind": chunk.kind,
            "images": ",".join(chunk.images),
        },
    )


def _from_chroma(document: Document, score: float) -> Document:
    raw_images = document.metadata.get("images") or ""
    images = [item for item in str(raw_images).split(",") if item]
    excerpt = document.page_content.replace("\n", " ")
    if len(excerpt) > 180:
        excerpt = excerpt[:180] + "…"
    return Document(
        page_content=document.page_content,
        metadata={
            "n": 0,
            "id": document.metadata.get("id", ""),
            "title": document.metadata.get("title", ""),
            "source": document.metadata.get("source", ""),
            "score": score,
            "excerpt": excerpt,
            "images": images,
        },
    )
