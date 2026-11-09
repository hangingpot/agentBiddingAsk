"""离线混合索引：字符 n-gram TF-IDF（绝对分）+ jieba BM25（相对分）。

不下载模型，也不调用向量云接口。绝对分用 TF-IDF 余弦，避免无关问题被 min-max 归一化抬到 1。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from bidding_qa.corpus import Chunk, fingerprint, load_corpus
from bidding_qa.lexicon import DOMAIN_WORDS, STOPWORDS

logger = logging.getLogger(__name__)

_PUNCT = set(" \t\r\n，。！？、；：""''（）()【】[]《》<>—…·,.!?;:'`~|/\\@#$%^&*+-=＿")
_jieba_ready = False


def _prepare_jieba() -> None:
    global _jieba_ready
    if _jieba_ready:
        return
    import jieba

    jieba.setLogLevel(logging.ERROR)
    for word in DOMAIN_WORDS:
        jieba.add_word(word)
    _jieba_ready = True


def tokenize(text: str) -> list[str]:
    _prepare_jieba()
    import jieba

    tokens: list[str] = []
    for raw in jieba.lcut(text.lower(), cut_all=False):
        token = raw.strip().lower()
        if not token or token in STOPWORDS:
            continue
        if all(char in _PUNCT or char.isspace() for char in token):
            continue
        if len(token) == 1 and not ("\u4e00" <= token <= "\u9fff" or token.isalnum()):
            continue
        if len(token) == 1 and token.isascii() and not token.isdigit():
            continue
        tokens.append(token)
    if tokens:
        return tokens
    return [char for char in text.lower() if char.isalnum()]


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def expand_query(query: str) -> str:
    """把口头说法补成资料里的写法。半年在百问榜里写成 6 个月。"""
    extras: list[str] = []
    if "半年" in query:
        extras.append("6个月")
    if "多少钱" in query:
        extras.append("费用")
    lowered = query.lower()
    if "pin" in lowered and "pin码" not in lowered and "PIN码" not in query:
        extras.append("PIN码")
    if not extras:
        return query
    return f"{query} {' '.join(extras)}"


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    tfidf: float
    bm25: float
    title_score: float

    def public(self) -> dict:
        excerpt = self.chunk.text.replace("\n", " ")
        if len(excerpt) > 180:
            excerpt = excerpt[:180] + "…"
        return {
            "id": self.chunk.id,
            "title": self.chunk.title,
            "source": self.chunk.source,
            "score": round(float(self.score), 4),
            "excerpt": excerpt,
        }


class HybridIndex:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("索引为空")
        self.chunks = list(chunks)
        self._titles = [_bigrams(chunk.question or chunk.title) for chunk in self.chunks]
        self._tokenized = [tokenize(chunk.search_text) for chunk in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized)
        self._vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True,
            norm=None,
        )
        matrix = self._vectorizer.fit_transform(chunk.search_text for chunk in self.chunks)
        self._matrix = normalize(matrix, norm="l2", copy=False)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        query = expand_query(query.strip())
        if not query:
            return []
        k = max(1, min(k, len(self.chunks)))
        query_vec = normalize(self._vectorizer.transform([query]), norm="l2")
        tfidf = np.asarray(query_vec.dot(self._matrix.T).todense()).ravel()
        bm25_raw = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)
        peak = float(bm25_raw.max()) if len(bm25_raw) else 0.0
        bm25 = bm25_raw / peak if peak > 1e-9 else np.zeros_like(bm25_raw)
        query_grams = _bigrams(query)
        title_scores = np.array([_jaccard(query_grams, grams) for grams in self._titles], dtype=float)
        # BM25 只在该片段已经有一点字面重合时参与，避免无关问题被相对分抬高。
        gate = np.clip(tfidf / 0.08, 0.0, 1.0)
        fused = 0.62 * tfidf + 0.23 * (bm25 * gate) + 0.15 * title_scores
        order = np.argsort(-fused)[:k]
        hits: list[Hit] = []
        for index in order:
            hits.append(
                Hit(
                    chunk=self.chunks[int(index)],
                    score=float(fused[index]),
                    tfidf=float(tfidf[index]),
                    bm25=float(bm25[index]),
                    title_score=float(title_scores[index]),
                )
            )
        return hits

    def search_many(self, queries: list[str], k: int = 4, limit: int | None = None) -> list[Hit]:
        unique: list[str] = []
        seen: set[str] = set()
        for query in queries:
            text = query.strip()
            key = re.sub(r"\s+", "", text)
            if text and key not in seen:
                seen.add(key)
                unique.append(text)
        if not unique:
            return []
        best: dict[str, Hit] = {}
        per_query_best: list[Hit] = []
        for query in unique:
            hits = self.search(query, k=k)
            if hits:
                per_query_best.append(hits[0])
            for hit in hits:
                previous = best.get(hit.chunk.id)
                if previous is None or hit.score > previous.score:
                    best[hit.chunk.id] = hit
        if limit is None:
            limit = k if len(unique) == 1 else min(8, 3 * len(unique))
        ranked = sorted(best.values(), key=lambda item: item.score, reverse=True)
        if not ranked:
            return []
        selected = ranked[:limit]
        selected_ids = {hit.chunk.id for hit in selected}
        for hit in per_query_best:
            if hit.chunk.id not in selected_ids and hit.score >= ranked[0].score * 0.45:
                selected.append(hit)
                selected_ids.add(hit.chunk.id)
        selected.sort(key=lambda item: item.score, reverse=True)
        return selected[:8]


def save_chunks(chunks: list[Chunk], index_dir: Path, data_fingerprint: str) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = index_dir / "chunks.jsonl"
    with chunk_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_json(), ensure_ascii=False) + "\n")
    manifest = {
        "fingerprint": data_fingerprint,
        "chunk_count": len(chunks),
        "embedding_backend": "local-hybrid-tfidf-bm25",
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_chunks(index_dir: Path) -> tuple[list[Chunk], str]:
    chunk_path = index_dir / "chunks.jsonl"
    manifest_path = index_dir / "manifest.json"
    if not chunk_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("索引不存在，请先执行 python -m bidding_qa ingest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    with chunk_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(Chunk.from_json(json.loads(line)))
    return chunks, str(manifest.get("fingerprint", ""))


def build_index(data_dir: Path, index_dir: Path, force: bool = False) -> HybridIndex:
    """语料有变化时重建 chunks.jsonl，并在内存中拟合向量。"""
    current = fingerprint(data_dir)
    if not force:
        try:
            chunks, saved = load_chunks(index_dir)
            if saved == current and chunks:
                logger.info("使用已有索引，%s 条", len(chunks))
                return HybridIndex(chunks)
        except FileNotFoundError:
            pass
    chunks = load_corpus(data_dir)
    save_chunks(chunks, index_dir, current)
    logger.info("已索引 %s 条片段", len(chunks))
    return HybridIndex(chunks)
