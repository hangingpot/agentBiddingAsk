"""把本地混合索引包成 LangChain Retriever。不调用任何嵌入接口。"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from bidding_qa.index_store import Hit, HybridIndex


class LocalHybridRetriever(BaseRetriever):
    """单问走 `invoke`，复合问题走 `retrieve_many`。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: HybridIndex
    k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        return hits_to_documents(self.index.search(query, k=self.k))

    def retrieve_many(self, queries: list[str]) -> list[Document]:
        return hits_to_documents(self.index.search_many(queries, k=self.k))


def hits_to_documents(hits: list[Hit]) -> list[Document]:
    documents: list[Document] = []
    for rank, hit in enumerate(hits, start=1):
        text = hit.chunk.text.strip()
        if len(text) > 1800:
            text = text[:1800] + "…"
        public = hit.public()
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "n": rank,
                    "id": hit.chunk.id,
                    "title": hit.chunk.title,
                    "source": hit.chunk.source,
                    "score": float(hit.score),
                    "excerpt": public["excerpt"],
                    "images": list(hit.chunk.images),
                },
            )
        )
    return documents


def document_public(document: Document) -> dict[str, Any]:
    images = document.metadata.get("images") or []
    if isinstance(images, str):
        images = [item for item in images.split(",") if item]
    return {
        "id": document.metadata["id"],
        "title": document.metadata["title"],
        "source": document.metadata["source"],
        "score": round(float(document.metadata["score"]), 4),
        "excerpt": document.metadata["excerpt"],
        "images": images,
    }
