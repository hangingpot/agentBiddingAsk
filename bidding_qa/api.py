"""本地 HTTP 接口和单页演示。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bidding_qa.config import Settings, load_settings
from bidding_qa.index_store import HybridIndex, build_index
from bidding_qa.llm import LLMError, LLMNotConfigured
from bidding_qa.pipeline import Answer, answer_question
from bidding_qa.retriever import LocalHybridRetriever, document_public

STATIC_DIR = Path(__file__).resolve().parent / "static"


class HistoryMessage(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[HistoryMessage] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=5, ge=1, le=10)


def create_app(
    settings: Settings | None = None,
    index: HybridIndex | None = None,
    complete_fn=None,
) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="招标百问", version="1.0.0")
    holder: dict = {"settings": settings, "index": index, "complete_fn": complete_fn}
    assets = settings.index_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    def get_index() -> HybridIndex:
        if holder["index"] is None:
            holder["index"] = build_index(
                settings.data_dir,
                settings.index_dir,
                manuals_dir=settings.manuals_dir,
                embedding_backend=settings.embedding_backend,
                embedding_model=settings.embedding_model,
            )
        return holder["index"]

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/health")
    def health() -> dict:
        try:
            current = get_index()
            chunks = len(current.chunks)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "detail": str(exc),
                "llm_configured": settings.llm_configured,
                "model": settings.model,
                "embedding_backend": settings.embedding_backend,
            }
        return {
            "status": "ok",
            "chunks": chunks,
            "llm_configured": settings.llm_configured,
            "model": settings.model,
            "base_url": settings.base_url,
            "embedding_backend": settings.embedding_backend,
            "embedding_model": settings.embedding_model,
            "manuals_dir": str(settings.manuals_dir or ""),
        }

    @app.post("/api/retrieve")
    def retrieve(body: RetrieveRequest) -> dict:
        retriever = LocalHybridRetriever(index=get_index(), k=body.k)
        documents = retriever.invoke(body.query)
        return {"query": body.query, "hits": [document_public(doc) for doc in documents]}

    @app.post("/api/chat")
    def chat(body: ChatRequest) -> dict:
        history = [{"role": item.role, "content": item.content} for item in body.history]
        history = [item for item in history if item["role"] in {"user", "assistant"}]
        try:
            result = answer_question(
                body.message,
                history,
                get_index(),
                settings,
                complete_fn=holder["complete_fn"],
            )
        except LLMNotConfigured as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return _payload(result)

    return app


def _payload(result: Answer) -> dict:
    return {
        "answer": result.answer,
        "refused": result.refused,
        "reason": result.reason,
        "sources": result.sources,
        "rewritten_query": result.rewritten_query,
        "subqueries": result.subqueries,
    }


@lru_cache(maxsize=1)
def _cached_app() -> FastAPI:
    return create_app()


app = _cached_app()
