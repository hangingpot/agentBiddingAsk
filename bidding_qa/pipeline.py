"""问答主流程：改写 → 范围闸门 → LangChain 检索 → stuff documents 链。"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document

from bidding_qa.chain import CompleteFn, build_answer_chain, history_messages
from bidding_qa.config import Settings
from bidding_qa.dialogue import decompose, rewrite_query
from bidding_qa.gate import (
    ALLOW,
    NO_EVIDENCE_TEXT,
    REFUSE_COMPLIANCE,
    REFUSE_SCOPE,
    classify,
)
from bidding_qa.index_store import HybridIndex
from bidding_qa.llm import LLMError, LLMNotConfigured
from bidding_qa.retriever import LocalHybridRetriever, document_public
from bidding_qa.vector_store import merge_vector_hits

# 绝对融合分。无关问句的最高分应低于 MIN_CONFIDENT，改写后的平台问句应高于它。
MIN_HIT = 0.12
MIN_CONFIDENT = 0.20


@dataclass
class Answer:
    answer: str
    refused: bool
    reason: str | None = None
    sources: list[dict] = field(default_factory=list)
    rewritten_query: str = ""
    subqueries: list[str] = field(default_factory=list)


def answer_question(
    question: str,
    history: list[dict] | None,
    index: HybridIndex,
    settings: Settings,
    complete_fn: CompleteFn | None = None,
) -> Answer:
    history = history or []
    original = question.strip()
    rewritten = rewrite_query(original, history)
    gate = classify(original, rewritten)
    if gate.decision == REFUSE_COMPLIANCE:
        return Answer(
            answer=gate.answer,
            refused=True,
            reason=REFUSE_COMPLIANCE,
            rewritten_query=rewritten,
        )
    if gate.decision == REFUSE_SCOPE:
        return Answer(
            answer=gate.answer,
            refused=True,
            reason=REFUSE_SCOPE,
            rewritten_query=rewritten,
        )

    parts = decompose(rewritten)
    queries = [rewritten, *parts] if len(parts) > 1 else parts
    retriever = LocalHybridRetriever(index=index, k=settings.top_k)
    documents = merge_vector_hits(
        retriever.retrieve_many(queries),
        queries,
        index_dir=settings.index_dir,
        backend=settings.embedding_backend,
        model=settings.embedding_model,
        k=settings.top_k,
        allow_new=gate.decision == ALLOW,
    )
    confident = [doc for doc in documents if float(doc.metadata["score"]) >= MIN_CONFIDENT]
    subqueries = parts
    if not confident:
        kept = [doc for doc in documents if float(doc.metadata["score"]) >= MIN_HIT]
        if gate.decision == ALLOW:
            return Answer(
                answer=NO_EVIDENCE_TEXT,
                refused=False,
                reason="no_evidence",
                sources=[document_public(doc) for doc in kept[:3]],
                rewritten_query=rewritten,
                subqueries=subqueries,
            )
        from bidding_qa.gate import REFUSE_SCOPE_TEXT

        return Answer(
            answer=REFUSE_SCOPE_TEXT,
            refused=True,
            reason=REFUSE_SCOPE,
            rewritten_query=rewritten,
            subqueries=subqueries,
        )

    for rank, doc in enumerate(confident, start=1):
        doc.metadata["n"] = rank
    note = ""
    if rewritten and rewritten != original:
        note = f"结合上文后的完整问题：{rewritten}\n"
    chain = build_answer_chain(settings, complete_fn)
    try:
        text = chain.invoke(
            {
                "question": original,
                "context": confident,
                "chat_history": history_messages(history, settings.history_turns),
                "rewrite_note": note,
            }
        )
    except (LLMNotConfigured, LLMError):
        raise
    except Exception as exc:  # noqa: BLE001 — 把模型 SDK 的异常收成一句中文
        raise LLMError(f"调用 Kimi 失败：{exc}") from exc
    text = (text or "").strip()
    if not text:
        text = NO_EVIDENCE_TEXT
    return Answer(
        answer=text,
        refused=False,
        reason=None,
        sources=[document_public(doc) for doc in confident],
        rewritten_query=rewritten,
        subqueries=subqueries,
    )


def confident_documents(documents: list[Document]) -> list[Document]:
    return [doc for doc in documents if float(doc.metadata["score"]) >= MIN_CONFIDENT]
