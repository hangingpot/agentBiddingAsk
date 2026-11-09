"""问答主流程：改写 → 范围闸门 → 多路检索 → 依据生成。"""

from __future__ import annotations

from dataclasses import dataclass, field

from bidding_qa.config import Settings
from bidding_qa.dialogue import decompose, recent_history, rewrite_query
from bidding_qa.gate import (
    ALLOW,
    NO_EVIDENCE_TEXT,
    REFUSE_COMPLIANCE,
    REFUSE_SCOPE,
    UNDECIDED,
    classify,
)
from bidding_qa.index_store import Hit, HybridIndex
from bidding_qa.llm import CompleteFn, LLMError, LLMNotConfigured, complete

# 绝对融合分。无关问句的最高分应低于 MIN_CONFIDENT，改写后的平台问句应高于它。
MIN_HIT = 0.12
MIN_CONFIDENT = 0.20

SYSTEM_PROMPT = """你是「招标百问」助手，只回答中国石化物资招投标相关问题。范围限于：
- 易派客（www.epec.com）
- 中国石化电子招标投标交易平台，也叫电招平台（bidding.sinopec.com）
- 中国石化物资采购电子商务平台，也叫 EC 平台（ec.sinopec.com）
上的业务知识和平台操作。

回答规则：
1. 只使用下面给出的参考资料。资料没有写的步骤、金额、时限、电话和菜单，不要补充，直接说明知识库未覆盖。
2. 复合问题按小问分点回答，每个小问都要能在资料中找到依据。
3. 在使用了某条资料的句子末尾标注【资料n】，n 是资料编号。
4. 用简洁中文，保留按钮名称、菜单路径、金额和时间。
5. 多轮对话里的「那」「这个」指代，按已给出的历史理解，仍然只能依据资料作答。
6. 不要输出思考过程。"""


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
    hits = index.search_many(queries, k=settings.top_k)
    kept = [hit for hit in hits if hit.score >= MIN_HIT]
    confident = [hit for hit in kept if hit.score >= MIN_CONFIDENT]
    subqueries = parts
    if not confident:
        if gate.decision == ALLOW:
            return Answer(
                answer=NO_EVIDENCE_TEXT,
                refused=False,
                reason="no_evidence",
                sources=[hit.public() for hit in kept[:3]],
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

    caller = complete_fn or (lambda messages: complete(messages, settings))
    messages = _messages(original, rewritten, history, confident, settings)
    try:
        text = caller(messages).strip()
    except (LLMNotConfigured, LLMError):
        raise
    if not text:
        text = NO_EVIDENCE_TEXT
    return Answer(
        answer=text,
        refused=False,
        reason=None,
        sources=[hit.public() for hit in confident],
        rewritten_query=rewritten,
        subqueries=subqueries,
    )


def _messages(
    question: str,
    rewritten: str,
    history: list[dict],
    hits: list[Hit],
    settings: Settings,
) -> list[dict]:
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        body = hit.chunk.text.strip()
        if len(body) > 1800:
            body = body[:1800] + "…"
        blocks.append(f"【资料{index}】来源：{hit.chunk.source}｜{hit.chunk.title}\n{body}")
    context = "\n\n".join(blocks)
    user = f"参考资料：\n{context}\n\n用户原话：{question}"
    if rewritten and rewritten != question:
        user += f"\n结合上文后的完整问题：{rewritten}"
    user += "\n请只依据参考资料回答用户原话。"
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(recent_history(history, settings.history_turns))
    messages.append({"role": "user", "content": user})
    return messages
