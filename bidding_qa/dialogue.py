"""多轮指代改写，以及复合问题拆分。规则可单测，不依赖模型。"""

from __future__ import annotations

import re

from bidding_qa.lexicon import MEDIUM_ANCHORS, STRONG_ANCHORS

_FOLLOWUP_PREFIX = re.compile(
    r"^(那|那么|这|这个|这些|其中|继续|还有|然后|另外|它|该|上面|前述|此)"
)
_SELF_CONTAINED = re.compile(r"如何|怎么|怎样|为什么|为何")


def is_followup(question: str, history: list[dict]) -> bool:
    if not any(item.get("role") == "user" and str(item.get("content", "")).strip() for item in history):
        return False
    return _followup_shape(question)


def _followup_shape(question: str) -> bool:
    text = question.strip()
    if not text:
        return False
    if _FOLLOWUP_PREFIX.match(text):
        return True
    if text.endswith(("呢", "呢？", "呢?")) and len(text) <= 20 and not _SELF_CONTAINED.search(text):
        return True
    if len(text) <= 8 and not _SELF_CONTAINED.search(text):
        return True
    return False


def rewrite_query(question: str, history: list[dict]) -> str:
    """把「那费用呢」补成带上一问主题的独立检索句。"""
    text = question.strip()
    if not is_followup(text, history):
        return text
    topic = _previous_topic(history)
    if not topic:
        return text
    tail = _FOLLOWUP_PREFIX.sub("", text).strip()
    return f"{topic} {tail or text}".strip()


def _previous_topic(history: list[dict]) -> str:
    from bidding_qa.gate import ALLOW, UNDECIDED, classify

    users = [str(item.get("content", "")).strip() for item in history if item.get("role") == "user"]
    users = [item for item in users if item]
    for content in reversed(users):
        if _followup_shape(content):
            continue
        decision = classify(content).decision
        if decision in {ALLOW, UNDECIDED}:
            return content
    return ""


def decompose(question: str) -> list[str]:
    """一个问句里有多处问题时拆开，并给后半句补上主题词。"""
    text = question.strip()
    if not text:
        return []
    parts = _split(text)
    if len(parts) <= 1:
        return [text]
    anchors = _anchors(text)
    enriched: list[str] = []
    for index, part in enumerate(parts):
        if index and anchors and not any(anchor in part.lower() for anchor in anchors):
            enriched.append(f"{anchors[0]} {part}".strip())
        else:
            enriched.append(part)
    # 去重并保持顺序
    seen: set[str] = set()
    result: list[str] = []
    for part in enriched:
        key = re.sub(r"\s+", "", part)
        if key in seen:
            continue
        seen.add(key)
        result.append(part)
    return result or [text]


def _split(text: str) -> list[str]:
    pieces = [part.strip() for part in re.split(r"(?<=[？?])", text) if part.strip()]
    refined: list[str] = []
    for piece in pieces:
        bits = re.split(r"[；;]|(?:^|[，,])\s*(?:另外|此外|同时|还有|以及)\s*", piece)
        for bit in bits:
            cleaned = bit.strip(" ，,、;；")
            if len(cleaned) >= 4:
                refined.append(cleaned)
    if len(refined) <= 1:
        comma_parts = [part.strip() for part in re.split(r"[，,]", text) if part.strip()]
        if len(comma_parts) == 2 and all(_looks_like_ask(part) for part in comma_parts):
            return comma_parts
        return [text] if not refined else refined
    return refined


def _looks_like_ask(text: str) -> bool:
    cues = ("如何", "怎么", "怎样", "吗", "呢", "多少", "是否", "哪里", "哪些", "什么", "为何", "为什么", "多久")
    return len(text) >= 6 and any(cue in text for cue in cues)


def _anchors(text: str) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for anchor in STRONG_ANCHORS + MEDIUM_ANCHORS:
        if anchor.lower() in lowered:
            found.append(anchor)
    # 较长的专名优先，避免只用「招标」这种宽词去补后半句。
    found.sort(key=len, reverse=True)
    return found


def recent_history(history: list[dict], turns: int) -> list[dict]:
    """保留最近若干轮，并截断过长的单条内容。"""
    cleaned: list[dict] = []
    for item in history:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        cleaned.append({"role": role, "content": content[:1500]})
    keep = max(1, turns) * 2
    return cleaned[-keep:]
