"""意图范围闸门：平台知识与操作才进入检索，其余礼貌拒绝。"""

from __future__ import annotations

from dataclasses import dataclass

from bidding_qa.lexicon import (
    GREETINGS,
    MEDIUM_ANCHORS,
    NON_COMPLIANT,
    OTHER_SUBJECTS,
    PAYMENT_APPS,
    PAYMENT_CONTEXT,
    STRONG_ANCHORS,
)

ALLOW = "allow"
REFUSE_SCOPE = "out_of_scope"
REFUSE_COMPLIANCE = "non_compliant"
UNDECIDED = "undecided"

REFUSE_SCOPE_TEXT = (
    "这个问题超出了我的范围。我只回答中国石化物资招投标相关的知识和平台操作，"
    "包括易派客、电子招标投标交易平台（电招）和物资采购电子商务平台（EC）。"
    "请直接描述您在这些平台上遇到的业务或操作问题。"
)
REFUSE_COMPLIANCE_TEXT = (
    "这类请求涉及串通投标、伪造材料或攻击平台等不合规情形，我不能提供帮助。"
    "如果您要了解报名、购标、CA、保证金或开标解密等正常流程，可以直接说出具体步骤。"
)
GREETING_TEXT = (
    "您好，我是招标百问助手，可以解答易派客、电招平台和 EC 平台上的会员、投标、"
    "CA 证书、费用和支付操作。请直接提出具体问题。"
)
NO_EVIDENCE_TEXT = (
    "我在现有知识库里没有检索到足够依据，不能凭空回答。"
    "请补充是在易派客、电招平台还是 EC 平台，并写上菜单名称或报错原文。"
)


@dataclass(frozen=True)
class GateResult:
    decision: str
    answer: str = ""

    @property
    def refused(self) -> bool:
        return self.decision in {REFUSE_SCOPE, REFUSE_COMPLIANCE}


def classify(original: str, rewritten: str = "") -> GateResult:
    """先看用户原话里的越界信号，再用改写后的句子判断是不是平台问题。"""
    raw = _norm(original)
    expanded = _norm(rewritten or original)
    if _non_compliant(raw) or _non_compliant(expanded):
        return GateResult(REFUSE_COMPLIANCE, REFUSE_COMPLIANCE_TEXT)
    if _greeting(raw):
        return GateResult(REFUSE_SCOPE, GREETING_TEXT)
    # 点名了其他平台或生活话题时，仅有「招标/投标」这类宽词不能放行。
    if _foreign(raw) or _foreign(expanded):
        return GateResult(REFUSE_SCOPE, REFUSE_SCOPE_TEXT)
    if _anchored(raw) or _anchored(expanded):
        return GateResult(ALLOW)
    return GateResult(UNDECIDED)


def _norm(text: str) -> str:
    return text.strip().lower()


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word.lower() in text for word in words)


def _non_compliant(text: str) -> bool:
    return _contains_any(text, NON_COMPLIANT)


def _greeting(text: str) -> bool:
    compact = "".join(text.split())
    compact = compact.strip("!！。.?？~ ")
    return compact in {item.lower() for item in GREETINGS}


def _foreign(text: str) -> bool:
    """其他平台或无关话题。已经点名易派客/电招等专名时不在这里拦截。"""
    if _contains_any(text, STRONG_ANCHORS):
        return False
    if _contains_any(text, OTHER_SUBJECTS):
        return True
    if _contains_any(text, PAYMENT_APPS) and not _contains_any(text, PAYMENT_CONTEXT):
        return True
    return False


def _anchored(text: str) -> bool:
    if _contains_any(text, STRONG_ANCHORS) or _contains_any(text, MEDIUM_ANCHORS):
        return True
    if _contains_any(text, PAYMENT_APPS) and _contains_any(text, PAYMENT_CONTEXT):
        return True
    return False
