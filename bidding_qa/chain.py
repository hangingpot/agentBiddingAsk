"""LangChain stuff-documents 链：把检索片段和历史交给 Kimi。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableLambda

from bidding_qa.config import Settings
from bidding_qa.dialogue import recent_history
from bidding_qa.llm import LLMError, build_chat_model, message_text

CompleteFn = Callable[[Sequence[BaseMessage]], str]

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


class AnswerTextParser(StrOutputParser):
    """取出正文。Kimi 若只返回思考字段，也把它收成可见回答。"""

    def parse_result(self, result: list, *, partial: bool = False) -> str:
        del partial
        generation = result[0]
        message = getattr(generation, "message", None)
        text = message_text(message) if message is not None else ""
        if not text:
            text = (getattr(generation, "text", None) or "").strip()
        if not text:
            raise LLMError("Kimi 返回了空内容。")
        return text


def build_answer_chain(settings: Settings, complete_fn: CompleteFn | None = None) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            (
                "human",
                "参考资料：\n{context}\n\n用户原话：{question}\n{rewrite_note}请只依据参考资料回答用户原话。",
            ),
        ]
    )
    document_prompt = PromptTemplate.from_template(
        "【资料{n}】来源：{source}｜{title}\n{page_content}"
    )
    model: Runnable = (
        _scripted_model(complete_fn) if complete_fn is not None else build_chat_model(settings)
    )
    return create_stuff_documents_chain(
        model,
        prompt,
        output_parser=AnswerTextParser(),
        document_prompt=document_prompt,
    )


def history_messages(history: list[dict], turns: int) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in recent_history(history, turns):
        content = item["content"]
        if item["role"] == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


def _scripted_model(complete_fn: CompleteFn) -> Runnable:
    def run(prompt_value: PromptValue | Sequence[BaseMessage]) -> AIMessage:
        if isinstance(prompt_value, PromptValue):
            messages = prompt_value.to_messages()
        else:
            messages = list(prompt_value)
        return AIMessage(content=complete_fn(messages))

    return RunnableLambda(run)
