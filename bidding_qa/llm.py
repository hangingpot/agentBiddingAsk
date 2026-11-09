"""Kimi 对话模型：LangChain ChatOpenAI，指向 Moonshot 的 OpenAI 兼容接口。"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from bidding_qa.config import Settings


class LLMNotConfigured(RuntimeError):
    """未配置 Kimi 密钥。"""


class LLMError(RuntimeError):
    """模型调用失败。"""


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """构造只走 Chat Completions 的 Kimi 客户端。

    K3 不接受 temperature。reasoning_effort 只对 kimi-k3 系列附带。
    """
    if not settings.llm_configured:
        raise LLMNotConfigured(
            "未配置 Kimi 密钥。请复制 .env.example 为 .env，并填写 KIMI_API_KEY（或 MOONSHOT_API_KEY）。"
        )
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        "temperature": None,
        "max_tokens": settings.max_completion_tokens,
        "use_responses_api": False,
        "max_retries": 1,
    }
    if settings.model.lower().startswith("kimi-k3") and settings.reasoning_effort:
        kwargs["reasoning_effort"] = settings.reasoning_effort
    return ChatOpenAI(**kwargs)


def message_text(message: Any) -> str:
    """从 AIMessage 或普通字符串里取出给用户看的正文。"""
    if message is None:
        return ""
    if isinstance(message, str):
        return message.strip()
    content = getattr(message, "content", None)
    text = _content_to_text(content)
    if text:
        return text
    extra = getattr(message, "additional_kwargs", None) or {}
    if isinstance(extra, dict):
        return _content_to_text(extra.get("reasoning_content"))
    return ""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif hasattr(item, "text") and item.text:
                parts.append(str(item.text))
        return "".join(parts).strip()
    return ""
