"""Kimi（Moonshot）OpenAI 兼容 Chat Completions。"""

from __future__ import annotations

from typing import Callable

from openai import OpenAI

from bidding_qa.config import Settings

CompleteFn = Callable[[list[dict]], str]


class LLMNotConfigured(RuntimeError):
    """未配置 Kimi 密钥。"""


class LLMError(RuntimeError):
    """模型调用失败。"""


def complete(messages: list[dict], settings: Settings) -> str:
    if not settings.llm_configured:
        raise LLMNotConfigured(
            "未配置 Kimi 密钥。请复制 .env.example 为 .env，并填写 KIMI_API_KEY（或 MOONSHOT_API_KEY）。"
        )
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    kwargs: dict = {
        "model": settings.model,
        "messages": messages,
        "max_completion_tokens": settings.max_completion_tokens,
    }
    if settings.model.lower().startswith("kimi-k3") and settings.reasoning_effort:
        kwargs["reasoning_effort"] = settings.reasoning_effort
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 — 把 SDK 错误收成给演示者看的一句中文
        raise LLMError(f"调用 Kimi 失败：{exc}") from exc
    choice = response.choices[0].message
    text = _message_text(choice)
    if not text:
        raise LLMError("Kimi 返回了空内容。")
    return text


def _message_text(message: object) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
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
        joined = "".join(parts).strip()
        if joined:
            return joined
    dump = message.model_dump() if hasattr(message, "model_dump") else {}
    for key in ("content", "reasoning_content"):
        value = dump.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
