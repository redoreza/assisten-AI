"""OpenRouter LLM provider — wraps the OpenAI SDK pointed at OpenRouter.

Same public surface as GroqLLM (generate / generate_stream / complete_with_tools)
so the router can swap providers freely.

OpenRouter (https://openrouter.ai) is an OpenAI-compatible aggregator that
exposes many models through a single API key. We use only the `:free` tagged
models so usage is rate-limited but never costs money.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from app.config import settings
from app.services.llm_groq import ChatMessage, ToolCall, ToolResponse


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterLLM:
    name = "openrouter"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or settings.openrouter_api_key
        if not key:
            raise ValueError("OPENROUTER_API_KEY is not set in .env")
        self._client = AsyncOpenAI(
            api_key=key,
            base_url=OPENROUTER_BASE_URL,
            # Optional headers OpenRouter likes for analytics
            default_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Pointer Campus Assistant",
            },
        )
        self._model = model or settings.openrouter_model

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        content = resp.choices[0].message.content or ""
        logger.debug(f"OpenRouter ({self._model}) non-stream reply ({len(content)} chars)")
        return content

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    async def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.5,
        max_tokens: int = 512,
        tool_choice: str = "auto",
    ) -> ToolResponse:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        msg = resp.choices[0].message
        calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    }
                )
        return {"content": msg.content, "tool_calls": calls}


_singleton: OpenRouterLLM | None = None


def get_openrouter_llm() -> OpenRouterLLM:
    global _singleton
    if _singleton is None:
        _singleton = OpenRouterLLM()
    return _singleton
