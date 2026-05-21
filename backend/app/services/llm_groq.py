"""Groq LLM service — async streaming wrapper around the official groq SDK.

Single responsibility: take a list of OpenAI-style chat messages, return either
the full assistant reply or an async iterator of token chunks.

Designed so a future GeminiLLM or LocalLLM can implement the same two methods
and be swapped in via the orchestrator.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, TypedDict

from groq import AsyncGroq
from loguru import logger

from app.config import settings


class ChatMessage(TypedDict, total=False):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None
    # When role="assistant" and LLM emitted tool calls, content is None and
    # this list is populated. When role="tool", tool_call_id matches the id
    # of the assistant tool_call we're replying to.
    tool_calls: list[dict[str, Any]]
    tool_call_id: str
    name: str


class ToolCall(TypedDict):
    id: str
    name: str
    arguments: str  # JSON string


class ToolResponse(TypedDict):
    content: str | None
    tool_calls: list[ToolCall]


class GroqLLM:
    name = "groq"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or settings.groq_api_key
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to .env at the project root."
            )
        self._client = AsyncGroq(api_key=key)
        self._model = model or settings.llm_model

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
        """Non-streaming completion. Returns the full assistant reply."""
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        content = resp.choices[0].message.content or ""
        logger.debug(f"Groq non-stream reply ({len(content)} chars)")
        return content

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        """Streaming completion. Yields token chunks as they arrive."""
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
        """Non-streaming completion with tool calling enabled.

        Returns the assistant message: either `content` (no tool needed) or
        `tool_calls` (LLM wants the backend to execute one or more tools).
        """
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


def get_llm():
    """Lazy singleton — returns the multi-provider router so callers don't
    need to know whether OpenRouter is configured. The router exposes the
    same method signatures as GroqLLM.

    (Defined as a thin forwarder to llm_router.get_router so we avoid an
    import cycle at module load.)
    """
    from app.services.llm_router import get_router

    return get_router()
