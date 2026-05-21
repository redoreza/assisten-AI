"""LLM provider router — round-robin Groq + OpenRouter with auto-fallback.

Goals
-----
- Spread load roughly evenly across providers so neither hits free-tier
  rate limits faster than necessary.
- Auto-fall back to a healthy provider when one errors (rate-limited, 5xx,
  network blip). Skip a provider after N consecutive failures, retry it
  later after a cool-off.
- Streaming calls don't mid-stream fallback (would corrupt sentence order
  in our TTS pipeline) but they do try a different provider on retry of
  the next call.

Public surface mirrors GroqLLM so the orchestrator doesn't care which
provider answered.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any, Protocol

from loguru import logger

from app.services.llm_groq import ChatMessage, ToolResponse

# Allow N consecutive failures before we mark a provider unhealthy
_UNHEALTHY_FAIL_THRESHOLD = 3
# Re-check an unhealthy provider after this long
_UNHEALTHY_COOLDOWN_S = 60.0


class LLMProvider(Protocol):
    name: str
    model: str

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = ...,
        max_tokens: int = ...,
    ) -> str: ...

    def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = ...,
        max_tokens: int = ...,
    ) -> AsyncIterator[str]: ...

    async def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        *,
        temperature: float = ...,
        max_tokens: int = ...,
        tool_choice: str = ...,
    ) -> ToolResponse: ...


class _ProviderStats:
    def __init__(self) -> None:
        self.consecutive_failures = 0
        self.total_calls = 0
        self.total_failures = 0
        self.last_failure_at: float = 0.0


class LLMRouter:
    """Round-robin + fallback wrapper for a list of LLM providers."""

    name = "router"

    def __init__(self, providers: list[LLMProvider]) -> None:
        if not providers:
            raise ValueError("LLMRouter needs at least one provider")
        self._providers: list[LLMProvider] = providers
        self._stats: dict[str, _ProviderStats] = {p.name: _ProviderStats() for p in providers}
        self._next_idx = 0

    @property
    def model(self) -> str:
        return ", ".join(f"{p.name}:{p.model}" for p in self._providers)

    def _is_healthy(self, p: LLMProvider) -> bool:
        s = self._stats[p.name]
        if s.consecutive_failures < _UNHEALTHY_FAIL_THRESHOLD:
            return True
        return time.monotonic() - s.last_failure_at >= _UNHEALTHY_COOLDOWN_S

    def _record_success(self, p: LLMProvider) -> None:
        s = self._stats[p.name]
        s.consecutive_failures = 0
        s.total_calls += 1

    def _record_failure(self, p: LLMProvider, exc: BaseException) -> None:
        s = self._stats[p.name]
        s.consecutive_failures += 1
        s.total_failures += 1
        s.last_failure_at = time.monotonic()
        logger.warning(
            f"[llm_router] {p.name} failed (consecutive={s.consecutive_failures}, "
            f"total_fails={s.total_failures}): {exc}"
        )

    def _candidates(self) -> list[LLMProvider]:
        n = len(self._providers)
        start = self._next_idx
        self._next_idx = (self._next_idx + 1) % n
        ordered = [self._providers[(start + i) % n] for i in range(n)]
        healthy = [p for p in ordered if self._is_healthy(p)]
        unhealthy = [p for p in ordered if not self._is_healthy(p)]
        return healthy + unhealthy

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            name: {
                "consecutive_failures": s.consecutive_failures,
                "total_calls": s.total_calls,
                "total_failures": s.total_failures,
            }
            for name, s in self._stats.items()
        }

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        last_exc: BaseException | None = None
        for p in self._candidates():
            try:
                result = await p.generate(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
                self._record_success(p)
                return result
            except Exception as exc:
                self._record_failure(p, exc)
                last_exc = exc
        raise RuntimeError(f"all LLM providers failed; last error: {last_exc}")

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        last_exc: BaseException | None = None
        for p in self._candidates():
            try:
                stream = p.generate_stream(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
                aiter = stream.__aiter__()
                try:
                    first = await aiter.__anext__()
                    self._record_success(p)
                except StopAsyncIteration:
                    self._record_success(p)
                    return
                yield first
                async for chunk in aiter:
                    yield chunk
                return
            except Exception as exc:
                self._record_failure(p, exc)
                last_exc = exc
        raise RuntimeError(f"all LLM providers failed; last error: {last_exc}")

    async def complete_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.5,
        max_tokens: int = 512,
        tool_choice: str = "auto",
    ) -> ToolResponse:
        last_exc: BaseException | None = None
        for p in self._candidates():
            try:
                result = await p.complete_with_tools(
                    messages,
                    tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tool_choice=tool_choice,
                )
                self._record_success(p)
                return result
            except Exception as exc:
                self._record_failure(p, exc)
                last_exc = exc
        raise RuntimeError(f"all LLM providers failed; last error: {last_exc}")


def build_router() -> LLMRouter:
    """Construct the router from whatever providers are configured.

    Priority order (first one called first on round-robin):
        1. NVIDIA NIM  (set NVIDIA_API_KEY)
        2. Groq        (set GROQ_API_KEY — required)
        3. OpenRouter  (set OPENROUTER_API_KEY)

    Each new round of requests advances the cursor by one, so over time
    all healthy providers share the load roughly equally.
    """
    from app.config import settings
    from app.services.llm_groq import GroqLLM

    providers: list[LLMProvider] = []
    if settings.nvidia_api_key:
        from app.services.llm_nvidia import NvidiaLLM

        providers.append(NvidiaLLM())
    providers.append(GroqLLM())
    if settings.openrouter_api_key:
        from app.services.llm_openrouter import OpenRouterLLM

        providers.append(OpenRouterLLM())

    names = " + ".join(p.name for p in providers)
    logger.info(f"[llm_router] providers ({len(providers)}, round-robin): {names}")
    return LLMRouter(providers)


_singleton: LLMRouter | None = None


def get_router() -> LLMRouter:
    global _singleton
    if _singleton is None:
        _singleton = build_router()
    return _singleton
