"""REST chat endpoints — Phase 1.

POST /api/chat         → JSON {reply: str, persona_id, latency_ms}
POST /api/chat/stream  → Server-Sent Events: each `data:` line is a token chunk,
                        terminated by `data: [DONE]`.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from app.core.persona import persona_manager
from app.services.llm_groq import ChatMessage, get_llm

router = APIRouter(tags=["chat"])


class HistoryTurn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    persona_id: str | None = None
    history: list[HistoryTurn] = Field(default_factory=list)
    stream: bool = False


class ChatResponse(BaseModel):
    reply: str
    persona_id: str
    model: str
    latency_ms: int


def _build_messages(req: ChatRequest) -> tuple[list[ChatMessage], str]:
    persona = persona_manager.get(req.persona_id or settings.default_persona)
    system = persona.render_system_prompt()
    messages: list[ChatMessage] = [{"role": "system", "content": system}]
    for turn in req.history[-settings.memory_max_turns :]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": req.message})
    return messages, persona.id


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(400, "message must not be empty")

    messages, persona_id = _build_messages(req)
    try:
        llm = get_llm()
    except ValueError as exc:
        raise HTTPException(500, str(exc)) from exc

    t0 = time.perf_counter()
    reply = await llm.generate(messages)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(f"/chat persona={persona_id} latency={latency_ms}ms reply_chars={len(reply)}")
    return ChatResponse(
        reply=reply,
        persona_id=persona_id,
        model=llm.model,
        latency_ms=latency_ms,
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    if not req.message.strip():
        raise HTTPException(400, "message must not be empty")

    messages, persona_id = _build_messages(req)
    try:
        llm = get_llm()
    except ValueError as exc:
        raise HTTPException(500, str(exc)) from exc

    async def event_source() -> AsyncIterator[str]:
        t0 = time.perf_counter()
        first_token_ms: int | None = None
        total_chars = 0
        meta = {"persona_id": persona_id, "model": llm.model}
        yield f"data: {json.dumps({'event': 'meta', **meta})}\n\n"
        try:
            async for chunk in llm.generate_stream(messages):
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - t0) * 1000)
                total_chars += len(chunk)
                yield f"data: {json.dumps({'event': 'token', 'text': chunk})}\n\n"
        except Exception as exc:
            logger.exception("LLM stream error")
            yield f"data: {json.dumps({'event': 'error', 'message': str(exc)})}\n\n"
            return
        total_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            f"/chat/stream persona={persona_id} first_token={first_token_ms}ms "
            f"total={total_ms}ms chars={total_chars}"
        )
        yield f"data: {json.dumps({'event': 'done', 'first_token_ms': first_token_ms, 'total_ms': total_ms})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/personas")
async def list_personas() -> dict:
    return {"personas": persona_manager.list_ids(), "default": settings.default_persona}
