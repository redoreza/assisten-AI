"""FastAPI application entrypoint.

Run with:
    uv run uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.chat import router as chat_router
from app.api.face import router as face_router
from app.api.websocket import router as ws_router
from app.config import settings
from app.core.persona import persona_manager


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<7}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    logger.info(f"Starting up — model={settings.llm_model}, default_persona={settings.default_persona}")
    persona_manager.load_all()
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY is empty — /api/chat will return 500 until you set it in .env")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="AI Avatar Companion — Backend",
    version="0.2.0",
    description="Phase 2: voice pipeline (STT Groq Whisper + TTS Edge + WS) over Phase 1 text chat.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(face_router, prefix="/api")
app.include_router(ws_router)


@app.get("/")
async def root() -> dict:
    return {
        "name": "ai-avatar-backend",
        "version": "0.3.0",
        "phase": "R1 — face recognition",
        "endpoints": [
            "/api/chat",
            "/api/chat/stream",
            "/api/personas",
            "/api/face/recognize",
            "/api/face/enroll",
            "/api/face/persons",
            "/api/face/warmup",
            "/ws",
            "/health",
            "/docs",
        ],
    }


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "groq_key_configured": bool(settings.groq_api_key),
        "personas_loaded": persona_manager.list_ids(),
        "model": settings.llm_model,
    }
