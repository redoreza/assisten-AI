"""STTAgent — thin wrapper around the Groq Whisper STT service.

Owns the session-aware bias prompt so neither the orchestrator nor the
WebSocket layer need to import or compute it themselves.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# Bias prompt fed to Whisper — keeps it in Indonesian campus vocabulary
# and reduces misheard names.
_BIAS_BASE = (
    "Percakapan santai antara user dengan asisten virtual kampus bernama Pointer. "
    "Bahasa Indonesia sehari-hari, kadang mahasiswa, dosen, kelas, jadwal, kampus, "
    "ruang kuliah, fakultas, prodi."
)


def _bias_prompt(session: Any) -> str:
    if session.last_engaged_name:
        return _BIAS_BASE + f" Nama lawan bicara: {session.last_engaged_name}."
    return _BIAS_BASE


class STTAgent:
    def __init__(self) -> None:
        from app.services.stt_groq import get_stt
        self._stt = get_stt()

    async def transcribe(
        self,
        audio: bytes,
        *,
        format: str = "webm",
        session: Any,
    ) -> str:
        """Transcribe audio bytes to Indonesian text with session-aware hints."""
        prompt = _bias_prompt(session)
        try:
            return await self._stt.transcribe(
                audio,
                format=format,
                language="id",
                prompt=prompt,
            )
        except Exception:
            logger.exception("STTAgent transcribe failed")
            raise


_singleton: STTAgent | None = None


def get_stt_agent() -> STTAgent:
    global _singleton
    if _singleton is None:
        _singleton = STTAgent()
    return _singleton
