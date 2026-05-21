"""Speech-to-Text via Groq Whisper-large-v3-turbo.

Groq's transcription endpoint accepts mp3/mp4/m4a/wav/webm/ogg, so the browser's
default MediaRecorder output (webm/opus) is sent as-is — no FFmpeg conversion
needed for the happy path.
"""

from __future__ import annotations

import io

from groq import AsyncGroq
from loguru import logger

from app.config import settings


class GroqSTT:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or settings.groq_api_key
        if not key:
            raise ValueError("GROQ_API_KEY is not set. Add it to .env at the project root.")
        self._client = AsyncGroq(api_key=key)
        self._model = model or settings.stt_model

    @property
    def model(self) -> str:
        return self._model

    async def transcribe(
        self,
        audio: bytes,
        *,
        format: str = "webm",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Send raw audio bytes to Groq Whisper, return text.

        Args:
            audio: raw audio bytes.
            format: file extension hint (`webm`, `mp3`, `wav`, `m4a`, `ogg`). Groq
                uses the filename to detect the format.
            language: ISO-639-1 hint (e.g. `id`, `en`). Omit to let Whisper auto-detect.
            prompt: optional context to bias decoding (e.g. uncommon names).
        """
        if not audio:
            return ""
        buf = io.BytesIO(audio)
        buf.name = f"audio.{format}"
        kwargs: dict = {"file": buf, "model": self._model}
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt
        resp = await self._client.audio.transcriptions.create(**kwargs)
        text = (resp.text or "").strip()
        logger.debug(f"Groq STT ({format}, {len(audio)} bytes) → '{text[:80]}'")
        return text


_singleton: GroqSTT | None = None


def get_stt() -> GroqSTT:
    global _singleton
    if _singleton is None:
        _singleton = GroqSTT()
    return _singleton
