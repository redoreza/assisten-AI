"""LLMStreamAgent — LLM streaming → sentence splitting → per-sentence TTS.

Accepts a fully-prepared messages list and a Persona, then:
  1. Streams tokens from the LLM
  2. Splits the stream into sentences on punctuation boundaries
  3. TTS each sentence in parallel as it's recognized
  4. Emits events in order: ai_text (partial), viseme, audio, ai_text (final),
     timing, done

The 'done' event carries a 'full_reply' key with the complete assistant text.
Callers should persist that value, then re-emit {"type": "done"} to the client.
"""

from __future__ import annotations

import asyncio
import base64
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from app.core.persona import Persona
from app.core.viseme import boundaries_to_visemes
from app.services.llm_groq import ChatMessage

_SENTENCE_END = re.compile(r"([\.!\?…]+)(\s|$)")
_MAX_SENTENCE_CHARS = 90  # flush at word boundary even without punctuation


class LLMStreamAgent:
    def __init__(self, llm: Any, tts: Any, speech_prep: Any) -> None:
        self._llm = llm
        self._tts = tts
        self._speech_prep = speech_prep

    async def _tts_sentence(
        self,
        sentence: str,
        idx: int,
        seq: int,
        persona: Persona,
        t0: float,
        first_audio_ref: list[int | None],
    ) -> dict[str, Any]:
        """Prep + synthesize one sentence; returns audio + viseme payloads."""
        tts_t0 = time.perf_counter()
        spoken = await self._speech_prep.prepare(sentence)
        if spoken != sentence:
            logger.debug(f"[speech-prep] {sentence[:60]!r} -> {spoken[:60]!r}")
        result = await self._tts.synthesize(
            spoken,
            voice=persona.voice.voice_id,
            rate=persona.voice.rate,
            pitch=persona.voice.pitch,
        )
        tts_ms = int((time.perf_counter() - tts_t0) * 1000)
        if first_audio_ref[0] is None:
            first_audio_ref[0] = int((time.perf_counter() - t0) * 1000)
        visemes = boundaries_to_visemes(result["word_boundaries"])
        return {
            "audio_event": {
                "type": "audio",
                "data": base64.b64encode(result["audio"]).decode("ascii"),
                "format": "mp3",
                "sequence": seq,
                "sentence_idx": idx,
                "tts_ms": tts_ms,
            },
            "viseme_event": {
                "type": "viseme",
                "events": visemes,
                "audio_seq": seq,
            },
        }

    async def run(
        self,
        messages: list[ChatMessage],
        persona: Persona,
        t0: float,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream LLM → sentence split → TTS → events.

        Yields ai_text, viseme, audio, then a final ai_text (is_final=True),
        timing, and done. The done event includes 'full_reply' for the caller
        to persist before forwarding {"type": "done"} to the client.
        """
        first_token_ms: int | None = None
        # Mutable reference so _tts_sentence can write first_audio_ms back
        first_audio_ref: list[int | None] = [None]
        sentence_idx = 0
        audio_seq = 0
        buffer = ""
        full_reply = ""
        pending: list[asyncio.Task[dict[str, Any]]] = []

        try:
            async for chunk in self._llm.generate_stream(messages, temperature=0.5):
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - t0) * 1000)
                buffer += chunk
                full_reply += chunk
                while True:
                    match = _SENTENCE_END.search(buffer)
                    if match:
                        end = match.end()
                        sentence = buffer[:end].strip()
                        buffer = buffer[end:]
                        if not sentence:
                            continue
                    elif len(buffer) >= _MAX_SENTENCE_CHARS:
                        split_pos = buffer.rfind(' ')
                        if split_pos > 20:
                            sentence = buffer[:split_pos].strip()
                            buffer = buffer[split_pos:].lstrip()
                        else:
                            break
                    else:
                        break
                    yield {
                        "type": "ai_text",
                        "text": sentence,
                        "is_final": False,
                        "sentence_idx": sentence_idx,
                    }
                    pending.append(
                        asyncio.create_task(
                            self._tts_sentence(
                                sentence, sentence_idx, audio_seq,
                                persona, t0, first_audio_ref,
                            )
                        )
                    )
                    sentence_idx += 1
                    audio_seq += 1
        except Exception as exc:
            logger.exception("LLM stream failed")
            yield {"type": "error", "message": f"LLM failed: {exc}"}
            return

        # Flush any remaining buffer after the stream ends
        tail = buffer.strip()
        if tail:
            yield {
                "type": "ai_text",
                "text": tail,
                "is_final": False,
                "sentence_idx": sentence_idx,
            }
            pending.append(
                asyncio.create_task(
                    self._tts_sentence(
                        tail, sentence_idx, audio_seq,
                        persona, t0, first_audio_ref,
                    )
                )
            )
            sentence_idx += 1

        # Drain TTS tasks in order — viseme before audio so client has lip-sync
        # data ready before the audio element starts playing.
        for task in pending:
            try:
                payload = await task
            except Exception as exc:
                logger.exception("TTS task failed")
                yield {"type": "error", "message": f"TTS failed: {exc}"}
                continue
            yield payload["viseme_event"]
            yield payload["audio_event"]

        total_ms = int((time.perf_counter() - t0) * 1000)
        reply_text = full_reply.strip()
        yield {
            "type": "ai_text",
            "text": reply_text,
            "is_final": True,
            "sentence_idx": sentence_idx,
        }
        yield {
            "type": "timing",
            "first_token_ms": first_token_ms,
            "first_audio_ms": first_audio_ref[0],
            "total_ms": total_ms,
            "sentences": sentence_idx,
        }
        # Carry full_reply so the caller can persist it before forwarding done.
        yield {"type": "done", "full_reply": reply_text}


_singleton: LLMStreamAgent | None = None


def get_llm_stream_agent() -> LLMStreamAgent:
    global _singleton
    if _singleton is None:
        from app.services.llm_groq import get_llm
        from app.services.tts_edge import get_tts
        from app.agents.speech_prep_agent import get_speech_prep_agent
        _singleton = LLMStreamAgent(get_llm(), get_tts(), get_speech_prep_agent())
    return _singleton
