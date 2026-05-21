"""Text-to-Speech via Microsoft Edge TTS (free, unlimited, multilingual).

Returns MP3 bytes plus boundary events for lip-sync.

Edge TTS v7 only emits SentenceBoundary (not WordBoundary) for most voices, so
when WordBoundary is absent we synthesize per-word timings by distributing the
sentence's duration across its words weighted by character count. Not
sub-phoneme accurate, but visually convincing for blendshape lip-sync.

This module also applies light preprocessing/prosody adjustment to make the
voice less monotone:
- Convert "..." and " - " into natural pauses.
- Bias rate down a bit for questions (pacing for reflection).
- Bias pitch + rate slightly up for exclamations (engagement).
- A small deterministic-but-varied jitter per sentence so consecutive
  sentences never use identical rate/pitch.
"""

from __future__ import annotations

import hashlib
import re
from typing import TypedDict

import edge_tts
from loguru import logger

from app.config import settings


class WordBoundary(TypedDict):
    text: str
    offset: float
    duration: float


class SentenceBoundary(TypedDict):
    text: str
    offset: float
    duration: float


class TTSResult(TypedDict):
    audio: bytes
    word_boundaries: list[WordBoundary]
    sentence_boundaries: list[SentenceBoundary]
    voice: str


_TICKS_PER_SECOND = 10_000_000
_WORD_RE = re.compile(r"\S+")


def _estimate_word_boundaries(sentence: SentenceBoundary) -> list[WordBoundary]:
    words = _WORD_RE.findall(sentence["text"])
    if not words:
        return []
    weights = [max(len(w), 1) for w in words]
    total_weight = sum(weights)
    base_offset = sentence["offset"]
    base_duration = sentence["duration"]
    out: list[WordBoundary] = []
    cursor = 0.0
    for word, w in zip(words, weights, strict=True):
        dur = base_duration * (w / total_weight)
        out.append({"text": word, "offset": base_offset + cursor, "duration": dur})
        cursor += dur
    return out


def _preprocess_text(text: str) -> str:
    """Light text grooming to coax more natural delivery from Edge TTS."""
    t = text.strip()
    # Long pauses
    t = re.sub(r"\.{3,}", "…", t)
    # Hyphen-as-dash → comma break
    t = re.sub(r"\s-\s", ", ", t)
    # Collapse spammy punctuation
    t = re.sub(r"([?!])\1+", r"\1", t)
    t = re.sub(r"\.\.", ".", t)
    return t


def _parse_offset(spec: str, unit: str) -> int:
    m = re.match(rf"^([+\-]?\d+){re.escape(unit)}$", spec.strip())
    if not m:
        return 0
    return int(m.group(1))


def _format_offset(value: int, unit: str) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value}{unit}"


def _adjust_prosody(text: str, base_rate: str, base_pitch: str) -> tuple[str, str]:
    """Return (rate, pitch) adjusted for sentence content.

    Adjustments are intentionally small (≤ ±6 units) — bigger swings sound
    cartoonish coming from Edge TTS's neural model.
    """
    rate_n = _parse_offset(base_rate, "%")
    pitch_n = _parse_offset(base_pitch, "Hz")

    if "?" in text:
        rate_n -= 4
        pitch_n += 1
    elif "!" in text:
        rate_n += 5
        pitch_n += 2

    # Stable jitter per text → same input always gets same output (so an audio
    # cache later in the pipeline can hit), but consecutive sentences differ.
    digest = hashlib.md5(text.encode("utf-8")).digest()
    rate_jitter = (digest[0] % 7) - 3   # -3..+3
    pitch_jitter = (digest[1] % 5) - 2  # -2..+2

    rate_n += rate_jitter
    pitch_n += pitch_jitter

    rate_n = max(-15, min(25, rate_n))
    pitch_n = max(-10, min(10, pitch_n))

    return _format_offset(rate_n, "%"), _format_offset(pitch_n, "Hz")


class EdgeTTS:
    def __init__(self, default_voice: str | None = None) -> None:
        self._default_voice = default_voice or settings.tts_voice_default

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> TTSResult:
        voice_id = voice or self._default_voice
        processed = _preprocess_text(text)
        rate_eff, pitch_eff = _adjust_prosody(processed, rate, pitch)

        comm = edge_tts.Communicate(processed, voice_id, rate=rate_eff, pitch=pitch_eff)
        audio = bytearray()
        words: list[WordBoundary] = []
        sentences: list[SentenceBoundary] = []
        async for chunk in comm.stream():
            t = chunk["type"]
            if t == "audio":
                audio.extend(chunk["data"])
            elif t == "WordBoundary":
                words.append(
                    {
                        "text": chunk["text"],
                        "offset": chunk["offset"] / _TICKS_PER_SECOND,
                        "duration": chunk["duration"] / _TICKS_PER_SECOND,
                    }
                )
            elif t == "SentenceBoundary":
                sentences.append(
                    {
                        "text": chunk["text"],
                        "offset": chunk["offset"] / _TICKS_PER_SECOND,
                        "duration": chunk["duration"] / _TICKS_PER_SECOND,
                    }
                )

        if not words and sentences:
            for s in sentences:
                words.extend(_estimate_word_boundaries(s))
            logger.debug(
                f"Edge TTS '{voice_id}' estimated {len(words)} word boundaries "
                f"from {len(sentences)} sentences"
            )

        logger.debug(
            f"Edge TTS '{voice_id}' chars={len(text)} rate={rate_eff} pitch={pitch_eff} → "
            f"{len(audio)} bytes audio, {len(words)} words, {len(sentences)} sentences"
        )
        return {
            "audio": bytes(audio),
            "word_boundaries": words,
            "sentence_boundaries": sentences,
            "voice": voice_id,
        }


_singleton: EdgeTTS | None = None


def get_edge_tts() -> EdgeTTS:
    global _singleton
    if _singleton is None:
        _singleton = EdgeTTS()
    return _singleton


_AZURE_COOLDOWN_S = 120.0  # retry a failed Azure key after this many seconds


class TTSWithFallback:
    """Azure primary → Azure backup key → Edge TTS fallback chain.

    Failure modes handled:
    - Init failure on a key  → skip that key, try next in chain at startup.
    - Per-call failure (rate limit / quota / 5xx) → mark that key on cooldown,
      retry the same sentence with the next entry immediately; retry the key
      again after _AZURE_COOLDOWN_S seconds in case of transient failures.
    Chain: [Azure key1] → [Azure key2 if configured] → [Edge TTS]
    """

    def __init__(self) -> None:
        import time as _time
        from app.services.tts_azure import AzureTTS

        self._chain: list[AzureTTS] = []
        self._edge = get_edge_tts()
        # Maps chain index → timestamp of last failure (0 = never failed)
        self._failed_at: dict[int, float] = {}
        self._time = _time

        for idx, key in enumerate(
            [settings.azure_speech_key, settings.azure_speech_key_2]
        ):
            if not key:
                continue
            try:
                self._chain.append(
                    AzureTTS(default_voice=settings.tts_voice_default, api_key=key)
                )
                label = "primary" if idx == 0 else "backup"
                logger.info(f"TTSWithFallback: Azure {label} key loaded (idx={idx})")
            except Exception as exc:
                logger.warning(f"TTSWithFallback: Azure key[{idx}] init failed: {exc}")

        if self._chain:
            logger.info(
                f"TTSWithFallback: {len(self._chain)} Azure key(s) + Edge fallback"
            )
        else:
            logger.info("TTSWithFallback: no Azure key configured, using Edge only")

    def _is_available(self, idx: int) -> bool:
        t = self._failed_at.get(idx, 0.0)
        return t == 0.0 or (self._time.monotonic() - t) >= _AZURE_COOLDOWN_S

    async def synthesize(self, text: str, **kwargs) -> TTSResult:
        for idx, provider in enumerate(self._chain):
            if not self._is_available(idx):
                continue
            try:
                return await provider.synthesize(text, **kwargs)
            except Exception as exc:
                self._failed_at[idx] = self._time.monotonic()
                remaining = sum(
                    1 for i in range(idx + 1, len(self._chain))
                    if self._is_available(i)
                )
                next_label = f"Azure key[{idx + 1}]" if remaining > 0 else "Edge TTS"
                logger.warning(
                    f"TTSWithFallback: Azure key[{idx}] failed → {next_label} "
                    f"(will retry in {_AZURE_COOLDOWN_S:.0f}s): {exc}"
                )
        return await self._edge.synthesize(text, **kwargs)


_tts_with_fallback: TTSWithFallback | None = None


def get_tts() -> TTSWithFallback:
    """Public TTS accessor used by the orchestrator.

    Returns a TTSWithFallback instance: Azure primary when AZURE_SPEECH_KEY
    is set, Edge TTS as automatic fallback on init or per-call failure.
    """
    global _tts_with_fallback
    if _tts_with_fallback is None:
        _tts_with_fallback = TTSWithFallback()
    return _tts_with_fallback
