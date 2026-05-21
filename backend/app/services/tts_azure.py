"""Azure Speech Services TTS — same public interface as `tts_edge.EdgeTTS`.

Why this exists alongside tts_edge.py:
- Edge TTS is the free anonymous endpoint to Microsoft's neural TTS. It accepts
  only plain text (no SSML), so we have no control over pauses, prosody, or
  break timing — the engine has to infer everything from punctuation.
- Azure Speech is the authenticated endpoint to the SAME engines. It accepts
  full SSML, so we can wrap each utterance with `<prosody>` + `<break>` tags
  to give Pointer a more conversational rhythm.

Output contract mirrors tts_edge.TTSResult exactly so the orchestrator can
swap providers without code changes:
    {audio: bytes (mp3), word_boundaries: [...], sentence_boundaries: [...]}

Voice choice:
- `id-ID-GadisNeural` (default) — same voice as Edge but speaks via SSML; the
  engine sounds identical, the *rhythm* sounds noticeably more natural.
- `id-ID-ArdiNeural` — male equivalent.
- HD multilingual voices (e.g. `en-US-Ava:DragonHDLatestNeural`,
  `en-US-Andrew:DragonHDLatestNeural`) — Microsoft's newest neural voices,
  closer to ElevenLabs quality. They speak Indonesian with mild English
  accent. Availability varies by region (works in eastus/westus/etc.; check
  before switching the persona for southeastasia).
"""

from __future__ import annotations

import asyncio
import re
import threading
from typing import TypedDict
from xml.sax.saxutils import escape as xml_escape

import azure.cognitiveservices.speech as speechsdk
from loguru import logger

from app.config import settings


class WordBoundary(TypedDict):
    text: str
    offset: float  # seconds from start of audio
    duration: float  # seconds


class SentenceBoundary(TypedDict):
    text: str
    offset: float
    duration: float


class TTSResult(TypedDict):
    audio: bytes
    word_boundaries: list[WordBoundary]
    sentence_boundaries: list[SentenceBoundary]
    voice: str


# Pause durations chosen by ear for conversational Indonesian: a comma gives a
# subtle breath, an ellipsis a noticeable beat.
_COMMA_BREAK_MS = 120
_ELLIPSIS_BREAK_MS = 350


def _preprocess_text(text: str) -> str:
    """Same light grooming as tts_edge so the two providers behave alike."""
    t = text.strip()
    t = re.sub(r"\.{3,}", "…", t)
    t = re.sub(r"\s-\s", ", ", t)
    t = re.sub(r"([?!])\1+", r"\1", t)
    return t


def _build_inner_ssml(text: str) -> str:
    """Escape `text` for SSML and inject `<break>` tags at natural pause points.

    Returns the inner body that should sit inside `<voice>…</voice>` — does NOT
    include the voice or speak wrapper. Embedded XML tags here are intentional.
    """
    parts = re.split(r"(,\s*|…\s*)", text)
    pieces: list[str] = []
    for chunk in parts:
        if not chunk:
            continue
        if chunk.startswith(","):
            pieces.append(", ")
            pieces.append(f'<break time="{_COMMA_BREAK_MS}ms"/>')
        elif chunk.startswith("…"):
            pieces.append("… ")
            pieces.append(f'<break time="{_ELLIPSIS_BREAK_MS}ms"/>')
        else:
            pieces.append(xml_escape(chunk))
    return "".join(pieces)


def _voice_lang(voice: str) -> str:
    """Determine xml:lang for an SSML utterance.

    HD multilingual voices (DragonHDLatestNeural and the *Multilingual* family)
    are nominally en-US but are designed to speak any of ~70 languages — the
    xml:lang attribute tells the engine which language to render. Since this
    assistant always speaks Bahasa Indonesia, force id-ID for those voices.

    For locale-specific voices (id-ID-GadisNeural, etc.) just mirror the
    voice's own locale prefix.
    """
    if "Dragon" in voice or "Multilingual" in voice:
        return "id-ID"
    m = re.match(r"([a-z]{2}-[A-Z]{2})", voice)
    return m.group(1) if m else "id-ID"


def build_ssml(text: str, voice: str, rate: str, pitch: str) -> str:
    body = _build_inner_ssml(text)
    return (
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="{_voice_lang(voice)}">'
        f'<voice name="{voice}">'
        f'<prosody rate="{rate}" pitch="{pitch}">'
        f'{body}'
        '</prosody>'
        '</voice>'
        '</speak>'
    )


class AzureTTS:
    def __init__(
        self,
        default_voice: str | None = None,
        api_key: str | None = None,
    ) -> None:
        key = api_key or settings.azure_speech_key
        if not key:
            raise ValueError("AZURE_SPEECH_KEY not configured")
        self._region = settings.azure_speech_region
        self._default_voice = default_voice or settings.tts_voice_default
        # 24kHz 160kbps mono MP3 matches Edge TTS output format so the frontend
        # AudioQueue behaves the same regardless of provider.
        fmt = speechsdk.SpeechSynthesisOutputFormat.Audio24Khz160KBitRateMonoMp3

        config = speechsdk.SpeechConfig(subscription=key, region=self._region)
        config.set_speech_synthesis_output_format(fmt)
        config.set_property(
            speechsdk.PropertyId.SpeechServiceResponse_RequestWordBoundary,
            "true",
        )
        # Persistent synthesizer — keeps the underlying WebSocket connection open
        # between calls so subsequent syntheses avoid the ~1s connection setup.
        # audio_config=None → audio bytes in result.audio_data only, no playback.
        self._synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=config, audio_config=None
        )
        # Serialize concurrent calls; the SDK is not thread-safe.
        self._lock = threading.Lock()
        # Per-call boundary accumulators; reset inside the lock before each call.
        self._cur_words: list[WordBoundary] = []
        self._cur_sentences: list[SentenceBoundary] = []

        # Register a single persistent callback — EventSignal.connect() is the
        # only registration API; there is no disconnect(). The callback writes
        # to `_cur_words`/`_cur_sentences` which are cleared before each call.
        sentence_type = speechsdk.SpeechSynthesisBoundaryType.Sentence
        punct_type = speechsdk.SpeechSynthesisBoundaryType.Punctuation

        def _on_boundary(evt):
            btype = getattr(evt, "boundary_type", None)
            if btype == punct_type:
                return
            entry: WordBoundary | SentenceBoundary = {
                "text": evt.text,
                "offset": evt.audio_offset / 10_000_000.0,  # 100ns → s
                "duration": evt.duration.total_seconds(),
            }
            if btype == sentence_type:
                self._cur_sentences.append(entry)
            else:
                self._cur_words.append(entry)

        self._synthesizer.synthesis_word_boundary.connect(_on_boundary)

        logger.info(
            f"AzureTTS ready (region={self._region}, default_voice={self._default_voice})"
        )
        # Open the WebSocket now so the first real synthesis call is fast.
        threading.Thread(target=self._warmup, daemon=True).start()

    def _warmup(self) -> None:
        try:
            ssml = build_ssml(".", self._default_voice, "+0%", "+0Hz")
            with self._lock:
                self._cur_words = []
                self._cur_sentences = []
                self._synthesizer.speak_ssml(ssml)
            logger.debug("AzureTTS WebSocket pre-warmed")
        except Exception as exc:
            logger.debug(f"AzureTTS warmup skipped: {exc}")

    def _synth_sync(
        self, ssml: str
    ) -> tuple[bytes, list[WordBoundary], list[SentenceBoundary]]:
        with self._lock:
            self._cur_words = []
            self._cur_sentences = []
            result = self._synthesizer.speak_ssml(ssml)
            # Snapshot while still holding the lock so a concurrent warmup
            # can't clobber the lists before we return them.
            words = list(self._cur_words)
            sentences = list(self._cur_sentences)

        # Re-warm the connection immediately after synthesis so the next call
        # finds the WebSocket already open.
        threading.Thread(target=self._warmup, daemon=True).start()

        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = f"reason={result.reason}"
            if result.reason == speechsdk.ResultReason.Canceled:
                cancel = speechsdk.CancellationDetails.from_result(result)
                details += (
                    f" cancel_reason={cancel.reason} "
                    f"error={cancel.error_details}"
                )
            raise RuntimeError(f"Azure TTS failed: {details}")
        return bytes(result.audio_data), words, sentences

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
        ssml = build_ssml(processed, voice_id, rate, pitch)
        audio, words, sentences = await asyncio.to_thread(self._synth_sync, ssml)
        logger.debug(
            f"Azure TTS '{voice_id}' chars={len(text)} rate={rate} pitch={pitch} → "
            f"{len(audio)} bytes, {len(words)} words, {len(sentences)} sentences"
        )
        return {
            "audio": audio,
            "word_boundaries": words,
            "sentence_boundaries": sentences,
            "voice": voice_id,
        }


_singleton: AzureTTS | None = None


def get_azure_tts() -> AzureTTS:
    global _singleton
    if _singleton is None:
        _singleton = AzureTTS()
    return _singleton
