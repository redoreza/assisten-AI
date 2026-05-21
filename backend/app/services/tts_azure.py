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
        self._key = key
        self._region = settings.azure_speech_region
        self._default_voice = default_voice or settings.tts_voice_default
        # 24kHz 160kbps mono MP3 matches Edge TTS output format so the frontend
        # AudioQueue behaves the same regardless of provider.
        self._format = (
            speechsdk.SpeechSynthesisOutputFormat.Audio24Khz160KBitRateMonoMp3
        )
        logger.info(
            f"AzureTTS ready (region={self._region}, default_voice={self._default_voice})"
        )

    def _synth_sync(
        self, ssml: str
    ) -> tuple[bytes, list[WordBoundary], list[SentenceBoundary]]:
        config = speechsdk.SpeechConfig(
            subscription=self._key, region=self._region
        )
        config.set_speech_synthesis_output_format(self._format)
        config.set_property(
            speechsdk.PropertyId.SpeechServiceResponse_RequestWordBoundary,
            "true",
        )
        # audio_config=None → keep bytes in result.audio_data (no playback / no file)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=config, audio_config=None
        )

        words: list[WordBoundary] = []
        sentences: list[SentenceBoundary] = []

        # In the Python SDK, both word- and sentence-level events arrive on the
        # same `synthesis_word_boundary` signal, distinguished by `boundary_type`
        # (Word / Sentence / Punctuation).
        sentence_type = speechsdk.SpeechSynthesisBoundaryType.Sentence
        punct_type = speechsdk.SpeechSynthesisBoundaryType.Punctuation

        def on_boundary(evt):
            btype = getattr(evt, "boundary_type", None)
            # Skip punctuation markers — they have duration=0 and would
            # produce empty viseme spans.
            if btype == punct_type:
                return
            entry = {
                "text": evt.text,
                "offset": evt.audio_offset / 10_000_000.0,  # 100ns → s
                "duration": evt.duration.total_seconds(),
            }
            if btype == sentence_type:
                sentences.append(entry)
            else:
                words.append(entry)

        synthesizer.synthesis_word_boundary.connect(on_boundary)

        result = synthesizer.speak_ssml(ssml)
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
