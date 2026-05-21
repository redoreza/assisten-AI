"""Voice pipeline orchestrator — coordinates STT → LLM → per-sentence TTS.

Emits a stream of typed events that the WebSocket layer forwards to the client:

    {"type": "transcript", "text": ...}
    {"type": "ai_text", "text": ..., "is_final": False, "sentence_idx": 0}
    {"type": "audio", "data": <base64 mp3>, "sequence": 0, "sentence_idx": 0}
    {"type": "viseme", "events": [...], "audio_seq": 0}
    {"type": "timing", "first_token_ms": ..., "first_audio_ms": ...}
    {"type": "done"}
    {"type": "error", "message": ...}

Conversation history is per-session in-memory for Phase 2; SQLite persistence
lands in Phase 5.
"""

from __future__ import annotations

import asyncio
import base64
import re
import time
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from loguru import logger

from app.config import settings
from app.core.persona import Persona, persona_manager
from app.core.viseme import boundaries_to_visemes
from app.services.chat_history import get_chat_history
from app.services.llm_groq import ChatMessage, get_llm
from app.services.search_tavily import get_search
from app.services.stt_groq import get_stt
from app.services.tts_edge import get_tts

# Tool schema for OpenAI/Groq function calling. Description is the LLM's guide
# for WHEN to call this — phrasing matters a lot for trigger accuracy.
SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Cari informasi terkini di internet. Gunakan HANYA kalau pertanyaan "
            "user butuh info aktual yang kamu tidak tahu pasti — misalnya: berita "
            "hari ini, kurs / harga, cuaca, hasil pertandingan, info publik tentang "
            "tokoh/tempat/event, jadwal acara, atau fakta spesifik yang mungkin sudah "
            "berubah. JANGAN panggil tool ini untuk obrolan biasa, sapaan, atau "
            "pertanyaan yang bisa kamu jawab dari pengetahuan umum."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Kata kunci pencarian yang ringkas (bahasa Indonesia atau Inggris)",
                }
            },
            "required": ["query"],
        },
    },
}

Mode = Literal["companion", "customer_service"]

_SENTENCE_END = re.compile(r"([\.!\?…]+)(\s|$)")

# Bias prompt for Whisper STT — gives the model context so it leans toward
# Indonesian campus-conversation vocabulary instead of guessing English words.
# Including a few words the user is likely to say also reduces misheard names.
_STT_BIAS_BASE = (
    "Percakapan santai antara user dengan asisten virtual kampus bernama Pointer. "
    "Bahasa Indonesia sehari-hari, kadang mahasiswa, dosen, kelas, jadwal, kampus, "
    "ruang kuliah, fakultas, prodi."
)


def _stt_bias_prompt(session: SessionState) -> str:
    """Build a per-session Whisper prompt. Include the engaged person's name so
    STT can recognize it when they introduce themselves or mention themselves."""
    if session.last_engaged_name:
        return _STT_BIAS_BASE + f" Nama lawan bicara: {session.last_engaged_name}."
    return _STT_BIAS_BASE


# Llama 3.3 on Groq occasionally emits tool calls as text content instead of
# using the structured `tool_calls` field. We detect a few known patterns and
# synthesize a tool call so the orchestrator can still execute it.
_INLINE_TOOL_PATTERNS = [
    # <function=NAME>{"args":...}</function>
    re.compile(r"<function=(\w+)>(\{.*?\})</function>", re.DOTALL),
    # <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
]


def _parse_inline_tool_calls(content: str) -> list[dict[str, Any]]:
    """Pull tool calls out of a Llama text response. Returns the same shape as
    the structured tool_calls path: [{id, name, arguments}, ...]"""
    out: list[dict[str, Any]] = []
    for i, m in enumerate(_INLINE_TOOL_PATTERNS[0].finditer(content)):
        out.append({"id": f"inline_{i}", "name": m.group(1), "arguments": m.group(2)})
    if out:
        return out
    for i, m in enumerate(_INLINE_TOOL_PATTERNS[1].finditer(content)):
        try:
            import json as _json

            obj = _json.loads(m.group(1))
            name = obj.get("name") or obj.get("function")
            args = obj.get("arguments") or obj.get("parameters") or {}
            if name:
                arg_str = args if isinstance(args, str) else _json.dumps(args)
                out.append({"id": f"inline_{i}", "name": name, "arguments": arg_str})
        except Exception:
            continue
    return out


def _strip_tool_markup(text: str) -> str:
    """Remove any leftover tool-call markup from text that may end up in TTS."""
    s = text
    for pat in _INLINE_TOOL_PATTERNS:
        s = pat.sub("", s)
    # Also strip Llama's special tokens that occasionally leak through
    s = re.sub(r"<\|python_tag\|>|<\|eom_id\|>|<\|eot_id\|>", "", s)
    return s.strip()


_ID_DAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

# Cheap heuristic to decide whether to invoke the phonetic-respeller LLM call:
# match English-typical letter clusters that virtually never appear in pure
# Indonesian text. Saves a ~300ms LLM call on every plain-Indonesian sentence.
_LIKELY_ENGLISH = re.compile(
    r"\b("
    r"the|of|and|in|on|with|for|is|are|to|by|from|"
    r"machine|learning|deep|computer|vision|neural|network|model|object|"
    r"detection|recognition|training|algorithm|programming|software|"
    r"hardware|cloud|server|client|frontend|backend|database|api|interface|"
    r"sign|design|system|prompt|token|stream|tool|search|engine|"
    r"face|prompt|once|only|look|you|your|with|what|how|"
    r"AI|ML|DL|CV|NLP|GPU|CPU|SDK|API|UI|UX|"
    r"yolo|chatgpt|llama|whisper|edge[- ]?tts"
    r")\b",
    re.IGNORECASE,
)

# Formal Bahasa Indonesia markers — when present we run the naturalization LLM
# call to rewrite the sentence into spoken style. Tuned to catch the most
# common formal-writing patterns Llama emits despite our system prompt.
_FORMAL_ID = re.compile(
    r"\b("
    r"tidak|saya|anda|akan|dapat|adalah|merupakan|"
    r"sehingga|namun|tetapi|apabila|jika|maupun|"
    r"sangat|sekali|sangatlah|sebagai|secara|"
    r"sudah|belum|akan|telah|sedang|"
    r"bagaimana|mengapa|seperti|sebagaimana|"
    r"silakan|silahkan|terima\s+kasih|mohon"
    r")\b",
    re.IGNORECASE,
)
_ID_MONTHS = [
    "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]


def _now_indonesian() -> str:
    """Return a human-readable Bahasa Indonesia datetime string for the
    Asia/Jakarta (WIB) timezone — injected into the LLM system prompt so
    Pointer can answer 'jam berapa?' / 'hari apa?' factually."""
    try:
        now = datetime.now(ZoneInfo("Asia/Jakarta"))
    except Exception:
        now = datetime.now()
    day = _ID_DAYS[now.weekday()]
    month = _ID_MONTHS[now.month]
    return f"{day}, {now.day} {month} {now.year}, pukul {now:%H:%M} WIB"


class SessionState:
    """Per-WebSocket session state. Cheap, lives only while the WS is open."""

    def __init__(self, persona_id: str, mode: Mode = "companion") -> None:
        self.persona_id = persona_id
        self.mode: Mode = mode
        self.history: list[ChatMessage] = []
        # Face engagement state — drives auto-greet + enrollment flow (R4)
        # engaged_person_id: who we last greeted (so we don't re-greet on every frame)
        self.engaged_person_id: int | None = None
        # greeted_person_ids: avoid greeting the same person twice in one session
        self.greeted_person_ids: set[int] = set()
        # awaiting_name: True after we asked an unknown face for their name;
        # next user text/voice input is interpreted as the name to enroll
        self.awaiting_name: bool = False
        # The face image we captured of the unknown person — used to enroll
        self.pending_enroll_image_b64: str | None = None
        # When an unknown face first appeared (None = no unknown currently);
        # used so we only ask after the face has been stable for ≥ N seconds
        self.unknown_first_seen_at: float | None = None
        # Last time we sent any face-driven greeting/prompt (debounce)
        self.last_face_action_at: float = 0.0
        # Most recent base64-JPEG of the currently-engaged known face — kept
        # fresh on each face_present event so the correction flow can re-enroll
        # under a new name without needing to re-prompt the user to face the
        # camera again.
        self.last_known_face_image_b64: str | None = None
        # Name we last enrolled the engaged person under; used by the LLM
        # correction-intent prompt and to delete the old DB entry on rename.
        self.last_engaged_name: str | None = None

    def append(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        max_msgs = settings.memory_max_turns * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def reset_face(self) -> None:
        """Call when face is lost. Keeps greeted_person_ids (persons may return)."""
        self.engaged_person_id = None
        self.awaiting_name = False
        self.pending_enroll_image_b64 = None
        self.unknown_first_seen_at = None
        self.last_known_face_image_b64 = None
        self.last_engaged_name = None


class Orchestrator:
    def __init__(self) -> None:
        self._stt = get_stt()
        self._llm = get_llm()
        self._tts = get_tts()

    async def handle_audio(
        self,
        session: SessionState,
        audio: bytes,
        *,
        audio_format: str = "webm",
    ) -> AsyncIterator[dict[str, Any]]:
        t0 = time.perf_counter()
        try:
            transcript = await self._stt.transcribe(
                audio,
                format=audio_format,
                language="id",
                prompt=_stt_bias_prompt(session),
            )
        except Exception as exc:
            logger.exception("STT failed")
            yield {"type": "error", "message": f"STT failed: {exc}"}
            return
        stt_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(f"STT {stt_ms}ms → '{transcript[:80]}'")
        yield {"type": "transcript", "text": transcript, "latency_ms": stt_ms}
        if not transcript.strip():
            yield {"type": "done", "reason": "empty_transcript"}
            return
        async for ev in self.handle_text(session, transcript, _t0=t0):
            yield ev

    async def handle_text(
        self,
        session: SessionState,
        message: str,
        *,
        _t0: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        t0 = _t0 if _t0 is not None else time.perf_counter()
        persona = persona_manager.get(session.persona_id)

        session.append("user", message)
        # Persist user turn to disk (under engaged person if known; else NULL).
        # Fire-and-forget would lose order guarantees — keep awaited but cheap.
        try:
            await get_chat_history().append(
                session.engaged_person_id, "user", message
            )
        except Exception as exc:
            logger.warning(f"chat_history append(user) failed: {exc}")

        # ── Search classifier (if Tavily configured) ────────────────────────
        # Llama's tool-calling on Groq is unreliable — sometimes emits tool
        # calls as text. We use a deterministic 2-stage approach instead:
        # 1. Cheap classifier call: "does this need web search? reply SEARCH: <q> or NO"
        # 2. If SEARCH: run Tavily, inject result as context, then stream normal answer
        # 3. If NO: stream answer immediately
        search_context: str | None = None
        if settings.tavily_api_key:
            search_query = await self._classify_search_need(
                message, history=session.history[:-1]  # exclude the just-added user msg
            )
            if search_query:
                logger.info(f"[handle_text] classifier triggered search: {search_query!r}")
                yield {"type": "tool_use", "tools": ["search_web"], "query": search_query}
                try:
                    result = await get_search().search_and_extract(search_query)
                    search_context = result["llm_context"]
                    logger.info(
                        f"[handle_text] search 1 ({len(search_context)} chars, "
                        f"{len(result['sources'])} sources)"
                    )

                    # Agentic step: evaluate if the result is sufficient. If
                    # not, do one (and only one) refined retry. Cap at 1 retry
                    # to bound the worst-case latency.
                    retry_query = await self._evaluate_search_sufficiency(
                        message, search_context
                    )
                    if retry_query and retry_query != search_query:
                        logger.info(f"[handle_text] retry search with: {retry_query!r}")
                        yield {
                            "type": "tool_use",
                            "tools": ["search_web"],
                            "query": retry_query,
                        }
                        retry_result = await get_search().search_and_extract(retry_query)
                        if retry_result["sources"]:
                            # Merge: prepend new context as primary, then original
                            search_context = (
                                retry_result["llm_context"]
                                + "\n\n--- additional context from previous search ---\n"
                                + search_context
                            )
                            # Merge sources (dedupe by URL)
                            seen_urls = {s["url"] for s in retry_result["sources"]}
                            for s in result["sources"]:
                                if s["url"] not in seen_urls:
                                    retry_result["sources"].append(s)
                                    seen_urls.add(s["url"])
                            result = retry_result
                            search_query = retry_query
                            logger.info(
                                f"[handle_text] search 2 merged "
                                f"({len(search_context)} chars, {len(result['sources'])} sources)"
                            )

                    if result["sources"]:
                        yield {
                            "type": "search_sources",
                            "query": search_query,
                            "sources": result["sources"],
                        }
                except Exception as exc:
                    logger.warning(f"search failed: {exc}")
                    search_context = None

        messages = self._build_messages_with_session(persona, session)
        if search_context:
            # Inject search context as a system addendum, immediately before
            # the user's message in the LLM's view.
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Hasil pencarian web untuk pertanyaan user:\n\n{search_context}\n\n"
                        "Gunakan info di atas untuk jawab pertanyaan user secara ringkas dan natural "
                        "dalam bahasa Indonesia. JANGAN tempel URL/link mentah-mentah ke jawaban suara. "
                        "Kalau hasil kurang relevan, bilang saja kamu tidak yakin."
                    ),
                }
            )

        first_token_ms: int | None = None
        first_audio_ms: int | None = None
        sentence_idx = 0
        audio_seq = 0
        buffer = ""
        full_reply = ""

        async def flush_sentence(sentence: str, idx: int, seq: int) -> dict[str, Any]:
            nonlocal first_audio_ms
            tts_t0 = time.perf_counter()
            spoken = await self._prepare_for_speech(sentence)
            if spoken != sentence:
                logger.debug(f"[speech-prep] {sentence[:60]!r} -> {spoken[:60]!r}")
            result = await self._tts.synthesize(
                spoken,
                voice=persona.voice.voice_id,
                rate=persona.voice.rate,
                pitch=persona.voice.pitch,
            )
            tts_ms = int((time.perf_counter() - tts_t0) * 1000)
            if first_audio_ms is None:
                first_audio_ms = int((time.perf_counter() - t0) * 1000)
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

        pending_tts: list[asyncio.Task[dict[str, Any]]] = []

        try:
            # Lower temperature (0.5) for more deterministic, on-character replies
            # — defaults tended toward 0.7 which gave too much creative drift.
            async for chunk in self._llm.generate_stream(messages, temperature=0.5):
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - t0) * 1000)
                buffer += chunk
                full_reply += chunk
                while True:
                    match = _SENTENCE_END.search(buffer)
                    if not match:
                        break
                    end = match.end()
                    sentence = buffer[:end].strip()
                    buffer = buffer[end:]
                    if not sentence:
                        continue
                    yield {
                        "type": "ai_text",
                        "text": sentence,
                        "is_final": False,
                        "sentence_idx": sentence_idx,
                    }
                    pending_tts.append(
                        asyncio.create_task(
                            flush_sentence(sentence, sentence_idx, audio_seq)
                        )
                    )
                    sentence_idx += 1
                    audio_seq += 1
        except Exception as exc:
            logger.exception("LLM stream failed")
            yield {"type": "error", "message": f"LLM failed: {exc}"}
            return

        tail = buffer.strip()
        if tail:
            yield {
                "type": "ai_text",
                "text": tail,
                "is_final": False,
                "sentence_idx": sentence_idx,
            }
            pending_tts.append(
                asyncio.create_task(flush_sentence(tail, sentence_idx, audio_seq))
            )
            sentence_idx += 1

        for task in pending_tts:
            try:
                payload = await task
            except Exception as exc:
                logger.exception("TTS task failed")
                yield {"type": "error", "message": f"TTS failed: {exc}"}
                continue
            # Emit viseme first so the client has lip-sync data ready BEFORE
            # the audio element starts playing — otherwise the first 100ms of
            # speech happens with no mouth movement.
            yield payload["viseme_event"]
            yield payload["audio_event"]

        if full_reply.strip():
            final_text = full_reply.strip()
            session.append("assistant", final_text)
            try:
                await get_chat_history().append(
                    session.engaged_person_id, "assistant", final_text
                )
            except Exception as exc:
                logger.warning(f"chat_history append(assistant) failed: {exc}")

        total_ms = int((time.perf_counter() - t0) * 1000)
        yield {
            "type": "ai_text",
            "text": full_reply.strip(),
            "is_final": True,
            "sentence_idx": sentence_idx,
        }
        yield {
            "type": "timing",
            "first_token_ms": first_token_ms,
            "first_audio_ms": first_audio_ms,
            "total_ms": total_ms,
            "sentences": sentence_idx,
        }
        yield {"type": "done"}

    def _build_messages(
        self, persona: Persona, history: list[ChatMessage]
    ) -> list[ChatMessage]:
        system: ChatMessage = {"role": "system", "content": persona.render_system_prompt()}
        return [system, *history[-(settings.memory_max_turns * 2) :]]

    async def _prepare_for_speech(self, text: str) -> str:
        """Naturalize formal Bahasa Indonesia into spoken style AND respell
        English words phonetically for Indonesian TTS — both in one LLM call to
        keep latency manageable.

        Triggered when the input contains either English markers OR formal
        Indonesian markers. Plain casual Indonesian passes through untouched.

        Adds ~200-400ms but only when needed.
        """
        if not text:
            return text
        has_english = bool(_LIKELY_ENGLISH.search(text))
        has_formal = bool(_FORMAL_ID.search(text))
        if not (has_english or has_formal):
            return text  # already casual + Indonesian-only — nothing to do

        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    "Kamu adalah editor teks untuk text-to-speech Bahasa Indonesia. "
                    "Tugasmu: rewrite teks input agar terdengar NATURAL diucapkan, dengan DUA langkah:\n\n"
                    "1. NATURALISASI: ubah kata/struktur tulis-formal menjadi gaya LISAN sehari-hari.\n"
                    "   - 'tidak' -> 'nggak' atau 'gak'\n"
                    "   - 'saya' -> 'aku' (kecuali konteks sangat formal)\n"
                    "   - 'apabila' / 'jika' -> 'kalau'\n"
                    "   - 'bagaimana' -> 'gimana'\n"
                    "   - 'sudah' -> 'udah'\n"
                    "   - 'akan' -> 'bakal' atau dihilangkan\n"
                    "   - 'merupakan' -> 'adalah' atau dihilangkan ('X merupakan Y' -> 'X itu Y')\n"
                    "   - 'sehingga' -> 'jadi'\n"
                    "   - 'namun' -> 'tapi'\n"
                    "   - 'tetapi' -> 'tapi'\n"
                    "   - 'sangat' -> 'banget' (di akhir frasa) atau dihilangkan\n"
                    "   - Pecah kalimat panjang jadi 2-3 kalimat pendek kalau perlu, biar mengalir.\n"
                    "   - JANGAN menambah informasi baru. JANGAN menghilangkan fakta penting.\n"
                    "   - Pertahankan nama orang, tempat, istilah teknis.\n\n"
                    "2. RESPELL INGGRIS: kata Inggris diubah ke ejaan fonetik Indonesia "
                    "supaya pembaca Indonesia mengucapkannya seperti bunyi Inggris aslinya.\n"
                    "   - 'machine learning' -> 'masyin lerning'\n"
                    "   - 'You Only Look Once' -> 'Yu Onli Luk Wans'\n"
                    "   - 'deep learning' -> 'diip lerning'\n"
                    "   - 'object detection' -> 'obyek diteksyen'\n"
                    "   - 'computer vision' -> 'kompyuter visyen'\n"
                    "   - 'AI' -> 'ei ai', 'API' -> 'ei pi ai'\n"
                    "   - Nama produk/orang (Pointer, YOLO, Polinela) -> biarkan apa adanya.\n\n"
                    "OUTPUT: HANYA teks hasil rewrite, tanpa penjelasan, tanpa label, tanpa tanda kutip.\n\n"
                    "Contoh lengkap:\n"
                    "INPUT: 'Saya akan menjelaskan apa itu machine learning. Machine learning merupakan teknik yang sangat populer.'\n"
                    "OUTPUT: 'Aku jelasin ya apa itu masyin lerning. Masyin lerning itu teknik yang lagi populer banget.'\n\n"
                    "INPUT: 'Apabila kamu tidak mengerti, silakan bertanya kembali.'\n"
                    "OUTPUT: 'Kalau kamu nggak ngerti, tanya aja lagi.'\n\n"
                    "INPUT: 'aku suka kopi'\n"
                    "OUTPUT: 'aku suka kopi'  (sudah natural, biarkan)"
                ),
            },
            {"role": "user", "content": text},
        ]
        try:
            raw = await self._llm.generate(messages, temperature=0.2, max_tokens=400)
        except Exception as exc:
            logger.warning(f"_prepare_for_speech failed, using original text: {exc}")
            return text
        out = raw.strip().strip("\"'")
        if not out:
            return text
        return out

    async def _evaluate_search_sufficiency(
        self, user_message: str, search_context: str
    ) -> str | None:
        """After an initial search, decide whether the result is enough to
        answer the user. If not, return a refined follow-up query.

        Return value:
            None             → result is sufficient, proceed to answer
            <query string>   → run another search with this refined query
        """
        # Skip eval for very long contexts (likely already comprehensive)
        if len(search_context) > 8000:
            return None
        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    "Kamu adalah evaluator hasil pencarian web. Tugasmu:\n"
                    "Lihat (a) pertanyaan user dan (b) hasil pencarian yang sudah ada.\n"
                    "Tentukan: apakah hasil ini SUDAH CUKUP untuk menjawab user secara faktual?\n\n"
                    "Jawab PERSIS satu baris, salah satu format:\n"
                    "  OK\n"
                    "  RETRY: <query baru>\n\n"
                    "Pakai RETRY hanya kalau hasil:\n"
                    "- Tidak ada/error/kosong\n"
                    "- Topiknya berbeda dari pertanyaan user (off-target)\n"
                    "- Cuma menyentuh permukaan, padahal user minta detail spesifik\n\n"
                    "Pakai OK kalau hasil sudah memuat fakta yang user butuhkan, walau pendek.\n"
                    "JANGAN RETRY hanya karena ingin info lebih banyak — hanya kalau hasil benar-benar tidak menjawab.\n"
                    "Query RETRY harus SELF-CONTAINED dan spesifik."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PERTANYAAN USER:\n{user_message}\n\n"
                    f"HASIL PENCARIAN:\n{search_context[:6000]}"
                ),
            },
        ]
        try:
            raw = await self._llm.generate(messages, temperature=0.0, max_tokens=64)
        except Exception as exc:
            logger.warning(f"_evaluate_search_sufficiency failed: {exc}")
            return None
        line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        if line.upper().startswith("OK"):
            return None
        if not line.upper().startswith("RETRY"):
            return None
        after = line.split(":", 1)[1] if ":" in line else line.split(" ", 1)[1] if " " in line else ""
        query = after.strip().strip('"').strip("'")
        if not query or len(query) > 200:
            return None
        return query

    async def _classify_search_need(
        self, user_text: str, history: list[ChatMessage] | None = None
    ) -> str | None:
        """Ask the LLM whether this user message needs a web search.

        Returns the search query string if yes, else None. Uses a tight
        instruction format (output must be 'SEARCH: <query>' or 'NO') so we
        don't have to deal with structured tool-calling on Llama (which is
        flaky on Groq — sometimes the model emits the call as raw text).

        When `history` is provided, the classifier sees the last few turns so
        anaphoric/elliptical follow-ups ("sejarah-nya gimana?", "YOLO?") get
        rewritten into self-contained queries like
        "sejarah program studi sains data terapan Polinela".
        """
        if not user_text.strip():
            return None

        # Build a short context summary of the last ~4 turns
        context_lines: list[str] = []
        if history:
            for turn in history[-4:]:
                role = turn.get("role", "?")
                content = (turn.get("content") or "").strip()
                if not content:
                    continue
                if len(content) > 200:
                    content = content[:197] + "…"
                speaker = "User" if role == "user" else "Pointer"
                context_lines.append(f"{speaker}: {content}")
        context_block = ""
        if context_lines:
            context_block = (
                "\nKONTEKS PERCAKAPAN (jangan jawab ini, hanya untuk rewrite query):\n"
                + "\n".join(context_lines)
                + "\n"
            )

        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    "Tugasmu: tentukan apakah pesan user terakhir butuh pencarian web "
                    "untuk dijawab dengan benar.\n"
                    "Jawab PERSIS satu baris, salah satu format ini, tanpa apapun lain:\n"
                    "  SEARCH: <query>\n"
                    "  NO\n\n"
                    "PENTING — query harus SELF-CONTAINED. Pakai konteks percakapan untuk:\n"
                    "- Mengganti kata ganti ('itu', 'nya', 'tersebut') dengan subjek aslinya\n"
                    "- Tambah nama institusi/lokasi yang sedang dibahas (mis. nama kampus)\n"
                    "- Lengkapi follow-up: 'sejarahnya?' -> 'sejarah X' (X dari konteks)\n\n"
                    "Pakai SEARCH untuk:\n"
                    "- Pertanyaan tentang fakta aktual / terkini (berita, harga, cuaca, kurs)\n"
                    "- Tokoh/organisasi/produk yang tidak umum diketahui\n"
                    "- 'apa itu X' untuk istilah yang kamu tidak yakin\n"
                    "- Event / jadwal / pengumuman publik\n"
                    "- Pertanyaan 'siapa', 'kapan', 'di mana', 'berapa' yang butuh fakta\n"
                    "- Detail spesifik institusi/program (jumlah prodi, tahun berdiri, dll)\n\n"
                    "Pakai NO untuk:\n"
                    "- Sapaan, basa-basi, chit-chat\n"
                    "- Pertanyaan tentang dirimu/sifat/persona\n"
                    "- Perintah perilaku ('panggil aku X', 'jangan begitu')\n"
                    "- Pertanyaan umum yang bisa dijawab dari pengetahuan dasar\n"
                    "- Klarifikasi pesan ('apa maksudmu?')\n\n"
                    "Contoh:\n"
                    "- 'halo apa kabar' -> NO\n"
                    "- 'siapa presiden indonesia sekarang' -> SEARCH: presiden indonesia 2026\n"
                    "- (Konteks: kampus Polinela) + 'jumlah prodi?' -> SEARCH: jumlah program studi Polinela Politeknik Negeri Lampung\n"
                    "- (Konteks: prodi Sains Data Terapan Polinela) + 'sejarahnya?' -> SEARCH: sejarah program studi Sains Data Terapan Polinela\n"
                    "- (Konteks: deteksi objek) + 'YOLO?' -> SEARCH: YOLO object detection model\n"
                    "- 'jam berapa sekarang' -> NO\n"
                    "- 'ganti namaku jadi reza' -> NO\n"
                    "- 'apa maksudmu?' -> NO"
                    + context_block
                ),
            },
            {"role": "user", "content": user_text},
        ]
        try:
            raw = await self._llm.generate(messages, temperature=0.0, max_tokens=48)
        except Exception as exc:
            logger.warning(f"_classify_search_need LLM failed: {exc}")
            return None
        line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        if line.upper() == "NO" or line.upper().startswith("NO"):
            return None
        if not line.upper().startswith("SEARCH"):
            return None
        # Accept both 'SEARCH: <q>' and 'SEARCH <q>'
        after = line.split(":", 1)[1] if ":" in line else line.split(" ", 1)[1] if " " in line else ""
        query = after.strip().strip('"').strip("'")
        if not query or len(query) > 200:
            return None
        return query

    async def _execute_tool(self, name: str, args_json: str) -> str:
        """Execute a tool call from the LLM. Returns text the LLM will see next."""
        import json

        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as exc:
            return f"TOOL_ERROR: invalid arguments JSON: {exc}"

        if name == "search_web":
            query = args.get("query", "").strip()
            if not query:
                return "TOOL_ERROR: missing query"
            try:
                return await get_search().search(query)
            except Exception as exc:
                logger.exception(f"search_web tool failed: {exc}")
                return f"TOOL_ERROR: {exc}"

        return f"TOOL_ERROR: unknown tool '{name}'"

    def _build_messages_with_session(
        self, persona: Persona, session: SessionState
    ) -> list[ChatMessage]:
        """Same as _build_messages but injects engaged-person + realtime context."""
        content = persona.render_system_prompt()
        # Inject current time — LLM has no realtime clock, so tell it explicitly
        # each request. Without this, 'jam berapa?' yields hallucinated answers.
        content += f"\n\nWAKTU SEKARANG: {_now_indonesian()}"
        if session.last_engaged_name:
            content += (
                f"\n\nKONTEKS PERCAKAPAN INI:\n"
                f"Kamu sedang ngobrol dengan {session.last_engaged_name}. "
                f"Sapa dengan namanya saat natural saja, tidak setiap pesan."
            )
        system: ChatMessage = {"role": "system", "content": content}
        return [system, *session.history[-(settings.memory_max_turns * 2) :]]

    # ─── Canned-speech path (R4) ────────────────────────────────────────────
    # Bypasses the LLM for the simple greet/ask-name lines so we avoid extra
    # latency + LLM creative reinterpretation. Uses the same TTS pipeline so
    # the audio + viseme + lip-sync work identically to LLM-generated speech.

    async def speak(
        self, session: SessionState, text: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Render `text` as if Pointer said it: split into sentences, TTS each
        in parallel, emit ai_text + audio + viseme events in order."""
        t0 = time.perf_counter()
        persona = persona_manager.get(session.persona_id)

        sentences: list[str] = []
        buf = ""
        for ch in text:
            buf += ch
            if ch in ".!?…":
                s = buf.strip()
                if s:
                    sentences.append(s)
                buf = ""
        if buf.strip():
            sentences.append(buf.strip())
        if not sentences:
            yield {"type": "done", "reason": "empty_text"}
            return

        async def render_sentence(
            sentence: str, idx: int, seq: int
        ) -> dict[str, Any]:
            tts_t0 = time.perf_counter()
            # Naturalize formal Indonesian -> spoken style + respell English
            # phonetically. UI sees the LLM's original text; TTS sees the
            # rewritten version. Adds latency only when needed (heuristic skip).
            spoken = await self._prepare_for_speech(sentence)
            if spoken != sentence:
                logger.debug(f"[speech-prep] {sentence[:60]!r} -> {spoken[:60]!r}")
            result = await self._tts.synthesize(
                spoken,
                voice=persona.voice.voice_id,
                rate=persona.voice.rate,
                pitch=persona.voice.pitch,
            )
            tts_ms = int((time.perf_counter() - tts_t0) * 1000)
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

        tasks = [
            asyncio.create_task(render_sentence(s, i, i))
            for i, s in enumerate(sentences)
        ]

        # Emit ai_text + audio + viseme for each sentence in order
        for i, (s, task) in enumerate(zip(sentences, tasks, strict=True)):
            yield {
                "type": "ai_text",
                "text": s,
                "is_final": False,
                "sentence_idx": i,
            }
            try:
                payload = await task
            except Exception as exc:
                logger.exception("TTS task failed")
                yield {"type": "error", "message": f"TTS failed: {exc}"}
                continue
            yield payload["viseme_event"]
            yield payload["audio_event"]

        full = " ".join(sentences)
        session.append("assistant", full)
        try:
            await get_chat_history().append(
                session.engaged_person_id, "assistant", full
            )
        except Exception as exc:
            logger.warning(f"chat_history append(assistant/speak) failed: {exc}")
        total_ms = int((time.perf_counter() - t0) * 1000)
        yield {
            "type": "ai_text",
            "text": full,
            "is_final": True,
            "sentence_idx": len(sentences),
        }
        yield {
            "type": "timing",
            "first_token_ms": None,
            "first_audio_ms": None,
            "total_ms": total_ms,
            "sentences": len(sentences),
        }
        yield {"type": "done"}

    async def greet_known(
        self, session: SessionState, person_name: str
    ) -> AsyncIterator[dict[str, Any]]:
        line = f"Halo {person_name}, ada yang bisa aku bantu?"
        logger.info(f"[face] greet_known('{person_name}')")
        async for ev in self.speak(session, line):
            yield ev

    async def ask_name(
        self, session: SessionState
    ) -> AsyncIterator[dict[str, Any]]:
        line = "Halo, sepertinya kita belum berkenalan. Boleh aku tau namamu?"
        logger.info("[face] ask_name()")
        async for ev in self.speak(session, line):
            yield ev

    async def confirm_enrolled(
        self, session: SessionState, person_name: str
    ) -> AsyncIterator[dict[str, Any]]:
        line = f"Senang berkenalan, {person_name}."
        logger.info(f"[face] confirm_enrolled('{person_name}')")
        async for ev in self.speak(session, line):
            yield ev

    async def extract_name(self, user_text: str) -> str | None:
        """Use the LLM to pull a clean name from a free-form response.

        Returns the name string, or None if the response doesn't contain one.
        """
        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    "Tugasmu: dari teks user yang membalas pertanyaan 'siapa namamu?', "
                    "ekstrak HANYA namanya. Format keluaran: nama saja, tanpa kalimat, "
                    "tanpa tanda kutip. Kapitalisasi yang benar (mis. 'Rezar Aulia').\n"
                    "Kalau teks tidak mengandung nama yang jelas, jawab: UNCLEAR\n"
                    "Contoh:\n"
                    "- 'Aku Reza' → Reza\n"
                    "- 'Nama saya Rezar Aulia' → Rezar Aulia\n"
                    "- 'panggil aja reza aja' → Reza\n"
                    "- 'I am John' → John\n"
                    "- 'halo apa kabar' → UNCLEAR"
                ),
            },
            {"role": "user", "content": user_text},
        ]
        try:
            raw = await self._llm.generate(messages, temperature=0.2, max_tokens=24)
        except Exception as exc:
            logger.warning(f"extract_name LLM call failed: {exc}")
            return None
        name = raw.strip().strip("\"'.").splitlines()[0].strip()
        if not name or name.upper() == "UNCLEAR":
            return None
        if len(name) > 64 or "\n" in name:
            return None
        return name

    async def detect_correction(
        self, user_text: str, current_name: str
    ) -> str | None:
        """If the user is correcting their name, return the new name; else None.

        Two-stage to avoid the LLM cost on every chat message:
        1. Cheap regex pre-filter for correction keywords.
        2. LLM classifier only when the pre-filter hits.
        """
        if not _CORRECTION_KEYWORDS.search(user_text):
            return None
        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    "User berbicara dengan asisten kampus. Asisten baru saja menyimpan "
                    f"nama user sebagai '{current_name}'.\n"
                    "Tugasmu: tentukan apakah teks user ini adalah KOREKSI NAMA "
                    "(meminta asisten mengganti nama tersimpan), atau bukan.\n"
                    "Format keluaran (PERSIS satu baris, tanpa apapun lain):\n"
                    "- 'CORRECT: <NamaBaru>' jika koreksi DAN nama baru jelas\n"
                    "- 'CORRECT: ?'        jika koreksi tapi nama baru belum jelas\n"
                    "- 'NO'                jika bukan koreksi (chat biasa)\n"
                    "Contoh:\n"
                    "- 'nama saya salah, aku Reza' → CORRECT: Reza\n"
                    "- 'bukan, namaku Rezar Aulia' → CORRECT: Rezar Aulia\n"
                    "- 'panggil saja Andi'         → CORRECT: Andi\n"
                    "- 'nama saya salah'           → CORRECT: ?\n"
                    "- 'bagaimana cuaca hari ini?' → NO\n"
                    "- 'halo apa kabar Tiago'      → NO"
                ),
            },
            {"role": "user", "content": user_text},
        ]
        try:
            raw = await self._llm.generate(messages, temperature=0.1, max_tokens=32)
        except Exception as exc:
            logger.warning(f"detect_correction LLM call failed: {exc}")
            return None
        line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        if line.upper().startswith("NO") or not line:
            return None
        if not line.upper().startswith("CORRECT:"):
            return None
        new_name = line.split(":", 1)[1].strip().strip("\"'.")
        if not new_name or new_name == "?":
            return None
        if len(new_name) > 64 or "\n" in new_name:
            return None
        return new_name

    async def confirm_correction(
        self, session: SessionState, old_name: str, new_name: str
    ) -> AsyncIterator[dict[str, Any]]:
        line = (
            f"Oh, maaf salah catat. Sudah aku perbaiki — sekarang aku ingat kamu sebagai {new_name}."
        )
        logger.info(f"[face] confirm_correction('{old_name}' -> '{new_name}')")
        async for ev in self.speak(session, line):
            yield ev


# Cheap keyword filter: skip the LLM correction-detection call entirely unless
# the user's text contains at least one of these tokens. Tuned for Bahasa Indonesia
# with a few English variants ('wrong', 'not', 'call me').
_CORRECTION_KEYWORDS = re.compile(
    r"\b(salah|bukan|ganti|perbaiki|ralat|koreksi|panggil|wrong|not\s+\w+|call\s+me)\b",
    re.IGNORECASE,
)


_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
