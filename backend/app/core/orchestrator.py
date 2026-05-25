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
import random
import time
from collections import deque
from collections.abc import AsyncIterator
from typing import Any, Literal

from loguru import logger

from app.config import settings
from app.core.persona import Persona, persona_manager
from app.core.viseme import boundaries_to_visemes
from app.agents.context_builder_agent import get_context_builder_agent
from app.agents.history_hydration_agent import HistoryHydrationAgent
from app.agents.light_chat_agent import get_light_chat_agent
from app.agents.llm_stream_agent import get_llm_stream_agent
from app.agents.search_agent import SearchAgent, SearchResult
from app.agents.speech_prep_agent import get_speech_prep_agent
from app.agents.stt_agent import get_stt_agent
from app.services.chat_history import get_chat_history
from app.services.llm_groq import ChatMessage, get_llm
from app.services.tts_edge import get_tts

Mode = Literal["companion", "customer_service"]

# Phrases Pointer speaks the moment a web search is triggered — user hears an
# immediate acknowledgement instead of silence while search+TTS spin up.
_SEARCH_PHRASES = (
    "Oke, tunggu sebentar ya, aku coba cari dulu.",
    "Hmm, biar aku cek dulu infonya ya.",
    "Sebentar ya, aku cari dulu.",
    "Boleh aku cek sebentar? Satu momen ya.",
    "Oke, aku cari dulu, tunggu ya.",
    "Wah menarik nih! Bentar, aku googling dulu.",
    "Hmm, aku kurang hafal detailnya — cek dulu ya.",
    "Satu momen ya, biar aku pastiin dulu infonya.",
    "Bentar, aku cari info yang akurat dulu biar nggak salah.",
    "Oh menarik! Aku lihat dulu supaya jawabannya tepat ya.",
    "Sabar ya, aku cek dulu biar beneran akurat nih.",
    "Hmm, biar aku cari yang terkini — tunggu sebentar ya.",
)

# Short filler phrases Pointer speaks if the search is still running after the
# initial hold-on phrase plays out. Keeps conversation alive during slow searches.
_SEARCH_FILLER_PHRASES = (
    "Hampir ketemu nih, bentar lagi.",
    "Lagi aku baca hasilnya...",
    "Wah banyak infonya, aku ambil yang paling relevan ya.",
    "Dikit lagi ya, hampir selesai nih.",
    "Hmm, aku sortir dulu biar jawabannya pas.",
)


def _pick_search_phrase() -> str:
    return random.choice(_SEARCH_PHRASES)



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
        # Light chat: queue of messages received while main pipeline is busy.
        # Processed one-by-one after each filler phrase in _yield_fillers_while_waiting.
        self.pending_light_chat_queue: deque[str] = deque()
        # Conversation history kept alive during one wait window so back-and-forth
        # stays coherent. Cleared by orchestrator after the main answer is delivered.
        self.light_chat_history: list[dict] = []
        # Pipeline busy flag — set by websocket.py when a pipeline task is running.
        # Used by face_engagement_agent to skip proactive probes while a turn is active.
        self.pipeline_busy: bool = False
        # Periodic expression check — last time mood was probed and the last mood seen.
        # Initialised to 0 so the first face_present after greet_known starts the clock.
        self.last_expression_check_at: float = 0.0
        self.last_known_mood: str = "neutral"

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
        self._stt_agent = get_stt_agent()
        self._llm = get_llm()
        self._tts = get_tts()
        self._search_agent = SearchAgent(self._llm)
        self._light_chat_agent = get_light_chat_agent()
        self._hydration_agent = HistoryHydrationAgent()
        self._speech_prep = get_speech_prep_agent()
        self._llm_stream_agent = get_llm_stream_agent()

    def _persist(self, person_id: int | None, role: str, content: str) -> None:
        """Fire-and-forget chat history write — never blocks the streaming path."""
        async def _write() -> None:
            try:
                await get_chat_history().append(person_id, role, content)
            except Exception as exc:
                logger.warning(f"chat_history append({role}) failed: {exc}")
        asyncio.create_task(_write())

    async def handle_audio(
        self,
        session: SessionState,
        audio: bytes,
        *,
        audio_format: str = "webm",
    ) -> AsyncIterator[dict[str, Any]]:
        t0 = time.perf_counter()
        try:
            transcript = await self._stt_agent.transcribe(
                audio,
                format=audio_format,
                session=session,
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
        self._persist(session.engaged_person_id, "user", message)

        # ── Classify search need, then run search + hydration in parallel ──
        # Hydration is a catch-up path: normally history is loaded when a face
        # is first recognised. This branch handles the race where the user
        # speaks before the face_present event fires.
        needs_hydration = (
            session.engaged_person_id is not None
            and session.engaged_person_id not in session.greeted_person_ids
        )

        search_query: str | None = None
        if settings.tavily_api_key:
            search_query = await self._search_agent.classify(
                message, history=session.history[:-1]
            )

        search_result: SearchResult | None = None
        hydrated_turns: list = []

        if search_query:
            # Kick off search + hydration as background tasks, then speak the
            # hold-on phrase while they run — search latency is largely hidden
            # behind the TTS playback time.
            search_task = asyncio.create_task(
                self._search_agent.run_with_query(search_query, message)
            )
            hydration_task = (
                asyncio.create_task(self._hydration_agent.run(session.engaged_person_id))
                if needs_hydration else None
            )
            # Only forward audio-production events from the hold-on phrase.
            # Suppress terminal events so the frontend state machine doesn't
            # think the turn is complete while search is still running.
            async for ev in self.speak(session, _pick_search_phrase(), ephemeral=True):
                ev_type = ev.get("type")
                if ev_type in ("done", "timing"):
                    continue
                if ev_type == "ai_text" and ev.get("is_final"):
                    continue
                yield ev
            # Speak filler phrases if search takes too long after the hold-on
            # phrase finishes, then collect the result.
            try:
                async for ev in self._yield_fillers_while_waiting(session, search_task):
                    yield ev
                # Safety net: filler loop may exit while task is still running
                try:
                    search_result = await asyncio.wait_for(
                        search_task, timeout=settings.search_timeout_s
                    )
                except asyncio.TimeoutError:
                    logger.warning("[handle_text] search_task hung, cancelling")
                    search_task.cancel()
                    search_result = None
            except asyncio.CancelledError:
                search_task.cancel()
                if hydration_task is not None:
                    hydration_task.cancel()
                raise
            if hydration_task is not None:
                hydrated_turns = await hydration_task
        elif needs_hydration:
            hydrated_turns = await self._hydration_agent.run(session.engaged_person_id)

        if hydrated_turns and session.engaged_person_id is not None:
            # Prepend past turns before the current user message.
            # Clamp to memory_max_turns * 2 so hydration can't overflow the cap.
            current_msg = session.history[-1]
            max_msgs = settings.memory_max_turns * 2
            session.history = (hydrated_turns + [current_msg])[-max_msgs:]
            session.greeted_person_ids.add(session.engaged_person_id)
            logger.info(
                f"[handle_text] catch-up hydrated {len(hydrated_turns)} turns "
                f"for person_id={session.engaged_person_id}"
            )

        search_context: str | None = None
        if search_result:
            for ev in search_result["ws_events"]:
                yield ev
            search_context = search_result["context"]

        # ── Knowledge base retrieval (NVIDIA NIM RAG) ───────────────────────
        # Runs only when NVIDIA_API_KEY is set and the KB has documents indexed.
        kb_context: str | None = None
        if settings.nvidia_api_key:
            try:
                from app.services.knowledge_base import get_knowledge_base
                kb_context = await get_knowledge_base().build_context(message)
            except Exception as exc:
                logger.debug(f"[KB] query skipped: {exc}")

        messages = get_context_builder_agent().build_with_session(persona, session)
        if kb_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Informasi relevan dari knowledge base Pointer:\n\n"
                        f"{kb_context}\n\n"
                        "Gunakan informasi ini jika relevan dengan pertanyaan user. "
                        "Jika tidak relevan, abaikan dan jawab dari pengetahuanmu sendiri."
                    ),
                }
            )
        if search_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Hasil pencarian web untuk pertanyaan user:\n\n{search_context}\n\n"
                        "Gunakan info di atas untuk jawab pertanyaan user secara ringkas dan natural "
                        "dalam bahasa Indonesia. JANGAN tempel URL/link mentah-mentah ke jawaban suara. "
                        "Kalau hasil kurang relevan, bilang saja kamu tidak yakin.\n\n"
                        "PENTING: kamu tadi sudah bilang ke user bahwa kamu sedang mencari. "
                        "Awali jawabanmu dengan kalimat natural singkat yang menandakan kamu sudah "
                        "selesai mencari — misalnya 'Oke, udah dapat nih—', 'Nah ini yang aku "
                        "temukan—', atau 'Sudah ketemu, jadi—'. Lanjutkan langsung ke isi jawaban."
                    ),
                }
            )

        # Drain light-chat queue before LLM stream starts.
        # Covers: (1) fast search that exits filler loop before any filler fires,
        # (2) non-search path that never enters the filler loop at all.
        async for ev in self.handle_light_chat_if_pending(session):
            yield ev

        async for ev in self._llm_stream_agent.run(messages, persona, t0):
            if ev.get("type") == "done":
                full_reply = ev.get("full_reply", "")
                if full_reply:
                    session.append("assistant", full_reply)
                    self._persist(session.engaged_person_id, "assistant", full_reply)
                # Main answer delivered — reset light chat state for next turn
                session.light_chat_history.clear()
                session.pending_light_chat_queue.clear()
                yield {"type": "done"}
            else:
                yield ev

    async def handle_light_chat_if_pending(
        self,
        session: SessionState,
    ) -> AsyncIterator[dict[str, Any]]:
        """Process all queued light-chat messages, generating a brief reply for each.

        Called between filler phrases so users are acknowledged while the main
        search/pipeline runs in the background. Maintains conversation history
        within the wait window so back-and-forth feels natural. Replies are
        ephemeral — not added to the main conversation history.
        """
        if not session.pending_light_chat_queue:
            return

        # Main question being processed (last user turn in main history)
        main_question: str | None = None
        for turn in reversed(session.history):
            if turn.get("role") == "user":
                main_question = turn.get("content") or None
                break

        # Offset sequences so light-chat audio never collides with the main
        # pipeline's 0-based sequence counter (viseme buffer key conflicts
        # cause wrong lip-sync).
        SEQ_OFFSET = 10_000
        lc_idx = 0
        while session.pending_light_chat_queue:
            message = session.pending_light_chat_queue.popleft()
            reply = await self._light_chat_agent.generate(
                message, main_question, session.light_chat_history
            )
            base = SEQ_OFFSET + lc_idx * 100
            lc_idx += 1
            async for ev in self.speak(session, reply, ephemeral=True):
                ev_type = ev.get("type")
                if ev_type in ("done", "timing"):
                    continue
                # Let is_final:True flow through so the frontend can finalize
                # the light-chat bubble and reset its ref for the main answer.
                if ev_type == "audio":
                    ev = {**ev, "sequence": ev["sequence"] + base}
                elif ev_type == "viseme":
                    ev = {**ev, "audio_seq": ev["audio_seq"] + base}
                yield ev

    async def _yield_fillers_while_waiting(
        self,
        session: SessionState,
        task: asyncio.Task,
    ) -> AsyncIterator[dict[str, Any]]:
        """Speak short filler phrases while `task` (search) is still running.

        Uses asyncio.shield so the original task is never cancelled by the
        wait_for timeout. Yields audio/viseme events just like the hold-on
        phrase, suppressing terminal events from each filler speak().

        After each filler, any pending light-chat message is handled so the
        user feels heard while the search runs in the background.
        """
        fillers = list(_SEARCH_FILLER_PHRASES)
        random.shuffle(fillers)
        idx = 0
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=1.5)
                break  # task finished within timeout window
            except asyncio.TimeoutError:
                if idx >= len(fillers):
                    # Out of fillers; handle any pending light chat then wait
                    async for ev in self.handle_light_chat_if_pending(session):
                        yield ev
                    try:
                        await asyncio.wait_for(asyncio.shield(task), timeout=10.0)
                    except asyncio.TimeoutError:
                        pass
                    break
                filler = fillers[idx]
                idx += 1
                async for ev in self.speak(session, filler, ephemeral=True):
                    ev_type = ev.get("type")
                    if ev_type in ("done", "timing"):
                        continue
                    if ev_type == "ai_text" and ev.get("is_final"):
                        continue
                    yield ev
                # After each filler, acknowledge any queued light-chat message
                async for ev in self.handle_light_chat_if_pending(session):
                    yield ev
            except asyncio.CancelledError:
                raise

    # ─── Canned-speech path (R4) ────────────────────────────────────────────
    # Bypasses the LLM for the simple greet/ask-name lines so we avoid extra
    # latency + LLM creative reinterpretation. Uses the same TTS pipeline so
    # the audio + viseme + lip-sync work identically to LLM-generated speech.

    async def speak(
        self, session: SessionState, text: str, *, ephemeral: bool = False
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

        # Batch-prep all sentences in one LLM call (all text is known upfront
        # here, unlike handle_text where sentences arrive one-by-one).
        prepped = await self._speech_prep.prepare_batch(sentences)
        for orig, spk in zip(sentences, prepped):
            if spk != orig:
                logger.debug(f"[speech-prep] {orig[:60]!r} -> {spk[:60]!r}")

        async def render_sentence(
            spoken: str, idx: int, seq: int
        ) -> dict[str, Any]:
            tts_t0 = time.perf_counter()
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
            asyncio.create_task(render_sentence(prepped[i], i, i))
            for i in range(len(sentences))
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
        if not ephemeral:
            session.append("assistant", full)
            self._persist(session.engaged_person_id, "assistant", full)
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
        self, session: SessionState, person_name: str, mood: str = "neutral"
    ) -> AsyncIterator[dict[str, Any]]:
        _greet_lines: dict[str, str] = {
            "happy":    f"Halo {person_name}! Kamu terlihat semangat hari ini, ada yang bisa aku bantu?",
            "focused":  f"Halo {person_name}, sepertinya sedang sibuk. Ada yang perlu aku bantu?",
            "confused": f"Halo {person_name}! Sepertinya ada yang membingungkan? Aku siap membantu.",
            "tired":    f"Halo {person_name}. Semoga harimu baik-baik saja, ada yang bisa aku bantu?",
            "neutral":  f"Halo {person_name}, ada yang bisa aku bantu?",
        }
        line = _greet_lines.get(mood, _greet_lines["neutral"])
        logger.info(f"[face] greet_known('{person_name}', mood='{mood}')")
        async for ev in self.speak(session, line):
            yield ev

    async def ask_name(
        self, session: SessionState, mood: str = "neutral"
    ) -> AsyncIterator[dict[str, Any]]:
        _ask_lines: dict[str, str] = {
            "happy":    "Halo! Senang melihatmu di sini! Sepertinya kita belum berkenalan, boleh aku tau namamu?",
            "focused":  "Halo, ada yang bisa aku bantu? Oh ya, boleh aku tau namamu dulu?",
            "confused": "Halo! Kelihatannya ada yang membingungkan? Boleh aku tau namamu dulu supaya aku bisa bantu lebih baik?",
            "tired":    "Halo, sepertinya hari yang panjang ya. Boleh aku tau namamu?",
            "neutral":  "Halo, sepertinya kita belum berkenalan. Boleh aku tau namamu?",
        }
        line = _ask_lines.get(mood, _ask_lines["neutral"])
        logger.info(f"[face] ask_name(mood='{mood}')")
        async for ev in self.speak(session, line):
            yield ev

    async def confirm_enrolled(
        self, session: SessionState, person_name: str
    ) -> AsyncIterator[dict[str, Any]]:
        line = f"Senang berkenalan, {person_name}."
        logger.info(f"[face] confirm_enrolled('{person_name}')")
        async for ev in self.speak(session, line):
            yield ev

    async def confirm_correction(
        self, session: SessionState, old_name: str, new_name: str
    ) -> AsyncIterator[dict[str, Any]]:
        line = (
            f"Oh, maaf salah catat. Sudah aku perbaiki — sekarang aku ingat kamu sebagai {new_name}."
        )
        logger.info(f"[face] confirm_correction('{old_name}' -> '{new_name}')")
        async for ev in self.speak(session, line):
            yield ev

    async def probe_mood(
        self, session: SessionState, person_name: str, mood: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Proactively comment when a known user's mood shifts to something noteworthy."""
        _mood_lines: dict[str, str] = {
            "tired":    f"{person_name}, kamu kelihatan lelah. Mau istirahat sebentar, atau ada yang bisa aku bantu biar lebih ringan?",
            "confused": f"Hei {person_name}, sepertinya ada yang bikin bingung? Cerita aja, aku siap bantu.",
            "happy":    f"Wah, kamu kelihatan happy nih {person_name}! Ada kabar baik yang mau diceritain?",
            "focused":  f"Tetap semangat ya {person_name}, aku di sini kalau butuh bantuan.",
        }
        line = _mood_lines.get(mood)
        if not line:
            return
        logger.info(f"[face] probe_mood('{person_name}', mood='{mood}')")
        async for ev in self.speak(session, line, ephemeral=True):
            yield ev


_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
