"""LightChatAgent — interactive brief responses while the main pipeline is busy.

Runs on a DEDICATED fast LLM (llama-3.1-8b-instant on Groq by default) so it
does not consume the main pipeline's tokens-per-day quota.

Key design:
  • Own GroqLLM instance → separate quota, faster model
  • history list is passed in and mutated in-place so back-and-forth conversation
    stays coherent across multiple user messages during a single wait window
  • Hard limit: max 2 sentences per reply (enforced post-generation)

Rules enforced via system prompt:
  BOLEH  : basa-basi, konfirmasi pertanyaan, estimasi tunggu, fakta umum
  DILARANG: pencarian baru, respons > 2 kalimat, data real-time
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.config import settings

_SYSTEM = (
    "Kamu adalah Pointer, asisten virtual kampus yang sedang sibuk memproses "
    "pertanyaan utama user di latar belakang.\n\n"
    "Sementara menunggu, kamu bisa MENGOBROL RINGAN dengan user.\n\n"
    "BOLEH:\n"
    "- Menjawab basa-basi / small talk secara natural\n"
    "- Mengkonfirmasi atau meringkas pertanyaan user\n"
    "- Memberikan estimasi waktu tunggu\n"
    "- Menjawab pertanyaan sederhana dari pengetahuan umum (tanpa internet)\n"
    "- Melanjutkan obrolan ringan dari percakapan sebelumnya dalam sesi ini\n\n"
    "DILARANG:\n"
    "- Menjawab lebih dari 2 kalimat\n"
    "- Memulai pencarian internet baru\n"
    "- Menjawab pertanyaan faktual terkini (berita / harga / cuaca / data real-time)\n"
    "  → Untuk ini, katakan 'nanti aku jawab sekalian ya' atau 'tunggu sebentar'\n\n"
    "Pertahankan nada hangat, santai, dan singkat."
)


class LightChatAgent:
    """Brief interactive responses during search/LLM waiting time.

    Uses a dedicated GroqLLM(llama-3.1-8b-instant) instance so it does not
    compete with the main pipeline's llama-3.3-70b-versatile TPD quota.
    """

    def __init__(self) -> None:
        self._llm = self._build_llm()

    def _build_llm(self) -> Any:
        """Dedicated fast LLM; falls back to main router on init failure."""
        try:
            from app.services.llm_nvidia import NvidiaLLM
            llm = NvidiaLLM(model=settings.light_chat_model)
            logger.info(f"[LightChatAgent] dedicated LLM: nvidia/{settings.light_chat_model}")
            return llm
        except Exception as exc:
            logger.warning(
                f"[LightChatAgent] dedicated LLM init failed ({exc}), "
                "falling back to main router"
            )
            from app.services.llm_router import get_router
            return get_router()

    async def generate(
        self,
        user_message: str,
        main_question: str | None,
        history: list[dict],
    ) -> str:
        """Generate a brief reply and append the exchange to `history` in-place.

        history: mutable list of {"role": str, "content": str} dicts stored in
                 SessionState.light_chat_history. Grows across turns so
                 back-and-forth stays coherent during one wait window.
                 Cleared by the orchestrator after the main answer is delivered.
        """
        context_note = ""
        if main_question:
            context_note = f"\n\nPertanyaan utama yang sedang diproses: '{main_question[:100]}'"

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM + context_note},
            *history,
            {"role": "user", "content": user_message},
        ]

        try:
            raw = await self._llm.generate(messages, temperature=0.75, max_tokens=80)
        except Exception as exc:
            logger.warning(f"[LightChatAgent] LLM failed: {exc}")
            raw = "Sebentar ya, hampir selesai nih!"

        reply = (raw or "").strip() or "Hampir selesai, tunggu ya!"

        # Hard cap: keep at most 2 sentences
        sentences: list[str] = []
        buf = ""
        for ch in reply:
            buf += ch
            if ch in ".!?…" and buf.strip():
                sentences.append(buf.strip())
                buf = ""
                if len(sentences) >= 2:
                    break
        if buf.strip() and len(sentences) < 2:
            sentences.append(buf.strip())
        reply = " ".join(sentences) if sentences else reply[:120]

        # Append exchange so next call has context
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})

        logger.info(f"[LightChatAgent] '{user_message[:50]}' → '{reply}'")
        return reply


_agent: LightChatAgent | None = None


def get_light_chat_agent() -> LightChatAgent:
    global _agent
    if _agent is None:
        _agent = LightChatAgent()
    return _agent
