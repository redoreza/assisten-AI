"""NameExtractionAgent — pull a clean name from a free-form enrollment reply.

Called once when `session.awaiting_name` is True and the user has responded
to "siapa namamu?". Returns the extracted name or None if the reply is
unclear.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from app.services.llm_groq import ChatMessage


class NameExtractionAgent:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def run(self, user_text: str) -> str | None:
        """Extract a name from a free-form reply. Returns None if unclear."""
        if not user_text.strip():
            return None
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
            logger.warning(f"[NameExtractionAgent] LLM call failed: {exc}")
            return None
        name = raw.strip().strip("\"'.").splitlines()[0].strip()
        if not name or name.upper() == "UNCLEAR":
            return None
        if len(name) > 64 or "\n" in name:
            return None
        return name


_singleton: NameExtractionAgent | None = None


def get_name_extraction_agent() -> NameExtractionAgent:
    global _singleton
    if _singleton is None:
        from app.services.llm_groq import get_llm
        _singleton = NameExtractionAgent(get_llm())
    return _singleton
