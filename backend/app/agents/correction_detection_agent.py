"""CorrectionDetectionAgent — detect if the user is correcting their stored name.

Two-stage to minimise latency on every chat message:
  1. Cheap regex pre-filter (_CORRECTION_KEYWORDS) — skips the LLM entirely
     unless the message contains a correction-related keyword.
  2. LLM classifier only when the pre-filter fires — returns the new name or
     None.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from app.services.llm_groq import ChatMessage

# Cheap keyword filter. Tuned for Bahasa Indonesia with common English variants.
_CORRECTION_KEYWORDS = re.compile(
    r"\b(salah|bukan|ganti|perbaiki|ralat|koreksi|panggil|wrong|not\s+\w+|call\s+me)\b",
    re.IGNORECASE,
)


class CorrectionDetectionAgent:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def run(self, user_text: str, current_name: str) -> str | None:
        """Return the corrected name if the user is correcting theirs, else None.

        Skips the LLM call entirely when the regex gate does not fire.
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
            logger.warning(f"[CorrectionDetectionAgent] LLM call failed: {exc}")
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


_singleton: CorrectionDetectionAgent | None = None


def get_correction_detection_agent() -> CorrectionDetectionAgent:
    global _singleton
    if _singleton is None:
        from app.services.llm_groq import get_llm
        _singleton = CorrectionDetectionAgent(get_llm())
    return _singleton
