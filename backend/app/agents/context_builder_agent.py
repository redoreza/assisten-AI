"""ContextBuilderAgent — assemble the message list sent to the LLM.

Two entry points:
  • build(persona, history)              — bare history, no session state
  • build_with_session(persona, session) — injects realtime clock + person name

Both trim the history to the last memory_max_turns * 2 messages (one turn =
one user + one assistant message) so the LLM context stays bounded.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.config import settings
from app.core.persona import Persona
from app.services.llm_groq import ChatMessage

_ID_DAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
_ID_MONTHS = [
    "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]


def _now_indonesian() -> str:
    """Return a human-readable Bahasa Indonesia datetime string (WIB)."""
    try:
        now = datetime.now(ZoneInfo("Asia/Jakarta"))
    except Exception:
        now = datetime.now()
    day = _ID_DAYS[now.weekday()]
    month = _ID_MONTHS[now.month]
    return f"{day}, {now.day} {month} {now.year}, pukul {now:%H:%M} WIB"


class ContextBuilderAgent:
    def build(
        self, persona: Persona, history: list[ChatMessage]
    ) -> list[ChatMessage]:
        """Bare context: system prompt + trimmed history."""
        system: ChatMessage = {"role": "system", "content": persona.render_system_prompt()}
        return [system, *history[-(settings.memory_max_turns * 2) :]]

    def build_with_session(
        self, persona: Persona, session: Any
    ) -> list[ChatMessage]:
        """Full context: system prompt + realtime clock + engaged-person hint + history."""
        content = persona.render_system_prompt()
        content += f"\n\nWAKTU SEKARANG: {_now_indonesian()}"
        if session.last_engaged_name:
            content += (
                f"\n\nKONTEKS PERCAKAPAN INI:\n"
                f"Kamu sedang ngobrol dengan {session.last_engaged_name}. "
                f"Sapa dengan namanya saat natural saja, tidak setiap pesan."
            )
        system: ChatMessage = {"role": "system", "content": content}
        return [system, *session.history[-(settings.memory_max_turns * 2) :]]


_singleton: ContextBuilderAgent | None = None


def get_context_builder_agent() -> ContextBuilderAgent:
    global _singleton
    if _singleton is None:
        _singleton = ContextBuilderAgent()
    return _singleton
