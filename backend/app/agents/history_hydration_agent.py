"""HistoryHydrationAgent — load a known person's past chat turns from SQLite.

Called when a recognized face is first seen in a session (websocket face_present
path) and as a parallel catch-up inside handle_text if hydration has not yet
happened before the user's first utterance.

Returns an empty list on any error so callers never have to handle exceptions.
"""

from __future__ import annotations

from loguru import logger

from app.services.chat_history import get_chat_history
from app.services.llm_groq import ChatMessage


class HistoryHydrationAgent:
    async def run(self, person_id: int, max_turns: int = 12) -> list[ChatMessage]:
        try:
            past = await get_chat_history().recent(person_id, max_turns=max_turns)
            turns: list[ChatMessage] = [
                {"role": role, "content": content} for role, content, _ in past
            ]
            if turns:
                logger.info(
                    f"[HistoryHydrationAgent] {len(turns)} turns "
                    f"for person_id={person_id}"
                )
            return turns
        except Exception as exc:
            logger.warning(
                f"[HistoryHydrationAgent] failed for person_id={person_id}: {exc}"
            )
            return []


_singleton: HistoryHydrationAgent | None = None


def get_history_hydration_agent() -> HistoryHydrationAgent:
    global _singleton
    if _singleton is None:
        _singleton = HistoryHydrationAgent()
    return _singleton
