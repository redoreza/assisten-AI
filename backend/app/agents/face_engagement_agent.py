"""FaceEngagementAgent — face presence state machine.

Encapsulates the _handle_face_present logic that was previously inline in
websocket.py. Responsible for:
  - Debouncing unknown faces (UNKNOWN_STABLE_SECONDS)
  - Enforcing per-action cooldown (FACE_ACTION_COOLDOWN_S)
  - Triggering greet_known / ask_name on the orchestrator
  - Hydrating per-person history via HistoryHydrationAgent
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import WebSocket
from loguru import logger

from app.agents.expression_analysis_agent import get_expression_analysis_agent
from app.agents.history_hydration_agent import get_history_hydration_agent

UNKNOWN_STABLE_SECONDS = 3.0
FACE_ACTION_COOLDOWN_S = 5.0
EXPRESSION_CHECK_INTERVAL_S = 60.0
_PROACTIVE_MOODS = {"tired", "confused", "happy"}


async def _dispatch(ws: WebSocket, iterator) -> None:
    async for event in iterator:
        await ws.send_json(event)


class FaceEngagementAgent:
    async def handle(
        self,
        ws: WebSocket,
        session: Any,
        orch: Any,
        msg: dict[str, Any],
    ) -> None:
        """Process one face_present event against the current session state."""
        if session.awaiting_name:
            return

        match_name = msg.get("match_name")
        match_person_id = msg.get("match_person_id")
        image_b64 = msg.get("image_base64")
        now = time.monotonic()

        # ── Known face ───────────────────────────────────────────────────────
        if match_person_id is not None and isinstance(match_name, str):
            session.unknown_first_seen_at = None
            if isinstance(image_b64, str) and image_b64:
                session.last_known_face_image_b64 = image_b64
            session.last_engaged_name = match_name
            session.engaged_person_id = match_person_id

            # Initial greeting (once per person per session)
            if match_person_id not in session.greeted_person_ids:
                if now - session.last_face_action_at < FACE_ACTION_COOLDOWN_S:
                    return
                session.history = await get_history_hydration_agent().run(match_person_id)
                if session.history:
                    logger.info(
                        f"[face] hydrated {len(session.history)} turns for "
                        f"person_id={match_person_id} ({match_name})"
                    )
                session.greeted_person_ids.add(match_person_id)
                session.last_face_action_at = now
                mood = await get_expression_analysis_agent().analyze(image_b64 or "")
                session.last_known_mood = mood
                session.last_expression_check_at = now
                await _dispatch(ws, orch.greet_known(session, match_name, mood=mood))
                return

            # Periodic expression check — runs every EXPRESSION_CHECK_INTERVAL_S
            # when pipeline is idle so we don't interrupt an active turn.
            if (
                isinstance(image_b64, str) and image_b64
                and not session.pipeline_busy
                and now - session.last_expression_check_at >= EXPRESSION_CHECK_INTERVAL_S
            ):
                session.last_expression_check_at = now
                new_mood = await get_expression_analysis_agent().analyze(image_b64)
                logger.info(
                    f"[face] periodic check: {session.last_known_mood} → {new_mood} "
                    f"for {match_name}"
                )
                if new_mood in _PROACTIVE_MOODS and new_mood != session.last_known_mood:
                    session.last_known_mood = new_mood
                    await _dispatch(ws, orch.probe_mood(session, match_name, new_mood))
                else:
                    session.last_known_mood = new_mood
            return

        # ── Unknown face ─────────────────────────────────────────────────────
        if not isinstance(image_b64, str) or not image_b64:
            return
        if session.unknown_first_seen_at is None:
            session.unknown_first_seen_at = now
            logger.debug(f"[face] unknown face first seen at {now:.1f}")
            return
        elapsed = now - session.unknown_first_seen_at
        if elapsed < UNKNOWN_STABLE_SECONDS:
            return
        if now - session.last_face_action_at < FACE_ACTION_COOLDOWN_S:
            return
        session.pending_enroll_image_b64 = image_b64
        session.awaiting_name = True
        session.last_face_action_at = now
        logger.info(f"[face] unknown stable for {elapsed:.1f}s, asking for name")
        mood = await get_expression_analysis_agent().analyze(image_b64)
        await ws.send_json({"type": "face_awaiting_name"})
        await _dispatch(ws, orch.ask_name(session, mood=mood))


_singleton: FaceEngagementAgent | None = None


def get_face_engagement_agent() -> FaceEngagementAgent:
    global _singleton
    if _singleton is None:
        _singleton = FaceEngagementAgent()
    return _singleton
