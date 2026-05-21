"""WebSocket endpoint for the voice pipeline.

Protocol (extended in R4 for face engagement):

Client → Server:
    {"type": "audio_chunk", "data": <base64>, "format": "webm"}
    {"type": "audio_end"}
    {"type": "text", "message": "..."}
    {"type": "set_persona", "persona_id": "pointer"}
    {"type": "set_mode", "mode": "companion"|"customer_service"}
    {"type": "clear_history"}
    # R4 face engagement:
    {"type": "face_present", "match_name": str|null, "match_person_id": int|null,
     "similarity": float, "image_base64": str}
    {"type": "face_lost"}

Server → Client: see orchestrator event types.
"""

from __future__ import annotations

import base64
import binascii
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.config import settings
from app.core.orchestrator import SessionState, get_orchestrator
from app.services.chat_history import get_chat_history
from app.services.face_recognition import get_face_recognition

router = APIRouter()

# How long an unknown face must remain visible before we ask their name —
# prevents reacting to brief glances or people walking past.
UNKNOWN_STABLE_SECONDS = 3.0
# Debounce on any face-driven speech — avoid firing two greetings in quick succession
FACE_ACTION_COOLDOWN_S = 5.0


async def _dispatch_orchestrator(ws: WebSocket, iterator) -> None:
    """Forward orchestrator events to the WS until the iterator completes."""
    async for event in iterator:
        await ws.send_json(event)


def _strip_data_url(b: str) -> str:
    if "," in b and b.startswith("data:"):
        return b.split(",", 1)[1]
    return b


@router.websocket("/ws")
async def voice_ws(ws: WebSocket) -> None:
    await ws.accept()
    session = SessionState(persona_id=settings.default_persona, mode="companion")
    orch = get_orchestrator()
    face_svc = get_face_recognition()

    audio_buf = bytearray()
    audio_format = "webm"
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
    logger.info(f"WS connect from {client}")

    async def handle_user_input(text: str) -> None:
        """Route user text/audio: if we're awaiting a name, treat it as the name
        and run enrollment. Otherwise normal chat."""
        if session.awaiting_name and session.pending_enroll_image_b64:
            image_b64 = session.pending_enroll_image_b64
            # Clear state immediately so a slow LLM doesn't double-fire
            session.awaiting_name = False
            session.pending_enroll_image_b64 = None
            logger.info(f"[face] extracting name from: '{text[:80]}'")
            extracted = await orch.extract_name(text)
            if not extracted:
                logger.info("[face] no name extracted; falling back to chat")
                await _dispatch_orchestrator(ws, orch.handle_text(session, text))
                return
            try:
                image_bytes = base64.b64decode(_strip_data_url(image_b64))
            except (binascii.Error, ValueError) as exc:
                await ws.send_json({"type": "error", "message": f"bad image: {exc}"})
                return
            try:
                result = await face_svc.enroll(extracted, [image_bytes])
            except ValueError as exc:
                logger.warning(f"[face] enroll failed: {exc}")
                await ws.send_json(
                    {"type": "error", "message": f"enroll failed: {exc}"}
                )
                return
            logger.info(
                f"[face] enrolled '{extracted}' person_id={result['person_id']}"
            )
            session.engaged_person_id = result["person_id"]
            session.greeted_person_ids.add(result["person_id"])
            session.last_engaged_name = extracted
            session.last_known_face_image_b64 = image_b64  # keep for potential correction
            session.last_face_action_at = time.monotonic()
            await ws.send_json(
                {
                    "type": "face_enrolled",
                    "person_id": result["person_id"],
                    "name": extracted,
                }
            )
            await _dispatch_orchestrator(ws, orch.confirm_enrolled(session, extracted))
            return

        # ── Correction intent? ─────────────────────────────────────────────
        # Only check if we have an engaged person with a known name AND a
        # cached face image to re-enroll under the corrected name.
        logger.debug(
            f"[face] correction-check: engaged={session.engaged_person_id} "
            f"name={session.last_engaged_name!r} "
            f"has_img={session.last_known_face_image_b64 is not None}"
        )
        if (
            session.engaged_person_id is not None
            and session.last_engaged_name
            and session.last_known_face_image_b64
        ):
            new_name = await orch.detect_correction(text, session.last_engaged_name)
            logger.info(
                f"[face] correction LLM returned: {new_name!r} "
                f"(current: {session.last_engaged_name!r})"
            )
            if new_name and new_name != session.last_engaged_name:
                logger.info(
                    f"[face] correction: '{session.last_engaged_name}' -> '{new_name}'"
                )
                old_name = session.last_engaged_name
                old_pid = session.engaged_person_id
                # Delete old DB entry
                try:
                    await face_svc.delete_person(old_pid)
                except Exception as exc:
                    logger.warning(f"delete_person({old_pid}) failed: {exc}")
                # Enroll under new name with cached face image
                try:
                    image_bytes = base64.b64decode(
                        _strip_data_url(session.last_known_face_image_b64)
                    )
                    result = await face_svc.enroll(new_name, [image_bytes])
                except (binascii.Error, ValueError) as exc:
                    logger.warning(f"correction enroll failed: {exc}")
                    await ws.send_json(
                        {"type": "error", "message": f"correction failed: {exc}"}
                    )
                    return
                # Re-engage with the new person
                session.engaged_person_id = result["person_id"]
                # greeted_person_ids may still contain old_pid; replace it
                session.greeted_person_ids.discard(old_pid)
                session.greeted_person_ids.add(result["person_id"])
                session.last_engaged_name = new_name
                session.last_face_action_at = time.monotonic()
                await ws.send_json(
                    {
                        "type": "face_enrolled",
                        "person_id": result["person_id"],
                        "name": new_name,
                    }
                )
                await _dispatch_orchestrator(
                    ws, orch.confirm_correction(session, old_name, new_name)
                )
                return

        await _dispatch_orchestrator(ws, orch.handle_text(session, text))

    try:
        await ws.send_json(
            {
                "type": "ready",
                "persona_id": session.persona_id,
                "mode": session.mode,
            }
        )
        while True:
            msg: dict[str, Any] = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "audio_chunk":
                data = msg.get("data")
                fmt = msg.get("format", "webm")
                if not isinstance(data, str):
                    await ws.send_json(
                        {"type": "error", "message": "audio_chunk.data must be base64 string"}
                    )
                    continue
                try:
                    audio_buf.extend(base64.b64decode(data))
                except Exception as exc:
                    await ws.send_json({"type": "error", "message": f"base64 decode failed: {exc}"})
                    continue
                audio_format = fmt

            elif mtype == "audio_end":
                if not audio_buf:
                    await ws.send_json({"type": "error", "message": "no audio buffered"})
                    continue
                audio_bytes = bytes(audio_buf)
                audio_buf.clear()
                logger.info(f"WS audio_end: {len(audio_bytes)} bytes ({audio_format})")
                # All audio goes through STT here so handle_user_input can decide
                # the routing (name-extraction / correction intent / normal chat).
                # Previously normal-chat audio bypassed handle_user_input and so
                # name corrections spoken via mic were never detected.
                stt_t0 = time.monotonic()
                from app.core.orchestrator import _stt_bias_prompt
                try:
                    transcript = await orch._stt.transcribe(  # type: ignore[attr-defined]
                        audio_bytes,
                        format=audio_format,
                        language="id",
                        prompt=_stt_bias_prompt(session),
                    )
                except Exception as exc:
                    logger.exception("STT failed")
                    await ws.send_json(
                        {"type": "error", "message": f"STT failed: {exc}"}
                    )
                    continue
                stt_ms = int((time.monotonic() - stt_t0) * 1000)
                await ws.send_json(
                    {"type": "transcript", "text": transcript, "latency_ms": stt_ms}
                )
                if transcript.strip():
                    await handle_user_input(transcript)
                else:
                    await ws.send_json({"type": "done", "reason": "empty_transcript"})

            elif mtype == "text":
                message = msg.get("message", "")
                if not isinstance(message, str) or not message.strip():
                    await ws.send_json({"type": "error", "message": "text.message required"})
                    continue
                logger.info(f"WS text: '{message[:80]}'")
                await handle_user_input(message)

            elif mtype == "face_present":
                await _handle_face_present(ws, session, orch, msg)

            elif mtype == "face_lost":
                if session.engaged_person_id is not None or session.awaiting_name:
                    logger.info("[face] face_lost — reset engagement")
                session.reset_face()

            elif mtype == "set_persona":
                pid = msg.get("persona_id")
                if not isinstance(pid, str):
                    await ws.send_json(
                        {"type": "error", "message": "set_persona.persona_id required"}
                    )
                    continue
                session.persona_id = pid
                await ws.send_json({"type": "ack", "field": "persona_id", "value": pid})

            elif mtype == "set_mode":
                mode = msg.get("mode")
                if mode not in ("companion", "customer_service"):
                    await ws.send_json({"type": "error", "message": "invalid mode"})
                    continue
                session.mode = mode
                await ws.send_json({"type": "ack", "field": "mode", "value": mode})

            elif mtype == "clear_history":
                session.history.clear()
                session.greeted_person_ids.clear()
                session.reset_face()
                audio_buf.clear()
                await ws.send_json({"type": "ack", "field": "history", "value": "cleared"})

            else:
                await ws.send_json({"type": "error", "message": f"unknown type: {mtype}"})

    except WebSocketDisconnect:
        logger.info(f"WS disconnect from {client}")
    except Exception:
        logger.exception("WS handler crashed")
        try:
            await ws.close(code=1011)
        except Exception:
            pass


async def _handle_face_present(
    ws: WebSocket,
    session: SessionState,
    orch,
    msg: dict[str, Any],
) -> None:
    """Drive the face engagement state machine from one face_present event."""
    if session.awaiting_name:
        return  # already asked; wait for user's response

    match_name = msg.get("match_name")
    match_person_id = msg.get("match_person_id")
    image_b64 = msg.get("image_base64")
    now = time.monotonic()

    # ── Known face ───────────────────────────────────────────────────────────
    if match_person_id is not None and isinstance(match_name, str):
        session.unknown_first_seen_at = None
        # Always keep the freshest face image cached — needed if the user
        # later says their name is wrong and we re-enroll under a new name.
        if isinstance(image_b64, str) and image_b64:
            session.last_known_face_image_b64 = image_b64
        session.last_engaged_name = match_name
        # Always restore engagement on every known-face frame; without this, a
        # face_lost → re-entry sequence would leave engaged_person_id=None and
        # break the correction flow even though greeted_person_ids still holds
        # the person.
        session.engaged_person_id = match_person_id
        if match_person_id in session.greeted_person_ids:
            return  # already greeted this session — don't re-greet
        if now - session.last_face_action_at < FACE_ACTION_COOLDOWN_S:
            return
        # Hydrate cross-session memory: pull this person's last N turns from
        # disk so Pointer can reference what they talked about previously.
        # Replace any in-memory history (which would be either empty or from
        # a different person on the same WS connection).
        try:
            past = await get_chat_history().recent(match_person_id, max_turns=12)
            if past:
                session.history = [
                    {"role": role, "content": content} for role, content, _ in past
                ]
                logger.info(
                    f"[face] hydrated {len(past)} past turns for "
                    f"person_id={match_person_id} ({match_name})"
                )
            else:
                session.history = []
        except Exception as exc:
            logger.warning(f"chat_history hydrate failed: {exc}")
        session.greeted_person_ids.add(match_person_id)
        session.last_face_action_at = now
        await _dispatch_orchestrator(ws, orch.greet_known(session, match_name))
        return

    # ── Unknown face ─────────────────────────────────────────────────────────
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
    await ws.send_json({"type": "face_awaiting_name"})
    await _dispatch_orchestrator(ws, orch.ask_name(session))
