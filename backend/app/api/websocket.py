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

import asyncio
import base64
import binascii
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.agents.correction_detection_agent import get_correction_detection_agent
from app.agents.face_engagement_agent import get_face_engagement_agent
from app.agents.name_extraction_agent import get_name_extraction_agent
from app.agents.stt_agent import get_stt_agent
from app.config import settings
from app.core.orchestrator import SessionState, get_orchestrator
from app.services.face_recognition import get_face_recognition

router = APIRouter()


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

    pipeline_task: asyncio.Task | None = None

    def cancel_pipeline() -> None:
        nonlocal pipeline_task
        if pipeline_task and not pipeline_task.done():
            pipeline_task.cancel()
        pipeline_task = None
        session.pipeline_busy = False

    async def run_pipeline(text: str) -> None:
        session.pipeline_busy = True
        try:
            await handle_user_input(text)
        except asyncio.CancelledError:
            logger.info("[WS] pipeline cancelled (barge-in)")
            raise
        finally:
            session.pipeline_busy = False

    async def handle_user_input(text: str) -> None:
        """Route user text/audio: if we're awaiting a name, treat it as the name
        and run enrollment. Otherwise normal chat."""
        if session.awaiting_name and session.pending_enroll_image_b64:
            image_b64 = session.pending_enroll_image_b64
            # Clear state immediately so a slow LLM doesn't double-fire
            session.awaiting_name = False
            session.pending_enroll_image_b64 = None
            logger.info(f"[face] extracting name from: '{text[:80]}'")
            extracted = await get_name_extraction_agent().run(text)
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
            new_name = await get_correction_detection_agent().run(text, session.last_engaged_name)
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
                try:
                    transcript = await get_stt_agent().transcribe(
                        audio_bytes,
                        format=audio_format,
                        session=session,
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
                    if pipeline_task and not pipeline_task.done():
                        # Pipeline busy — queue as light chat instead of barge-in
                        session.pending_light_chat_queue.append(transcript)
                        logger.info(f"[WS] light chat queued (audio): '{transcript[:80]}'")
                    else:
                        cancel_pipeline()
                        pipeline_task = asyncio.create_task(run_pipeline(transcript))
                else:
                    await ws.send_json({"type": "done", "reason": "empty_transcript"})

            elif mtype == "text":
                message = msg.get("message", "")
                if not isinstance(message, str) or not message.strip():
                    await ws.send_json({"type": "error", "message": "text.message required"})
                    continue
                logger.info(f"WS text: '{message[:80]}'")
                if pipeline_task and not pipeline_task.done():
                    # Main pipeline is busy — queue as light chat instead of barge-in.
                    # Use the 'interrupt' message type to force a barge-in when needed.
                    session.pending_light_chat_queue.append(message)
                    logger.info(f"[WS] light chat queued: '{message[:80]}'")
                else:
                    cancel_pipeline()
                    pipeline_task = asyncio.create_task(run_pipeline(message))

            elif mtype == "interrupt":
                cancel_pipeline()
                logger.info("[WS] interrupt received")

            elif mtype == "face_present":
                await get_face_engagement_agent().handle(ws, session, orch, msg)

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
                session.light_chat_history.clear()
                session.pending_light_chat_queue.clear()
                await ws.send_json({"type": "ack", "field": "history", "value": "cleared"})

            else:
                await ws.send_json({"type": "error", "message": f"unknown type: {mtype}"})

    except WebSocketDisconnect:
        cancel_pipeline()
        logger.info(f"WS disconnect from {client}")
    except Exception:
        cancel_pipeline()
        logger.exception("WS handler crashed")
        try:
            await ws.close(code=1011)
        except Exception:
            pass
