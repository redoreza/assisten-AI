"""Face recognition REST endpoints.

Two input modes for image payloads:
- JSON body with `image_base64: str` (best for browser → backend over HTTPS;
  decode is fast, no multipart parsing).
- multipart form (`image=<file>`) for `curl -F` / Postman style testing.

Endpoints:
    POST   /api/face/recognize        → list[FaceMatch]
    POST   /api/face/enroll           → EnrollResult
    GET    /api/face/persons          → list[{person_id, name, embedding_count}]
    DELETE /api/face/persons/{pid}    → {deleted: bool}
    POST   /api/face/warmup           → pre-load model + DB (one-time, ~30-90s)
"""

from __future__ import annotations

import base64
import binascii

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel, Field

from app.services.face_recognition import get_face_recognition

router = APIRouter(tags=["face"])


class RecognizeBody(BaseModel):
    image_base64: str = Field(min_length=1)
    threshold: float | None = None


class EnrollJsonBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    images_base64: list[str] = Field(min_length=1, max_length=10)


def _decode_b64(b: str) -> bytes:
    # Allow `data:image/jpeg;base64,...` prefix from canvas.toDataURL
    if "," in b and b.startswith("data:"):
        b = b.split(",", 1)[1]
    try:
        return base64.b64decode(b, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(400, f"invalid base64: {exc}") from exc


@router.post("/face/warmup")
async def warmup() -> dict:
    """Trigger model + DB load. Call once at startup to avoid first-request latency."""
    try:
        return await get_face_recognition().warmup()
    except Exception as exc:
        logger.exception("Face warmup failed")
        raise HTTPException(500, f"warmup failed: {exc}") from exc


@router.post("/face/recognize")
async def recognize_json(body: RecognizeBody) -> dict:
    image = _decode_b64(body.image_base64)
    try:
        faces = await get_face_recognition().recognize(image, threshold=body.threshold)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Recognize failed")
        raise HTTPException(500, str(exc)) from exc
    return {"faces": faces, "count": len(faces)}


@router.post("/face/recognize/multipart")
async def recognize_multipart(
    image: UploadFile = File(...),
    threshold: float | None = Form(None),
) -> dict:
    image_bytes = await image.read()
    try:
        faces = await get_face_recognition().recognize(image_bytes, threshold=threshold)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Recognize (multipart) failed")
        raise HTTPException(500, str(exc)) from exc
    return {"faces": faces, "count": len(faces)}


@router.post("/face/enroll")
async def enroll_json(body: EnrollJsonBody) -> dict:
    images = [_decode_b64(b) for b in body.images_base64]
    try:
        return await get_face_recognition().enroll(body.name, images)  # type: ignore[return-value]
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Enroll failed")
        raise HTTPException(500, str(exc)) from exc


@router.post("/face/enroll/multipart")
async def enroll_multipart(
    name: str = Form(..., min_length=1, max_length=80),
    images: list[UploadFile] = File(...),
) -> dict:
    if not images:
        raise HTTPException(400, "at least one image is required")
    image_bytes_list = [await img.read() for img in images]
    try:
        return await get_face_recognition().enroll(name, image_bytes_list)  # type: ignore[return-value]
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Enroll (multipart) failed")
        raise HTTPException(500, str(exc)) from exc


@router.get("/face/persons")
async def list_persons() -> dict:
    try:
        persons = await get_face_recognition().list_persons()
    except Exception as exc:
        logger.exception("list_persons failed")
        raise HTTPException(500, str(exc)) from exc
    return {"persons": persons, "count": len(persons)}


@router.delete("/face/persons/{person_id}")
async def delete_person(person_id: int) -> dict:
    try:
        deleted = await get_face_recognition().delete_person(person_id)
    except Exception as exc:
        logger.exception("delete_person failed")
        raise HTTPException(500, str(exc)) from exc
    if not deleted:
        raise HTTPException(404, f"person_id={person_id} not found")
    return {"deleted": True, "person_id": person_id}
