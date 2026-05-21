"""Face recognition service — InsightFace buffalo_l (SCRFD detector + ArcFace
embedder) wrapped behind a small task-oriented API.

`buffalo_l` ships as a single ~300 MB bundle that auto-downloads to
`~/.insightface/models/buffalo_l/` on first use. We do this lazily on the first
recognize/enroll call so FastAPI startup stays snappy; hit `/api/face/warmup`
to pre-pay that cost out-of-band.

Persistence lives in `app.recognition.face_database.SQLiteFaceDB`, copied
verbatim from the detection-engine prototype.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from loguru import logger

from app.config import settings
from app.recognition.face_database import SQLiteFaceDB


class FaceBox(TypedDict):
    x: int
    y: int
    width: int
    height: int


class FaceMatch(TypedDict):
    bbox: FaceBox
    det_score: float
    match_name: str | None
    match_person_id: int | None
    similarity: float  # similarity of the top match, even if below threshold


class EnrollResult(TypedDict):
    person_id: int
    name: str
    images_provided: int
    embeddings_added: int


def _decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Empty image bytes")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image — not a valid JPEG/PNG/WebP")
    return img


def _bbox_to_xywh(bbox: np.ndarray) -> FaceBox:
    """Convert InsightFace [x1, y1, x2, y2] to {x, y, width, height}."""
    x1, y1, x2, y2 = bbox.astype(int).tolist()
    return {"x": int(x1), "y": int(y1), "width": int(x2 - x1), "height": int(y2 - y1)}


class FaceRecognition:
    """InsightFace + SQLiteFaceDB wrapper.

    InsightFace's `FaceAnalysis.get()` is not async-safe (single ONNX runtime
    session under the GIL), so all inference calls are funneled through one
    `asyncio.Lock` and dispatched via `asyncio.to_thread`. For our 1-2 fps
    workload that's plenty.
    """

    def __init__(
        self,
        db_path: Path,
        threshold: float,
        det_size: int,
        model_name: str,
        use_gpu: bool = True,
        adaptive_enabled: bool = True,
        adaptive_threshold: float = 0.65,
        max_emb_per_person: int = 20,
    ) -> None:
        self._db_path = db_path
        self._threshold = threshold
        self._det_size = det_size
        self._model_name = model_name
        self._use_gpu = use_gpu
        self._adaptive_enabled = adaptive_enabled
        self._adaptive_threshold = adaptive_threshold
        self._max_emb_per_person = max_emb_per_person
        self._app: FaceAnalysis | None = None
        self._db: SQLiteFaceDB | None = None
        self._lock = asyncio.Lock()

    def _ensure_loaded_sync(self) -> tuple[FaceAnalysis, SQLiteFaceDB]:
        """Lazy init — first call may take 30-90s on a cold cache (model download)."""
        if self._app is None:
            import onnxruntime as ort

            available = ort.get_available_providers()
            if self._use_gpu:
                # Prefer CUDA (needs cuDNN), fall back to DirectML (Windows DX12),
                # then CPU.
                gpu_providers = [
                    p for p in ["CUDAExecutionProvider", "DmlExecutionProvider"]
                    if p in available
                ]
                providers = gpu_providers + ["CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            logger.info(
                f"Loading InsightFace {self._model_name} "
                f"(providers={providers}, auto-downloads on first run)..."
            )
            self._app = FaceAnalysis(name=self._model_name, providers=providers)
            self._app.prepare(ctx_id=0, det_size=(self._det_size, self._det_size))
            # Report which provider each sub-model actually loaded onto
            for m in self._app.models.values():
                try:
                    active = m.session.get_providers()[0]
                    logger.info(f"  {m.__class__.__name__}: {active}")
                except Exception:
                    pass
            logger.info(
                f"InsightFace ready (model={self._model_name}, det_size={self._det_size})"
            )
        if self._db is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = SQLiteFaceDB(self._db_path)
            logger.info(f"Face DB at {self._db_path}: {self._db.stats()}")
        return self._app, self._db

    async def warmup(self) -> dict:
        async with self._lock:
            _, db = await asyncio.to_thread(self._ensure_loaded_sync)
            return {"ready": True, "db_stats": db.stats()}

    async def recognize(
        self, image_bytes: bytes, *, threshold: float | None = None
    ) -> list[FaceMatch]:
        thr = threshold if threshold is not None else self._threshold
        async with self._lock:
            return await asyncio.to_thread(self._recognize_sync, image_bytes, thr)

    def _recognize_sync(self, image_bytes: bytes, threshold: float) -> list[FaceMatch]:
        app, db = self._ensure_loaded_sync()
        img = _decode_image(image_bytes)
        faces = app.get(img)
        out: list[FaceMatch] = []
        # Per-person current embedding count, cached to avoid hitting DB per face
        per_person_count: dict[int, int] | None = None
        adaptive_adds: list[tuple[str, np.ndarray]] = []
        for f in faces:
            bbox = _bbox_to_xywh(f.bbox)
            matches = db.search(f.embedding, top_k=1)
            top = matches[0] if matches else None
            top_sim = float(top.similarity) if top else 0.0
            if top and top_sim >= threshold:
                out.append(
                    {
                        "bbox": bbox,
                        "det_score": float(f.det_score),
                        "match_name": top.name,
                        "match_person_id": top.person_id,
                        "similarity": top_sim,
                    }
                )
                # Continuous learning: high-confidence matches grow the person's
                # embedding profile, up to max_emb_per_person.
                if self._adaptive_enabled and top_sim >= self._adaptive_threshold:
                    if per_person_count is None:
                        per_person_count = {pid: n for pid, _, n in db.list_persons()}
                    cur = per_person_count.get(top.person_id, 0)
                    if cur < self._max_emb_per_person:
                        adaptive_adds.append((top.name, f.embedding))
                        per_person_count[top.person_id] = cur + 1
            else:
                out.append(
                    {
                        "bbox": bbox,
                        "det_score": float(f.det_score),
                        "match_name": None,
                        "match_person_id": None,
                        "similarity": top_sim,
                    }
                )

        if adaptive_adds:
            # Group by name; one db.enroll per name collapses N centroid rebuilds into 1
            by_name: dict[str, list[np.ndarray]] = {}
            for name, emb in adaptive_adds:
                by_name.setdefault(name, []).append(emb)
            for name, embs in by_name.items():
                try:
                    db.enroll(name, embs, source="adaptive:recognize")
                    logger.debug(
                        f"Adaptive: +{len(embs)} embedding(s) for '{name}'"
                    )
                except Exception as exc:
                    logger.warning(f"Adaptive enroll for '{name}' failed: {exc}")

        return out

    async def enroll(self, name: str, image_bytes_list: list[bytes]) -> EnrollResult:
        if not name.strip():
            raise ValueError("name must not be empty")
        if not image_bytes_list:
            raise ValueError("at least one image is required")
        async with self._lock:
            return await asyncio.to_thread(self._enroll_sync, name.strip(), image_bytes_list)

    def _enroll_sync(self, name: str, image_bytes_list: list[bytes]) -> EnrollResult:
        app, db = self._ensure_loaded_sync()
        embeddings: list[np.ndarray] = []
        for img_bytes in image_bytes_list:
            try:
                img = _decode_image(img_bytes)
            except ValueError as exc:
                logger.warning(f"Skipping invalid image during enroll: {exc}")
                continue
            faces = app.get(img)
            if not faces:
                continue
            # Use the largest face — typical single-subject enrollment
            largest = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            embeddings.append(largest.embedding)
        if not embeddings:
            raise ValueError(
                f"No faces detected in any of the {len(image_bytes_list)} provided images"
            )
        person_id = db.enroll(name, embeddings, source="api_enroll")
        logger.info(
            f"Enrolled '{name}' → person_id={person_id} "
            f"with {len(embeddings)}/{len(image_bytes_list)} usable images"
        )
        return {
            "person_id": person_id,
            "name": name,
            "images_provided": len(image_bytes_list),
            "embeddings_added": len(embeddings),
        }

    async def list_persons(self) -> list[dict]:
        async with self._lock:
            return await asyncio.to_thread(self._list_persons_sync)

    def _list_persons_sync(self) -> list[dict]:
        _, db = self._ensure_loaded_sync()
        return [
            {"person_id": pid, "name": name, "embedding_count": count}
            for pid, name, count in db.list_persons()
        ]

    async def delete_person(self, person_id: int) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self._delete_person_sync, person_id)

    def _delete_person_sync(self, person_id: int) -> bool:
        _, db = self._ensure_loaded_sync()
        return db.delete_person(person_id)


_singleton: FaceRecognition | None = None


def get_face_recognition() -> FaceRecognition:
    global _singleton
    if _singleton is None:
        _singleton = FaceRecognition(
            db_path=settings.faces_db_full_path,
            threshold=settings.face_match_threshold,
            det_size=settings.face_det_size,
            model_name=settings.face_model_name,
            use_gpu=settings.face_use_gpu,
            adaptive_enabled=settings.face_adaptive_enabled,
            adaptive_threshold=settings.face_adaptive_threshold,
            max_emb_per_person=settings.face_max_emb_per_person,
        )
    return _singleton
