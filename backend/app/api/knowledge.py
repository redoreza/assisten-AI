"""HTTP endpoints for Pointer's knowledge base (Phase 5 RAG).

POST /kb/upload       — upload PDF/DOCX/TXT file
GET  /kb/sources      — list all indexed document sources
DELETE /kb/source/{name} — remove a document from the index
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from loguru import logger

from app.services.knowledge_base import get_knowledge_base

router = APIRouter(prefix="/kb", tags=["knowledge-base"])

_ALLOWED_SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".md"}
_MAX_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("/upload")
async def upload_document(file: UploadFile):
    """Upload a document to Pointer's knowledge base."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type '{suffix}'. Allowed: {sorted(_ALLOWED_SUFFIXES)}",
        )
    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB)")
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        kb = get_knowledge_base()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        n_chunks = await kb.add_document(tmp_path, source_name=file.filename)
    except Exception as exc:
        logger.exception(f"[KB] add_document failed for '{file.filename}'")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"filename": file.filename, "chunks_indexed": n_chunks}


@router.get("/sources")
def list_sources():
    try:
        return {"sources": get_knowledge_base().list_sources()}
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.delete("/source/{source_name}")
def delete_source(source_name: str):
    try:
        n = get_knowledge_base().delete_source(source_name)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"deleted_chunks": n, "source": source_name}
