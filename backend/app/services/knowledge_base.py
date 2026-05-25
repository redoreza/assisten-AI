"""Knowledge base service — NVIDIA NIM embeddings + ChromaDB vector store.

Uses baai/bge-m3 (multilingual, free tier) via the NVIDIA NIM OpenAI-compatible
embeddings endpoint. Documents are chunked, embedded, and stored in ChromaDB.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import NamedTuple

from loguru import logger
from openai import AsyncOpenAI

from app.config import settings

_COLLECTION_NAME = "pointer_kb"
_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class KBChunk(NamedTuple):
    text: str
    source: str
    similarity: float


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    if suffix in (".docx", ".doc"):
        from docx import Document
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    return path.read_text(encoding="utf-8", errors="replace")


class KnowledgeBase:
    def __init__(self) -> None:
        if not settings.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY not configured")
        self._client = AsyncOpenAI(
            base_url=_NVIDIA_BASE_URL,
            api_key=settings.nvidia_api_key,
        )
        self._model = settings.embedding_model  # baai/bge-m3
        self._chunk_size = settings.rag_chunk_size
        self._overlap = settings.rag_chunk_overlap
        self._top_k = settings.rag_top_k
        self._min_sim = settings.rag_min_similarity
        self._collection = self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb not installed — stop the backend and run: uv pip install chromadb"
            ) from exc
        db_path = settings.chroma_full_path
        db_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(db_path))
        return client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    async def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        """Call NVIDIA NIM embeddings API. Batches ≤ 96 texts per request."""
        batch_size = 96
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self._client.embeddings.create(
                input=batch,
                model=self._model,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "END"},
            )
            all_embeddings.extend(e.embedding for e in response.data)
        return all_embeddings

    async def add_document(self, path: Path, source_name: str | None = None) -> int:
        """Chunk, embed, and upsert a document. Returns number of chunks stored."""
        source = source_name or path.name
        text = await asyncio.to_thread(_extract_text, path)
        if not text.strip():
            logger.warning(f"[KB] empty text from '{source}'")
            return 0

        chunks = _chunk_text(text, self._chunk_size, self._overlap)
        if not chunks:
            return 0

        logger.info(f"[KB] embedding {len(chunks)} chunks from '{source}'")
        embeddings = await self._embed(chunks, input_type="passage")

        prefix = hashlib.md5(source.encode()).hexdigest()[:8]
        ids = [f"{prefix}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk_idx": i} for i in range(len(chunks))]

        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
        )
        logger.info(f"[KB] stored {len(chunks)} chunks for '{source}'")
        return len(chunks)

    async def query(self, question: str, top_k: int | None = None) -> list[KBChunk]:
        k = top_k or self._top_k
        if self._collection.count() == 0:
            return []

        q_emb = (await self._embed([question], input_type="query"))[0]

        res = self._collection.query(
            query_embeddings=[q_emb],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[KBChunk] = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            sim = 1.0 - dist / 2.0  # cosine distance [0,2] → similarity [0,1]
            if sim >= self._min_sim:
                chunks.append(KBChunk(text=doc, source=meta["source"], similarity=sim))
        return chunks

    async def build_context(self, question: str) -> str | None:
        chunks = await self.query(question)
        if not chunks:
            return None
        return "\n\n".join(f"[{c.source}] {c.text}" for c in chunks)

    def list_sources(self) -> list[str]:
        if self._collection.count() == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        return sorted({m["source"] for m in result["metadatas"]})

    def delete_source(self, source_name: str) -> int:
        result = self._collection.get(where={"source": source_name}, include=["metadatas"])
        ids = result["ids"]
        if ids:
            self._collection.delete(ids=ids)
        logger.info(f"[KB] deleted {len(ids)} chunks for '{source_name}'")
        return len(ids)


_singleton: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    global _singleton
    if _singleton is None:
        _singleton = KnowledgeBase()
    return _singleton
