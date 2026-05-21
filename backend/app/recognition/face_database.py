"""SQLite-backed face database with in-memory numpy centroid index.

Design
------
- Two tables: `persons` (metadata) + `embeddings` (raw 512-float vectors as BLOB).
- Centroid per person computed at load time, refreshed on enroll/delete.
- Search = single numpy matmul against centroid matrix (BLAS-accelerated).
- All vectors stored L2-normalized → dot product = cosine similarity directly.
- Public surface routed through `FaceDatabaseProtocol` so a future Qdrant /
  FAISS backend can be dropped in without API churn.

Out of scope (KISS for MVP)
---------------------------
- Connection pooling (single-process workload).
- Concurrent writes (read-mostly: enroll rarely, search often).
- Schema migrations (single shot CREATE IF NOT EXISTS).
- Multi-model embeddings (only `buffalo_l` ArcFace 512-dim for MVP).
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger

EMBEDDING_DIM = 512  # InsightFace buffalo_l ArcFace (w600k_r50)
SCHEMA_VERSION = 1
DEFAULT_K_CENTROIDS = 3  # K-means clusters per person; pose-robust without bloat


def _kmeans_unit_sphere(
    vectors: np.ndarray, k: int, max_iter: int = 20, seed: int = 42
) -> np.ndarray:
    """Simple cosine-similarity K-means on already L2-normalized vectors.

    Returns up to `k` L2-normalized centroids. When `len(vectors) <= k`, the
    centroids degenerate to the input vectors themselves (no clustering needed).

    For face embeddings this means:
    - N=1 emb  → 1 centroid (= the emb), self-match sim = 1.0
    - N=k emb  → k centroids (= each emb), each pose matches its own perfectly
    - N>k emb  → k K-means cluster centers, query matches nearest pose-cluster
    """
    n = vectors.shape[0]
    if n <= k:
        return vectors.copy()

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = vectors[init_idx].copy()

    for _ in range(max_iter):
        sims = vectors @ centroids.T  # (n, k); both are unit-norm so dot = cosine
        assignments = sims.argmax(axis=1)

        new_centroids = np.empty_like(centroids)
        for j in range(k):
            mask = assignments == j
            if mask.any():
                mean = vectors[mask].mean(axis=0)
                norm = float(np.linalg.norm(mean))
                new_centroids[j] = mean / norm if norm > 1e-12 else centroids[j]
            else:
                new_centroids[j] = centroids[j]  # keep prior if no members

        if float(np.linalg.norm(new_centroids - centroids)) < 1e-6:
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids.astype(np.float32, copy=False)


@dataclass(frozen=True)
class PersonMatch:
    person_id: int
    name: str
    similarity: float


class FaceDatabaseProtocol(Protocol):
    """Abstract interface — SQLite / Qdrant / FAISS implementations interchangeable."""

    def enroll(self, name: str, embeddings: list[np.ndarray], source: str = "") -> int: ...

    def search(self, query: np.ndarray, top_k: int = 1) -> list[PersonMatch]: ...

    def list_persons(self) -> list[tuple[int, str, int]]: ...

    def delete_person(self, person_id: int) -> bool: ...

    def stats(self) -> dict: ...

    def close(self) -> None: ...


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    source      TEXT,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        raise ValueError("Cannot L2-normalize a near-zero embedding")
    return (v / norm).astype(np.float32, copy=False)


def _embedding_to_blob(v: np.ndarray) -> bytes:
    if v.shape != (EMBEDDING_DIM,):
        raise ValueError(f"Expected shape ({EMBEDDING_DIM},), got {v.shape}")
    return v.astype(np.float32, copy=False).tobytes()


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class SQLiteFaceDB:
    """SQLite + in-memory centroid index. Implements `FaceDatabaseProtocol`."""

    def __init__(self, db_path: Path | str, k_centroids: int = DEFAULT_K_CENTROIDS) -> None:
        if k_centroids < 1:
            raise ValueError(f"k_centroids must be >= 1, got {k_centroids}")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False lets the GUI worker thread create the connection
        # while the main thread queries it; we serialize access via self._lock below.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

        self._k = k_centroids
        self._lock = threading.RLock()
        self._person_ids: list[int] = []
        self._person_names: list[str] = []
        # Flat centroid matrix; multiple rows can belong to one person.
        # `_centroid_to_person_idx[i]` is the index into _person_ids for row i.
        # Rows are ordered by person, so per-person max can use np.maximum.reduceat.
        self._centroid_matrix: np.ndarray = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self._centroid_to_person_idx: np.ndarray = np.zeros(0, dtype=np.int32)
        # Lazy rebuild: enroll/delete just set dirty=True. search() rebuilds before
        # query. Batches many writes between searches into one expensive recompute,
        # critical for adaptive-enrollment workloads where many writes happen rapidly.
        self._dirty = False
        self._rebuild_centroids()

    def _rebuild_centroids(self) -> None:
        """Recompute per-person K-means centroids from current DB state.

        Result: contiguous `_centroid_matrix` with `_centroid_to_person_idx`
        giving the person-index of each row. Rows are grouped & ordered by
        person, enabling vectorized per-person max via `np.maximum.reduceat`.
        """
        rows = self._conn.execute("SELECT id, name FROM persons ORDER BY id").fetchall()
        ids: list[int] = []
        names: list[str] = []
        all_centroids: list[np.ndarray] = []
        owner_indices: list[int] = []

        for pid, name in rows:
            emb_rows = self._conn.execute(
                "SELECT vector FROM embeddings WHERE person_id = ?",
                (pid,),
            ).fetchall()
            if not emb_rows:
                continue
            vectors = np.stack([_blob_to_embedding(b) for (b,) in emb_rows])
            # Vectors are already L2-normalized at enroll time (see _embedding_to_blob path).
            centroids = _kmeans_unit_sphere(vectors, k=self._k)
            person_idx = len(ids)
            ids.append(int(pid))
            names.append(name)
            for c in centroids:
                all_centroids.append(c)
                owner_indices.append(person_idx)

        self._person_ids = ids
        self._person_names = names
        if all_centroids:
            self._centroid_matrix = np.stack(all_centroids).astype(np.float32, copy=False)
            self._centroid_to_person_idx = np.array(owner_indices, dtype=np.int32)
        else:
            self._centroid_matrix = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            self._centroid_to_person_idx = np.zeros(0, dtype=np.int32)

    def enroll(self, name: str, embeddings: list[np.ndarray], source: str = "") -> int:
        """Add embeddings under `name`. Creates the person if new. Returns person_id."""
        if not embeddings:
            raise ValueError("enroll() requires at least one embedding")
        now = _now_iso()

        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO persons(name, created_at) VALUES (?, ?)",
                (name, now),
            )
            if cur.rowcount == 0:
                row = self._conn.execute(
                    "SELECT id FROM persons WHERE name = ?", (name,)
                ).fetchone()
                person_id = int(row[0])
            else:
                person_id = int(cur.lastrowid)

            for emb in embeddings:
                normalized = _l2_normalize(emb)
                self._conn.execute(
                    "INSERT INTO embeddings(person_id, vector, source, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (person_id, _embedding_to_blob(normalized), source, now),
                )
            self._conn.commit()
            # Mark dirty; next search() will rebuild. Batches multiple enrolls
            # into a single rebuild instead of recomputing per-add.
            self._dirty = True
        logger.debug("Enrolled '{}' (id={}) +{} embeddings", name, person_id, len(embeddings))
        return person_id

    def search(self, query: np.ndarray, top_k: int = 1) -> list[PersonMatch]:
        """Return top-k matches by **per-person max cosine sim** across multi-centroids.

        With K=3 centroids per person, the query is matched against the nearest
        pose-cluster (not the average), which is robust to pose / lighting
        variation. Returns matches sorted by similarity descending.
        """
        with self._lock:
            if self._dirty:
                self._rebuild_centroids()
                self._dirty = False
            if self._centroid_matrix.shape[0] == 0:
                return []
            q = _l2_normalize(query)
            sims = self._centroid_matrix @ q  # (n_centroids,)

            # Per-person max: rows are person-ordered, find segment boundaries
            # and use np.maximum.reduceat for a vectorized reduction.
            owners = self._centroid_to_person_idx
            # First index of each person's centroid run:
            segment_starts = np.r_[0, np.where(np.diff(owners) != 0)[0] + 1]
            person_max = np.maximum.reduceat(sims, segment_starts)

            n_persons = len(self._person_ids)
            top_k = min(top_k, n_persons)
            if top_k == 1:
                i = int(np.argmax(person_max))
                return [
                    PersonMatch(
                        person_id=self._person_ids[i],
                        name=self._person_names[i],
                        similarity=float(person_max[i]),
                    )
                ]
            idx = np.argpartition(-person_max, top_k - 1)[:top_k]
            idx = idx[np.argsort(-person_max[idx])]
            return [
                PersonMatch(
                    person_id=self._person_ids[int(i)],
                    name=self._person_names[int(i)],
                    similarity=float(person_max[int(i)]),
                )
                for i in idx
            ]

    def list_persons(self) -> list[tuple[int, str, int]]:
        """Return [(person_id, name, n_embeddings), ...] sorted by id."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT p.id, p.name, COUNT(e.id) AS n
                FROM persons p LEFT JOIN embeddings e ON e.person_id = p.id
                GROUP BY p.id
                ORDER BY p.id
                """
            ).fetchall()
        return [(int(r[0]), r[1], int(r[2])) for r in rows]

    def delete_person(self, person_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            self._conn.commit()
            if cur.rowcount > 0:
                self._dirty = True
                return True
        return False

    def stats(self) -> dict:
        with self._lock:
            n_persons = self._conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            n_embeddings = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "n_persons": int(n_persons),
            "n_embeddings": int(n_embeddings),
            "db_size_mb": round(size_bytes / (1024 * 1024), 3),
            "db_path": str(self.db_path),
            "embedding_dim": EMBEDDING_DIM,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SQLiteFaceDB:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
