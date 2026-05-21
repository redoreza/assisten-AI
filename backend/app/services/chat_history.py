"""Per-person conversation history persistence.

Each enrolled face has a `person_id` from face_database. We log every chat
turn under that id so when Pointer recognizes the user later, it can pull
the most recent conversations and remember context.

Turns with `person_id = NULL` are anonymous (no recognized face) — kept
ephemeral and not loaded on session start.

Schema is intentionally minimal: id, person_id, role, content, ts_iso (ISO 8601 UTC).
The face_database already lives at data/sqlite/faces.db; we use a
separate file `data/sqlite/app.db` (already configured via settings) so
schemas remain decoupled.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from app.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    ts_iso      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chat_turns_person_ts
    ON chat_turns(person_id, ts_iso);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatHistory:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info(f"ChatHistory ready at {self._db_path}")

    def _append_sync(self, person_id: int | None, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO chat_turns(person_id, role, content, ts_iso) "
                "VALUES (?, ?, ?, ?)",
                (person_id, role, content[:8000], _now_iso()),
            )
            self._conn.commit()

    async def append(self, person_id: int | None, role: str, content: str) -> None:
        """Persist a single chat turn. Truncates content >8000 chars."""
        if not content.strip():
            return
        await asyncio.to_thread(self._append_sync, person_id, role, content)

    def _recent_sync(
        self, person_id: int, max_turns: int
    ) -> list[tuple[str, str, str]]:
        """Return up to `max_turns` most recent (role, content, ts_iso) tuples
        for the given person, ordered oldest→newest."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, ts_iso FROM chat_turns "
                "WHERE person_id = ? ORDER BY id DESC LIMIT ?",
                (person_id, max_turns),
            ).fetchall()
        return list(reversed(rows))

    async def recent(
        self, person_id: int, max_turns: int = 12
    ) -> list[tuple[str, str, str]]:
        return await asyncio.to_thread(self._recent_sync, person_id, max_turns)

    def _stats_sync(self) -> dict:
        with self._lock:
            total = self._conn.execute(
                "SELECT COUNT(*) FROM chat_turns"
            ).fetchone()[0]
            anon = self._conn.execute(
                "SELECT COUNT(*) FROM chat_turns WHERE person_id IS NULL"
            ).fetchone()[0]
            unique_persons = self._conn.execute(
                "SELECT COUNT(DISTINCT person_id) FROM chat_turns "
                "WHERE person_id IS NOT NULL"
            ).fetchone()[0]
        return {
            "total_turns": total,
            "anonymous_turns": anon,
            "unique_persons": unique_persons,
            "db_path": str(self._db_path),
        }

    async def stats(self) -> dict:
        return await asyncio.to_thread(self._stats_sync)


_singleton: ChatHistory | None = None


def get_chat_history() -> ChatHistory:
    global _singleton
    if _singleton is None:
        _singleton = ChatHistory(settings.sqlite_full_path)
    return _singleton
