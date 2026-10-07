"""Local, unencrypted SQLite storage for Painthaker's chat history.

One row per conversation, plus one row per saved ChatContext item (as JSON).
Only complete turns are written, each in a single transaction, so the file
never holds half a turn. System instructions and the injected language/date
notes are never stored: the agent rebuilds them on every call.

Location: $PAINTHAKER_HISTORY_DB if set, else
$XDG_DATA_HOME/painthaker/history.sqlite3 (default ~/.local/share/...).
"""

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_LANGUAGES = ("fr", "en")

_SCHEMA = """
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    language TEXT NOT NULL CHECK (language IN ('fr', 'en')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE items (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    turn INTEGER NOT NULL,
    item TEXT NOT NULL,
    PRIMARY KEY (conversation_id, position)
);
CREATE INDEX items_by_turn ON items (conversation_id, turn);
"""


class HistoryError(Exception):
    """The history database can't be used; the message says what to do."""


@dataclass(frozen=True)
class ConversationInfo:
    id: str
    title: str
    language: str
    created_at: datetime
    updated_at: datetime
    turns: int

    @property
    def short_id(self) -> str:
        return self.id[:8]


def default_db_path() -> Path:
    override = os.environ.get("PAINTHAKER_HISTORY_DB")
    if override:
        return Path(override).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data_home) / "painthaker" / "history.sqlite3"


def new_conversation_id() -> str:
    return uuid.uuid4().hex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class HistoryStore:
    """Thread-safe (one lock) so callers can run each call in a worker thread."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        is_new = not path.exists()
        try:
            self._conn = sqlite3.connect(
                path, isolation_level=None, check_same_thread=False, timeout=5
            )
        except sqlite3.Error as exc:
            raise HistoryError(
                f"Can't open the history database {path}: {exc}"
            ) from exc
        try:
            self._prepare(is_new)
        except (sqlite3.Error, HistoryError) as exc:
            self._conn.close()
            if isinstance(exc, HistoryError):
                raise
            raise HistoryError(
                f"The history database {path} is unreadable ({exc}). It was left "
                "unchanged. Move it aside, or set PAINTHAKER_HISTORY_DB to another file."
            ) from exc
        if is_new:
            os.chmod(path, 0o600)

    def _prepare(self, is_new: bool) -> None:
        conn = self._conn
        conn.execute("PRAGMA foreign_keys = ON")
        # Zero deleted content inside this file. Not secure erasure: the rollback
        # journal is deleted unwritten, and filesystem/SSD/backups may keep copies.
        conn.execute("PRAGMA secure_delete = ON")
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise sqlite3.DatabaseError("integrity check failed")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            return
        if version > SCHEMA_VERSION:
            raise HistoryError(
                f"The history database {self.path} was written by a newer version "
                f"(schema {version}); this version reads schema {SCHEMA_VERSION}. "
                "It was left unchanged."
            )
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        if tables and not is_new:
            raise HistoryError(
                f"{self.path} is a SQLite file that isn't a Painthaker history "
                "database. It was left unchanged; set PAINTHAKER_HISTORY_DB to "
                "another file."
            )
        with self._transaction():
            for statement in _SCHEMA.split(";"):
                if statement.strip():
                    conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def append_turn(
        self,
        conversation_id: str,
        *,
        title: str,
        language: str,
        items: list[dict[str, Any]],
    ) -> None:
        """Save one complete turn; creates the conversation on its first turn."""
        if language not in _LANGUAGES:
            raise ValueError(f"unsupported language {language!r}")
        encoded = [json.dumps(item, ensure_ascii=False) for item in items]
        now = _now()
        with self._lock, self._wrap_errors("save the turn"), self._transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO conversations"
                " (id, title, language, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, title, language, now, now),
            )
            turn, position = db.execute(
                "SELECT COALESCE(MAX(turn), 0) + 1, COALESCE(MAX(position), 0)"
                " FROM items WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            db.executemany(
                "INSERT INTO items (conversation_id, position, turn, item)"
                " VALUES (?, ?, ?, ?)",
                [
                    (conversation_id, position + offset, turn, item)
                    for offset, item in enumerate(encoded, start=1)
                ],
            )
            db.execute(
                "UPDATE conversations SET language = ?, updated_at = ? WHERE id = ?",
                (language, now, conversation_id),
            )

    def list_conversations(self, limit: int = 20) -> list[ConversationInfo]:
        with self._lock, self._wrap_errors("list conversations"):
            rows = self._conn.execute(
                "SELECT c.id, c.title, c.language, c.created_at, c.updated_at,"
                " (SELECT COUNT(DISTINCT turn) FROM items WHERE conversation_id = c.id)"
                " FROM conversations c ORDER BY c.updated_at DESC, c.rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._info(row) for row in rows]

    def count(self) -> int:
        with self._lock, self._wrap_errors("count conversations"):
            return self._conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[
                0
            ]

    def find(self, prefix: str) -> ConversationInfo:
        """The conversation whose ID starts with prefix (at least 4 characters)."""
        prefix = prefix.strip().lower()
        if len(prefix) < 4 or not all(c in "0123456789abcdef" for c in prefix):
            raise HistoryError("Use the ID shown by /list (at least 4 characters).")
        with self._lock, self._wrap_errors("find the conversation"):
            rows = self._conn.execute(
                "SELECT c.id, c.title, c.language, c.created_at, c.updated_at,"
                " (SELECT COUNT(DISTINCT turn) FROM items WHERE conversation_id = c.id)"
                " FROM conversations c WHERE c.id LIKE ? || '%' LIMIT 2",
                (prefix,),
            ).fetchall()
        if not rows:
            raise HistoryError(f"No saved conversation with ID {prefix}.")
        if len(rows) > 1:
            raise HistoryError(
                f"Several conversations start with {prefix}; use more characters."
            )
        return self._info(rows[0])

    def load_items(
        self, conversation_id: str, *, last_turns: int | None = None
    ) -> list[dict[str, Any]]:
        """Saved items in order; with last_turns, only the most recent turns."""
        with self._lock, self._wrap_errors("load the conversation"):
            rows = self._conn.execute(
                "SELECT item FROM items WHERE conversation_id = ?"
                " AND turn > (SELECT COALESCE(MAX(turn), 0) FROM items"
                "             WHERE conversation_id = ?) - ?"
                " ORDER BY position",
                (conversation_id, conversation_id, last_turns or 2**62),
            ).fetchall()
        try:
            return [json.loads(row[0]) for row in rows]
        except json.JSONDecodeError as exc:
            raise HistoryError(
                f"A saved message in {conversation_id[:8]} is corrupt: {exc}"
            ) from exc

    def delete(self, conversation_id: str) -> bool:
        with (
            self._lock,
            self._wrap_errors("delete the conversation"),
            self._transaction() as db,
        ):
            deleted = db.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
            return deleted.rowcount > 0

    @contextmanager
    def _wrap_errors(self, action: str) -> Iterator[None]:
        try:
            yield
        except sqlite3.Error as exc:
            raise HistoryError(f"Couldn't {action} in {self.path}: {exc}") from exc

    @staticmethod
    def _info(row: tuple[Any, ...]) -> ConversationInfo:
        return ConversationInfo(
            id=row[0],
            title=row[1],
            language=row[2],
            created_at=datetime.fromisoformat(row[3]),
            updated_at=datetime.fromisoformat(row[4]),
            turns=row[5],
        )
