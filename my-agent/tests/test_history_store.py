# Offline tests for src/history.py (SQLite storage). No LLM calls.

import json
import os
import sqlite3
import stat
import uuid
from pathlib import Path

import pytest

import history
from history import HistoryError, HistoryStore, default_db_path, new_conversation_id

CODE = (
    'def salut(nom):\n    """Dit bonjour."""\n\treturn f"Bonjour {nom} 👋 — ça va ?"\n'
)


def _save(store: HistoryStore, cid: str, **kwargs) -> bool:
    return store.append_turn(cid, exchange_id=uuid.uuid4().hex, **kwargs)


def _turn(question: str, answer: str) -> list[dict]:
    return [
        {"type": "message", "role": "user", "content": [question], "id": "u"},
        {"type": "message", "role": "assistant", "content": [answer], "id": "a"},
    ]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "history.sqlite3"


def test_unicode_and_multiline_code_round_trip(db_path: Path) -> None:
    store = HistoryStore(db_path)
    cid = new_conversation_id()
    _save(store, cid, title="Code 👋", language="fr", items=_turn(CODE, "Réponse ✓"))
    assert store.load_items(cid) == _turn(CODE, "Réponse ✓")


def test_persists_after_closing_and_reopening(db_path: Path) -> None:
    store = HistoryStore(db_path)
    cid = new_conversation_id()
    _save(store, cid, title="Premier", language="en", items=_turn("q1", "a1"))
    _save(store, cid, title="Premier", language="en", items=_turn("q2", "a2"))
    store.close()

    reopened = HistoryStore(db_path)
    info = reopened.find(cid[:8])
    assert (info.title, info.language, info.turns) == ("Premier", "en", 2)
    assert reopened.load_items(cid) == _turn("q1", "a1") + _turn("q2", "a2")
    assert reopened.load_items(cid, last_turns=1) == _turn("q2", "a2")


def test_conversations_are_isolated(db_path: Path) -> None:
    store = HistoryStore(db_path)
    first, second = new_conversation_id(), new_conversation_id()
    _save(store, first, title="A", language="fr", items=_turn("a?", "a!"))
    _save(store, second, title="B", language="fr", items=_turn("b?", "b!"))
    assert store.load_items(first) == _turn("a?", "a!")
    assert store.load_items(second) == _turn("b?", "b!")
    assert {info.title for info in store.list_conversations()} == {"A", "B"}


def test_delete_removes_the_conversation_and_its_items(db_path: Path) -> None:
    store = HistoryStore(db_path)
    keep, drop = new_conversation_id(), new_conversation_id()
    _save(store, keep, title="keep", language="fr", items=_turn("k", "k"))
    _save(store, drop, title="drop", language="fr", items=_turn("d", "d"))
    assert store.delete(drop) is True
    assert store.load_items(drop) == []
    assert [info.id for info in store.list_conversations()] == [keep]
    raw = sqlite3.connect(db_path)
    assert raw.execute(
        "SELECT COUNT(*) FROM items WHERE conversation_id = ?", (drop,)
    ).fetchone() == (0,)


def test_failed_save_writes_nothing(db_path: Path) -> None:
    store = HistoryStore(db_path)
    cid = new_conversation_id()
    _save(store, cid, title="t", language="fr", items=_turn("q1", "a1"))
    store._conn.execute(
        "CREATE TRIGGER boom BEFORE INSERT ON items WHEN NEW.position = 4 "
        "BEGIN SELECT RAISE(ABORT, 'disk full'); END"
    )
    with pytest.raises(HistoryError, match="disk full"):
        _save(store, cid, title="t", language="en", items=_turn("q2", "a2"))
    assert store.load_items(cid) == _turn("q1", "a1")
    assert store.find(cid[:8]).language == "fr"


def test_find_rejects_unknown_ambiguous_or_malformed_ids(db_path: Path) -> None:
    store = HistoryStore(db_path)
    with pytest.raises(HistoryError, match="No saved conversation"):
        store.find("abcd1234")
    with pytest.raises(HistoryError, match="at least 4"):
        store.find("ab")
    with pytest.raises(HistoryError):
        store.find("'; DROP TABLE items; --")
    _save(store, "aaaa0001" + "0" * 24, title="1", language="fr", items=_turn("q", "a"))
    _save(store, "aaaa0002" + "0" * 24, title="2", language="fr", items=_turn("q", "a"))
    with pytest.raises(HistoryError, match="Several"):
        store.find("aaaa")


def test_unreadable_database_is_reported_and_left_unchanged(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True)
    garbage = b"this is not a database" * 100
    db_path.write_bytes(garbage)
    with pytest.raises(HistoryError, match="left unchanged"):
        HistoryStore(db_path)
    assert db_path.read_bytes() == garbage


def test_foreign_sqlite_file_is_not_modified(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True)
    other = sqlite3.connect(db_path)
    other.execute("CREATE TABLE notes (body TEXT)")
    other.commit()
    other.close()
    before = db_path.read_bytes()
    with pytest.raises(HistoryError, match="isn't a Painthaker"):
        HistoryStore(db_path)
    assert db_path.read_bytes() == before


def test_newer_schema_is_refused(db_path: Path) -> None:
    HistoryStore(db_path).close()
    raw = sqlite3.connect(db_path)
    raw.execute("PRAGMA user_version = 99")
    raw.close()
    with pytest.raises(HistoryError, match="newer version"):
        HistoryStore(db_path)


def test_new_database_is_private_to_the_user(db_path: Path) -> None:
    HistoryStore(db_path).close()
    assert stat.S_IMODE(os.stat(db_path).st_mode) == 0o600


def test_default_location_and_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("PAINTHAKER_HISTORY_DB", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert default_db_path() == tmp_path / "xdg" / "painthaker" / "history.sqlite3"
    monkeypatch.delenv("XDG_DATA_HOME")
    assert default_db_path() == Path.home() / ".local/share/painthaker/history.sqlite3"
    monkeypatch.setenv("PAINTHAKER_HISTORY_DB", "~/elsewhere.db")
    assert default_db_path() == Path.home() / "elsewhere.db"


def test_retrying_an_already_saved_exchange_writes_nothing(db_path: Path) -> None:
    store = HistoryStore(db_path)
    cid, exchange = new_conversation_id(), uuid.uuid4().hex
    kwargs = {"title": "t", "language": "fr", "items": _turn("q", "a")}
    assert store.append_turn(cid, exchange_id=exchange, **kwargs) is True
    assert store.append_turn(cid, exchange_id=exchange, **kwargs) is False
    assert store.load_items(cid) == _turn("q", "a")
    assert store.find(cid[:8]).turns == 1


def test_exchange_ids_are_scoped_to_their_conversation(db_path: Path) -> None:
    store = HistoryStore(db_path)
    first, second = new_conversation_id(), new_conversation_id()
    kwargs = {"exchange_id": "same-id", "title": "t", "language": "fr"}
    assert store.append_turn(first, items=_turn("a", "1"), **kwargs)
    assert store.append_turn(second, items=_turn("b", "2"), **kwargs)
    assert store.load_items(second) == _turn("b", "2")


_V1_SCHEMA = """
CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL,
    language TEXT NOT NULL CHECK (language IN ('fr', 'en')),
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE items (conversation_id TEXT NOT NULL REFERENCES conversations(id)
    ON DELETE CASCADE, position INTEGER NOT NULL, turn INTEGER NOT NULL,
    item TEXT NOT NULL, PRIMARY KEY (conversation_id, position));
CREATE INDEX items_by_turn ON items (conversation_id, turn);
PRAGMA user_version = 1;
"""


def test_version_1_database_is_migrated_without_loss(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True)
    old = sqlite3.connect(db_path)
    old.executescript(_V1_SCHEMA)
    old.execute(
        "INSERT INTO conversations VALUES (?, 'Baobab', 'fr', ?, ?)",
        ("a" * 32, "2026-10-07T10:00:00+00:00", "2026-10-07T10:05:00+00:00"),
    )
    rows = [*_turn("8 ordinateurs ?", "Oui."), *_turn("Et le Wi-Fi ?", "Bien.")]
    old.executemany(
        "INSERT INTO items VALUES (?, ?, ?, ?)",
        [
            ("a" * 32, n, (n + 1) // 2, json.dumps(item))
            for n, item in enumerate(rows, 1)
        ],
    )
    old.commit()
    old.close()

    store = HistoryStore(db_path)
    assert store.load_items("a" * 32) == rows
    assert store.find("aaaa").turns == 2
    legacy = store._conn.execute(
        "SELECT id, turn FROM exchanges ORDER BY turn"
    ).fetchall()
    assert legacy == [("legacy-1", 1), ("legacy-2", 2)]
    kwargs = {"exchange_id": "new-1", "title": "Baobab", "language": "fr"}
    assert store.append_turn("a" * 32, items=_turn("Suite ?", "Oui."), **kwargs)
    assert not store.append_turn("a" * 32, items=_turn("Suite ?", "Oui."), **kwargs)
    assert store.find("aaaa").turns == 3
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == 2


def test_new_database_is_private_before_sqlite_first_opens_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o755)
    os.chmod(parent, 0o755)  # an existing, non-private folder
    path = parent / "history.sqlite3"
    modes_at_open: list[int] = []
    real_connect = sqlite3.connect

    def spy(database, *args, **kwargs):
        modes_at_open.append(stat.S_IMODE(os.stat(database).st_mode))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(history.sqlite3, "connect", spy)
    old_umask = os.umask(0o022)
    try:
        HistoryStore(path).close()
    finally:
        os.umask(old_umask)

    assert modes_at_open == [0o600]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(parent).st_mode) == 0o755  # folder untouched


def test_existing_database_keeps_its_mode_and_content(db_path: Path) -> None:
    store = HistoryStore(db_path)
    cid = new_conversation_id()
    _save(store, cid, title="t", language="fr", items=_turn("q", "a"))
    store.close()
    os.chmod(db_path, 0o640)  # chosen by the user; not ours to change

    reopened = HistoryStore(db_path)
    assert stat.S_IMODE(os.stat(db_path).st_mode) == 0o640
    assert reopened.load_items(cid) == _turn("q", "a")
