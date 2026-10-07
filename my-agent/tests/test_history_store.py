# Offline tests for src/history.py (SQLite storage). No LLM calls.

import os
import sqlite3
import stat
from pathlib import Path

import pytest

from history import HistoryError, HistoryStore, default_db_path, new_conversation_id

CODE = (
    'def salut(nom):\n    """Dit bonjour."""\n\treturn f"Bonjour {nom} 👋 — ça va ?"\n'
)


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
    store.append_turn(
        cid, title="Code 👋", language="fr", items=_turn(CODE, "Réponse ✓")
    )
    assert store.load_items(cid) == _turn(CODE, "Réponse ✓")


def test_persists_after_closing_and_reopening(db_path: Path) -> None:
    store = HistoryStore(db_path)
    cid = new_conversation_id()
    store.append_turn(cid, title="Premier", language="en", items=_turn("q1", "a1"))
    store.append_turn(cid, title="Premier", language="en", items=_turn("q2", "a2"))
    store.close()

    reopened = HistoryStore(db_path)
    info = reopened.find(cid[:8])
    assert (info.title, info.language, info.turns) == ("Premier", "en", 2)
    assert reopened.load_items(cid) == _turn("q1", "a1") + _turn("q2", "a2")
    assert reopened.load_items(cid, last_turns=1) == _turn("q2", "a2")


def test_conversations_are_isolated(db_path: Path) -> None:
    store = HistoryStore(db_path)
    first, second = new_conversation_id(), new_conversation_id()
    store.append_turn(first, title="A", language="fr", items=_turn("a?", "a!"))
    store.append_turn(second, title="B", language="fr", items=_turn("b?", "b!"))
    assert store.load_items(first) == _turn("a?", "a!")
    assert store.load_items(second) == _turn("b?", "b!")
    assert {info.title for info in store.list_conversations()} == {"A", "B"}


def test_delete_removes_the_conversation_and_its_items(db_path: Path) -> None:
    store = HistoryStore(db_path)
    keep, drop = new_conversation_id(), new_conversation_id()
    store.append_turn(keep, title="keep", language="fr", items=_turn("k", "k"))
    store.append_turn(drop, title="drop", language="fr", items=_turn("d", "d"))
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
    store.append_turn(cid, title="t", language="fr", items=_turn("q1", "a1"))
    store._conn.execute(
        "CREATE TRIGGER boom BEFORE INSERT ON items WHEN NEW.position = 4 "
        "BEGIN SELECT RAISE(ABORT, 'disk full'); END"
    )
    with pytest.raises(HistoryError, match="disk full"):
        store.append_turn(cid, title="t", language="en", items=_turn("q2", "a2"))
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
    store.append_turn(
        "aaaa0001" + "0" * 24, title="1", language="fr", items=_turn("q", "a")
    )
    store.append_turn(
        "aaaa0002" + "0" * 24, title="2", language="fr", items=_turn("q", "a")
    )
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
