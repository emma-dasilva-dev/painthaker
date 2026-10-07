# Offline tests for user preferences: the model (src/preferences.py) and their
# storage in the history database (src/history.py). No LLM calls.

import sqlite3
import uuid
from pathlib import Path

import pytest

from history import HistoryError, HistoryStore, new_conversation_id
from preferences import Preferences, preferences_note


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "history.sqlite3"


def test_defaults_when_nothing_is_saved(db_path: Path) -> None:
    store = HistoryStore(db_path)
    assert store.load_preferences() == {}
    assert Preferences.from_stored({}) == Preferences("auto", "beginner", "concise")


def test_saved_values_survive_closing_and_reopening(db_path: Path) -> None:
    store = HistoryStore(db_path)
    store.save_preference("language", "en")
    store.save_preference("level", "advanced")
    store.save_preference("level", "intermediate")  # overwrite, not duplicate
    store.close()
    stored = HistoryStore(db_path).load_preferences()
    assert stored == {"language": "en", "level": "intermediate"}
    assert Preferences.from_stored(stored) == Preferences(
        "en", "intermediate", "concise"
    )


def test_reset_removes_every_saved_value(db_path: Path) -> None:
    store = HistoryStore(db_path)
    store.save_preference("length", "detailed")
    store.reset_preferences()
    assert store.load_preferences() == {}


def test_unknown_or_invalid_stored_values_fall_back_to_defaults() -> None:
    stored = {"language": "de", "level": "advanced", "colour": "blue"}
    assert Preferences.from_stored(stored) == Preferences("auto", "advanced", "concise")
    with pytest.raises(ValueError):
        Preferences().with_value("language", "de")


def test_existing_database_gains_preferences_and_keeps_conversations(
    db_path: Path,
) -> None:
    # A database written before this feature: no preferences table.
    store = HistoryStore(db_path)
    cid = new_conversation_id()
    turn = [
        {"type": "message", "role": "user", "content": ["q"]},
        {"type": "message", "role": "assistant", "content": ["a"]},
    ]
    store.append_turn(
        cid, exchange_id=uuid.uuid4().hex, title="t", language="fr", items=turn
    )
    store._conn.execute("DROP TABLE preferences")
    store.close()

    reopened = HistoryStore(db_path)
    assert reopened.load_preferences() == {}
    assert reopened.load_items(cid) == turn
    assert reopened._conn.execute("PRAGMA user_version").fetchone()[0] == 2


def test_failed_save_raises_and_keeps_the_previous_value(db_path: Path) -> None:
    store = HistoryStore(db_path)
    store.save_preference("length", "concise")
    store._conn.execute(
        "CREATE TRIGGER no_writes BEFORE UPDATE ON preferences "
        "BEGIN SELECT RAISE(ABORT, 'disk full'); END"
    )
    with pytest.raises(HistoryError, match="disk full"):
        store.save_preference("length", "detailed")
    assert store.load_preferences() == {"length": "concise"}


def test_preferences_note_reflects_values_and_keeps_correctness() -> None:
    concise = preferences_note(Preferences())
    assert "a beginner" in concise and "concise" in concise
    assert "Never drop a caveat, a security warning or a step" in concise
    assert "for this reply only" in concise and "does not change" in concise
    detailed = preferences_note(Preferences("en", "advanced", "detailed"))
    assert "an advanced practitioner" in detailed and "detailed" in detailed
    # Language is set by its own note; this one must not compete with it.
    assert "language" not in detailed.lower() and "French" not in detailed


def test_raw_sqlite_has_no_personal_columns(db_path: Path) -> None:
    HistoryStore(db_path).close()
    columns = [
        row[1]
        for row in sqlite3.connect(db_path).execute("PRAGMA table_info(preferences)")
    ]
    assert columns == ["key", "value", "updated_at"]
