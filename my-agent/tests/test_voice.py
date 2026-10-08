# Offline tests for the LiveKit (voice) entrypoint's preparation: text vs spoken
# output rules, the session note, read-only preferences and turn handling.
# No LLM, STT or TTS calls, and no LiveKit connection.

import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest
from livekit.agents import AgentSession, llm
from scripted_llm import ScriptedLLM

from agent import (
    LIVEKIT_SESSION_NOTE,
    PREEMPTIVE_GENERATION,
    SPOKEN_INSTRUCTIONS,
    TEXT_INSTRUCTIONS,
    Painthaker,
    load_saved_preferences,
    speaks_aloud,
)
from history import HistoryError, HistoryStore, read_saved_preferences
from preferences import Preferences


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test-key")


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "history.sqlite3"


def _system_texts(chat_ctx: llm.ChatContext) -> list[str]:
    return [
        item.text_content or ""
        for item in chat_ctx.items
        if item.type == "message" and item.role == "system"
    ]


def _with_instructions(user_text: str) -> llm.ChatContext:
    # As the framework builds it: the agent's instructions first, then the turn.
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content=TEXT_INSTRUCTIONS)
    chat_ctx.add_message(role="user", content=user_text)
    return chat_ctx


# --- Output rules -----------------------------------------------------------


def test_terminal_agent_keeps_the_text_instructions() -> None:
    assert Painthaker().instructions == TEXT_INSTRUCTIONS
    assert "Replies are rendered as Markdown in a" in TEXT_INSTRUCTIONS
    assert "spoken aloud" not in TEXT_INSTRUCTIONS


def test_spoken_variant_differs_only_in_its_output_rules() -> None:
    text_head, text_rules = TEXT_INSTRUCTIONS.split("# Output rules")
    spoken_head, spoken_rules = SPOKEN_INSTRUCTIONS.split("# Output rules")
    assert spoken_head == text_head  # teaching, accuracy, review, notes...
    assert text_rules.split("# Guardrails")[1] == spoken_rules.split("# Guardrails")[1]
    assert "spoken aloud by a text-to-speech voice" in spoken_rules
    assert "No Markdown" in spoken_rules
    assert "never offer or promise to" in spoken_rules
    assert "fenced code blocks" not in spoken_rules


@dataclass
class _Output:
    audio: object | None
    audio_enabled: bool = True


def test_only_an_enabled_audio_output_means_spoken() -> None:
    # Console text mode, the debugger and text simulations have no audio output.
    assert not speaks_aloud(_Output(audio=None))
    assert speaks_aloud(_Output(audio=object()))
    # A frontend can switch audio off for a hybrid session.
    assert not speaks_aloud(_Output(audio=object(), audio_enabled=False))


def test_spoken_turn_swaps_the_instructions_without_touching_the_session() -> None:
    chat_ctx = _with_instructions("Qu'est-ce qu'une injection SQL ?")
    original = chat_ctx.items[0]
    spoken = _system_texts(Painthaker().turn_context(chat_ctx, spoken=True))
    assert spoken[0] == SPOKEN_INSTRUCTIONS
    assert TEXT_INSTRUCTIONS not in spoken
    assert "Reply language for this turn: French" in spoken[1]
    assert chat_ctx.items[0] is original
    assert original.text_content == TEXT_INSTRUCTIONS


def test_text_turn_is_left_as_it_was() -> None:
    chat_ctx = _with_instructions("What is XSS?")
    text = _system_texts(Painthaker().turn_context(chat_ctx))
    assert text[0] == TEXT_INSTRUCTIONS
    assert SPOKEN_INSTRUCTIONS not in text


def test_spoken_rules_are_added_if_the_instructions_message_is_missing() -> None:
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Bonjour")
    spoken = _system_texts(Painthaker().turn_context(chat_ctx, spoken=True))
    assert spoken[0] == SPOKEN_INSTRUCTIONS


async def test_a_session_without_audio_output_gets_the_text_rules() -> None:
    # The same path as the debugger and console text mode: no audio output.
    scripted = ScriptedLLM(["Bonjour !"])
    async with AgentSession() as session:
        await session.start(Painthaker(model=scripted, app_note=LIVEKIT_SESSION_NOTE))
        await session.run(user_input="Bonjour")
    system = _system_texts(scripted.requests[0])
    assert TEXT_INSTRUCTIONS in system
    assert SPOKEN_INSTRUCTIONS not in system
    assert LIVEKIT_SESSION_NOTE in system


# --- Session note -------------------------------------------------------------


def test_session_note_rules_out_saving_resuming_and_sending_code() -> None:
    note = LIVEKIT_SESSION_NOTE
    assert "Nothing in it is saved and it can't be resumed" in note
    assert "/list and /resume <id>" in note
    assert "never offer or promise to" in note
    assert "/prefs in the terminal chat" in note


# --- Read-only preferences ------------------------------------------------------


def _snapshot(directory: Path) -> dict[str, tuple[int, bytes]]:
    return {
        p.name: (p.stat().st_mtime_ns, p.read_bytes())
        for p in directory.iterdir()
        if p.is_file()
    }


def test_missing_database_is_not_created(db_path: Path) -> None:
    assert read_saved_preferences(db_path) is None
    assert not db_path.parent.exists()


def test_saved_values_are_read_without_changing_the_file(db_path: Path) -> None:
    store = HistoryStore(db_path)
    store.save_preference("language", "en")
    store.save_preference("length", "detailed")
    store.close()
    before = _snapshot(db_path.parent)
    assert read_saved_preferences(db_path) == {"language": "en", "length": "detailed"}
    assert _snapshot(db_path.parent) == before  # no journal left, same bytes


def test_database_from_before_prefs_reads_as_no_preferences(db_path: Path) -> None:
    store = HistoryStore(db_path)
    store.close()
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE preferences")
    conn.close()
    before = _snapshot(db_path.parent)
    assert read_saved_preferences(db_path) == {}
    assert _snapshot(db_path.parent) == before  # the table isn't added back


def test_empty_file_reads_as_no_preferences(db_path: Path) -> None:
    db_path.parent.mkdir()
    db_path.touch()
    assert read_saved_preferences(db_path) == {}
    assert db_path.stat().st_size == 0


@pytest.mark.parametrize(
    "setup, message",
    [
        (lambda p: p.write_bytes(b"not a database" * 100), "Couldn't read"),
        (
            lambda p: sqlite3.connect(p).execute("CREATE TABLE notes (x)"),
            "isn't a Painthaker history database",
        ),
        (
            lambda p: sqlite3.connect(p).execute("PRAGMA user_version = 99"),
            "written by a newer version",
        ),
    ],
    ids=["corrupt", "foreign", "newer"],
)
def test_unreadable_database_is_reported_and_left_unchanged(
    db_path: Path, setup, message: str
) -> None:
    db_path.parent.mkdir()
    setup(db_path)
    before = _snapshot(db_path.parent)
    with pytest.raises(HistoryError, match=message):
        read_saved_preferences(db_path)
    assert _snapshot(db_path.parent) == before


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_file_permissions_are_reported(db_path: Path) -> None:
    HistoryStore(db_path).close()
    db_path.chmod(0o000)
    try:
        with pytest.raises(HistoryError):
            read_saved_preferences(db_path)
    finally:
        db_path.chmod(0o600)


async def test_voice_session_uses_saved_preferences(db_path: Path) -> None:
    store = HistoryStore(db_path)
    store.save_preference("level", "advanced")
    store.close()
    assert await load_saved_preferences(db_path) == Preferences(level="advanced")


async def test_voice_session_uses_defaults_without_a_database(
    db_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="agent"):
        assert await load_saved_preferences(db_path) == Preferences()
    assert not db_path.exists()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_voice_session_reports_a_database_error_and_uses_defaults(
    db_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    db_path.parent.mkdir()
    db_path.write_bytes(b"not a database" * 100)
    with caplog.at_level(logging.INFO, logger="agent"):
        assert await load_saved_preferences(db_path) == Preferences()
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert str(db_path) in errors[0].getMessage()


# --- Turn handling --------------------------------------------------------------


def test_preemptive_generation_is_off_for_the_first_voice_tests() -> None:
    assert PREEMPTIVE_GENERATION == {"enabled": False}
