# Offline tests for the LiveKit (voice) entrypoint's preparation: text vs spoken
# output rules, the session note, read-only preferences, turn handling and the
# voice per reply language.
# No LLM, STT or TTS calls, and no LiveKit connection.

import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from livekit.agents import Agent, AgentSession, ModelSettings, inference, llm
from scripted_llm import ScriptedLLM

from agent import (
    LIVEKIT_SESSION_NOTE,
    LUCAS_VOICE,
    PREEMPTIVE_GENERATION,
    SAWYER_VOICE,
    SPOKEN_INSTRUCTIONS,
    TEXT_INSTRUCTIONS,
    VOICE_BY_LANGUAGE,
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


# --- Voice per reply language ---------------------------------------------------


class _RecordingTTS(inference.TTS):
    """The real Inference TTS, recording option changes. It connects only when
    a stream opens, and these tests never open one."""

    def __init__(self) -> None:
        super().__init__(
            "fishaudio/s2.1-pro",
            voice=LUCAS_VOICE,
            language="fr",
            api_key="offline-test-key",
            api_secret="offline-test-secret",
        )
        self.updates: list[dict[str, Any]] = []

    def update_options(self, **options: Any) -> None:
        self.updates.append(options)
        super().update_options(**options)


@dataclass
class _Session:
    output: _Output
    tts: object | None


class _SessionAgent(Painthaker):
    # Agent.session needs a running AgentSession; llm_node only reads its
    # output and tts.
    def __init__(self, session: _Session, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._test_session = session

    @property
    def session(self) -> Any:
        return self._test_session


@pytest.fixture
def llm_requests(monkeypatch: pytest.MonkeyPatch) -> list[llm.ChatContext]:
    # Stands in for the model call: records what llm_node sends it.
    requests: list[llm.ChatContext] = []

    async def fake_llm_node(agent, chat_ctx, tools, model_settings):
        requests.append(chat_ctx)
        yield "ok"

    monkeypatch.setattr(Agent.default, "llm_node", fake_llm_node)
    return requests


async def _reply(agent: Painthaker, chat_ctx: llm.ChatContext) -> None:
    async for _ in agent.llm_node(chat_ctx, [], ModelSettings()):
        pass


def _spoken_session() -> _Session:
    return _Session(output=_Output(audio=object()), tts=_RecordingTTS())


def test_each_reply_language_has_its_native_voice() -> None:
    assert VOICE_BY_LANGUAGE == {"fr": LUCAS_VOICE, "en": SAWYER_VOICE}
    assert LUCAS_VOICE == "f69bca092b674168a8d02d61ca20943c"
    assert SAWYER_VOICE == "fa4c9eb3dccc4806b382b40d61c6b10a"


@pytest.mark.parametrize(
    "message, voice, language, note",
    [
        ("Qu'est-ce qu'une injection SQL ?", LUCAS_VOICE, "fr", "French"),
        ("What is an SQL injection?", SAWYER_VOICE, "en", "English"),
    ],
    ids=["french-lucas", "english-sawyer"],
)
async def test_spoken_reply_uses_the_voice_of_its_language(
    llm_requests: list[llm.ChatContext],
    message: str,
    voice: str,
    language: str,
    note: str,
) -> None:
    session = _spoken_session()
    await _reply(_SessionAgent(session), _with_instructions(message))
    assert session.tts.updates == [{"voice": voice, "language": language}]
    # Applied through the real update_options, as the next stream will read it.
    assert session.tts._opts.voice == voice
    assert session.tts._opts.language == language
    # The reply text gets the same language.
    assert f"Reply language for this turn: {note}" in _system_texts(llm_requests[0])[1]


async def test_voice_follows_language_switches_in_one_session(
    llm_requests: list[llm.ChatContext],
) -> None:
    session = _spoken_session()
    agent = _SessionAgent(session)
    chat_ctx = _with_instructions("Bonjour, qu'est-ce qu'un pare-feu ?")
    for message in [
        None,
        "Can you explain that in English, please?",
        "Merci, et en français ?",
    ]:
        if message:
            chat_ctx.add_message(role="assistant", content="ok")
            chat_ctx.add_message(role="user", content=message)
        await _reply(agent, chat_ctx)
    assert [u["voice"] for u in session.tts.updates] == [
        LUCAS_VOICE,
        SAWYER_VOICE,
        LUCAS_VOICE,
    ]
    assert [u["language"] for u in session.tts.updates] == ["fr", "en", "fr"]
    notes = [_system_texts(request)[1] for request in llm_requests]
    assert "French" in notes[0]
    assert "English" in notes[1]
    assert "French" in notes[2]


async def test_language_preference_also_picks_the_voice(
    llm_requests: list[llm.ChatContext],
) -> None:
    # conversation_language() decides for both: with /prefs language en, a
    # French message still gets an English reply, so Sawyer speaks it.
    session = _spoken_session()
    agent = _SessionAgent(session, preferences=Preferences(language="en"))
    await _reply(agent, _with_instructions("Qu'est-ce qu'un pare-feu ?"))
    assert session.tts.updates == [{"voice": SAWYER_VOICE, "language": "en"}]
    assert "English" in _system_texts(llm_requests[0])[1]


@pytest.mark.parametrize(
    "output",
    [_Output(audio=None), _Output(audio=object(), audio_enabled=False)],
    ids=["no-audio", "audio-off"],
)
async def test_text_reply_leaves_the_voice_alone(
    llm_requests: list[llm.ChatContext], output: _Output
) -> None:
    tts = _RecordingTTS()
    await _reply(
        _SessionAgent(_Session(output=output, tts=tts)),
        _with_instructions("What is an SQL injection?"),
    )
    assert tts.updates == []
    assert tts._opts.voice == LUCAS_VOICE
    system = _system_texts(llm_requests[0])
    assert system[0] == TEXT_INSTRUCTIONS
    assert "English" in system[1]


async def test_spoken_reply_without_an_inference_tts_still_runs(
    llm_requests: list[llm.ChatContext],
) -> None:
    session = _Session(output=_Output(audio=object()), tts=None)
    await _reply(_SessionAgent(session), _with_instructions("Bonjour"))
    assert _system_texts(llm_requests[0])[0] == SPOKEN_INSTRUCTIONS


def test_turn_context_uses_a_language_it_is_given() -> None:
    chat_ctx = _with_instructions("Qu'est-ce qu'un pare-feu ?")
    agent = Painthaker()
    assert "French" in _system_texts(agent.turn_context(chat_ctx))[1]
    given = _system_texts(agent.turn_context(chat_ctx, language="en"))
    assert "Reply language for this turn: English" in given[1]
