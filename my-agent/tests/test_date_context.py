# Offline tests for current-date handling in src/agent.py: a controlled clock
# replaces the real one, and no LLM is called.

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from livekit.agents import llm

from agent import DEFAULT_TIMEZONE, Painthaker, app_timezone, date_note

PORTO_NOVO = ZoneInfo("Africa/Porto-Novo")


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Painthaker() builds its Gemini client; give it a key when none is configured.
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test-key")


def _system_notes(agent: Painthaker, user_text: str) -> str:
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content=user_text)
    turn_ctx = agent.turn_context(chat_ctx)
    return "\n".join(
        item.text_content or ""
        for item in turn_ctx.items
        if item.type == "message" and item.role == "system"
    )


def test_date_note_states_the_clock_date_and_timezone() -> None:
    note = date_note(datetime(2026, 10, 7, 10, 42, tzinfo=PORTO_NOVO))
    assert "Wednesday 7 October 2026, 10:42" in note
    assert "Africa/Porto-Novo, UTC+01:00" in note
    assert "ISO date 2026-10-07" in note
    assert "nothing about current events" in note


def test_date_note_rejects_a_naive_clock() -> None:
    with pytest.raises(ValueError):
        date_note(datetime(2026, 10, 7, 10, 42))


def test_timezone_defaults_to_porto_novo_and_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PAINTHAKER_TIMEZONE", raising=False)
    assert app_timezone().key == DEFAULT_TIMEZONE == "Africa/Porto-Novo"
    monkeypatch.setenv("PAINTHAKER_TIMEZONE", "Europe/Paris")
    assert app_timezone().key == "Europe/Paris"


def test_each_llm_call_gets_the_clock_date() -> None:
    agent = Painthaker(clock=lambda: datetime(2026, 10, 7, 9, 0, tzinfo=PORTO_NOVO))
    notes = _system_notes(agent, "what year are we in?")
    assert "Wednesday 7 October 2026" in notes
    assert "Reply language for this turn: English" in notes


def test_date_refreshes_when_a_session_crosses_midnight() -> None:
    times = iter(
        [
            datetime(2026, 12, 31, 23, 59, 30, tzinfo=PORTO_NOVO),
            datetime(2027, 1, 1, 0, 0, 30, tzinfo=PORTO_NOVO),
        ]
    )
    agent = Painthaker(clock=lambda: next(times))

    before = _system_notes(agent, "On est quel jour ?")
    after = _system_notes(agent, "Et maintenant ?")

    assert "Thursday 31 December 2026" in before
    assert "Friday 1 January 2027" in after


def test_timezone_decides_the_date_near_midnight() -> None:
    # 23:30 UTC on 7 October is already 8 October in Porto-Novo (UTC+1).
    utc_time = datetime(2026, 10, 7, 23, 30, tzinfo=ZoneInfo("UTC"))
    assert "Thursday 8 October 2026, 00:30" in date_note(
        utc_time.astimezone(PORTO_NOVO)
    )


def test_original_chat_context_is_not_modified() -> None:
    agent = Painthaker(clock=lambda: datetime(2026, 10, 7, 9, 0, tzinfo=PORTO_NOVO))
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Bonjour")
    agent.turn_context(chat_ctx)
    assert len(chat_ctx.items) == 1


def test_context_notes_only_when_something_is_missing() -> None:
    clock = lambda: datetime(2026, 10, 7, 9, 0, tzinfo=PORTO_NOVO)  # noqa: E731
    assert "Context note" not in _system_notes(Painthaker(clock=clock), "Bonjour")
    notes = _system_notes(
        Painthaker(clock=clock, hidden_turns=3, missed_user_turn=True), "Bonjour"
    )
    assert "3 earlier exchange(s)" in notes
    assert "never received it" in notes
