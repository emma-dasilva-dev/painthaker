# Offline tests for src/conversation.py: turn validation, context limit,
# titles and command parsing.

import pytest

from conversation import (
    complete_turns,
    context_char_budget,
    is_complete_turn,
    is_conversation_id,
    make_title,
    parse_command,
    recent_context,
)


def user(text: str) -> dict:
    return {"type": "message", "role": "user", "content": [text]}


def reply(text: str) -> dict:
    return {"type": "message", "role": "assistant", "content": [text]}


def call(call_id: str) -> dict:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": "inspect_code",
        "arguments": "{}",
    }


def output(call_id: str) -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": "{}",
        "is_error": False,
    }


SYSTEM = {"type": "message", "role": "system", "content": ["instructions"]}


def test_complete_turns_with_and_without_tools() -> None:
    assert is_complete_turn([user("q"), reply("a")])
    assert is_complete_turn([user("q"), call("c1"), output("c1"), reply("a")])
    assert is_complete_turn(
        [user("q"), reply("je regarde"), call("c1"), output("c1"), reply("a")]
    )


def test_rejects_orphaned_or_unanswered_tool_history() -> None:
    assert not is_complete_turn([user("q"), call("c1"), reply("a")])  # tool never ran
    assert not is_complete_turn(
        [user("q"), output("c1"), reply("a")]
    )  # result without call
    assert not is_complete_turn([user("q"), call("c1"), output("c2"), reply("a")])
    assert not is_complete_turn([user("q"), call("c1"), output("c1")])  # no reply
    assert not is_complete_turn([user("q")])  # interrupted before any reply
    assert not is_complete_turn([user("q"), reply("   ")])
    assert not is_complete_turn([reply("a")])


def test_complete_turns_drops_only_the_broken_turn() -> None:
    items = [user("1"), reply("a1"), user("2"), call("c"), user("3"), reply("a3")]
    assert complete_turns(items) == [[user("1"), reply("a1")], [user("3"), reply("a3")]]


def test_recent_context_keeps_whole_turns_and_instructions() -> None:
    items = [
        SYSTEM,
        user("1"),
        reply("a1"),
        user("2"),
        call("c2"),
        output("c2"),
        reply("a2"),
        user("3"),
        call("c3"),
        output("c3"),
        reply("a3"),
    ]
    kept = recent_context(items, max_turns=2)
    assert kept == [SYSTEM, *items[3:]]
    assert recent_context(items, max_turns=5) == items


def test_recent_context_keeps_an_in_progress_turn_intact() -> None:
    items = [user("1"), reply("a1"), user("2"), call("c2"), output("c2")]
    assert recent_context(items, max_turns=1) == [user("2"), call("c2"), output("c2")]


def test_titles_come_from_the_first_line_of_prose() -> None:
    assert make_title("Bonjour, peux-tu vérifier ce code ?\n```python\nx = 1\n```") == (
        "Bonjour, peux-tu vérifier ce code ?"
    )
    assert make_title("```python\nimport os\nos.system(cmd)\n```") == "Code : import os"
    long = make_title("mot " * 40)
    assert len(long) == 60 and long.endswith("…")


def test_command_parsing() -> None:
    assert parse_command("/new") == ("new", None)
    assert parse_command("  /resume 3f2a9c1d ") == ("resume", "3f2a9c1d")
    assert parse_command("/delete abcd") == ("delete", "abcd")
    assert parse_command("/frobnicate") == ("unknown", "frobnicate")


def test_messages_that_look_like_commands_are_still_messages() -> None:
    assert parse_command("/etc/passwd est lisible par tous, c'est grave ?") is None
    assert parse_command("/new\nmais en fait voici mon code") is None
    assert parse_command("Comment marche /resume ?") is None


def test_char_budget_drops_oldest_whole_turns() -> None:
    items = [
        SYSTEM,
        user("q1"),
        reply("x" * 50),
        user("q2"),
        call("c2"),
        output("c2"),
        reply("y" * 50),
        user("q3"),
        reply("z" * 10),
    ]
    # Newest turn = 12 chars; with q2's turn (2 + 14 + 2 + 50 = 68) = 80; q1 adds 52.
    assert recent_context(items, max_turns=20, max_chars=80) == [SYSTEM, *items[3:]]
    assert recent_context(items, max_turns=20, max_chars=79) == [SYSTEM, *items[7:]]
    assert recent_context(items, max_turns=20, max_chars=1000) == items


def test_both_limits_apply_whichever_is_reached_first() -> None:
    items = [user("a"), reply("1"), user("b"), reply("2"), user("c"), reply("3")]
    assert recent_context(items, max_turns=2, max_chars=1000) == items[2:]
    assert recent_context(items, max_turns=20, max_chars=4) == items[2:]


def test_newest_turn_is_kept_even_when_it_alone_exceeds_the_budget() -> None:
    # e.g. a large tool output in the turn being answered: never cut inside it.
    items = [
        user("old"),
        reply("ok"),
        user("q"),
        call("c"),
        output("c") | {"output": "o" * 500},
    ]
    assert recent_context(items, max_turns=20, max_chars=100) == items[2:]


def test_char_budget_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PAINTHAKER_CONTEXT_CHARS", raising=False)
    assert context_char_budget() == 60_000
    monkeypatch.setenv("PAINTHAKER_CONTEXT_CHARS", "120000")
    assert context_char_budget() == 120_000
    for bad in ("0", "-5", "lots", "1e5"):
        monkeypatch.setenv("PAINTHAKER_CONTEXT_CHARS", bad)
        with pytest.raises(ValueError, match="positive integer"):
            context_char_budget()


def test_known_commands_with_malformed_arguments_stay_commands() -> None:
    # Handled locally (usage message) instead of reaching the model.
    assert parse_command("/resume ID") == ("resume", "ID")
    assert parse_command("/resume la conversation sur Baobab") == (
        "resume",
        "la conversation sur Baobab",
    )
    assert parse_command("/delete toutes les conversations") == (
        "delete",
        "toutes les conversations",
    )
    assert parse_command("/new conversation sur les MAC") == (
        "new",
        "conversation sur les MAC",
    )
    assert parse_command("/RESUME 7a073292") == ("resume", "7a073292")


def test_paths_and_pastes_starting_with_a_slash_remain_messages() -> None:
    assert parse_command("/etc/passwd est lisible par tous ?") is None
    assert parse_command("/tmp est plein, que faire ?") is None
    assert parse_command("/usr/bin/python3 -V") is None
    assert parse_command("/resume 7a073292\nvoici aussi mon code :\n    x = 1") is None


def test_conversation_ids() -> None:
    assert is_conversation_id("7a073292")
    assert is_conversation_id("7A07")
    assert not is_conversation_id("ID")
    assert not is_conversation_id("7a0")
    assert not is_conversation_id("la conversation")
    assert not is_conversation_id("7a07; DROP")
