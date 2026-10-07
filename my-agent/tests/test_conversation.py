# Offline tests for src/conversation.py: turn validation, context limit,
# titles and command parsing.

from conversation import (
    complete_turns,
    is_complete_turn,
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
