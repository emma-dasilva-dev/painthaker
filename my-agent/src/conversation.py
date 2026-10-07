"""Pure helpers for Painthaker's saved conversations: turns, titles, commands.

A *turn* starts at a user message and contains everything up to the next user
message: the user message, any tool calls with their outputs, and the
assistant's reply. Saving, restoring and the context limit all work on whole
turns, so a tool call is never separated from its output.
"""

import re
from collections.abc import Sequence
from typing import Any

# Context policy: the model sees at most this many of the most recent turns
# (plus the current system instructions). The full transcript stays on disk.
MAX_CONTEXT_TURNS = 20

# The ChatContext item types that make up a conversation. System messages,
# handoffs and config updates are rebuilt by the agent, never saved.
CONVERSATION_TYPES = frozenset({"message", "function_call", "function_call_output"})

COMMANDS = ("new", "list", "resume", "delete", "help")


def _type(item: Any) -> str:
    return item["type"] if isinstance(item, dict) else item.type


def _role(item: Any) -> str | None:
    if _type(item) != "message":
        return None
    return item["role"] if isinstance(item, dict) else item.role


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name)


def text_of(item: Any) -> str:
    """The text of a message item (dict or ChatMessage)."""
    content = _field(item, "content") or []
    return "\n".join(part for part in content if isinstance(part, str))


def is_conversation_item(item: Any) -> bool:
    if _type(item) not in CONVERSATION_TYPES:
        return False
    return _type(item) != "message" or _role(item) in ("user", "assistant")


def split_turns(items: Sequence[Any]) -> list[list[Any]]:
    """Group items into turns. Items before the first user message are dropped."""
    turns: list[list[Any]] = []
    for item in items:
        if _role(item) == "user":
            turns.append([item])
        elif turns:
            turns[-1].append(item)
    return turns


def is_complete_turn(turn: Sequence[Any]) -> bool:
    """True for a user message, matched tool calls/outputs, then an assistant reply.

    Rejects orphaned tool calls (no output: the tool never ran or the turn was
    cut off), outputs without a call, and turns with no final reply.
    """
    if not turn or _role(turn[0]) != "user":
        return False
    pending_calls: set[str] = set()
    for item in turn[1:]:
        kind = _type(item)
        if kind == "function_call":
            pending_calls.add(_field(item, "call_id"))
        elif kind == "function_call_output":
            call_id = _field(item, "call_id")
            if call_id not in pending_calls:
                return False
            pending_calls.remove(call_id)
        elif _role(item) == "user":
            return False
    last = turn[-1]
    return (
        not pending_calls and _role(last) == "assistant" and bool(text_of(last).strip())
    )


def complete_turns(items: Sequence[Any]) -> list[list[Any]]:
    """The complete turns in items; anything incomplete is left out."""
    return [turn for turn in split_turns(items) if is_complete_turn(turn)]


def recent_context(
    items: Sequence[Any], max_turns: int = MAX_CONTEXT_TURNS
) -> list[Any]:
    """System/developer messages plus the last max_turns turns, cut only at user messages."""
    user_positions = [i for i, item in enumerate(items) if _role(item) == "user"]
    if len(user_positions) <= max_turns:
        return list(items)
    cut = user_positions[-max_turns]
    kept_instructions = [
        item for item in items[:cut] if _role(item) in ("system", "developer")
    ]
    return kept_instructions + list(items[cut:])


_CODE_BLOCK = re.compile(r"```.*?(?:```|$)", re.DOTALL)


def make_title(first_message: str, max_length: int = 60) -> str:
    """A local title from the first user message: its first line of prose."""
    prose = _CODE_BLOCK.sub(" ", first_message)
    line = next((line.strip() for line in prose.splitlines() if line.strip()), "")
    if not line:
        code_line = next(
            (
                line.strip()
                for line in first_message.splitlines()
                if line.strip() and not line.strip().startswith("```")
            ),
            "",
        )
        line = f"Code : {code_line}" if code_line else "Conversation"
    line = re.sub(r"\s+", " ", line)
    return line if len(line) <= max_length else line[: max_length - 1].rstrip() + "…"


def parse_command(text: str) -> tuple[str, str | None] | None:
    """('name', argument) for a local command, ('unknown', name) for /word that
    isn't one, or None for a normal message (including multiline text and
    paths such as /etc/passwd)."""
    stripped = text.strip()
    if "\n" in stripped:
        return None
    match = re.fullmatch(r"/([a-z]+)(?:\s+(\S+))?\s*", stripped)
    if not match:
        return None
    name, argument = match.group(1), match.group(2)
    if name not in COMMANDS:
        return ("unknown", name)
    return (name, argument)
