"""Pure helpers for Painthaker's saved conversations: turns, titles, commands.

A *turn* starts at a user message and contains everything up to the next user
message: the user message, any tool calls with their outputs, and the
assistant's reply. Saving, restoring and the context limit all work on whole
turns, so a tool call is never separated from its output.
"""

import os
import re
from collections.abc import Sequence
from typing import Any

# Context policy: the model sees the current system instructions plus the most
# recent whole turns, at most MAX_CONTEXT_TURNS of them and at most
# MAX_CONTEXT_CHARS characters of conversation (message text, tool arguments and
# outputs; instructions and notes not counted). Characters, not tokens: no
# tokenizer needed, and roughly 4 characters per token for Gemini, so the
# default is about 15k tokens. The full transcript stays on disk.
MAX_CONTEXT_TURNS = 20
MAX_CONTEXT_CHARS = 60_000

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


def item_chars(item: Any) -> int:
    """Characters an item contributes to the context budget: message text, tool
    names and arguments, tool outputs. System messages aren't counted."""
    kind = _type(item)
    if kind == "message":
        return 0 if _role(item) in ("system", "developer") else len(text_of(item))
    if kind == "function_call":
        return len(_field(item, "name") or "") + len(_field(item, "arguments") or "")
    if kind == "function_call_output":
        return len(_field(item, "output") or "")
    return 0


def context_char_budget() -> int:
    """PAINTHAKER_CONTEXT_CHARS if set (a positive integer), else the default."""
    raw = os.environ.get("PAINTHAKER_CONTEXT_CHARS", "").strip()
    if not raw:
        return MAX_CONTEXT_CHARS
    if not raw.isdigit() or int(raw) <= 0:
        raise ValueError(
            f"PAINTHAKER_CONTEXT_CHARS must be a positive integer, got {raw!r}"
        )
    return int(raw)


def recent_context(
    items: Sequence[Any],
    max_turns: int = MAX_CONTEXT_TURNS,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> list[Any]:
    """System/developer messages plus the most recent whole turns that fit both
    limits. The newest turn is always kept (it's the one being answered);
    older turns are added newest-first until either limit would be exceeded."""
    user_positions = [i for i, item in enumerate(items) if _role(item) == "user"]
    if not user_positions:
        return list(items)
    bounds = [*user_positions, len(items)]
    cut = user_positions[-1]
    used = sum(item_chars(item) for item in items[cut:])
    kept = 1
    for start, end in reversed(list(zip(bounds[:-2], bounds[1:-1], strict=True))):
        size = sum(item_chars(item) for item in items[start:end])
        if kept >= max_turns or used + size > max_chars:
            break
        cut, used, kept = start, used + size, kept + 1
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
