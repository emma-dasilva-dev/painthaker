"""Local text chat with Painthaker that accepts pasted text and multiline code.

`lk agent console --text` (lk 2.18.8) drops bracketed pastes and its input is a
single line capped at 1000 characters, so code snippets can't be pasted into it.
This runs the same Painthaker agent in-process in text mode (no LiveKit room,
no STT/TTS) behind a prompt_toolkit input:

- A paste lands in the input for review, newlines and indentation intact, and
  is never sent until you press Enter. Each Enter sends one message.
- Alt+Enter (or Ctrl+J) inserts a newline while typing.

Run from the my-agent directory so .env.local is found:

    uv run --no-sync python src/chat.py
"""

import asyncio
import logging
from typing import Any

from livekit.agents import AgentSession
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown

from agent import Painthaker

HELP = (
    "Painthaker — Entrée : envoyer · Alt+Entrée : nouvelle ligne · "
    "Ctrl+C : effacer la saisie · Ctrl+D : quitter"
)


def build_prompt_session(**kwargs: Any) -> PromptSession[str]:
    bindings = KeyBindings()

    @bindings.add("enter")
    def _send(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    @bindings.add("c-j")
    def _newline(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    # multiline=True lets the buffer hold and display several lines; the bindings
    # above make Enter send instead of inserting a newline. Pasted text arrives
    # as one bracketed-paste event, so its newlines never trigger Enter.
    return PromptSession(
        multiline=True,
        key_bindings=bindings,
        prompt_continuation=lambda width, line_number, is_soft_wrap: " " * width,
        **kwargs,
    )


async def main() -> None:
    logging.basicConfig(level=logging.ERROR)
    console = Console()
    prompt = build_prompt_session()

    async with AgentSession() as session:
        await session.start(Painthaker())
        console.print(HELP, style="dim")

        while True:
            try:
                text = await prompt.prompt_async("vous> ")
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            if not text.strip():
                continue

            try:
                result = await session.run(user_input=text)
            except Exception as exc:
                console.print(f"Erreur : {exc}", style="red", markup=False)
                continue

            for event in result.events:
                if event.type == "function_call":
                    console.print(f"  tool: {event.item.name}", style="dim")
                elif event.type == "message" and event.item.role == "assistant":
                    console.print("Painthaker>", style="bold")
                    console.print(Markdown(event.item.text_content or ""))


if __name__ == "__main__":
    asyncio.run(main())
