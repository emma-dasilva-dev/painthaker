"""Local text chat with Painthaker that accepts pasted text and multiline code,
and saves conversations so they can be resumed later.

`lk agent console --text` (lk 2.18.8) drops bracketed pastes and its input is a
single line capped at 1000 characters, so code snippets can't be pasted into it.
This runs the same Painthaker agent in-process in text mode (no LiveKit room,
no STT/TTS) behind a prompt_toolkit input:

- A paste lands in the input for review, newlines and indentation intact, and
  is never sent until you press Enter. Each Enter sends one message.
- Alt+Enter (or Ctrl+J) inserts a newline while typing.
- /new, /list, /resume <id>, /delete <id> and /help are handled locally and
  never sent to the model. Each completed turn is saved to a local SQLite file
  (see history.py); a failed or interrupted turn is discarded, not saved.

Start it with scripts/chat.sh (Ubuntu/WSL).
"""

import asyncio
import contextlib
import logging
import signal
from collections.abc import Awaitable, Callable
from typing import Any

from livekit.agents import AgentSession, llm
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown

from agent import Painthaker, app_timezone, reply_language
from conversation import (
    MAX_CONTEXT_TURNS,
    complete_turns,
    is_complete_turn,
    is_conversation_item,
    make_title,
    parse_command,
    text_of,
)
from history import (
    ConversationInfo,
    HistoryError,
    HistoryStore,
    default_db_path,
    new_conversation_id,
)

KEYS = (
    "Entrée : envoyer · Alt+Entrée : nouvelle ligne · "
    "Ctrl+C : effacer la saisie ou interrompre une réponse · Ctrl+D : quitter"
)
HELP = (
    """\
Commandes (traitées localement, jamais envoyées au modèle) :
  /new            nouvelle conversation
  /list           conversations enregistrées
  /resume <id>    reprendre une conversation (ID affiché par /list)
  /delete <id>    supprimer une conversation (confirmation demandée)
  /help           cette aide
"""
    + KEYS
)

AgentFactory = Callable[..., Painthaker]
Confirm = Callable[[str], Awaitable[str]]


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


def _serialize(items: list[llm.ChatItem]) -> list[dict[str, Any]]:
    return llm.ChatContext(items).to_dict(
        exclude_timestamp=False, exclude_metrics=True
    )["items"]


class ChatApp:
    """The chat loop's logic, separate from the terminal so tests can drive it."""

    def __init__(
        self,
        store: HistoryStore,
        console: Console,
        confirm: Confirm,
        agent_factory: AgentFactory = Painthaker,
        max_turns: int = MAX_CONTEXT_TURNS,
    ) -> None:
        self.store = store
        self.console = console
        self.confirm = confirm
        self.agent_factory = agent_factory
        self.max_turns = max_turns
        self.session: AgentSession | None = None
        self.agent: Painthaker | None = None
        self._reset_state()

    def _reset_state(self) -> None:
        self.conversation_id: str | None = None
        self.title: str | None = None
        self.language = "fr"
        # Valid turns the model can see (serialized), and turns not yet on disk.
        self.turns: list[list[dict[str, Any]]] = []
        self.unsaved: list[tuple[list[dict[str, Any]], str]] = []

    # --- session lifecycle -------------------------------------------------

    async def start(self) -> None:
        await self._start_session()
        try:
            latest = await asyncio.to_thread(self.store.list_conversations, 1)
        except HistoryError as exc:
            self.console.print(str(exc), style="red", markup=False)
            latest = []
        self._print_start_hint(latest)

    async def close(self) -> None:
        await self._save_unsaved()
        if self.unsaved:
            self.console.print(
                f"Attention : {len(self.unsaved)} tour(s) n'ont pas pu être "
                "enregistrés et sont perdus.",
                style="bold red",
            )
        await self._stop_session()

    async def _start_session(self) -> None:
        """(Re)start the agent with the current valid turns as its context."""
        await self._stop_session()
        items = [item for turn in self.turns[-self.max_turns :] for item in turn]
        chat_ctx = llm.ChatContext.from_dict({"items": items})
        self.agent = self.agent_factory(chat_ctx=chat_ctx, language=self.language)
        self.session = AgentSession()
        await self.session.start(self.agent)

    async def _stop_session(self) -> None:
        if self.session is not None:
            await self.session.aclose()
            self.session = None

    # --- input -------------------------------------------------------------

    async def handle_input(self, text: str) -> None:
        if not text.strip():
            return
        command = parse_command(text)
        if command is None:
            await self.send(text)
            return
        name, argument = command
        if name == "unknown":
            self.console.print(f"Commande inconnue : /{argument}. Tapez /help.")
        elif name == "help":
            self.console.print(HELP, markup=False)
        elif name == "new":
            await self.new_conversation()
        elif name == "list":
            await self.list_conversations()
        elif name in ("resume", "delete") and not argument:
            self.console.print(f"Usage : /{name} <id> (voir /list).")
        elif name == "resume":
            await self.resume(argument)
        elif name == "delete":
            await self.delete(argument)

    async def send(self, text: str) -> None:
        assert self.session is not None and self.agent is not None
        seen = {item.id for item in self.session.history.items}
        turn_task = asyncio.ensure_future(self.session.run(user_input=text))
        try:
            with self._interrupt_on_ctrl_c(turn_task):
                result = await turn_task
        except asyncio.CancelledError:
            await self._discard_turn("Réponse interrompue")
            return
        except Exception as exc:
            await self._discard_turn(f"Erreur : {exc}")
            return

        new_items = [
            item
            for item in self.session.history.items
            if item.id not in seen and is_conversation_item(item)
        ]
        if not is_complete_turn(new_items):
            await self._discard_turn(
                "Tour incomplet (outil sans résultat ou pas de réponse)"
            )
            return

        for event in result.events:
            if event.type == "function_call":
                self.console.print(f"  tool: {event.item.name}", style="dim")
            elif event.type == "message" and event.item.role == "assistant":
                self.console.print("Painthaker>", style="bold")
                self.console.print(Markdown(event.item.text_content or ""))

        if self.conversation_id is None:
            self.conversation_id = new_conversation_id()
            self.title = make_title(text)
        turn = _serialize(new_items)
        self.turns.append(turn)
        self.language = reply_language(
            [text_of(item) for item in turn if item.get("role") == "user"],
            initial=self.language,
        )
        self.unsaved.append((turn, self.language))
        await self._save_unsaved()

    async def _discard_turn(self, reason: str) -> None:
        """Drop everything the failed turn added, so the model never sees it."""
        self.console.print(
            f"{reason}. Ce tour n'a pas été enregistré ; vous pouvez renvoyer le message.",
            style="red",
            markup=False,
        )
        await self._start_session()

    async def _save_unsaved(self) -> None:
        while self.unsaved and self.conversation_id is not None:
            turn, language = self.unsaved[0]
            try:
                await asyncio.to_thread(
                    self.store.append_turn,
                    self.conversation_id,
                    title=self.title or "Conversation",
                    language=language,
                    items=turn,
                )
            except HistoryError as exc:
                self.console.print(
                    f"Non enregistré : {exc}\nLe tour reste en mémoire ; "
                    "nouvel essai au prochain message et à la fermeture.",
                    style="bold red",
                    markup=False,
                )
                return
            self.unsaved.pop(0)

    @contextlib.contextmanager
    def _interrupt_on_ctrl_c(self, task: asyncio.Future[Any]) -> Any:
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, task.cancel)
        except (NotImplementedError, RuntimeError, ValueError):
            yield  # no signal support here (tests, non-main thread)
            return
        try:
            yield
        finally:
            loop.remove_signal_handler(signal.SIGINT)

    # --- commands ----------------------------------------------------------

    async def new_conversation(self) -> None:
        await self._save_unsaved()
        self._reset_state()
        await self._start_session()
        self.console.print("Nouvelle conversation.")

    async def list_conversations(self) -> None:
        try:
            conversations = await asyncio.to_thread(self.store.list_conversations)
        except HistoryError as exc:
            self.console.print(str(exc), style="red", markup=False)
            return
        if not conversations:
            self.console.print("Aucune conversation enregistrée.")
            return
        for info in conversations:
            marker = "*" if info.id == self.conversation_id else " "
            self.console.print(
                f"{marker} {info.short_id}  {_local_time(info)}  "
                f"{info.turns:>3} échange(s)  {info.title}",
                markup=False,
            )
        self.console.print("Reprendre : /resume <id>", style="dim")

    async def resume(self, prefix: str) -> None:
        try:
            info = await asyncio.to_thread(self.store.find, prefix)
            items = await asyncio.to_thread(
                self.store.load_items, info.id, last_turns=self.max_turns
            )
        except HistoryError as exc:
            self.console.print(str(exc), style="red", markup=False)
            return
        await self._save_unsaved()
        self._reset_state()
        self.conversation_id, self.title, self.language = (
            info.id,
            info.title,
            info.language,
        )
        self.turns = complete_turns(items)
        await self._start_session()
        self.console.print(
            f"Reprise de « {info.title} » ({info.short_id}, {info.turns} échange(s), "
            f"dernier : {_local_time(info)}).",
            style="bold",
            markup=False,
        )
        if info.turns > len(self.turns):
            self.console.print(
                f"Le modèle voit les {len(self.turns)} derniers échanges ; "
                "la transcription complète reste enregistrée.",
                style="dim",
            )
        self._print_recap()

    async def delete(self, prefix: str) -> None:
        try:
            info = await asyncio.to_thread(self.store.find, prefix)
        except HistoryError as exc:
            self.console.print(str(exc), style="red", markup=False)
            return
        answer = await self.confirm(
            f"Supprimer définitivement « {info.title} » ({info.short_id}, "
            f"{info.turns} échange(s)) ? Tapez « oui » pour confirmer : "
        )
        if answer.strip().lower() not in ("oui", "o", "yes", "y"):
            self.console.print("Suppression annulée.")
            return
        try:
            await asyncio.to_thread(self.store.delete, info.id)
        except HistoryError as exc:
            self.console.print(f"Non supprimée : {exc}", style="red", markup=False)
            return
        self.console.print(f"Conversation {info.short_id} supprimée.")
        if info.id == self.conversation_id:
            self._reset_state()
            await self._start_session()
            self.console.print("Nouvelle conversation.")

    # --- display -----------------------------------------------------------

    def _print_start_hint(self, latest: list[ConversationInfo]) -> None:
        self.console.print(f"Painthaker — nouvelle conversation. {KEYS}", style="dim")
        if latest:
            info = latest[0]
            self.console.print(
                f"Dernière conversation : « {info.title} » — /resume {info.short_id} "
                "· /list pour toutes · /help",
                style="dim",
                markup=False,
            )

    def _print_recap(self, turns: int = 2) -> None:
        for turn in self.turns[-turns:]:
            question = text_of(turn[0])
            lines = question.splitlines()
            shown = "\n".join(lines[:8]) + ("\n…" if len(lines) > 8 else "")
            self.console.print("vous>", style="bold", end=" ")
            self.console.print(shown, markup=False, highlight=False)
            reply = text_of(turn[-1])
            if len(reply) > 800:
                reply = reply[:800].rstrip() + " …"
            self.console.print("Painthaker>", style="bold")
            self.console.print(Markdown(reply))
        self.console.print("— suite de la conversation —", style="dim")


def _local_time(info: ConversationInfo) -> str:
    return info.updated_at.astimezone(app_timezone()).strftime("%Y-%m-%d %H:%M")


async def main() -> None:
    logging.basicConfig(level=logging.ERROR)
    console = Console()
    try:
        store = HistoryStore(default_db_path())
    except HistoryError as exc:
        console.print(
            f"Historique indisponible : {exc}", style="bold red", markup=False
        )
        raise SystemExit(1) from exc

    prompt = build_prompt_session()
    plain_prompt: PromptSession[str] = PromptSession()

    async def confirm(question: str) -> str:
        try:
            return await plain_prompt.prompt_async(question)
        except (KeyboardInterrupt, EOFError):
            return ""

    app = ChatApp(store, console, confirm)
    await app.start()
    try:
        while True:
            try:
                text = await prompt.prompt_async("vous> ")
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            await app.handle_input(text)
    finally:
        await app.close()
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
