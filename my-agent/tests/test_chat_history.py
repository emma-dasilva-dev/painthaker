# Offline end-to-end tests of the chat's history behavior: ChatApp drives the
# real AgentSession and Painthaker agent, with a scripted LLM instead of Gemini.

import asyncio
import io
import os
import signal
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from livekit.agents import llm
from rich.console import Console
from scripted_llm import Hang, ScriptedFailureError, ScriptedLLM, ToolCall

from agent import MISSED_TURN_NOTE, Painthaker
from chat import TERMINAL_CHAT_NOTE, ChatApp
from history import HistoryError, HistoryStore

CODE = "import os\n\ndef run(cmd):\n    os.system(cmd)  # indentation kept\n"
CLOCK = datetime(2026, 10, 7, 9, 30, tzinfo=ZoneInfo("Africa/Porto-Novo"))


class Harness:
    def __init__(self, db: Path, steps: list, answers: list[str] | None = None) -> None:
        self.llm = ScriptedLLM(steps)
        self.answers = list(answers or [])
        self.questions: list[str] = []
        self.output = io.StringIO()
        self.clock: Callable[[], datetime] = lambda: CLOCK
        self.store = HistoryStore(db)
        self.app = ChatApp(
            self.store,
            Console(file=self.output, width=120, color_system=None),
            self._confirm,
            agent_factory=lambda **kw: Painthaker(self.clock, model=self.llm, **kw),
        )

    async def _confirm(self, question: str) -> str:
        self.questions.append(question)
        return self.answers.pop(0)

    def printed(self) -> str:
        # Whitespace-normalized: the console wraps long lines.
        return " ".join(self.output.getvalue().split())

    def agent_items(self) -> list[llm.ChatItem]:
        return [
            item
            for item in self.app.agent.chat_ctx.items
            if not (item.type == "message" and item.role == "system")
            and item.type != "agent_config_update"
        ]

    async def close(self) -> None:
        await self.app.close()
        self.store.close()


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "offline-test-key")


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "history.sqlite3"


def _texts(items: list) -> list[tuple[str, str]]:
    out = []
    for item in items:
        kind = item["type"] if isinstance(item, dict) else item.type
        if kind == "message":
            role = item["role"] if isinstance(item, dict) else item.role
            content = item["content"] if isinstance(item, dict) else item.content
            out.append((role, "".join(c for c in content if isinstance(c, str))))
        else:
            out.append((kind, ""))
    return out


async def test_resume_after_restart_restores_context_language_and_fresh_date(
    db: Path,
) -> None:
    first = Harness(db, ["Noted.", "Sure, Falcon-7."])
    await first.app.start()
    assert "nouvelle conversation" in first.printed()
    await first.app.handle_input(
        "Can we continue in English please? My test robot is Rivet."
    )
    await first.app.handle_input("ok, and the code word is Falcon-7")
    conversation_id = first.app.conversation_id
    await first.close()

    second = Harness(db, ["Your robot is Rivet."])
    second.clock = lambda: datetime(
        2026, 10, 8, 0, 5, tzinfo=ZoneInfo("Africa/Porto-Novo")
    )
    await second.app.start()
    assert "Dernière conversation" in second.printed()
    await second.app.handle_input("/resume " + conversation_id[:8])
    assert "Reprise de" in second.printed() and "Falcon-7" in second.printed()
    assert second.app.language == "en"

    await second.app.handle_input("ok")  # no language cue: the saved English must hold
    sent = second.llm.requests[-1]
    system = "\n".join(
        item.text_content or ""
        for item in sent.items
        if item.type == "message" and item.role == "system"
    )
    assert "Reply language for this turn: English" in system
    assert "Thursday 8 October 2026, 00:05" in system  # fresh clock, not the saved one
    assert "Painthaker" in system  # current instructions applied
    assert (
        "user",
        "Can we continue in English please? My test robot is Rivet.",
    ) in _texts(sent.items)
    await second.close()

    stored = HistoryStore(db).load_items(conversation_id)
    assert [role for role, _ in _texts(stored)].count(
        "system"
    ) == 0  # no prompts or date notes saved
    assert len([i for i in stored if i.get("role") == "user"]) == 3


async def test_tool_turn_is_saved_with_matching_call_and_output(db: Path) -> None:
    h = Harness(
        db,
        [
            ToolCall(
                "inspect_code", '{"code": "eval(x)", "language": "python"}', "call-1"
            ),
            "Pattern detected.",
        ],
    )
    await h.app.start()
    await h.app.handle_input("Review:\n```python\n" + CODE + "```")
    stored = h.store.load_items(h.app.conversation_id)
    kinds = [item["type"] for item in stored]
    assert kinds == ["message", "function_call", "function_call_output", "message"]
    assert stored[1]["call_id"] == stored[2]["call_id"] == "call-1"
    assert CODE in stored[0]["content"][0]  # multiline code and indentation intact
    assert "tool: inspect_code" in h.printed()
    await h.close()


async def test_failed_turn_is_not_saved_and_leaves_no_trace_in_context(
    db: Path,
) -> None:
    h = Harness(
        db, ["First answer.", ScriptedFailureError("network down"), "Second answer."]
    )
    await h.app.start()
    await h.app.handle_input("Première question")
    await h.app.handle_input("Question qui échoue")
    assert "pas été enregistré" in h.printed()
    assert [t for _, t in _texts(h.agent_items())] == [
        "Première question",
        "First answer.",
    ]

    await h.app.handle_input("Deuxième question")
    stored = h.store.load_items(h.app.conversation_id)
    assert [t for _, t in _texts(stored)] == [
        "Première question",
        "First answer.",
        "Deuxième question",
        "Second answer.",
    ]
    sent_users = [t for r, t in _texts(h.llm.requests[-1].items) if r == "user"]
    assert "Question qui échoue" not in sent_users
    await h.close()


async def test_tool_call_without_a_final_answer_is_discarded(db: Path) -> None:
    # The model asks for a tool, the tool runs, then the follow-up call fails:
    # the call/output pair must not be saved without a reply.
    h = Harness(
        db,
        [
            ToolCall("inspect_code", '{"code": "x", "language": "python"}', "c9"),
            ScriptedFailureError("timeout"),
        ],
    )
    await h.app.start()
    await h.app.handle_input("Vérifie ce code : x")
    assert h.app.conversation_id is None
    assert h.store.list_conversations() == []
    assert h.agent_items() == []
    await h.close()


async def test_first_turn_creates_the_record_and_commands_never_reach_the_model(
    db: Path,
) -> None:
    h = Harness(db, ["Réponse."])
    await h.app.start()
    for command in ("/help", "/list", "/new", "/frobnicate"):
        await h.app.handle_input(command)
    assert h.llm.requests == []
    assert h.store.list_conversations() == []
    assert "Commande inconnue" in h.printed() and "/resume <id>" in h.printed()

    await h.app.handle_input("Bonjour, qu'est-ce qu'un CVE ?")
    [info] = h.store.list_conversations()
    assert info.title == "Bonjour, qu'est-ce qu'un CVE ?"
    assert len(h.llm.requests) == 1
    await h.close()


async def test_delete_requires_confirmation(db: Path) -> None:
    h = Harness(db, ["A.", "B."], answers=["non", "oui"])
    await h.app.start()
    await h.app.handle_input("Première conversation")
    keep_id = h.app.conversation_id
    await h.app.handle_input("/new")
    await h.app.handle_input("Seconde conversation")
    current_id = h.app.conversation_id

    await h.app.handle_input("/delete " + current_id[:8])
    assert "Suppression annulée" in h.printed()
    assert len(h.store.list_conversations()) == 2

    await h.app.handle_input("/delete " + current_id[:8])
    assert "supprimée" in h.printed()
    assert [i.id for i in h.store.list_conversations()] == [keep_id]
    assert (
        h.app.conversation_id is None
    )  # deleting the open conversation starts a new one
    assert all("Supprimer définitivement" in q for q in h.questions)
    await h.close()


async def test_resume_limits_model_context_to_recent_complete_turns(db: Path) -> None:
    h = Harness(db, [f"a{n}" for n in range(1, 5)] + ["fin"])
    h.app.max_turns = 2
    await h.app.start()
    for n in range(1, 5):
        await h.app.handle_input(f"q{n}")
    cid = h.app.conversation_id
    await h.app.handle_input("/new")
    await h.app.handle_input("/resume " + cid[:8])
    assert [t for _, t in _texts(h.agent_items())] == ["q3", "a3", "q4", "a4"]
    assert "derniers échanges" in h.printed()
    assert len(h.store.load_items(cid)) == 8  # the full transcript stays on disk
    await h.close()


async def test_ctrl_c_during_a_reply_interrupts_without_saving(db: Path) -> None:
    h = Harness(db, [Hang(), "After the interruption."])
    await h.app.start()
    turn = asyncio.ensure_future(h.app.handle_input("Une question très longue"))
    while not h.llm.requests:  # wait until the model call is in flight
        await asyncio.sleep(0.01)
    os.kill(os.getpid(), signal.SIGINT)  # what Ctrl+C sends
    await asyncio.wait_for(turn, timeout=10)

    assert "Réponse interrompue" in h.printed()
    assert h.store.list_conversations() == []
    assert h.agent_items() == []

    await h.app.handle_input("Nouvelle tentative")
    stored = h.store.load_items(h.app.conversation_id)
    assert [t for _, t in _texts(stored)] == [
        "Nouvelle tentative",
        "After the interruption.",
    ]
    await h.close()


def _fail_saves(store: HistoryStore, times: int) -> None:
    """Make the next `times` saves fail, as a full disk or locked file would."""
    real = store.append_turn
    remaining = [times]

    def flaky(*args, **kwargs):
        if remaining[0] > 0:
            remaining[0] -= 1
            raise HistoryError("Couldn't save the turn: database or disk is full")
        return real(*args, **kwargs)

    store.append_turn = flaky  # type: ignore[method-assign]


async def test_oversized_message_is_refused_not_truncated(db: Path) -> None:
    h = Harness(db, [])
    h.app.max_chars = 1_000
    await h.app.start()
    big = "def f():\n" + "    x = 1\n" * 200  # 2 009 characters
    await h.app.handle_input(big)
    assert "Message non envoyé" in h.printed() and "pas été tronqué" in h.printed()
    assert "1 000" in h.printed()
    assert h.llm.requests == []
    assert h.store.list_conversations() == []
    await h.close()


async def test_failed_save_is_retried_and_reported(db: Path) -> None:
    h = Harness(db, ["A1.", "A2."])
    await h.app.start()
    _fail_saves(h.store, 1)
    await h.app.handle_input("Question 1")
    assert "Non enregistré" in h.printed()
    assert h.store.list_conversations() == []
    await h.app.handle_input("Question 2")  # retries the pending turn first
    stored = h.store.load_items(h.app.conversation_id)
    assert [t for _, t in _texts(stored)] == ["Question 1", "A1.", "Question 2", "A2."]
    await h.close()


@pytest.mark.parametrize("command", ["/new", "/resume"])
async def test_switching_with_unsaved_turns_needs_explicit_consent(
    db: Path, command: str
) -> None:
    h = Harness(db, ["Autre.", "A1.", "A2."], answers=["non", "oui"])
    await h.app.start()
    await h.app.handle_input("Une autre conversation")
    other = h.app.conversation_id
    await h.app.handle_input("/new")
    _fail_saves(h.store, 10)
    await h.app.handle_input("Question importante")
    current = h.app.conversation_id
    target = command if command == "/new" else f"/resume {other[:8]}"

    await h.app.handle_input(target)  # answered "non"
    assert "n'ont pas pu être enregistrés et seraient perdus" in h.questions[-1]
    assert "Annulé" in h.printed()
    assert h.app.conversation_id == current and len(h.app.unsaved) == 1
    assert [t for _, t in _texts(h.agent_items())] == ["Question importante", "A1."]

    await h.app.handle_input(target)  # answered "oui"
    assert "abandonné" in h.printed()
    assert h.app.conversation_id == (None if command == "/new" else other)
    assert h.app.unsaved == []
    await h.close()


async def test_switch_after_a_recovered_save_asks_nothing(db: Path) -> None:
    h = Harness(db, ["A1."])
    await h.app.start()
    _fail_saves(h.store, 1)
    await h.app.handle_input("Question")
    cid = h.app.conversation_id
    await h.app.handle_input("/new")  # the retry succeeds, so no question
    assert h.questions == []
    assert [t for _, t in _texts(h.store.load_items(cid))] == ["Question", "A1."]
    await h.close()


async def test_deleting_the_current_conversation_mentions_unsaved_turns(
    db: Path,
) -> None:
    h = Harness(db, ["A1.", "A2."], answers=["non"])
    await h.app.start()
    await h.app.handle_input("Q1")
    _fail_saves(h.store, 10)
    await h.app.handle_input("Q2")
    await h.app.handle_input("/delete " + h.app.conversation_id[:8])
    assert "1 échange(s) enregistré(s)" in h.questions[-1]
    assert "1 échange(s) non enregistré(s) seront aussi perdus" in h.questions[-1]
    assert "Suppression annulée" in h.printed()
    assert len(h.app.unsaved) == 1
    await h.close()


async def test_tool_call_metadata_survives_save_restart_and_resume(db: Path) -> None:
    extra = {"provider": {"signature": "c2lnbmF0dXJlLWJ5dGVz"}}
    first = Harness(
        db,
        [
            ToolCall(
                "inspect_code", '{"code": "x", "language": "python"}', "c1", extra
            ),
            "Reviewed.",
        ],
    )
    await first.app.start()
    await first.app.handle_input("Review: x")
    cid = first.app.conversation_id
    await first.close()
    [stored_call] = [
        i for i in HistoryStore(db).load_items(cid) if i["type"] == "function_call"
    ]
    assert stored_call["extra"] == extra

    second = Harness(db, ["Follow-up."])
    await second.app.start()
    await second.app.handle_input("/resume " + cid[:8])
    await second.app.handle_input("And now?")
    [sent_call] = [
        i for i in second.llm.requests[-1].items if i.type == "function_call"
    ]
    assert sent_call.extra == extra and sent_call.call_id == "c1"
    [sent_output] = [
        i for i in second.llm.requests[-1].items if i.type == "function_call_output"
    ]
    assert sent_output.call_id == "c1"
    await second.close()


REVIEW = (
    "Peux-tu vérifier ce code ?\n```python\nimport subprocess\n\n"
    "def show(name):\n    subprocess.run('ls ' + name, shell=True)\n```"
)
FOLLOW_UP = "Et dans la fonction que tu as vérifiée, quel argument faut-il retirer ?"


def _system_notes(request: llm.ChatContext) -> str:
    return "\n".join(
        item.text_content or ""
        for item in request.items
        if item.type == "message" and item.role == "system"
    )


def _user_texts(request: llm.ChatContext) -> list[str]:
    return [t for r, t in _texts(request.items) if r == "user"]


async def test_failed_review_then_follow_up_says_the_code_is_not_in_context(
    db: Path,
) -> None:
    h = Harness(db, [ScriptedFailureError("504"), "Je n'ai pas reçu ce code.", "Ok."])
    await h.app.start()
    await h.app.handle_input(REVIEW)  # fails: nothing saved, session rebuilt
    await h.app.handle_input(FOLLOW_UP)
    follow_up_request = h.llm.requests[1]
    assert REVIEW not in _user_texts(follow_up_request)  # the code really is absent
    assert MISSED_TURN_NOTE in _system_notes(follow_up_request)

    await h.app.handle_input("Voici le code à nouveau : x = 1")
    assert MISSED_TURN_NOTE not in _system_notes(h.llm.requests[2])  # cleared
    await h.close()


async def test_resumed_conversation_with_the_code_needs_no_warning(db: Path) -> None:
    first = Harness(db, ["Injection de commande via shell=True."])
    await first.app.start()
    await first.app.handle_input(REVIEW)
    cid = first.app.conversation_id
    await first.close()

    second = Harness(db, ["Retirez shell=True."])
    await second.app.start()
    await second.app.handle_input("/resume " + cid[:8])
    await second.app.handle_input(FOLLOW_UP)
    request = second.llm.requests[-1]
    assert REVIEW in _user_texts(request)  # the reviewed code is there to use
    notes = _system_notes(request)
    assert "not included in what you can see" not in notes
    assert MISSED_TURN_NOTE not in notes
    await second.close()


async def test_code_trimmed_from_context_is_reported_as_hidden(db: Path) -> None:
    h = Harness(db, ["Revu.", "Oui.", "Je ne vois plus ce code."])
    h.app.max_chars = 120  # the newer exchanges fill this, so the review is dropped
    await h.app.start()
    await h.app.handle_input(REVIEW)
    await h.app.handle_input("Merci, c'est clair.")
    await h.app.handle_input(FOLLOW_UP)
    request = h.llm.requests[-1]
    assert REVIEW not in _user_texts(request)
    assert "1 earlier exchange(s) of this conversation are not included" in (
        _system_notes(request)
    )
    await h.close()


async def test_resume_reports_exchanges_that_were_never_loaded(db: Path) -> None:
    h = Harness(db, ["a1", "a2", "a3", "a4"])
    await h.app.start()
    for n in (1, 2, 3):
        await h.app.handle_input(f"q{n}")
    cid = h.app.conversation_id
    h.app.max_turns = 1
    await h.app.handle_input("/new")
    await h.app.handle_input("/resume " + cid[:8])
    await h.app.handle_input("q4")
    request = h.llm.requests[-1]
    assert _user_texts(request) == ["q3", "q4"]
    assert "2 earlier exchange(s)" in _system_notes(request)
    await h.close()


@pytest.mark.parametrize(
    "command",
    [
        "/resume ID",
        "/resume la conversation sur Baobab",
        "/resume",
        "/delete toutes",
        "/new conversation sur les MAC",
        "/list des conversations",
    ],
)
async def test_malformed_commands_show_usage_and_never_reach_the_model(
    db: Path, command: str
) -> None:
    h = Harness(db, ["Réponse."])
    await h.app.start()
    await h.app.handle_input("Bonjour")
    cid = h.app.conversation_id
    requests_before = len(h.llm.requests)

    await h.app.handle_input(command)

    assert len(h.llm.requests) == requests_before
    assert "Rien n'a été envoyé au modèle" in h.printed()
    assert h.app.conversation_id == cid  # nothing switched, created or deleted
    assert [i.id for i in h.store.list_conversations()] == [cid]
    await h.close()


async def test_paths_and_multiline_pastes_starting_with_a_slash_are_sent(
    db: Path,
) -> None:
    h = Harness(db, ["Réponse 1.", "Réponse 2."])
    await h.app.start()
    await h.app.handle_input("/etc/passwd est lisible par tous, c'est grave ?")
    await h.app.handle_input("/resume 7a073292\nvoici aussi mon code :\n    x = 1")
    sent = [t for r, t in _texts(h.llm.requests[-1].items) if r == "user"]
    assert sent == [
        "/etc/passwd est lisible par tous, c'est grave ?",
        "/resume 7a073292\nvoici aussi mon code :\n    x = 1",
    ]
    await h.close()


async def test_the_model_is_told_how_saved_history_works(db: Path) -> None:
    h = Harness(db, ["Réponse."])
    await h.app.start()
    await h.app.handle_input("Tu te souviens de notre conversation d'hier ?")
    assert TERMINAL_CHAT_NOTE in _system_notes(h.llm.requests[-1])
    await h.close()


async def test_retry_after_an_interrupted_save_acknowledgement_adds_no_duplicate(
    db: Path,
) -> None:
    h = Harness(db, ["Réponse."])
    await h.app.start()
    real = h.store.append_turn
    calls: list[bool] = []

    def commits_then_interrupted(*args, **kwargs):
        calls.append(real(*args, **kwargs))  # the write commits...
        if len(calls) == 1:
            raise asyncio.CancelledError  # ...but the caller never learns it

    h.store.append_turn = commits_then_interrupted  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        await h.app.handle_input("Question unique")
    assert len(h.app.unsaved) == 1  # still pending from the app's point of view

    await h.app.close()  # quitting retries pending saves
    assert calls == [True, False]  # the retry was recognized as already saved
    stored = h.store.load_items(h.app.conversation_id)
    assert [t for _, t in _texts(stored)] == ["Question unique", "Réponse."]
    assert h.app.unsaved == []
    h.store.close()


async def test_search_notes_tool_runs_and_its_sources_are_saved(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notes_dir = Path(__file__).parent / "fixtures" / "notes"
    monkeypatch.setenv("PAINTHAKER_NOTES_DIR", str(notes_dir))
    h = Harness(
        db,
        [
            ToolCall("search_notes", '{"query": "canal Wi-Fi routeur Baobab"}', "n1"),
            "D'après tes notes, le canal est 11 (reseau.md:1-8).",
        ],
    )
    await h.app.start()
    assert "search_notes" in {tool.info.name for tool in h.app.agent.tools}
    await h.app.handle_input("D'après mes notes, quel canal Wi-Fi utilise le routeur ?")

    stored = h.store.load_items(h.app.conversation_id)
    [output] = [i for i in stored if i["type"] == "function_call_output"]
    assert output["call_id"] == "n1" and not output["is_error"]
    assert "'status': 'ok'" in output["output"]
    assert "'source': 'reseau.md:" in output["output"]
    assert "Le routeur principal utilise le canal Wi-Fi 11." in output["output"]
    assert "canal 6" not in output["output"]  # hidden draft never read
    assert "tool: search_notes" in h.printed()
    await h.close()


async def test_search_notes_without_a_folder_explains_how_to_enable_it(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PAINTHAKER_NOTES_DIR", raising=False)
    h = Harness(
        db,
        [
            ToolCall("search_notes", '{"query": "routeur"}', "n2"),
            "La recherche dans les notes n'est pas activée.",
        ],
    )
    await h.app.start()
    await h.app.handle_input("Cherche dans mes notes : routeur")
    [output] = [
        i
        for i in h.store.load_items(h.app.conversation_id)
        if i["type"] == "function_call_output"
    ]
    assert "not_configured" in output["output"]
    assert "PAINTHAKER_NOTES_DIR" in output["output"]
    await h.close()
