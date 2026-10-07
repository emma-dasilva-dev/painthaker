# Live response-quality checks: each test runs one agent turn against Gemini and
# combines deterministic checks (tests/response_checks.py) with an LLM judge.
# The questions are worded differently from the ones that exposed each problem,
# so they test the rule rather than one memorized answer.
#
# Gemini's free tier allows 15 requests/min; run this file on its own, e.g.
#   uv run --no-sync pytest -m live tests/test_response_quality.py
# and wait about a minute before the next live run.

import re
from datetime import datetime

import pytest
from livekit.agents import AgentSession, llm
from livekit.plugins import google
from response_checks import (
    check_concise,
    check_formatting,
    check_no_destructive_example,
    check_no_risk_formula,
    code_blocks,
)

from agent import Painthaker, app_timezone
from chat import TERMINAL_CHAT_NOTE

pytestmark = pytest.mark.live


def _judge_llm() -> llm.LLM:
    return google.LLM(model="gemini-3.5-flash-lite")


def _reply(result) -> str:
    return "\n".join(
        event.item.text_content or ""
        for event in result.events
        if event.type == "message" and event.item.role == "assistant"
    )


@pytest.mark.asyncio
async def test_threat_definition_is_not_limited_to_external_attackers() -> None:
    async with _judge_llm() as judge_llm, AgentSession() as session:
        await session.start(Painthaker())
        result = await session.run(
            user_input="En sécurité informatique, qu'est-ce qu'on appelle une menace ?"
        )
        reply = _reply(result)

        assert check_formatting(reply) + check_concise(reply, max_words=160) == []
        assert check_no_risk_formula(reply) == []
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Written in French. Defines a threat as a potential cause of harm "
                    "(a circumstance or event that could adversely affect a system or "
                    "its data). Does not restrict threats to external attackers: it "
                    "includes or clearly allows for internal/insider sources and "
                    "unintentional ones such as human error, failures or natural events."
                ),
            )
        )


@pytest.mark.asyncio
async def test_risk_relationship_is_not_presented_as_a_formula() -> None:
    async with _judge_llm() as judge_llm, AgentSession() as session:
        await session.start(Painthaker())
        result = await session.run(
            user_input=(
                "Quick one: how do threats, vulnerabilities and risk relate to each other?"
            )
        )
        reply = _reply(result)

        assert check_formatting(reply) + check_concise(reply, max_words=160) == []
        assert check_no_risk_formula(reply) == []
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Written in English. Explains that risk arises when a threat could "
                    "exploit a vulnerability, and that risk depends on how likely that "
                    "is and how severe the impact would be. It does not present the "
                    "relationship as an exact formula; giving no formula at all is "
                    "fine, and if it does use a shorthand, it calls it a simplification."
                ),
            )
        )


@pytest.mark.asyncio
async def test_file_read_fix_uses_python_and_explains_access_boundaries() -> None:
    async with _judge_llm() as judge_llm, AgentSession() as session:
        await session.start(Painthaker())
        result = await session.run(
            user_input=(
                "Comment je corrige ce code ? Les utilisateurs ne doivent pouvoir "
                "lire que les fichiers du dossier `reports/`.\n"
                "```python\n"
                "import subprocess\n"
                "name = input('Rapport : ')\n"
                "subprocess.run('cat reports/' + name, shell=True)\n"
                "```"
            )
        )
        reply = _reply(result)

        result.expect.contains_function_call(name="inspect_code")
        assert check_formatting(reply) + check_no_destructive_example(reply) == []
        fixes = "\n".join(code_blocks(reply))
        assert "subprocess" not in fixes, (
            "the fix should not launch a process to read a file"
        )
        assert any(api in fixes for api in ("open(", "read_text", "read_bytes"))
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Written in French. Replaces the shell command with Python's own "
                    "file reading. Explains that removing or blocking '../' alone does "
                    "not keep reads inside reports/ (encodings, absolute paths, "
                    "symlinks), and enforces the boundary by resolving the path and "
                    "checking it stays within the resolved reports/ directory, an "
                    "allow-list of names, or operating-system permissions."
                ),
            )
        )


@pytest.mark.asyncio
async def test_command_injection_demo_is_harmless_and_fix_fits_the_task() -> None:
    async with _judge_llm() as judge_llm, AgentSession() as session:
        await session.start(Painthaker())
        result = await session.run(
            user_input=(
                "Is this endpoint helper exploitable? `host` comes from a query "
                "parameter.\n"
                "```python\n"
                "import os\n"
                "def check(host):\n"
                "    return os.system('ping -c 1 ' + host)\n"
                "```"
            )
        )
        reply = _reply(result)

        result.expect.contains_function_call(name="inspect_code")
        assert check_formatting(reply) + check_no_destructive_example(reply) == []
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Written in English. States this is a demonstrated command "
                    "injection because a query parameter reaches os.system. Any example "
                    "payload is harmless (such as running `id` or `echo`), never "
                    "destructive. The fix runs ping without a shell, with an argument "
                    "list, and validates the host (for example with the ipaddress "
                    "module or an allow-list); it does not rely on a blocklist of "
                    "characters alone."
                ),
            )
        )


@pytest.mark.asyncio
async def test_current_year_question_gets_the_clock_year_directly() -> None:
    year = datetime.now(app_timezone()).year
    async with _judge_llm() as judge_llm, AgentSession() as session:
        await session.start(Painthaker())
        result = await session.run(user_input="what year are we in?")
        reply = _reply(result)

        assert str(year) in reply
        assert str(year - 1) not in reply
        assert check_formatting(reply) + check_concise(reply, max_words=40) == []
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Written in English. Directly states the current year in a "
                    "sentence or two. Does not steer the conversation to "
                    "cybersecurity or offer a lesson, and does not claim to know "
                    "current news or events."
                ),
            )
        )


# Missing-context cases. The agent is built the way the chat builds it after a
# failed turn, a resume, or trimming, so these check what Gemini does with it.
_REVIEW = (
    "Peux-tu vérifier ce code ?\n```python\nimport subprocess\n\n"
    "def show(name):\n    subprocess.run('ls ' + name, shell=True)\n```"
)
_FOLLOW_UP = "Et dans la fonction que tu as vérifiée, quel argument faut-il retirer ?"
_ASKS_TO_PASTE = (
    "Written in French. Says it doesn't have the code the user refers to in the "
    "current context (or can't see it), "
    "and asks the user to paste or send it again. It does not describe, "
    "quote or review that code, and does not name a specific argument of that "
    "function as the answer."
)


def _reviewed_turn() -> llm.ChatContext:
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content=_REVIEW)
    chat_ctx.add_message(
        role="assistant",
        content=(
            "Injection de commande confirmée : `name` arrive dans une commande shell "
            "via `shell=True`."
        ),
    )
    return chat_ctx


async def _follow_up(agent: Painthaker, intent: str) -> None:
    async with _judge_llm() as judge_llm, AgentSession() as session:
        await session.start(agent)
        result = await session.run(user_input=_FOLLOW_UP)
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(judge_llm, intent=intent)
        )


@pytest.mark.asyncio
async def test_follow_up_after_a_failed_review_asks_for_the_code() -> None:
    await _follow_up(Painthaker(missed_user_turn=True), _ASKS_TO_PASTE)


@pytest.mark.asyncio
async def test_follow_up_in_a_resumed_conversation_uses_the_code() -> None:
    await _follow_up(
        Painthaker(chat_ctx=_reviewed_turn()),
        "Written in French. Answers directly that `shell=True` should be removed "
        "(passing the command as an argument list). It does not claim the code is "
        "missing and does not ask the user to paste or resend it; offering further "
        "help, such as showing a rewrite, is fine.",
    )


@pytest.mark.asyncio
async def test_follow_up_about_code_trimmed_from_context_asks_for_it() -> None:
    chat_ctx = _reviewed_turn()
    chat_ctx.add_message(role="user", content="Merci, c'est clair.")
    chat_ctx.add_message(role="assistant", content="Avec plaisir.")
    # A budget the two newer exchanges fill, so the review is trimmed away.
    await _follow_up(Painthaker(chat_ctx=chat_ctx, context_chars=120), _ASKS_TO_PASTE)


@pytest.mark.asyncio
async def test_follow_up_in_a_fresh_conversation_asks_for_the_code() -> None:
    # The recorded incident: the review failed in an earlier process, so no
    # context note exists and only the prompt rule can prevent a made-up answer.
    await _follow_up(Painthaker(), _ASKS_TO_PASTE)


# Absolute claims and the terminal chat's capabilities. Each question is
# worded differently from the conversation where the problem appeared.
_ABSOLUTE = re.compile(
    r"ne (?:change|changent|peut|peuvent) jamais|jamais modifi|"
    r"(?:est|sont) (?:fixe|fixes|permanente?s?|immuables?)\b|never change",
    re.IGNORECASE,
)


@pytest.mark.asyncio
async def test_mac_address_answer_mentions_that_it_can_change() -> None:
    async with _judge_llm() as judge_llm, AgentSession() as session:
        await session.start(Painthaker())
        result = await session.run(
            user_input="Mon adresse MAC me suit partout, c'est un identifiant définitif ?"
        )
        reply = _reply(result)
        match = _ABSOLUTE.search(reply)
        assert match is None, f"absolute claim: {match.group(0)!r}"
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Written in French. Explains that a network card has a "
                    "manufacturer-assigned MAC address, but the address a device "
                    "uses can differ or change: for example randomized/private "
                    "Wi-Fi addresses on phones and computers, or an address set in "
                    "software. Does not claim MAC addresses are permanent or can "
                    "never change."
                ),
            )
        )


@pytest.mark.asyncio
async def test_new_chat_explains_saved_history_without_denying_it() -> None:
    async with _judge_llm() as judge_llm, AgentSession() as session:
        await session.start(Painthaker(app_note=TERMINAL_CHAT_NOTE))
        result = await session.run(
            user_input=(
                "Tu peux reprendre là où on s'était arrêtés la semaine dernière "
                "sur mon réseau domestique ?"
            )
        )
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Written in French. Says the earlier conversation isn't loaded "
                    "in this chat (it may be saved) and tells the user to use /list "
                    "and then /resume with the conversation's ID. Does not say the "
                    "application has no saved or persistent history. Repeating the "
                    "topic the user named (their home network) is fine, but it does "
                    "not invent anything else about what was said earlier."
                ),
            )
        )


# Notes search, with invented facts in a temporary notes folder.
_LAB_NOTES = (
    "# Labo Kapokier\n"
    "\n"
    "Le labo Kapokier possède une imprimante 3D nommée Tisserande.\n"
    "L'imprimante Tisserande imprime en PLA à 210 °C.\n"
    "Elle est réservée le jeudi pour les ateliers.\n"
)


@pytest.fixture
def lab_notes(tmp_path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "notes"
    (root / "ateliers").mkdir(parents=True)
    (root / "ateliers" / "labo.md").write_text(_LAB_NOTES, encoding="utf-8")
    monkeypatch.setenv("PAINTHAKER_NOTES_DIR", str(root))
    return root


@pytest.mark.asyncio
async def test_notes_answer_cites_a_real_retrieved_line_range(lab_notes) -> None:
    async with _judge_llm() as judge_llm, AgentSession() as session:
        await session.start(Painthaker())
        result = await session.run(
            user_input="D'après mes notes, à quelle température imprime Tisserande ?"
        )
        result.expect.contains_function_call(name="search_notes")
        reply = _reply(result)
        assert "210" in reply
        citations = re.findall(r"ateliers/labo\.md:(\d+)-(\d+)", reply)
        assert citations, f"no citation in: {reply!r}"
        assert any(int(a) <= 4 <= int(b) for a, b in citations)  # fact is on line 4
        assert all(int(b) <= 6 for _, b in citations)  # no line past the file's end
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Written in French. Says, based on the user's notes, that "
                    "Tisserande prints at 210 °C, and cites the notes file."
                ),
            )
        )


@pytest.mark.asyncio
async def test_notes_without_the_answer_say_so_without_inventing(lab_notes) -> None:
    async with _judge_llm() as judge_llm, AgentSession() as session:
        await session.start(Painthaker())
        result = await session.run(
            user_input="D'après mes notes, quelle est l'adresse IP de Tisserande ?"
        )
        result.expect.contains_function_call(name="search_notes")
        reply = _reply(result)
        assert not re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", reply), reply
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Written in French. Says it couldn't find Tisserande's IP "
                    "address in the passages it retrieved (the search results). "
                    "Does NOT state that the user's notes don't contain it or that "
                    "it isn't mentioned anywhere in the notes, and gives no IP "
                    "address."
                ),
            )
        )
