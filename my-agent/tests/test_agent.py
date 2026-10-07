# Turn-level checks for Painthaker's text-chat behavior. They run the agent
# in-process against its real LLM (Gemini, using GOOGLE_API_KEY from .env.local)
# without a LiveKit room, so `uv run pytest` costs only a few LLM calls. Whole
# conversations are covered by the simulations in scenarios.yaml.
#
# LLM output is non-deterministic: a pass shows the behavior is likely, not
# guaranteed. Re-run a failing test before concluding it regressed.

import pytest
from livekit.agents import AgentSession, llm
from livekit.plugins import google

from agent import Painthaker


def _judge_llm() -> llm.LLM:
    return google.LLM(model="gemini-3.5-flash-lite")


def _assistant_text(result) -> str:
    return "\n".join(
        event.item.text_content or ""
        for event in result.events
        if event.type == "message" and event.item.role == "assistant"
    )


@pytest.mark.asyncio
async def test_defaults_to_french() -> None:
    async with _judge_llm() as judge_llm, AgentSession() as session:
        await session.start(Painthaker())

        result = await session.run(user_input="Bonjour, c'est quoi un pare-feu ?")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(judge_llm, intent="Explains what a firewall is, written in French.")
        )
        assert "<expr" not in _assistant_text(result)


@pytest.mark.asyncio
async def test_switches_to_english_on_request() -> None:
    async with _judge_llm() as judge_llm, AgentSession() as session:
        await session.start(Painthaker())

        await session.run(user_input="Salut, c'est quoi le hachage ?")
        result = await session.run(
            user_input="Can we continue in English please? What is XSS?"
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent="Explains cross-site scripting (XSS), written entirely in English.",
            )
        )


@pytest.mark.asyncio
async def test_code_review_calls_inspect_code() -> None:
    async with _judge_llm() as judge_llm, AgentSession() as session:
        await session.start(Painthaker())

        result = await session.run(
            user_input=(
                "Please review this Python code for security:\n"
                "```python\n"
                "import subprocess\n"
                "filename = input('File to show: ')\n"
                "subprocess.run('cat ' + filename, shell=True)\n"
                "```"
            )
        )

        result.expect.contains_function_call(name="inspect_code")
        reply = _assistant_text(result)
        assert "<expr" not in reply
        assert "```" in reply, "expected the fix as a fenced code block"
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Written in English. "
                    "Identifies command injection because user input reaches a "
                    "shell command via shell=True, and shows a safer version "
                    "using exact Python syntax (an argument list without shell=True)."
                ),
            )
        )


@pytest.mark.asyncio
async def test_harmless_pattern_is_not_called_confirmed_vulnerability() -> None:
    async with _judge_llm() as judge_llm, AgentSession() as session:
        await session.start(Painthaker())

        result = await session.run(
            user_input=(
                "Est-ce que ce code est vulnérable ?\n"
                "```python\n"
                "import subprocess\n"
                "subprocess.run('ls -la /tmp', shell=True)\n"
                "```"
            )
        )

        result.expect.contains_function_call(name="inspect_code")
        await (
            result.expect[-1]
            .is_message(role="assistant")
            .judge(
                judge_llm,
                intent=(
                    "Written in French. Notes shell=True as a risky pattern or bad "
                    "practice, but does not claim this snippet is a confirmed or "
                    "exploitable vulnerability, because the command is a fixed "
                    "string with no user-controlled input."
                ),
            )
        )
