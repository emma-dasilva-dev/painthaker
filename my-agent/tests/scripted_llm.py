"""A scripted LLM for offline tests: replays planned replies, tool calls or
failures through the real AgentSession, and records what it was sent."""

import asyncio
from dataclasses import dataclass
from typing import Any

from livekit.agents import APIConnectionError, llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions


@dataclass
class ToolCall:
    name: str
    arguments: str
    call_id: str


class ScriptedFailureError(Exception):
    """Make this LLM call fail (a non-retryable provider error)."""


class Hang:
    """Make this LLM call wait forever (until the turn is interrupted)."""


class ScriptedLLM(llm.LLM):
    def __init__(self, steps: list[str | ToolCall | ScriptedFailureError]) -> None:
        super().__init__()
        self.steps = list(steps)
        self.requests: list[llm.ChatContext] = []

    @property
    def model(self) -> str:
        return "scripted"

    @property
    def provider(self) -> str:
        return "test"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> llm.LLMStream:
        self.requests.append(chat_ctx.copy())
        if not self.steps:
            raise AssertionError("ScriptedLLM ran out of steps")
        return _Stream(
            self,
            step=self.steps.pop(0),
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class _Stream(llm.LLMStream):
    def __init__(self, scripted: ScriptedLLM, *, step: Any, **kwargs: Any) -> None:
        super().__init__(scripted, **kwargs)
        self._step = step

    async def _run(self) -> None:
        step = self._step
        if isinstance(step, Hang):
            await asyncio.Event().wait()
        if isinstance(step, ScriptedFailureError):
            raise APIConnectionError(str(step) or "scripted failure", retryable=False)
        if isinstance(step, ToolCall):
            delta = llm.ChoiceDelta(
                role="assistant",
                tool_calls=[
                    llm.FunctionToolCall(
                        name=step.name, arguments=step.arguments, call_id=step.call_id
                    )
                ],
            )
        else:
            delta = llm.ChoiceDelta(role="assistant", content=step)
        self._event_ch.send_nowait(llm.ChatChunk(id="scripted", delta=delta))
