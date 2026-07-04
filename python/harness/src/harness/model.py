"""Small model and event helpers for tests."""

from __future__ import annotations

from collections.abc import Sequence

from kernel import AgentEvent, AgentLoop, Message, ModelRequest, ModelResponse, RuntimeContext


class ScriptedModel:
    """Model client that returns a fixed sequence of responses."""

    def __init__(self, steps: Sequence[ModelResponse]) -> None:
        self._steps = list(steps)
        self.calls = 0
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        self.requests.append(request)
        if self.calls >= len(self._steps):
            raise AssertionError("scripted model exhausted")
        response = self._steps[self.calls]
        self.calls += 1
        return ModelResponse.from_dict(response.to_dict())


async def collect_events(agent: AgentLoop, messages: Sequence[Message]) -> list[AgentEvent]:
    return [event async for event in agent.run_events(messages)]
