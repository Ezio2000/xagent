"""Minimal runtime example."""

from __future__ import annotations

import asyncio

from harness import AgentHarness
from kernel import (
    ModelRequest,
    ModelResponse,
    RuntimeContext,
    ToolCall,
    ToolObservation,
    ToolSpec,
)
from toolkit import ToolExecutionContext, ToolInvocation


class EchoTool:
    spec = ToolSpec(
        name="echo",
        description="Return the input text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        return ToolObservation.text(str(invocation.arguments.get("text", "")))


class DemoModel:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "hello"})]
            )
        return ModelResponse.text(f"Tool said: {request.messages[-1].text}")


async def main() -> None:
    agent = AgentHarness(model=DemoModel(), tools=[EchoTool()])
    async for event in agent.stream_events("Say hello through a tool"):
        print(event.to_dict())


if __name__ == "__main__":
    asyncio.run(main())
