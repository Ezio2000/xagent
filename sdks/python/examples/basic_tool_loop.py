"""Minimal agent-runtime example."""

from __future__ import annotations

import asyncio
from typing import Any

from agent_runtime import (
    AgentLoop,
    Message,
    ModelRequest,
    ModelResponse,
    RuntimeContext,
    ToolCall,
    ToolResult,
    ToolSpec,
)


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

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        return ToolResult.text(str(arguments.get("text", "")))


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
    agent = AgentLoop(model=DemoModel(), tools=[EchoTool()])
    async for event in agent.run_events([Message.user_text("Say hello through a tool")]):
        print(event.to_dict())


if __name__ == "__main__":
    asyncio.run(main())
