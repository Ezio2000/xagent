"""Minimal complete model → tool → model invocation."""

from __future__ import annotations

import asyncio

from jharness.kernel import (
    Completed,
    ContentPart,
    DeltaSink,
    EventKind,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RunContext,
    Runtime,
    SettledResult,
    ToolCall,
    ToolContext,
    ToolResult,
    ToolSpec,
    ToolSuccess,
)
from jharness.toolkit import ToolRegistry


class EchoTool:
    spec = ToolSpec(
        "echo",
        "Return the input text.",
        {
            "type": "object",
            "required": ["text"],
            "properties": {"text": {"type": "string"}},
            "additionalProperties": False,
        },
    )

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        del context
        text = str(call.arguments["text"])
        return SettledResult(ToolSuccess((ContentPart.text_part(text),), {"text": text}))


class DemoModel:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        del context, stream, emit_delta
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                tool_calls=(ToolCall("call-1", "echo", {"text": "hello"}),),
                finish_reason="tool_calls",
            )
        outcome = request.messages[-1].outcome
        if not isinstance(outcome, ToolSuccess):
            raise RuntimeError("echo did not produce a successful tool outcome")
        tool_text = outcome.parts[0].text or ""
        return ModelResponse(
            (ContentPart.text_part(f"Tool said: {tool_text}"),),
            finish_reason="end_turn",
        )


async def main() -> None:
    runtime = Runtime(model=DemoModel(), tools=ToolRegistry((EchoTool(),)))
    invocation = runtime.start((Message.user("Say hello through a tool"),))
    events = [event async for event in invocation.events()]
    checkpoint = await invocation.result()
    state = checkpoint.snapshot.state
    if not isinstance(state, Completed):
        raise RuntimeError(f"run stopped with {checkpoint.snapshot.status}")

    print("".join(part.text or "" for part in state.parts))
    commit_count = sum(event.kind is EventKind.CHECKPOINT_COMMITTED for event in events)
    print(f"commits={commit_count} status={checkpoint.snapshot.status}")


if __name__ == "__main__":
    asyncio.run(main())
