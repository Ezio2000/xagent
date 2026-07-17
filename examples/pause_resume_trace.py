"""Durable tool wait, checkpoint wire round-trip, resume, and trace verification."""

from __future__ import annotations

import asyncio

from jharness.kernel import (
    Checkpoint,
    Completed,
    ContentPart,
    DeltaSink,
    Event,
    Invocation,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RunContext,
    Runtime,
    Suspension,
    SuspensionSelector,
    ToolCall,
    ToolContext,
    ToolResult,
    ToolSpec,
    ToolWaiting,
    WaitingResult,
)
from jharness.kernel.diagnostics import build_trace, verify_trace
from jharness.kernel.wire import decode_checkpoint, encode_checkpoint
from jharness.toolkit import ToolRegistry


class ExternalWaitTool:
    spec = ToolSpec(
        "external_wait",
        "Suspend until a host callback arrives.",
        {
            "type": "object",
            "required": ["wait_id"],
            "properties": {"wait_id": {"type": "string", "minLength": 1}},
            "additionalProperties": False,
        },
    )

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        del context
        wait_id = str(call.arguments["wait_id"])
        return WaitingResult(
            ToolWaiting((ContentPart.text_part(f"waiting for {wait_id}"),)),
            Suspension("external_callback", "external_wait", wait_id),
        )


class DemoModel:
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
        if not any(message.role == "tool" for message in request.messages):
            return ModelResponse(
                tool_calls=(ToolCall("call-1", "external_wait", {"wait_id": "job-1"}),),
                finish_reason="tool_calls",
            )
        return ModelResponse(
            (ContentPart.text_part("job-1 completed after resume"),),
            finish_reason="end_turn",
        )


async def observe(invocation: Invocation) -> tuple[Checkpoint, tuple[Event, ...]]:
    events = tuple([event async for event in invocation.events()])
    return await invocation.result(), events


async def main() -> None:
    runtime = Runtime(model=DemoModel(), tools=ToolRegistry((ExternalWaitTool(),)))
    paused, paused_events = await observe(runtime.start((Message.user("start external job"),)))
    restored = decode_checkpoint(encode_checkpoint(paused))
    paused_trace = build_trace(paused_events, "start")
    verification = verify_trace(paused_trace)

    resumed, resumed_events = await observe(
        runtime.resume(
            restored,
            selector=SuspensionSelector(wait_id="job-1"),
            append_messages=(Message.external("job-1 callback received"),),
        )
    )
    resumed_state = resumed.snapshot.state
    if not isinstance(resumed_state, Completed):
        raise RuntimeError(f"resume stopped with {resumed.snapshot.status}")
    verify_trace(build_trace(resumed_events, "resume"))

    print(f"paused={paused.snapshot.status} replay_commits={verification.checkpoint_count}")
    final_text = "".join(part.text or "" for part in resumed_state.parts)
    print(f"resumed={resumed.snapshot.status} final={final_text}")


if __name__ == "__main__":
    asyncio.run(main())
