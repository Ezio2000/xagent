from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any

from kernel import (
    AgentEvent,
    AgentLoop,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    CheckpointSummary,
    ContentPart,
    EventEmitter,
    InvalidToolCall,
    JournalRecord,
    Message,
    ModelCapabilities,
    ModelContentDelta,
    ModelRequest,
    ModelResponse,
    ModelStreamCompleted,
    ModelStreamEvent,
    RunJournal,
    RunSnapshot,
    RunStore,
    RuntimeContext,
    StoredCheckpoint,
    ToolCall,
    ToolObservation,
    ToolOutput,
    ToolRegistryProtocol,
    ToolSpec,
)


class EchoRegistry(ToolRegistryProtocol):
    def __init__(self) -> None:
        self._spec = ToolSpec(
            name="echo",
            description="Echo text back to the model.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            annotations={
                "parallel_safe": True,
                "read_only": True,
                "idempotent": True,
                "risk": {"requires_approval": False},
            },
        )

    def specs(self) -> tuple[ToolSpec, ...]:
        return (ToolSpec.from_dict(self._spec.to_dict()),)

    def spec_for(self, name: str) -> ToolSpec | None:
        if name != self._spec.name:
            return None
        return ToolSpec.from_dict(self._spec.to_dict())

    def validate_call(self, call: ToolCall) -> None:
        if call.name != self._spec.name:
            raise InvalidToolCall(f"unknown tool: {call.name}")
        if call.mode != "execute":
            raise InvalidToolCall(f"unsupported mode: {call.mode}")
        if not isinstance(call.arguments.get("text"), str):
            raise InvalidToolCall("echo.text must be a string")

    async def invoke(
        self,
        call: ToolCall,
        context: RuntimeContext,
        *,
        progress_emitter: Callable[[Mapping[str, Any]], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> ToolOutput:
        _ = context
        self.validate_call(call)
        if progress_emitter is not None:
            progress_emitter({"phase": "echo"})
        if cancel_checker is not None and cancel_checker():
            return ToolObservation.text("cancelled", is_error=True)
        return ToolObservation.text(str(call.arguments["text"]))


class ScriptedModel:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name=request.tools[0].name,
                        mode="execute",
                        arguments={"text": "hello"},
                    )
                ]
            )
        return ModelResponse.text(f"tool said: {request.messages[-1].text}")


class StreamingTextModel:
    capabilities = ModelCapabilities(streaming=True)

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        return ModelResponse.text("fallback")

    async def stream(
        self, request: ModelRequest, context: RuntimeContext
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = request, context
        yield ModelContentDelta(index=0, text_delta="stream")
        yield ModelContentDelta(index=0, text_delta="ed")
        yield ModelStreamCompleted(ModelResponse.text("streamed"))


class AllowAllPolicy(ApprovalPolicy):
    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        _ = request
        return ApprovalDecision.allow()


class MemoryRunStore(RunStore):
    def __init__(self) -> None:
        self.checkpoints: list[StoredCheckpoint] = []

    async def save_checkpoint(self, checkpoint: StoredCheckpoint) -> None:
        self.checkpoints.append(StoredCheckpoint.from_dict(checkpoint.to_dict()))

    async def load_checkpoint(self, run_id: str, checkpoint_id: str | None = None) -> RunSnapshot:
        candidates = [item for item in self.checkpoints if item.run_id == run_id]
        if checkpoint_id is not None:
            candidates = [item for item in candidates if item.checkpoint_id == checkpoint_id]
        if not candidates:
            raise KeyError(run_id)
        return candidates[-1].snapshot

    async def list_checkpoints(self, run_id: str) -> Sequence[CheckpointSummary]:
        return [item.summary() for item in self.checkpoints if item.run_id == run_id]


class MemoryRunJournal(RunJournal):
    def __init__(self) -> None:
        self.records: list[JournalRecord] = []

    async def append(self, record: JournalRecord) -> None:
        self.records.append(JournalRecord.from_dict(record.to_dict()))

    async def read(
        self, run_id: str, *, after_sequence: int | None = None
    ) -> AsyncIterator[JournalRecord]:
        for record in self.records:
            if record.run_id != run_id:
                continue
            if after_sequence is not None and record.sequence <= after_sequence:
                continue
            yield JournalRecord.from_dict(record.to_dict())


class EventCountingHook:
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_event(
        self,
        event: AgentEvent,
        context: RuntimeContext,
        emitter: EventEmitter,
    ) -> None:
        _ = context, emitter
        self.events.append(event.type)


async def main() -> None:
    store = MemoryRunStore()
    journal = MemoryRunJournal()
    hook = EventCountingHook()
    loop = AgentLoop(
        model=ScriptedModel(),
        tools=EchoRegistry(),
        approval_policy=AllowAllPolicy(),
        run_store=store,
        run_journal=journal,
        hooks=[hook],
    )
    result = await loop.run([Message.user([ContentPart.text_part("use echo")])])

    streaming = AgentLoop(model=StreamingTextModel())
    stream_result = await streaming.run(
        [Message.user([ContentPart.text_part("stream please")])],
        stream=True,
    )

    print(
        {
            "status": result.status.value,
            "final": result.final_parts[0].text,
            "checkpoints": len(store.checkpoints),
            "journal_records": len(journal.records),
            "hook_events": len(hook.events),
            "stream_final": stream_result.final_parts[0].text,
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
