from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from agent_runtime import (
    AgentEvent,
    AgentLoop,
    AgentStatus,
    ContentPart,
    EventTypes,
    LoopLimits,
    Message,
    ModelContentDelta,
    ModelRequest,
    ModelResponse,
    ModelToolCallDelta,
    RunSnapshot,
    RuntimeContext,
    ToolCall,
    ToolResult,
    ToolSpec,
)


class ScriptedModel:
    def __init__(self, steps: Sequence[ModelResponse]) -> None:
        self._steps = list(steps)
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        if self.calls >= len(self._steps):
            return ModelResponse.text("fallback")
        response = self._steps[self.calls]
        self.calls += 1
        return response


class StreamedCaseModel(ScriptedModel):
    def __init__(
        self, steps: Sequence[ModelResponse], stream_steps: Sequence[dict[str, Any]]
    ) -> None:
        super().__init__(steps)
        self._stream_steps = list(stream_steps)
        self.stream_calls = 0

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = request, context
        if self.stream_calls >= len(self._stream_steps):
            yield ModelContentDelta(index=0, text_delta="fallback")
            return

        step = self._stream_steps[self.stream_calls]
        self.stream_calls += 1
        for raw_event in cast(list[dict[str, Any]], step.get("events") or []):
            event_type = str(raw_event["type"])
            if event_type == "text_delta":
                yield ModelContentDelta(
                    index=int(raw_event.get("index", 0)),
                    text_delta=str(raw_event.get("text_delta", "")),
                    part_type=str(raw_event.get("part_type", "text")),
                )
            elif event_type == "tool_call_delta":
                yield ModelToolCallDelta(
                    index=int(raw_event.get("index", 0)),
                    id=cast(str | None, raw_event.get("id")),
                    name=cast(str | None, raw_event.get("name")),
                    arguments_delta=cast(str | None, raw_event.get("arguments_delta")),
                )
            elif event_type == "sleep":
                await asyncio.sleep(float(raw_event["seconds"]))
            else:
                raise AssertionError(f"unsupported stream event type: {event_type}")


class EchoTool:
    spec = ToolSpec(
        name="echo",
        description="Return input text.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        return ToolResult.text(str(arguments.get("text", "")))


class FailTool:
    spec = ToolSpec(
        name="fail",
        description="Raise an error.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = arguments, context
        raise RuntimeError("tool failed")


class DelayedEchoTool:
    spec = ToolSpec(
        name="delayed_echo",
        description="Return input text after an optional delay.",
        input_schema={"type": "object", "properties": {}},
        annotations={"parallel_safe": True, "read_only": True, "idempotent": True},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        await asyncio.sleep(float(arguments.get("delay", 0)))
        return ToolResult.text(str(arguments.get("text", "")))


CASES_DIR = Path(__file__).resolve().parents[3] / "conformance" / "cases"


def load_cases() -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(CASES_DIR.glob("*.json"))]


def content_part_from_case(part: dict[str, Any]) -> ContentPart:
    return ContentPart.from_dict(part)


def model_response_from_case_step(step: dict[str, Any]) -> ModelResponse:
    calls = [ToolCall.from_dict(call) for call in step.get("tool_calls", [])]
    parts = [
        content_part_from_case(part) for part in cast(list[dict[str, Any]], step.get("parts", []))
    ]
    return ModelResponse(parts=parts, tool_calls=calls)


def limits_from_case(case: dict[str, Any]) -> LoopLimits:
    raw_limits = cast(dict[str, Any], case.get("limits") or {})
    return LoopLimits(
        max_iterations=int(raw_limits.get("max_iterations", 8)),
        max_total_tool_calls=int(raw_limits.get("max_total_tool_calls", 20)),
        timeout_seconds=cast(float | None, raw_limits.get("timeout_seconds")),
        max_parallel_tool_calls=int(raw_limits.get("max_parallel_tool_calls", 1)),
    )


async def collect_case_events(
    case: dict[str, Any],
    steps: Sequence[ModelResponse],
    stream_steps: Sequence[dict[str, Any]],
) -> list[AgentEvent]:
    model = StreamedCaseModel(steps, stream_steps) if stream_steps else ScriptedModel(steps)
    return [
        event
        async for event in AgentLoop(
            model=model,
            tools=[EchoTool(), FailTool(), DelayedEchoTool()],
            limits=limits_from_case(case),
        ).run_events([Message.user_text("run conformance case")], stream=bool(stream_steps))
    ]


def assert_event_stream_invariants(events: Sequence[AgentEvent], expected: AgentStatus) -> None:
    assert events
    assert events[0].type == EventTypes.RUN_STARTED
    assert events[-1].type == EventTypes.RUN_COMPLETED
    assert events[-1].data["state"]["status"] == expected.value

    run_ids = {event.run_id for event in events}
    assert len(run_ids) == 1
    assert next(iter(run_ids))

    sequences = [event.sequence for event in events]
    assert sequences == sorted(sequences)
    assert len(sequences) == len(set(sequences))

    terminal_state_changed_index: int | None = None
    checkpoint_after_terminal_index: int | None = None
    for index, event in enumerate(events):
        envelope = event.to_dict()
        assert set(envelope) == {
            "type",
            "data",
            "run_id",
            "sequence",
            "created_at",
            "schema_version",
        }
        assert isinstance(envelope["data"], dict)

        if event.type == EventTypes.CHECKPOINT:
            snapshot = RunSnapshot.from_dict(event.data)
            assert snapshot.context.run_id == event.run_id
            assert snapshot.context.sequence == event.sequence
            if terminal_state_changed_index is not None and snapshot.state.status is expected:
                checkpoint_after_terminal_index = index

        if event.type == EventTypes.STATE_CHANGED and event.data.get("to") == expected.value:
            terminal_state_changed_index = index

    assert terminal_state_changed_index is not None
    assert checkpoint_after_terminal_index is not None
    assert checkpoint_after_terminal_index > terminal_state_changed_index

    if expected is AgentStatus.COMPLETED:
        final_indexes = [
            index for index, event in enumerate(events) if event.type == EventTypes.FINAL
        ]
        assert final_indexes
        assert checkpoint_after_terminal_index < final_indexes[-1] < len(events) - 1
    else:
        error_indexes = [
            index for index, event in enumerate(events) if event.type == EventTypes.ERROR
        ]
        assert error_indexes
        assert checkpoint_after_terminal_index < error_indexes[-1] < len(events) - 1


@pytest.mark.asyncio
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: str(case["name"]))
async def test_conformance_case(case: dict[str, Any]) -> None:
    steps = [model_response_from_case_step(step) for step in case["model_steps"]]
    stream_steps = cast(list[dict[str, Any]], case.get("stream_model_steps") or [])
    expected_status = AgentStatus(case["expected_status"])
    model = StreamedCaseModel(steps, stream_steps) if stream_steps else ScriptedModel(steps)
    result = await AgentLoop(
        model=model,
        tools=[EchoTool(), FailTool(), DelayedEchoTool()],
        limits=limits_from_case(case),
    ).run([Message.user_text("run conformance case")], stream=bool(stream_steps))
    events = await collect_case_events(case, steps, stream_steps)

    assert result.status is expected_status
    assert result.total_tool_calls == case["expected_tool_calls"]
    assert_event_stream_invariants(events, expected_status)
    if "expected_message_roles" in case:
        assert result.state is not None
        assert [message.role for message in result.state.messages] == case["expected_message_roles"]
    if "expected_final_text" in case:
        assert (
            "".join(part.text or "" for part in result.final_parts) == case["expected_final_text"]
        )
    if "expected_tool_texts" in case:
        assert [message.text for message in result.messages if message.role == "tool"] == case[
            "expected_tool_texts"
        ]
    if "expected_model_deltas" in case:
        assert [
            dict(event.data) for event in events if event.type == EventTypes.MODEL_DELTA
        ] == case["expected_model_deltas"]
    if "forbidden_checkpoint_tool_counts" in case:
        forbidden = set(cast(list[int], case["forbidden_checkpoint_tool_counts"]))
        checkpoint_counts = [
            RunSnapshot.from_dict(event.data).state.total_tool_calls
            for event in events
            if event.type == EventTypes.CHECKPOINT
        ]
        assert not (forbidden & set(checkpoint_counts))
    if "forbidden_checkpoint_message_roles" in case:
        forbidden_roles = [
            tuple(item)
            for item in cast(list[list[str]], case["forbidden_checkpoint_message_roles"])
        ]
        checkpoint_roles = [
            tuple(message.role for message in RunSnapshot.from_dict(event.data).state.messages)
            for event in events
            if event.type == EventTypes.CHECKPOINT
        ]
        assert not (set(forbidden_roles) & set(checkpoint_roles))
