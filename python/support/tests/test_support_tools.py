from __future__ import annotations

import asyncio

import pytest
from kernel import (
    ContentPart,
    InvalidToolCall,
    RuntimeContext,
    ToolCall,
    ToolObservation,
    ToolOutput,
    ToolSpec,
)
from support import (
    EchoFixtureTool,
    FixtureToolRegistry,
    RecordingToolRegistry,
    ScriptedToolRegistry,
)
from toolkit import ToolRegistry


@pytest.mark.asyncio
async def test_recording_tool_registry_records_calls_and_returns_copies() -> None:
    registry = RecordingToolRegistry()
    context = RuntimeContext(run_id="run-1", started_at=1.0)

    output = await registry.invoke(
        ToolCall(id="call-1", name="record", arguments={"id": "job"}),
        context,
    )

    assert output.text_content == "job"
    assert registry.calls == ["job"]
    assert registry.records[0].call.id == "call-1"
    assert registry.records[0].context.run_id == "run-1"


@pytest.mark.asyncio
async def test_scripted_tool_registry_validates_name_and_mode() -> None:
    registry = ScriptedToolRegistry(
        [
            ToolSpec(
                name="handoff",
                description="Custom handoff.",
                input_schema={"type": "object", "properties": {}},
                modes=("handoff",),
            )
        ],
        {
            "handoff": lambda call, context: ToolOutput(
                kind="handoff",
                parts=[ContentPart.text_part(f"{context.run_id}:{call.id}")],
            )
        },
    )

    output = await registry.invoke(
        ToolCall(id="call-1", name="handoff", mode="handoff", arguments={}),
        RuntimeContext(run_id="run-1", started_at=1.0),
    )

    assert output.kind == "handoff"
    assert output.text_content == "run-1:call-1"
    with pytest.raises(InvalidToolCall, match="unknown tool"):
        registry.validate_call(ToolCall(id="call-2", name="missing", arguments={}))
    with pytest.raises(InvalidToolCall, match="unsupported tool mode"):
        registry.validate_call(ToolCall(id="call-3", name="handoff", arguments={}))


def test_scripted_tool_registry_rejects_duplicate_specs_and_unknown_handlers() -> None:
    spec = ToolSpec(
        name="record",
        description="Record.",
        input_schema={"type": "object", "properties": {}},
    )

    with pytest.raises(ValueError, match="unique"):
        ScriptedToolRegistry(
            [spec, spec], {"record": lambda call, context: ToolObservation.text("")}
        )
    with pytest.raises(ValueError, match="without specs"):
        ScriptedToolRegistry([spec], {"missing": lambda call, context: ToolObservation.text("")})


@pytest.mark.asyncio
async def test_fixture_tool_registry_echo_fail_wait_and_metadata_tools() -> None:
    registry = FixtureToolRegistry("echo", "fail", "wait", "metadata_tool")
    context = RuntimeContext(run_id="run-1", started_at=1.0)

    echo = await registry.invoke(
        ToolCall(id="echo-1", name="echo", arguments={"text": "hello"}),
        context,
    )
    wait = await registry.invoke(
        ToolCall(id="wait-1", name="wait", arguments={"wait_id": "job-1"}),
        context,
    )
    metadata = await registry.invoke(
        ToolCall(id="meta-1", name="metadata_tool", arguments={}),
        context,
    )
    fail = await registry.invoke(ToolCall(id="fail-1", name="fail", arguments={}), context)

    assert echo.text_content == "hello"
    assert wait.text_content == "external job started"
    assert wait.pause is not None
    assert wait.pause.wait_id == "job-1"
    assert metadata.text_content == "tool"
    assert metadata.parts[0].metadata == {"secret": "part"}
    assert metadata.metadata == {"secret": "result"}
    assert fail.text_content == "tool failed"
    assert fail.is_error is True


@pytest.mark.asyncio
async def test_fixture_tool_registry_parallel_wait_and_slow_tools() -> None:
    registry = FixtureToolRegistry("parallel_wait", "slow")
    context = RuntimeContext(run_id="run-1", started_at=1.0)

    waiting = await registry.invoke(
        ToolCall(id="wait-1", name="parallel_wait", arguments={"wait_id": "job-1"}),
        context,
    )

    spec = registry.spec_for("parallel_wait")
    assert spec is not None
    assert spec.annotations["parallel_safe"] is True
    assert waiting.text_content == "job-1"
    assert waiting.pause is not None
    assert waiting.pause.wait_id == "job-1"
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            registry.invoke(ToolCall(id="slow-1", name="slow", arguments={}), context),
            timeout=0.01,
        )


def test_fixture_tool_registry_rejects_unknown_builtin_tool_name() -> None:
    with pytest.raises(ValueError, match="unknown fixture tool"):
        FixtureToolRegistry("missing")


@pytest.mark.asyncio
async def test_toolkit_fixture_tools_work_with_toolkit_registry() -> None:
    registry = ToolRegistry([EchoFixtureTool()])

    output = await registry.invoke(
        ToolCall(id="call-1", name="echo", arguments={"text": "hello"}),
        RuntimeContext(run_id="run-1", metadata={}),
    )

    assert output.text_content == "hello"
