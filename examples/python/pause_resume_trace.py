"""Pause, resume, and run-trace example."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from diagnostics import RunTrace, replay_trace
from kernel import (
    AgentLoop,
    ModelRequest,
    ModelResponse,
    PauseSelector,
    ResumeInput,
    RunSnapshot,
    RuntimeContext,
    ToolCall,
    ToolObservation,
    ToolSpec,
)
from prompting import user_text
from toolkit import ToolExecutionContext, ToolInvocation, ToolRegistry


class ExternalWaitTool:
    spec = ToolSpec(
        name="external_wait",
        description="Start external work and pause the run until a callback arrives.",
        input_schema={
            "type": "object",
            "properties": {
                "wait_id": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["wait_id"],
        },
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        return ToolObservation.waiting(
            str(invocation.arguments.get("description", "external work started")),
            wait_id=str(invocation.arguments["wait_id"]),
            reason="external_callback",
            pause_metadata={"example": "pause_resume_trace"},
        )


class DemoModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        if not any(message.role == "tool" for message in request.messages):
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="external_wait",
                        arguments={
                            "wait_id": "job-1",
                            "description": "waiting for job-1 callback",
                        },
                    )
                ]
            )
        return ModelResponse.text(f"resumed after callback: {request.messages[-1].text}")


def print_trace_summary(label: str, trace: Mapping[str, Any]) -> None:
    trace_dict = RunTrace.from_dict(trace).to_dict()
    replay = replay_trace(trace)
    print(
        json.dumps(
            {
                "label": label,
                "steps": len(trace_dict["steps"]),
                "replay_valid": replay.valid,
                "final_status": None if replay.final_status is None else replay.final_status.value,
            },
            indent=2,
            sort_keys=True,
        )
    )


async def main() -> None:
    agent = AgentLoop(model=DemoModel(), tools=ToolRegistry([ExternalWaitTool()]))

    paused = await agent.run([user_text("start external job")])
    if paused.snapshot is None or paused.trace is None:
        raise RuntimeError("expected paused run with snapshot and trace")

    print(f"paused status: {paused.status.value}")
    print_trace_summary("initial paused run", paused.trace)

    saved_snapshot_payload = paused.snapshot.to_dict()
    restored_snapshot = RunSnapshot.from_dict(saved_snapshot_payload)
    resume_agent = AgentLoop(model=DemoModel(), tools=ToolRegistry([ExternalWaitTool()]))

    resumed = await resume_agent.run_snapshot(
        ResumeInput(
            snapshot=restored_snapshot,
            append_messages=[user_text("job-1 completed successfully")],
            expected_pause=PauseSelector(
                source="tool",
                wait_id="job-1",
                metadata={"example": "pause_resume_trace"},
            ),
        )
    )
    if resumed.trace is None:
        raise RuntimeError("expected resume trace")

    print(f"resumed status: {resumed.status.value}")
    print(f"final text: {''.join(part.text or '' for part in resumed.final_parts)}")
    print_trace_summary("resume run", resumed.trace)


if __name__ == "__main__":
    asyncio.run(main())
