from __future__ import annotations

import pytest
from harness import (
    KernelScenario,
    ScriptedModel,
    assert_event_type_order,
    assert_result_status,
    assert_trace_replays,
    user_message,
)
from kernel import AgentStatus, EventTypes, ModelResponse


@pytest.mark.asyncio
async def test_kernel_scenario_runs_and_collects_events() -> None:
    scenario = KernelScenario(
        model=ScriptedModel([ModelResponse.text("done"), ModelResponse.text("done")])
    )

    result = await scenario.run([user_message("hello")])
    events = await scenario.events([user_message("hello")])

    assert_result_status(result, AgentStatus.COMPLETED)
    assert_event_type_order(
        events,
        [
            EventTypes.RUN_STARTED,
            EventTypes.MODEL_STARTED,
            EventTypes.MODEL_COMPLETED,
            EventTypes.RUN_COMPLETED,
        ],
    )
    assert result.trace is not None
    assert_trace_replays(result.trace)
