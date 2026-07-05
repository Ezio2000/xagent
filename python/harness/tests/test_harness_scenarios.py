from __future__ import annotations

import pytest
from harness import (
    KernelScenario,
)
from kernel import (
    AgentStatus,
    ContentPart,
    EventTypes,
    Message,
    ModelRequest,
    ModelResponse,
    RuntimeContext,
)


class StaticModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        return ModelResponse.text("done")


@pytest.mark.asyncio
async def test_kernel_scenario_runs_and_collects_events() -> None:
    scenario = KernelScenario(model=StaticModel())

    message = Message.user([ContentPart.text_part("hello")])
    result = await scenario.run([message])
    events = await scenario.events([message])

    assert result.status is AgentStatus.COMPLETED
    event_types = [event.type for event in events]
    expected = [
        EventTypes.RUN_STARTED,
        EventTypes.MODEL_STARTED,
        EventTypes.MODEL_COMPLETED,
        EventTypes.RUN_COMPLETED,
    ]
    position = 0
    for event_type in event_types:
        if position < len(expected) and event_type == expected[position]:
            position += 1
    assert position == len(expected)
    assert result.trace is not None
