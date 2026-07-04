from __future__ import annotations

import pytest
from harness import ScriptedModel, collect_events
from kernel import AgentLoop, EventTypes, ModelResponse
from prompting import user_text


@pytest.mark.asyncio
async def test_scripted_model_records_requests_and_returns_copies() -> None:
    model = ScriptedModel([ModelResponse.text("done")])

    result = await AgentLoop(model=model).run([user_text("hello")])

    assert result.final_parts[0].text == "done"
    assert model.calls == 1
    assert model.requests[0].messages[0].text == "hello"


@pytest.mark.asyncio
async def test_collect_events_returns_ordered_event_stream() -> None:
    events = await collect_events(
        AgentLoop(model=ScriptedModel([ModelResponse.text("done")])),
        [user_text("hello")],
    )

    assert events[0].type == EventTypes.RUN_STARTED
    assert events[-1].type == EventTypes.RUN_COMPLETED
