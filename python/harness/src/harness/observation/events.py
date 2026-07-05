"""Event-stream drivers and collectors for runtime scenarios."""

from __future__ import annotations

from collections.abc import Sequence

from kernel import AgentEvent, AgentLoop, Message, RunController, RuntimeContext


async def collect_events(
    agent: AgentLoop,
    messages: Sequence[Message],
    *,
    context: RuntimeContext | None = None,
    stream: bool = False,
    controller: RunController | None = None,
) -> list[AgentEvent]:
    """Collect one run event stream in order."""

    return [
        event
        async for event in agent.run_events(
            messages,
            context=context,
            stream=stream,
            controller=controller,
        )
    ]
