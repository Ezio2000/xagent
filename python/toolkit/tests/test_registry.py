from __future__ import annotations

from typing import Any

import pytest
import toolkit.registry as registry_module
from kernel import (
    RuntimeContext,
    ToolCall,
    ToolExecutionContext,
    ToolInvocation,
    ToolObservation,
    ToolSpec,
)
from toolkit import ToolRegistry


class StrictCountTool:
    def __init__(self) -> None:
        self.calls = 0

    spec = ToolSpec(
        name="strict_count",
        description="Require an integer count.",
        input_schema={
            "type": "object",
            "required": ["count"],
            "properties": {"count": {"type": "integer"}},
            "additionalProperties": False,
        },
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        self.calls += 1
        return ToolObservation.text(str(invocation.arguments["count"]))


@pytest.mark.asyncio
async def test_tool_registry_reuses_cached_input_schema_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ToolRegistry([StrictCountTool()])

    def fail_validator(schema: object) -> Any:
        _ = schema
        raise AssertionError("validator must be cached at registration")

    monkeypatch.setattr(registry_module, "Draft202012Validator", fail_validator)

    result = await registry.invoke(
        ToolCall(id="call-1", name="strict_count", arguments={"count": 3}),
        RuntimeContext(),
    )

    assert result.text_content == "3"
