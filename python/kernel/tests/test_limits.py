from __future__ import annotations

from typing import Any, cast

import pytest
from kernel import AgentState, AgentStatus, LimitReasons, LoopLimits


def test_limit_validation() -> None:
    with pytest.raises(ValueError, match="max_iterations"):
        LoopLimits(max_iterations=0)

    with pytest.raises(ValueError, match="max_total_tool_calls"):
        LoopLimits(max_total_tool_calls=-1)

    with pytest.raises(ValueError, match="timeout_seconds"):
        LoopLimits(timeout_seconds=0)

    with pytest.raises(ValueError, match="max_parallel_tool_calls"):
        LoopLimits(max_parallel_tool_calls=0)

    with pytest.raises(ValueError, match="max_total_tokens"):
        LoopLimits(max_total_tokens=-1)

    with pytest.raises(ValueError, match="max_model_retries"):
        LoopLimits(max_model_retries=-1)

    with pytest.raises(TypeError, match="max_iterations"):
        LoopLimits(max_iterations=True)

    with pytest.raises(TypeError, match="timeout_seconds"):
        LoopLimits(timeout_seconds=True)

    with pytest.raises(TypeError, match="stop_on_tool_error"):
        LoopLimits(stop_on_tool_error=cast(Any, 1))

    with pytest.raises(TypeError, match="max_total_tokens"):
        LoopLimits(max_total_tokens=cast(Any, True))

    with pytest.raises(TypeError, match="max_model_retries"):
        LoopLimits(max_model_retries=cast(Any, True))


def test_limit_reason_constants_match_runtime_reason_strings() -> None:
    state = AgentState(status=AgentStatus.PLANNING, messages=[])
    state.iterations = 8
    state.total_tool_calls = 20

    assert LoopLimits().iteration_reason(state) == LimitReasons.MAX_ITERATIONS
    assert LoopLimits().tool_call_reason(state) == LimitReasons.MAX_TOTAL_TOOL_CALLS
    assert LimitReasons.TIMEOUT_SECONDS == "timeout_seconds"


def test_loop_limits_round_trips_wire_shape() -> None:
    limits = LoopLimits(
        max_iterations=3,
        max_total_tool_calls=5,
        timeout_seconds=1.5,
        stop_on_tool_error=True,
        max_parallel_tool_calls=2,
        max_total_tokens=100,
        max_model_retries=1,
    )

    restored = LoopLimits.from_dict(limits.to_dict())

    assert restored == limits


def test_loop_limits_from_dict_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown"):
        LoopLimits.from_dict({"legacy_budget": 1})
