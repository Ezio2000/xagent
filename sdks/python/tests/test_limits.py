from __future__ import annotations

from typing import Any, cast

import pytest

from agent_runtime import LoopLimits


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
