from __future__ import annotations

from typing import Any, cast

import pytest
from kernel import (
    AgentError,
    ModelCapabilities,
    ModelContentDelta,
    ModelStreamAccumulator,
    ModelToolCallDelta,
    model_capabilities,
)
from modelkit import ModelStreamAccumulator as ModelkitModelStreamAccumulator
from modelkit import model_capabilities as modelkit_model_capabilities


def test_modelkit_reexports_kernel_stream_helpers() -> None:
    assert ModelkitModelStreamAccumulator is ModelStreamAccumulator
    assert modelkit_model_capabilities is model_capabilities


def test_stream_accumulator_rejects_inconsistent_content_part_type() -> None:
    accumulator = ModelkitModelStreamAccumulator()

    accumulator.apply(ModelContentDelta(index=0, part_type="text", text_delta="a"))

    with pytest.raises(AgentError, match="part_type changed"):
        accumulator.apply(ModelContentDelta(index=0, part_type="custom", text_delta="b"))


def test_stream_accumulator_rejects_non_object_tool_arguments() -> None:
    accumulator = ModelkitModelStreamAccumulator()

    accumulator.apply(ModelToolCallDelta(index=0, id="call-1", name="search"))
    accumulator.apply(ModelToolCallDelta(index=0, arguments_delta="[]"))

    with pytest.raises(AgentError, match="decode to an object"):
        accumulator.response()


def test_model_capabilities_accepts_value_mapping_and_callable() -> None:
    class ValueCapabilitiesModel:
        capabilities = ModelCapabilities(multimodal_input=True)

    class MappingCapabilitiesModel:
        capabilities = {"streaming": True}

    class CallableCapabilitiesModel:
        def capabilities(self) -> dict[str, bool]:
            return {"tools": True}

    assert modelkit_model_capabilities(ValueCapabilitiesModel()).multimodal_input
    assert modelkit_model_capabilities(MappingCapabilitiesModel()).streaming
    assert modelkit_model_capabilities(CallableCapabilitiesModel()).tools


def test_model_capabilities_rejects_invalid_capability_source() -> None:
    class InvalidCapabilitiesModel:
        capabilities = cast(Any, 1)

    with pytest.raises(TypeError, match="model capabilities"):
        modelkit_model_capabilities(InvalidCapabilitiesModel())
