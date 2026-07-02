from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

from agent_runtime import (
    AgentError,
    DuplicateToolError,
    InvalidToolCall,
    LimitExceeded,
    ModelError,
    ModelErrorInfo,
    ModelProviderError,
    ToolError,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_ERROR_SCHEMA = json.loads((REPO_ROOT / "spec" / "v0" / "model-error.schema.json").read_text())
MODEL_ERROR_SCHEMA_VALIDATOR: Any = Draft202012Validator(MODEL_ERROR_SCHEMA)


def test_error_hierarchy_is_available_from_package_root() -> None:
    assert issubclass(ModelError, AgentError)
    assert issubclass(ModelProviderError, ModelError)
    assert issubclass(ToolError, AgentError)
    assert issubclass(LimitExceeded, AgentError)
    assert issubclass(InvalidToolCall, AgentError)
    assert issubclass(DuplicateToolError, AgentError)


def test_model_error_info_constructor_rejects_invalid_core_types() -> None:
    with pytest.raises(TypeError, match="message"):
        ModelErrorInfo(message=cast(Any, 123))

    with pytest.raises(TypeError, match="provider"):
        ModelErrorInfo(message="failed", provider=cast(Any, 123))

    with pytest.raises(TypeError, match="status_code"):
        ModelErrorInfo(message="failed", status_code=cast(Any, True))

    with pytest.raises(TypeError, match="retryable"):
        ModelErrorInfo(message="failed", retryable=cast(Any, 1))

    with pytest.raises(TypeError, match="metadata"):
        ModelErrorInfo(message="failed", metadata=cast(Any, []))

    with pytest.raises(TypeError, match="ModelErrorInfo"):
        ModelProviderError(cast(Any, {"message": "failed"}))

    with pytest.raises(ValueError, match="provider"):
        ModelErrorInfo(message="failed", provider="")

    with pytest.raises(ValueError, match="code"):
        ModelErrorInfo(message="failed", code="")

    with pytest.raises(ValueError, match="request_id"):
        ModelErrorInfo(message="failed", request_id="")


def test_model_error_info_from_dict_rejects_schema_invalid_fields() -> None:
    with pytest.raises(TypeError, match="message"):
        ModelErrorInfo.from_dict({"message": 123})

    with pytest.raises(TypeError, match="retryable"):
        ModelErrorInfo.from_dict({"message": "failed", "retryable": "false"})

    with pytest.raises(TypeError, match="metadata"):
        ModelErrorInfo.from_dict({"message": "failed", "metadata": None})

    with pytest.raises(ValueError, match="unknown"):
        ModelErrorInfo.from_dict({"message": "failed", "provider_payload": {}})

    with pytest.raises(ValueError, match="provider"):
        ModelErrorInfo.from_dict({"message": "failed", "provider": ""})

    with pytest.raises(ValueError, match="code"):
        ModelErrorInfo.from_dict({"message": "failed", "code": ""})

    with pytest.raises(ValueError, match="request_id"):
        ModelErrorInfo.from_dict({"message": "failed", "request_id": ""})


def test_model_error_info_matches_schema() -> None:
    info = ModelErrorInfo(
        message="rate limited",
        provider="test-provider",
        code="rate_limit",
        status_code=429,
        retryable=True,
        request_id="req-1",
        metadata={"tenant": "acme"},
    )
    payload = info.to_dict()

    assert not list(MODEL_ERROR_SCHEMA_VALIDATOR.iter_errors(payload))
    assert ModelErrorInfo.from_dict(payload).to_dict() == payload


def test_model_error_info_schema_accepts_explicit_null_optional_scalars() -> None:
    payload = {
        "message": "failed",
        "provider": None,
        "code": None,
        "status_code": None,
        "retryable": None,
        "request_id": None,
    }

    assert not list(MODEL_ERROR_SCHEMA_VALIDATOR.iter_errors(payload))
    assert ModelErrorInfo.from_dict(payload).to_dict() == {
        "message": "failed",
        "retryable": False,
    }
