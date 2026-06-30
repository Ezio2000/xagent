"""Runtime error hierarchy."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast


def _empty_mapping() -> Mapping[str, Any]:
    return {}


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return deepcopy(dict(value or {}))


class AgentError(Exception):
    """Base class for runtime errors."""


@dataclass(slots=True, frozen=True)
class ModelErrorInfo:
    """Provider-neutral model failure details."""

    message: str
    provider: str | None = None
    code: str | None = None
    status_code: int | None = None
    retryable: bool = False
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError("model error message must not be empty")
        if self.status_code is not None and self.status_code < 100:
            raise ValueError("model error status_code must be >= 100")
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelErrorInfo:
        return cls(
            message=str(value["message"]),
            provider=cast(str | None, value.get("provider")),
            code=cast(str | None, value.get("code")),
            status_code=cast(int | None, value.get("status_code")),
            retryable=bool(value.get("retryable", False)),
            request_id=cast(str | None, value.get("request_id")),
            metadata=_copy_mapping(cast(Mapping[str, Any] | None, value.get("metadata"))),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.provider is not None:
            data["provider"] = self.provider
        if self.code is not None:
            data["code"] = self.code
        if self.status_code is not None:
            data["status_code"] = self.status_code
        if self.request_id is not None:
            data["request_id"] = self.request_id
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        return data


class ModelError(AgentError):
    """The model client failed."""


class ModelProviderError(ModelError):
    """A provider-backed model client failed with structured details."""

    info: ModelErrorInfo

    def __init__(self, info: ModelErrorInfo) -> None:
        self.info = info
        super().__init__(info.message)


class ToolError(AgentError):
    """A tool failed before it could return a ToolResult."""


class LimitExceeded(AgentError):
    """A configured loop limit was exceeded."""


class InvalidToolCall(AgentError):
    """The model requested an invalid or unknown tool call."""


class DuplicateToolError(AgentError):
    """A tool registry received duplicate tool names."""
