"""Runtime error hierarchy."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast


def _empty_mapping() -> Mapping[str, Any]:
    return {}


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(dict(_expect_mapping(value, "mapping")))


def _expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _expect_optional_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string or null")
    return value


def _expect_optional_non_empty_str(value: object, label: str) -> str | None:
    text = _expect_optional_str(value, label)
    if text == "":
        raise ValueError(f"{label} must not be empty")
    return text


def _expect_optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer or null")
    return value


def _expect_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _expect_present_optional_str(
    value: Mapping[str, Any],
    key: str,
    label: str,
) -> str | None:
    if key not in value:
        return None
    raw = value[key]
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TypeError(f"{label} must be a string or null")
    return raw


def _expect_present_optional_int(
    value: Mapping[str, Any],
    key: str,
    label: str,
) -> int | None:
    if key not in value:
        return None
    raw = value[key]
    if raw is None:
        return None
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise TypeError(f"{label} must be an integer or null")
    return raw


def _expect_present_optional_bool(
    value: Mapping[str, Any],
    key: str,
    label: str,
) -> bool | None:
    if key not in value:
        return None
    raw = value[key]
    if raw is None:
        return None
    if not isinstance(raw, bool):
        raise TypeError(f"{label} must be a boolean or null")
    return raw


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{label} has unknown field(s): {names}")


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
        object.__setattr__(self, "message", _expect_str(self.message, "model error message"))
        object.__setattr__(
            self,
            "provider",
            _expect_optional_non_empty_str(self.provider, "model error provider"),
        )
        object.__setattr__(
            self, "code", _expect_optional_non_empty_str(self.code, "model error code")
        )
        object.__setattr__(
            self,
            "status_code",
            _expect_optional_int(self.status_code, "model error status_code"),
        )
        object.__setattr__(self, "retryable", _expect_bool(self.retryable, "model error retryable"))
        object.__setattr__(
            self,
            "request_id",
            _expect_optional_non_empty_str(self.request_id, "model error request_id"),
        )
        if not self.message:
            raise ValueError("model error message must not be empty")
        if self.status_code is not None and self.status_code < 100:
            raise ValueError("model error status_code must be >= 100")
        object.__setattr__(
            self,
            "metadata",
            deepcopy(dict(_expect_mapping(self.metadata, "model error metadata"))),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelErrorInfo:
        known = {
            "message",
            "provider",
            "code",
            "status_code",
            "retryable",
            "request_id",
            "metadata",
        }
        _reject_unknown_keys(value, known, "model error info")
        raw_metadata: object = value.get("metadata", {})
        return cls(
            message=_expect_str(value["message"], "model error message"),
            provider=_expect_present_optional_str(value, "provider", "model error provider"),
            code=_expect_present_optional_str(value, "code", "model error code"),
            status_code=_expect_present_optional_int(
                value, "status_code", "model error status_code"
            ),
            retryable=_expect_present_optional_bool(value, "retryable", "model error retryable")
            or False,
            request_id=_expect_present_optional_str(value, "request_id", "model error request_id"),
            metadata=_expect_mapping(raw_metadata, "model error metadata"),
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

    def __init__(self, info: object) -> None:
        if not isinstance(info, ModelErrorInfo):
            raise TypeError("model provider error info must be a ModelErrorInfo")
        self.info = info
        super().__init__(info.message)


class ToolError(AgentError):
    """A tool failed before it could return a tool output."""


class LimitExceeded(AgentError):
    """A configured loop limit was exceeded."""


class InvalidToolCall(AgentError):
    """The model requested an invalid or unknown tool call."""


class DuplicateToolError(AgentError):
    """A tool registry received duplicate tool names."""
