"""Stable kernel error hierarchy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jharness.kernel._validation import (
    expect_bool,
    expect_instance,
    expect_non_empty_str,
    expect_optional_int,
    expect_optional_nonnegative_int,
    expect_optional_str,
    freeze_mapping,
)

if TYPE_CHECKING:
    from jharness.kernel.checkpoint import Checkpoint


class KernelError(Exception):
    """Base class for public kernel failures."""


class ProtocolError(KernelError):
    """A value or extension violated a kernel protocol invariant."""

    __slots__ = ("code",)

    def __init__(self, message: str, *, code: str = "protocol_error") -> None:
        super().__init__(expect_non_empty_str(message, "protocol error message"))
        self.code = expect_non_empty_str(code, "protocol error code")


class RequestError(KernelError):
    """A runtime request violated a stable request-level invariant."""

    __slots__ = ("code",)

    def __init__(self, code: str, message: str) -> None:
        super().__init__(expect_non_empty_str(message, "request error message"))
        self.code = expect_non_empty_str(code, "request error code")


class ToolError(KernelError):
    """A tool catalog, binding, or progress boundary failed."""


class RepositoryError(KernelError):
    """A repository operation failed."""


@dataclass(frozen=True, slots=True)
class RevisionConflict(RepositoryError):
    """Optimistic repository revision comparison failed."""

    run_id: str
    expected_revision: int | None
    actual_revision: int | None

    def __post_init__(self) -> None:
        expect_non_empty_str(self.run_id, "revision conflict run_id")
        expect_optional_nonnegative_int(
            self.expected_revision,
            "revision conflict expected_revision",
        )
        expect_optional_nonnegative_int(
            self.actual_revision,
            "revision conflict actual_revision",
        )

    def __str__(self) -> str:
        return (
            f"run {self.run_id!r} revision conflict: expected "
            f"{self.expected_revision!r}, found {self.actual_revision!r}"
        )


class CommitError(RepositoryError):
    """A durable commit failed; `last_checkpoint` remains authoritative."""

    __slots__ = ("last_checkpoint",)

    def __init__(self, message: str, *, last_checkpoint: Checkpoint | None) -> None:
        super().__init__(message)
        self.last_checkpoint: Checkpoint | None = last_checkpoint


@dataclass(frozen=True, slots=True)
class ModelErrorInfo:
    """Provider-neutral model failure details."""

    code: str
    message: str
    provider: str | None = None
    status_code: int | None = None
    retryable: bool = False
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        expect_non_empty_str(self.code, "model error code")
        expect_non_empty_str(self.message, "model error message")
        expect_optional_str(self.provider, "model error provider")
        expect_optional_int(self.status_code, "model error status_code")
        expect_bool(self.retryable, "model error retryable")
        expect_optional_str(self.request_id, "model error request_id")
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, "model error metadata"),
        )


class ModelError(KernelError):
    """A model client failed with provider-neutral details."""

    __slots__ = ("info",)

    def __init__(self, info: ModelErrorInfo) -> None:
        info = expect_instance(info, ModelErrorInfo, "model error info")
        super().__init__(info.message)
        self.info = info
