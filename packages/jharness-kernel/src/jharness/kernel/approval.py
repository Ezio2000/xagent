"""Ordered approval values and policy port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

from jharness.kernel._validation import expect_instance, expect_int, expect_non_empty_str
from jharness.kernel.messages import ToolCall
from jharness.kernel.state import Suspension
from jharness.kernel.tools import ToolRisk


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    batch_id: str
    index: int
    call: ToolCall
    risk: ToolRisk

    def __post_init__(self) -> None:
        expect_non_empty_str(self.batch_id, "approval batch_id")
        if expect_int(self.index, "approval index") < 0:
            raise ValueError("approval index must be >= 0")
        expect_instance(self.call, ToolCall, "approval call")
        expect_instance(self.risk, ToolRisk, "approval risk")


@dataclass(frozen=True, slots=True)
class ApprovalAllow:
    call_id: str

    def __post_init__(self) -> None:
        expect_non_empty_str(self.call_id, "approval call_id")


@dataclass(frozen=True, slots=True)
class ApprovalDeny:
    call_id: str
    reason: str

    def __post_init__(self) -> None:
        expect_non_empty_str(self.call_id, "approval call_id")
        expect_non_empty_str(self.reason, "approval reason")


@dataclass(frozen=True, slots=True)
class ApprovalSuspend:
    call_id: str
    suspension: Suspension

    def __post_init__(self) -> None:
        expect_non_empty_str(self.call_id, "approval call_id")
        expect_instance(self.suspension, Suspension, "approval suspension")


ApprovalDecision: TypeAlias = ApprovalAllow | ApprovalDeny | ApprovalSuspend


@runtime_checkable
class ApprovalPolicy(Protocol):
    """Decide one ordered, already-bound batch."""

    async def decide(
        self, requests: tuple[ApprovalRequest, ...]
    ) -> tuple[ApprovalDecision, ...]: ...
