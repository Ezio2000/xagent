"""Scripted approval policy doubles for runtime scenarios."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from kernel import ApprovalDecision, ApprovalRequest


class StaticApprovalPolicy:
    """Approval policy that always returns the same decision."""

    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.requests: list[ApprovalRequest] = []

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return self.decision


class ApprovalPolicyByCall:
    """Approval policy keyed by tool call id, defaulting to allow."""

    def __init__(self, decisions: Mapping[str, ApprovalDecision]) -> None:
        self.decisions = dict(decisions)
        self.requests: list[ApprovalRequest] = []

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return self.decisions.get(request.tool_call.id, ApprovalDecision.allow())


class SequencedApprovalPolicy:
    """Approval policy that returns decisions in order, then defaults to allow."""

    def __init__(self, decisions: Sequence[ApprovalDecision]) -> None:
        self.decisions = list(decisions)
        self.requests: list[ApprovalRequest] = []

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        if not self.decisions:
            return ApprovalDecision.allow()
        return self.decisions.pop(0)


class FailingApprovalPolicy:
    """Approval policy double that raises on every decision."""

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        _ = request
        raise RuntimeError("approval backend unavailable")
