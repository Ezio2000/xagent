"""Host-owned execution port used by the Agent preset tools."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jharness.kernel import RunContext
from jharness.tools.agent.models import AgentRequest, AgentSnapshot


@runtime_checkable
class AgentBackend(Protocol):
    """Create, observe, wait for, and cancel Host-owned child Agents.

    Implementations own authorization, idempotency, persistence, supervision, and
    deriving a fresh Child Runtime from the trusted parent configuration. Methods must
    be safe for concurrent calls from multiple immutable tool instances.
    """

    async def start_or_get(
        self,
        request: AgentRequest,
        *,
        parent: RunContext,
        parent_tool_call_id: str,
    ) -> AgentSnapshot:
        """Idempotently create or return the Agent for one parent tool call.

        For foreground requests, creation must also establish durable completion
        delivery so a fast Child cannot race the parent's waiting checkpoint.
        """

        ...

    async def get(
        self,
        agent_id: str,
        *,
        requester: RunContext,
    ) -> AgentSnapshot:
        """Return the current Agent snapshot without waiting."""

        ...

    async def wait_or_get(
        self,
        agent_id: str,
        *,
        requester: RunContext,
        requester_tool_call_id: str,
    ) -> AgentSnapshot:
        """Atomically register a durable waiter or return a terminal snapshot."""

        ...

    async def cancel(
        self,
        agent_id: str,
        *,
        requester: RunContext,
        requester_tool_call_id: str,
    ) -> AgentSnapshot:
        """Idempotently request cancellation and return the resulting snapshot."""

        ...


__all__ = ["AgentBackend"]
