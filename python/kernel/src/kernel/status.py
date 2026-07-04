"""Core run status vocabulary."""

from __future__ import annotations

from enum import StrEnum


class AgentStatus(StrEnum):
    """Runtime state machine status values."""

    PLANNING = "planning"
    EXECUTING_TOOLS = "executing_tools"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    LIMIT_EXCEEDED = "limit_exceeded"


# Legal working statuses restored from a resumable checkpoint or pause payload.
CHECKPOINT_RESUME_STATUSES = frozenset(
    {
        AgentStatus.PLANNING,
        AgentStatus.EXECUTING_TOOLS,
    }
)

TERMINAL_STATUSES = frozenset(
    {
        AgentStatus.PAUSED,
        AgentStatus.COMPLETED,
        AgentStatus.FAILED,
        AgentStatus.LIMIT_EXCEEDED,
    }
)
