"""State assertions for controlled runtime tests."""

from __future__ import annotations

from kernel import AgentResult, AgentStatus


def assert_result_status(result: AgentResult, status: AgentStatus) -> None:
    """Assert a terminal result status."""

    assert result.status is status
