from __future__ import annotations

import agent_runtime


def test_public_all_is_sorted_and_resolves_exports() -> None:
    assert list(agent_runtime.__all__) == sorted(agent_runtime.__all__)
    for name in agent_runtime.__all__:
        assert hasattr(agent_runtime, name), name


def test_p0_extension_protocol_exports_are_present() -> None:
    assert {
        "ApprovalAction",
        "ApprovalDecision",
        "ApprovalPolicy",
        "ApprovalRequest",
        "CheckpointSummary",
        "JournalRecord",
        "RunJournal",
        "RunStore",
        "StoredCheckpoint",
    } <= set(agent_runtime.__all__)
