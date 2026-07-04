from __future__ import annotations

import kernel


def test_public_all_is_sorted_and_resolves_exports() -> None:
    assert list(kernel.__all__) == sorted(kernel.__all__)
    for name in kernel.__all__:
        assert hasattr(kernel, name), name


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
        "ToolRegistryProtocol",
    } <= set(kernel.__all__)
