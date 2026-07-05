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
        "AfterModelHook",
        "AfterToolHook",
        "BeforeModelHook",
        "BeforeToolHook",
        "CheckpointSummary",
        "EventHook",
        "JournalRecord",
        "ModelErrorHook",
        "RunJournal",
        "RunStore",
        "StoredCheckpoint",
        "TransitionHook",
        "ToolRegistryProtocol",
    } <= set(kernel.__all__)


def test_sibling_owned_helpers_are_not_kernel_root_exports() -> None:
    assert {
        "AcceptableTool",
        "ExecutableTool",
        "InvocableTool",
        "RuntimeContextSnapshot",
        "RunTrace",
        "Tool",
        "ToolCancelChecker",
        "ToolExecutionContext",
        "ToolInvocation",
        "ToolProgressEmitter",
        "ToolRegistry",
        "replay_trace",
        "user_text",
    }.isdisjoint(kernel.__all__)
