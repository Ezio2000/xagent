from __future__ import annotations

import diagnostics
import harness
import kernel
import modelkit
import prompting
import toolkit


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


def test_sibling_package_root_exports_are_sorted_and_resolve() -> None:
    for package in (diagnostics, harness, modelkit, prompting, toolkit):
        assert list(package.__all__) == sorted(package.__all__)
        for name in package.__all__:
            assert hasattr(package, name), name


def test_extracted_helpers_are_not_kernel_root_exports() -> None:
    assert {
        "ModelStreamAccumulator",
        "RunTrace",
        "ToolRegistry",
        "model_capabilities",
        "replay_trace",
        "user_text",
    }.isdisjoint(kernel.__all__)
