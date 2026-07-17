from __future__ import annotations

import importlib

import conformance
import jharness.kernel as kernel
import jharness.kernel.diagnostics as diagnostics
import jharness.kernel.wire as wire
import jharness.models as models
import jharness.toolkit as toolkit
import jharness.tools as tools


def test_package_all_exports_exist() -> None:
    for module in (kernel, toolkit, tools, diagnostics, wire, conformance):
        exports = set(module.__all__)
        assert exports
        assert all(hasattr(module, name) for name in exports)
    assert models.__all__ == []


def test_kernel_root_contains_the_documented_protocol_families() -> None:
    required = {
        "Runtime",
        "Invocation",
        "Checkpoint",
        "RunSnapshot",
        "Planning",
        "ToolsPending",
        "Suspended",
        "Completed",
        "Failed",
        "Limited",
        "RunRepository",
        "Model",
        "ToolCatalogProvider",
        "ApprovalPolicy",
        "HistoryReducer",
        "BatchPolicy",
    }
    assert required <= set(kernel.__all__)
    assert {"build_trace", "verify_trace", "RunTrace"} <= set(diagnostics.__all__)
    assert {"encode_checkpoint", "decode_checkpoint", "StartRequest"} <= set(wire.__all__)


def test_only_documented_model_namespaces_are_public() -> None:
    for namespace in (
        "jharness.models.openai",
        "jharness.models.anthropic",
        "jharness.models.deepseek",
    ):
        module = importlib.import_module(namespace)
        assert module.__all__
        assert all(hasattr(module, name) for name in module.__all__)
    for implementation in (
        "jharness.models.openai.chat_completions",
        "jharness.models.anthropic.messages_api",
    ):
        assert importlib.import_module(implementation).__all__ == []
