from __future__ import annotations

import importlib
import subprocess
import sys

import conformance
import jharness.kernel as kernel
import jharness.kernel.diagnostics as diagnostics
import jharness.kernel.wire as wire
import jharness.models as models
import jharness.repository as repository
import jharness.toolkit as toolkit
import jharness.tools as tools


def test_package_all_exports_exist() -> None:
    for module in (kernel, toolkit, tools, repository, diagnostics, wire, conformance):
        exports = set(module.__all__)
        assert exports
        assert all(hasattr(module, name) for name in exports)
    assert models.__all__ == []


def test_repository_root_exports_all_supported_backends() -> None:
    assert set(repository.__all__) == {
        "MemoryRunRepository",
        "MySQLRunRepository",
        "RedisRunRepository",
        "SQLiteRunRepository",
    }


def test_repository_base_import_and_embedded_backends_need_no_optional_drivers() -> None:
    program = """
import asyncio
import builtins

original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and (name == "pymysql" or name == "redis" or name.startswith("redis.")):
        raise AssertionError(f"base repository import loaded optional driver: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

from jharness.repository import (
    MemoryRunRepository,
    MySQLRunRepository,
    RedisRunRepository,
    SQLiteRunRepository,
)

async def verify():
    memory = MemoryRunRepository()
    assert await memory.get_head("missing") is None
    sqlite = SQLiteRunRepository(":memory:")
    await sqlite.initialize()
    assert await sqlite.get_head("missing") is None
    await sqlite.close()

asyncio.run(verify())
assert MySQLRunRepository and RedisRunRepository
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


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
