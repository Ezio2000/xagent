# pyright: reportPrivateUsage=false, reportPrivateImportUsage=false
from __future__ import annotations

import asyncio
import errno
import hashlib
import os
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from typing import Any, cast

import pytest

from jharness.kernel import (
    RunContext,
    SettledResult,
    ToolCall,
    ToolContext,
    ToolFailure,
    ToolSuccess,
    thaw_json_value,
)
from jharness.toolkit import Tool, ToolRegistry
from jharness.tools import EditTool, ReadTool, WriteTool
from jharness.tools.filesystem import _write_io
from jharness.tools.filesystem import write as write_module
from jharness.tools.filesystem._common import FilesystemFailure, OperationCancelled, Workspace


async def _emit_progress(_progress: Mapping[str, Any]) -> None:
    return None


@contextmanager
def _path_based_parent(
    _target: _write_io.MutationTarget,
) -> Generator[None, None, None]:
    """Exercise the path-based fallback used when directory handles are unavailable."""

    yield None


def _invoke(
    tool: Tool,
    arguments: Mapping[str, Any],
    *,
    is_cancelled: Callable[[], bool] = lambda: False,
    through_registry: bool = False,
) -> ToolSuccess | ToolFailure:
    async def invoke() -> ToolSuccess | ToolFailure:
        context = ToolContext(
            RunContext("mutation-run", time.monotonic() + 60),
            _emit_progress,
            is_cancelled,
        )
        call = ToolCall("mutation-call", tool.spec.name, arguments)
        if through_registry:
            catalog = await ToolRegistry((tool,)).open_catalog()
            result = await catalog.bind(call).invoke(context)
        else:
            result = await tool.invoke(call, context)
        assert isinstance(result, SettledResult)
        assert isinstance(result.outcome, ToolSuccess | ToolFailure)
        return result.outcome

    return asyncio.run(invoke())


def _success(outcome: ToolSuccess | ToolFailure) -> tuple[str, dict[str, object]]:
    assert isinstance(outcome, ToolSuccess)
    assert outcome.parts[0].text is not None
    structured = thaw_json_value(outcome.structured_content)
    assert isinstance(structured, dict)
    return outcome.parts[0].text, cast(dict[str, object], structured)


def _failure(outcome: ToolSuccess | ToolFailure, code: str) -> str:
    assert isinstance(outcome, ToolFailure)
    assert outcome.error.code == code
    return outcome.error.message


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _temp_files(root: Path) -> list[Path]:
    return list(root.rglob("*.jharness-*.tmp"))


def _sha256_schema() -> dict[str, object]:
    return {
        "type": "string",
        "minLength": 64,
        "maxLength": 64,
        "pattern": "^[0-9a-f]{64}$",
    }


def test_mutation_tool_specs_are_exact_and_registry_validates_them(tmp_path: Path) -> None:
    edit = EditTool(tmp_path, max_file_bytes=17)
    write = WriteTool(tmp_path, max_file_bytes=17)
    sha256 = _sha256_schema()

    for tool in (edit, write):
        assert tool.root == tmp_path.resolve()
        assert tool.spec.execution.concurrency == "serial"
        assert tool.spec.execution.read_only is False
        assert tool.spec.execution.idempotent is False
        assert tool.spec.parallel_safe is False
        assert tool.spec.risk.filesystem == "write"
        assert tool.spec.risk.network is None
        assert tool.spec.risk.subprocess is None
        assert tool.spec.risk.destructive is True
        assert tool.spec.risk.requires_approval is True
        assert tool.spec.risk.extra == {}

    assert thaw_json_value(edit.spec.input_schema) == {
        "type": "object",
        "required": ["file_path", "old_string", "new_string", "expected_sha256"],
        "properties": {
            "file_path": {"type": "string", "minLength": 1},
            "old_string": {"type": "string", "minLength": 1, "maxLength": 17},
            "new_string": {"type": "string", "maxLength": 17},
            "replace_all": {"type": "boolean", "default": False},
            "expected_sha256": sha256,
        },
        "additionalProperties": False,
    }
    assert thaw_json_value(edit.spec.output_schema) == {
        "anyOf": [
            {
                "type": "object",
                "required": [
                    "path",
                    "replacements",
                    "previous_sha256",
                    "sha256",
                    "bytes_written",
                ],
                "properties": {
                    "path": {"type": "string"},
                    "replacements": {"type": "integer", "minimum": 1},
                    "previous_sha256": sha256,
                    "sha256": sha256,
                    "bytes_written": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": False,
            },
            {"type": "null"},
        ]
    }
    assert thaw_json_value(write.spec.input_schema) == {
        "type": "object",
        "required": ["file_path", "content", "expected_sha256"],
        "properties": {
            "file_path": {"type": "string", "minLength": 1},
            "content": {"type": "string", "maxLength": 17},
            "expected_sha256": {"anyOf": [sha256, {"type": "null"}]},
        },
        "additionalProperties": False,
    }
    assert thaw_json_value(write.spec.output_schema) == {
        "anyOf": [
            {
                "type": "object",
                "required": [
                    "path",
                    "operation",
                    "previous_sha256",
                    "sha256",
                    "bytes_written",
                ],
                "properties": {
                    "path": {"type": "string"},
                    "operation": {"enum": ["created", "overwritten"]},
                    "previous_sha256": {"anyOf": [sha256, {"type": "null"}]},
                    "sha256": sha256,
                    "bytes_written": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": False,
            },
            {"type": "null"},
        ]
    }

    text, result = _success(
        _invoke(
            write,
            {"file_path": "catalog.txt", "content": "valid", "expected_sha256": None},
            through_registry=True,
        )
    )
    assert text == "Created catalog.txt (5 bytes)."
    assert result["operation"] == "created"


def test_read_reports_sha256_of_all_raw_bytes_for_partial_bom_read(tmp_path: Path) -> None:
    raw = b"\xef\xbb\xbffirst\r\nsecond\nthird\r"
    (tmp_path / "mixed.txt").write_bytes(raw)

    text, result = _success(
        _invoke(ReadTool(tmp_path), {"file_path": "mixed.txt", "offset": 2, "limit": 1})
    )

    assert text == f"SHA-256 (raw file bytes): {_digest(raw)}\n\n2: second"
    assert result == {
        "path": "mixed.txt",
        "content": "second",
        "sha256": _digest(raw),
        "start_line": 2,
        "end_line": 2,
        "next_offset": 3,
        "truncated": True,
    }
    output_schema = cast(dict[str, object], thaw_json_value(ReadTool(tmp_path).spec.output_schema))
    variants = cast(list[dict[str, object]], output_schema["anyOf"])
    properties = cast(dict[str, object], variants[0]["properties"])
    assert properties["sha256"] == _sha256_schema()


@pytest.mark.parametrize(
    ("name", "content"),
    (("empty.txt", ""), ("unicode.txt", "你好, JHarness 🌍")),
)
def test_write_creates_empty_and_unicode_files(
    tmp_path: Path,
    name: str,
    content: str,
) -> None:
    encoded = content.encode("utf-8")

    text, result = _success(
        _invoke(
            WriteTool(tmp_path),
            {"file_path": name, "content": content, "expected_sha256": None},
            through_registry=True,
        )
    )

    assert text == f"Created {name} ({len(encoded)} bytes)."
    assert result == {
        "path": name,
        "operation": "created",
        "previous_sha256": None,
        "sha256": _digest(encoded),
        "bytes_written": len(encoded),
    }
    assert (tmp_path / name).read_bytes() == encoded
    assert _temp_files(tmp_path) == []


def test_write_overwrites_only_the_expected_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "state.txt"
    previous = b"old state\r\n"
    updated = "new 🌱\n"
    path.write_bytes(previous)

    text, result = _success(
        _invoke(
            WriteTool(tmp_path),
            {
                "file_path": "state.txt",
                "content": updated,
                "expected_sha256": _digest(previous),
            },
            through_registry=True,
        )
    )

    encoded = updated.encode()
    assert text == f"Overwritten state.txt ({len(encoded)} bytes)."
    assert result == {
        "path": "state.txt",
        "operation": "overwritten",
        "previous_sha256": _digest(previous),
        "sha256": _digest(encoded),
        "bytes_written": len(encoded),
    }
    assert path.read_bytes() == encoded
    assert _temp_files(tmp_path) == []


def test_path_based_parent_fallback_creates_and_replaces_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_write_io, "_open_parent", _path_based_parent)
    tool = WriteTool(tmp_path)

    _, created = _success(
        _invoke(
            tool,
            {"file_path": "fallback.txt", "content": "created", "expected_sha256": None},
        )
    )
    created_raw = b"created"
    assert created["operation"] == "created"
    assert created["sha256"] == _digest(created_raw)
    assert (tmp_path / "fallback.txt").read_bytes() == created_raw

    _, overwritten = _success(
        _invoke(
            tool,
            {
                "file_path": "fallback.txt",
                "content": "replaced",
                "expected_sha256": _digest(created_raw),
            },
        )
    )
    replaced_raw = b"replaced"
    assert overwritten["operation"] == "overwritten"
    assert overwritten["previous_sha256"] == _digest(created_raw)
    assert overwritten["sha256"] == _digest(replaced_raw)
    assert (tmp_path / "fallback.txt").read_bytes() == replaced_raw
    assert _temp_files(tmp_path) == []


def test_write_same_content_satisfies_cas_without_allocating_a_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "same.txt"
    raw = b"unchanged"
    path.write_bytes(raw)

    def unexpected_temporary(*_args: object, **_kwargs: object) -> tuple[int, str]:
        pytest.fail("an unchanged CAS write must not allocate a temporary")

    monkeypatch.setattr(_write_io, "_open_temporary", unexpected_temporary)
    _, result = _success(
        _invoke(
            WriteTool(tmp_path),
            {
                "file_path": "same.txt",
                "content": "unchanged",
                "expected_sha256": _digest(raw),
            },
        )
    )
    assert result["operation"] == "overwritten"
    assert result["previous_sha256"] == _digest(raw)
    assert result["sha256"] == _digest(raw)
    assert path.read_bytes() == raw


def test_write_rejects_stale_create_and_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "state.txt"
    path.write_bytes(b"current")
    tool = WriteTool(tmp_path)

    _failure(
        _invoke(tool, {"file_path": "state.txt", "content": "new", "expected_sha256": None}),
        "stale_file",
    )
    _failure(
        _invoke(
            tool,
            {
                "file_path": "state.txt",
                "content": "new",
                "expected_sha256": "0" * 64,
            },
        ),
        "stale_file",
    )
    _failure(
        _invoke(
            tool,
            {
                "file_path": "missing.txt",
                "content": "new",
                "expected_sha256": "0" * 64,
            },
        ),
        "stale_file",
    )
    assert path.read_bytes() == b"current"
    assert not (tmp_path / "missing.txt").exists()
    assert _temp_files(tmp_path) == []


def test_write_rejects_missing_parent_and_workspace_escape(tmp_path: Path) -> None:
    tool = WriteTool(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.unlink(missing_ok=True)

    _failure(
        _invoke(
            tool,
            {"file_path": "missing/child.txt", "content": "x", "expected_sha256": None},
        ),
        "path_not_found",
    )
    _failure(
        _invoke(
            tool,
            {"file_path": str(outside), "content": "x", "expected_sha256": None},
        ),
        "path_outside_workspace",
    )
    assert not (tmp_path / "missing").exists()
    assert not outside.exists()
    assert _temp_files(tmp_path) == []


@pytest.mark.parametrize(
    ("content", "max_file_bytes", "code"),
    (
        ("a\x00b", 10, "binary_content"),
        ("\ud800", 10, "invalid_utf8"),
        ("éé", 3, "content_too_large"),
    ),
)
def test_write_rejects_non_text_and_oversized_encoded_content(
    tmp_path: Path,
    content: str,
    max_file_bytes: int,
    code: str,
) -> None:
    _failure(
        _invoke(
            WriteTool(tmp_path, max_file_bytes=max_file_bytes),
            {"file_path": "bad.txt", "content": content, "expected_sha256": None},
        ),
        code,
    )
    assert not (tmp_path / "bad.txt").exists()
    assert _temp_files(tmp_path) == []


def test_write_honors_immediate_cancellation_without_side_effects(tmp_path: Path) -> None:
    _failure(
        _invoke(
            WriteTool(tmp_path),
            {"file_path": "cancelled.txt", "content": "never", "expected_sha256": None},
            is_cancelled=lambda: True,
        ),
        "cancelled",
    )
    assert not (tmp_path / "cancelled.txt").exists()
    assert _temp_files(tmp_path) == []


@pytest.mark.parametrize(
    ("file_path", "code"),
    (("", "invalid_path"), ("bad\x00path", "invalid_path"), (".", "not_a_file")),
)
def test_write_rejects_invalid_target_paths(
    tmp_path: Path,
    file_path: str,
    code: str,
) -> None:
    _failure(
        _invoke(
            WriteTool(tmp_path),
            {"file_path": file_path, "content": "x", "expected_sha256": None},
        ),
        code,
    )
    assert _temp_files(tmp_path) == []


@pytest.mark.skipif(os.name != "nt", reason="drive-relative paths are a Windows concept")
def test_write_rejects_drive_relative_target(tmp_path: Path) -> None:
    _failure(
        _invoke(
            WriteTool(tmp_path),
            {"file_path": "C:relative.txt", "content": "x", "expected_sha256": None},
        ),
        "path_outside_workspace",
    )


def test_write_normalizes_path_resolution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = WriteTool(tmp_path)

    def fail_abspath(_path: object) -> str:
        raise OSError("unresolvable")

    monkeypatch.setattr(_write_io.os.path, "abspath", fail_abspath)
    _failure(
        _invoke(
            tool,
            {"file_path": "file.txt", "content": "x", "expected_sha256": None},
        ),
        "invalid_path",
    )


def test_edit_unique_match_and_delete(tmp_path: Path) -> None:
    path = tmp_path / "app.py"
    original = b"before\nold value\nafter\n"
    path.write_bytes(original)
    tool = EditTool(tmp_path)

    _, replaced = _success(
        _invoke(
            tool,
            {
                "file_path": "app.py",
                "old_string": "old value",
                "new_string": "new value",
                "expected_sha256": _digest(original),
            },
            through_registry=True,
        )
    )
    first_update = b"before\nnew value\nafter\n"
    assert replaced == {
        "path": "app.py",
        "replacements": 1,
        "previous_sha256": _digest(original),
        "sha256": _digest(first_update),
        "bytes_written": len(first_update),
    }

    _, deleted = _success(
        _invoke(
            tool,
            {
                "file_path": "app.py",
                "old_string": "new value\n",
                "new_string": "",
                "expected_sha256": _digest(first_update),
            },
        )
    )
    final = b"before\nafter\n"
    assert deleted["previous_sha256"] == _digest(first_update)
    assert deleted["sha256"] == _digest(final)
    assert deleted["bytes_written"] == len(final)
    assert path.read_bytes() == final
    assert _temp_files(tmp_path) == []


def test_edit_rejects_empty_old_string_even_without_registry_validation(tmp_path: Path) -> None:
    path = tmp_path / "text.txt"
    raw = b"text"
    path.write_bytes(raw)

    _failure(
        _invoke(
            EditTool(tmp_path),
            {
                "file_path": "text.txt",
                "old_string": "",
                "new_string": "new",
                "expected_sha256": _digest(raw),
            },
        ),
        "invalid_input",
    )
    assert path.read_bytes() == raw


def test_edit_replace_all_reports_exact_count(tmp_path: Path) -> None:
    path = tmp_path / "many.txt"
    original = b"old middle old oldish\n"
    path.write_bytes(original)

    _, result = _success(
        _invoke(
            EditTool(tmp_path),
            {
                "file_path": "many.txt",
                "old_string": "old",
                "new_string": "new",
                "replace_all": True,
                "expected_sha256": _digest(original),
            },
        )
    )

    updated = b"new middle new newish\n"
    assert result["replacements"] == 3
    assert result["sha256"] == _digest(updated)
    assert path.read_bytes() == updated


def test_edit_preserves_bom_and_untouched_mixed_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "mixed.txt"
    original = b"\xef\xbb\xbfone\r\ntarget\r\nx\ntail\r"
    path.write_bytes(original)

    _, result = _success(
        _invoke(
            EditTool(tmp_path),
            {
                "file_path": "mixed.txt",
                "old_string": "target\nx",
                "new_string": "done\nz",
                "expected_sha256": _digest(original),
            },
        )
    )

    updated = b"\xef\xbb\xbfone\r\ndone\r\nz\ntail\r"
    assert path.read_bytes() == updated
    assert result["previous_sha256"] == _digest(original)
    assert result["sha256"] == _digest(updated)


def test_edit_rejects_stale_missing_nonunique_not_found_and_no_changes(tmp_path: Path) -> None:
    path = tmp_path / "text.txt"
    original = b"same same\n"
    path.write_bytes(original)
    tool = EditTool(tmp_path)
    base = {
        "file_path": "text.txt",
        "new_string": "new",
        "expected_sha256": _digest(original),
    }

    _failure(_invoke(tool, {**base, "old_string": "same"}), "old_string_not_unique")
    _failure(_invoke(tool, {**base, "old_string": "absent"}), "old_string_not_found")
    _failure(
        _invoke(tool, {**base, "old_string": "same", "new_string": "same"}),
        "no_changes",
    )
    _failure(
        _invoke(tool, {**base, "old_string": "same", "expected_sha256": "0" * 64}),
        "stale_file",
    )
    _failure(
        _invoke(
            tool,
            {
                "file_path": "missing.txt",
                "old_string": "old",
                "new_string": "new",
                "expected_sha256": "0" * 64,
            },
        ),
        "path_not_found",
    )
    assert path.read_bytes() == original
    assert _temp_files(tmp_path) == []


@pytest.mark.parametrize(
    ("raw", "max_file_bytes", "code"),
    ((b"a\x00b", 10, "binary_file"), (b"\xff", 10, "invalid_utf8"), (b"four", 3, "file_too_large")),
)
def test_edit_rejects_binary_invalid_utf8_and_oversized_source(
    tmp_path: Path,
    raw: bytes,
    max_file_bytes: int,
    code: str,
) -> None:
    path = tmp_path / "bad.txt"
    path.write_bytes(raw)

    _failure(
        _invoke(
            EditTool(tmp_path, max_file_bytes=max_file_bytes),
            {
                "file_path": "bad.txt",
                "old_string": "a",
                "new_string": "b",
                "expected_sha256": _digest(raw),
            },
        ),
        code,
    )
    assert path.read_bytes() == raw
    assert _temp_files(tmp_path) == []


def test_edit_rejects_oversized_result(tmp_path: Path) -> None:
    path = tmp_path / "small.txt"
    original = b"a"
    path.write_bytes(original)

    _failure(
        _invoke(
            EditTool(tmp_path, max_file_bytes=3),
            {
                "file_path": "small.txt",
                "old_string": "a",
                "new_string": "éé",
                "expected_sha256": _digest(original),
            },
        ),
        "content_too_large",
    )
    assert path.read_bytes() == original
    assert _temp_files(tmp_path) == []


def test_edit_rejects_oversized_direct_input_before_matching(tmp_path: Path) -> None:
    path = tmp_path / "small.txt"
    original = b"a"
    path.write_bytes(original)

    _failure(
        _invoke(
            EditTool(tmp_path, max_file_bytes=3),
            {
                "file_path": "small.txt",
                "old_string": "a",
                "new_string": "four",
                "expected_sha256": _digest(original),
            },
        ),
        "content_too_large",
    )
    assert path.read_bytes() == original
    assert _temp_files(tmp_path) == []


def test_edit_rejects_replace_all_expansion_before_joining(tmp_path: Path) -> None:
    path = tmp_path / "small.txt"
    original = b"aa"
    path.write_bytes(original)

    _failure(
        _invoke(
            EditTool(tmp_path, max_file_bytes=3),
            {
                "file_path": "small.txt",
                "old_string": "a",
                "new_string": "xx",
                "replace_all": True,
                "expected_sha256": _digest(original),
            },
        ),
        "content_too_large",
    )
    assert path.read_bytes() == original
    assert _temp_files(tmp_path) == []


def test_edit_rejects_expansion_when_untouched_tail_exceeds_limit(tmp_path: Path) -> None:
    path = tmp_path / "small.txt"
    original = b"abbb"
    path.write_bytes(original)

    _failure(
        _invoke(
            EditTool(tmp_path, max_file_bytes=4),
            {
                "file_path": "small.txt",
                "old_string": "a",
                "new_string": "xx",
                "expected_sha256": _digest(original),
            },
        ),
        "content_too_large",
    )
    assert path.read_bytes() == original
    assert _temp_files(tmp_path) == []


def test_write_rejects_character_count_over_limit_before_encoding(tmp_path: Path) -> None:
    _failure(
        _invoke(
            WriteTool(tmp_path, max_file_bytes=3),
            {"file_path": "large.txt", "content": "four", "expected_sha256": None},
        ),
        "content_too_large",
    )
    assert not (tmp_path / "large.txt").exists()


def test_edit_honors_immediate_cancellation_without_side_effects(tmp_path: Path) -> None:
    path = tmp_path / "cancelled.txt"
    original = b"old"
    path.write_bytes(original)

    _failure(
        _invoke(
            EditTool(tmp_path),
            {
                "file_path": "cancelled.txt",
                "old_string": "old",
                "new_string": "new",
                "expected_sha256": _digest(original),
            },
            is_cancelled=lambda: True,
        ),
        "cancelled",
    )
    assert path.read_bytes() == original
    assert _temp_files(tmp_path) == []


def test_mid_write_cancellation_removes_temporary_and_preserves_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state.txt"
    original = b"old"
    path.write_bytes(original)

    def cancel_after_partial_write(
        descriptor: int,
        data: bytes,
        _cancelled: Callable[[], bool],
    ) -> None:
        assert data == b"replacement"
        assert os.write(descriptor, data[:2]) == 2
        raise OperationCancelled

    monkeypatch.setattr(_write_io, "_write_all", cancel_after_partial_write)
    _failure(
        _invoke(
            WriteTool(tmp_path),
            {
                "file_path": "state.txt",
                "content": "replacement",
                "expected_sha256": _digest(original),
            },
        ),
        "cancelled",
    )

    assert path.read_bytes() == original
    assert _temp_files(tmp_path) == []


def test_second_cas_check_detects_a_racing_overwrite_and_removes_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state.txt"
    original = b"old"
    raced = b"external update"
    path.write_bytes(original)
    original_write_all = _write_io._write_all

    def write_then_race(
        descriptor: int,
        data: bytes,
        cancelled: Callable[[], bool],
    ) -> None:
        original_write_all(descriptor, data, cancelled)
        path.write_bytes(raced)

    monkeypatch.setattr(_write_io, "_write_all", write_then_race)
    _failure(
        _invoke(
            WriteTool(tmp_path),
            {
                "file_path": "state.txt",
                "content": "ours",
                "expected_sha256": _digest(original),
            },
        ),
        "stale_file",
    )
    assert path.read_bytes() == raced
    assert _temp_files(tmp_path) == []


@pytest.mark.parametrize("worker_outcome", ["return", "cancel", "error"])
def test_run_mutation_settles_worker_after_task_cancellation(worker_outcome: str) -> None:
    started = Event()

    def worker(cancelled: Callable[[], bool]) -> str:
        started.set()
        while not cancelled():
            time.sleep(0.001)
        if worker_outcome == "cancel":
            raise OperationCancelled
        if worker_outcome == "error":
            raise RuntimeError("worker failed after cancellation")
        return "committed"

    async def cancel_task() -> str:
        task = asyncio.create_task(_write_io.run_mutation(worker, lambda: False))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        if worker_outcome == "return":
            return await task
        with pytest.raises(asyncio.CancelledError):
            await task
        return "cancelled"

    expected = "committed" if worker_outcome == "return" else "cancelled"
    assert asyncio.run(cancel_task()) == expected


def test_newline_detection_covers_all_supported_styles() -> None:
    assert _write_io.detect_newline("a\r\nb\r\n") == "\r\n"
    assert _write_io.detect_newline("a\rb\r") == "\r"
    assert _write_io.detect_newline("a\nb") == "\n"


def test_target_lock_wait_is_cooperatively_cancellable(tmp_path: Path) -> None:
    target = _write_io.resolve_mutation_target(Workspace.create(tmp_path), "locked.txt")
    checks = iter((False, True))

    with (
        _write_io._target_lock(target, lambda: False),
        pytest.raises(OperationCancelled),
        _write_io._target_lock(target, lambda: next(checks, True)),
    ):
        pytest.fail("a second lock owner must not enter")
    assert _write_io._PATH_LOCKS == {}


def test_private_atomic_write_rejects_oversized_bytes(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path)
    with (
        _write_io.mutation_session(workspace, "large.txt", lambda: False) as session,
        pytest.raises(FilesystemFailure) as captured,
    ):
        _write_io.atomic_write(
            session,
            b"too large",
            expected_sha256=None,
            max_file_bytes=1,
            cancelled=lambda: False,
        )
    assert captured.value.code == "content_too_large"
    assert not (tmp_path / "large.txt").exists()


@pytest.mark.parametrize(
    ("error_number", "code"),
    (
        (errno.ENOENT, "path_not_found"),
        (errno.ENOTDIR, "not_a_directory"),
        (errno.ELOOP, "unsafe_path"),
        (errno.EACCES, "filesystem_error"),
    ),
)
def test_parent_os_errors_have_stable_failure_codes(
    tmp_path: Path,
    error_number: int,
    code: str,
) -> None:
    target = _write_io.resolve_mutation_target(Workspace.create(tmp_path), "child/file.txt")
    failure = _write_io._parent_failure(target, OSError(error_number, "test failure"))
    assert failure.code == code


def test_resolve_mutation_target_rejects_a_drive_relative_windows_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DriveRelativePath:
        drive = "C:"

        def expanduser(self) -> DriveRelativePath:
            return self

        def is_absolute(self) -> bool:
            return False

    def drive_relative_path(_value: object) -> DriveRelativePath:
        return DriveRelativePath()

    monkeypatch.setattr(_write_io, "Path", drive_relative_path)
    with pytest.raises(FilesystemFailure) as captured:
        _write_io.resolve_mutation_target(Workspace.create(tmp_path), "C:relative.txt")
    assert captured.value.code == "path_outside_workspace"


def test_edit_rejects_a_non_regular_opened_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "target.txt"
    raw = b"old"
    path.write_bytes(raw)

    def never_regular(_mode: int) -> bool:
        return False

    monkeypatch.setattr(_write_io.stat, "S_ISREG", never_regular)

    _failure(
        _invoke(
            EditTool(tmp_path),
            {
                "file_path": "target.txt",
                "old_string": "old",
                "new_string": "new",
                "expected_sha256": _digest(raw),
            },
        ),
        "not_a_file",
    )
    assert path.read_bytes() == raw


def test_mutation_rejects_a_reparse_target_without_following_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "target.txt"
    raw = b"old"
    path.write_bytes(raw)

    def always_reparse(_status: os.stat_result) -> bool:
        return True

    monkeypatch.setattr(_write_io, "_is_reparse", always_reparse)
    monkeypatch.setattr(_write_io, "_open_parent", _path_based_parent)

    _failure(
        _invoke(
            EditTool(tmp_path),
            {
                "file_path": "target.txt",
                "old_string": "old",
                "new_string": "new",
                "expected_sha256": _digest(raw),
            },
        ),
        "unsafe_path",
    )
    _failure(
        _invoke(
            WriteTool(tmp_path),
            {"file_path": "target.txt", "content": "new", "expected_sha256": None},
        ),
        "unsafe_path",
    )
    assert path.read_bytes() == raw


def test_edit_detects_target_identity_change_during_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "target.txt"
    raw = b"old"
    path.write_bytes(raw)
    identities = iter(((1, 1, 1, 1), (2, 2, 2, 2)))

    def next_identity(_status: os.stat_result) -> tuple[int, int, int, int]:
        return next(identities)

    monkeypatch.setattr(_write_io, "_identity", next_identity)
    monkeypatch.setattr(_write_io, "_open_parent", _path_based_parent)

    _failure(
        _invoke(
            EditTool(tmp_path),
            {
                "file_path": "target.txt",
                "old_string": "old",
                "new_string": "new",
                "expected_sha256": _digest(raw),
            },
        ),
        "stale_file",
    )
    assert path.read_bytes() == raw


@pytest.mark.parametrize(
    ("error_number", "code"),
    ((errno.ELOOP, "unsafe_path"), (errno.EACCES, "filesystem_error")),
)
def test_edit_normalizes_existing_target_open_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
    code: str,
) -> None:
    path = tmp_path / "target.txt"
    raw = b"old"
    path.write_bytes(raw)

    def fail_open(*_args: object, **_kwargs: object) -> int:
        raise OSError(error_number, "denied")

    monkeypatch.setattr(_write_io.os, "open", fail_open)
    monkeypatch.setattr(_write_io, "_open_parent", _path_based_parent)
    _failure(
        _invoke(
            EditTool(tmp_path),
            {
                "file_path": "target.txt",
                "old_string": "old",
                "new_string": "new",
                "expected_sha256": _digest(raw),
            },
        ),
        code,
    )
    assert path.read_bytes() == raw


def test_edit_rejects_opened_handle_outside_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "target.txt"
    raw = b"old"
    path.write_bytes(raw)

    def outside_path(_descriptor: int) -> Path:
        return tmp_path.parent / "outside.txt"

    monkeypatch.setattr(_write_io, "opened_file_path", outside_path)
    monkeypatch.setattr(_write_io, "_open_parent", _path_based_parent)

    _failure(
        _invoke(
            EditTool(tmp_path),
            {
                "file_path": "target.txt",
                "old_string": "old",
                "new_string": "new",
                "expected_sha256": _digest(raw),
            },
        ),
        "unsafe_path",
    )
    assert path.read_bytes() == raw


def test_temporary_name_collision_exhaustion_is_a_stable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def collide(*_args: object, **_kwargs: object) -> int:
        nonlocal attempts
        attempts += 1
        raise FileExistsError

    monkeypatch.setattr(_write_io.os, "open", collide)
    monkeypatch.setattr(_write_io, "_open_parent", _path_based_parent)
    _failure(
        _invoke(
            WriteTool(tmp_path),
            {"file_path": "new.txt", "content": "new", "expected_sha256": None},
        ),
        "filesystem_error",
    )
    assert attempts == 32
    assert not (tmp_path / "new.txt").exists()
    assert _temp_files(tmp_path) == []


def test_non_regular_temporary_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never_regular(_mode: int) -> bool:
        return False

    monkeypatch.setattr(_write_io.stat, "S_ISREG", never_regular)
    _failure(
        _invoke(
            WriteTool(tmp_path),
            {"file_path": "new.txt", "content": "new", "expected_sha256": None},
        ),
        "filesystem_error",
    )
    assert not (tmp_path / "new.txt").exists()
    assert _temp_files(tmp_path) == []


def test_temporary_handle_escape_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def escaped_path(_descriptor: int) -> Path:
        return tmp_path.parent / "escaped.tmp"

    monkeypatch.setattr(_write_io, "opened_file_path", escaped_path)
    monkeypatch.setattr(_write_io, "_open_parent", _path_based_parent)
    _failure(
        _invoke(
            WriteTool(tmp_path),
            {"file_path": "new.txt", "content": "new", "expected_sha256": None},
        ),
        "unsafe_path",
    )
    assert not (tmp_path / "new.txt").exists()
    assert _temp_files(tmp_path) == []


def test_zero_progress_write_is_normalized_and_temporary_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_progress(_descriptor: int, _data: bytes) -> int:
        return 0

    monkeypatch.setattr(_write_io.os, "write", no_progress)
    _failure(
        _invoke(
            WriteTool(tmp_path),
            {"file_path": "new.txt", "content": "new", "expected_sha256": None},
        ),
        "filesystem_error",
    )
    assert not (tmp_path / "new.txt").exists()
    assert _temp_files(tmp_path) == []


def test_create_commit_collision_is_stale_and_removes_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def collide(_session: object, _name: str, _identity: object) -> None:
        raise FileExistsError

    monkeypatch.setattr(_write_io, "_commit_create", collide)
    _failure(
        _invoke(
            WriteTool(tmp_path),
            {"file_path": "new.txt", "content": "new", "expected_sha256": None},
        ),
        "stale_file",
    )
    assert not (tmp_path / "new.txt").exists()
    assert _temp_files(tmp_path) == []


def test_disappearing_temporary_is_rejected_without_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_lstat = Path.lstat

    def disappear(path: Path) -> os.stat_result:
        if ".jharness-" in path.name:
            path.unlink()
            raise FileNotFoundError
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", disappear)
    monkeypatch.setattr(_write_io, "_open_parent", _path_based_parent)
    _failure(
        _invoke(
            WriteTool(tmp_path),
            {"file_path": "new.txt", "content": "new", "expected_sha256": None},
        ),
        "unsafe_path",
    )
    assert not (tmp_path / "new.txt").exists()
    assert _temp_files(tmp_path) == []


def test_changed_temporary_identity_is_rejected_without_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = iter(((1, 1, 1, 1), (2, 2, 2, 2)))

    def next_identity(_status: os.stat_result) -> tuple[int, int, int, int]:
        return next(identities)

    monkeypatch.setattr(_write_io, "_identity", next_identity)
    _failure(
        _invoke(
            WriteTool(tmp_path),
            {"file_path": "new.txt", "content": "new", "expected_sha256": None},
        ),
        "unsafe_path",
    )
    assert not (tmp_path / "new.txt").exists()
    assert _temp_files(tmp_path) == []


def test_cleanup_never_deletes_a_temporary_with_an_alien_inode(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path)
    target = _write_io.resolve_mutation_target(workspace, "result.txt")
    session = _write_io.MutationSession(target, None)
    temporary_name = ".result.txt.jharness-alien.tmp"
    temporary = tmp_path / temporary_name
    temporary.write_bytes(b"attacker replacement")
    status = temporary.stat()

    _write_io._remove_temporary(
        session,
        temporary_name,
        (status.st_dev, status.st_ino + 1),
    )

    assert temporary.read_bytes() == b"attacker replacement"
    temporary.unlink()


def test_write_normalizes_unexpected_atomic_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_atomic(*_args: object, **_kwargs: object) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(write_module, "atomic_write", fail_atomic)
    _failure(
        _invoke(
            WriteTool(tmp_path),
            {"file_path": "new.txt", "content": "new", "expected_sha256": None},
        ),
        "filesystem_error",
    )
    assert not (tmp_path / "new.txt").exists()


def test_mutation_refuses_file_symlinks_or_reparse_points(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    link = tmp_path / "link.txt"
    target.write_bytes(b"target")
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("file symlinks are unavailable for this test account")

    _failure(
        _invoke(
            WriteTool(tmp_path),
            {"file_path": "link.txt", "content": "new", "expected_sha256": None},
        ),
        "unsafe_path",
    )
    _failure(
        _invoke(
            EditTool(tmp_path),
            {
                "file_path": "link.txt",
                "old_string": "target",
                "new_string": "new",
                "expected_sha256": _digest(b"target"),
            },
        ),
        "unsafe_path",
    )
    assert target.read_bytes() == b"target"
    assert link.is_symlink()
    assert _temp_files(tmp_path) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission mode semantics")
def test_overwrite_preserves_posix_mode(tmp_path: Path) -> None:
    path = tmp_path / "executable.sh"
    original = b"#!/bin/sh\nexit 0\n"
    path.write_bytes(original)
    path.chmod(0o751)

    _success(
        _invoke(
            WriteTool(tmp_path),
            {
                "file_path": "executable.sh",
                "content": "#!/bin/sh\nexit 1\n",
                "expected_sha256": _digest(original),
            },
        )
    )

    assert path.stat().st_mode & 0o777 == 0o751
    assert _temp_files(tmp_path) == []
