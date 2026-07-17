# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
from threading import Event
from typing import Any, cast

import pytest
import regex

import jharness.tools as tools
from jharness.kernel import (
    RunContext,
    SettledResult,
    ToolCall,
    ToolContext,
    ToolError,
    ToolFailure,
    ToolSuccess,
    thaw_json_value,
)
from jharness.toolkit import Tool, ToolRegistry
from jharness.tools.filesystem import _common
from jharness.tools.filesystem._common import (
    FilesystemFailure,
    SearchBudget,
    Workspace,
    excluded_names,
    glob_filter_matches,
    glob_matches,
    is_excluded,
    is_in_excluded_directory,
    positive_float,
    positive_int,
    read_bytes_bounded,
    run_blocking,
    validate_glob_pattern,
)


async def _emit_progress(_progress: Mapping[str, Any]) -> None:
    return None


def _invoke(
    tool: Tool,
    arguments: Mapping[str, Any],
    *,
    is_cancelled: Callable[[], bool] = lambda: False,
) -> ToolSuccess | ToolFailure:
    async def invoke() -> ToolSuccess | ToolFailure:
        context = ToolContext(
            RunContext("run-1", time.monotonic() + 60),
            _emit_progress,
            is_cancelled,
        )
        result = await tool.invoke(ToolCall("call-1", tool.spec.name, arguments), context)
        assert isinstance(result, SettledResult)
        assert isinstance(result.outcome, ToolSuccess | ToolFailure)
        return result.outcome

    return asyncio.run(invoke())


def _success(outcome: ToolSuccess | ToolFailure) -> tuple[str, object]:
    assert isinstance(outcome, ToolSuccess)
    assert outcome.parts[0].text is not None
    return outcome.parts[0].text, thaw_json_value(outcome.structured_content)


def _failure(outcome: ToolSuccess | ToolFailure, code: str) -> str:
    assert isinstance(outcome, ToolFailure)
    assert outcome.error.code == code
    return outcome.error.message


def test_public_api_and_contracts(tmp_path: Path) -> None:
    assert tools.__all__ == [
        "AgentCancelTool",
        "AgentGetTool",
        "AgentTool",
        "AgentWaitTool",
        "AskQuestionTool",
        "BashTool",
        "EditTool",
        "GlobTool",
        "GrepTool",
        "ReadTool",
        "WriteTool",
    ]
    read_presets = (tools.ReadTool(tmp_path), tools.GlobTool(tmp_path), tools.GrepTool(tmp_path))
    write_presets = (tools.EditTool(tmp_path), tools.WriteTool(tmp_path))
    presets = (*read_presets, *write_presets)
    assert all(isinstance(tool, Tool) for tool in presets)
    assert [tool.spec.name for tool in presets] == ["Read", "Glob", "Grep", "Edit", "Write"]
    for tool in read_presets:
        assert tool.root == tmp_path.resolve()
        assert tool.spec.execution.concurrency == "parallel"
        assert tool.spec.execution.read_only is True
        assert tool.spec.execution.idempotent is True
        assert tool.spec.risk.filesystem == "read"
        assert tool.spec.risk.destructive is False
        assert tool.spec.risk.requires_approval is None
        input_schema = cast(dict[str, object], thaw_json_value(tool.spec.input_schema))
        properties = cast(dict[str, dict[str, object]], input_schema["properties"])
        if "pattern" in properties:
            assert properties["pattern"]["maxLength"] == 4_096
    for tool in write_presets:
        assert tool.root == tmp_path.resolve()
        assert tool.spec.execution.concurrency == "serial"
        assert tool.spec.execution.read_only is False
        assert tool.spec.execution.idempotent is False
        assert tool.spec.risk.filesystem == "write"
        assert tool.spec.risk.destructive is True
        assert tool.spec.risk.requires_approval is True

    async def open_catalog() -> tuple[str, ...]:
        catalog = await ToolRegistry(presets).open_catalog()
        return tuple(spec.name for spec in catalog.specs())

    assert asyncio.run(open_catalog()) == ("Read", "Glob", "Grep", "Edit", "Write")


def test_registry_validates_inputs_and_all_output_branches(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("needle\n", encoding="utf-8")
    presets = (tools.ReadTool(tmp_path), tools.GlobTool(tmp_path), tools.GrepTool(tmp_path))

    async def invoke_through_registry() -> list[str]:
        catalog = await ToolRegistry(presets).open_catalog()
        context = ToolContext(
            RunContext("run-1", time.monotonic() + 60),
            _emit_progress,
            lambda: False,
        )
        calls = (
            ToolCall("read", "Read", {"file_path": "app.py"}),
            ToolCall("glob", "Glob", {"pattern": "*.py"}),
            ToolCall("grep", "Grep", {"pattern": "needle"}),
            ToolCall(
                "grep-content",
                "Grep",
                {"pattern": "needle", "output_mode": "content"},
            ),
            ToolCall(
                "grep-count",
                "Grep",
                {"pattern": "needle", "output_mode": "count"},
            ),
            ToolCall("failure", "Read", {"file_path": "missing.py"}),
        )
        outcomes: list[str] = []
        for call in calls:
            result = await catalog.bind(call).invoke(context)
            outcomes.append(result.outcome.kind)
        with pytest.raises(ToolError, match="do not match input_schema"):
            catalog.bind(ToolCall("invalid", "Read", {"file_path": "app.py", "offset": 0}))
        return outcomes

    assert asyncio.run(invoke_through_registry()) == [
        "success",
        "success",
        "success",
        "success",
        "success",
        "failure",
    ]


@pytest.mark.parametrize("value", [0, -1, True, "1"])
def test_positive_int_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="size must be a positive integer"):
        positive_int(value, "size")


@pytest.mark.parametrize("value", [0, -1, True, "1", float("inf"), float("nan")])
def test_positive_float_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="timeout must be a positive finite number"):
        positive_float(value, "timeout")
    assert positive_float(1, "timeout") == 1.0


def test_search_budget_failures() -> None:
    timed_out = SearchBudget(lambda: False, float("-inf"), 1)
    with pytest.raises(FilesystemFailure) as timeout:
        timed_out.checkpoint()
    assert timeout.value.code == "search_timeout"

    entries = SearchBudget(lambda: False, float("inf"), 1)
    assert entries.remaining_byte_limit(5) == 5
    entries.consume_entry()
    with pytest.raises(FilesystemFailure) as entry_limit:
        entries.consume_entry()
    assert entry_limit.value.code == "search_budget_exceeded"

    byte_budget = SearchBudget(lambda: False, float("inf"), 1, max_bytes=0)
    assert byte_budget.remaining_byte_limit(5) == 0
    with pytest.raises(FilesystemFailure) as byte_limit:
        byte_budget.consume_bytes(1)
    assert byte_limit.value.code == "search_budget_exceeded"


def test_constructor_validation(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="does not exist"):
        tools.ReadTool(missing)
    file_root = tmp_path / "file.txt"
    file_root.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        tools.ReadTool(file_root)
    with pytest.raises(ValueError, match="cannot exceed"):
        tools.ReadTool(tmp_path, default_limit=2, max_limit=1)
    with pytest.raises(ValueError, match="cannot exceed"):
        tools.GlobTool(tmp_path, default_limit=2, max_limit=1)
    with pytest.raises(ValueError, match="cannot exceed"):
        tools.GrepTool(tmp_path, default_limit=2, max_limit=1)
    with pytest.raises(ValueError, match="max_file_bytes"):
        tools.ReadTool(tmp_path, max_file_bytes=0)
    with pytest.raises(ValueError, match="max_context"):
        tools.GrepTool(tmp_path, max_context=0)
    with pytest.raises(ValueError, match="max_file_bytes"):
        tools.GrepTool(tmp_path, max_file_bytes=0)
    with pytest.raises(ValueError, match="max_line_chars"):
        tools.GrepTool(tmp_path, max_line_chars=0)
    with pytest.raises(ValueError, match="regex_timeout_seconds"):
        tools.GrepTool(tmp_path, regex_timeout_seconds=0)
    with pytest.raises(ValueError, match="max_pattern_chars"):
        tools.GlobTool(tmp_path, max_pattern_chars=0)
    with pytest.raises(ValueError, match="max_pattern_components"):
        tools.GrepTool(tmp_path, max_pattern_components=0)
    with pytest.raises(ValueError, match="max_search_seconds"):
        tools.GlobTool(tmp_path, max_search_seconds=0)
    with pytest.raises(ValueError, match="max_scanned_entries"):
        tools.GrepTool(tmp_path, max_scanned_entries=0)
    with pytest.raises(ValueError, match="max_total_bytes"):
        tools.GrepTool(tmp_path, max_total_bytes=0)
    with pytest.raises(ValueError, match="individual path names"):
        tools.GlobTool(tmp_path, excluded_directory_names=("a/b",))
    with pytest.raises(ValueError, match="individual path names"):
        tools.GrepTool(tmp_path, excluded_directory_names=("",))


def test_workspace_boundaries_and_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = Workspace.create(tmp_path)
    assert workspace.display(tmp_path) == "."
    assert workspace.safe_match(tmp_path) is None
    assert workspace.safe_match(tmp_path / "missing") is None
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    assert workspace.safe_match(outside) is None
    with pytest.raises(FilesystemFailure) as absolute:
        workspace.resolve(str(outside))
    assert absolute.value.code == "path_outside_workspace"
    with pytest.raises(FilesystemFailure) as relative:
        workspace.resolve("../outside.txt")
    assert relative.value.code == "path_outside_workspace"

    original_resolve = Path.resolve

    def broken_resolve(path: Path, strict: bool = False) -> Path:
        if path.name == "broken":
            raise OSError("broken")
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", broken_resolve)
    with pytest.raises(FilesystemFailure) as broken:
        workspace.resolve("broken")
    assert broken.value.code == "filesystem_error"

    names = excluded_names((".git", "node_modules"))
    assert is_excluded("src/.git/config", names)
    assert is_excluded("src/.GIT/config", names)
    assert not is_excluded("src/app.py", names)
    assert is_in_excluded_directory("src/.git", names, is_file=False)
    assert not is_in_excluded_directory("src/.git", names, is_file=True)
    with pytest.raises(ValueError, match="individual path names"):
        excluded_names(("..",))


@pytest.mark.skipif(sys.platform != "darwin", reason="F_GETPATH is a Darwin API")
def test_darwin_opened_paths_fit_the_fcntl_buffer(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("content", encoding="utf-8")
    file_descriptor = os.open(source, os.O_RDONLY)
    directory_descriptor = os.open(tmp_path, os.O_RDONLY)
    try:
        assert _common.opened_file_path(file_descriptor) == source.resolve()
        assert _common.opened_file_path(directory_descriptor) == tmp_path.resolve()
    finally:
        os.close(file_descriptor)
        os.close(directory_descriptor)


@pytest.mark.parametrize("pattern", ["../*.py", "", "bad\x00pattern", "src\\*.py"])
def test_invalid_glob_patterns(pattern: str) -> None:
    with pytest.raises(FilesystemFailure) as captured:
        validate_glob_pattern(pattern)
    assert captured.value.code == "invalid_glob_pattern"


def test_glob_match_semantics() -> None:
    assert glob_matches("app.py", "*.py")
    assert not glob_matches("src/app.py", "*.py")
    assert glob_matches("src/app.py", "**/*.py")
    assert glob_matches("src/deep/app.py", "**/*.py")
    assert glob_matches("a/b/c.py", "**/**/c.py")
    assert not glob_matches("a/b/c.py", "**/**/missing.py")
    assert glob_matches("src/app.py", "src/*.py")
    assert not glob_matches("other/app.py", "src/*.py")
    assert glob_filter_matches("src/app.py", "*.py")
    assert not glob_filter_matches("src/app.txt", "*.py")
    assert glob_matches("a", "**/" * 2_000 + "a")


def test_read_success_ranges_and_bom(tmp_path: Path) -> None:
    path = tmp_path / "source.py"
    raw = b"\xef\xbb\xbffirst\r\nsecond\nthird\n"
    path.write_bytes(raw)
    tool = tools.ReadTool(tmp_path, default_limit=2)
    text, result = _success(_invoke(tool, {"file_path": str(path), "offset": 2}))
    digest = sha256(raw).hexdigest()
    assert text == f"SHA-256 (raw file bytes): {digest}\n\n2: second\n3: third"
    assert result == {
        "path": "source.py",
        "content": "second\nthird",
        "sha256": digest,
        "start_line": 2,
        "end_line": 3,
        "next_offset": None,
        "truncated": False,
    }

    _, first = _success(_invoke(tool, {"file_path": "source.py", "limit": 1}))
    assert cast(dict[str, object], first)["next_offset"] == 2
    empty_text, empty = _success(_invoke(tool, {"file_path": "source.py", "offset": 99}))
    assert empty_text == (
        f"SHA-256 (raw file bytes): {digest}\n\nNo lines found in the requested range of source.py."
    )
    assert cast(dict[str, object], empty)["start_line"] is None


def test_read_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = tools.ReadTool(tmp_path, max_file_bytes=3)
    (tmp_path / "large.txt").write_text("four", encoding="utf-8")
    assert "3-byte" in _failure(_invoke(tool, {"file_path": "large.txt"}), "file_too_large")
    (tmp_path / "binary.dat").write_bytes(b"a\x00b")
    _failure(_invoke(tool, {"file_path": "binary.dat"}), "binary_file")
    (tmp_path / "invalid.txt").write_bytes(b"\xff")
    _failure(_invoke(tool, {"file_path": "invalid.txt"}), "invalid_utf8")
    _failure(_invoke(tool, {"file_path": "missing.txt"}), "path_not_found")
    _failure(_invoke(tool, {"file_path": "."}), "not_a_file")
    _failure(_invoke(tool, {"file_path": "../outside.txt"}), "path_outside_workspace")
    _failure(_invoke(tool, {"file_path": "large.txt"}, is_cancelled=lambda: True), "cancelled")

    readable = tmp_path / "readable.txt"
    readable.write_text("ok", encoding="utf-8")

    def fail_open(_path: Path) -> int:
        raise OSError("denied")

    monkeypatch.setattr(_common, "_open_readonly", fail_open)
    _failure(_invoke(tools.ReadTool(tmp_path), {"file_path": "readable.txt"}), "filesystem_error")


def test_bounded_read_revalidates_opened_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "inside.txt"
    path.write_text("inside", encoding="utf-8")
    workspace = Workspace.create(tmp_path)
    assert read_bytes_bounded(workspace, path, 100) == (b"inside", False)

    def is_not_regular(_mode: int) -> bool:
        return False

    monkeypatch.setattr(_common.stat, "S_ISREG", is_not_regular)
    with pytest.raises(FilesystemFailure) as not_regular:
        read_bytes_bounded(workspace, path, 100)
    assert not_regular.value.code == "not_a_file"
    monkeypatch.undo()

    outside = tmp_path.parent / "outside-handle.txt"
    outside.write_text("outside", encoding="utf-8")

    def outside_opened_path(_descriptor: int) -> Path:
        return outside

    monkeypatch.setattr(_common, "opened_file_path", outside_opened_path)
    with pytest.raises(FilesystemFailure) as escaped:
        read_bytes_bounded(workspace, path, 100)
    assert escaped.value.code == "path_outside_workspace"


def test_read_rejects_resolved_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = tools.ReadTool(tmp_path)
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    original_resolve = Path.resolve

    def escaped_resolve(path: Path, strict: bool = False) -> Path:
        if path.name == "link.txt":
            return outside
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", escaped_resolve)
    _failure(_invoke(tool, {"file_path": "link.txt"}), "path_outside_workspace")


def _search_fixture(root: Path) -> None:
    (root / "src" / "deep").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / ".venv").mkdir()
    (root / "alpha.py").write_text("alpha\nbeta\nalpha\n", encoding="utf-8")
    (root / "zeta.txt").write_text("alpha\n", encoding="utf-8")
    (root / "src" / "bravo.py").write_text("before\nalpha\nafter\n", encoding="utf-8")
    (root / "src" / "deep" / "charlie.py").write_text("ALPHA\n", encoding="utf-8")
    (root / ".git" / "hidden.py").write_text("alpha\n", encoding="utf-8")
    (root / ".venv" / "hidden.py").write_text("alpha\n", encoding="utf-8")


def test_glob_success_sorting_exclusions_and_limits(tmp_path: Path) -> None:
    _search_fixture(tmp_path)
    tool = tools.GlobTool(tmp_path, default_limit=2)
    text, result = _success(_invoke(tool, {"pattern": "**/*.py"}))
    assert text == "alpha.py\nsrc/bravo.py"
    assert result == {"matches": ["alpha.py", "src/bravo.py"], "truncated": True}

    _, nested = _success(_invoke(tool, {"pattern": "*.py", "path": "src", "limit": 10}))
    assert nested == {"matches": ["src/bravo.py"], "truncated": False}
    no_match_text, no_match = _success(_invoke(tool, {"pattern": "*.go"}))
    assert no_match_text == "No files matched the glob pattern."
    assert no_match == {"matches": [], "truncated": False}
    _, excluded = _success(_invoke(tool, {"pattern": "**/*.py", "path": ".git", "limit": 10}))
    assert excluded == {"matches": [], "truncated": False}


def test_exclusions_apply_to_directories_not_same_named_files(tmp_path: Path) -> None:
    (tmp_path / "venv").write_text("needle", encoding="utf-8")
    (tmp_path / ".GIT").mkdir()
    (tmp_path / ".GIT" / "hidden.txt").write_text("needle", encoding="utf-8")

    _, glob_result = _success(_invoke(tools.GlobTool(tmp_path), {"pattern": "*"}))
    assert cast(dict[str, object], glob_result)["matches"] == ["venv"]
    _, grep_result = _success(_invoke(tools.GrepTool(tmp_path), {"pattern": "needle"}))
    assert cast(dict[str, object], grep_result)["files"] == ["venv"]


def test_search_work_budgets_are_model_visible_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.txt").write_text("needle", encoding="utf-8")
    (tmp_path / "b.txt").write_text("needle", encoding="utf-8")
    _failure(
        _invoke(tools.GlobTool(tmp_path, max_scanned_entries=1), {"pattern": "*"}),
        "search_budget_exceeded",
    )
    _failure(
        _invoke(tools.GrepTool(tmp_path, max_total_bytes=1), {"pattern": "needle"}),
        "search_budget_exceeded",
    )

    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(_common, "monotonic", lambda: next(ticks, 2.0))
    _failure(
        _invoke(tools.GlobTool(tmp_path, max_search_seconds=1), {"pattern": "*"}),
        "search_timeout",
    )


def test_glob_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    tool = tools.GlobTool(tmp_path)
    _failure(_invoke(tool, {"pattern": "../*.py"}), "invalid_glob_pattern")
    limited_tool = tools.GlobTool(tmp_path, max_pattern_components=2)
    _failure(_invoke(limited_tool, {"pattern": "**/**/x"}), "invalid_glob_pattern")
    _failure(_invoke(tool, {"pattern": "*", "path": "missing"}), "path_not_found")
    _failure(_invoke(tool, {"pattern": "*", "path": "file.txt"}), "not_a_directory")
    _failure(_invoke(tool, {"pattern": "*"}, is_cancelled=lambda: True), "cancelled")

    def fail_scan(_workspace: Workspace, _directory: Path) -> None:
        raise OSError("denied")

    monkeypatch.setattr("jharness.tools.filesystem.glob.secure_scandir", fail_scan)
    _failure(_invoke(tool, {"pattern": "*"}), "filesystem_error")


def test_glob_skips_unresolvable_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "file.py").write_text("x", encoding="utf-8")

    def no_safe_match(_workspace: Workspace, _path: Path) -> None:
        return None

    monkeypatch.setattr(Workspace, "safe_match", no_safe_match)
    _, result = _success(_invoke(tools.GlobTool(tmp_path), {"pattern": "*.py"}))
    assert result == {"matches": [], "truncated": False}


def test_grep_modes_filters_context_and_limits(tmp_path: Path) -> None:
    _search_fixture(tmp_path)
    tool = tools.GrepTool(tmp_path, default_limit=10)

    text, files = _success(_invoke(tool, {"pattern": "alpha", "glob": "*.py"}))
    assert text == "alpha.py\nsrc/bravo.py"
    assert cast(dict[str, object], files)["files"] == ["alpha.py", "src/bravo.py"]

    count_text, counts = _success(
        _invoke(
            tool,
            {
                "pattern": "alpha",
                "output_mode": "count",
                "case_insensitive": True,
                "limit": 2,
            },
        )
    )
    assert count_text == "alpha.py:2\nsrc/bravo.py:1"
    assert cast(dict[str, object], counts)["truncated"] is True

    content_text, content = _success(
        _invoke(
            tool,
            {
                "pattern": "alpha",
                "path": "src/bravo.py",
                "output_mode": "content",
                "context": 1,
            },
        )
    )
    assert content_text == ("src/bravo.py-1-before\nsrc/bravo.py:2:alpha\nsrc/bravo.py-3-after")
    assert len(cast(dict[str, list[object]], content)["lines"]) == 3

    no_match_text, no_match = _success(_invoke(tool, {"pattern": "absent"}))
    assert no_match_text == "No matches found."
    assert cast(dict[str, object], no_match)["files"] == []


def test_grep_bounds_long_content_lines(tmp_path: Path) -> None:
    (tmp_path / "long.txt").write_text("prefix needle suffix", encoding="utf-8")
    tool = tools.GrepTool(tmp_path, max_line_chars=8)
    text, result = _success(_invoke(tool, {"pattern": "needle", "output_mode": "content"}))
    assert text == "long.txt:1:prefix n…"
    lines = cast(dict[str, list[dict[str, object]]], result)["lines"]
    assert lines == [
        {
            "path": "long.txt",
            "line_number": 1,
            "text": "prefix n",
            "is_match": True,
            "line_truncated": True,
        }
    ]


def test_grep_content_limit_counts_matches_not_context(tmp_path: Path) -> None:
    (tmp_path / "context.txt").write_text(
        "before\nneedle one\nafter\nneedle two\nend\n",
        encoding="utf-8",
    )
    text, result = _success(
        _invoke(
            tools.GrepTool(tmp_path),
            {"pattern": "needle", "output_mode": "content", "context": 1, "limit": 1},
        )
    )
    assert text == "context.txt-1-before\ncontext.txt:2:needle one\ncontext.txt-3-after"
    assert cast(dict[str, object], result)["truncated"] is True

    (tmp_path / "overlap.txt").write_text("needle one\nmiddle\nneedle two", encoding="utf-8")
    _, overlapping = _success(
        _invoke(
            tools.GrepTool(tmp_path),
            {
                "pattern": "needle",
                "path": "overlap.txt",
                "output_mode": "content",
                "context": 2,
            },
        )
    )
    lines = cast(dict[str, list[dict[str, object]]], overlapping)["lines"]
    assert [line["line_number"] for line in lines] == [1, 2, 3]
    assert [line["is_match"] for line in lines] == [True, False, True]


def test_grep_regex_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "app.txt").write_text("needle", encoding="utf-8")

    class TimedOutExpression:
        def search(self, _string: str, *, timeout: float) -> None:
            assert timeout == 0.25
            raise TimeoutError

    def compile_timeout(*_args: object, **_kwargs: object) -> TimedOutExpression:
        return TimedOutExpression()

    monkeypatch.setattr(regex, "compile", compile_timeout)
    tool = tools.GrepTool(tmp_path, regex_timeout_seconds=0.25)
    _failure(_invoke(tool, {"pattern": "needle"}), "regex_timeout")


def test_grep_skips_non_text_large_and_excluded_files(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "secret.txt").write_text("needle", encoding="utf-8")
    (tmp_path / "binary.dat").write_bytes(b"needle\x00")
    (tmp_path / "invalid.txt").write_bytes(b"needle\xff")
    (tmp_path / "large.txt").write_text("needle long", encoding="utf-8")
    (tmp_path / "ok.txt").write_text("needle", encoding="utf-8")
    tool = tools.GrepTool(tmp_path, max_file_bytes=10)
    _, result = _success(_invoke(tool, {"pattern": "needle", "limit": 10}))
    structured = cast(dict[str, object], result)
    assert structured["files"] == ["ok.txt"]
    assert structured["files_searched"] == 1
    assert structured["files_skipped"] == 3

    _, excluded = _success(_invoke(tool, {"pattern": "needle", "path": ".git"}))
    assert cast(dict[str, object], excluded)["files"] == []


def test_grep_failures_and_file_filters(tmp_path: Path) -> None:
    path = tmp_path / "app.py"
    path.write_text("needle", encoding="utf-8")
    tool = tools.GrepTool(tmp_path)
    _failure(_invoke(tool, {"pattern": "["}), "invalid_regex")
    _failure(_invoke(tool, {"pattern": "x", "glob": "../*.py"}), "invalid_glob_pattern")
    _failure(_invoke(tool, {"pattern": "x", "path": "missing"}), "path_not_found")
    _failure(_invoke(tool, {"pattern": "x"}, is_cancelled=lambda: True), "cancelled")
    bounded = tools.GrepTool(tmp_path, max_pattern_chars=3, max_pattern_components=1)
    _failure(_invoke(bounded, {"pattern": "four"}), "invalid_regex")
    _failure(_invoke(bounded, {"pattern": "x", "glob": "a/b"}), "invalid_glob_pattern")

    _, filtered = _success(_invoke(tool, {"pattern": "needle", "path": "app.py", "glob": "*.txt"}))
    assert cast(dict[str, object], filtered)["files"] == []
    _, matched = _success(_invoke(tool, {"pattern": "needle", "path": "app.py", "glob": "*.py"}))
    assert cast(dict[str, object], matched)["files"] == ["app.py"]


def test_grep_settles_regex_compiler_recursion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app.txt").write_text("needle", encoding="utf-8")

    def recursive_compile(*_args: object, **_kwargs: object) -> None:
        raise RecursionError

    monkeypatch.setattr(regex, "compile", recursive_compile)
    _failure(_invoke(tools.GrepTool(tmp_path), {"pattern": "needle"}), "invalid_regex")


def test_grep_path_and_enumeration_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = tools.GrepTool(tmp_path)
    original_resolve = Workspace.resolve

    def resolve_unsupported(workspace: Workspace, value: str) -> Path:
        if value == "unsupported":
            return tmp_path / "not-created"
        return original_resolve(workspace, value)

    monkeypatch.setattr(Workspace, "resolve", resolve_unsupported)
    _failure(
        _invoke(tool, {"pattern": "x", "path": "unsupported"}),
        "unsupported_path",
    )
    monkeypatch.setattr(Workspace, "resolve", original_resolve)

    def fail_scandir(_workspace: Workspace, _directory: Path) -> None:
        raise OSError("denied")

    monkeypatch.setattr("jharness.tools.filesystem.grep.secure_scandir", fail_scandir)
    _failure(_invoke(tool, {"pattern": "x"}), "filesystem_error")


def test_grep_skips_unresolvable_and_symlinked_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "app.py").write_text("needle", encoding="utf-8")
    original_is_symlink = Path.is_symlink

    def marks_source_as_symlink(path: Path) -> bool:
        return path.name == "src" or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", marks_source_as_symlink)
    _, symlinked = _success(_invoke(tools.GrepTool(tmp_path), {"pattern": "needle"}))
    assert cast(dict[str, object], symlinked)["files"] == []
    monkeypatch.setattr(Path, "is_symlink", original_is_symlink)

    (tmp_path / "root.py").write_text("needle", encoding="utf-8")

    def no_safe_match(_workspace: Workspace, _path: Path) -> None:
        return None

    monkeypatch.setattr(Workspace, "safe_match", no_safe_match)
    _, unresolved = _success(_invoke(tools.GrepTool(tmp_path), {"pattern": "needle"}))
    assert cast(dict[str, object], unresolved)["files"] == []


def test_grep_handles_read_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "app.py"
    path.write_text("needle", encoding="utf-8")

    def fail_open(_path: Path) -> int:
        raise OSError("denied")

    monkeypatch.setattr(_common, "_open_readonly", fail_open)
    _, result = _success(_invoke(tools.GrepTool(tmp_path), {"pattern": "needle"}))
    structured = cast(dict[str, object], result)
    assert structured["files_skipped"] == 1
    assert structured["files"] == []


@pytest.mark.parametrize("worker_fails", [False, True])
def test_run_blocking_settles_worker_on_cancellation(worker_fails: bool) -> None:
    started = Event()
    settled = Event()

    def worker(cancelled: Callable[[], bool]) -> str:
        started.set()
        while not cancelled():
            time.sleep(0.001)
        settled.set()
        if worker_fails:
            raise RuntimeError("worker failed")
        return "done"

    async def cancel_worker() -> None:
        task = asyncio.create_task(run_blocking(worker, lambda: False))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_worker())
    assert settled.is_set()
