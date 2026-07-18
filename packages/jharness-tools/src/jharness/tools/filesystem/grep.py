"""The workspace-scoped Grep preset."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from heapq import heappop, heappush
from io import StringIO
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, TypedDict, cast

import regex

from jharness.kernel import ToolCall, ToolContext, ToolExecution, ToolResult, ToolRisk, ToolSpec
from jharness.tools.filesystem._common import (
    DEFAULT_EXCLUDED_DIRECTORY_NAMES,
    FilesystemFailure,
    OperationCancelled,
    PathInput,
    SearchBudget,
    Workspace,
    cancelled,
    excluded_names,
    failure,
    glob_filter_matches,
    is_in_excluded_directory,
    nullable_output,
    positive_float,
    positive_int,
    read_bytes_bounded,
    run_blocking,
    secure_scandir,
    success,
    validate_glob_pattern,
)

OutputMode: TypeAlias = Literal["content", "files_with_matches", "count"]


class ContentEntry(TypedDict):
    path: str
    line_number: int
    text: str
    is_match: bool
    line_truncated: bool


class CountEntry(TypedDict):
    path: str
    count: int


GrepEntry: TypeAlias = str | ContentEntry | CountEntry

_DEFAULT_LIMIT = 100
_MAX_LIMIT = 1_000
_MAX_CONTEXT = 10
_MAX_FILE_BYTES = 4 * 1024 * 1024
_MAX_LINE_CHARS = 2_000
_REGEX_TIMEOUT_SECONDS = 0.1
_MAX_PATTERN_CHARS = 4_096
_MAX_PATTERN_COMPONENTS = 256
_MAX_SEARCH_SECONDS = 10.0
_MAX_SCANNED_ENTRIES = 100_000
_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_OUTPUT_BYTES = 4 * 1024 * 1024


class _SearchExpression(Protocol):
    def search(self, string: str, *, timeout: float) -> object | None: ...


@dataclass(frozen=True, slots=True, init=False)
class GrepTool:
    """Search UTF-8 text files inside one workspace."""

    workspace: Workspace
    default_limit: int
    max_limit: int
    max_context: int
    max_file_bytes: int
    max_line_chars: int
    regex_timeout_seconds: float
    max_pattern_chars: int
    max_pattern_components: int
    max_search_seconds: float
    max_scanned_entries: int
    max_total_bytes: int
    max_output_bytes: int
    excluded_directory_names: frozenset[str]
    spec: ToolSpec = field(repr=False)

    def __init__(
        self,
        root: PathInput,
        *,
        default_limit: int = _DEFAULT_LIMIT,
        max_limit: int = _MAX_LIMIT,
        max_context: int = _MAX_CONTEXT,
        max_file_bytes: int = _MAX_FILE_BYTES,
        max_line_chars: int = _MAX_LINE_CHARS,
        regex_timeout_seconds: float = _REGEX_TIMEOUT_SECONDS,
        max_pattern_chars: int = _MAX_PATTERN_CHARS,
        max_pattern_components: int = _MAX_PATTERN_COMPONENTS,
        max_search_seconds: float = _MAX_SEARCH_SECONDS,
        max_scanned_entries: int = _MAX_SCANNED_ENTRIES,
        max_total_bytes: int = _MAX_TOTAL_BYTES,
        max_output_bytes: int = _MAX_OUTPUT_BYTES,
        excluded_directory_names: Iterable[str] = DEFAULT_EXCLUDED_DIRECTORY_NAMES,
    ) -> None:
        default_limit = positive_int(default_limit, "default_limit")
        max_limit = positive_int(max_limit, "max_limit")
        max_context = positive_int(max_context, "max_context")
        max_file_bytes = positive_int(max_file_bytes, "max_file_bytes")
        max_line_chars = positive_int(max_line_chars, "max_line_chars")
        regex_timeout_seconds = positive_float(regex_timeout_seconds, "regex_timeout_seconds")
        max_pattern_chars = positive_int(max_pattern_chars, "max_pattern_chars")
        max_pattern_components = positive_int(max_pattern_components, "max_pattern_components")
        max_search_seconds = positive_float(max_search_seconds, "max_search_seconds")
        max_scanned_entries = positive_int(max_scanned_entries, "max_scanned_entries")
        max_total_bytes = positive_int(max_total_bytes, "max_total_bytes")
        max_output_bytes = positive_int(max_output_bytes, "max_output_bytes")
        if default_limit > max_limit:
            raise ValueError("default_limit cannot exceed max_limit")
        object.__setattr__(self, "workspace", Workspace.create(root))
        object.__setattr__(self, "default_limit", default_limit)
        object.__setattr__(self, "max_limit", max_limit)
        object.__setattr__(self, "max_context", max_context)
        object.__setattr__(self, "max_file_bytes", max_file_bytes)
        object.__setattr__(self, "max_line_chars", max_line_chars)
        object.__setattr__(self, "regex_timeout_seconds", regex_timeout_seconds)
        object.__setattr__(self, "max_pattern_chars", max_pattern_chars)
        object.__setattr__(self, "max_pattern_components", max_pattern_components)
        object.__setattr__(self, "max_search_seconds", max_search_seconds)
        object.__setattr__(self, "max_scanned_entries", max_scanned_entries)
        object.__setattr__(self, "max_total_bytes", max_total_bytes)
        object.__setattr__(self, "max_output_bytes", max_output_bytes)
        object.__setattr__(
            self,
            "excluded_directory_names",
            excluded_names(excluded_directory_names),
        )
        object.__setattr__(
            self,
            "spec",
            _spec(default_limit, max_limit, max_context, max_pattern_chars),
        )

    @property
    def root(self) -> Path:
        return self.workspace.root

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        pattern = cast(str, call.arguments["pattern"])
        path = cast(str, call.arguments.get("path", "."))
        glob = cast(str | None, call.arguments.get("glob"))
        mode = cast(OutputMode, call.arguments.get("output_mode", "files_with_matches"))
        case_insensitive = cast(bool, call.arguments.get("case_insensitive", False))
        context_lines = cast(int, call.arguments.get("context", 0))
        limit = cast(int, call.arguments.get("limit", self.default_limit))
        try:
            return await run_blocking(
                lambda cancelled_check: self._grep(
                    pattern,
                    path,
                    glob,
                    mode,
                    case_insensitive,
                    context_lines,
                    limit,
                    cancelled_check,
                ),
                lambda: context.cancel_requested,
            )
        except OperationCancelled:
            return cancelled("Grep")
        except FilesystemFailure as exc:
            return failure(exc)

    def _grep(
        self,
        pattern: str,
        path_value: str,
        glob: str | None,
        mode: OutputMode,
        case_insensitive: bool,
        context_lines: int,
        limit: int,
        cancelled_check: Callable[[], bool],
    ) -> ToolResult:
        budget = SearchBudget.create(
            cancelled_check,
            self.max_search_seconds,
            self.max_scanned_entries,
            self.max_total_bytes,
        )
        budget.checkpoint()
        expression = self._prepare_expression(pattern, glob, case_insensitive)
        target = self.workspace.resolve(path_value)
        if not target.is_file() and not target.is_dir():
            message = f"Path is not a file or directory: {path_value}"
            raise FilesystemFailure("unsupported_path", message)

        entries: list[GrepEntry] = []
        files_searched = 0
        files_skipped = 0
        matched_lines = 0
        truncated = False
        for file_path, display_path in self._files(target, glob, budget):
            budget.checkpoint()
            content = self._text_content(file_path, budget)
            if content is None:
                files_skipped += 1
                continue
            files_searched += 1
            if mode == "content":
                file_entries, file_matches, has_more = self._content_entries(
                    expression,
                    display_path,
                    content,
                    context_lines,
                    limit - matched_lines,
                    budget,
                )
                entries.extend(file_entries)
                matched_lines += file_matches
                if has_more:
                    truncated = True
                    break
                continue

            only_detect = len(entries) >= limit
            match_count = self._match_count(
                expression,
                content,
                budget,
                stop_after_first=mode == "files_with_matches" or only_detect,
            )
            if match_count == 0:
                continue
            if only_detect:
                truncated = True
                break
            if mode == "count":
                entries.append(CountEntry(path=display_path, count=match_count))
            else:
                entries.append(display_path)

        return self._bounded_result(
            mode,
            entries,
            truncated,
            files_searched,
            files_skipped,
        )

    def _bounded_result(
        self,
        mode: OutputMode,
        entries: list[GrepEntry],
        truncated: bool,
        files_searched: int,
        files_skipped: int,
    ) -> ToolResult:
        while True:
            structured = self._structured(
                mode,
                entries,
                truncated,
                files_searched,
                files_skipped,
            )
            text = self._text(mode, entries)
            structured_json = json.dumps(
                structured,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            output_bytes = len(text.encode("utf-8")) + len(structured_json.encode("utf-8"))
            if output_bytes <= self.max_output_bytes:
                return success(text, structured)
            if not entries:
                raise FilesystemFailure(
                    "output_budget_exceeded",
                    "Grep output metadata exceeds the configured byte limit.",
                )
            entries.pop()
            truncated = True

    def _prepare_expression(
        self,
        pattern: str,
        glob: str | None,
        case_insensitive: bool,
    ) -> _SearchExpression:
        if glob is not None:
            validate_glob_pattern(glob, self.max_pattern_chars, self.max_pattern_components)
        if len(pattern) > self.max_pattern_chars:
            raise FilesystemFailure(
                "invalid_regex",
                "Regular expression exceeds the configured length limit.",
            )
        return self._compile_expression(pattern, case_insensitive)

    @staticmethod
    def _compile_expression(pattern: str, case_insensitive: bool) -> _SearchExpression:
        try:
            return regex.compile(pattern, regex.IGNORECASE if case_insensitive else 0)
        except (MemoryError, RecursionError, regex.error) as exc:
            raise FilesystemFailure("invalid_regex", f"Invalid regular expression: {exc}") from exc

    def _match_count(
        self,
        expression: _SearchExpression,
        content: str,
        budget: SearchBudget,
        *,
        stop_after_first: bool,
    ) -> int:
        count = 0
        for _, line in self._iter_lines(content):
            budget.checkpoint()
            if not self._line_matches(expression, line, budget):
                continue
            count += 1
            if stop_after_first:
                break
        return count

    def _content_entries(
        self,
        expression: _SearchExpression,
        display_path: str,
        content: str,
        context_lines: int,
        remaining_matches: int,
        budget: SearchBudget,
    ) -> tuple[list[GrepEntry], int, bool]:
        previous: deque[tuple[int, str]] = deque(maxlen=context_lines)
        emitted: set[int] = set()
        entries: list[GrepEntry] = []
        matches = 0
        following = 0
        for line_number, line in self._iter_lines(content):
            budget.checkpoint()
            is_match = self._line_matches(expression, line, budget)
            if is_match:
                if matches >= remaining_matches:
                    return entries, matches, True
                matches += 1
                for previous_number, previous_line in previous:
                    self._append_content_entry(
                        entries,
                        emitted,
                        display_path,
                        previous_number,
                        previous_line,
                        is_match=False,
                    )
                self._append_content_entry(
                    entries,
                    emitted,
                    display_path,
                    line_number,
                    line,
                    is_match=True,
                )
                following = context_lines
            elif following:
                self._append_content_entry(
                    entries,
                    emitted,
                    display_path,
                    line_number,
                    line,
                    is_match=False,
                )
                following -= 1
            previous.append((line_number, line))
        return entries, matches, False

    def _append_content_entry(
        self,
        entries: list[GrepEntry],
        emitted: set[int],
        display_path: str,
        line_number: int,
        line: str,
        *,
        is_match: bool,
    ) -> None:
        if line_number in emitted:
            return
        emitted.add(line_number)
        entries.append(
            ContentEntry(
                path=display_path,
                line_number=line_number,
                text=line[: self.max_line_chars],
                is_match=is_match,
                line_truncated=len(line) > self.max_line_chars,
            )
        )

    def _line_matches(
        self,
        expression: _SearchExpression,
        line: str,
        budget: SearchBudget,
    ) -> bool:
        try:
            timeout = budget.operation_timeout(self.regex_timeout_seconds)
            return expression.search(line, timeout=timeout) is not None
        except TimeoutError as exc:
            budget.checkpoint()
            raise FilesystemFailure(
                "regex_timeout",
                "Regular expression search exceeded the configured time limit.",
            ) from exc

    @staticmethod
    def _iter_lines(content: str) -> Iterator[tuple[int, str]]:
        with StringIO(content, newline=None) as stream:
            for line_number, line in enumerate(stream, 1):
                yield line_number, line[:-1] if line.endswith("\n") else line

    def _files(
        self,
        target: Path,
        glob: str | None,
        budget: SearchBudget,
    ) -> Iterator[tuple[Path, str]]:
        target_display = self.workspace.display(target)
        if is_in_excluded_directory(
            target_display,
            self.excluded_directory_names,
            is_file=target.is_file(),
        ):
            return
        if target.is_file():
            if glob is None or glob_filter_matches(target_display, glob):
                yield target, target_display
            return
        pending: list[tuple[str, str, Path]] = []
        self._push_directory_entries(target, pending, budget)
        while pending:
            budget.checkpoint()
            _, _, candidate = heappop(pending)
            if candidate.is_dir() and not candidate.is_symlink():
                self._push_directory_entries(candidate, pending, budget)
                continue
            match = self.workspace.safe_match(candidate)
            if match is None:
                continue
            resolved, display_path = match
            if glob is None or glob_filter_matches(display_path, glob):
                yield resolved, display_path

    def _push_directory_entries(
        self,
        directory: Path,
        pending: list[tuple[str, str, Path]],
        budget: SearchBudget,
    ) -> None:
        try:
            with secure_scandir(self.workspace, directory) as entries:
                for entry in entries:
                    budget.consume_entry()
                    path = directory / entry.name
                    display_path = self.workspace.display(path)
                    is_directory = entry.is_dir(follow_symlinks=False)
                    if is_directory and entry.name.casefold() in self.excluded_directory_names:
                        continue
                    sort_path = f"{display_path}/" if is_directory else display_path
                    heappush(pending, (sort_path.casefold(), sort_path, path))
        except OSError as exc:
            raise FilesystemFailure(
                "filesystem_error",
                f"Cannot search directory: {self.workspace.display(directory)}",
            ) from exc

    def _text_content(self, path: Path, budget: SearchBudget) -> str | None:
        try:
            read_limit = budget.remaining_byte_limit(self.max_file_bytes)
            data, too_large = read_bytes_bounded(
                self.workspace,
                path,
                read_limit,
                budget.checkpoint,
            )
            budget.consume_bytes(len(data))
            if too_large:
                return None
        except OSError:
            return None
        if b"\x00" in data:
            return None
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            return None

    @staticmethod
    def _structured(
        mode: OutputMode,
        entries: list[GrepEntry],
        truncated: bool,
        files_searched: int,
        files_skipped: int,
    ) -> dict[str, object]:
        common = {
            "output_mode": mode,
            "truncated": truncated,
            "files_searched": files_searched,
            "files_skipped": files_skipped,
        }
        if mode == "content":
            return {**common, "lines": entries}
        if mode == "count":
            return {**common, "counts": entries}
        return {**common, "files": entries}

    @staticmethod
    def _text(mode: OutputMode, entries: list[GrepEntry]) -> str:
        if not entries:
            return "No matches found."
        if mode == "content":
            content_entries = cast(list[ContentEntry], entries)
            return "\n".join(
                f"{entry['path']}:{entry['line_number']}:{entry['text']}"
                f"{'…' if entry['line_truncated'] else ''}"
                if entry["is_match"]
                else f"{entry['path']}-{entry['line_number']}-{entry['text']}"
                f"{'…' if entry['line_truncated'] else ''}"
                for entry in content_entries
            )
        if mode == "count":
            count_entries = cast(list[CountEntry], entries)
            return "\n".join(f"{entry['path']}:{entry['count']}" for entry in count_entries)
        return "\n".join(cast(list[str], entries))


def _spec(
    default_limit: int,
    max_limit: int,
    max_context: int,
    max_pattern_chars: int,
) -> ToolSpec:
    common_properties = {
        "output_mode": {"type": "string"},
        "truncated": {"type": "boolean"},
        "files_searched": {"type": "integer", "minimum": 0},
        "files_skipped": {"type": "integer", "minimum": 0},
    }
    common_required = ["output_mode", "truncated", "files_searched", "files_skipped"]
    output = {
        "oneOf": [
            {
                "type": "object",
                "required": [*common_required, "lines"],
                "properties": {
                    **common_properties,
                    "output_mode": {"const": "content"},
                    "lines": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "path",
                                "line_number",
                                "text",
                                "is_match",
                                "line_truncated",
                            ],
                            "properties": {
                                "path": {"type": "string"},
                                "line_number": {"type": "integer", "minimum": 1},
                                "text": {"type": "string"},
                                "is_match": {"type": "boolean"},
                                "line_truncated": {"type": "boolean"},
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
            {
                "type": "object",
                "required": [*common_required, "files"],
                "properties": {
                    **common_properties,
                    "output_mode": {"const": "files_with_matches"},
                    "files": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
            {
                "type": "object",
                "required": [*common_required, "counts"],
                "properties": {
                    **common_properties,
                    "output_mode": {"const": "count"},
                    "counts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["path", "count"],
                            "properties": {
                                "path": {"type": "string"},
                                "count": {"type": "integer", "minimum": 1},
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
        ]
    }
    return ToolSpec(
        name="Grep",
        description=(
            "Search UTF-8 text files with a regular expression inside the configured workspace. "
            f"Output limit defaults to {default_limit}."
        ),
        input_schema={
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": max_pattern_chars,
                    "description": "Single-line regular expression to search for.",
                },
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "default": ".",
                    "description": "File or directory inside the workspace to search.",
                },
                "glob": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": max_pattern_chars,
                    "description": "Optional file glob; basename patterns match at any depth.",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "default": "files_with_matches",
                    "description": (
                        "Return matching lines, matching paths, or per-file line counts."
                    ),
                },
                "case_insensitive": {
                    "type": "boolean",
                    "default": False,
                    "description": "Match without case sensitivity when true.",
                },
                "context": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": max_context,
                    "default": 0,
                    "description": "Context lines around matches in content mode.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_limit,
                    "default": default_limit,
                    "description": "Maximum matching lines (content) or files (other modes).",
                },
            },
            "additionalProperties": False,
        },
        output_schema=nullable_output(output),
        execution=ToolExecution(concurrency="parallel", read_only=True, idempotent=True),
        risk=ToolRisk(filesystem="read", destructive=False),
    )
