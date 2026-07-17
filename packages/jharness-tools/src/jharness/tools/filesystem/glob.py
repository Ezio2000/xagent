"""The workspace-scoped Glob preset."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

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
    glob_matches,
    is_excluded,
    nullable_output,
    positive_float,
    positive_int,
    run_blocking,
    secure_scandir,
    success,
    validate_glob_pattern,
)

_DEFAULT_LIMIT = 100
_MAX_LIMIT = 1_000
_MAX_PATTERN_CHARS = 4_096
_MAX_PATTERN_COMPONENTS = 256
_MAX_SEARCH_SECONDS = 10.0
_MAX_SCANNED_ENTRIES = 100_000


@dataclass(frozen=True, slots=True, init=False)
class GlobTool:
    """Find files by glob pattern inside one workspace."""

    workspace: Workspace
    default_limit: int
    max_limit: int
    max_pattern_chars: int
    max_pattern_components: int
    max_search_seconds: float
    max_scanned_entries: int
    excluded_directory_names: frozenset[str]
    spec: ToolSpec = field(repr=False)

    def __init__(
        self,
        root: PathInput,
        *,
        default_limit: int = _DEFAULT_LIMIT,
        max_limit: int = _MAX_LIMIT,
        max_pattern_chars: int = _MAX_PATTERN_CHARS,
        max_pattern_components: int = _MAX_PATTERN_COMPONENTS,
        max_search_seconds: float = _MAX_SEARCH_SECONDS,
        max_scanned_entries: int = _MAX_SCANNED_ENTRIES,
        excluded_directory_names: Iterable[str] = DEFAULT_EXCLUDED_DIRECTORY_NAMES,
    ) -> None:
        default_limit = positive_int(default_limit, "default_limit")
        max_limit = positive_int(max_limit, "max_limit")
        max_pattern_chars = positive_int(max_pattern_chars, "max_pattern_chars")
        max_pattern_components = positive_int(max_pattern_components, "max_pattern_components")
        max_search_seconds = positive_float(max_search_seconds, "max_search_seconds")
        max_scanned_entries = positive_int(max_scanned_entries, "max_scanned_entries")
        if default_limit > max_limit:
            raise ValueError("default_limit cannot exceed max_limit")
        object.__setattr__(self, "workspace", Workspace.create(root))
        object.__setattr__(self, "default_limit", default_limit)
        object.__setattr__(self, "max_limit", max_limit)
        object.__setattr__(self, "max_pattern_chars", max_pattern_chars)
        object.__setattr__(self, "max_pattern_components", max_pattern_components)
        object.__setattr__(self, "max_search_seconds", max_search_seconds)
        object.__setattr__(self, "max_scanned_entries", max_scanned_entries)
        object.__setattr__(
            self,
            "excluded_directory_names",
            excluded_names(excluded_directory_names),
        )
        object.__setattr__(self, "spec", _spec(default_limit, max_limit, max_pattern_chars))

    @property
    def root(self) -> Path:
        return self.workspace.root

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        pattern = cast(str, call.arguments["pattern"])
        base_path = cast(str, call.arguments.get("path", "."))
        limit = cast(int, call.arguments.get("limit", self.default_limit))
        try:
            return await run_blocking(
                lambda cancelled_check: self._glob(pattern, base_path, limit, cancelled_check),
                lambda: context.cancel_requested,
            )
        except OperationCancelled:
            return cancelled("Glob")
        except FilesystemFailure as exc:
            return failure(exc)

    def _glob(
        self,
        pattern: str,
        base_path: str,
        limit: int,
        cancelled_check: Callable[[], bool],
    ) -> ToolResult:
        budget = SearchBudget.create(
            cancelled_check,
            self.max_search_seconds,
            self.max_scanned_entries,
        )
        budget.checkpoint()
        validate_glob_pattern(pattern, self.max_pattern_chars, self.max_pattern_components)
        directory = self.workspace.directory(base_path)
        try:
            smallest = heapq.nsmallest(
                limit + 1,
                self._matches(directory, pattern, budget),
                key=lambda value: (value.casefold(), value),
            )
        except OSError as exc:
            raise FilesystemFailure(
                "filesystem_error",
                f"Cannot search directory: {base_path}",
            ) from exc
        truncated = len(smallest) > limit
        matches = smallest[:limit]
        text = "\n".join(matches) if matches else "No files matched the glob pattern."
        return success(text, {"matches": matches, "truncated": truncated})

    def _matches(
        self,
        directory: Path,
        pattern: str,
        budget: SearchBudget,
    ) -> Iterator[str]:
        if is_excluded(self.workspace.display(directory), self.excluded_directory_names):
            return
        pending = [directory]
        while pending:
            budget.checkpoint()
            current = pending.pop()
            with secure_scandir(self.workspace, current) as entries:
                for entry in entries:
                    budget.consume_entry()
                    candidate = current / entry.name
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name.casefold() not in self.excluded_directory_names:
                            pending.append(candidate)
                        continue
                    match = self.workspace.safe_match(candidate)
                    if match is None:
                        continue
                    _, display_path = match
                    search_path = candidate.relative_to(directory).as_posix()
                    if glob_matches(search_path, pattern):
                        yield display_path


def _spec(default_limit: int, max_limit: int, max_pattern_chars: int) -> ToolSpec:
    output = {
        "type": "object",
        "required": ["matches", "truncated"],
        "properties": {
            "matches": {"type": "array", "items": {"type": "string"}},
            "truncated": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    return ToolSpec(
        name="Glob",
        description=(
            "Find files by a relative glob pattern within the configured workspace. "
            f"Results are sorted and limit defaults to {default_limit}."
        ),
        input_schema={
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": max_pattern_chars,
                    "description": "Relative file glob using '/' separators.",
                },
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "default": ".",
                    "description": "Directory inside the workspace to search from.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_limit,
                    "default": default_limit,
                    "description": "Maximum number of sorted file paths to return.",
                },
            },
            "additionalProperties": False,
        },
        output_schema=nullable_output(output),
        execution=ToolExecution(concurrency="parallel", read_only=True, idempotent=True),
        risk=ToolRisk(filesystem="read", destructive=False),
    )
