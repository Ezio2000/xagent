"""The workspace-scoped Read preset."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import cast

from jharness.kernel import ToolCall, ToolContext, ToolExecution, ToolResult, ToolRisk, ToolSpec
from jharness.tools.filesystem._common import (
    FilesystemFailure,
    OperationCancelled,
    PathInput,
    Workspace,
    cancelled,
    check_cancelled,
    failure,
    nullable_output,
    positive_int,
    read_bytes_bounded,
    run_blocking,
    success,
)
from jharness.tools.filesystem._content import digest_bytes, sha256_schema

_DEFAULT_LIMIT = 200
_MAX_LIMIT = 2_000
_MAX_FILE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True, init=False)
class ReadTool:
    """Read bounded UTF-8 text from files inside one workspace."""

    workspace: Workspace
    default_limit: int
    max_limit: int
    max_file_bytes: int
    spec: ToolSpec = field(repr=False)

    def __init__(
        self,
        root: PathInput,
        *,
        default_limit: int = _DEFAULT_LIMIT,
        max_limit: int = _MAX_LIMIT,
        max_file_bytes: int = _MAX_FILE_BYTES,
    ) -> None:
        default_limit = positive_int(default_limit, "default_limit")
        max_limit = positive_int(max_limit, "max_limit")
        max_file_bytes = positive_int(max_file_bytes, "max_file_bytes")
        if default_limit > max_limit:
            raise ValueError("default_limit cannot exceed max_limit")
        object.__setattr__(self, "workspace", Workspace.create(root))
        object.__setattr__(self, "default_limit", default_limit)
        object.__setattr__(self, "max_limit", max_limit)
        object.__setattr__(self, "max_file_bytes", max_file_bytes)
        object.__setattr__(self, "spec", _spec(default_limit, max_limit))

    @property
    def root(self) -> Path:
        return self.workspace.root

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        file_path = cast(str, call.arguments["file_path"])
        offset = cast(int, call.arguments.get("offset", 1))
        limit = cast(int, call.arguments.get("limit", self.default_limit))
        try:
            return await run_blocking(
                lambda cancelled_check: self._read(file_path, offset, limit, cancelled_check),
                lambda: context.cancel_requested,
            )
        except OperationCancelled:
            return cancelled("Read")
        except FilesystemFailure as exc:
            return failure(exc)

    def _read(
        self,
        file_path: str,
        offset: int,
        limit: int,
        cancelled_check: Callable[[], bool],
    ) -> ToolResult:
        check_cancelled(cancelled_check)
        path = self.workspace.file(file_path)
        try:
            data, too_large = read_bytes_bounded(
                self.workspace,
                path,
                self.max_file_bytes,
                lambda: check_cancelled(cancelled_check),
            )
            if too_large:
                raise FilesystemFailure(
                    "file_too_large",
                    f"File exceeds the configured {self.max_file_bytes}-byte limit: {file_path}",
                )
        except FilesystemFailure:
            raise
        except OSError as exc:
            raise FilesystemFailure("filesystem_error", f"Cannot read file: {file_path}") from exc
        check_cancelled(cancelled_check)
        if b"\x00" in data:
            raise FilesystemFailure("binary_file", f"File appears to be binary: {file_path}")
        try:
            content = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise FilesystemFailure(
                "invalid_utf8",
                f"File is not valid UTF-8 text: {file_path}",
            ) from exc
        selected: list[str] = []
        truncated = False
        with StringIO(content, newline=None) as stream:
            for line_number, line in enumerate(stream, 1):
                check_cancelled(cancelled_check)
                if line_number < offset:
                    continue
                if len(selected) >= limit:
                    truncated = True
                    break
                selected.append(line[:-1] if line.endswith("\n") else line)
        display_path = self.workspace.display(path)
        digest = digest_bytes(data)
        body = "\n".join(f"{number}: {line}" for number, line in enumerate(selected, offset))
        if not body:
            body = f"No lines found in the requested range of {display_path}."
        text = f"SHA-256 (raw file bytes): {digest}\n\n{body}"
        end_line = offset + len(selected) - 1 if selected else None
        return success(
            text,
            {
                "path": display_path,
                "content": "\n".join(selected),
                "sha256": digest,
                "start_line": offset if selected else None,
                "end_line": end_line,
                "next_offset": end_line + 1 if truncated and end_line is not None else None,
                "truncated": truncated,
            },
        )


def _spec(default_limit: int, max_limit: int) -> ToolSpec:
    output = {
        "type": "object",
        "required": [
            "path",
            "content",
            "sha256",
            "start_line",
            "end_line",
            "next_offset",
            "truncated",
        ],
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "sha256": sha256_schema(),
            "start_line": {"type": ["integer", "null"], "minimum": 1},
            "end_line": {"type": ["integer", "null"], "minimum": 1},
            "next_offset": {"type": ["integer", "null"], "minimum": 1},
            "truncated": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    return ToolSpec(
        name="Read",
        description=(
            "Read a UTF-8 text file within the configured workspace. "
            "The model-visible result begins with the SHA-256 of the complete raw file bytes. "
            f"Lines are one-based; limit defaults to {default_limit}."
        ),
        input_schema={
            "type": "object",
            "required": ["file_path"],
            "properties": {
                "file_path": {
                    "type": "string",
                    "minLength": 1,
                    "description": "File path inside the Host-configured workspace.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "One-based line number at which reading starts.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_limit,
                    "default": default_limit,
                    "description": "Maximum number of lines to return.",
                },
            },
            "additionalProperties": False,
        },
        output_schema=nullable_output(output),
        execution=ToolExecution(concurrency="parallel", read_only=True, idempotent=True),
        risk=ToolRisk(filesystem="read", destructive=False),
    )
