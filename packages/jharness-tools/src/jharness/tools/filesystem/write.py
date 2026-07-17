"""The workspace-scoped Write preset."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from jharness.kernel import ToolCall, ToolContext, ToolExecution, ToolResult, ToolRisk, ToolSpec
from jharness.tools.filesystem._common import (
    FilesystemFailure,
    OperationCancelled,
    PathInput,
    Workspace,
    cancelled,
    failure,
    nullable_output,
    positive_int,
    success,
)
from jharness.tools.filesystem._content import digest_bytes, sha256_schema
from jharness.tools.filesystem._write_io import (
    atomic_write,
    encode_text_content,
    mutation_session,
    run_mutation,
)

_MAX_FILE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True, init=False)
class WriteTool:
    """Create or CAS-overwrite one UTF-8 workspace file."""

    workspace: Workspace
    max_file_bytes: int
    spec: ToolSpec = field(repr=False)

    def __init__(self, root: PathInput, *, max_file_bytes: int = _MAX_FILE_BYTES) -> None:
        max_file_bytes = positive_int(max_file_bytes, "max_file_bytes")
        object.__setattr__(self, "workspace", Workspace.create(root))
        object.__setattr__(self, "max_file_bytes", max_file_bytes)
        object.__setattr__(self, "spec", _spec(max_file_bytes))

    @property
    def root(self) -> Path:
        return self.workspace.root

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        file_path = cast(str, call.arguments["file_path"])
        content = cast(str, call.arguments["content"])
        expected_sha256 = cast(str | None, call.arguments["expected_sha256"])
        try:
            return await run_mutation(
                lambda cancelled_check: self._write(
                    file_path,
                    content,
                    expected_sha256,
                    cancelled_check,
                ),
                lambda: context.cancel_requested,
            )
        except OperationCancelled:
            return cancelled("Write")
        except FilesystemFailure as exc:
            return failure(exc)
        except OSError:
            return failure(FilesystemFailure("filesystem_error", f"Cannot write file: {file_path}"))

    def _write(
        self,
        file_path: str,
        content: str,
        expected_sha256: str | None,
        cancelled_check: Callable[[], bool],
    ) -> ToolResult:
        encoded = encode_text_content(content, max_file_bytes=self.max_file_bytes)
        with mutation_session(self.workspace, file_path, cancelled_check) as session:
            previous = atomic_write(
                session,
                encoded,
                expected_sha256=expected_sha256,
                max_file_bytes=self.max_file_bytes,
                cancelled=cancelled_check,
            )
        digest = digest_bytes(encoded)
        display_path = session.display
        operation = "overwritten" if previous is not None else "created"
        return success(
            f"{operation.capitalize()} {display_path} ({len(encoded)} bytes).",
            {
                "path": display_path,
                "operation": operation,
                "previous_sha256": previous,
                "sha256": digest,
                "bytes_written": len(encoded),
            },
        )


def _spec(max_file_bytes: int) -> ToolSpec:
    output = {
        "type": "object",
        "required": ["path", "operation", "previous_sha256", "sha256", "bytes_written"],
        "properties": {
            "path": {"type": "string"},
            "operation": {"enum": ["created", "overwritten"]},
            "previous_sha256": {"anyOf": [sha256_schema(), {"type": "null"}]},
            "sha256": sha256_schema(),
            "bytes_written": {"type": "integer", "minimum": 0},
        },
        "additionalProperties": False,
    }
    return ToolSpec(
        name="Write",
        description=(
            "Create or completely replace a UTF-8 file within the configured workspace. "
            "Set expected_sha256 to null to require a new path, or to the sha256 returned by "
            "Read to conditionally overwrite an existing file. Parent directories must exist."
        ),
        input_schema={
            "type": "object",
            "required": ["file_path", "content", "expected_sha256"],
            "properties": {
                "file_path": {"type": "string", "minLength": 1},
                "content": {"type": "string", "maxLength": max_file_bytes},
                "expected_sha256": {
                    "anyOf": [sha256_schema(), {"type": "null"}],
                },
            },
            "additionalProperties": False,
        },
        output_schema=nullable_output(output),
        execution=ToolExecution(concurrency="serial", read_only=False, idempotent=False),
        risk=ToolRisk(
            filesystem="write",
            destructive=True,
            requires_approval=True,
        ),
    )
