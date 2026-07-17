"""The workspace-scoped Edit preset."""

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
    apply_text_edit,
    atomic_write,
    mutation_session,
    read_text_snapshot,
    run_mutation,
)

_MAX_FILE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True, init=False)
class EditTool:
    """Replace exact text in one existing UTF-8 workspace file."""

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
        old_string = cast(str, call.arguments["old_string"])
        new_string = cast(str, call.arguments["new_string"])
        replace_all = cast(bool, call.arguments.get("replace_all", False))
        expected_sha256 = cast(str, call.arguments["expected_sha256"])
        try:
            return await run_mutation(
                lambda cancelled_check: self._edit(
                    file_path,
                    old_string,
                    new_string,
                    replace_all,
                    expected_sha256,
                    cancelled_check,
                ),
                lambda: context.cancel_requested,
            )
        except OperationCancelled:
            return cancelled("Edit")
        except FilesystemFailure as exc:
            return failure(exc)
        except OSError:
            return failure(FilesystemFailure("filesystem_error", f"Cannot edit file: {file_path}"))

    def _edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,
        expected_sha256: str,
        cancelled_check: Callable[[], bool],
    ) -> ToolResult:
        with mutation_session(self.workspace, file_path, cancelled_check) as session:
            snapshot = read_text_snapshot(
                session,
                self.max_file_bytes,
                cancelled_check,
            )
            if snapshot.digest != expected_sha256:
                raise FilesystemFailure(
                    "stale_file",
                    f"File changed since it was read: {session.display}",
                )
            encoded, replacements = apply_text_edit(
                snapshot,
                old_string,
                new_string,
                replace_all=replace_all,
                max_file_bytes=self.max_file_bytes,
                cancelled=cancelled_check,
            )
            atomic_write(
                session,
                encoded,
                expected_sha256=snapshot.digest,
                expected_identity=snapshot.identity,
                max_file_bytes=self.max_file_bytes,
                cancelled=cancelled_check,
            )
        digest = digest_bytes(encoded)
        display_path = session.display
        noun = "replacement" if replacements == 1 else "replacements"
        return success(
            f"Updated {display_path} with {replacements} {noun}.",
            {
                "path": display_path,
                "replacements": replacements,
                "previous_sha256": snapshot.digest,
                "sha256": digest,
                "bytes_written": len(encoded),
            },
        )


def _spec(max_file_bytes: int) -> ToolSpec:
    output = {
        "type": "object",
        "required": ["path", "replacements", "previous_sha256", "sha256", "bytes_written"],
        "properties": {
            "path": {"type": "string"},
            "replacements": {"type": "integer", "minimum": 1},
            "previous_sha256": sha256_schema(),
            "sha256": sha256_schema(),
            "bytes_written": {"type": "integer", "minimum": 0},
        },
        "additionalProperties": False,
    }
    return ToolSpec(
        name="Edit",
        description=(
            "Replace exact text in an existing UTF-8 file within the configured workspace. "
            "Use the sha256 returned by Read and provide enough old_string context to be unique."
        ),
        input_schema={
            "type": "object",
            "required": [
                "file_path",
                "old_string",
                "new_string",
                "expected_sha256",
            ],
            "properties": {
                "file_path": {"type": "string", "minLength": 1},
                "old_string": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": max_file_bytes,
                },
                "new_string": {"type": "string", "maxLength": max_file_bytes},
                "replace_all": {"type": "boolean", "default": False},
                "expected_sha256": sha256_schema(),
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
