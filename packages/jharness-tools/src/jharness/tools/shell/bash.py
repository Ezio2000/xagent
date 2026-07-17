"""The workspace-rooted Bash preset."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import cast

from jharness.kernel import (
    ContentPart,
    SettledResult,
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolFailure,
    ToolResult,
    ToolRisk,
    ToolSpec,
    ToolSuccess,
)
from jharness.tools.filesystem._common import (
    FilesystemFailure,
    Workspace,
    nullable_output,
    positive_float,
    positive_int,
)
from jharness.tools.shell._runner import (
    CommandCancelled,
    CommandExecutionFailed,
    CommandOutcome,
    CommandSpawnFailed,
    run_bash,
)

PathInput = str | PathLike[str]

_MAX_COMMAND_CHARS = 32_768
_TIMEOUT_SECONDS = 120
_MAX_STDOUT_BYTES = 128 * 1024
_MAX_STDERR_BYTES = 128 * 1024
_TERMINATE_GRACE_SECONDS = 1


@dataclass(frozen=True, slots=True, init=False)
class BashTool:
    """Run one bounded, non-interactive foreground Bash command."""

    workspace: Workspace
    bash_path: str
    environment: Mapping[str, str] | None
    max_command_chars: int
    timeout_seconds: float
    max_stdout_bytes: int
    max_stderr_bytes: int
    terminate_grace_seconds: float
    spec: ToolSpec = field(repr=False)

    def __init__(
        self,
        root: PathInput,
        *,
        bash_path: PathInput | None = None,
        environment: Mapping[str, str] | None = None,
        max_command_chars: int = _MAX_COMMAND_CHARS,
        timeout_seconds: float = _TIMEOUT_SECONDS,
        max_stdout_bytes: int = _MAX_STDOUT_BYTES,
        max_stderr_bytes: int = _MAX_STDERR_BYTES,
        terminate_grace_seconds: float = _TERMINATE_GRACE_SECONDS,
    ) -> None:
        max_command_chars = positive_int(max_command_chars, "max_command_chars")
        timeout_seconds = positive_float(timeout_seconds, "timeout_seconds")
        max_stdout_bytes = positive_int(max_stdout_bytes, "max_stdout_bytes")
        max_stderr_bytes = positive_int(max_stderr_bytes, "max_stderr_bytes")
        terminate_grace_seconds = positive_float(
            terminate_grace_seconds,
            "terminate_grace_seconds",
        )
        selected_bash = _bash_path(bash_path)
        selected_environment = _environment(environment)
        object.__setattr__(self, "workspace", Workspace.create(root))
        object.__setattr__(self, "bash_path", selected_bash)
        object.__setattr__(self, "environment", selected_environment)
        object.__setattr__(self, "max_command_chars", max_command_chars)
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(self, "max_stdout_bytes", max_stdout_bytes)
        object.__setattr__(self, "max_stderr_bytes", max_stderr_bytes)
        object.__setattr__(self, "terminate_grace_seconds", terminate_grace_seconds)
        object.__setattr__(self, "spec", _spec(max_command_chars))

    @property
    def root(self) -> Path:
        return self.workspace.root

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        command_value = call.arguments.get("command")
        working_directory_value = call.arguments.get("working_directory", ".")
        if (
            not isinstance(command_value, str)
            or not command_value
            or len(command_value) > self.max_command_chars
        ):
            return _failure(
                "invalid_command",
                "Command is empty or exceeds the configured character limit.",
            )
        command = command_value
        if "\x00" in command:
            return _failure("invalid_command", "Command contains a null character.")
        if (
            not isinstance(working_directory_value, str)
            or not working_directory_value
            or "\x00" in working_directory_value
        ):
            return _failure(
                "invalid_working_directory",
                "Working directory must be non-empty text without null characters.",
            )
        working_directory = working_directory_value
        if context.cancel_requested:
            return _failure("cancelled", "Bash was cancelled.")
        try:
            directory = self.workspace.directory(working_directory)
        except FilesystemFailure as exc:
            return _failure(exc.code, str(exc))
        try:
            outcome = await run_bash(
                bash_path=self.bash_path,
                command=command,
                working_directory=os.fspath(directory),
                environment=self.environment,
                timeout_seconds=self.timeout_seconds,
                max_stdout_bytes=self.max_stdout_bytes,
                max_stderr_bytes=self.max_stderr_bytes,
                terminate_grace_seconds=self.terminate_grace_seconds,
                cancelled=lambda: context.cancel_requested,
            )
        except CommandCancelled:
            return _failure("cancelled", "Bash was cancelled.")
        except CommandSpawnFailed:
            return _failure(
                "command_spawn_failed",
                "Could not start the Host-configured Bash executable.",
            )
        except CommandExecutionFailed:
            return _failure(
                "command_execution_failed",
                "The Bash command could not be observed or cleaned up safely.",
            )
        structured = _structured(outcome)
        if outcome.status == "timeout":
            message = _observation(
                f"Command exceeded the configured {self.timeout_seconds:g}-second timeout.",
                outcome,
            )
            return _failure("command_timeout", message, structured_content=structured)
        return SettledResult(
            ToolSuccess(
                (
                    ContentPart.text_part(
                        _observation(f"Command exited with code {outcome.exit_code}.", outcome)
                    ),
                ),
                structured_content=structured,
            )
        )


def _bash_path(value: PathInput | None) -> str:
    if value is None:
        return shutil.which("bash") or "bash"
    path = os.fspath(value)
    if not path or "\x00" in path:
        raise ValueError("bash_path must be a non-empty path without null characters")
    return path


def _environment(value: object) -> Mapping[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("environment must be a string mapping or None")
    copied: dict[str, str] = {}
    for key, item in cast(Mapping[object, object], value).items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError("environment keys and values must be strings")
        if not key or "=" in key or "\x00" in key or "\x00" in item:
            raise ValueError("environment contains an invalid key or value")
        copied[key] = item
    return MappingProxyType(copied)


def _structured(outcome: CommandOutcome) -> dict[str, object]:
    return {
        "status": outcome.status,
        "exit_code": outcome.exit_code,
        "stdout": outcome.stdout.text,
        "stderr": outcome.stderr.text,
        "duration_ms": outcome.duration_ms,
        "stdout_bytes": outcome.stdout.bytes,
        "stderr_bytes": outcome.stderr.bytes,
        "stdout_truncated": outcome.stdout.truncated,
        "stderr_truncated": outcome.stderr.truncated,
    }


def _observation(prefix: str, outcome: CommandOutcome) -> str:
    sections = [prefix]
    if outcome.stdout.text:
        sections.append(f"stdout:\n{outcome.stdout.text}")
    if outcome.stderr.text:
        sections.append(f"stderr:\n{outcome.stderr.text}")
    if outcome.stdout.truncated or outcome.stderr.truncated:
        sections.append(
            "Output capture was incomplete or exceeded the Host-configured byte limits."
        )
    return "\n\n".join(sections)


def _failure(
    code: str,
    message: str,
    *,
    structured_content: object = None,
) -> ToolResult:
    return SettledResult(
        ToolFailure.from_error(code, message, structured_content=structured_content)
    )


def _spec(max_command_chars: int) -> ToolSpec:
    output = {
        "type": "object",
        "required": [
            "status",
            "exit_code",
            "stdout",
            "stderr",
            "duration_ms",
            "stdout_bytes",
            "stderr_bytes",
            "stdout_truncated",
            "stderr_truncated",
        ],
        "properties": {
            "status": {"enum": ["exit", "timeout"]},
            "exit_code": {"type": ["integer", "null"]},
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "duration_ms": {"type": "integer", "minimum": 0},
            "stdout_bytes": {"type": "integer", "minimum": 0},
            "stderr_bytes": {"type": "integer", "minimum": 0},
            "stdout_truncated": {"type": "boolean"},
            "stderr_truncated": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    return ToolSpec(
        name="Bash",
        description=(
            "Run one non-interactive foreground Bash command from a directory inside the "
            "Host-configured workspace. Output and duration are Host-bounded. The workspace "
            "directory is not a sandbox; commands may access files and networks allowed by "
            "the Host."
        ),
        input_schema={
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": max_command_chars,
                },
                "working_directory": {
                    "type": "string",
                    "minLength": 1,
                    "default": ".",
                },
            },
            "additionalProperties": False,
        },
        output_schema=nullable_output(output),
        execution=ToolExecution(concurrency="serial", read_only=False, idempotent=False),
        risk=ToolRisk(
            filesystem="write",
            network="unrestricted",
            subprocess=True,
            destructive=True,
            requires_approval=True,
        ),
    )
