# pyright: reportPrivateUsage=false, reportPrivateImportUsage=false
from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import signal
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any, cast

import pytest

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
from jharness.tools.shell import _output, _runner
from jharness.tools.shell import bash as bash_module

_POSIX_KILL_SIGNAL = cast(signal.Signals, getattr(signal, "SIGKILL", signal.SIGTERM))


async def _emit_progress(_progress: Mapping[str, Any]) -> None:
    return None


def _bash_path() -> Path:
    path = shutil.which("bash")
    if path is None:
        pytest.skip("Bash is not installed")
    return Path(path).resolve()


async def _invoke_async(
    tool: Tool,
    arguments: Mapping[str, Any],
    *,
    is_cancelled: Callable[[], bool] = lambda: False,
    through_registry: bool = False,
) -> ToolSuccess | ToolFailure:
    context = ToolContext(
        RunContext("bash-run", time.monotonic() + 60),
        _emit_progress,
        is_cancelled,
    )
    call = ToolCall("bash-call", tool.spec.name, arguments)
    if through_registry:
        catalog = await ToolRegistry((tool,)).open_catalog()
        result = await catalog.bind(call).invoke(context)
    else:
        result = await tool.invoke(call, context)
    assert isinstance(result, SettledResult)
    assert isinstance(result.outcome, ToolSuccess | ToolFailure)
    return result.outcome


def _invoke(
    tool: Tool,
    arguments: Mapping[str, Any],
    *,
    is_cancelled: Callable[[], bool] = lambda: False,
    through_registry: bool = False,
) -> ToolSuccess | ToolFailure:
    return asyncio.run(
        _invoke_async(
            tool,
            arguments,
            is_cancelled=is_cancelled,
            through_registry=through_registry,
        )
    )


def _success(outcome: ToolSuccess | ToolFailure) -> tuple[str, dict[str, object]]:
    assert isinstance(outcome, ToolSuccess)
    assert len(outcome.parts) == 1
    assert outcome.parts[0].text is not None
    structured = thaw_json_value(outcome.structured_content)
    assert isinstance(structured, dict)
    return outcome.parts[0].text, cast(dict[str, object], structured)


def _failure(
    outcome: ToolSuccess | ToolFailure,
    code: str,
) -> tuple[str, dict[str, object] | None]:
    assert isinstance(outcome, ToolFailure)
    assert outcome.error.code == code
    structured = thaw_json_value(outcome.structured_content)
    assert structured is None or isinstance(structured, dict)
    return outcome.error.message, cast(dict[str, object] | None, structured)


def _output_schema() -> dict[str, object]:
    return {
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


def test_bash_public_contract_and_exact_schemas(tmp_path: Path) -> None:
    bash_path = _bash_path()
    tool = tools.BashTool(
        tmp_path,
        bash_path=bash_path,
        environment={"JHARNESS_BASH_CONTRACT": "configured"},
        max_command_chars=41,
        timeout_seconds=2,
        max_stdout_bytes=43,
        max_stderr_bytes=47,
        terminate_grace_seconds=0.25,
    )

    assert "BashTool" in tools.__all__
    assert isinstance(tool, Tool)
    assert tool.root == tmp_path.resolve()
    assert tool.spec.name == "Bash"
    assert tool.spec.execution.concurrency == "serial"
    assert tool.spec.execution.read_only is False
    assert tool.spec.execution.idempotent is False
    assert tool.spec.parallel_safe is False
    assert tool.spec.risk.filesystem == "write"
    assert tool.spec.risk.network == "unrestricted"
    assert tool.spec.risk.subprocess is True
    assert tool.spec.risk.destructive is True
    assert tool.spec.risk.requires_approval is True
    assert tool.spec.risk.extra == {}
    assert tool.inherit_environment is False
    assert tool.environment["JHARNESS_BASH_CONTRACT"] == "configured"
    assert thaw_json_value(tool.spec.input_schema) == {
        "type": "object",
        "required": ["command"],
        "properties": {
            "command": {"type": "string", "minLength": 1, "maxLength": 41},
            "working_directory": {"type": "string", "minLength": 1, "default": "."},
        },
        "additionalProperties": False,
    }
    assert thaw_json_value(tool.spec.output_schema) == {
        "anyOf": [_output_schema(), {"type": "null"}]
    }

    async def open_catalog() -> tuple[str, ...]:
        catalog = await ToolRegistry((tool,)).open_catalog()
        return tuple(spec.name for spec in catalog.specs())

    assert asyncio.run(open_catalog()) == ("Bash",)


def test_bash_constructor_rejects_non_positive_limits(tmp_path: Path) -> None:
    invalid: tuple[tuple[str, Callable[[Path], object]], ...] = (
        (
            "max_command_chars",
            lambda root: tools.BashTool(root, bash_path=_bash_path(), max_command_chars=0),
        ),
        (
            "max_stdout_bytes",
            lambda root: tools.BashTool(root, bash_path=_bash_path(), max_stdout_bytes=0),
        ),
        (
            "max_stderr_bytes",
            lambda root: tools.BashTool(root, bash_path=_bash_path(), max_stderr_bytes=0),
        ),
        (
            "timeout_seconds",
            lambda root: tools.BashTool(root, bash_path=_bash_path(), timeout_seconds=0),
        ),
        (
            "terminate_grace_seconds",
            lambda root: tools.BashTool(
                root,
                bash_path=_bash_path(),
                terminate_grace_seconds=0,
            ),
        ),
    )
    for keyword, construct in invalid:
        with pytest.raises(ValueError, match=keyword):
            construct(tmp_path)


@pytest.mark.parametrize("bash_path", ["", "bad\x00path"])
def test_bash_constructor_rejects_invalid_executable_paths(
    tmp_path: Path,
    bash_path: str,
) -> None:
    with pytest.raises(ValueError, match="bash_path"):
        tools.BashTool(tmp_path, bash_path=bash_path)


@pytest.mark.parametrize(
    ("environment", "error"),
    [
        (1, TypeError),
        ({1: "value"}, TypeError),
        ({"KEY": 1}, TypeError),
        ({"": "value"}, ValueError),
        ({"BAD=KEY": "value"}, ValueError),
        ({"BAD\x00KEY": "value"}, ValueError),
        ({"KEY": "bad\x00value"}, ValueError),
    ],
)
def test_bash_constructor_validates_environment(
    tmp_path: Path,
    environment: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="environment"):
        tools.BashTool(tmp_path, bash_path=_bash_path(), environment=cast(Any, environment))


def test_bash_constructor_validates_environment_inheritance_flag(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="inherit_environment"):
        tools.BashTool(
            tmp_path,
            bash_path=_bash_path(),
            inherit_environment=cast(Any, 1),
        )


def test_bash_registry_rejects_invalid_model_arguments(tmp_path: Path) -> None:
    tool = tools.BashTool(tmp_path, bash_path=_bash_path(), max_command_chars=4)

    async def validate() -> None:
        catalog = await ToolRegistry((tool,)).open_catalog()
        invalid: tuple[Mapping[str, Any], ...] = (
            {},
            {"command": ""},
            {"command": 1},
            {"command": "12345"},
            {"command": "true", "working_directory": ""},
            {"command": "true", "working_directory": None},
            {"command": "true", "unexpected": True},
        )
        for index, arguments in enumerate(invalid):
            with pytest.raises(ToolError, match="do not match input_schema"):
                catalog.bind(ToolCall(f"invalid-{index}", "Bash", arguments))

    asyncio.run(validate())


def test_bash_direct_invocation_returns_stable_validation_failures(tmp_path: Path) -> None:
    tool = tools.BashTool(tmp_path, bash_path=_bash_path(), max_command_chars=4)
    invalid: tuple[tuple[Mapping[str, Any], str], ...] = (
        ({}, "invalid_command"),
        ({"command": ""}, "invalid_command"),
        ({"command": "12345"}, "invalid_command"),
        ({"command": "x\x00y"}, "invalid_command"),
        ({"command": 1}, "invalid_command"),
        ({"command": "true", "working_directory": ""}, "invalid_working_directory"),
        ({"command": "true", "working_directory": "bad\x00path"}, "invalid_working_directory"),
        ({"command": "true", "working_directory": 1}, "invalid_working_directory"),
    )

    for arguments, code in invalid:
        _failure(_invoke(tool, arguments), code)
    _failure(_invoke(tool, {"command": "true"}, is_cancelled=lambda: True), "cancelled")


def test_bash_default_executable_and_empty_output(tmp_path: Path) -> None:
    tool = tools.BashTool(tmp_path)

    assert tool.bash_path
    text, result = _success(_invoke(tool, {"command": "true"}, through_registry=True))
    assert result["status"] == "exit"
    assert result["exit_code"] == 0
    assert result["stdout"] == ""
    assert result["stderr"] == ""
    assert result["stdout_bytes"] == 0
    assert result["stderr_bytes"] == 0
    assert result["stdout_truncated"] is False
    assert result["stderr_truncated"] is False
    assert "code 0" in text


def test_bash_captures_stdout_stderr_and_unicode(tmp_path: Path) -> None:
    tool = tools.BashTool(tmp_path, bash_path=_bash_path())

    text, result = _success(
        _invoke(
            tool,
            {"command": "printf 'hello 你好 🌍'; printf 'warning ⚠' >&2"},
            through_registry=True,
        )
    )

    assert result == {
        "status": "exit",
        "exit_code": 0,
        "stdout": "hello 你好 🌍",
        "stderr": "warning ⚠",
        "duration_ms": result["duration_ms"],
        "stdout_bytes": len("hello 你好 🌍".encode()),
        "stderr_bytes": len("warning ⚠".encode()),
        "stdout_truncated": False,
        "stderr_truncated": False,
    }
    assert isinstance(result["duration_ms"], int)
    assert result["duration_ms"] >= 0
    assert "hello" in text
    assert "warning" in text


def test_bash_nonzero_exit_is_a_success(tmp_path: Path) -> None:
    tool = tools.BashTool(tmp_path, bash_path=_bash_path())

    _, result = _success(_invoke(tool, {"command": "printf 'partial'; printf 'bad' >&2; exit 23"}))

    assert result["status"] == "exit"
    assert result["exit_code"] == 23
    assert result["stdout"] == "partial"
    assert result["stderr"] == "bad"
    assert result["stdout_truncated"] is False
    assert result["stderr_truncated"] is False


def test_bash_uses_workspace_relative_working_directory(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "marker.txt").write_text("marker", encoding="utf-8")
    tool = tools.BashTool(tmp_path, bash_path=_bash_path())

    _, result = _success(
        _invoke(
            tool,
            {
                "command": "test -f marker.txt && printf 'nested-ok'",
                "working_directory": "nested",
            },
        )
    )

    assert result["exit_code"] == 0
    assert result["stdout"] == "nested-ok"


def test_bash_rejects_invalid_working_directories(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("not a directory", encoding="utf-8")
    tool = tools.BashTool(tmp_path, bash_path=_bash_path())

    _failure(
        _invoke(tool, {"command": "true", "working_directory": ".."}),
        "path_outside_workspace",
    )
    _failure(
        _invoke(tool, {"command": "true", "working_directory": "missing"}),
        "path_not_found",
    )
    _failure(
        _invoke(tool, {"command": "true", "working_directory": "file.txt"}),
        "not_a_directory",
    )


def test_bash_receives_host_configured_environment_without_ambient_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JHARNESS_BASH_NOT_CONFIGURED", "host-secret")
    environment: dict[str, str] = {}
    if os.name == "nt":
        for key in ("SYSTEMROOT", "WINDIR"):
            if key in os.environ:
                environment[key] = os.environ[key]
    environment["JHARNESS_BASH_CONFIGURED"] = "visible"
    tool = tools.BashTool(tmp_path, bash_path=_bash_path(), environment=environment)
    environment["JHARNESS_BASH_CONFIGURED"] = "mutated-after-construction"

    _, result = _success(
        _invoke(
            tool,
            {
                "command": (
                    "printf '%s|%s' \"$JHARNESS_BASH_CONFIGURED\" "
                    '"${JHARNESS_BASH_NOT_CONFIGURED-unset}"'
                )
            },
        )
    )

    assert result["exit_code"] == 0
    assert result["stdout"] == "visible|unset"


def test_bash_environment_inheritance_requires_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JHARNESS_BASH_AMBIENT_SECRET", "host-secret")
    minimal = tools.BashTool(tmp_path, bash_path=_bash_path())
    inherited = tools.BashTool(
        tmp_path,
        bash_path=_bash_path(),
        inherit_environment=True,
    )

    assert minimal.inherit_environment is False
    assert "JHARNESS_BASH_AMBIENT_SECRET" not in minimal.environment
    assert inherited.inherit_environment is True
    assert inherited.environment["JHARNESS_BASH_AMBIENT_SECRET"] == "host-secret"

    _, minimal_result = _success(
        _invoke(
            minimal,
            {"command": "printf '%s' \"${JHARNESS_BASH_AMBIENT_SECRET-unset}\""},
        )
    )
    _, inherited_result = _success(
        _invoke(inherited, {"command": "printf '%s' \"$JHARNESS_BASH_AMBIENT_SECRET\""})
    )
    assert minimal_result["stdout"] == "unset"
    assert inherited_result["stdout"] == "host-secret"


def test_bash_bounds_and_drains_both_output_pipes(tmp_path: Path) -> None:
    chunks = 512
    chunk_size = 1_024
    size = chunks * chunk_size
    tool = tools.BashTool(
        tmp_path,
        bash_path=_bash_path(),
        max_stdout_bytes=97,
        max_stderr_bytes=89,
        timeout_seconds=10,
    )
    command = (
        f"printf -v out '%{chunk_size}s' ''; out=${{out// /O}}; "
        f"printf -v err '%{chunk_size}s' ''; err=${{err// /E}}; "
        f'i=0; while [ "$i" -lt {chunks} ]; do '
        "printf '%s' \"$out\"; printf '%s' \"$err\" >&2; i=$((i + 1)); done"
    )

    _, result = _success(_invoke(tool, {"command": command}, through_registry=True))

    assert result["exit_code"] == 0
    assert result["stdout_bytes"] == size
    assert result["stderr_bytes"] == size
    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is True
    assert set(cast(str, result["stdout"])) <= {"O"}
    assert set(cast(str, result["stderr"])) <= {"E"}
    assert 0 < len(cast(str, result["stdout"]).encode()) <= 97
    assert 0 < len(cast(str, result["stderr"]).encode()) <= 89


def test_bash_truncation_does_not_return_invalid_unicode(tmp_path: Path) -> None:
    value = "🌍" * 16
    tool = tools.BashTool(
        tmp_path,
        bash_path=_bash_path(),
        max_stdout_bytes=1,
        max_stderr_bytes=7,
    )

    _, result = _success(_invoke(tool, {"command": f"printf '{value}'"}))

    assert result["stdout_bytes"] == len(value.encode())
    assert result["stdout_truncated"] is True
    stdout = cast(str, result["stdout"])
    assert stdout.encode("utf-8").decode("utf-8") == stdout


def test_bash_timeout_is_a_failure_with_bounded_partial_output(tmp_path: Path) -> None:
    tool = tools.BashTool(
        tmp_path,
        bash_path=_bash_path(),
        timeout_seconds=1.0 if os.name == "nt" else 0.1,
        terminate_grace_seconds=0.05,
        max_stdout_bytes=64,
        max_stderr_bytes=64,
    )

    started = time.monotonic()
    message, partial = _failure(
        _invoke(
            tool,
            {"command": "printf 'before-timeout'; printf 'warning' >&2; sleep 30"},
            through_registry=True,
        ),
        "command_timeout",
    )

    assert time.monotonic() - started < 5
    assert "timeout" in message.lower() or "timed out" in message.lower()
    assert partial is not None
    assert partial["status"] == "timeout"
    assert partial["exit_code"] is None
    assert partial["stdout"] == "before-timeout"
    assert partial["stderr"] == "warning"
    assert partial["stdout_bytes"] == len(b"before-timeout")
    assert partial["stderr_bytes"] == len(b"warning")
    assert partial["stdout_truncated"] is False
    assert partial["stderr_truncated"] is False


def test_bash_cooperative_cancellation_settles_promptly(tmp_path: Path) -> None:
    tool = tools.BashTool(
        tmp_path,
        bash_path=_bash_path(),
        timeout_seconds=30,
        terminate_grace_seconds=0.05,
    )
    cancelled = Event()

    async def cancel_running_command() -> ToolSuccess | ToolFailure:
        invocation = asyncio.create_task(
            _invoke_async(
                tool,
                {"command": "printf 'started'; sleep 30"},
                is_cancelled=cancelled.is_set,
            )
        )
        await asyncio.sleep(0.1)
        cancelled.set()
        return await asyncio.wait_for(invocation, timeout=5)

    started = time.monotonic()
    message, structured = _failure(asyncio.run(cancel_running_command()), "cancelled")

    assert time.monotonic() - started < 5
    assert "cancel" in message.lower()
    assert structured is None


def test_bash_python_task_cancellation_waits_for_process_cleanup(tmp_path: Path) -> None:
    tool = tools.BashTool(
        tmp_path,
        bash_path=_bash_path(),
        timeout_seconds=30,
        terminate_grace_seconds=0.05,
    )

    async def cancel_task() -> None:
        invocation = asyncio.create_task(
            _invoke_async(tool, {"command": "printf 'started'; sleep 30"})
        )
        await asyncio.sleep(0.1)
        invocation.cancel()
        try:
            await asyncio.wait_for(invocation, timeout=5)
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("task cancellation must propagate after process cleanup")
        assert invocation.cancelled()

    started = time.monotonic()
    asyncio.run(cancel_task())
    assert time.monotonic() - started < 5


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup is POSIX-specific")
def test_bash_reaps_a_pipe_holding_descendant_after_parent_exit(tmp_path: Path) -> None:
    child_pid = tmp_path / "early-child.pid"
    leaked = tmp_path / "early-child-leaked.txt"
    delay = 0.5
    program = (
        "import os, time\n"
        "from pathlib import Path\n"
        f"Path({child_pid.name!r}).write_text(str(os.getpid()), encoding='utf-8')\n"
        f"time.sleep({delay!r})\n"
        f"Path({leaked.name!r}).write_text('leaked', encoding='utf-8')\n"
    )
    command = (
        f"{shlex.quote(sys.executable)} -c {shlex.quote(program)} & "
        f"while [ ! -s {shlex.quote(child_pid.name)} ]; do :; done"
    )
    tool = tools.BashTool(
        tmp_path,
        bash_path=_bash_path(),
        timeout_seconds=5,
        terminate_grace_seconds=0.05,
    )

    started = time.monotonic()
    _, result = _success(_invoke(tool, {"command": command}))
    assert time.monotonic() - started < 5
    assert result["exit_code"] == 0
    assert child_pid.exists()

    pid = int(child_pid.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        if time.monotonic() >= deadline:
            raise AssertionError(f"descendant process {pid} survived BashTool settlement")
        time.sleep(0.01)
    time.sleep(delay + 0.1)
    assert not leaked.exists()


@pytest.mark.skipif(os.name != "posix", reason="setsid is a POSIX API")
def test_bash_bounds_pipe_drain_when_a_descendant_escapes_the_process_group(
    tmp_path: Path,
) -> None:
    child_pid = tmp_path / "escaped-child.pid"
    program = (
        "import os, time\n"
        "from pathlib import Path\n"
        "os.setsid()\n"
        f"Path({child_pid.name!r}).write_text(str(os.getpid()), encoding='utf-8')\n"
        "time.sleep(30)\n"
    )
    command = (
        f"{shlex.quote(sys.executable)} -c {shlex.quote(program)} & "
        f"while [ ! -s {shlex.quote(child_pid.name)} ]; do :; done"
    )
    tool = tools.BashTool(
        tmp_path,
        bash_path=_bash_path(),
        timeout_seconds=5,
        terminate_grace_seconds=0.05,
    )
    pid: int | None = None

    try:
        started = time.monotonic()
        _, result = _success(_invoke(tool, {"command": command}))
        assert time.monotonic() - started < 2
        assert result["exit_code"] == 0
        assert result["stdout_truncated"] is True
        assert result["stderr_truncated"] is True
        pid = int(child_pid.read_text(encoding="utf-8"))
        os.kill(pid, 0)
    finally:
        if pid is None and child_pid.exists():
            pid = int(child_pid.read_text(encoding="utf-8"))
        if pid is not None:
            with suppress(ProcessLookupError):
                os.kill(pid, _POSIX_KILL_SIGNAL)
            deadline = time.monotonic() + 5
            while True:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                if time.monotonic() >= deadline:
                    raise AssertionError(f"escaped process {pid} was not reaped after test cleanup")
                time.sleep(0.01)

    assert pid is not None


def test_bash_missing_executable_is_a_stable_spawn_failure(tmp_path: Path) -> None:
    missing = tmp_path / "missing-bash"
    tool = tools.BashTool(tmp_path, bash_path=missing)

    message, structured = _failure(
        _invoke(tool, {"command": "printf 'must-not-run'"}, through_registry=True),
        "command_spawn_failed",
    )

    assert "start" in message.lower() or "bash" in message.lower()
    assert structured is None


def test_bash_normalizes_post_spawn_execution_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(**_kwargs: object) -> Any:
        raise _runner.CommandExecutionFailed

    monkeypatch.setattr(bash_module, "run_bash", fail)
    message, structured = _failure(
        _invoke(tools.BashTool(tmp_path, bash_path=_bash_path()), {"command": "true"}),
        "command_execution_failed",
    )
    assert "observed" in message or "cleaned up" in message
    assert structured is None


def test_bounded_output_keeps_a_fixed_head_and_tail() -> None:
    output = _output.BoundedOutput(4)
    output.feed(b"")
    output.feed(b"a")
    output.feed(b"bcdef")
    output.feed(b"g")
    assert output.capture() == _output.CapturedOutput("abfg", 7, True)

    exact = _output.BoundedOutput(4)
    exact.feed(b"abcd")
    assert exact.capture() == _output.CapturedOutput("abcd", 4, False)

    one_byte = _output.BoundedOutput(1)
    one_byte.feed(b"xy")
    assert one_byte.capture() == _output.CapturedOutput("x", 2, True)

    incomplete = _output.BoundedOutput(4)
    incomplete.feed(b"ok")
    assert incomplete.capture(incomplete=True) == _output.CapturedOutput("ok", 2, True)


def test_bash_default_path_has_a_stable_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_bash(_name: str) -> None:
        return None

    monkeypatch.setattr(bash_module.shutil, "which", no_bash)
    assert tools.BashTool(tmp_path).bash_path == "bash"


def test_running_process_rejects_missing_pipes() -> None:
    process = SimpleNamespace(stdout=None, stderr=None)
    with pytest.raises(RuntimeError, match="pipes"):
        _runner._RunningProcess.create(cast(Any, process), 1, 1)


def test_process_group_helpers_normalize_os_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, object]] = []

    def available(group: int, selected_signal: object) -> None:
        calls.append((group, selected_signal))

    monkeypatch.setattr(_runner.os, "killpg", available, raising=False)
    assert _runner._signal_group(7, signal.SIGTERM) is True
    assert _runner._group_exists(8) is True
    assert calls == [(7, signal.SIGTERM), (8, 0)]

    def missing(_group: int, _selected_signal: object) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(_runner.os, "killpg", missing)
    assert _runner._signal_group(7, signal.SIGTERM) is False
    assert _runner._group_exists(8) is False

    def denied(_group: int, _selected_signal: object) -> None:
        raise PermissionError

    monkeypatch.setattr(_runner.os, "killpg", denied)
    assert _runner._signal_group(7, signal.SIGTERM) is True
    assert _runner._group_exists(8) is True


def test_windows_job_closes_once_and_exposes_kill_on_close_layout() -> None:
    class Kernel:
        def __init__(self) -> None:
            self.terminated: list[tuple[int, int]] = []
            self.closed: list[int] = []

        def TerminateJobObject(self, handle: int, exit_code: int) -> bool:
            self.terminated.append((handle, exit_code))
            return True

        def CloseHandle(self, handle: int) -> bool:
            self.closed.append(handle)
            return True

    kernel = Kernel()
    job = _runner._WindowsJob(17, kernel)
    job.terminate()
    job.close()
    job.terminate()
    job.close()

    assert kernel.terminated == [(17, 1)]
    assert kernel.closed == [17]
    information = cast(Any, _runner._job_limit_information())
    assert information.BasicLimitInformation.LimitFlags == 0x00002000


def test_posix_termination_covers_force_grace_and_missing_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[signal.Signals] = []

    def record_signal(_group: int, selected: signal.Signals) -> bool:
        signals.append(selected)
        return True

    def group_missing(_group: int, _selected: signal.Signals) -> bool:
        return False

    def group_present(_group: int) -> bool:
        return True

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "_signal_group", record_signal)
        asyncio.run(_runner._terminate_posix_group(11, 1, force=True))
    assert signals == [_POSIX_KILL_SIGNAL]

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "_signal_group", group_missing)
        asyncio.run(_runner._terminate_posix_group(11, 1, force=False))

    signals.clear()
    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "_signal_group", record_signal)
        scoped.setattr(_runner, "_group_exists", group_present)
        asyncio.run(_runner._terminate_posix_group(11, 0, force=False))
    assert signals == [signal.SIGTERM, _POSIX_KILL_SIGNAL]

    group_checks = iter((True, False, False))
    clock = iter((0.0, 0.0, 0.5))

    async def no_wait(_seconds: float) -> None:
        return None

    def signal_succeeds(_group: int, _selected: signal.Signals) -> bool:
        return True

    def next_group_check(_group: int) -> bool:
        return next(group_checks)

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "monotonic", lambda: next(clock))
        scoped.setattr(_runner, "_signal_group", signal_succeeds)
        scoped.setattr(_runner, "_group_exists", next_group_check)
        scoped.setattr(_runner.asyncio, "sleep", no_wait)
        asyncio.run(_runner._terminate_posix_group(11, 1, force=False))


def test_taskkill_success_timeout_and_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Killer:
        def __init__(self, exit_code: int = 0) -> None:
            self.exit_code = exit_code
            self.waited = False
            self.killed = False

        async def wait(self) -> int:
            self.waited = True
            return self.exit_code

        def kill(self) -> None:
            self.killed = True

    killer = Killer()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def spawn(*args: object, **kwargs: object) -> Any:
        calls.append((args, kwargs))
        return killer

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner.asyncio, "create_subprocess_exec", spawn)
        asyncio.run(_runner._taskkill(42))
    assert killer.waited is True
    assert calls[0][0][:3] == ("taskkill.exe", "/PID", "42")

    async def timeout(_awaitable: object, *, timeout: float) -> object:
        del timeout
        if hasattr(_awaitable, "close"):
            cast(Any, _awaitable).close()
        raise TimeoutError

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner.asyncio, "create_subprocess_exec", spawn)
        scoped.setattr(_runner.asyncio, "wait_for", timeout)
        with pytest.raises(_runner.CommandExecutionFailed, match="did not settle"):
            asyncio.run(_runner._taskkill(43))
    assert killer.killed is True

    async def missing(*_args: object, **_kwargs: object) -> Any:
        raise FileNotFoundError

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner.asyncio, "create_subprocess_exec", missing)
        with pytest.raises(_runner.CommandExecutionFailed, match="start taskkill"):
            asyncio.run(_runner._taskkill(44))

    nonzero = Killer(3)

    async def spawn_nonzero(*_args: object, **_kwargs: object) -> Any:
        return nonzero

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner.asyncio, "create_subprocess_exec", spawn_nonzero)
        with pytest.raises(_runner.CommandExecutionFailed, match="code 3"):
            asyncio.run(_runner._taskkill(45))


@pytest.mark.parametrize(
    ("platform_name", "expected_option"),
    [("posix", "start_new_session"), ("nt", "creationflags"), ("other", None)],
)
def test_spawn_bash_selects_platform_options_and_copies_environment(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    expected_option: str | None,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    process = cast(asyncio.subprocess.Process, object())

    async def spawn(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        calls.append((args, kwargs))
        return process

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "os", SimpleNamespace(name=platform_name))
        scoped.setattr(_runner.asyncio, "create_subprocess_exec", spawn)
        result = asyncio.run(
            _runner._spawn_bash(
                bash_path="configured-bash",
                command="true",
                working_directory="workspace",
                environment={"KEY": "value"},
            )
        )

    assert result is process
    args, kwargs = calls[0]
    assert args[:5] == ("configured-bash", "--noprofile", "--norc", "-c", "true")
    assert kwargs["cwd"] == "workspace"
    assert kwargs["env"] == {"KEY": "value"}
    assert ("creationflags" in kwargs) is (expected_option == "creationflags")
    assert ("start_new_session" in kwargs) is (expected_option == "start_new_session")


def test_spawn_bash_normalizes_subprocess_argument_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def invalid(*_args: object, **_kwargs: object) -> Any:
        raise ValueError("invalid spawn arguments")

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner.asyncio, "create_subprocess_exec", invalid)
        with pytest.raises(_runner.CommandSpawnFailed):
            asyncio.run(
                _runner._spawn_bash(
                    bash_path="configured-bash",
                    command="true",
                    working_directory="workspace",
                    environment=None,
                )
            )


def test_running_process_selects_generic_and_windows_tree_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        def __init__(self, returncode: int | None = None) -> None:
            self.returncode = returncode
            self.pid = 91
            self.killed = 0
            self.terminated = 0

        def kill(self) -> None:
            self.killed += 1

        def terminate(self) -> None:
            self.terminated += 1

    class State:
        def __init__(self, process: Process) -> None:
            self.process = process
            self.process_group: int | None = None
            self.windows_calls: list[tuple[float, bool]] = []

        async def _terminate_windows(self, grace: float, *, force: bool) -> None:
            self.windows_calls.append((grace, force))

    async def terminate(state: State, *, force: bool) -> None:
        await _runner._RunningProcess._terminate_tree(
            cast(Any, state),
            0.25,
            force=force,
        )

    process = Process()
    state = State(process)
    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "os", SimpleNamespace(name="other"))
        asyncio.run(terminate(state, force=True))
        asyncio.run(terminate(state, force=False))
    assert process.killed == 1
    assert process.terminated == 1

    completed = State(Process(0))
    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "os", SimpleNamespace(name="other"))
        asyncio.run(terminate(completed, force=False))
    assert completed.process.terminated == 0

    windows = State(Process())
    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "os", SimpleNamespace(name="nt"))
        asyncio.run(terminate(windows, force=True))
    assert windows.windows_calls == [(0.25, True)]

    posix = State(Process())
    posix.process_group = 91
    posix_calls: list[tuple[int, float, bool]] = []

    async def terminate_posix(group: int, grace: float, *, force: bool) -> None:
        posix_calls.append((group, grace, force))

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "os", SimpleNamespace(name="posix"))
        scoped.setattr(_runner, "_terminate_posix_group", terminate_posix)
        asyncio.run(terminate(posix, force=True))
    assert posix_calls == [(91, 0.25, True)]


def test_windows_tree_cleanup_uses_job_or_taskkill_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        def __init__(self, returncode: int | None) -> None:
            self.returncode = returncode
            self.pid = 73
            self.signals: list[object] = []

        def send_signal(self, selected: object) -> None:
            self.signals.append(selected)

    class Job:
        def __init__(self, *, closed: bool = False) -> None:
            self.closed = closed
            self.terminations = 0

        def terminate(self) -> None:
            self.terminations += 1

    class State:
        def __init__(self, process: Process, job: Job | None) -> None:
            self.process = process
            self.windows_job = job

    killed: list[int] = []

    async def taskkill(process_id: int) -> None:
        killed.append(process_id)

    async def no_wait(_seconds: float) -> None:
        return None

    async def terminate(state: State, *, force: bool) -> None:
        await _runner._RunningProcess._terminate_windows(
            cast(Any, state),
            0.01,
            force=force,
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "_taskkill", taskkill)
        scoped.setattr(_runner.asyncio, "sleep", no_wait)

        active_job = Job()
        asyncio.run(terminate(State(Process(None), active_job), force=True))
        assert active_job.terminations == 1

        closed_job = Job(closed=True)
        asyncio.run(terminate(State(Process(None), closed_job), force=True))

        graceful_job = Job()
        graceful_process = Process(None)
        asyncio.run(terminate(State(graceful_process, graceful_job), force=False))
        assert graceful_process.signals == [_runner._WINDOWS_CTRL_BREAK_EVENT]
        assert graceful_job.terminations == 1

        asyncio.run(terminate(State(Process(0), None), force=False))

        fallback_process = Process(None)
        asyncio.run(terminate(State(fallback_process, None), force=False))
        assert fallback_process.signals == [_runner._WINDOWS_CTRL_BREAK_EVENT]

    assert killed == [73, 73]


def test_windows_tree_cleanup_suppresses_signal_races(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 81
        returncode = None

        def send_signal(self, _selected: object) -> None:
            raise ProcessLookupError

    class Job:
        closed = False

        def __init__(self) -> None:
            self.terminations = 0

        def terminate(self) -> None:
            self.terminations += 1

    state = SimpleNamespace(process=Process(), windows_job=Job())

    async def no_wait(_seconds: float) -> None:
        return None

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner.asyncio, "sleep", no_wait)
        asyncio.run(
            _runner._RunningProcess._terminate_windows(
                cast(Any, state),
                0,
                force=False,
            )
        )
    assert state.windows_job.terminations == 1


def test_running_process_finish_always_closes_windows_job() -> None:
    class Job:
        def __init__(self) -> None:
            self.closes = 0

        def close(self) -> None:
            self.closes += 1

    class State:
        def __init__(self, job: Job) -> None:
            self.windows_job = job
            self.stdout_output = _output.BoundedOutput(4)
            self.stderr_output = _output.BoundedOutput(4)
            self.stdout_output.feed(b"out")
            self.stderr_output.feed(b"err")

        async def _terminate_tree(self, _grace: float, *, force: bool) -> None:
            assert force is False

        async def _settle_pipes(self) -> tuple[bool, bool]:
            return False, False

    async def finish() -> tuple[_output.CapturedOutput, _output.CapturedOutput, Job]:
        job = Job()
        state = State(job)
        result = await _runner._RunningProcess.finish(
            cast(Any, state),
            terminate_grace_seconds=0,
            force=False,
        )
        return result[0], result[1], job

    stdout, stderr, job = asyncio.run(finish())
    assert stdout == _output.CapturedOutput("out", 3, False)
    assert stderr == _output.CapturedOutput("err", 3, False)
    assert job.closes == 1


def test_running_process_bounds_pipe_settlement_and_closes_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Transport:
        def __init__(self) -> None:
            self.closes = 0

        def close(self) -> None:
            self.closes += 1

    class Process:
        def __init__(self, transport: Transport) -> None:
            self._transport = transport
            self.returncode = 0

    async def never() -> Any:
        await asyncio.Event().wait()

    async def settle() -> tuple[Any, Transport, tuple[bool, bool]]:
        transport = Transport()
        state = SimpleNamespace(
            process=Process(transport),
            wait_task=asyncio.create_task(never()),
            stdout_task=asyncio.create_task(never()),
            stderr_task=asyncio.create_task(never()),
        )
        incomplete = await _runner._RunningProcess._settle_pipes(cast(Any, state))
        return state, transport, incomplete

    monkeypatch.setattr(_runner, "_PIPE_DRAIN_SECONDS", 0.001)
    state, transport, incomplete = asyncio.run(settle())
    assert transport.closes == 1
    assert incomplete == (True, True)
    assert state.wait_task.cancelled()
    assert state.stdout_task.cancelled()
    assert state.stderr_task.cancelled()


def test_running_process_fails_when_the_root_cannot_be_reaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Transport:
        def close(self) -> None:
            return None

    class Process:
        returncode = None

        def __init__(self) -> None:
            self._transport = Transport()
            self.kills = 0

        def kill(self) -> None:
            self.kills += 1

    async def never() -> Any:
        await asyncio.Event().wait()

    async def settle() -> Process:
        process = Process()
        state = SimpleNamespace(
            process=process,
            wait_task=asyncio.create_task(never()),
            stdout_task=asyncio.create_task(never()),
            stderr_task=asyncio.create_task(never()),
        )
        with pytest.raises(_runner.CommandExecutionFailed, match="could not be reaped"):
            await _runner._RunningProcess._settle_pipes(cast(Any, state))
        return process

    monkeypatch.setattr(_runner, "_PIPE_DRAIN_SECONDS", 0.001)
    process = asyncio.run(settle())
    assert process.kills == 1


def test_running_process_cancels_only_pending_pipe_readers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Transport:
        def __init__(self) -> None:
            self.closes = 0

        def close(self) -> None:
            self.closes += 1

    async def value(result: Any) -> Any:
        return result

    async def never() -> Any:
        await asyncio.Event().wait()

    async def settle() -> tuple[tuple[bool, bool], Any, Transport]:
        transport = Transport()
        stdout_task = asyncio.create_task(value(_output.CapturedOutput("done", 4, False)))
        wait_task = asyncio.create_task(value(0))
        await asyncio.sleep(0)
        state = SimpleNamespace(
            process=SimpleNamespace(_transport=transport, returncode=0),
            wait_task=wait_task,
            stdout_task=stdout_task,
            stderr_task=asyncio.create_task(never()),
        )
        incomplete = await _runner._RunningProcess._settle_pipes(cast(Any, state))
        return incomplete, state, transport

    monkeypatch.setattr(_runner, "_PIPE_DRAIN_SECONDS", 0.001)
    incomplete, state, transport = asyncio.run(settle())
    assert incomplete == (False, True)
    assert state.stdout_task.cancelled() is False
    assert state.stderr_task.cancelled() is True
    assert transport.closes == 1


def test_running_process_propagates_reader_errors() -> None:
    async def settle() -> None:
        async def value(result: Any) -> Any:
            return result

        async def broken() -> Any:
            raise OSError("pipe read failed")

        state = SimpleNamespace(
            process=SimpleNamespace(),
            wait_task=asyncio.create_task(value(0)),
            stdout_task=asyncio.create_task(broken()),
            stderr_task=asyncio.create_task(value(_output.CapturedOutput("", 0, False))),
        )
        with pytest.raises(OSError, match="pipe read failed"):
            await _runner._RunningProcess._settle_pipes(cast(Any, state))

    asyncio.run(settle())


def test_close_process_transport_accepts_an_absent_transport() -> None:
    _runner._close_process_transport(cast(Any, SimpleNamespace()))


def test_running_process_create_handles_non_posix_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.stdout.feed_eof()
            self.stderr.feed_eof()
            self.pid = 99
            self.returncode = 0

        async def wait(self) -> int:
            return 0

    async def create_and_finish() -> _runner._RunningProcess:
        process = Process()
        state = _runner._RunningProcess.create(cast(Any, process), 4, 4)
        await state.finish(terminate_grace_seconds=0, force=False)
        return state

    def no_job(_pid: int) -> None:
        return None

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "os", SimpleNamespace(name="other"))
        scoped.setattr(_runner._WindowsJob, "attach", no_job)
        state = asyncio.run(create_and_finish())
    assert state.process_group is None


def test_settle_after_task_cancellation_suppresses_cleanup_failures() -> None:
    class BrokenState:
        async def finish(self, *, terminate_grace_seconds: float, force: bool) -> None:
            assert terminate_grace_seconds == 0
            assert force is True
            raise RuntimeError("cleanup failed")

    asyncio.run(_runner._settle_after_task_cancellation(cast(Any, BrokenState())))


def test_settle_after_task_cancellation_resists_repeated_cancellation() -> None:
    class State:
        async def finish(self, *, terminate_grace_seconds: float, force: bool) -> None:
            assert terminate_grace_seconds == 0
            assert force is True
            await asyncio.sleep(0)

    async def settle() -> None:
        current = asyncio.current_task()
        assert current is not None
        asyncio.get_running_loop().call_soon(current.cancel)
        await _runner._settle_after_task_cancellation(cast(Any, State()))
        while current.cancelling():
            current.uncancel()

    asyncio.run(settle())


def test_await_task_resists_cancellation_until_the_owned_task_finishes() -> None:
    async def run() -> None:
        async def value() -> int:
            await asyncio.sleep(0)
            return 17

        owned = asyncio.create_task(value())
        current = asyncio.current_task()
        assert current is not None
        asyncio.get_running_loop().call_soon(current.cancel)
        assert await _runner._await_task_resisting_cancellation(owned) == 17
        assert current.cancelling() == 0

    asyncio.run(run())


def test_clear_cancellation_requests_accepts_no_current_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_current_task() -> None:
        return None

    monkeypatch.setattr(_runner.asyncio, "current_task", no_current_task)
    _runner._clear_cancellation_requests()


def test_spawn_cancellation_reaps_a_process_created_during_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    process = cast(asyncio.subprocess.Process, object())
    finishes: list[tuple[float, bool]] = []

    class State:
        async def finish(self, *, terminate_grace_seconds: float, force: bool) -> None:
            finishes.append((terminate_grace_seconds, force))

    async def delayed_spawn(**_kwargs: object) -> asyncio.subprocess.Process:
        started.set()
        await release.wait()
        return process

    def capture(
        selected: asyncio.subprocess.Process,
        _stdout: int,
        _stderr: int,
    ) -> Any:
        assert selected is process
        return State()

    async def exercise() -> None:
        invocation = asyncio.create_task(
            _runner._spawn_owned_bash(
                bash_path="bash",
                command="true",
                working_directory="workspace",
                environment=None,
                max_stdout_bytes=1,
                max_stderr_bytes=1,
            )
        )
        await started.wait()
        invocation.cancel()
        release.set()
        try:
            await invocation
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("spawn cancellation must propagate after process cleanup")

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "_spawn_bash", delayed_spawn)
        scoped.setattr(_runner, "_capture_process", capture)
        asyncio.run(exercise())
    assert finishes == [(0.0, True)]


def test_spawn_cancellation_preserves_cancellation_when_spawn_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def failed_spawn(**_kwargs: object) -> Any:
        started.set()
        await release.wait()
        raise _runner.CommandSpawnFailed

    async def exercise() -> None:
        invocation = asyncio.create_task(
            _runner._spawn_owned_bash(
                bash_path="bash",
                command="true",
                working_directory="workspace",
                environment=None,
                max_stdout_bytes=1,
                max_stderr_bytes=1,
            )
        )
        await started.wait()
        invocation.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await invocation

    monkeypatch.setattr(_runner, "_spawn_bash", failed_spawn)
    asyncio.run(exercise())


def test_run_bash_rejects_pre_spawn_cancellation(tmp_path: Path) -> None:
    with pytest.raises(_runner.CommandCancelled):
        asyncio.run(
            _runner.run_bash(
                bash_path=str(tmp_path / "missing"),
                command="true",
                working_directory=str(tmp_path),
                environment=None,
                timeout_seconds=1,
                max_stdout_bytes=1,
                max_stderr_bytes=1,
                terminate_grace_seconds=1,
                cancelled=lambda: True,
            )
        )


def test_run_bash_maps_expected_wait_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SimpleNamespace(wait_task=SimpleNamespace(done=lambda: True))
    cleanups: list[object] = []

    async def spawn(**_kwargs: object) -> Any:
        return object()

    async def fail_wait(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("wait failed")

    async def cleanup(selected: object) -> None:
        cleanups.append(selected)

    def create_state(
        _process: asyncio.subprocess.Process,
        _stdout: int,
        _stderr: int,
    ) -> Any:
        return state

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "_spawn_bash", spawn)
        scoped.setattr(_runner._RunningProcess, "create", create_state)
        scoped.setattr(_runner, "_wait_reason", fail_wait)
        scoped.setattr(_runner, "_settle_after_task_cancellation", cleanup)
        with pytest.raises(_runner.CommandExecutionFailed):
            asyncio.run(
                _runner.run_bash(
                    bash_path="bash",
                    command="true",
                    working_directory="workspace",
                    environment=None,
                    timeout_seconds=1,
                    max_stdout_bytes=1,
                    max_stderr_bytes=1,
                    terminate_grace_seconds=1,
                    cancelled=lambda: False,
                )
            )
    assert cleanups == [state]


def test_run_bash_reaps_after_unexpected_wait_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WaitTask:
        def done(self) -> bool:
            return True

    state = SimpleNamespace(wait_task=WaitTask())
    cleanups: list[object] = []

    async def spawn(**_kwargs: object) -> Any:
        return object()

    async def fail_wait(*_args: object, **_kwargs: object) -> Any:
        raise ValueError("wait failed")

    async def cleanup(selected: object) -> None:
        cleanups.append(selected)

    def create_state(
        _process: asyncio.subprocess.Process,
        _stdout: int,
        _stderr: int,
    ) -> Any:
        return state

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "_spawn_bash", spawn)
        scoped.setattr(_runner._RunningProcess, "create", create_state)
        scoped.setattr(_runner, "_wait_reason", fail_wait)
        scoped.setattr(_runner, "_settle_after_task_cancellation", cleanup)
        with pytest.raises(ValueError, match="wait failed"):
            asyncio.run(
                _runner.run_bash(
                    bash_path="bash",
                    command="true",
                    working_directory="workspace",
                    environment=None,
                    timeout_seconds=1,
                    max_stdout_bytes=1,
                    max_stderr_bytes=1,
                    terminate_grace_seconds=1,
                    cancelled=lambda: False,
                )
            )
    assert cleanups == [state]


def test_run_bash_clamps_negative_clock_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _output.CapturedOutput("", 0, False)

    class WaitTask:
        def done(self) -> bool:
            return True

        def result(self) -> int:
            return 5

    class State:
        wait_task = WaitTask()
        process = SimpleNamespace(returncode=5)

        async def finish(
            self,
            *,
            terminate_grace_seconds: float,
            force: bool,
        ) -> tuple[_output.CapturedOutput, _output.CapturedOutput]:
            del terminate_grace_seconds
            assert force is False
            return captured, captured

    state = State()
    clock = iter((2.0, 3.0, 1.0))

    async def spawn(**_kwargs: object) -> Any:
        return object()

    async def exited(*_args: object, **_kwargs: object) -> str:
        return "exit"

    def create_state(
        _process: asyncio.subprocess.Process,
        _stdout: int,
        _stderr: int,
    ) -> Any:
        return state

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "monotonic", lambda: next(clock))
        scoped.setattr(_runner, "_spawn_bash", spawn)
        scoped.setattr(_runner._RunningProcess, "create", create_state)
        scoped.setattr(_runner, "_wait_reason", exited)
        outcome = asyncio.run(
            _runner.run_bash(
                bash_path="bash",
                command="true",
                working_directory="workspace",
                environment=None,
                timeout_seconds=1,
                max_stdout_bytes=1,
                max_stderr_bytes=1,
                terminate_grace_seconds=1,
                cancelled=lambda: False,
            )
        )
    assert outcome.exit_code == 5
    assert outcome.duration_ms == 0


def test_run_bash_does_not_repeat_cleanup_after_full_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _output.CapturedOutput("", 0, False)
    cleanups: list[object] = []

    class State:
        process = SimpleNamespace(returncode=0)

        async def finish(
            self,
            *,
            terminate_grace_seconds: float,
            force: bool,
        ) -> tuple[_output.CapturedOutput, _output.CapturedOutput]:
            del terminate_grace_seconds
            assert force is False
            return captured, captured

    state = State()
    clock = iter((1.0, 2.0))

    def fail_after_settlement() -> float:
        try:
            return next(clock)
        except StopIteration as exc:
            raise RuntimeError("clock failed after settlement") from exc

    async def spawn(**_kwargs: object) -> Any:
        return object()

    async def exited(*_args: object, **_kwargs: object) -> str:
        return "exit"

    async def cleanup(selected: object) -> None:
        cleanups.append(selected)

    def create_state(
        _process: asyncio.subprocess.Process,
        _stdout: int,
        _stderr: int,
    ) -> Any:
        return state

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "monotonic", fail_after_settlement)
        scoped.setattr(_runner, "_spawn_bash", spawn)
        scoped.setattr(_runner._RunningProcess, "create", create_state)
        scoped.setattr(_runner, "_wait_reason", exited)
        scoped.setattr(_runner, "_settle_after_task_cancellation", cleanup)
        with pytest.raises(_runner.CommandExecutionFailed):
            asyncio.run(
                _runner.run_bash(
                    bash_path="bash",
                    command="true",
                    working_directory="workspace",
                    environment=None,
                    timeout_seconds=1,
                    max_stdout_bytes=1,
                    max_stderr_bytes=1,
                    terminate_grace_seconds=1,
                    cancelled=lambda: False,
                )
            )
    assert cleanups == []


def test_run_bash_rejects_an_exit_without_a_root_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _output.CapturedOutput("", 0, False)
    cleanups: list[object] = []

    class State:
        process = SimpleNamespace(returncode=None)

        async def finish(
            self,
            *,
            terminate_grace_seconds: float,
            force: bool,
        ) -> tuple[_output.CapturedOutput, _output.CapturedOutput]:
            del terminate_grace_seconds
            assert force is False
            return captured, captured

    state = State()

    async def spawn(**_kwargs: object) -> Any:
        return object()

    async def exited(*_args: object, **_kwargs: object) -> str:
        return "exit"

    async def cleanup(selected: object) -> None:
        cleanups.append(selected)

    def create_state(
        _process: asyncio.subprocess.Process,
        _stdout: int,
        _stderr: int,
    ) -> Any:
        return state

    with monkeypatch.context() as scoped:
        scoped.setattr(_runner, "_spawn_bash", spawn)
        scoped.setattr(_runner._RunningProcess, "create", create_state)
        scoped.setattr(_runner, "_wait_reason", exited)
        scoped.setattr(_runner, "_settle_after_task_cancellation", cleanup)
        with pytest.raises(_runner.CommandExecutionFailed, match="exit code"):
            asyncio.run(
                _runner.run_bash(
                    bash_path="bash",
                    command="true",
                    working_directory="workspace",
                    environment=None,
                    timeout_seconds=1,
                    max_stdout_bytes=1,
                    max_stderr_bytes=1,
                    terminate_grace_seconds=1,
                    cancelled=lambda: False,
                )
            )
    assert cleanups == [state]
