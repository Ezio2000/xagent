from __future__ import annotations

import asyncio
import ctypes
import os
import shlex
import shutil
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from jharness.kernel import (
    ApprovalAllow,
    ApprovalDecision,
    ApprovalDeny,
    ApprovalPolicy,
    ApprovalRequest,
    Checkpoint,
    ContentPart,
    DeltaSink,
    Event,
    EventKind,
    Invocation,
    Limited,
    LimitReason,
    Message,
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RunContext,
    RunLimits,
    Runtime,
    ToolCall,
    ToolFailure,
    ToolSuccess,
    thaw_json_value,
)
from jharness.toolkit import ToolRegistry
from jharness.tools import BashTool

ResponseFactory = Callable[[int, ModelRequest], ModelResponse]


class _DeterministicModel(Model):
    def __init__(self, respond: ResponseFactory) -> None:
        self._respond = respond
        self.turns = 0

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        del context, stream, emit_delta
        response = self._respond(self.turns, request)
        self.turns += 1
        return response


class _AllowAll(ApprovalPolicy):
    async def decide(self, requests: tuple[ApprovalRequest, ...]) -> tuple[ApprovalDecision, ...]:
        return tuple(ApprovalAllow(request.call.id) for request in requests)


class _DenyAll(ApprovalPolicy):
    async def decide(self, requests: tuple[ApprovalRequest, ...]) -> tuple[ApprovalDecision, ...]:
        return tuple(ApprovalDeny(request.call.id, "test denial") for request in requests)


def _bash_path() -> str:
    path = shutil.which("bash")
    if path is None:
        pytest.skip("Bash is required for BashTool end-to-end tests")
    return path


def _final(text: str = "done") -> ModelResponse:
    return ModelResponse((ContentPart.text_part(text),), finish_reason="stop")


def _structured_success(message: Message) -> dict[str, object]:
    outcome = message.outcome
    assert isinstance(outcome, ToolSuccess)
    structured = thaw_json_value(outcome.structured_content)
    assert isinstance(structured, dict)
    return cast(dict[str, object], structured)


def _tool_messages(request: ModelRequest) -> list[Message]:
    return [message for message in request.messages if message.role == "tool"]


async def _collect(invocation: Invocation) -> tuple[Checkpoint, list[Event]]:
    events = invocation.events()
    result_task = asyncio.create_task(invocation.result())
    observed = [event async for event in events]
    return await result_task, observed


async def _wait_for_file(path: Path, *, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not path.exists():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out waiting for {path.name}")
        await asyncio.sleep(0.01)


def _process_exists(pid: int) -> bool:
    if os.name == "nt":  # pragma: no cover - exercised by Windows CI
        return _windows_process_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _windows_process_exists(pid: int) -> bool:  # pragma: no cover - exercised by Windows CI
    from ctypes import wintypes

    kernel32 = cast(
        Any,
        ctypes.WinDLL("kernel32", use_last_error=True),  # type: ignore[attr-defined]
    )
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, wintypes.LPDWORD)
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        error = cast(int, ctypes.get_last_error())  # type: ignore[attr-defined]
        if error == 87:
            return False
        raise OSError(error, f"OpenProcess failed for child process {pid}")
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            error = cast(int, ctypes.get_last_error())  # type: ignore[attr-defined]
            raise OSError(error, f"GetExitCodeProcess failed for child process {pid}")
        return exit_code.value == 259
    finally:
        kernel32.CloseHandle(handle)


async def _wait_for_process_exit(pid: int, *, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while _process_exists(pid):
        if loop.time() >= deadline:
            raise AssertionError(f"process {pid} survived BashTool settlement")
        await asyncio.sleep(0.01)


def _delayed_descendant_command(started: str, leaked: str, delay: float) -> str:
    program = (
        "import os, time\n"
        "from pathlib import Path\n"
        f"Path({started!r}).write_text(str(os.getpid()), encoding='utf-8')\n"
        f"time.sleep({delay!r})\n"
        f"Path({leaked!r}).write_text('leaked', encoding='utf-8')\n"
    )
    executable = sys.executable.replace("\\", "/")
    return f"{shlex.quote(executable)} -c {shlex.quote(program)} &\nwait"


def _early_exit_descendant_command(started: str, leaked: str, delay: float) -> str:
    program = (
        "import os, time\n"
        "from pathlib import Path\n"
        f"Path({started!r}).write_text(str(os.getpid()), encoding='utf-8')\n"
        f"time.sleep({delay!r})\n"
        f"Path({leaked!r}).write_text('leaked', encoding='utf-8')\n"
    )
    executable = sys.executable.replace("\\", "/")
    return (
        f"{shlex.quote(executable)} -c {shlex.quote(program)} &\n"
        f"while [ ! -s {shlex.quote(started)} ]; do :; done"
    )


def test_runtime_model_calls_bash_and_observes_result_before_final_response(
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    def respond(turn: int, request: ModelRequest) -> ModelResponse:
        assert [spec.name for spec in request.tools] == ["Bash"]
        if turn == 0:
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        "bash-call",
                        "Bash",
                        {
                            "command": "printf 'hello\\n'; printf 'warning\\n' >&2; exit 7",
                            "working_directory": ".",
                        },
                    ),
                )
            )
        if turn == 1:
            tool_messages = _tool_messages(request)
            assert len(tool_messages) == 1
            observed.update(_structured_success(tool_messages[0]))
            return _final("handled Bash result")
        raise AssertionError("unexpected model turn")

    model = _DeterministicModel(respond)
    checkpoint, events = asyncio.run(
        _collect(
            Runtime(
                model=model,
                tools=ToolRegistry((BashTool(tmp_path, bash_path=_bash_path()),)),
                approval_policy=_AllowAll(),
            ).start((Message.user("run a harmless diagnostic"),))
        )
    )

    assert observed["status"] == "exit"
    assert observed["exit_code"] == 7
    assert observed["stdout"] == "hello\n"
    assert observed["stderr"] == "warning\n"
    assert observed["stdout_bytes"] == 6
    assert observed["stderr_bytes"] == 8
    assert observed["stdout_truncated"] is False
    assert observed["stderr_truncated"] is False
    duration_ms = observed["duration_ms"]
    assert isinstance(duration_ms, int)
    assert duration_ms >= 0
    assert model.turns == 2
    assert checkpoint.snapshot.status == "completed"
    assert checkpoint.snapshot.history[-1].parts[0].text == "handled Bash result"
    assert [event.kind for event in events].count(EventKind.TOOL_STARTED) == 1
    assert [event.kind for event in events].count(EventKind.TOOL_FINISHED) == 1


def test_runtime_bash_approval_deny_never_starts_or_changes_workspace(tmp_path: Path) -> None:
    target = tmp_path / "denied.txt"

    def respond(turn: int, request: ModelRequest) -> ModelResponse:
        if turn == 0:
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        "denied-bash",
                        "Bash",
                        {
                            "command": "printf forbidden > denied.txt",
                            "working_directory": ".",
                        },
                    ),
                )
            )
        if turn == 1:
            outcome = request.messages[-1].outcome
            assert isinstance(outcome, ToolFailure)
            assert outcome.error.code == "approval_denied"
            return _final()
        raise AssertionError("unexpected model turn")

    checkpoint, events = asyncio.run(
        _collect(
            Runtime(
                model=_DeterministicModel(respond),
                tools=ToolRegistry((BashTool(tmp_path, bash_path=_bash_path()),)),
                approval_policy=_DenyAll(),
            ).start((Message.user("attempt a denied command"),))
        )
    )

    assert not target.exists()
    outcome = checkpoint.snapshot.history[2].outcome
    assert isinstance(outcome, ToolFailure)
    assert outcome.error.code == "approval_denied"
    kinds = [event.kind for event in events]
    assert EventKind.APPROVAL_REQUESTED in kinds
    assert EventKind.APPROVAL_DECIDED in kinds
    assert EventKind.TOOL_STARTED not in kinds


def test_runtime_two_bash_calls_execute_strictly_serially(tmp_path: Path) -> None:
    def respond(turn: int, request: ModelRequest) -> ModelResponse:
        if turn == 0:
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        "bash-first",
                        "Bash",
                        {
                            "command": (
                                "printf 'first\\n' > serial.txt; "
                                "sleep 0.15; "
                                "printf 'finished\\n' >> serial.txt"
                            ),
                            "working_directory": ".",
                        },
                    ),
                    ToolCall(
                        "bash-second",
                        "Bash",
                        {
                            "command": (
                                "content=$(<serial.txt); "
                                "[[ \"$content\" == $'first\\nfinished' ]] || exit 91; "
                                "printf 'second\\n' >> serial.txt"
                            ),
                            "working_directory": ".",
                        },
                    ),
                )
            )
        if turn == 1:
            results = [_structured_success(message) for message in _tool_messages(request)]
            assert [result["exit_code"] for result in results] == [0, 0]
            return _final()
        raise AssertionError("unexpected model turn")

    checkpoint, events = asyncio.run(
        _collect(
            Runtime(
                model=_DeterministicModel(respond),
                tools=ToolRegistry((BashTool(tmp_path, bash_path=_bash_path()),)),
                approval_policy=_AllowAll(),
            ).start((Message.user("run two ordered commands"),))
        )
    )

    lifecycle: list[tuple[str, str]] = []
    starts = [event for event in events if event.kind is EventKind.TOOL_STARTED]
    for event in events:
        if event.kind is EventKind.TOOL_STARTED:
            call = event.data["call"]
            assert isinstance(call, Mapping)
            lifecycle.append(("started", cast(str, call["id"])))
        elif event.kind is EventKind.TOOL_FINISHED:
            lifecycle.append(("finished", cast(str, event.data["tool_call_id"])))

    assert (tmp_path / "serial.txt").read_text() == "first\nfinished\nsecond\n"
    assert [cast(bool, event.data["parallel"]) for event in starts] == [False, False]
    assert lifecycle == [
        ("started", "bash-first"),
        ("finished", "bash-first"),
        ("started", "bash-second"),
        ("finished", "bash-second"),
    ]
    assert all(
        isinstance(message.outcome, ToolSuccess)
        for message in checkpoint.snapshot.history
        if message.role == "tool"
    )


def test_runtime_active_cancel_cleans_up_bash_descendants(tmp_path: Path) -> None:
    started = tmp_path / "cancel-child.pid"
    leaked = tmp_path / "cancel-leaked.txt"
    delay = 0.6

    def respond(turn: int, request: ModelRequest) -> ModelResponse:
        if turn == 0:
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        "cancel-bash",
                        "Bash",
                        {
                            "command": _delayed_descendant_command(
                                started.name, leaked.name, delay
                            ),
                            "working_directory": ".",
                        },
                    ),
                )
            )
        if turn == 1:
            outcome = request.messages[-1].outcome
            assert isinstance(outcome, ToolFailure)
            assert outcome.error.code == "cancelled"
            return _final()
        raise AssertionError("unexpected model turn")

    async def run() -> tuple[Checkpoint, list[Event]]:
        invocation = Runtime(
            model=_DeterministicModel(respond),
            tools=ToolRegistry(
                (
                    BashTool(
                        tmp_path,
                        bash_path=_bash_path(),
                        terminate_grace_seconds=0.05,
                    ),
                )
            ),
            approval_policy=_AllowAll(),
        ).start((Message.user("cancel the active command"),))
        events = invocation.events()
        result_task = asyncio.create_task(invocation.result())
        observed: list[Event] = []
        async for event in events:
            observed.append(event)
            if event.kind is EventKind.TOOL_STARTED:
                await _wait_for_file(started)
                invocation.cancel_tool("cancel-bash")
        checkpoint = await result_task
        pid = int(started.read_text())
        assert not _process_exists(pid), "active cancellation returned before child cleanup"
        await _wait_for_process_exit(pid)
        await asyncio.sleep(delay + 0.1)
        return checkpoint, observed

    checkpoint, events = asyncio.run(run())
    kinds = [event.kind for event in events]
    assert kinds.index(EventKind.TOOL_CANCEL_REQUESTED) < kinds.index(EventKind.TOOL_FINISHED)
    assert not leaked.exists()
    outcome = checkpoint.snapshot.history[2].outcome
    assert isinstance(outcome, ToolFailure)
    assert outcome.error.code == "cancelled"


def test_runtime_bash_root_exit_cleans_up_background_descendant(tmp_path: Path) -> None:
    started = tmp_path / "root-exit-child.pid"
    leaked = tmp_path / "root-exit-leaked.txt"
    delay = 0.6

    def respond(turn: int, request: ModelRequest) -> ModelResponse:
        if turn == 0:
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        "root-exit-bash",
                        "Bash",
                        {
                            "command": _early_exit_descendant_command(
                                started.name,
                                leaked.name,
                                delay,
                            ),
                            "working_directory": ".",
                        },
                    ),
                )
            )
        if turn == 1:
            result = _structured_success(_tool_messages(request)[0])
            assert result["status"] == "exit"
            assert result["exit_code"] == 0
            return _final()
        raise AssertionError("unexpected model turn")

    async def run() -> tuple[Checkpoint, list[Event]]:
        checkpoint, events = await _collect(
            Runtime(
                model=_DeterministicModel(respond),
                tools=ToolRegistry((BashTool(tmp_path, bash_path=_bash_path()),)),
                approval_policy=_AllowAll(),
            ).start((Message.user("clean up a background child after Bash exits"),))
        )
        pid = int(started.read_text())
        assert not _process_exists(pid), "Bash returned before background child cleanup"
        await _wait_for_process_exit(pid)
        await asyncio.sleep(delay + 0.1)
        return checkpoint, events

    checkpoint, events = asyncio.run(run())
    assert checkpoint.snapshot.status == "completed"
    assert EventKind.TOOL_FINISHED in [event.kind for event in events]
    assert not leaked.exists()


def test_runtime_deadline_cleans_up_bash_descendants_before_return(tmp_path: Path) -> None:
    started = tmp_path / "deadline-child.pid"
    leaked = tmp_path / "deadline-leaked.txt"
    delay = 4.0 if os.name == "nt" else 1.5

    def respond(turn: int, request: ModelRequest) -> ModelResponse:
        if turn == 0:
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        "deadline-bash",
                        "Bash",
                        {
                            "command": _delayed_descendant_command(
                                started.name, leaked.name, delay
                            ),
                            "working_directory": ".",
                        },
                    ),
                )
            )
        raise AssertionError("the runtime deadline must stop before another model turn")

    async def run() -> tuple[Checkpoint, list[Event]]:
        checkpoint, events = await _collect(
            Runtime(
                model=_DeterministicModel(respond),
                tools=ToolRegistry(
                    (
                        BashTool(
                            tmp_path,
                            bash_path=_bash_path(),
                            terminate_grace_seconds=0.05,
                        ),
                    )
                ),
                limits=RunLimits(timeout_seconds=3.0 if os.name == "nt" else 1.0),
                approval_policy=_AllowAll(),
            ).start((Message.user("run until the invocation deadline"),))
        )
        await _wait_for_file(started)
        pid = int(started.read_text())
        assert not _process_exists(pid), "Runtime deadline returned before child cleanup"
        await _wait_for_process_exit(pid)
        await asyncio.sleep(delay + 0.1)
        return checkpoint, events

    checkpoint, events = asyncio.run(run())
    assert checkpoint.snapshot.status == "limited"
    assert isinstance(checkpoint.snapshot.state, Limited)
    assert checkpoint.snapshot.state.reason is LimitReason.DEADLINE
    assert EventKind.TOOL_STARTED in [event.kind for event in events]
    assert not leaked.exists()
