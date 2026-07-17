"""Async Bash process lifecycle with bounded output and tree cleanup."""

from __future__ import annotations

import asyncio
import ctypes
import os
import signal
import subprocess
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from time import monotonic
from typing import Any, Literal, TypeVar, cast

from jharness.tools.shell._output import BoundedOutput, CapturedOutput, drain_stream

_CANCEL_POLL_SECONDS = 0.02
_PIPE_DRAIN_SECONDS = 0.1
_FORCED_EXIT_CODE = 1
_WINDOWS_CREATE_NEW_PROCESS_GROUP = 0x00000200
_WINDOWS_CTRL_BREAK_EVENT = 1
_POSIX_KILL_SIGNAL = cast(signal.Signals, getattr(signal, "SIGKILL", signal.SIGTERM))

_T = TypeVar("_T")


class CommandCancelled(Exception):
    """The Host requested cancellation of an active command."""


class CommandSpawnFailed(Exception):
    """The configured Bash process could not be started."""


class CommandExecutionFailed(Exception):
    """A started command could not be observed or settled safely."""


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """A reaped Bash root process and its bounded observations."""

    status: Literal["exit", "timeout"]
    exit_code: int | None
    stdout: CapturedOutput
    stderr: CapturedOutput
    duration_ms: int


@dataclass(slots=True)
class _WindowsJob:  # pragma: no cover - exercised by Windows CI
    """Best-effort Windows Job Object ownership for descendant cleanup."""

    handle: int
    _kernel32: Any
    closed: bool = False

    @classmethod
    def attach(cls, process_id: int) -> _WindowsJob | None:
        if os.name != "nt":
            return None
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
            kernel32.CreateJobObjectW.restype = ctypes.c_void_p
            kernel32.SetInformationJobObject.argtypes = (
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint32,
            )
            kernel32.SetInformationJobObject.restype = ctypes.c_int
            kernel32.OpenProcess.argtypes = (
                ctypes.c_uint32,
                ctypes.c_int,
                ctypes.c_uint32,
            )
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.AssignProcessToJobObject.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
            kernel32.AssignProcessToJobObject.restype = ctypes.c_int
            kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
            kernel32.CloseHandle.restype = ctypes.c_int
            kernel32.TerminateJobObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
            kernel32.TerminateJobObject.restype = ctypes.c_int

            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return None
            information = _job_limit_information()
            configured = kernel32.SetInformationJobObject(
                job,
                9,  # JobObjectExtendedLimitInformation
                ctypes.byref(information),
                ctypes.sizeof(information),
            )
            if not configured:
                kernel32.CloseHandle(job)
                return None
            process_handle = kernel32.OpenProcess(
                0x0100 | 0x0200 | 0x0400 | 0x0001,
                False,
                process_id,
            )
            if not process_handle:
                kernel32.CloseHandle(job)
                return None
            try:
                assigned = kernel32.AssignProcessToJobObject(job, process_handle)
            finally:
                kernel32.CloseHandle(process_handle)
            if not assigned:
                kernel32.CloseHandle(job)
                return None
            return cls(cast(int, job), kernel32)
        except (AttributeError, OSError, TypeError, ValueError):
            return None

    def terminate(self) -> None:
        if not self.closed and not self._kernel32.TerminateJobObject(
            self.handle,
            _FORCED_EXIT_CODE,
        ):
            raise OSError("TerminateJobObject failed")

    def close(self) -> None:
        if not self.closed:
            if not self._kernel32.CloseHandle(self.handle):
                raise OSError("CloseHandle failed")
            self.closed = True


def _job_limit_information() -> ctypes.Structure:  # pragma: no cover - Windows only
    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    information = _ExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = 0x00002000
    return information


@dataclass(slots=True)
class _RunningProcess:
    process: asyncio.subprocess.Process
    wait_task: asyncio.Task[int]
    stdout_output: BoundedOutput
    stderr_output: BoundedOutput
    stdout_task: asyncio.Task[CapturedOutput]
    stderr_task: asyncio.Task[CapturedOutput]
    process_group: int | None
    windows_job: _WindowsJob | None

    @classmethod
    def create(
        cls,
        process: asyncio.subprocess.Process,
        stdout_limit: int,
        stderr_limit: int,
    ) -> _RunningProcess:
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("subprocess pipes were not created")
        stdout_output = BoundedOutput(stdout_limit)
        stderr_output = BoundedOutput(stderr_limit)
        return cls(
            process,
            asyncio.create_task(process.wait()),
            stdout_output,
            stderr_output,
            asyncio.create_task(drain_stream(process.stdout, stdout_output)),
            asyncio.create_task(drain_stream(process.stderr, stderr_output)),
            process.pid if os.name == "posix" else None,
            _WindowsJob.attach(process.pid),
        )

    async def finish(
        self,
        *,
        terminate_grace_seconds: float,
        force: bool,
    ) -> tuple[CapturedOutput, CapturedOutput]:
        try:
            await self._terminate_tree(terminate_grace_seconds, force=force)
            stdout_incomplete, stderr_incomplete = await self._settle_pipes()
            return (
                self.stdout_output.capture(incomplete=stdout_incomplete),
                self.stderr_output.capture(incomplete=stderr_incomplete),
            )
        finally:
            if self.windows_job is not None:
                self.windows_job.close()

    async def _settle_pipes(self) -> tuple[bool, bool]:
        tasks = (self.wait_task, self.stdout_task, self.stderr_task)
        _, pending = await asyncio.wait(tasks, timeout=_PIPE_DRAIN_SECONDS)
        incomplete = (self.stdout_task in pending, self.stderr_task in pending)
        if pending:
            _close_process_transport(self.process)
            for task in (self.stdout_task, self.stderr_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(self.stdout_task, self.stderr_task, return_exceptions=True)
            if not self.wait_task.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(self.wait_task),
                        timeout=_PIPE_DRAIN_SECONDS,
                    )
                except TimeoutError as exc:
                    self.wait_task.cancel()
                    await asyncio.gather(self.wait_task, return_exceptions=True)
                    if self.process.returncode is None:
                        with suppress(ProcessLookupError):
                            self.process.kill()
                        raise CommandExecutionFailed(
                            "Bash root process could not be reaped"
                        ) from exc
        for task in (self.stdout_task, self.stderr_task):
            if not task.cancelled() and (error := task.exception()) is not None:
                raise error
        return incomplete

    async def _terminate_tree(self, grace_seconds: float, *, force: bool) -> None:
        if os.name == "posix" and self.process_group is not None:
            await _terminate_posix_group(self.process_group, grace_seconds, force=force)
            return
        if os.name == "nt":  # pragma: no cover - exercised by Windows CI
            await self._terminate_windows(grace_seconds, force=force)
            return
        if self.process.returncode is None:  # pragma: no cover - unsupported OS fallback
            if force:
                self.process.kill()
            else:
                self.process.terminate()

    async def _terminate_windows(  # pragma: no cover - exercised by Windows CI
        self,
        grace_seconds: float,
        *,
        force: bool,
    ) -> None:
        if force:
            if self.windows_job is not None and not self.windows_job.closed:
                self.windows_job.terminate()
            elif self.process.returncode is None:
                await _taskkill(self.process.pid)
            return
        if self.process.returncode is None:
            with suppress(ProcessLookupError, OSError):
                self.process.send_signal(cast(signal.Signals, _WINDOWS_CTRL_BREAK_EVENT))
            await asyncio.sleep(grace_seconds)
        if self.windows_job is not None:
            self.windows_job.terminate()
        elif self.process.returncode is None:
            await _taskkill(self.process.pid)


async def _terminate_posix_group(group: int, grace_seconds: float, *, force: bool) -> None:
    if force:
        _signal_group(group, _POSIX_KILL_SIGNAL)
        return
    if not _signal_group(group, signal.SIGTERM):
        return
    deadline = monotonic() + grace_seconds
    while _group_exists(group):
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        await asyncio.sleep(min(_CANCEL_POLL_SECONDS, remaining))
    if _group_exists(group):
        _signal_group(group, _POSIX_KILL_SIGNAL)


def _kill_process_group(group: int, selected_signal: signal.Signals | int) -> None:
    killpg = cast(Callable[[int, int], None] | None, vars(os).get("killpg"))
    if killpg is None:  # pragma: no cover - guarded by the POSIX execution path
        raise OSError("process groups are unavailable")
    killpg(group, int(selected_signal))


def _signal_group(group: int, selected_signal: signal.Signals) -> bool:
    try:
        _kill_process_group(group, selected_signal)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _group_exists(group: int) -> bool:
    try:
        _kill_process_group(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _close_process_transport(process: asyncio.subprocess.Process) -> None:
    """Close inherited pipes after the bounded drain window expires."""

    transport = cast(Any, getattr(process, "_transport", None))
    if transport is not None:
        transport.close()


async def _taskkill(process_id: int) -> None:  # pragma: no cover - Windows only
    """Use the Windows tree-aware fallback when a Job Object is unavailable."""

    try:
        killer = await asyncio.create_subprocess_exec(
            "taskkill.exe",
            "/PID",
            str(process_id),
            "/T",
            "/F",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError) as exc:
        raise CommandExecutionFailed("Could not start taskkill.exe") from exc
    wait_task = asyncio.create_task(killer.wait())
    try:
        exit_code = await asyncio.wait_for(asyncio.shield(wait_task), timeout=5.0)
    except TimeoutError as exc:
        with suppress(ProcessLookupError):
            killer.kill()
        await _await_task_resisting_cancellation(wait_task)
        raise CommandExecutionFailed("taskkill.exe did not settle") from exc
    except asyncio.CancelledError:
        with suppress(ProcessLookupError):
            killer.kill()
        await _await_task_resisting_cancellation(wait_task)
        raise
    if exit_code != 0:
        raise CommandExecutionFailed(f"taskkill.exe exited with code {exit_code}")


async def _await_task_resisting_cancellation(task: asyncio.Task[_T]) -> _T:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            _clear_cancellation_requests()
    return task.result()


async def _settle_after_task_cancellation(state: _RunningProcess) -> None:
    """Do not let repeated cancellation interrupt force-kill and reap."""

    cleanup = asyncio.create_task(state.finish(terminate_grace_seconds=0.0, force=True))
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            _clear_cancellation_requests()
            continue
        except Exception:
            break
    with suppress(asyncio.CancelledError, Exception):
        cleanup.result()


def _clear_cancellation_requests() -> None:
    current = asyncio.current_task()
    if current is not None:
        while current.cancelling():
            current.uncancel()


async def _spawn_bash(
    *,
    bash_path: str,
    command: str,
    working_directory: str,
    environment: Mapping[str, str] | None,
) -> asyncio.subprocess.Process:
    spawn_options: dict[str, Any] = {}
    if os.name == "posix":
        spawn_options["start_new_session"] = True
    elif os.name == "nt":  # pragma: no cover - exercised by Windows CI
        spawn_options["creationflags"] = _WINDOWS_CREATE_NEW_PROCESS_GROUP
    try:
        return await asyncio.create_subprocess_exec(
            bash_path,
            "--noprofile",
            "--norc",
            "-c",
            command,
            cwd=working_directory,
            env=None if environment is None else dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **spawn_options,
        )
    except (OSError, ValueError) as exc:
        raise CommandSpawnFailed from exc


async def _wait_reason(
    state: _RunningProcess,
    deadline: float,
    cancelled: Callable[[], bool],
) -> Literal["exit", "timeout", "cancelled"]:
    while True:
        if cancelled():
            return "cancelled"
        if state.process.returncode is not None or state.wait_task.done():
            return "exit"
        remaining = deadline - monotonic()
        if remaining <= 0:
            return "timeout"
        await asyncio.wait(
            (state.wait_task,),
            timeout=min(_CANCEL_POLL_SECONDS, remaining),
        )


def _capture_process(
    process: asyncio.subprocess.Process,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
) -> _RunningProcess:
    try:
        return _RunningProcess.create(process, max_stdout_bytes, max_stderr_bytes)
    except RuntimeError as exc:  # pragma: no cover - asyncio guarantees PIPE creation
        with suppress(ProcessLookupError):
            process.kill()
        raise CommandExecutionFailed from exc


async def _spawn_owned_bash(
    *,
    bash_path: str,
    command: str,
    working_directory: str,
    environment: Mapping[str, str] | None,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
) -> asyncio.subprocess.Process:
    spawn_task = asyncio.create_task(
        _spawn_bash(
            bash_path=bash_path,
            command=command,
            working_directory=working_directory,
            environment=environment,
        )
    )
    try:
        return await asyncio.shield(spawn_task)
    except asyncio.CancelledError as cancellation:
        _clear_cancellation_requests()
        try:
            process = await _await_task_resisting_cancellation(spawn_task)
            state = _capture_process(process, max_stdout_bytes, max_stderr_bytes)
        except (CommandExecutionFailed, CommandSpawnFailed):
            _clear_cancellation_requests()
            raise cancellation from None
        await _settle_after_task_cancellation(state)
        _clear_cancellation_requests()
        raise cancellation


async def run_bash(
    *,
    bash_path: str,
    command: str,
    working_directory: str,
    environment: Mapping[str, str] | None,
    timeout_seconds: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    terminate_grace_seconds: float,
    cancelled: Callable[[], bool],
) -> CommandOutcome:
    """Run Bash and settle its root process, pipes, and managed descendants."""

    if cancelled():
        raise CommandCancelled
    started = monotonic()
    process = await _spawn_owned_bash(
        bash_path=bash_path,
        command=command,
        working_directory=working_directory,
        environment=environment,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )
    state = _capture_process(process, max_stdout_bytes, max_stderr_bytes)
    deadline = monotonic() + timeout_seconds
    settled = False
    try:
        reason = await _wait_reason(state, deadline, cancelled)
        stdout, stderr = await state.finish(
            terminate_grace_seconds=terminate_grace_seconds,
            force=False,
        )
        if reason == "cancelled":
            settled = True
            raise CommandCancelled
        exit_code = state.process.returncode if reason == "exit" else None
        if reason == "exit" and exit_code is None:
            raise CommandExecutionFailed("Bash root process did not report an exit code")
        settled = True
        return CommandOutcome(
            reason,
            exit_code,
            stdout,
            stderr,
            max(0, int((monotonic() - started) * 1_000)),
        )
    except asyncio.CancelledError:
        _clear_cancellation_requests()
        await _settle_after_task_cancellation(state)
        _clear_cancellation_requests()
        raise
    except (OSError, RuntimeError) as exc:
        if not settled:
            await _settle_after_task_cancellation(state)
        raise CommandExecutionFailed from exc
    except BaseException:
        if not settled:
            await _settle_after_task_cancellation(state)
        raise
