"""Shared filesystem boundaries and result helpers."""

from __future__ import annotations

import asyncio
import ctypes
import errno
import math
import os
import stat
import sys
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from fnmatch import fnmatchcase
from importlib import import_module
from os import PathLike
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Event
from time import monotonic
from typing import Any, TypeAlias, TypeVar, cast

from jharness.kernel import ContentPart, SettledResult, ToolFailure, ToolResult, ToolSuccess

PathInput: TypeAlias = str | PathLike[str]

_T = TypeVar("_T")

DEFAULT_EXCLUDED_DIRECTORY_NAMES = (
    ".bzr",
    ".git",
    ".hg",
    ".jj",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".sl",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
)


class FilesystemFailure(Exception):
    """One stable, model-visible filesystem failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class OperationCancelled(Exception):
    """Internal cooperative-cancellation signal."""


@dataclass(frozen=True, slots=True)
class DirectoryEntry:
    """The directory-entry facts needed by the preset walkers."""

    name: str
    directory: bool

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        del follow_symlinks
        return self.directory


@dataclass(slots=True)
class SearchBudget:
    """Host-owned time and work limits for one filesystem search."""

    cancelled: Callable[[], bool]
    deadline: float
    max_entries: int
    max_bytes: int | None = None
    entries: int = 0
    bytes_read: int = 0

    @classmethod
    def create(
        cls,
        cancelled: Callable[[], bool],
        max_seconds: float,
        max_entries: int,
        max_bytes: int | None = None,
    ) -> SearchBudget:
        return cls(cancelled, monotonic() + max_seconds, max_entries, max_bytes)

    def checkpoint(self) -> None:
        check_cancelled(self.cancelled)
        if monotonic() >= self.deadline:
            raise FilesystemFailure(
                "search_timeout",
                "Search exceeded the configured time limit.",
            )

    def consume_entry(self) -> None:
        self.entries += 1
        if self.entries > self.max_entries:
            raise FilesystemFailure(
                "search_budget_exceeded",
                "Search exceeded the configured entry limit.",
            )
        self.checkpoint()

    def consume_bytes(self, size: int) -> None:
        self.bytes_read += size
        if self.max_bytes is not None and self.bytes_read > self.max_bytes:
            raise FilesystemFailure(
                "search_budget_exceeded",
                "Search exceeded the configured byte limit.",
            )
        self.checkpoint()

    def operation_timeout(self, maximum: float) -> float:
        self.checkpoint()
        return max(1e-9, min(maximum, self.deadline - monotonic()))

    def remaining_byte_limit(self, maximum: int) -> int:
        self.checkpoint()
        if self.max_bytes is None:
            return maximum
        return min(maximum, max(0, self.max_bytes - self.bytes_read))


@dataclass(frozen=True, slots=True)
class Workspace:
    """A resolved Host-owned filesystem boundary."""

    root: Path
    identity: tuple[int, int]

    @classmethod
    def create(cls, root: PathInput) -> Workspace:
        try:
            resolved = Path(root).expanduser().resolve(strict=True)
            status = resolved.stat()
        except OSError as exc:
            message = f"workspace root does not exist or cannot be resolved: {root!s}"
            raise ValueError(message) from exc
        if not stat.S_ISDIR(status.st_mode):
            raise ValueError(f"workspace root is not a directory: {resolved}")
        return cls(resolved, (status.st_dev, status.st_ino))

    def resolve(self, value: str) -> Path:
        candidate = Path(value).expanduser()
        if candidate.is_absolute() and not candidate.is_relative_to(self.root):
            raise FilesystemFailure(
                "path_outside_workspace",
                f"Path is outside the configured workspace: {value}",
            )
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FilesystemFailure("path_not_found", f"Path does not exist: {value}") from exc
        except OSError as exc:
            raise FilesystemFailure("filesystem_error", f"Cannot resolve path: {value}") from exc
        if not resolved.is_relative_to(self.root):
            raise FilesystemFailure(
                "path_outside_workspace",
                f"Path is outside the configured workspace: {value}",
            )
        return resolved

    def file(self, value: str) -> Path:
        resolved = self.resolve(value)
        if not resolved.is_file():
            raise FilesystemFailure("not_a_file", f"Path is not a file: {value}")
        return resolved

    def directory(self, value: str) -> Path:
        resolved = self.resolve(value)
        if not resolved.is_dir():
            raise FilesystemFailure("not_a_directory", f"Path is not a directory: {value}")
        return resolved

    def display(self, path: Path) -> str:
        relative = path.relative_to(self.root)
        return "." if not relative.parts else relative.as_posix()

    def safe_match(self, path: Path) -> tuple[Path, str] | None:
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            return None
        if not resolved.is_file() or not resolved.is_relative_to(self.root):
            return None
        return resolved, path.relative_to(self.root).as_posix()


def positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def positive_float(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a positive finite number")
    return float(value)


def excluded_names(values: Iterable[str]) -> frozenset[str]:
    names = frozenset(values)
    if any(
        not name or "/" in name or "\\" in name or Path(name).name != name or name in {".", ".."}
        for name in names
    ):
        raise ValueError("excluded directory names must be individual path names")
    return frozenset(name.casefold() for name in names)


def is_excluded(path: str, names: frozenset[str]) -> bool:
    return any(part.casefold() in names for part in PurePosixPath(path).parts)


def is_in_excluded_directory(
    path: str,
    names: frozenset[str],
    *,
    is_file: bool,
) -> bool:
    parts = PurePosixPath(path).parts
    directory_parts = parts[:-1] if is_file else parts
    return any(part.casefold() in names for part in directory_parts)


def validate_glob_pattern(
    pattern: str,
    max_chars: int = 4_096,
    max_components: int = 256,
) -> None:
    posix_path = PurePosixPath(pattern)
    windows_path = PureWindowsPath(pattern)
    if (
        not pattern
        or len(pattern) > max_chars
        or len(posix_path.parts) > max_components
        or "\x00" in pattern
        or "\\" in pattern
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or ".." in posix_path.parts
    ):
        raise FilesystemFailure(
            "invalid_glob_pattern",
            "Glob pattern is invalid or exceeds the configured complexity limit.",
        )


def glob_matches(path: str, pattern: str) -> bool:
    path_parts = PurePosixPath(path).parts
    pattern_parts = PurePosixPath(pattern).parts
    reachable = [True, *([False] * len(path_parts))]
    for component in pattern_parts:
        next_reachable = [False] * (len(path_parts) + 1)
        if component == "**":
            seen_reachable = False
            for index, is_reachable in enumerate(reachable):
                seen_reachable = seen_reachable or is_reachable
                next_reachable[index] = seen_reachable
        else:
            for index, is_reachable in enumerate(reachable[:-1]):
                if is_reachable and fnmatchcase(path_parts[index], component):
                    next_reachable[index + 1] = True
        reachable = next_reachable
        if not any(reachable):
            return False
    return reachable[-1]


def glob_filter_matches(path: str, pattern: str) -> bool:
    candidate = PurePosixPath(path)
    return glob_matches(path, pattern) or candidate.match(pattern)


def check_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise OperationCancelled


def _open_readonly(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def windows_kernel32() -> Any:  # pragma: no cover - Windows only
    return ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined, no-any-return]


def _windows_last_error() -> int:  # pragma: no cover - Windows only
    return ctypes.get_last_error()  # type: ignore[attr-defined, no-any-return]


def _windows_set_last_error(value: int) -> None:  # pragma: no cover - Windows only
    ctypes.set_last_error(value)  # type: ignore[attr-defined]


def windows_error(value: int | None = None) -> OSError:  # pragma: no cover - Windows only
    code = _windows_last_error() if value is None else value
    return ctypes.WinError(code)  # type: ignore[attr-defined, no-any-return]


def _windows_opened_file_path(descriptor: int) -> Path:  # pragma: no cover
    import msvcrt

    handle = cast(int, msvcrt.get_osfhandle(descriptor))  # type: ignore[attr-defined]
    return windows_final_path(handle)


def windows_final_path(handle: int) -> Path:  # pragma: no cover
    from ctypes import wintypes

    kernel32 = windows_kernel32()
    get_final_path = kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = (
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    )
    get_final_path.restype = wintypes.DWORD
    required = get_final_path(handle, None, 0, 0)
    if required == 0:
        raise windows_error()
    buffer = ctypes.create_unicode_buffer(required + 1)
    written = get_final_path(handle, buffer, len(buffer), 0)
    if written == 0 or written >= len(buffer):
        raise windows_error()
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)


def _windows_parse_directory_buffer(
    buffer: ctypes.Array[ctypes.c_char],
) -> Iterator[DirectoryEntry]:  # pragma: no cover
    from ctypes import wintypes

    class FileIdBothDirectoryInfo(ctypes.Structure):
        _fields_ = [
            ("next_entry_offset", wintypes.DWORD),
            ("file_index", wintypes.DWORD),
            ("creation_time", ctypes.c_longlong),
            ("last_access_time", ctypes.c_longlong),
            ("last_write_time", ctypes.c_longlong),
            ("change_time", ctypes.c_longlong),
            ("end_of_file", ctypes.c_longlong),
            ("allocation_size", ctypes.c_longlong),
            ("file_attributes", wintypes.DWORD),
            ("file_name_length", wintypes.DWORD),
            ("ea_size", wintypes.DWORD),
            ("short_name_length", ctypes.c_byte),
            ("short_name", wintypes.WCHAR * 12),
            ("file_id", ctypes.c_longlong),
            ("file_name", wintypes.WCHAR * 1),
        ]

    offset = 0
    while True:
        entry = FileIdBothDirectoryInfo.from_buffer(buffer, offset)
        name_bytes = int(entry.file_name_length)
        name_offset = offset + FileIdBothDirectoryInfo.file_name.offset
        if name_bytes % 2 or name_offset + name_bytes > len(buffer):
            raise OSError(errno.EIO, "invalid directory information returned by Windows")
        name = ctypes.wstring_at(
            ctypes.addressof(buffer) + name_offset,
            name_bytes // ctypes.sizeof(wintypes.WCHAR),
        )
        attributes = int(entry.file_attributes)
        if name not in {".", ".."}:
            yield DirectoryEntry(
                name,
                bool(attributes & 0x0010) and not bool(attributes & 0x0400),
            )
        next_offset = int(entry.next_entry_offset)
        if next_offset == 0:
            return
        if next_offset % 8 or offset + next_offset >= len(buffer):
            raise OSError(errno.EIO, "invalid directory information offset returned by Windows")
        offset += next_offset


def _windows_directory_entries(handle: int) -> Iterator[DirectoryEntry]:  # pragma: no cover
    from ctypes import wintypes

    kernel32 = windows_kernel32()
    get_information = kernel32.GetFileInformationByHandleEx
    get_information.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    get_information.restype = wintypes.BOOL
    buffer = ctypes.create_string_buffer(64 * 1024)
    information_class = 11
    while True:
        _windows_set_last_error(0)
        if not get_information(
            handle,
            information_class,
            ctypes.byref(buffer),
            ctypes.sizeof(buffer),
        ):
            error = _windows_last_error()
            if error == 18:
                return
            raise windows_error(error)
        yield from _windows_parse_directory_buffer(buffer)
        information_class = 10


@contextmanager
def _windows_secure_scandir(
    workspace: Workspace,
    directory: Path,
) -> Generator[Iterator[DirectoryEntry], None, None]:  # pragma: no cover
    from ctypes import wintypes

    class FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD), ("reparse_tag", wintypes.DWORD)]

    kernel32 = windows_kernel32()
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    raw_handle = create_file(
        str(directory),
        0x0001,
        0x0001 | 0x0002,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if raw_handle is None or raw_handle == invalid_handle:
        raise windows_error()
    handle = int(raw_handle)
    try:
        get_information = kernel32.GetFileInformationByHandleEx
        get_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        get_information.restype = wintypes.BOOL
        information = FileAttributeTagInfo()
        if not get_information(
            handle,
            9,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            raise windows_error()
        if information.attributes & 0x0400:
            raise OSError(errno.ELOOP, "refusing to traverse a reparse-point directory")
        _validate_opened_path(workspace, directory, windows_final_path(handle))
        yield _windows_directory_entries(handle)
    finally:
        if not close_handle(handle):
            raise windows_error()


def _darwin_opened_file_path(descriptor: int) -> Path:  # pragma: no cover
    fcntl_module = import_module("fcntl")
    fcntl_call = cast(
        Callable[[int, int, bytes], bytes | int],
        fcntl_module.fcntl,
    )
    # CPython copies ``fcntl`` byte arguments through a fixed 1024-byte buffer.
    # Darwin's F_GETPATH uses MAXPATHLEN (also 1024), so a larger argument is
    # rejected before the system call is made.
    value = fcntl_call(descriptor, 50, b"\x00" * 1024)
    if not isinstance(value, bytes):
        raise OSError(errno.ENOTSUP, "F_GETPATH did not return a path")
    return Path(value.split(b"\x00", 1)[0].decode())


def opened_file_path(descriptor: int) -> Path:  # pragma: no cover
    if os.name == "nt":
        return _windows_opened_file_path(descriptor)
    if sys.platform == "darwin":
        return _darwin_opened_file_path(descriptor)
    for descriptor_root in ("/proc/self/fd", "/dev/fd"):
        try:
            value = os.readlink(f"{descriptor_root}/{descriptor}")
        except OSError:
            continue
        deleted_suffix = " (deleted)"
        if value.endswith(deleted_suffix):
            value = value[: -len(deleted_suffix)]
        return Path(value)
    raise OSError(errno.ENOTSUP, "cannot determine the opened file path on this platform")


def _validate_opened_path(workspace: Workspace, expected: Path, opened: Path) -> None:
    if not opened.is_absolute() or not opened.is_relative_to(workspace.root) or opened != expected:
        raise FilesystemFailure(
            "path_outside_workspace",
            "Opened path does not match its validated workspace path.",
        )


@contextmanager
def _posix_secure_scandir(
    workspace: Workspace,
    directory: Path,
) -> Generator[Iterator[DirectoryEntry], None, None]:  # pragma: no cover
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(directory, flags)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(errno.ENOTDIR, "opened path is not a directory")
        _validate_opened_path(workspace, directory, opened_file_path(descriptor))
        with os.scandir(descriptor) as entries:
            yield (
                DirectoryEntry(entry.name, entry.is_dir(follow_symlinks=False)) for entry in entries
            )
    finally:
        os.close(descriptor)


_secure_scandir_implementations = {  # pragma: no cover
    "nt": _windows_secure_scandir,
    "posix": _posix_secure_scandir,
}
_secure_scandir_impl = _secure_scandir_implementations[os.name]  # pragma: no cover


@contextmanager
def secure_scandir(
    workspace: Workspace,
    directory: Path,
) -> Generator[Iterator[DirectoryEntry], None, None]:
    with _secure_scandir_impl(workspace, directory) as entries:
        yield entries


def read_bytes_bounded(
    workspace: Workspace,
    path: Path,
    max_bytes: int,
    checkpoint: Callable[[], None] | None = None,
) -> tuple[bytes, bool]:
    descriptor = _open_readonly(path)
    try:
        if checkpoint is not None:
            checkpoint()
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FilesystemFailure("not_a_file", "Opened path is not a regular file.")
        opened_path = opened_file_path(descriptor)
        _validate_opened_path(workspace, path, opened_path)
        remaining = max_bytes + 1
        chunks: list[bytes] = []
        while remaining:
            if checkpoint is not None:
                checkpoint()
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        return data, len(data) > max_bytes
    finally:
        os.close(descriptor)


async def run_blocking(
    function: Callable[[Callable[[], bool]], _T],
    cancelled: Callable[[], bool],
) -> _T:
    """Run blocking I/O and settle the owned worker before cancellation escapes."""

    local_cancelled = Event()

    def invoke() -> _T:
        return function(lambda: local_cancelled.is_set() or cancelled())

    task = asyncio.create_task(asyncio.to_thread(invoke))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        local_cancelled.set()
        with suppress(Exception):
            await task
        raise


def success(text: str, structured_content: object) -> ToolResult:
    return SettledResult(
        ToolSuccess((ContentPart.text_part(text),), structured_content=structured_content)
    )


def failure(error: FilesystemFailure) -> ToolResult:
    return SettledResult(ToolFailure.from_error(error.code, str(error)))


def cancelled(name: str) -> ToolResult:
    return SettledResult(ToolFailure.from_error("cancelled", f"{name} was cancelled."))


def nullable_output(schema: Mapping[str, object]) -> dict[str, object]:
    return {"anyOf": [schema, {"type": "null"}]}
