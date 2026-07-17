"""CAS-guarded, workspace-scoped text mutation primitives."""

from __future__ import annotations

import asyncio
import ctypes
import errno
import os
import secrets
import stat
from _thread import LockType
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from threading import Event, Lock
from typing import Final, TypeVar, cast

from jharness.tools.filesystem._common import (
    FilesystemFailure,
    OperationCancelled,
    Workspace,
    check_cancelled,
    opened_file_path,
    windows_error,
    windows_final_path,
    windows_kernel32,
)
from jharness.tools.filesystem._content import digest_bytes

_UTF8_BOM: Final = b"\xef\xbb\xbf"
_WRITE_CHUNK_BYTES: Final = 64 * 1024
_LOCK_WAIT_SECONDS: Final = 0.05
_T = TypeVar("_T")
FileIdentity = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class TextSnapshot:
    """One bounded, securely opened UTF-8 file snapshot."""

    raw: bytes
    original_text: str
    text: str
    digest: str
    identity: FileIdentity
    bom: bool
    newline: str


@dataclass(frozen=True, slots=True)
class MutationTarget:
    """A lexical target path below one workspace root."""

    workspace: Workspace
    path: Path
    relative: Path

    @property
    def display(self) -> str:
        return self.relative.as_posix()


@dataclass(frozen=True, slots=True)
class MutationSession:
    """A target whose parent path is pinned for one mutation."""

    target: MutationTarget
    parent_descriptor: int | None

    @property
    def display(self) -> str:
        return self.target.display


@dataclass(frozen=True, slots=True)
class _CurrentFile:
    raw: bytes
    digest: str
    identity: FileIdentity
    mode: int


@dataclass(slots=True)
class _PathLock:
    lock: LockType = field(default_factory=Lock)
    users: int = 0


_PATH_LOCKS: dict[str, _PathLock] = {}
_PATH_LOCKS_GUARD = Lock()


def resolve_mutation_target(workspace: Workspace, value: str) -> MutationTarget:
    """Resolve a model path lexically without following filesystem links."""

    if not value or "\x00" in value:
        raise FilesystemFailure("invalid_path", "File path must be a non-empty text path.")
    try:
        candidate = Path(value).expanduser()
        if candidate.drive and not candidate.is_absolute():
            raise FilesystemFailure(
                "path_outside_workspace",
                f"Path is outside the configured workspace: {value}",
            )
        joined = candidate if candidate.is_absolute() else workspace.root / candidate
        absolute = Path(os.path.abspath(joined))
    except FilesystemFailure:
        raise
    except (OSError, RuntimeError) as exc:
        raise FilesystemFailure("invalid_path", f"File path is invalid: {value}") from exc
    if not absolute.is_relative_to(workspace.root):
        raise FilesystemFailure(
            "path_outside_workspace",
            f"Path is outside the configured workspace: {value}",
        )
    relative = absolute.relative_to(workspace.root)
    if not relative.parts:
        raise FilesystemFailure("not_a_file", f"Path is not a file: {value}")
    return MutationTarget(workspace, absolute, relative)


@contextmanager
def mutation_session(
    workspace: Workspace,
    value: str,
    cancelled: Callable[[], bool],
) -> Generator[MutationSession, None, None]:
    """Lock a target and pin its safe parent path for a complete mutation."""

    target = resolve_mutation_target(workspace, value)
    with _target_lock(target, cancelled), _open_parent(target) as parent_descriptor:
        check_cancelled(cancelled)
        yield MutationSession(target, parent_descriptor)


def read_text_snapshot(
    session: MutationSession,
    max_file_bytes: int,
    cancelled: Callable[[], bool],
) -> TextSnapshot:
    """Read an existing target as bounded UTF-8 without releasing its path lock."""

    current = _read_current(session, max_file_bytes, cancelled, missing_code="path_not_found")
    if b"\x00" in current.raw:
        raise FilesystemFailure(
            "binary_file",
            f"File appears to be binary: {session.display}",
        )
    try:
        decoded = current.raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise FilesystemFailure(
            "invalid_utf8",
            f"File is not valid UTF-8 text: {session.display}",
        ) from exc
    return TextSnapshot(
        current.raw,
        decoded,
        normalize_newlines(decoded),
        current.digest,
        current.identity,
        current.raw.startswith(_UTF8_BOM),
        detect_newline(decoded),
    )


def apply_text_edit(
    snapshot: TextSnapshot,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool,
    max_file_bytes: int,
    cancelled: Callable[[], bool],
) -> tuple[bytes, int]:
    """Apply exact normalized-text matches while preserving untouched line endings."""

    old, new, matches = _prepare_text_edit(
        snapshot,
        old_string,
        new_string,
        replace_all=replace_all,
        max_file_bytes=max_file_bytes,
    )
    spans = _iter_match_spans(snapshot.text, old)
    selected = spans if replace_all else islice(spans, 1)
    parts: list[str] = []
    original_cursor = 0
    normalized_cursor = 0
    output_cursor = 0
    projected_characters = 0
    replacements: dict[str, str] = {}
    for replacement_index, (start, end) in enumerate(selected):
        if replacement_index % 1024 == 0:
            check_cancelled(cancelled)
        original_start = _advance_original(
            snapshot.original_text,
            original_cursor,
            start - normalized_cursor,
            cancelled,
        )
        original_end = _advance_original(
            snapshot.original_text,
            original_start,
            end - start,
            cancelled,
        )
        matched = snapshot.original_text[original_start:original_end]
        newline = detect_newline(matched) if "\n" in old else snapshot.newline
        replacement = replacements.get(newline)
        if replacement is None:
            replacement = new.replace("\n", newline)
            replacements[newline] = replacement
        untouched = snapshot.original_text[output_cursor:original_start]
        projected_characters += len(untouched) + len(replacement)
        if projected_characters > max_file_bytes:
            raise _content_too_large(max_file_bytes)
        parts.extend((untouched, replacement))
        original_cursor = original_end
        normalized_cursor = end
        output_cursor = original_end
    tail = snapshot.original_text[output_cursor:]
    if projected_characters + len(tail) > max_file_bytes:
        raise _content_too_large(max_file_bytes)
    parts.append(tail)
    check_cancelled(cancelled)
    updated = "".join(parts)
    return encode_text_content(
        updated,
        max_file_bytes=max_file_bytes,
        bom=snapshot.bom,
    ), matches if replace_all else 1


def _prepare_text_edit(
    snapshot: TextSnapshot,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool,
    max_file_bytes: int,
) -> tuple[str, str, int]:
    if not old_string:
        raise FilesystemFailure("invalid_input", "old_string must not be empty.")
    if len(old_string) > max_file_bytes or len(new_string) > max_file_bytes:
        raise _content_too_large(max_file_bytes)
    old = normalize_newlines(old_string)
    new = normalize_newlines(new_string)
    if old == new:
        raise FilesystemFailure("no_changes", "old_string and new_string must differ.")
    matches = snapshot.text.count(old)
    if matches == 0:
        raise FilesystemFailure("old_string_not_found", "old_string was not found in the file.")
    if matches > 1 and not replace_all:
        raise FilesystemFailure(
            "old_string_not_unique",
            f"old_string matched {matches} locations; set replace_all=true or add context.",
        )
    return old, new, matches


def encode_text_content(
    value: str,
    *,
    max_file_bytes: int,
    bom: bool = False,
) -> bytes:
    """Encode bounded UTF-8 mutation content and reject binary text."""

    if len(value) > max_file_bytes:
        raise _content_too_large(max_file_bytes)
    if "\x00" in value:
        raise FilesystemFailure("binary_content", "File content cannot contain NUL bytes.")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise FilesystemFailure("invalid_utf8", "File content is not valid UTF-8 text.") from exc
    result = (_UTF8_BOM if bom else b"") + encoded
    if len(result) > max_file_bytes:
        raise _content_too_large(max_file_bytes)
    return result


def atomic_write(
    session: MutationSession,
    data: bytes,
    *,
    expected_sha256: str | None,
    expected_identity: FileIdentity | None = None,
    max_file_bytes: int,
    cancelled: Callable[[], bool],
) -> str | None:
    """CAS-write bytes with a same-directory temporary and one atomic commit."""

    if len(data) > max_file_bytes:
        raise _content_too_large(max_file_bytes)
    check_cancelled(cancelled)
    previous = _verify_precondition(
        session,
        expected_sha256,
        expected_identity,
        max_file_bytes,
        cancelled,
    )
    if previous is not None and previous.raw == data:
        return previous.digest
    descriptor, temporary_name, temporary_inode = _open_temporary(
        session,
        previous.mode if previous is not None else 0o666,
        preserve_mode=previous is not None,
    )
    committed = False
    try:
        _write_all(descriptor, data, cancelled)
        os.fsync(descriptor)
        temporary_identity = _identity(os.fstat(descriptor))
        os.close(descriptor)
        descriptor = -1
        check_cancelled(cancelled)
        _verify_precondition(
            session,
            expected_sha256,
            previous.identity if previous is not None else None,
            max_file_bytes,
            cancelled,
        )
        check_cancelled(cancelled)
        _verify_temporary(session, temporary_name, temporary_identity)
        _commit_temporary(
            session,
            temporary_name,
            create=previous is None,
            expected_identity=temporary_identity,
        )
        committed = True
    finally:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        if not committed:
            with suppress(OSError):
                _remove_temporary(session, temporary_name, temporary_inode)
    _sync_parent_best_effort(session)
    return previous.digest if previous is not None else None


async def run_mutation(
    function: Callable[[Callable[[], bool]], _T],
    cancelled: Callable[[], bool],
) -> _T:
    """Settle a mutation worker so a committed write is never reported as cancelled."""

    local_cancelled = Event()

    def invoke() -> _T:
        return function(lambda: local_cancelled.is_set() or cancelled())

    task = asyncio.create_task(asyncio.to_thread(invoke))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        local_cancelled.set()
        try:
            return await task
        except OperationCancelled:
            raise cancellation from None
        except BaseException:
            raise cancellation from None


def normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def detect_newline(value: str) -> str:
    crlf = value.count("\r\n")
    without_crlf = value.replace("\r\n", "")
    lf = without_crlf.count("\n")
    cr = without_crlf.count("\r")
    if crlf > lf and crlf >= cr:
        return "\r\n"
    if cr > lf:
        return "\r"
    return "\n"


def _iter_match_spans(value: str, needle: str) -> Generator[tuple[int, int], None, None]:
    start = 0
    while (match := value.find(needle, start)) >= 0:
        end = match + len(needle)
        yield match, end
        start = end


def _advance_original(
    value: str,
    index: int,
    characters: int,
    cancelled: Callable[[], bool],
) -> int:
    advanced = 0
    while advanced < characters:
        if advanced % 4096 == 0:
            check_cancelled(cancelled)
        if value[index] == "\r" and index + 1 < len(value) and value[index + 1] == "\n":
            index += 2
        else:
            index += 1
        advanced += 1
    return index


def _content_too_large(max_file_bytes: int) -> FilesystemFailure:
    return FilesystemFailure(
        "content_too_large",
        f"File content exceeds the configured {max_file_bytes}-byte limit.",
    )


@contextmanager
def _target_lock(
    target: MutationTarget,
    cancelled: Callable[[], bool],
) -> Generator[None, None, None]:
    key = os.path.normcase(str(target.path))
    with _PATH_LOCKS_GUARD:
        entry = _PATH_LOCKS.setdefault(key, _PathLock())
        entry.users += 1
    acquired = False
    try:
        check_cancelled(cancelled)
        while not entry.lock.acquire(timeout=_LOCK_WAIT_SECONDS):
            check_cancelled(cancelled)
        acquired = True
        check_cancelled(cancelled)
        yield
    finally:
        if acquired:
            entry.lock.release()
        with _PATH_LOCKS_GUARD:
            entry.users -= 1
            if entry.users == 0:
                del _PATH_LOCKS[key]


@contextmanager
def _open_parent(target: MutationTarget) -> Generator[int | None, None, None]:
    implementation = _OPEN_PARENT_IMPLEMENTATIONS[os.name]
    with implementation(target) as descriptor:
        yield descriptor


@contextmanager
def _posix_open_parent(  # pragma: no cover - exercised by POSIX CI
    target: MutationTarget,
) -> Generator[int | None, None, None]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    current = target.workspace.root
    try:
        root_descriptor = os.open(current, flags)
        descriptors.append(root_descriptor)
        status = os.fstat(root_descriptor)
        if (status.st_dev, status.st_ino) != target.workspace.identity:
            raise FilesystemFailure("unsafe_path", "Workspace root changed during mutation.")
        _validate_directory_descriptor(target.workspace, current, root_descriptor)
        for component in target.relative.parent.parts:
            current /= component
            try:
                descriptor = os.open(component, flags, dir_fd=descriptors[-1])
            except OSError as exc:
                raise _parent_failure(target, exc) from exc
            descriptors.append(descriptor)
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise FilesystemFailure(
                    "not_a_directory",
                    f"Parent path is not a directory: {target.display}",
                )
            _validate_directory_descriptor(target.workspace, current, descriptor)
        yield descriptors[-1]
    finally:
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)


@contextmanager
def _windows_open_parent(  # pragma: no cover - native Windows handle validation
    target: MutationTarget,
) -> Generator[int | None, None, None]:
    handles: list[int] = []
    current = target.workspace.root
    try:
        paths = [current]
        for component in target.relative.parent.parts:
            current /= component
            paths.append(current)
        for index, path in enumerate(paths):
            try:
                handle = _windows_open_directory(path)
            except OSError as exc:
                raise _parent_failure(target, exc) from exc
            handles.append(handle)
            if index == 0:
                status = path.stat()
                if (status.st_dev, status.st_ino) != target.workspace.identity:
                    raise FilesystemFailure(
                        "unsafe_path",
                        "Workspace root changed during mutation.",
                    )
        yield None
    finally:
        for handle in reversed(handles):
            _windows_close_handle(handle)


def _validate_directory_descriptor(  # pragma: no cover - exercised by POSIX CI
    workspace: Workspace,
    expected: Path,
    descriptor: int,
) -> None:
    opened = opened_file_path(descriptor)
    if opened != expected or not opened.is_relative_to(workspace.root):
        raise FilesystemFailure("unsafe_path", "Parent path changed during mutation.")


def _windows_open_directory(path: Path) -> int:  # pragma: no cover - Windows only
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
    raw_handle = create_file(
        str(path),
        0x0080,
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
        if not information.attributes & 0x0010:
            raise OSError(errno.ENOTDIR, "opened parent is not a directory")
        if information.attributes & 0x0400:
            raise OSError(errno.ELOOP, "refusing to traverse a reparse point")
        opened = _opened_file_path_from_windows_handle(handle)
        if opened != path:
            raise OSError(errno.ELOOP, "opened parent path changed")
        return handle
    except BaseException:
        _windows_close_handle(handle)
        raise


def _opened_file_path_from_windows_handle(  # pragma: no cover - Windows only
    handle: int,
) -> Path:
    return windows_final_path(handle)


def _windows_close_handle(handle: int) -> None:  # pragma: no cover - Windows only
    from ctypes import wintypes

    kernel32 = windows_kernel32()
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    close_handle(handle)


def _parent_failure(target: MutationTarget, error: OSError) -> FilesystemFailure:
    if error.errno in {errno.ENOENT} or getattr(error, "winerror", None) in {2, 3}:
        return FilesystemFailure(
            "path_not_found",
            f"Parent directory does not exist: {target.display}",
        )
    if error.errno in {errno.ENOTDIR} or getattr(error, "winerror", None) == 267:
        return FilesystemFailure(
            "not_a_directory",
            f"Parent path is not a directory: {target.display}",
        )
    if error.errno in {errno.ELOOP} or getattr(error, "winerror", None) == 4390:
        return FilesystemFailure(
            "unsafe_path",
            f"Refusing to traverse a symbolic link or reparse point: {target.display}",
        )
    return FilesystemFailure(
        "filesystem_error",
        f"Cannot open parent directory: {target.display}",
    )


def _read_current(
    session: MutationSession,
    max_file_bytes: int,
    cancelled: Callable[[], bool],
    *,
    missing_code: str,
) -> _CurrentFile:
    check_cancelled(cancelled)
    descriptor = _open_existing_target(session, missing_code)
    try:
        opened_status = os.fstat(descriptor)
        if not stat.S_ISREG(opened_status.st_mode):
            raise FilesystemFailure("not_a_file", f"Path is not a file: {session.display}")
        _validate_target_descriptor(session, descriptor)
        raw = _read_descriptor(descriptor, max_file_bytes, cancelled, session.display)
        return _CurrentFile(
            raw,
            digest_bytes(raw),
            _identity(opened_status),
            stat.S_IMODE(opened_status.st_mode),
        )
    finally:
        os.close(descriptor)


def _open_existing_target(session: MutationSession, missing_code: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        if session.parent_descriptor is None:
            status = session.target.path.lstat()
            if _is_reparse(status):
                raise FilesystemFailure(
                    "unsafe_path",
                    f"Refusing to mutate a symbolic link or reparse point: {session.display}",
                )
            descriptor = os.open(session.target.path, flags)
            if _identity(os.fstat(descriptor)) != _identity(status):
                os.close(descriptor)
                raise FilesystemFailure(
                    "stale_file",
                    f"File changed during mutation: {session.display}",
                )
            return descriptor
        return os.open(
            session.target.path.name,
            flags,
            dir_fd=session.parent_descriptor,
        )  # pragma: no cover - exercised by POSIX CI
    except FileNotFoundError as exc:
        raise FilesystemFailure(
            missing_code,
            f"Path does not exist: {session.display}",
        ) from exc
    except FilesystemFailure:
        raise
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise FilesystemFailure(
                "unsafe_path",
                f"Refusing to mutate a symbolic link or reparse point: {session.display}",
            ) from exc
        raise


def _validate_target_descriptor(session: MutationSession, descriptor: int) -> None:
    opened = opened_file_path(descriptor)
    expected = session.target.path
    if opened != expected or not opened.is_relative_to(session.target.workspace.root):
        raise FilesystemFailure(
            "unsafe_path",
            f"Opened target does not match the requested file: {session.display}",
        )


def _read_descriptor(
    descriptor: int,
    max_file_bytes: int,
    cancelled: Callable[[], bool],
    display: str,
) -> bytes:
    remaining = max_file_bytes + 1
    chunks: list[bytes] = []
    while remaining:
        check_cancelled(cancelled)
        chunk = os.read(descriptor, min(remaining, 1024 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    if len(raw) > max_file_bytes:
        raise FilesystemFailure(
            "file_too_large",
            f"File exceeds the configured {max_file_bytes}-byte limit: {display}",
        )
    return raw


def _verify_precondition(
    session: MutationSession,
    expected_sha256: str | None,
    expected_identity: FileIdentity | None,
    max_file_bytes: int,
    cancelled: Callable[[], bool],
) -> _CurrentFile | None:
    if expected_sha256 is None:
        _require_missing(session)
        return None
    current = _read_current(session, max_file_bytes, cancelled, missing_code="stale_file")
    if current.digest != expected_sha256 or (
        expected_identity is not None and current.identity != expected_identity
    ):
        raise FilesystemFailure(
            "stale_file",
            f"File changed since it was read: {session.display}",
        )
    return current


def _require_missing(session: MutationSession) -> None:
    try:
        if session.parent_descriptor is None:
            status = session.target.path.lstat()
        else:
            status = os.stat(
                session.target.path.name,
                dir_fd=session.parent_descriptor,
                follow_symlinks=False,
            )  # pragma: no cover - exercised by POSIX CI
    except FileNotFoundError:
        return
    if _is_reparse(status) or stat.S_ISLNK(status.st_mode):
        raise FilesystemFailure(
            "unsafe_path",
            f"Refusing to mutate a symbolic link or reparse point: {session.display}",
        )
    raise FilesystemFailure(
        "stale_file",
        f"Path already exists: {session.display}",
    )


def _open_temporary(
    session: MutationSession,
    mode: int,
    *,
    preserve_mode: bool,
) -> tuple[int, str, tuple[int, int]]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for _ in range(32):
        name = f".{session.target.path.name}.jharness-{secrets.token_hex(8)}.tmp"
        try:
            if session.parent_descriptor is None:
                descriptor = os.open(session.target.path.parent / name, flags, mode)
            else:
                descriptor = os.open(
                    name,
                    flags,
                    mode,
                    dir_fd=session.parent_descriptor,
                )  # pragma: no cover - exercised by POSIX CI
        except FileExistsError:
            continue
        temporary_inode: tuple[int, int] | None = None
        try:
            status = os.fstat(descriptor)
            temporary_inode = (status.st_dev, status.st_ino)
            if not stat.S_ISREG(status.st_mode):
                raise OSError(errno.EINVAL, "temporary output is not a regular file")
            _validate_temporary_descriptor(session, name, descriptor)
            if preserve_mode and os.name != "nt":  # pragma: no cover - exercised by POSIX CI
                os.fchmod(descriptor, mode)
            return descriptor, name, temporary_inode
        except BaseException:
            with suppress(OSError):
                os.close(descriptor)
            if temporary_inode is not None:  # pragma: no branch - fstat establishes the identity
                with suppress(OSError):
                    _remove_temporary(session, name, temporary_inode)
            raise
    raise FilesystemFailure("filesystem_error", "Cannot allocate a temporary output file.")


def _validate_temporary_descriptor(
    session: MutationSession,
    name: str,
    descriptor: int,
) -> None:
    expected = session.target.path.parent / name
    opened = opened_file_path(descriptor)
    if opened != expected or not opened.is_relative_to(session.target.workspace.root):
        raise FilesystemFailure("unsafe_path", "Temporary output escaped the workspace.")


def _write_all(
    descriptor: int,
    data: bytes,
    cancelled: Callable[[], bool],
) -> None:
    offset = 0
    while offset < len(data):
        check_cancelled(cancelled)
        written = os.write(descriptor, data[offset : offset + _WRITE_CHUNK_BYTES])
        if written < 1:
            raise OSError(errno.EIO, "write returned no progress")
        offset += written
    check_cancelled(cancelled)


def _commit_temporary(
    session: MutationSession,
    name: str,
    *,
    create: bool,
    expected_identity: FileIdentity,
) -> None:
    try:
        if create:
            _commit_create(session, name, expected_identity)
        elif session.parent_descriptor is None:
            os.replace(session.target.path.parent / name, session.target.path)
        else:
            os.replace(
                name,
                session.target.path.name,
                src_dir_fd=session.parent_descriptor,
                dst_dir_fd=session.parent_descriptor,
            )  # pragma: no cover - exercised by POSIX CI
    except FileExistsError as exc:
        raise FilesystemFailure(
            "stale_file",
            f"Path was created during mutation: {session.display}",
        ) from exc


def _commit_create(
    session: MutationSession,
    name: str,
    expected_identity: FileIdentity,
) -> None:
    if session.parent_descriptor is None:
        os.rename(session.target.path.parent / name, session.target.path)
        return
    os.link(
        name,
        session.target.path.name,
        src_dir_fd=session.parent_descriptor,
        dst_dir_fd=session.parent_descriptor,
        follow_symlinks=False,
    )  # pragma: no cover - exercised by POSIX CI
    with suppress(OSError):  # pragma: no cover - exercised by POSIX CI
        _remove_temporary(  # pragma: no cover - exercised by POSIX CI
            session,
            name,
            expected_identity[:2],
        )


def _verify_temporary(
    session: MutationSession,
    name: str,
    expected_identity: FileIdentity,
) -> None:
    try:
        if session.parent_descriptor is None:
            status = (session.target.path.parent / name).lstat()
        else:
            status = os.stat(
                name,
                dir_fd=session.parent_descriptor,
                follow_symlinks=False,
            )  # pragma: no cover - exercised by POSIX CI
    except FileNotFoundError as exc:
        raise FilesystemFailure("unsafe_path", "Temporary output disappeared.") from exc
    if _identity(status) != expected_identity or not stat.S_ISREG(status.st_mode):
        raise FilesystemFailure("unsafe_path", "Temporary output changed before commit.")


def _remove_temporary(
    session: MutationSession,
    name: str,
    expected_inode: tuple[int, int],
) -> None:
    try:
        if session.parent_descriptor is None:
            path = session.target.path.parent / name
            status = path.lstat()
            if (status.st_dev, status.st_ino) == expected_inode:
                os.unlink(path)
        else:
            status = os.stat(
                name,
                dir_fd=session.parent_descriptor,
                follow_symlinks=False,
            )  # pragma: no cover - exercised by POSIX CI
            if (  # pragma: no cover - exercised by POSIX CI
                status.st_dev,
                status.st_ino,
            ) == expected_inode:
                os.unlink(
                    name,
                    dir_fd=session.parent_descriptor,
                )  # pragma: no cover - exercised by POSIX CI
    except FileNotFoundError:
        return


def _sync_parent_best_effort(session: MutationSession) -> None:
    if session.parent_descriptor is None:
        return
    with suppress(OSError):  # pragma: no cover - exercised by POSIX CI
        os.fsync(session.parent_descriptor)


def _identity(status: os.stat_result) -> FileIdentity:
    return (status.st_dev, status.st_ino, status.st_mtime_ns, status.st_size)


def _is_reparse(status: os.stat_result) -> bool:
    attributes = cast(int, getattr(status, "st_file_attributes", 0))
    reparse = cast(int, getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400))
    return stat.S_ISLNK(status.st_mode) or bool(attributes & reparse)


_OPEN_PARENT_IMPLEMENTATIONS = {
    "nt": _windows_open_parent,
    "posix": _posix_open_parent,
}
