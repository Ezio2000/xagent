"""Small explicit digest writer for immutable kernel content."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Protocol, cast

from jharness.kernel.messages import ArtifactRef, ContentPart, ErrorInfo, Message, TaskRef, ToolCall
from jharness.kernel.tools import ToolAccepted, ToolFailure, ToolOutcome, ToolWaiting


class _Hash(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def digest(self) -> bytes: ...


class DigestWriter:
    """Domain-separated, length-prefixed writer with explicit scalar types."""

    __slots__ = ("_hash",)

    def __init__(self, domain: str) -> None:
        self._hash: _Hash = sha256()
        self._chunk(b"d", domain.encode())

    def field(self, name: str) -> None:
        self._chunk(b"k", name.encode())

    def none(self) -> None:
        self._hash.update(b"n")

    def boolean(self, value: bool) -> None:
        self._hash.update(b"b1" if value else b"b0")

    def integer(self, value: int) -> None:
        self._chunk(b"i", str(value).encode())

    def number(self, value: float) -> None:
        self._chunk(b"f", value.hex().encode())

    def string(self, value: str) -> None:
        self._chunk(b"s", value.encode())

    def bytes(self, value: bytes) -> None:
        self._chunk(b"x", value)

    def sequence(self, length: int) -> None:
        self._chunk(b"q", str(length).encode())

    def mapping(self, length: int) -> None:
        self._chunk(b"m", str(length).encode())

    def json(self, value: object) -> None:
        """Write a validated JSON value with stable mapping order and scalar types."""

        if value is None:
            self.none()
        elif isinstance(value, bool):
            self.boolean(value)
        elif isinstance(value, int):
            self.integer(value)
        elif isinstance(value, float):
            self.number(value)
        elif isinstance(value, str):
            self.string(value)
        elif isinstance(value, Mapping):
            self._json_mapping(cast(Mapping[object, object], value))
        elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
            items = cast(Sequence[object], value)
            self.sequence(len(items))
            for item in items:
                self.json(item)
        else:
            raise TypeError(f"unsupported JSON digest value: {type(value).__qualname__}")

    def finish(self) -> bytes:
        return self._hash.digest()

    def _json_mapping(self, value: Mapping[object, object]) -> None:
        if any(not isinstance(key, str) for key in value):
            raise TypeError("JSON digest mapping keys must be strings")
        items = cast(Mapping[str, object], value)
        keys = sorted(items)
        self.mapping(len(keys))
        for key in keys:
            self.string(key)
            self.json(items[key])

    def _chunk(self, tag: bytes, value: bytes) -> None:
        self._hash.update(tag)
        self._hash.update(len(value).to_bytes(8, "big"))
        self._hash.update(value)


def empty_history_digest() -> bytes:
    return DigestWriter("jharness.kernel.history.empty.v0").finish()


def append_history_digest(previous: bytes, message: Message) -> bytes:
    writer = DigestWriter("jharness.kernel.history.append.v0")
    writer.field("previous")
    writer.bytes(previous)
    writer.field("message")
    write_message(writer, message)
    return writer.finish()


def empty_tool_call_suffix_digest() -> bytes:
    return DigestWriter("jharness.kernel.tool_calls.suffix.empty.v1").finish()


def prepend_tool_call_digest(suffix: bytes, call: ToolCall) -> bytes:
    writer = DigestWriter("jharness.kernel.tool_calls.suffix.prepend.v1")
    writer.field("call")
    write_tool_call(writer, call)
    writer.field("suffix")
    writer.bytes(suffix)
    return writer.finish()


def empty_call_id_suffix_digest() -> bytes:
    return DigestWriter("jharness.kernel.tool_call_ids.suffix.empty.v1").finish()


def prepend_call_id_digest(suffix: bytes, call_id: str) -> bytes:
    writer = DigestWriter("jharness.kernel.tool_call_ids.suffix.prepend.v1")
    writer.field("call_id")
    writer.string(call_id)
    writer.field("suffix")
    writer.bytes(suffix)
    return writer.finish()


def compose_call_id_digest(prefix: Sequence[str], suffix: bytes) -> bytes:
    digest = suffix
    for call_id in reversed(prefix):
        digest = prepend_call_id_digest(digest, call_id)
    return digest


def write_message(writer: DigestWriter, value: Message) -> None:
    writer.field("role")
    writer.string(value.role)
    writer.field("parts")
    write_parts(writer, value.parts)
    writer.field("tool_calls")
    write_tool_calls(writer, value.tool_calls)
    writer.field("tool_call_id")
    write_optional_string(writer, value.tool_call_id)
    writer.field("outcome")
    if value.outcome is None:
        writer.none()
    else:
        _write_tool_outcome(writer, value.outcome)
    writer.field("metadata")
    writer.json(value.metadata)


def write_parts(writer: DigestWriter, values: tuple[ContentPart, ...]) -> None:
    writer.sequence(len(values))
    for value in values:
        _write_content_part(writer, value)


def _write_content_part(writer: DigestWriter, value: ContentPart) -> None:
    writer.field("content_part")
    writer.field("type")
    writer.string(value.type)
    writer.field("text")
    write_optional_string(writer, value.text)
    writer.field("uri")
    write_optional_string(writer, value.uri)
    writer.field("artifact")
    if value.artifact is None:
        writer.none()
    else:
        _write_artifact(writer, value.artifact)
    writer.field("media_type")
    write_optional_string(writer, value.media_type)
    writer.field("name")
    write_optional_string(writer, value.name)
    writer.field("data")
    writer.json(value.data)
    writer.field("metadata")
    writer.json(value.metadata)


def _write_artifact(writer: DigestWriter, value: ArtifactRef) -> None:
    writer.field("artifact_ref")
    writer.field("ref")
    writer.string(value.ref)
    writer.field("media_type")
    write_optional_string(writer, value.media_type)
    writer.field("name")
    write_optional_string(writer, value.name)
    writer.field("size_bytes")
    write_optional_integer(writer, value.size_bytes)
    writer.field("sha256")
    write_optional_string(writer, value.sha256)
    writer.field("metadata")
    writer.json(value.metadata)


def write_tool_calls(writer: DigestWriter, values: tuple[ToolCall, ...]) -> None:
    writer.sequence(len(values))
    for value in values:
        write_tool_call(writer, value)


def write_tool_call(writer: DigestWriter, value: ToolCall) -> None:
    writer.field("tool_call")
    writer.field("id")
    writer.string(value.id)
    writer.field("name")
    writer.string(value.name)
    writer.field("arguments")
    writer.json(value.arguments)


def _write_tool_outcome(writer: DigestWriter, value: ToolOutcome) -> None:
    writer.field("tool_outcome")
    writer.field("kind")
    writer.string(value.kind)
    writer.field("parts")
    write_parts(writer, value.parts)
    writer.field("structured_content")
    writer.json(value.structured_content)
    if isinstance(value, ToolFailure):
        writer.field("error")
        write_error(writer, value.error)
    elif isinstance(value, ToolAccepted):
        writer.field("correlation_id")
        writer.string(value.correlation_id)
        writer.field("task")
        _write_optional_task(writer, value.task)
    elif isinstance(value, ToolWaiting):
        writer.field("task")
        _write_optional_task(writer, value.task)


def _write_optional_task(writer: DigestWriter, value: TaskRef | None) -> None:
    if value is None:
        writer.none()
        return
    writer.field("task_ref")
    writer.field("id")
    writer.string(value.id)
    writer.field("status")
    writer.string(value.status)
    writer.field("metadata")
    writer.json(value.metadata)


def write_error(writer: DigestWriter, value: ErrorInfo) -> None:
    writer.field("error_info")
    writer.field("code")
    writer.string(value.code)
    writer.field("message")
    writer.string(value.message)


def write_optional_string(writer: DigestWriter, value: str | None) -> None:
    if value is None:
        writer.none()
    else:
        writer.string(value)


def write_optional_integer(writer: DigestWriter, value: int | None) -> None:
    if value is None:
        writer.none()
    else:
        writer.integer(value)
