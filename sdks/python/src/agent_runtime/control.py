"""Run-control primitives for pausing, interrupting, and inserting inputs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, TypeAlias, cast

from agent_runtime.messages import ContentPart, Message


def _empty_metadata() -> Mapping[str, Any]:
    return {}


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(dict(_expect_mapping(value, "mapping")))


def _expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _expect_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{label} must be an array")
    return cast(Sequence[object], value)


def _expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _expect_optional_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string or null")
    return value


def _expect_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{label} has unknown field(s): {names}")


@dataclass(slots=True, frozen=True)
class PauseRequest:
    """A request for the runtime to stop at a resumable boundary."""

    reason: str = "host_requested"
    source: str = "host"
    wait_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)
    interrupt: bool = False

    def __post_init__(self) -> None:
        _expect_str(self.reason, "pause request reason")
        _expect_str(self.source, "pause request source")
        _expect_optional_str(self.wait_id, "pause request wait_id")
        _expect_bool(self.interrupt, "pause request interrupt")
        if not self.reason:
            raise ValueError("pause request reason must not be empty")
        if not self.source:
            raise ValueError("pause request source must not be empty")
        object.__setattr__(
            self,
            "metadata",
            _copy_mapping(_expect_mapping(self.metadata, "pause request metadata")),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PauseRequest:
        known = {"reason", "source", "wait_id", "metadata", "interrupt"}
        _reject_unknown_keys(value, known, "pause request")
        raw_metadata = value["metadata"]
        raw_wait_id = value["wait_id"]
        return cls(
            reason=_expect_str(value["reason"], "pause reason"),
            source=_expect_str(value["source"], "pause source"),
            wait_id=_expect_optional_str(raw_wait_id, "pause wait_id"),
            metadata=_expect_mapping(raw_metadata, "pause metadata"),
            interrupt=_expect_bool(value["interrupt"], "pause interrupt"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "source": self.source,
            "wait_id": self.wait_id,
            "metadata": _copy_mapping(self.metadata),
            "interrupt": self.interrupt,
        }


@dataclass(slots=True, frozen=True)
class ConversationInsert:
    """External input that preempts a run and enters conversation history."""

    id: str
    source: str
    parts: tuple[ContentPart, ...]
    correlation_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _expect_str(self.id, "conversation insert id"))
        object.__setattr__(self, "source", _expect_str(self.source, "conversation insert source"))
        object.__setattr__(
            self,
            "correlation_id",
            _expect_optional_str(self.correlation_id, "conversation insert correlation_id"),
        )
        if not self.id:
            raise ValueError("conversation insert id must not be empty")
        if not self.source:
            raise ValueError("conversation insert source must not be empty")
        if self.correlation_id is not None and not self.correlation_id:
            raise ValueError("conversation insert correlation_id must not be empty")
        parts: list[ContentPart] = []
        for part in _expect_sequence(self.parts, "conversation insert parts"):
            if not isinstance(part, ContentPart):
                raise TypeError("conversation insert parts items must be ContentPart")
            parts.append(ContentPart.from_dict(part.to_dict()))
        object.__setattr__(self, "parts", tuple(parts))
        object.__setattr__(
            self,
            "metadata",
            _copy_mapping(_expect_mapping(self.metadata, "conversation insert metadata")),
        )

    @classmethod
    def text(
        cls,
        text: str,
        *,
        id: str,
        source: str,
        correlation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ConversationInsert:
        return cls(
            id=id,
            source=source,
            correlation_id=correlation_id,
            parts=(ContentPart.text_part(text),),
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConversationInsert:
        known = {"id", "source", "parts", "correlation_id", "metadata"}
        _reject_unknown_keys(value, known, "conversation insert")
        raw_metadata: object = value.get("metadata", {})
        return cls(
            id=_expect_str(value["id"], "conversation insert id"),
            source=_expect_str(value["source"], "conversation insert source"),
            correlation_id=_expect_optional_str(
                value.get("correlation_id"), "conversation insert correlation_id"
            ),
            parts=tuple(
                ContentPart.from_dict(_expect_mapping(part, "conversation insert part"))
                for part in _expect_sequence(value["parts"], "conversation insert parts")
            ),
            metadata=_expect_mapping(raw_metadata, "conversation insert metadata"),
        )

    def to_message(self) -> Message:
        return Message.external(
            self.parts,
            insert_id=self.id,
            source=self.source,
            correlation_id=self.correlation_id,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "parts": [part.to_dict() for part in self.parts],
        }
        if self.correlation_id is not None:
            data["correlation_id"] = self.correlation_id
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        return data


@dataclass(slots=True, frozen=True)
class ToolCancelRequest:
    """Host request for a running tool call to cancel cooperatively."""

    tool_call_id: str
    reason: str = "host_cancelled"
    source: str = "host"
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        tool_call_id = _expect_str(self.tool_call_id, "tool cancel tool_call_id")
        reason = _expect_str(self.reason, "tool cancel reason")
        source = _expect_str(self.source, "tool cancel source")
        if not tool_call_id:
            raise ValueError("tool cancel tool_call_id must not be empty")
        if not reason:
            raise ValueError("tool cancel reason must not be empty")
        if not source:
            raise ValueError("tool cancel source must not be empty")
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "metadata",
            _copy_mapping(_expect_mapping(self.metadata, "tool cancel metadata")),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolCancelRequest:
        known = {"tool_call_id", "reason", "source", "metadata"}
        _reject_unknown_keys(value, known, "tool cancel request")
        raw_metadata: object = value.get("metadata", {})
        return cls(
            tool_call_id=_expect_str(value["tool_call_id"], "tool cancel tool_call_id"),
            reason=_expect_str(value["reason"], "tool cancel reason"),
            source=_expect_str(value["source"], "tool cancel source"),
            metadata=_expect_mapping(raw_metadata, "tool cancel metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "reason": self.reason,
            "source": self.source,
            "metadata": _copy_mapping(self.metadata),
        }


ControlInterrupt: TypeAlias = PauseRequest | ConversationInsert


class RunController:
    """Mutable run-control handle shared with host code."""

    __slots__ = (
        "_cancelled_tool_call_ids",
        "_insert_ids",
        "_inserts",
        "_lock",
        "_pause_request",
        "_tool_cancel_requests",
        "_tool_cancel_waiters",
        "_waiters",
    )

    _cancelled_tool_call_ids: set[str]
    _insert_ids: set[str]
    _inserts: list[ConversationInsert]
    _lock: Lock
    _pause_request: PauseRequest | None
    _tool_cancel_requests: list[ToolCancelRequest]
    _tool_cancel_waiters: set[asyncio.Future[None]]
    _waiters: set[asyncio.Future[None]]

    def __init__(self) -> None:
        self._lock = Lock()
        self._pause_request = None
        self._inserts = []
        self._insert_ids = set()
        self._tool_cancel_requests = []
        self._cancelled_tool_call_ids = set()
        self._tool_cancel_waiters = set()
        self._waiters = set()

    @property
    def pause_request(self) -> PauseRequest | None:
        with self._lock:
            request = self._pause_request
            if request is None:
                return None
            return PauseRequest.from_dict(request.to_dict())

    @property
    def has_pause_request(self) -> bool:
        with self._lock:
            return self._pause_request is not None

    @property
    def has_insert(self) -> bool:
        with self._lock:
            return bool(self._inserts)

    def request_pause(
        self,
        *,
        reason: str = "host_requested",
        source: str = "host",
        wait_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PauseRequest:
        request = PauseRequest(
            reason=reason,
            source=source,
            wait_id=wait_id,
            metadata={} if metadata is None else metadata,
            interrupt=False,
        )
        self._set_pause_request(request)
        return PauseRequest.from_dict(request.to_dict())

    def interrupt(
        self,
        *,
        reason: str = "host_interrupted",
        source: str = "host",
        wait_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PauseRequest:
        request = PauseRequest(
            reason=reason,
            source=source,
            wait_id=wait_id,
            metadata={} if metadata is None else metadata,
            interrupt=True,
        )
        self._set_pause_request(request)
        return PauseRequest.from_dict(request.to_dict())

    def insert(self, insert: ConversationInsert) -> ConversationInsert:
        canonical = ConversationInsert.from_dict(insert.to_dict())
        with self._lock:
            if canonical.id not in self._insert_ids:
                self._insert_ids.add(canonical.id)
                self._inserts.append(canonical)
            waiters = tuple(self._waiters)
        self._notify_waiters(waiters)
        return ConversationInsert.from_dict(canonical.to_dict())

    def pop_insert(self) -> ConversationInsert | None:
        with self._lock:
            if not self._inserts:
                return None
            insert = self._inserts.pop(0)
            return ConversationInsert.from_dict(insert.to_dict())

    def clear_pause(self) -> None:
        with self._lock:
            self._pause_request = None
            waiters = tuple(self._waiters)
        self._notify_waiters(waiters)

    def cancel_tool(
        self,
        tool_call_id: str,
        *,
        reason: str = "host_cancelled",
        source: str = "host",
        metadata: Mapping[str, Any] | None = None,
    ) -> ToolCancelRequest:
        request = ToolCancelRequest(
            tool_call_id=tool_call_id,
            reason=reason,
            source=source,
            metadata={} if metadata is None else metadata,
        )
        with self._lock:
            self._cancelled_tool_call_ids.add(request.tool_call_id)
            self._tool_cancel_requests.append(request)
            waiters = tuple(self._tool_cancel_waiters)
        self._notify_waiters(waiters)
        return ToolCancelRequest.from_dict(request.to_dict())

    def is_tool_cancelled(self, tool_call_id: str) -> bool:
        with self._lock:
            return tool_call_id in self._cancelled_tool_call_ids

    def clear_tool_cancel(self, tool_call_id: str) -> None:
        with self._lock:
            self._cancelled_tool_call_ids.discard(tool_call_id)

    async def wait_for_tool_cancel(self) -> ToolCancelRequest:
        while True:
            with self._lock:
                if self._tool_cancel_requests:
                    request = self._tool_cancel_requests.pop(0)
                    return ToolCancelRequest.from_dict(request.to_dict())
                waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                self._tool_cancel_waiters.add(waiter)
            try:
                await waiter
            finally:
                with self._lock:
                    self._tool_cancel_waiters.discard(waiter)

    async def wait_for_interrupt_or_insert(self) -> ControlInterrupt:
        while True:
            with self._lock:
                if self._inserts:
                    insert = self._inserts.pop(0)
                    return ConversationInsert.from_dict(insert.to_dict())
                request = self._pause_request
                if request is not None and request.interrupt:
                    return PauseRequest.from_dict(request.to_dict())
                waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                self._waiters.add(waiter)
            try:
                await waiter
            finally:
                with self._lock:
                    self._waiters.discard(waiter)

    def _set_pause_request(self, request: PauseRequest) -> None:
        with self._lock:
            self._pause_request = PauseRequest.from_dict(request.to_dict())
            waiters = tuple(self._waiters)
        self._notify_waiters(waiters)

    @staticmethod
    def _notify_waiters(waiters: tuple[asyncio.Future[None], ...]) -> None:
        for waiter in waiters:
            loop = waiter.get_loop()
            if loop.is_closed():
                continue
            loop.call_soon_threadsafe(RunController._resolve_waiter, waiter)

    @staticmethod
    def _resolve_waiter(waiter: asyncio.Future[None]) -> None:
        if not waiter.done():
            waiter.set_result(None)
