"""Run-control primitives for pausing and interrupting agent runs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, cast


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


class PauseController:
    """Mutable run-control handle shared with host code."""

    __slots__ = ("_lock", "_request", "_waiters")

    _lock: Lock
    _request: PauseRequest | None
    _waiters: set[asyncio.Future[None]]

    def __init__(self) -> None:
        self._lock = Lock()
        self._request = None
        self._waiters = set()

    @property
    def request(self) -> PauseRequest | None:
        with self._lock:
            request = self._request
            if request is None:
                return None
            return PauseRequest.from_dict(request.to_dict())

    @property
    def is_requested(self) -> bool:
        with self._lock:
            return self._request is not None

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
        self._set_request(request)
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
        self._set_request(request)
        return PauseRequest.from_dict(request.to_dict())

    def clear(self) -> None:
        with self._lock:
            self._request = None
            waiters = tuple(self._waiters)
        self._notify_waiters(waiters)

    async def wait_for_interrupt(self) -> PauseRequest:
        while True:
            with self._lock:
                request = self._request
                if request is not None and request.interrupt:
                    return PauseRequest.from_dict(request.to_dict())
                waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                self._waiters.add(waiter)
            try:
                await waiter
            finally:
                with self._lock:
                    self._waiters.discard(waiter)

    def _set_request(self, request: PauseRequest) -> None:
        with self._lock:
            self._request = PauseRequest.from_dict(request.to_dict())
            waiters = tuple(self._waiters)
        self._notify_waiters(waiters)

    @staticmethod
    def _notify_waiters(waiters: tuple[asyncio.Future[None], ...]) -> None:
        for waiter in waiters:
            loop = waiter.get_loop()
            if loop.is_closed():
                continue
            loop.call_soon_threadsafe(PauseController._resolve_waiter, waiter)

    @staticmethod
    def _resolve_waiter(waiter: asyncio.Future[None]) -> None:
        if not waiter.done():
            waiter.set_result(None)
