"""Internal immutable mapping/list helpers for event and trace payloads."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any, NoReturn, SupportsIndex, cast


class FrozenList(list[Any]):
    """List copy that rejects mutation while preserving deepcopy to plain list."""

    __slots__ = ("_error_message",)

    def __init__(self, items: Iterable[Any], *, error_message: str) -> None:
        super().__init__(items)
        self._error_message = error_message

    def _readonly(self) -> NoReturn:
        raise TypeError(self._error_message)

    def __setitem__(self, key: SupportsIndex | slice, value: Any) -> None:
        _ = key, value
        self._readonly()

    def __delitem__(self, key: SupportsIndex | slice) -> None:
        _ = key
        self._readonly()

    def __iadd__(self, value: Iterable[Any]) -> FrozenList:
        _ = value
        self._readonly()

    def __imul__(self, value: SupportsIndex) -> FrozenList:
        _ = value
        self._readonly()

    def append(self, item: Any) -> None:
        _ = item
        self._readonly()

    def clear(self) -> None:
        self._readonly()

    def extend(self, items: Iterable[Any]) -> None:
        _ = items
        self._readonly()

    def insert(self, index: SupportsIndex, item: Any) -> None:
        _ = index, item
        self._readonly()

    def pop(self, index: SupportsIndex = -1) -> Any:
        _ = index
        self._readonly()

    def remove(self, item: Any) -> None:
        _ = item
        self._readonly()

    def reverse(self) -> None:
        self._readonly()

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        _ = key, reverse
        self._readonly()

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        return [deepcopy(item, memo) for item in self]


class FrozenDict(dict[str, Any]):
    """Dict copy that rejects mutation while preserving deepcopy to plain dict."""

    __slots__ = ("_error_message",)

    def __init__(self, items: Mapping[str, Any], *, error_message: str) -> None:
        super().__init__(items)
        self._error_message = error_message

    def _readonly(self) -> NoReturn:
        raise TypeError(self._error_message)

    def __setitem__(self, key: str, value: Any) -> None:
        _ = key, value
        self._readonly()

    def __delitem__(self, key: str) -> None:
        _ = key
        self._readonly()

    def clear(self) -> None:
        self._readonly()

    def pop(self, key: str, default: Any = None) -> Any:
        _ = key, default
        self._readonly()

    def popitem(self) -> tuple[str, Any]:
        self._readonly()

    def setdefault(self, key: str, default: Any = None) -> Any:
        _ = key, default
        self._readonly()

    def update(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        self._readonly()

    def __ior__(self, value: object) -> FrozenDict:
        _ = value
        self._readonly()

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        return {deepcopy(key, memo): deepcopy(value, memo) for key, value in self.items()}


def freeze_value(value: object, *, error_message: str) -> object:
    if isinstance(value, Mapping):
        return FrozenDict(
            {
                key: freeze_value(item, error_message=error_message)
                for key, item in cast(Mapping[str, Any], value).items()
            },
            error_message=error_message,
        )
    if isinstance(value, list | tuple):
        return FrozenList(
            [
                freeze_value(item, error_message=error_message)
                for item in cast(Sequence[object], value)
            ],
            error_message=error_message,
        )
    return deepcopy(value)


def thaw_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: thaw_value(item) for key, item in cast(Mapping[str, Any], value).items()}
    if isinstance(value, list | tuple):
        return [thaw_value(item) for item in cast(Sequence[object], value)]
    return deepcopy(value)
