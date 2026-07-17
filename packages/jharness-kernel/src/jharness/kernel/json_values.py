"""Canonical validation, copying, and freezing for JSON-compatible values."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, NoReturn, SupportsIndex, TypeAlias, cast

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class FrozenJsonList(Sequence[Any]):
    __slots__ = ("_error_message", "_items")

    def __init__(self, items: Iterable[Any], *, error_message: str) -> None:
        self._items = tuple(items)
        self._error_message = error_message

    def __getitem__(self, key: SupportsIndex | slice) -> Any:
        return self._items[key]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence) and not isinstance(other, str | bytes | bytearray):
            return list(self) == list(cast(Sequence[object], other))
        return False

    def __repr__(self) -> str:
        return repr(list(self))

    def _readonly(self) -> NoReturn:
        raise TypeError(self._error_message)

    def __setitem__(self, key: SupportsIndex | slice, value: Any) -> None:
        self._readonly()

    def __delitem__(self, key: SupportsIndex | slice) -> None:
        self._readonly()

    def __iadd__(self, value: Iterable[Any]) -> FrozenJsonList:
        self._readonly()

    def __imul__(self, value: SupportsIndex) -> FrozenJsonList:
        self._readonly()

    def append(self, item: Any) -> None:
        self._readonly()

    def clear(self) -> None:
        self._readonly()

    def extend(self, items: Iterable[Any]) -> None:
        self._readonly()

    def insert(self, index: SupportsIndex, item: Any) -> None:
        self._readonly()

    def pop(self, index: SupportsIndex = -1) -> Any:
        self._readonly()

    def remove(self, item: Any) -> None:
        self._readonly()

    def reverse(self) -> None:
        self._readonly()

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        self._readonly()

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        return [_copy_json_value(item, path="$") for item in self]


class FrozenJsonDict(Mapping[str, Any]):
    __slots__ = ("_error_message", "_items")

    def __init__(self, items: Mapping[str, Any], *, error_message: str) -> None:
        self._items = MappingProxyType(dict(items))
        self._error_message = error_message

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(cast(Mapping[object, object], other).items())
        return False

    def __repr__(self) -> str:
        return repr(dict(self.items()))

    def _readonly(self) -> NoReturn:
        raise TypeError(self._error_message)

    def __setitem__(self, key: str, value: Any) -> None:
        self._readonly()

    def __delitem__(self, key: str) -> None:
        self._readonly()

    def clear(self) -> None:
        self._readonly()

    def pop(self, key: str, default: Any = None) -> Any:
        self._readonly()

    def popitem(self) -> tuple[str, Any]:
        self._readonly()

    def setdefault(self, key: str, default: Any = None) -> Any:
        self._readonly()

    def update(self, *args: Any, **kwargs: Any) -> None:
        self._readonly()

    def __ior__(self, value: object) -> FrozenJsonDict:
        self._readonly()

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        return {key: _copy_json_value(value, path=f"$[{key!r}]") for key, value in self.items()}


FrozenJsonValue: TypeAlias = JsonScalar | FrozenJsonList | FrozenJsonDict
_NOT_JSON_SCALAR = object()


def _copy_json_scalar(value: object, path: str) -> JsonScalar | object:
    if value is None:
        copied: JsonScalar | object = value
    elif isinstance(value, bool):
        copied = bool(value)
    elif isinstance(value, str):
        copied = str(value)
    elif isinstance(value, int):
        copied = int(value)
    elif isinstance(value, float):
        number = float(value)
        if not isfinite(number):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        copied = number
    else:
        copied = _NOT_JSON_SCALAR
    return copied


@dataclass(slots=True)
class _JsonCopier:
    error_message: str | None
    active_containers: set[int]

    def copy(self, value: object, path: str) -> JsonValue | FrozenJsonValue:
        scalar = _copy_json_scalar(value, path)
        if scalar is not _NOT_JSON_SCALAR:
            return cast(JsonScalar, scalar)
        if isinstance(value, Mapping):
            return self._copy_mapping(cast(Mapping[object, object], value), path)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            return self._copy_sequence(cast(Sequence[object], value), path)
        raise TypeError(f"{path} must contain only JSON-compatible values")

    def _copy_mapping(
        self,
        value: Mapping[object, object],
        path: str,
    ) -> JsonValue | FrozenJsonValue:
        identity = self._enter_container(value, path)
        try:
            copied: dict[str, JsonValue | FrozenJsonValue] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} object keys must be strings")
                copied[key] = self.copy(item, f"{path}[{key!r}]")
            if self.error_message is not None:
                return FrozenJsonDict(copied, error_message=self.error_message)
            return cast(JsonValue, copied)
        finally:
            self.active_containers.remove(identity)

    def _copy_sequence(
        self,
        value: Sequence[object],
        path: str,
    ) -> JsonValue | FrozenJsonValue:
        identity = self._enter_container(value, path)
        try:
            copied = [self.copy(item, f"{path}[{index}]") for index, item in enumerate(value)]
            if self.error_message is not None:
                return FrozenJsonList(copied, error_message=self.error_message)
            return cast(JsonValue, copied)
        finally:
            self.active_containers.remove(identity)

    def _enter_container(self, value: object, path: str) -> int:
        identity = id(value)
        if identity in self.active_containers:
            raise ValueError(f"{path} must not contain a reference cycle")
        self.active_containers.add(identity)
        return identity


def _copy_json_value(
    value: object,
    *,
    path: str,
    error_message: str | None = None,
    active_containers: set[int] | None = None,
) -> JsonValue | FrozenJsonValue:
    active: set[int] = set() if active_containers is None else active_containers
    return _JsonCopier(error_message, active).copy(value, path)


def freeze_json_value(
    value: object,
    *,
    label: str = "value",
    error_message: str = "JSON value is immutable",
) -> FrozenJsonValue:
    """Validate and copy a value into recursively immutable JSON containers."""

    return cast(
        FrozenJsonValue,
        _copy_json_value(value, path=label, error_message=error_message),
    )


def thaw_json_value(value: object, *, label: str = "value") -> JsonValue:
    """Validate and copy a frozen JSON value into plain dict/list containers."""

    return cast(JsonValue, _copy_json_value(value, path=label))
