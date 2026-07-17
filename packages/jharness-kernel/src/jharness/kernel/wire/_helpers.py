"""Small strict helpers shared by explicit wire codecs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Set
from math import isfinite
from typing import Any, TypeVar, cast

from jharness.kernel.errors import ProtocolError
from jharness.kernel.json_values import thaw_json_value

T = TypeVar("T")


def decode_document(value: object, label: str, decoder: Callable[[object], T]) -> T:
    """Validate one host JSON tree and normalize decode failures."""

    try:
        _validate_json(value, label, set())
        return decoder(value)
    except ProtocolError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid {label}: {exc}") from exc


def object_fields(
    value: object,
    label: str,
    required: Set[str],
    optional: Set[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be an object")
    mapping = cast(Mapping[str, Any], value)
    keys = set(mapping)
    missing = required - keys
    if missing:
        raise ProtocolError(f"{label} is missing field(s): {', '.join(sorted(missing))}")
    unknown = keys - required - optional
    if unknown:
        raise ProtocolError(f"{label} has unknown field(s): {', '.join(sorted(unknown))}")
    return mapping


def array(value: object, label: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ProtocolError(f"{label} must be an array")
    return tuple(cast(Sequence[Any], value))


def string(value: object, label: str, *, non_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ProtocolError(f"{label} must be a string")
    if non_empty and not value:
        raise ProtocolError(f"{label} must not be empty")
    return value


def optional_string(value: object, label: str, *, non_empty: bool = False) -> str | None:
    if value is None:
        return None
    return string(value, label, non_empty=non_empty)


def integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ProtocolError(f"{label} must be >= {minimum}")
    return value


def optional_integer(value: object, label: str, *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    return integer(value, label, minimum=minimum)


def number(value: object, label: str, *, minimum: float | None = None) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ProtocolError(f"{label} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ProtocolError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ProtocolError(f"{label} must be >= {minimum}")
    return result


def optional_number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return number(value, label, minimum=minimum)


def boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(f"{label} must be a boolean")
    return value


def json_object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def enum_string(value: object, label: str, allowed: Set[str]) -> str:
    result = string(value, label)
    if result not in allowed:
        raise ProtocolError(f"{label} has unsupported value: {result}")
    return result


def unique_strings(
    value: object,
    label: str,
    *,
    non_empty_items: bool = False,
) -> tuple[str, ...]:
    values = tuple(
        string(item, f"{label} item", non_empty=non_empty_items) for item in array(value, label)
    )
    if len(values) != len(set(values)):
        raise ProtocolError(f"{label} items must be unique")
    return values


def thaw_object(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], thaw_json_value(value))


def thaw_value(value: object) -> Any:
    return thaw_json_value(value)


def _validate_json(value: object, label: str, active: set[int]) -> None:
    if value is None or isinstance(value, str | bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ProtocolError(f"{label} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        _validate_container(mapping, label, active)
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise ProtocolError(f"{label} object keys must be strings")
            _validate_json(item, f"{label}.{key}", active)
        active.remove(id(mapping))
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        sequence = cast(Sequence[object], value)
        _validate_container(sequence, label, active)
        for index, item in enumerate(sequence):
            _validate_json(item, f"{label}[{index}]", active)
        active.remove(id(sequence))
        return
    raise ProtocolError(f"{label} contains a non-JSON value")


def _validate_container(value: object, label: str, active: set[int]) -> None:
    identity = id(value)
    if identity in active:
        raise ProtocolError(f"{label} contains a cycle")
    active.add(identity)
