"""Internal validation helpers shared by kernel value objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, TypeVar, cast

from jharness.kernel.json_values import freeze_json_value

T = TypeVar("T")


def expect_instance(value: object, expected: type[T], label: str) -> T:
    if not isinstance(value, expected):
        raise TypeError(f"{label} must be {expected.__name__}")
    return value


def expect_instance_tuple(
    value: object,
    expected: type[T],
    label: str,
) -> tuple[T, ...]:
    items = expect_sequence(value, label)
    if any(not isinstance(item, expected) for item in items):
        raise TypeError(f"{label} must contain {expected.__name__} values")
    return cast(tuple[T, ...], tuple(items))


def expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in cast(Mapping[object, object], value)):
        raise TypeError(f"{label} keys must be strings")
    return cast(Mapping[str, Any], value)


def freeze_mapping(value: object, label: str) -> Mapping[str, Any]:
    return cast(
        Mapping[str, Any],
        freeze_json_value(
            expect_mapping(value, label),
            label=label,
            error_message=f"{label} is immutable",
        ),
    )


def expect_sequence(
    value: object,
    label: str,
    *,
    noun: str = "array",
) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{label} must be a {noun}")
    return cast(Sequence[object], value)


def expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def expect_non_empty_str(value: object, label: str) -> str:
    text = expect_str(value, label)
    if not text:
        raise ValueError(f"{label} must not be empty")
    return text


def expect_optional_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string or null")
    return value


def expect_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def expect_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    return value


def expect_nonnegative_int(value: object, label: str) -> int:
    number = expect_int(value, label)
    if number < 0:
        raise ValueError(f"{label} must be >= 0")
    return number


def expect_optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer or null")
    return value


def expect_optional_nonnegative_int(value: object, label: str) -> int | None:
    number = expect_optional_int(value, label)
    if number is not None and number < 0:
        raise ValueError(f"{label} must be >= 0")
    return number


def expect_number(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def expect_optional_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{label} must be a number or null")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number
