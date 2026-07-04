"""Internal validation helpers shared by kernel value objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast


def expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def expect_sequence(
    value: object,
    label: str,
    *,
    noun: str = "array",
) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{label} must be a {noun}")
    return cast(Sequence[object], value)


def expect_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return cast(list[object], value)


def expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def expect_optional_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string or null")
    return value


def expect_optional_non_empty_str(value: object, label: str) -> str | None:
    text = expect_optional_str(value, label)
    if text == "":
        raise ValueError(f"{label} must not be empty")
    return text


def expect_present_str(
    value: Mapping[str, Any],
    key: str,
    label: str,
) -> str | None:
    if key not in value:
        return None
    raw = value[key]
    if not isinstance(raw, str):
        raise TypeError(f"{label} must be a string")
    return raw


def expect_present_optional_str(
    value: Mapping[str, Any],
    key: str,
    label: str,
) -> str | None:
    if key not in value:
        return None
    raw = value[key]
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TypeError(f"{label} must be a string or null")
    return raw


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


def expect_present_optional_int(
    value: Mapping[str, Any],
    key: str,
    label: str,
) -> int | None:
    if key not in value:
        return None
    raw = value[key]
    if raw is None:
        return None
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise TypeError(f"{label} must be an integer or null")
    return raw


def expect_number(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{label} must be a number")
    return float(value)


def expect_optional_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{label} must be a number or null")
    return float(value)


def expect_present_optional_number(
    value: Mapping[str, Any],
    key: str,
    label: str,
) -> float | None:
    if key not in value:
        return None
    raw = value[key]
    if raw is None:
        return None
    if not isinstance(raw, int | float) or isinstance(raw, bool):
        raise TypeError(f"{label} must be a number or null")
    return float(raw)


def expect_present_optional_bool(
    value: Mapping[str, Any],
    key: str,
    label: str,
) -> bool | None:
    if key not in value:
        return None
    raw = value[key]
    if raw is None:
        return None
    if not isinstance(raw, bool):
        raise TypeError(f"{label} must be a boolean or null")
    return raw


def reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{label} has unknown field(s): {names}")
