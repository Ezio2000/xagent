"""Small validation vocabulary shared by provider profile values."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast


def required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def copy_json_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result: dict[str, Any] = {}
    for key, item in cast(Mapping[object, object], value).items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        result[key] = deepcopy(item)
    return result


def copy_string_mapping(
    value: object,
    label: str,
    *,
    entry_description: str = "non-empty strings",
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    result: dict[str, str] = {}
    for key, item in cast(Mapping[object, object], value).items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be {entry_description}")
        if not isinstance(item, str) or not item:
            raise ValueError(f"{label} values must be {entry_description}")
        result[key] = item
    return result
