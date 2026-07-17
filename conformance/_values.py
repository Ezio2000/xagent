"""Strict fixture-value parsing helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast


def load_object(path: Path, label: str = "JSON document") -> dict[str, Any]:
    try:
        value: object = json.loads(path.read_text())
    except OSError as exc:
        raise OSError(f"failed to read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{path}: {label} must be an object")
    return cast(dict[str, Any], value)


def mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    candidate = cast(Mapping[object, object], value)
    if any(not isinstance(key, str) for key in candidate):
        raise TypeError(f"{label} keys must be strings")
    return cast(Mapping[str, Any], value)


def sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{label} must be an array")
    return cast(Sequence[object], value)


def string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a number")
    return float(value)


def boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def string_list(value: object, label: str) -> list[str]:
    return [string(item, f"{label} item") for item in sequence(value, label)]
