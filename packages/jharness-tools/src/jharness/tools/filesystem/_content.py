"""Shared bounded-content helpers for filesystem presets."""

from __future__ import annotations

from hashlib import sha256
from typing import Final

SHA256_PATTERN: Final = "^[0-9a-f]{64}$"


def digest_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest for raw file bytes."""

    return sha256(value).hexdigest()


def sha256_schema() -> dict[str, object]:
    """Return a strict JSON Schema fragment for a lowercase SHA-256 digest."""

    return {
        "type": "string",
        "minLength": 64,
        "maxLength": 64,
        "pattern": SHA256_PATTERN,
    }
