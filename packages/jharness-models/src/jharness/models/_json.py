"""Small scalar guards shared by provider wire codecs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn, cast


@dataclass(frozen=True, slots=True)
class JsonValues:
    """Validate common JSON scalar shapes with a provider-local error type."""

    error_type: type[ValueError]

    def mapping(self, value: object, label: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            self._raise(f"{label} must be an object")
        return cast(Mapping[str, Any], value)

    def required_string(self, value: object, label: str) -> str:
        if not isinstance(value, str):
            self._raise(f"{label} must be a string")
        if not value:
            self._raise(f"{label} must not be empty")
        return value

    def optional_string(self, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            self._raise("expected string or null")
        return value

    def optional_integer(self, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            self._raise("expected integer or null")
        return value

    def _raise(self, message: str) -> NoReturn:
        raise self.error_type(message)
