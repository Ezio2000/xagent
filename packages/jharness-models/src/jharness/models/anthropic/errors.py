"""Anthropic Messages adapter errors."""

from __future__ import annotations

from jharness.models._json import JsonValues


class AnthropicError(ValueError):
    """The Anthropic Messages adapter could not encode or decode a request."""


ANTHROPIC_JSON = JsonValues(AnthropicError)
