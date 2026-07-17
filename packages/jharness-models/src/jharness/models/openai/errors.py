"""OpenAI Chat Completions adapter errors."""

from __future__ import annotations

from jharness.models._json import JsonValues


class OpenAIChatCompletionsError(ValueError):
    """The OpenAI Chat Completions adapter could not encode or decode a request."""


OPENAI_JSON = JsonValues(OpenAIChatCompletionsError)
