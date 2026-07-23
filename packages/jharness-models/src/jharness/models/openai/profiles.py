"""Profile knobs for OpenAI Chat Completions provider APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from jharness.models._profiles import copy_json_mapping, copy_string_mapping, required_string

MaxTokensField = Literal["max_tokens", "max_completion_tokens"]
ReasoningContentMode = Literal["live_only", "round_trip", "required_with_tools"]
SystemContentMode = Literal["string", "parts"]

_BOOLEAN_FIELDS = (
    "supports_streaming",
    "supports_tools",
    "supports_tool_choice",
    "supports_parallel_tool_calls",
    "supports_parallel_tool_call_control",
    "supports_image_input",
    "supports_video_input",
    "supports_file_input",
    "supports_json_object",
    "supports_json_schema",
    "supports_seed",
    "requires_assistant_content_for_tool_calls",
    "stream_include_usage",
)


@dataclass(frozen=True, slots=True)
class OpenAIChatCompletionsProfile:
    """OpenAI Chat Completions provider behavior.

    Many providers implement the same wire shape with small differences. Keep
    those differences explicit here instead of spreading provider checks through
    the codec or runtime.
    """

    name: str = "openai-chat-completions"
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_tool_choice: bool = True
    supports_parallel_tool_calls: bool = True
    supports_parallel_tool_call_control: bool = True
    supports_image_input: bool = True
    supports_video_input: bool = False
    supports_file_input: bool = False
    supports_json_object: bool = True
    supports_json_schema: bool = False
    supports_seed: bool = True
    requires_assistant_content_for_tool_calls: bool = False
    stream_include_usage: bool = True
    reasoning_content_mode: ReasoningContentMode = "live_only"
    max_tokens_field: MaxTokensField = "max_tokens"
    system_content_mode: SystemContentMode = "string"
    json_schema_name: str = "response"
    extra_request_body: Mapping[str, Any] = field(default_factory=dict[str, Any])
    finish_reason_map: Mapping[str, str] = field(default_factory=dict[str, str])

    def __post_init__(self) -> None:
        required_string(self.name, "profile name")
        for field_name in _BOOLEAN_FIELDS:
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        if not isinstance(
            cast(object, self.reasoning_content_mode), str
        ) or self.reasoning_content_mode not in {
            "live_only",
            "round_trip",
            "required_with_tools",
        }:
            raise ValueError(
                "reasoning_content_mode must be 'live_only', 'required_with_tools', or 'round_trip'"
            )
        if not isinstance(
            cast(object, self.max_tokens_field), str
        ) or self.max_tokens_field not in {
            "max_tokens",
            "max_completion_tokens",
        }:
            raise ValueError("max_tokens_field must be 'max_completion_tokens' or 'max_tokens'")
        if not isinstance(
            cast(object, self.system_content_mode), str
        ) or self.system_content_mode not in {
            "string",
            "parts",
        }:
            raise ValueError("system_content_mode must be 'parts' or 'string'")
        required_string(self.json_schema_name, "json_schema_name")
        object.__setattr__(
            self,
            "extra_request_body",
            copy_json_mapping(self.extra_request_body, "extra_request_body"),
        )
        object.__setattr__(
            self,
            "finish_reason_map",
            copy_string_mapping(
                self.finish_reason_map,
                "finish_reason_map",
                entry_description="a non-empty string",
            ),
        )

    def finish_reason(self, raw: str | None) -> str | None:
        if raw is None:
            return None
        return self.finish_reason_map.get(raw, raw)
