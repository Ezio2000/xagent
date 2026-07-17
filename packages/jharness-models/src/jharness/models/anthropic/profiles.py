"""Profile knobs for Anthropic Messages provider APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from jharness.models._profiles import copy_json_mapping, copy_string_mapping, required_string

AnthropicAuthScheme = Literal["x-api-key", "bearer"]
SystemContentMode = Literal["string", "blocks"]

_BOOLEAN_FIELDS = (
    "supports_streaming",
    "supports_tools",
    "supports_tool_choice",
    "supports_parallel_tool_calls",
    "supports_parallel_tool_call_control",
    "supports_image_input",
    "supports_file_input",
    "supports_json_object",
    "supports_json_schema",
    "stream_usage",
    "supports_mid_conversation_system",
)


@dataclass(frozen=True, slots=True)
class AnthropicProfile:
    """Anthropic Messages provider behavior."""

    name: str = "anthropic"
    anthropic_version: str = "2023-06-01"
    auth_scheme: AnthropicAuthScheme = "x-api-key"
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_tool_choice: bool = True
    supports_parallel_tool_calls: bool = True
    supports_parallel_tool_call_control: bool = True
    supports_image_input: bool = True
    supports_file_input: bool = True
    supports_json_object: bool = False
    supports_json_schema: bool = True
    stream_usage: bool = True
    default_max_tokens: int = 1024
    system_content_mode: SystemContentMode = "string"
    supports_mid_conversation_system: bool = False
    seed_field: str | None = None
    file_ref_beta_header: str | None = "files-api-2025-04-14"
    json_object_schema: Mapping[str, Any] = field(default_factory=lambda: {"type": "object"})
    extra_output_config: Mapping[str, Any] = field(default_factory=dict[str, Any])
    extra_request_body: Mapping[str, Any] = field(default_factory=dict[str, Any])
    extra_headers: Mapping[str, str] = field(default_factory=dict[str, str])
    finish_reason_map: Mapping[str, str] = field(default_factory=dict[str, str])

    def __post_init__(self) -> None:
        required_string(self.name, "profile name")
        required_string(self.anthropic_version, "anthropic_version")
        for field_name in _BOOLEAN_FIELDS:
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        if not isinstance(cast(object, self.auth_scheme), str) or self.auth_scheme not in {
            "x-api-key",
            "bearer",
        }:
            raise ValueError("auth_scheme must be 'x-api-key' or 'bearer'")
        if not isinstance(
            cast(object, self.system_content_mode), str
        ) or self.system_content_mode not in {
            "string",
            "blocks",
        }:
            raise ValueError("system_content_mode must be 'string' or 'blocks'")
        if not isinstance(cast(object, self.default_max_tokens), int) or isinstance(
            self.default_max_tokens, bool
        ):
            raise TypeError("default_max_tokens must be an integer")
        if self.default_max_tokens < 1:
            raise ValueError("default_max_tokens must be >= 1")
        _validate_optional_string(self.seed_field, "seed_field")
        _validate_optional_string(self.file_ref_beta_header, "file_ref_beta_header")
        object.__setattr__(
            self,
            "json_object_schema",
            copy_json_mapping(self.json_object_schema, "json_object_schema"),
        )
        object.__setattr__(
            self,
            "extra_output_config",
            copy_json_mapping(self.extra_output_config, "extra_output_config"),
        )
        object.__setattr__(
            self,
            "extra_request_body",
            copy_json_mapping(self.extra_request_body, "extra_request_body"),
        )
        object.__setattr__(
            self,
            "extra_headers",
            copy_string_mapping(self.extra_headers, "extra_headers"),
        )
        object.__setattr__(
            self,
            "finish_reason_map",
            copy_string_mapping(self.finish_reason_map, "finish_reason_map"),
        )

    def finish_reason(self, raw: str | None) -> str | None:
        if raw is None:
            return None
        return self.finish_reason_map.get(raw, raw)


def _validate_optional_string(value: object, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string when set")
