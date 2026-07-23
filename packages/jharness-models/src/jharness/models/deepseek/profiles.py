"""DeepSeek provider profiles."""

from __future__ import annotations

from typing import Literal

from jharness.models.anthropic import AnthropicProfile
from jharness.models.openai import OpenAIChatCompletionsProfile

DeepSeekThinkingEffort = Literal["high", "max"]

_THINKING_EFFORTS = frozenset({"high", "max"})


def deepseek_openai_chat_profile(
    *,
    thinking: bool,
    effort: DeepSeekThinkingEffort | None = None,
) -> OpenAIChatCompletionsProfile:
    """Return a DeepSeek profile for the OpenAI Chat Completions wire protocol."""

    thinking = _validate_options(thinking, effort)
    extra_request_body = _thinking_request_body(thinking=thinking, effort=effort)
    return OpenAIChatCompletionsProfile(
        name=_profile_name("deepseek-openai-chat", thinking),
        supports_streaming=True,
        supports_tools=True,
        supports_tool_choice=not thinking,
        supports_parallel_tool_calls=True,
        supports_parallel_tool_call_control=False,
        supports_image_input=False,
        supports_video_input=False,
        supports_file_input=False,
        supports_json_object=True,
        supports_json_schema=False,
        supports_seed=False,
        requires_assistant_content_for_tool_calls=thinking,
        reasoning_content_mode="required_with_tools" if thinking else "live_only",
        stream_include_usage=True,
        extra_request_body=extra_request_body,
    )


def deepseek_anthropic_profile(
    *,
    thinking: bool,
    effort: DeepSeekThinkingEffort | None = None,
) -> AnthropicProfile:
    """Return a DeepSeek profile for the Anthropic Messages wire protocol."""

    thinking = _validate_options(thinking, effort)
    extra_request_body = _thinking_request_body(thinking=thinking, effort=None)
    return AnthropicProfile(
        name=_profile_name("deepseek-anthropic", thinking),
        supports_streaming=True,
        supports_tools=True,
        supports_tool_choice=True,
        supports_parallel_tool_calls=True,
        supports_parallel_tool_call_control=False,
        supports_image_input=False,
        supports_file_input=False,
        supports_json_object=False,
        supports_json_schema=False,
        supports_redacted_thinking=False,
        stream_usage=True,
        extra_request_body=extra_request_body,
        extra_output_config={} if effort is None else {"effort": effort},
    )


def _thinking_request_body(
    *,
    thinking: bool,
    effort: DeepSeekThinkingEffort | None,
) -> dict[str, object]:
    body: dict[str, object] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
    if effort is not None:
        body["reasoning_effort"] = effort
    return body


def _validate_options(thinking: object, effort: object) -> bool:
    if not isinstance(thinking, bool):
        raise ValueError("thinking must be a bool")
    if effort is not None and not thinking:
        raise ValueError("effort is only valid when thinking=True")
    if effort is not None and effort not in _THINKING_EFFORTS:
        expected = ", ".join(sorted(_THINKING_EFFORTS))
        raise ValueError(f"effort must be one of: {expected}")
    return thinking


def _profile_name(prefix: str, thinking: bool) -> str:
    mode = "thinking" if thinking else "nonthinking"
    return f"{prefix}-{mode}"
