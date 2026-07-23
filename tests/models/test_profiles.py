from __future__ import annotations

from typing import Any, cast

import httpx
import pytest

from jharness.models._http import model_client_config
from jharness.models.anthropic import AnthropicModel, AnthropicProfile
from jharness.models.deepseek import deepseek_anthropic_profile, deepseek_openai_chat_profile
from jharness.models.openai import OpenAIChatCompletionsModel, OpenAIChatCompletionsProfile


@pytest.mark.parametrize("model_type", [OpenAIChatCompletionsModel, AnthropicModel])
def test_model_clients_share_constructor_validation(model_type: type[object]) -> None:
    constructor = cast(Any, model_type)
    for keywords, pattern in (
        ({"base_url": "", "api_key": "secret", "model": "model"}, "base_url"),
        ({"base_url": "https://x", "api_key": "", "model": "model"}, "api_key"),
        ({"base_url": "https://x", "api_key": "secret", "model": ""}, "model"),
    ):
        with pytest.raises(ValueError, match=pattern):
            constructor(**keywords)
    with pytest.raises(TypeError, match="unexpected keyword argument 'unknown'"):
        constructor(
            base_url="https://provider.test",
            api_key="secret",
            model="model",
            unknown=True,
        )
    configured = constructor(
        base_url="https://provider.test/",
        api_key="secret",
        model="model",
    )
    assert configured.base_url == "https://provider.test"
    assert not hasattr(configured, "api_key")
    assert configured._api_key == "secret"
    assert isinstance(configured._timeout, httpx.Timeout)
    assert configured._timeout.connect == 10.0
    assert configured._timeout.read == 60.0

    without_transport_timeout = constructor(
        base_url="https://provider.test",
        api_key="secret",
        model="model",
        timeout=None,
    )
    assert without_transport_timeout._timeout is None

    for keywords, pattern in (
        ({"max_sse_line_bytes": 0}, "max_sse_line_bytes"),
        (
            {"max_sse_line_bytes": 20, "max_sse_event_bytes": 10},
            "max_sse_event_bytes",
        ),
    ):
        with pytest.raises(ValueError, match=pattern):
            constructor(
                base_url="https://provider.test",
                api_key="secret",
                model="model",
                **keywords,
            )


def test_shared_transport_config_repr_redacts_api_key() -> None:
    config = model_client_config(
        base_url="https://provider.test",
        api_key="repr-secret",
        model="model",
        options={},
        default_profile=object(),
        constructor_name="test",
    )

    assert config.api_key == "repr-secret"
    assert "repr-secret" not in repr(config)


def test_deepseek_profiles_drive_capabilities_without_runtime_special_cases() -> None:
    thinking_profile = deepseek_openai_chat_profile(thinking=True, effort="high")
    plain_profile = deepseek_anthropic_profile(thinking=False)
    openai_model = OpenAIChatCompletionsModel(
        base_url="https://provider.test",
        api_key="secret",
        model="deepseek",
        profile=thinking_profile,
    )
    anthropic_model = AnthropicModel(
        base_url="https://provider.test",
        api_key="secret",
        model="deepseek",
        profile=plain_profile,
    )

    assert thinking_profile.extra_request_body["thinking"] == {"type": "enabled"}
    assert thinking_profile.extra_request_body["reasoning_effort"] == "high"
    assert thinking_profile.reasoning_content_mode == "required_with_tools"
    assert thinking_profile.supports_seed is False
    assert openai_model.capabilities.tools is True
    assert openai_model.capabilities.tool_choice is False
    assert openai_model.capabilities.multimodal_input is False
    assert plain_profile.supports_redacted_thinking is False
    assert anthropic_model.capabilities.tools is True
    assert anthropic_model.capabilities.multimodal_input is False


def test_openai_profile_validates_every_configuration_family() -> None:
    profile = OpenAIChatCompletionsProfile(finish_reason_map={"stop": "end_turn"})
    assert profile.finish_reason(None) is None
    assert profile.finish_reason("stop") == "end_turn"
    assert profile.finish_reason("length") == "length"

    invalid: tuple[tuple[dict[str, Any], type[Exception], str], ...] = (
        ({"name": ""}, ValueError, "profile name"),
        ({"supports_streaming": 1}, TypeError, "must be a bool"),
        ({"supports_seed": 1}, TypeError, "must be a bool"),
        (
            {"requires_assistant_content_for_tool_calls": 1},
            TypeError,
            "must be a bool",
        ),
        ({"reasoning_content_mode": "other"}, ValueError, "reasoning_content_mode"),
        ({"max_tokens_field": "other"}, ValueError, "max_tokens_field"),
        ({"system_content_mode": "other"}, ValueError, "system_content_mode"),
        ({"json_schema_name": ""}, ValueError, "json_schema_name"),
        ({"extra_request_body": 1}, TypeError, "must be a mapping"),
        ({"extra_request_body": {"": 1}}, ValueError, "keys must be non-empty"),
        ({"finish_reason_map": 1}, TypeError, "must be a mapping"),
        ({"finish_reason_map": {"": "x"}}, ValueError, "keys"),
        ({"finish_reason_map": {"x": ""}}, ValueError, "values"),
    )
    for keywords, error, pattern in invalid:
        with pytest.raises(error, match=pattern):
            OpenAIChatCompletionsProfile(**cast(Any, keywords))

    for finish_reason_map, message in (
        ({"": "stop"}, "finish_reason_map keys must be a non-empty string"),
        ({"stop": ""}, "finish_reason_map values must be a non-empty string"),
    ):
        with pytest.raises(ValueError) as caught:
            OpenAIChatCompletionsProfile(finish_reason_map=finish_reason_map)
        assert str(caught.value) == message


def test_anthropic_profile_validates_every_configuration_family() -> None:
    profile = AnthropicProfile(finish_reason_map={"end_turn": "stop"})
    assert profile.finish_reason(None) is None
    assert profile.finish_reason("end_turn") == "stop"
    assert profile.finish_reason("max_tokens") == "max_tokens"

    invalid: tuple[tuple[dict[str, Any], type[Exception], str], ...] = (
        ({"name": ""}, ValueError, "profile name"),
        ({"anthropic_version": ""}, ValueError, "anthropic_version"),
        ({"supports_streaming": 1}, TypeError, "must be a bool"),
        ({"supports_redacted_thinking": 1}, TypeError, "must be a bool"),
        ({"auth_scheme": "other"}, ValueError, "auth_scheme"),
        ({"system_content_mode": "other"}, ValueError, "system_content_mode"),
        ({"default_max_tokens": True}, TypeError, "must be an integer"),
        ({"default_max_tokens": 0}, ValueError, "must be >= 1"),
        ({"seed_field": ""}, ValueError, "seed_field"),
        ({"file_ref_beta_header": ""}, ValueError, "file_ref_beta_header"),
        ({"json_object_schema": 1}, TypeError, "must be a mapping"),
        ({"extra_output_config": {"": 1}}, ValueError, "keys must be non-empty"),
        ({"extra_headers": 1}, TypeError, "must be a mapping"),
        ({"extra_headers": {"": "x"}}, ValueError, "keys"),
        ({"extra_headers": {"x": ""}}, ValueError, "values"),
        ({"finish_reason_map": 1}, TypeError, "must be a mapping"),
        ({"finish_reason_map": {"": "x"}}, ValueError, "keys"),
        ({"finish_reason_map": {"x": ""}}, ValueError, "values"),
    )
    for keywords, error, pattern in invalid:
        with pytest.raises(error, match=pattern):
            AnthropicProfile(**cast(Any, keywords))


def test_deepseek_profiles_validate_thinking_and_effort_combinations() -> None:
    plain = deepseek_openai_chat_profile(thinking=False)
    openai_thinking = deepseek_openai_chat_profile(thinking=True, effort="high")
    thinking = deepseek_anthropic_profile(thinking=True, effort="max")
    assert plain.name.endswith("nonthinking")
    assert plain.extra_request_body["thinking"] == {"type": "disabled"}
    assert plain.reasoning_content_mode == "live_only"
    assert plain.supports_seed is False
    assert plain.supports_tools is True
    assert openai_thinking.extra_request_body == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }
    assert openai_thinking.reasoning_content_mode == "required_with_tools"
    assert openai_thinking.supports_tools is True
    assert openai_thinking.supports_tool_choice is False
    assert openai_thinking.supports_parallel_tool_calls is True
    assert openai_thinking.supports_parallel_tool_call_control is False
    assert openai_thinking.requires_assistant_content_for_tool_calls is True
    assert thinking.name.endswith("thinking")
    assert thinking.extra_request_body == {"thinking": {"type": "enabled"}}
    assert thinking.extra_output_config == {"effort": "max"}
    assert thinking.supports_redacted_thinking is False
    with pytest.raises(ValueError, match="thinking must be a bool"):
        deepseek_openai_chat_profile(thinking=cast(Any, 1))
    with pytest.raises(ValueError, match="effort must be one of"):
        deepseek_openai_chat_profile(thinking=True, effort=cast(Any, "low"))
    for factory in (deepseek_openai_chat_profile, deepseek_anthropic_profile):
        with pytest.raises(ValueError, match="only valid"):
            factory(thinking=False, effort="high")
