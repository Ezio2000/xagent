"""HTTP client for Anthropic Messages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict, Unpack, cast

import httpx

from jharness.kernel import (
    DeltaSink,
    ModelCapabilities,
    ModelDelta,
    ModelRequest,
    ModelResponse,
    RunContext,
)
from jharness.models._http import (
    ModelErrorPolicy,
    decode_json_object,
    invoke_json_model,
    invoke_sse_model,
    model_client_config,
    stream_body_error,
)
from jharness.models.anthropic.errors import AnthropicError
from jharness.models.anthropic.messages_api.codec import AnthropicCodec
from jharness.models.anthropic.messages_api.stream import AnthropicStreamDecoder
from jharness.models.anthropic.profiles import AnthropicProfile

_ADDITIONAL_RETRYABLE_STATUS_CODES = frozenset({529})
_REQUEST_ID_HEADERS = ("request-id", "x-request-id", "x-ds-request-id")


class _AnthropicModelOptions(TypedDict, total=False):
    profile: AnthropicProfile | None
    timeout: float | httpx.Timeout | None
    headers: Mapping[str, str] | None
    client: httpx.AsyncClient | None


class AnthropicModel:
    """Model implementation backed by Anthropic Messages."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        **options: Unpack[_AnthropicModelOptions],
    ) -> None:
        config = model_client_config(
            base_url=base_url,
            api_key=api_key,
            model=model,
            options=options,
            default_profile=AnthropicProfile(),
            constructor_name="AnthropicModel.__init__",
        )
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.model = config.model
        self.profile = config.profile
        self.codec = AnthropicCodec(model=config.model, profile=config.profile)
        self._timeout = config.timeout
        self._headers = dict(config.headers)
        self._client = config.client
        self._errors = ModelErrorPolicy(
            provider=config.profile.name,
            codec_error=AnthropicError,
            request_id_headers=_REQUEST_ID_HEADERS,
            error_code_keys=("type", "code"),
            additional_retryable_status_codes=_ADDITIONAL_RETRYABLE_STATUS_CODES,
            body_request_id_key="request_id",
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            streaming=self.profile.supports_streaming,
            tools=self.profile.supports_tools,
            tool_choice=self.profile.supports_tool_choice,
            parallel_tool_calls=self.profile.supports_parallel_tool_calls,
            multimodal_input=(
                self.profile.supports_image_input or self.profile.supports_file_input
            ),
            multimodal_output=False,
            structured_output=self.profile.supports_json_schema,
            json_mode=self.profile.supports_json_object,
            usage_reporting=True,
        )

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        if not stream and emit_delta is not None:
            raise ValueError("emit_delta requires stream=True")
        if stream:
            decoder = AnthropicStreamDecoder(self.profile)
            return await invoke_sse_model(
                client=self._client,
                timeout=self._timeout,
                url=self._messages_url(),
                payload=lambda: self.codec.encode_request(request, stream=True),
                headers=self._request_headers,
                decode_frame=lambda event, data: self._decode_sse_data(event, data, decoder),
                completed_response=decoder.completed_response,
                emit_delta=emit_delta,
                errors=self._errors,
                incomplete_error="Anthropic stream ended before message_stop",
            )
        return await invoke_json_model(
            client=self._client,
            timeout=self._timeout,
            url=self._messages_url(),
            payload=lambda: self.codec.encode_request(request, stream=False),
            headers=self._request_headers,
            decode=self.codec.decode_response,
            errors=self._errors,
            response_shape_error="Anthropic response must be an object",
        )

    def _decode_sse_data(
        self,
        event_name: str | None,
        frame_data: str,
        decoder: AnthropicStreamDecoder,
    ) -> tuple[bool, list[ModelDelta]]:
        data = frame_data.strip()
        if not data:
            return False, []
        parsed_mapping = decode_json_object(
            data,
            AnthropicError,
            "Anthropic stream event must be an object",
        )
        if (event_name == "error" or parsed_mapping.get("type") == "error") and (
            "error" in parsed_mapping
        ):
            raise stream_body_error(parsed_mapping, self._errors)
        return decoder.apply_event(event_name, parsed_mapping)

    def _messages_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/messages"
        return f"{self.base_url}/v1/messages"

    def _request_headers(self, payload: Mapping[str, object]) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.profile.anthropic_version,
            **self.profile.extra_headers,
            **self._headers,
        }
        if _uses_file_source(payload):
            beta_header = self.profile.file_ref_beta_header
            if beta_header is None:
                raise AnthropicError(f"{self.profile.name} does not support file ref inputs")
            _ensure_header_value(headers, "anthropic-beta", beta_header)
        if self.profile.auth_scheme == "bearer":
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers["x-api-key"] = self.api_key
        return headers


def _uses_file_source(value: object) -> bool:
    if isinstance(value, Mapping):
        value_mapping = cast(Mapping[str, object], value)
        if value_mapping.get("type") == "file" and isinstance(value_mapping.get("file_id"), str):
            return True
        return any(_uses_file_source(item) for item in value_mapping.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return any(_uses_file_source(item) for item in cast(Sequence[object], value))
    return False


def _ensure_header_value(headers: dict[str, str], name: str, value: str) -> None:
    expected = name.lower()
    for key, existing in list(headers.items()):
        if key.lower() != expected:
            continue
        existing_values = {item.strip() for item in existing.split(",") if item.strip()}
        if value not in existing_values:
            headers[key] = f"{existing}, {value}" if existing else value
        return
    headers[name] = value
