"""HTTP client for OpenAI Chat Completions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict, Unpack

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
from jharness.models.openai.chat_completions.codec import OpenAIChatCompletionsCodec
from jharness.models.openai.chat_completions.stream import OpenAIChatStreamDecoder
from jharness.models.openai.errors import OpenAIChatCompletionsError
from jharness.models.openai.profiles import OpenAIChatCompletionsProfile

_REQUEST_ID_HEADERS = ("x-request-id", "x-ds-request-id")


class _OpenAIModelOptions(TypedDict, total=False):
    profile: OpenAIChatCompletionsProfile | None
    timeout: float | httpx.Timeout | None
    headers: Mapping[str, str] | None
    client: httpx.AsyncClient | None
    max_sse_line_bytes: int
    max_sse_event_bytes: int


class OpenAIChatCompletionsModel:
    """Model implementation backed by OpenAI Chat Completions."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        **options: Unpack[_OpenAIModelOptions],
    ) -> None:
        config = model_client_config(
            base_url=base_url,
            api_key=api_key,
            model=model,
            options=options,
            default_profile=OpenAIChatCompletionsProfile(),
            constructor_name="OpenAIChatCompletionsModel.__init__",
        )
        self.base_url = config.base_url
        self._api_key = config.api_key
        self.model = config.model
        self.profile = config.profile
        self.codec = OpenAIChatCompletionsCodec(model=config.model, profile=config.profile)
        self._timeout = config.timeout
        self._max_sse_line_bytes = config.max_sse_line_bytes
        self._max_sse_event_bytes = config.max_sse_event_bytes
        self._headers = dict(config.headers)
        self._client = config.client
        self._errors = ModelErrorPolicy(
            provider=config.profile.name,
            codec_error=OpenAIChatCompletionsError,
            request_id_headers=_REQUEST_ID_HEADERS,
            error_code_keys=("code", "type"),
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            streaming=self.profile.supports_streaming,
            tools=self.profile.supports_tools,
            tool_choice=self.profile.supports_tool_choice,
            parallel_tool_calls=self.profile.supports_parallel_tool_calls,
            multimodal_input=(
                self.profile.supports_image_input
                or self.profile.supports_video_input
                or self.profile.supports_file_input
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
            decoder = OpenAIChatStreamDecoder(self.profile)
            return await invoke_sse_model(
                client=self._client,
                timeout=self._timeout,
                context=context,
                url=self._chat_completions_url(),
                payload=lambda: self.codec.encode_request(request, stream=True),
                headers=lambda _payload: self._request_headers(),
                decode_frame=lambda _event, data: self._decode_sse_data(data, decoder),
                completed_response=decoder.completed_response,
                emit_delta=emit_delta,
                errors=self._errors,
                incomplete_error="chat completion stream ended before [DONE]",
                max_sse_line_bytes=self._max_sse_line_bytes,
                max_sse_event_bytes=self._max_sse_event_bytes,
            )
        return await invoke_json_model(
            client=self._client,
            timeout=self._timeout,
            context=context,
            url=self._chat_completions_url(),
            payload=lambda: self.codec.encode_request(request, stream=False),
            headers=lambda _payload: self._request_headers(),
            decode=self.codec.decode_response,
            errors=self._errors,
            response_shape_error="chat completion response must be an object",
        )

    def _decode_sse_data(
        self,
        frame_data: str,
        decoder: OpenAIChatStreamDecoder,
    ) -> tuple[bool, list[ModelDelta]]:
        data = frame_data.strip()
        if not data:
            return False, []
        if data == "[DONE]":
            return True, []
        parsed_mapping = decode_json_object(
            data,
            OpenAIChatCompletionsError,
            "chat completion stream chunk must be an object",
        )
        if "error" in parsed_mapping:
            raise stream_body_error(parsed_mapping, self._errors)
        return False, decoder.apply_chunk(parsed_mapping)

    def _chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _request_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._headers,
        }
