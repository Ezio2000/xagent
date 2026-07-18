"""Shared HTTP transport primitives for provider adapters."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from time import time
from typing import Any, Generic, TypeVar, cast

import httpx

from jharness.kernel import (
    DeltaSink,
    ModelDelta,
    ModelError,
    ModelErrorInfo,
    ModelResponse,
    RunContext,
)

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
_CLIENT_OPTION_NAMES = frozenset(
    {
        "client",
        "headers",
        "max_sse_event_bytes",
        "max_sse_line_bytes",
        "profile",
        "timeout",
    }
)
_DEFAULT_HTTP_TIMEOUT_SECONDS = 60.0
_DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
_DEFAULT_MAX_SSE_LINE_BYTES = 256 * 1024
_DEFAULT_MAX_SSE_EVENT_BYTES = 1024 * 1024

ProfileT = TypeVar("ProfileT")

PayloadFactory = Callable[[], Mapping[str, object]]
HeadersFactory = Callable[[Mapping[str, object]], Mapping[str, str]]
ResponseDecoder = Callable[[Mapping[str, object]], ModelResponse]
FrameDecoder = Callable[[str | None, str], tuple[bool, Sequence[ModelDelta]]]


@dataclass(frozen=True, slots=True)
class ModelClientConfig(Generic[ProfileT]):
    """Validated transport configuration shared by concrete provider clients."""

    base_url: str
    api_key: str = field(repr=False)
    model: str
    profile: ProfileT
    timeout: float | httpx.Timeout | None
    max_sse_line_bytes: int
    max_sse_event_bytes: int
    headers: Mapping[str, str]
    client: httpx.AsyncClient | None


def model_client_config(
    *,
    base_url: str,
    api_key: str,
    model: str,
    options: Mapping[str, Any],
    default_profile: ProfileT,
    constructor_name: str,
) -> ModelClientConfig[ProfileT]:
    """Normalize the common constructor surface for one provider client."""

    unexpected = set(options).difference(_CLIENT_OPTION_NAMES)
    if unexpected:
        option = min(unexpected)
        raise TypeError(f"{constructor_name}() got an unexpected keyword argument {option!r}")
    if not base_url:
        raise ValueError("base_url must not be empty")
    if not api_key:
        raise ValueError("api_key must not be empty")
    if not model:
        raise ValueError("model must not be empty")
    profile = cast(ProfileT | None, options.get("profile")) or default_profile
    timeout = cast(
        float | httpx.Timeout | None,
        (
            options["timeout"]
            if "timeout" in options
            else httpx.Timeout(
                _DEFAULT_HTTP_TIMEOUT_SECONDS,
                connect=_DEFAULT_CONNECT_TIMEOUT_SECONDS,
            )
        ),
    )
    max_sse_line_bytes = _positive_int_option(
        options.get("max_sse_line_bytes", _DEFAULT_MAX_SSE_LINE_BYTES),
        "max_sse_line_bytes",
    )
    max_sse_event_bytes = _positive_int_option(
        options.get("max_sse_event_bytes", _DEFAULT_MAX_SSE_EVENT_BYTES),
        "max_sse_event_bytes",
    )
    if max_sse_event_bytes < max_sse_line_bytes:
        raise ValueError("max_sse_event_bytes must be >= max_sse_line_bytes")
    return ModelClientConfig(
        base_url.rstrip("/"),
        api_key,
        model,
        profile,
        timeout,
        max_sse_line_bytes,
        max_sse_event_bytes,
        dict(cast(Mapping[str, str] | None, options.get("headers")) or {}),
        cast(httpx.AsyncClient | None, options.get("client")),
    )


def _positive_int_option(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ModelErrorPolicy:
    """Model-specific details needed by the shared transport error boundary."""

    provider: str
    codec_error: type[ValueError]
    request_id_headers: tuple[str, ...]
    error_code_keys: tuple[str, ...]
    additional_retryable_status_codes: frozenset[int] = frozenset()
    retryable_error_codes: frozenset[str] = frozenset()
    body_request_id_key: str | None = None


@dataclass(frozen=True, slots=True)
class ServerSentEvent:
    """One decoded SSE frame."""

    event: str | None
    data: str


async def invoke_json_model(
    *,
    client: httpx.AsyncClient | None,
    timeout: float | httpx.Timeout | None,
    context: RunContext,
    url: str,
    payload: PayloadFactory,
    headers: HeadersFactory,
    decode: ResponseDecoder,
    errors: ModelErrorPolicy,
    response_shape_error: str,
) -> ModelResponse:
    """Execute one JSON model request through the shared provider error boundary."""

    response: httpx.Response | None = None
    try:
        body = payload()
        async with managed_async_client(client, timeout) as http:
            response = await http.post(
                url,
                headers=headers(body),
                json=body,
                timeout=_effective_timeout(timeout, context),
            )
            await ensure_success_response(response, errors)
            value: object = response.json()
            if not isinstance(value, Mapping):
                raise errors.codec_error(response_shape_error)
            decoded = cast(Mapping[str, object], value)
            if "error" in decoded:
                raise ModelError(
                    _body_error_info(
                        decoded,
                        errors,
                        status_code=response.status_code,
                        response_text="provider response error",
                        request_id=response_request_id(response, errors.request_id_headers),
                        metadata=response_error_metadata(response),
                    )
                )
            return decode(decoded)
    except (ModelError, httpx.HTTPError, ValueError) as exc:
        raise _model_error(exc, response, errors) from exc


async def invoke_sse_model(
    *,
    client: httpx.AsyncClient | None,
    timeout: float | httpx.Timeout | None,
    context: RunContext,
    url: str,
    payload: PayloadFactory,
    headers: HeadersFactory,
    decode_frame: FrameDecoder,
    completed_response: Callable[[], ModelResponse],
    emit_delta: DeltaSink | None,
    errors: ModelErrorPolicy,
    incomplete_error: str,
    max_sse_line_bytes: int = _DEFAULT_MAX_SSE_LINE_BYTES,
    max_sse_event_bytes: int = _DEFAULT_MAX_SSE_EVENT_BYTES,
) -> ModelResponse:
    """Execute one SSE model request and return its provider-assembled response."""

    steps = _decoded_sse_steps(
        client=client,
        timeout=timeout,
        context=context,
        url=url,
        payload=payload,
        headers=headers,
        decode_frame=decode_frame,
        completed_response=completed_response,
        errors=errors,
        incomplete_error=incomplete_error,
        max_sse_line_bytes=max_sse_line_bytes,
        max_sse_event_bytes=max_sse_event_bytes,
    )
    async with aclosing(steps):
        async for deltas, completed in steps:
            if emit_delta is not None:
                for delta in deltas:
                    await emit_delta(delta)
            if completed is not None:
                return completed
    raise RuntimeError("provider stream ended without a result")


async def _decoded_sse_steps(
    *,
    client: httpx.AsyncClient | None,
    timeout: float | httpx.Timeout | None,
    context: RunContext,
    url: str,
    payload: PayloadFactory,
    headers: HeadersFactory,
    decode_frame: FrameDecoder,
    completed_response: Callable[[], ModelResponse],
    errors: ModelErrorPolicy,
    incomplete_error: str,
    max_sse_line_bytes: int,
    max_sse_event_bytes: int,
) -> AsyncGenerator[tuple[Sequence[ModelDelta], ModelResponse | None]]:
    """Own provider resources while yielding outside the host sink boundary."""

    response: httpx.Response | None = None
    try:
        body = payload()
        async with (
            managed_async_client(client, timeout) as http,
            http.stream(
                "POST",
                url,
                headers=headers(body),
                json=body,
                timeout=_effective_timeout(timeout, context),
            ) as response,
        ):
            await ensure_success_response(response, errors)
            async for frame in iter_server_sent_events(
                response,
                max_line_bytes=max_sse_line_bytes,
                max_event_bytes=max_sse_event_bytes,
                error=errors.codec_error,
            ):
                done, deltas = decode_frame(frame.event, frame.data)
                completed = completed_response() if done else None
                yield deltas, completed
                if done:
                    return
            raise errors.codec_error(incomplete_error)
    except (ModelError, httpx.HTTPError, ValueError) as exc:
        raise _model_error(exc, response, errors) from exc


@asynccontextmanager
async def managed_async_client(
    client: httpx.AsyncClient | None,
    timeout: float | httpx.Timeout | None,
) -> AsyncGenerator[httpx.AsyncClient]:
    """Reuse an injected client or own a short-lived provider client."""

    if client is not None:
        yield client
        return
    async with httpx.AsyncClient(timeout=timeout) as owned_client:
        yield owned_client


def _effective_timeout(
    configured: float | httpx.Timeout | None,
    context: RunContext,
) -> float | httpx.Timeout | None:
    deadline = context.deadline
    if deadline is None:
        return configured
    remaining = deadline - time()
    if remaining <= 0:
        raise httpx.ReadTimeout("run deadline expired before provider request")
    if configured is None:
        return remaining
    if isinstance(configured, httpx.Timeout):
        return httpx.Timeout(
            connect=_clamp_timeout_phase(configured.connect, remaining),
            read=_clamp_timeout_phase(configured.read, remaining),
            write=_clamp_timeout_phase(configured.write, remaining),
            pool=_clamp_timeout_phase(configured.pool, remaining),
        )
    return min(float(configured), remaining)


def _clamp_timeout_phase(value: float | None, remaining: float) -> float:
    return remaining if value is None else min(value, remaining)


async def ensure_success_response(
    response: httpx.Response,
    policy: ModelErrorPolicy,
) -> None:
    """Accept only 2xx responses and preserve the complete HTTP error envelope."""

    if 200 <= response.status_code < 300:
        return
    try:
        _ = response.text
    except httpx.ResponseNotRead:
        await response.aread()
    try:
        body = response.json()
    except ValueError:
        body = None
    raise ModelError(
        _body_error_info(
            body,
            policy,
            status_code=response.status_code,
            response_text=response.text or response.reason_phrase or "provider error",
            request_id=response_request_id(response, policy.request_id_headers),
            metadata=response_error_metadata(response),
        )
    )


async def iter_server_sent_events(
    response: httpx.Response,
    *,
    max_line_bytes: int = _DEFAULT_MAX_SSE_LINE_BYTES,
    max_event_bytes: int = _DEFAULT_MAX_SSE_EVENT_BYTES,
    error: type[ValueError] = ValueError,
) -> AsyncIterator[ServerSentEvent]:
    """Parse an HTTP response body into SSE frames according to field boundaries."""

    event_name: str | None = None
    data_lines: list[str] = []
    event_bytes = 0
    async for line, line_bytes in _iter_sse_lines(
        response,
        max_line_bytes=max_line_bytes,
        error=error,
    ):
        if line == "":
            if data_lines:
                yield ServerSentEvent(event=event_name, data="\n".join(data_lines))
            event_name = None
            data_lines = []
            event_bytes = 0
            continue
        event_bytes += line_bytes + 1
        if event_bytes > max_event_bytes:
            raise error(f"SSE event exceeds the configured {max_event_bytes}-byte limit")
        if line.startswith(":"):
            continue
        field, separator, raw_value = line.partition(":")
        if not separator:
            raw_value = ""
        value = raw_value[1:] if raw_value.startswith(" ") else raw_value
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        yield ServerSentEvent(event=event_name, data="\n".join(data_lines))


async def _iter_sse_lines(
    response: httpx.Response,
    *,
    max_line_bytes: int,
    error: type[ValueError],
) -> AsyncIterator[tuple[str, int]]:
    pending = bytearray()
    skip_lf = False
    async for chunk in response.aiter_bytes():
        for item in chunk:
            if skip_lf:
                skip_lf = False
                if item == 0x0A:
                    continue
            if item == 0x0D:
                yield _decode_sse_line(pending, error), len(pending)
                pending.clear()
                skip_lf = True
                continue
            if item == 0x0A:
                yield _decode_sse_line(pending, error), len(pending)
                pending.clear()
                continue
            if len(pending) >= max_line_bytes:
                raise error(f"SSE line exceeds the configured {max_line_bytes}-byte limit")
            pending.append(item)
    if pending:
        yield _decode_sse_line(pending, error), len(pending)


def _decode_sse_line(value: bytearray, error: type[ValueError]) -> str:
    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise error("SSE stream contains invalid UTF-8") from exc


def response_error_metadata(response: httpx.Response) -> dict[str, str]:
    """Return portable HTTP context that is useful beyond provider error bodies."""

    metadata: dict[str, str] = {}
    location = response.headers.get("location")
    if location:
        metadata["location"] = location
    retry_after = response.headers.get("retry-after")
    if retry_after:
        metadata["retry_after"] = retry_after
    return metadata


def response_request_id(
    response: httpx.Response,
    header_names: Sequence[str],
) -> str | None:
    """Return the first non-empty request identifier from provider header names."""

    for header_name in header_names:
        request_id = response.headers.get(header_name)
        if request_id:
            return request_id
    return None


def stream_body_error(
    body: Mapping[str, object],
    policy: ModelErrorPolicy,
) -> ModelError:
    """Build the standard model error for an error carried by an SSE payload."""

    return ModelError(
        _body_error_info(
            body,
            policy,
            status_code=None,
            response_text="provider stream error",
            request_id=None,
        )
    )


def decode_json_object(
    data: str,
    error: type[ValueError],
    error_message: str,
) -> Mapping[str, object]:
    parsed: object = json.loads(data)
    if not isinstance(parsed, Mapping):
        raise error(error_message)
    return cast(Mapping[str, object], parsed)


def _model_error(
    exc: Exception,
    response: httpx.Response | None,
    policy: ModelErrorPolicy,
) -> ModelError:
    semantic_model_error = isinstance(exc, ModelError)
    if semantic_model_error:
        info = exc.info
    else:
        if isinstance(exc, httpx.TimeoutException):
            code, retryable = "timeout", True
        elif isinstance(exc, httpx.NetworkError | httpx.RemoteProtocolError | httpx.ProxyError):
            code, retryable = exc.__class__.__name__, True
        elif isinstance(exc, httpx.HTTPError):
            code, retryable = exc.__class__.__name__, False
        elif isinstance(exc, policy.codec_error):
            code, retryable = "codec_error", False
        elif isinstance(exc, json.JSONDecodeError):
            code, retryable = "invalid_json", False
        else:
            code, retryable = exc.__class__.__name__, False
        info = ModelErrorInfo(
            message=str(exc) or exc.__class__.__name__,
            provider=policy.provider,
            code=code,
            retryable=retryable,
        )
    if response is not None:
        metadata = dict(info.metadata)
        metadata.update(response_error_metadata(response))
        response_status = response.status_code
        status_code = info.status_code
        if status_code is None and (not semantic_model_error or not 200 <= response_status < 300):
            status_code = response_status
        info = ModelErrorInfo(
            message=info.message,
            provider=info.provider,
            code=info.code,
            status_code=status_code,
            retryable=info.retryable,
            request_id=(
                response_request_id(response, policy.request_id_headers) or info.request_id
            ),
            metadata=metadata,
        )
    return ModelError(info)


def _body_error_info(
    error_body: object,
    policy: ModelErrorPolicy,
    *,
    status_code: int | None,
    response_text: str,
    request_id: str | None,
    metadata: Mapping[str, object] | None = None,
) -> ModelErrorInfo:
    message = response_text
    code: str | None = None
    body_request_id: str | None = None
    if isinstance(error_body, Mapping):
        error_mapping = cast(Mapping[str, object], error_body)
        if policy.body_request_id_key is not None:
            raw_request_id = error_mapping.get(policy.body_request_id_key)
            if isinstance(raw_request_id, str) and raw_request_id:
                body_request_id = raw_request_id
        error_value = error_mapping.get("error")
        if isinstance(error_value, Mapping):
            nested = cast(Mapping[str, object], error_value)
            raw_message = nested.get("message")
            if isinstance(raw_message, str) and raw_message:
                message = raw_message
            for key in policy.error_code_keys:
                item = nested.get(key)
                if isinstance(item, str) and item:
                    code = item
                    break
        elif isinstance(error_value, str) and error_value:
            message = error_value
    return ModelErrorInfo(
        message=message,
        provider=policy.provider,
        code=code or "provider_error",
        status_code=status_code,
        retryable=(
            status_code in _RETRYABLE_STATUS_CODES
            or status_code in policy.additional_retryable_status_codes
            or code in policy.retryable_error_codes
        ),
        request_id=request_id or body_request_id,
        metadata={} if metadata is None else metadata,
    )
