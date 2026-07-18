from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from time import time

import httpx
import pytest

from jharness.kernel import (
    ContentPart,
    ModelContentDelta,
    ModelDelta,
    ModelError,
    ModelErrorInfo,
    ModelResponse,
    RunContext,
)
from jharness.models._http import (
    ModelErrorPolicy,
    invoke_json_model,
    invoke_sse_model,
    iter_server_sent_events,
    stream_body_error,
)


class CodecError(ValueError):
    pass


_POLICY = ModelErrorPolicy(
    provider="test-provider",
    codec_error=CodecError,
    request_id_headers=("x-request-id",),
    error_code_keys=("code", "type"),
    additional_retryable_status_codes=frozenset({529}),
    retryable_error_codes=frozenset({"overloaded"}),
    body_request_id_key="request_id",
)


def _decoded_response(_value: Mapping[str, object]) -> ModelResponse:
    return ModelResponse(parts=(ContentPart.text_part("ok"),))


async def _invoke_json(handler: Callable[[httpx.Request], httpx.Response]) -> ModelResponse:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await invoke_json_model(
            client=client,
            timeout=None,
            context=RunContext("run-1", time()),
            url="https://provider.test/model",
            payload=dict,
            headers=lambda _body: {},
            decode=_decoded_response,
            errors=_POLICY,
            response_shape_error="response must be an object",
        )


def _assert_error(
    info: ModelErrorInfo,
    *,
    code: str,
    status: int | None,
    retryable: bool,
    request_id: str | None,
    location: str | None,
    retry_after: str | None,
) -> None:
    assert (info.code, info.status_code, info.retryable, info.request_id) == (
        code,
        status,
        retryable,
        request_id,
    )
    expected_metadata = {}
    if location is not None:
        expected_metadata["location"] = location
    if retry_after is not None:
        expected_metadata["retry_after"] = retry_after
    assert info.metadata == expected_metadata


def _error_body(
    code: str, *, key: str = "code", request_id: str | None = None
) -> dict[str, object]:
    body: dict[str, object] = {"error": {"message": "error", key: code}}
    if request_id is not None:
        body["request_id"] = request_id
    return body


@pytest.mark.parametrize(
    "kind,status,body,headers,code,retryable,request_id",
    [
        ("json", 400, _error_body("invalid"), {}, "invalid", False, None),
        (
            "json",
            429,
            _error_body("rate_limit", request_id="body-id"),
            {
                "x-request-id": "header-id",
                "location": "https://provider.test/status",
                "retry-after": "17",
            },
            "rate_limit",
            True,
            "header-id",
        ),
        (
            "json",
            529,
            _error_body("overloaded", key="type", request_id="body-id"),
            {},
            "overloaded",
            True,
            "body-id",
        ),
        (
            "json",
            200,
            _error_body("bad_envelope", request_id="body-id"),
            {"x-request-id": "header-id", "location": "https://provider.test/result"},
            "bad_envelope",
            False,
            "header-id",
        ),
        ("json", 200, [], {}, "codec_error", False, None),
        ("invalid_json", 200, None, {}, "invalid_json", False, None),
        ("invalid_json", 400, None, {}, "provider_error", False, None),
        ("json", 400, {"error": "plain"}, {}, "provider_error", False, None),
        ("timeout", None, None, {}, "timeout", True, None),
        ("network", None, None, {}, "ConnectError", True, None),
    ],
)
async def test_json_and_transport_error_matrix(
    kind: str,
    status: int | None,
    body: object,
    headers: Mapping[str, str],
    code: str,
    retryable: bool,
    request_id: str | None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if kind == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        if kind == "network":
            raise httpx.ConnectError("offline", request=request)
        assert status is not None
        if kind == "invalid_json":
            return httpx.Response(status, content=b"not-json", request=request)
        return httpx.Response(status, json=body, headers=headers, request=request)

    with pytest.raises(ModelError) as caught:
        await _invoke_json(handler)
    _assert_error(
        caught.value.info,
        code=code,
        status=status,
        retryable=retryable,
        request_id=request_id,
        location=headers.get("location"),
        retry_after=headers.get("retry-after"),
    )


async def test_sse_error_envelope_keeps_http_context() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"data: error\n\n",
            headers={
                "x-request-id": "header-id",
                "location": "https://provider.test/stream",
            },
            request=request,
        )

    def decode_frame(_event: str | None, _data: str) -> tuple[bool, Sequence[ModelDelta]]:
        raise stream_body_error(
            _error_body("stream_error", request_id="body-id"),
            _POLICY,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ModelError) as caught:
            await invoke_sse_model(
                client=client,
                timeout=None,
                context=RunContext("run-1", time()),
                url="https://provider.test/model",
                payload=dict,
                headers=lambda _body: {},
                decode_frame=decode_frame,
                completed_response=lambda: _decoded_response({}),
                emit_delta=None,
                errors=_POLICY,
                incomplete_error="stream incomplete",
            )

    _assert_error(
        caught.value.info,
        code="stream_error",
        status=None,
        retryable=False,
        request_id="header-id",
        location="https://provider.test/stream",
        retry_after=None,
    )


class ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: Sequence[bytes]) -> None:
        self.chunks = tuple(chunks)
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


async def test_sse_parser_preserves_unicode_separators_and_crlf_boundaries() -> None:
    payload = 'event: message\r\ndata: {"text":"a\u2028b"}\r\n\r\n'.encode()
    separator = payload.index(b"\xe2")
    stream = ChunkedStream(
        (
            payload[: separator + 1],
            payload[separator + 1 : separator + 2],
            payload[separator + 2 : -2],
            payload[-2:-1],
            payload[-1:],
        )
    )
    response = httpx.Response(
        200,
        stream=stream,
        request=httpx.Request("POST", "https://provider.test/model"),
    )
    try:
        events = [event async for event in iter_server_sent_events(response)]
    finally:
        await response.aclose()

    assert events == [
        type(events[0])(event="message", data='{"text":"a\u2028b"}'),
    ]
    assert stream.closed


@pytest.mark.parametrize(
    "payload,limits,pattern",
    [
        (b"data: abc\n\n", {"max_line_bytes": 7}, "SSE line exceeds"),
        (
            b": comment\ndata: x\n\n",
            {"max_line_bytes": 32, "max_event_bytes": 10},
            "SSE event exceeds",
        ),
        (b"data: \xff\n\n", {}, "invalid UTF-8"),
    ],
)
async def test_sse_parser_rejects_invalid_or_over_limit_input(
    payload: bytes,
    limits: Mapping[str, int],
    pattern: str,
) -> None:
    response = httpx.Response(
        200,
        content=payload,
        request=httpx.Request("POST", "https://provider.test/model"),
    )
    with pytest.raises(CodecError, match=pattern):
        _ = [
            event
            async for event in iter_server_sent_events(
                response,
                error=CodecError,
                **limits,
            )
        ]


async def test_sse_limit_failure_is_a_structured_model_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"data: too-long\n\n", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ModelError) as caught:
            await invoke_sse_model(
                client=client,
                timeout=None,
                context=RunContext("run-1", time()),
                url="https://provider.test/model",
                payload=dict,
                headers=lambda _body: {},
                decode_frame=lambda _event, _data: (True, ()),
                completed_response=lambda: _decoded_response({}),
                emit_delta=None,
                errors=_POLICY,
                incomplete_error="stream incomplete",
                max_sse_line_bytes=8,
                max_sse_event_bytes=16,
            )

    assert caught.value.info.code == "codec_error"
    assert caught.value.info.status_code == 200


class SinkFailure(ValueError):
    pass


async def test_delta_sink_failure_propagates_unchanged_and_closes_response() -> None:
    stream = ChunkedStream((b"data: chunk\n\n",))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, request=request)

    failure = SinkFailure("sink failed")

    async def emit_delta(_delta: ModelDelta, /) -> None:
        raise failure

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(SinkFailure) as caught:
            await invoke_sse_model(
                client=client,
                timeout=None,
                context=RunContext("run-1", time()),
                url="https://provider.test/model",
                payload=dict,
                headers=lambda _body: {},
                decode_frame=lambda _event, _data: (
                    True,
                    (ModelContentDelta(0, "chunk"),),
                ),
                completed_response=lambda: _decoded_response({}),
                emit_delta=emit_delta,
                errors=_POLICY,
                incomplete_error="stream incomplete",
            )

    assert caught.value is failure
    assert stream.closed


async def test_semantic_stream_overload_is_retryable_without_http_200_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"data: error\n\n", request=request)

    def decode_frame(_event: str | None, _data: str) -> tuple[bool, Sequence[ModelDelta]]:
        raise stream_body_error(_error_body("overloaded"), _POLICY)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ModelError) as caught:
            await invoke_sse_model(
                client=client,
                timeout=None,
                context=RunContext("run-1", time()),
                url="https://provider.test/model",
                payload=dict,
                headers=lambda _body: {},
                decode_frame=decode_frame,
                completed_response=lambda: _decoded_response({}),
                emit_delta=None,
                errors=_POLICY,
                incomplete_error="stream incomplete",
            )

    assert caught.value.info.status_code is None
    assert caught.value.info.retryable is True


async def test_run_deadline_clamps_injected_client_timeout_and_short_circuits_expiry() -> None:
    observed: list[Mapping[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request.extensions["timeout"])
        return httpx.Response(
            200,
            json={"parts": "unused"},
            request=request,
        )

    started = time()
    context = RunContext("run-1", started, deadline=started + 0.5)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await invoke_json_model(
            client=client,
            timeout=httpx.Timeout(60.0, connect=10.0),
            context=context,
            url="https://provider.test/model",
            payload=dict,
            headers=lambda _body: {},
            decode=_decoded_response,
            errors=_POLICY,
            response_shape_error="response must be an object",
        )
        with pytest.raises(ModelError) as caught:
            await invoke_json_model(
                client=client,
                timeout=None,
                context=RunContext("run-2", started, deadline=started - 1),
                url="https://provider.test/model",
                payload=dict,
                headers=lambda _body: {},
                decode=_decoded_response,
                errors=_POLICY,
                response_shape_error="response must be an object",
            )

    timeout_values = observed[0]
    assert all(
        isinstance(value, int | float) and 0 < value <= 0.5 for value in timeout_values.values()
    )
    assert len(observed) == 1
    assert caught.value.info.code == "timeout"
