from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import httpx
import pytest

from jharness.kernel import ContentPart, ModelDelta, ModelError, ModelErrorInfo, ModelResponse
from jharness.models._http import (
    ModelErrorPolicy,
    invoke_json_model,
    invoke_sse_model,
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
    body_request_id_key="request_id",
)


def _decoded_response(_value: Mapping[str, object]) -> ModelResponse:
    return ModelResponse(parts=(ContentPart.text_part("ok"),))


async def _invoke_json(handler: Callable[[httpx.Request], httpx.Response]) -> ModelResponse:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await invoke_json_model(
            client=client,
            timeout=None,
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
) -> None:
    assert (info.code, info.status_code, info.retryable, info.request_id) == (
        code,
        status,
        retryable,
        request_id,
    )
    assert info.metadata == ({} if location is None else {"location": location})


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
            {"x-request-id": "header-id", "location": "https://provider.test/status"},
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
        status=200,
        retryable=False,
        request_id="header-id",
        location="https://provider.test/stream",
    )
