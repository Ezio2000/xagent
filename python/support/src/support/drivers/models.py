"""Scripted model drivers for controlled runtime scenarios."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from kernel import (
    ModelCapabilities,
    ModelContentDelta,
    ModelErrorInfo,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelToolCallDelta,
    RuntimeContext,
)


class ScriptedModel:
    """Model client driver that returns a fixed response sequence."""

    def __init__(self, steps: Sequence[ModelResponse]) -> None:
        self._steps = list(steps)
        self.calls = 0
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        self.requests.append(ModelRequest.from_dict(request.to_dict()))
        if self.calls >= len(self._steps):
            raise AssertionError("scripted model exhausted")
        response = self._steps[self.calls]
        self.calls += 1
        return ModelResponse.from_dict(response.to_dict())


class RequestCapturingModel:
    """Model driver that returns a final response and records the last request."""

    def __init__(self) -> None:
        self.request: ModelRequest | None = None

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        self.request = ModelRequest.from_dict(request.to_dict())
        return ModelResponse.text("done")


class StreamingTextModel:
    """Streaming model driver that emits two text deltas."""

    capabilities = ModelCapabilities(streaming=True)

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("stream path should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = request, context
        yield ModelContentDelta(index=0, text_delta="hel")
        yield ModelContentDelta(index=0, text_delta="lo")


class StreamingToolModel:
    """Streaming model driver that emits one tool call, then a final answer."""

    capabilities = ModelCapabilities(streaming=True)

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("stream path should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = context
        self.calls += 1
        if self.calls == 1:
            yield ModelToolCallDelta(index=0, id="call-1", name="echo")
            yield ModelToolCallDelta(index=0, arguments_delta='{"text":')
            yield ModelToolCallDelta(index=0, arguments_delta='"hello"}')
            return
        assert request.messages[-1].role == "tool"
        yield ModelContentDelta(index=0, text_delta="done")


class StreamingToolThenSlowModel:
    """Streaming model driver that hangs after a tool result delta."""

    capabilities = ModelCapabilities(streaming=True)

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("stream path should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = context
        self.calls += 1
        if self.calls == 1:
            yield ModelToolCallDelta(index=0, id="call-1", name="echo")
            yield ModelToolCallDelta(index=0, arguments_delta='{"text":"hello"}')
            return
        assert request.messages[-1].role == "tool"
        yield ModelContentDelta(index=0, text_delta="partial")
        await asyncio.sleep(1)


class SlowStreamingModel:
    """Streaming model driver that emits one delta and then blocks."""

    capabilities = ModelCapabilities(streaming=True)

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("stream path should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = request, context
        yield ModelContentDelta(index=0, text_delta="partial")
        await asyncio.sleep(1)


class FastStreamingModel:
    """Streaming model driver that emits two ready deltas."""

    capabilities = ModelCapabilities(streaming=True)

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("stream path should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = request, context
        yield ModelContentDelta(index=0, text_delta="first")
        yield ModelContentDelta(index=0, text_delta="second")


class CloseTrackingStreamingModel:
    """Streaming model driver that exposes whether its stream was closed."""

    capabilities = ModelCapabilities(streaming=True)

    def __init__(self) -> None:
        self.next_chunk_started = asyncio.Event()
        self.closed = asyncio.Event()

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("stream path should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = request, context
        try:
            yield ModelContentDelta(index=0, text_delta="partial")
            self.next_chunk_started.set()
            await asyncio.sleep(1)
        finally:
            self.closed.set()


class ProviderErrorModel:
    """Model driver that always raises a retryable provider error."""

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise ModelProviderError(
            ModelErrorInfo(
                message="provider unavailable",
                provider="test-provider",
                code="rate_limit",
                status_code=429,
                retryable=True,
                request_id="req-1",
            )
        )


class FlakyProviderErrorModel:
    """Model driver that fails once with a provider error and then recovers."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        self.calls += 1
        if self.calls == 1:
            raise ModelProviderError(
                ModelErrorInfo(
                    message="provider unavailable",
                    provider="test-provider",
                    code="rate_limit",
                    status_code=429,
                    retryable=True,
                    request_id="req-1",
                )
            )
        return ModelResponse.text("recovered")


class StreamingProviderErrorModel:
    """Streaming model driver that emits one delta and then raises a provider error."""

    capabilities = ModelCapabilities(streaming=True)

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("stream path should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = request, context
        yield ModelContentDelta(index=0, text_delta="partial")
        raise ModelProviderError(
            ModelErrorInfo(
                message="provider unavailable",
                provider="test-provider",
                code="rate_limit",
                status_code=429,
                retryable=True,
                request_id="req-1",
            )
        )


class SlowModel:
    """Model driver that blocks long enough for runtime deadline tests."""

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        await asyncio.sleep(1)
        return ModelResponse.text("late")


class AdapterTimeoutModel:
    """Model driver that raises an adapter timeout error."""

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise TimeoutError("provider timeout")


class CancellationConvertingModel:
    """Model driver that converts cancellation into a provider error."""

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError as exc:
            raise RuntimeError("provider converted cancellation") from exc
        return ModelResponse.text("late")


class CancellationSwallowingModel:
    """Model driver that swallows cancellation and completes late."""

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
        return ModelResponse.text("late")


class CancellationSwallowingThenFailingModel:
    """Model driver that swallows cancellation and then fails late."""

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)
            raise RuntimeError("late provider failure") from None
        return ModelResponse.text("late")


class ExternallyCancelledModel:
    """Model driver that marks its start before reacting to external cancellation."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        self.started.set()
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)
            raise RuntimeError("late provider failure") from None
        return ModelResponse.text("late")


class ContextInspectingModel:
    """Model driver that returns a selected runtime context metadata value."""

    def __init__(self, key: str = "tenant") -> None:
        self.key = key

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request
        return ModelResponse.text(str(context.metadata[self.key]))
