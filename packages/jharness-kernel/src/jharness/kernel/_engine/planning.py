"""One model effect translated into one typed Change."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from time import time
from typing import Any, Protocol, cast

from jharness.kernel._engine.change import Change, failed, insert, limited, suspend
from jharness.kernel._engine.deadline import (
    Deadline,
    EffectInterrupted,
    WorkDeadlineReached,
    await_effect,
)
from jharness.kernel._validation import expect_instance
from jharness.kernel.checkpoint import ModelTurnFact, ModelTurnResult
from jharness.kernel.control import ControlInbox, Insert, Pause
from jharness.kernel.errors import ModelError
from jharness.kernel.events import EventKind
from jharness.kernel.limits import LimitReason, RunLimits
from jharness.kernel.models import (
    Model,
    ModelCapabilities,
    ModelContentDelta,
    ModelDelta,
    ModelOptions,
    ModelReasoningDelta,
    ModelRequest,
    ModelResponse,
    ModelToolCallDelta,
    ModelUsage,
    ModelUsageDelta,
    ResponseFormat,
    ToolChoice,
)
from jharness.kernel.snapshot import RunSnapshot
from jharness.kernel.state import Completed, Limited, Planning, ToolsPending
from jharness.kernel.tools import ToolCatalog

_TEXT_LIKE = frozenset({"text", "reasoning", "thinking", "redacted_thinking", "refusal"})


class Emit(Protocol):
    def __call__(self, kind: EventKind, data: Mapping[str, Any]) -> Awaitable[None]: ...


class PlanningStep:
    __slots__ = (
        "_capabilities",
        "_catalog",
        "_emit",
        "_limits",
        "_model",
        "_options",
        "_response_format",
        "_stream",
        "_tool_choice",
    )

    def __init__(
        self,
        *,
        model: Model,
        capabilities: ModelCapabilities,
        catalog: ToolCatalog,
        limits: RunLimits,
        options: ModelOptions,
        tool_choice: ToolChoice,
        response_format: ResponseFormat | None,
        stream: bool,
        emit: Emit,
    ) -> None:
        self._model = model
        self._capabilities = capabilities
        self._catalog = catalog
        self._limits = limits
        self._options = options
        self._tool_choice = tool_choice
        self._response_format = response_format
        self._stream = stream
        self._emit = emit

    async def run(
        self, snapshot: RunSnapshot, *, deadline: Deadline, inbox: ControlInbox
    ) -> Change:
        request = ModelRequest(
            snapshot.history,
            self._catalog.specs(),
            self._options,
            self._tool_choice,
            self._response_format,
        )
        await self._emit(
            EventKind.MODEL_STARTED,
            {"planning_step": snapshot.metrics.planning_steps + 1},
        )
        try:
            response = await self._invoke(request, snapshot, deadline, inbox)
        except EffectInterrupted as interrupted:
            return _interrupted(interrupted)
        except WorkDeadlineReached:
            return limited(LimitReason.DEADLINE)
        except ModelError as exc:
            return failed("model_provider_error", exc.info.message)
        except Exception as exc:
            return failed("model_protocol_error", str(exc) or exc.__class__.__name__)

        await self._emit(
            EventKind.MODEL_FINISHED,
            {
                "finish_reason": response.finish_reason,
                "tool_call_count": len(response.tool_calls),
                "usage": usage_data(response.usage),
            },
        )
        return self._change(snapshot, response)

    async def _invoke(
        self,
        request: ModelRequest,
        snapshot: RunSnapshot,
        deadline: Deadline,
        inbox: ControlInbox,
    ) -> ModelResponse:
        _validate_request(request, self._capabilities)
        use_stream = self._stream and self._capabilities.streaming

        async def emit_delta(delta: ModelDelta) -> None:
            await self._emit(EventKind.MODEL_DELTA, delta_data(delta))

        response = await await_effect(
            self._model.invoke(
                request,
                snapshot.context,
                stream=use_stream,
                emit_delta=emit_delta if use_stream else None,
            ),
            deadline=deadline,
            inbox=inbox,
        )
        response = expect_instance(response, ModelResponse, "model response")
        _validate_response(response, request, self._capabilities)
        return response

    def _change(self, snapshot: RunSnapshot, response: ModelResponse) -> Change:
        total_tokens = snapshot.metrics.usage.total_tokens
        if response.usage is not None and response.usage.total_tokens is not None:
            total_tokens = (total_tokens or 0) + response.usage.total_tokens
        over_tokens = (
            self._limits.max_total_tokens is not None
            and total_tokens is not None
            and total_tokens > self._limits.max_total_tokens
        )
        if over_tokens:
            state = Limited(LimitReason.MAX_TOTAL_TOKENS)
        elif response.tool_calls:
            state = ToolsPending(response.tool_calls)
        else:
            state = Completed(response.parts)
        return Change(
            fact=ModelTurnFact(
                at=time(),
                result=ModelTurnResult(state.kind),
                part_count=len(response.parts),
                tool_call_ids=tuple(call.id for call in response.tool_calls),
                finish_reason=response.finish_reason,
                usage=response.usage,
                limit_reason=state.reason if isinstance(state, Limited) else None,
            ),
            state=state,
            append=(response.to_assistant_message(),),
            planning_steps=1,
            usage=response.usage,
        )


def _interrupted(interrupted: EffectInterrupted) -> Change:
    control = interrupted.control
    if isinstance(control, Pause):
        return suspend(Planning(), control.suspension)
    if isinstance(control, Insert):
        return insert(control)
    raise TypeError("unsupported planning interruption")


def usage_data(usage: ModelUsage | None) -> dict[str, int | None] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    }


def delta_data(delta: ModelDelta) -> Mapping[str, Any]:
    if isinstance(delta, ModelContentDelta):
        return {
            "kind": "content",
            "index": delta.index,
            "part_type": delta.part_type,
            "text_delta": delta.text_delta,
            "data": delta.data,
        }
    if isinstance(delta, ModelToolCallDelta):
        return {
            "kind": "tool_call",
            "index": delta.index,
            "id": delta.id,
            "name": delta.name,
            "arguments_delta": delta.arguments_delta,
        }
    if isinstance(delta, ModelReasoningDelta):
        return {"kind": "reasoning", "index": delta.index, "text_delta": delta.text_delta}
    if not isinstance(cast(object, delta), ModelUsageDelta):
        raise TypeError("model emitted an invalid delta")
    return {"kind": "usage", "usage": usage_data(delta.usage)}


def _validate_request(request: ModelRequest, capabilities: ModelCapabilities) -> None:
    if request.tools and not capabilities.tools:
        raise ValueError("model does not support tools")
    if request.tool_choice.type != "auto" and not capabilities.tool_choice:
        raise ValueError("model does not support explicit tool_choice")
    response_format = request.response_format
    if response_format is not None:
        if response_format.type == "json_object" and not capabilities.json_mode:
            raise ValueError("model does not support JSON object mode")
        if response_format.type == "json_schema" and not capabilities.structured_output:
            raise ValueError("model does not support structured output")
    if not capabilities.multimodal_input and any(
        part.type not in _TEXT_LIKE for message in request.messages for part in message.parts
    ):
        raise ValueError("model does not support multimodal input")


def _validate_response(
    response: ModelResponse,
    request: ModelRequest,
    capabilities: ModelCapabilities,
) -> None:
    calls = response.tool_calls
    choice = request.tool_choice
    if calls and not capabilities.tools:
        raise ValueError("model returned unsupported tool calls")
    if len(calls) > 1 and (
        not capabilities.parallel_tool_calls or not choice.allow_parallel_tool_calls
    ):
        raise ValueError("model returned disallowed parallel tool calls")
    if choice.type == "none" and calls:
        raise ValueError("model returned tool calls for tool_choice=none")
    if choice.type == "required" and not calls:
        raise ValueError("model omitted required tool call")
    if choice.type == "named" and any(call.name != choice.name for call in calls):
        raise ValueError("model returned a tool other than the named tool_choice")
    if not capabilities.multimodal_output and any(
        part.type not in _TEXT_LIKE for part in response.parts
    ):
        raise ValueError("model returned unsupported multimodal output")
