from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any, cast

import pytest

from jharness.kernel import (
    Checkpoint,
    ContentPart,
    DeltaSink,
    Event,
    EventKind,
    Failed,
    Invocation,
    Message,
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RequestError,
    RunContext,
    Runtime,
    Suspended,
    SuspensionSelector,
    ToolCall,
    ToolChoice,
)
from jharness.kernel.wire import decode_checkpoint, encode_checkpoint
from jharness.toolkit import ToolRegistry
from jharness.tools.interaction import (
    AskQuestionTool,
    QuestionResponse,
    extract_question_request,
    resume_question,
)


def _questions() -> list[dict[str, Any]]:
    return [
        {
            "id": "confirmed",
            "kind": "confirm",
            "prompt": "Proceed with the plan?",
            "required": True,
            "default": True,
        },
        {
            "id": "database",
            "kind": "single_choice",
            "prompt": "Choose a database",
            "options": [
                {"value": "postgres", "label": "PostgreSQL"},
                {"value": "sqlite", "label": "SQLite"},
            ],
            "allow_custom": True,
        },
        {
            "id": "features",
            "kind": "multiple_choice",
            "prompt": "Choose features",
            "options": [
                {"value": "cache", "label": "Cache"},
                {"value": "audit", "label": "Audit"},
                {"value": "search", "label": "Search"},
            ],
            "min_selections": 1,
            "max_selections": 2,
        },
        {
            "id": "notes",
            "kind": "text",
            "prompt": "Add implementation notes",
            "multiline": True,
            "min_length": 2,
            "max_length": 80,
        },
        {
            "id": "retries",
            "kind": "number",
            "prompt": "Choose a retry count",
            "minimum": 0,
            "maximum": 10,
            "step": 1,
            "integer_only": True,
        },
        {
            "id": "deadline",
            "kind": "date",
            "prompt": "Choose a deadline",
            "minimum": "2026-07-01",
            "maximum": "2026-12-31",
        },
        {
            "id": "confidence",
            "kind": "scale",
            "prompt": "Rate confidence",
            "minimum": 1,
            "maximum": 5,
            "step": 1,
            "minimum_label": "Low",
            "maximum_label": "High",
        },
        {
            "id": "priorities",
            "kind": "ranking",
            "prompt": "Rank priorities",
            "options": [
                {"value": "correctness", "label": "Correctness"},
                {"value": "speed", "label": "Speed"},
                {"value": "simplicity", "label": "Simplicity"},
            ],
            "min_ranked": 2,
            "max_ranked": 3,
        },
    ]


def _answers() -> dict[str, Any]:
    return {
        "confirmed": True,
        "database": "postgres",
        "features": ["cache", "audit"],
        "notes": "Keep it durable",
        "retries": 3,
        "deadline": "2026-08-15",
        "confidence": 4,
        "priorities": ["correctness", "simplicity", "speed"],
    }


def _external_payload(request: ModelRequest) -> dict[str, Any] | None:
    responses = [
        message
        for message in request.messages
        if message.role == "external" and message.metadata.get("kind") == "question_response"
    ]
    if not responses:
        return None
    assert len(responses) == 1
    message = responses[0]
    assert len(message.parts) == 1
    text = message.parts[0].text
    assert text is not None
    prefix = "AskQuestion response:\n"
    assert text.startswith(prefix)
    value = json.loads(text.removeprefix(prefix))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


class _TranscriptQuestionModel(Model):
    def __init__(self, questions: list[dict[str, Any]]) -> None:
        self._questions = questions
        self.requests: list[ModelRequest] = []
        self.observed_response: dict[str, Any] | None = None

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        del context, stream, emit_delta
        self.requests.append(request)
        assert [spec.name for spec in request.tools] == ["AskQuestion"]
        assert request.tool_choice.allow_parallel_tool_calls is False

        response = _external_payload(request)
        if response is None:
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        "ask-real",
                        "AskQuestion",
                        {"questions": self._questions},
                    ),
                )
            )

        self.observed_response = response
        if response["status"] == "cancelled":
            reason = response.get("reason") or "no reason"
            return ModelResponse(
                (ContentPart.text_part(f"Question was cancelled: {reason}"),),
                finish_reason="stop",
            )

        answers = cast(Mapping[str, Any], response["answers"])
        return ModelResponse(
            (
                ContentPart.text_part(
                    f"Selected {answers['database']} with {answers['retries']} retries"
                ),
            ),
            finish_reason="stop",
        )


class _MultipleQuestionCallsModel(Model):
    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        del context, stream, emit_delta
        assert request.tool_choice.allow_parallel_tool_calls is False
        question = {
            "id": "confirm",
            "kind": "confirm",
            "prompt": "Continue?",
        }
        return ModelResponse(
            tool_calls=(
                ToolCall("ask-first", "AskQuestion", {"questions": [question]}),
                ToolCall("ask-second", "AskQuestion", {"questions": [question]}),
            )
        )


async def _collect(invocation: Invocation) -> tuple[Checkpoint, list[Event]]:
    events = invocation.events()
    result_task = asyncio.create_task(invocation.result())
    observed = [event async for event in events]
    return await result_task, observed


def _runtime(model: Model) -> Runtime:
    return Runtime(
        model=model,
        tools=ToolRegistry((AskQuestionTool(),)),
        tool_choice=ToolChoice(allow_parallel_tool_calls=False),
    )


def test_runtime_question_checkpoint_json_roundtrip_and_fresh_runtime_resume() -> None:
    initial_model = _TranscriptQuestionModel(_questions())
    paused, initial_events = asyncio.run(
        _collect(_runtime(initial_model).start((Message.user("Configure the implementation"),)))
    )

    assert isinstance(paused.snapshot.state, Suspended)
    assert paused.snapshot.status == "suspended"
    assert len(initial_model.requests) == 1
    assert [event.kind for event in initial_events].count(EventKind.TOOL_STARTED) == 1
    assert [event.kind for event in initial_events].count(EventKind.TOOL_FINISHED) == 1

    wire_value = json.loads(json.dumps(encode_checkpoint(paused)))
    restored = decode_checkpoint(wire_value)
    assert restored == paused

    question_request = extract_question_request(restored)
    assert question_request.request_id.endswith(":ask-real")
    assert [question["id"] for question in question_request.questions] == [
        question["id"] for question in _questions()
    ]

    response = QuestionResponse.answered(
        question_request.request_id,
        "response-real",
        _answers(),
    )
    resumed_model = _TranscriptQuestionModel(_questions())
    resumed, resume_events = asyncio.run(
        _collect(resume_question(_runtime(resumed_model), restored, response))
    )

    assert resumed.snapshot.status == "completed"
    assert resumed.snapshot.history[-1].parts[0].text == "Selected postgres with 3 retries"
    assert len(resumed_model.requests) == 1
    assert resumed_model.observed_response == {
        "answers": _answers(),
        "request_id": question_request.request_id,
        "response_id": "response-real",
        "status": "answered",
    }
    assert EventKind.TOOL_STARTED not in [event.kind for event in resume_events]
    assert [message.role for message in resumed.snapshot.history] == [
        "user",
        "assistant",
        "tool",
        "external",
        "assistant",
    ]


def test_runtime_cancelled_question_is_visible_to_fresh_model() -> None:
    questions = [
        {
            "id": "confirmed",
            "kind": "confirm",
            "prompt": "Proceed?",
        }
    ]
    paused = asyncio.run(
        _runtime(_TranscriptQuestionModel(questions))
        .start((Message.user("Ask before proceeding"),))
        .result()
    )
    restored = decode_checkpoint(json.loads(json.dumps(encode_checkpoint(paused))))
    request = extract_question_request(restored)
    response = QuestionResponse.cancelled(
        request.request_id,
        "response-cancelled",
        "user chose not to decide",
    )

    resumed_model = _TranscriptQuestionModel(questions)
    completed = asyncio.run(resume_question(_runtime(resumed_model), restored, response).result())

    assert completed.snapshot.status == "completed"
    assert completed.snapshot.history[-1].parts[0].text == (
        "Question was cancelled: user chose not to decide"
    )
    assert resumed_model.observed_response == {
        "answers": {},
        "reason": "user chose not to decide",
        "request_id": request.request_id,
        "response_id": "response-cancelled",
        "status": "cancelled",
    }


def test_runtime_rejects_wrong_selector_and_response_request() -> None:
    model = _TranscriptQuestionModel(_questions())
    runtime = _runtime(model)
    paused = asyncio.run(runtime.start((Message.user("Ask me"),)).result())
    request = extract_question_request(paused)

    with pytest.raises(RequestError) as mismatch:
        runtime.resume(
            paused,
            selector=SuspensionSelector(wait_id="ask:not-this-request"),
        )
    assert mismatch.value.code == "suspension_mismatch"

    wrong_response = QuestionResponse.cancelled(
        "ask:not-this-request",
        "response-wrong",
    )
    with pytest.raises(ValueError, match="request_id"):
        resume_question(runtime, paused, wrong_response)

    correct_response = QuestionResponse.cancelled(
        request.request_id,
        "response-correct",
    )
    assert isinstance(resume_question(runtime, paused, correct_response), Invocation)


def test_runtime_disallowed_multiple_question_calls_fail_before_tools_start() -> None:
    checkpoint, events = asyncio.run(
        _collect(
            _runtime(_MultipleQuestionCallsModel()).start(
                (Message.user("Ask exactly one question batch"),)
            )
        )
    )

    state = checkpoint.snapshot.state
    assert isinstance(state, Failed)
    assert state.error.code == "model_protocol_error"
    assert "disallowed parallel tool calls" in state.error.message
    kinds = [event.kind for event in events]
    assert EventKind.MODEL_STARTED in kinds
    assert EventKind.TOOL_STARTED not in kinds
    assert EventKind.TOOL_FINISHED not in kinds
