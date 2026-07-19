# pyright: reportPrivateUsage=false
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

import jharness.tools as tools
from jharness.kernel import (
    Checkpoint,
    ContentPart,
    DeltaSink,
    Message,
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    PendingToolCalls,
    Planning,
    RunContext,
    Runtime,
    SettledResult,
    Suspended,
    Suspension,
    SuspensionView,
    ToolBatchFact,
    ToolCall,
    ToolContext,
    ToolError,
    ToolFailure,
    ToolOutcomeKind,
    ToolsPending,
    ToolSuccess,
    ToolWaiting,
    WaitingResult,
    thaw_json_value,
)
from jharness.toolkit import Tool, ToolRegistry
from jharness.tools.interaction import (
    AskQuestionTool,
    QuestionRequest,
    QuestionResponse,
    extract_question_request,
    question_response_message,
    validate_question_response,
)
from jharness.tools.interaction import _schema as question_schema
from jharness.tools.interaction import response as question_response_module


async def _emit_progress(_progress: Mapping[str, Any]) -> None:
    return None


async def _invoke_async(
    tool: AskQuestionTool,
    arguments: Mapping[str, Any],
    *,
    cancelled: bool = False,
    through_registry: bool = False,
) -> WaitingResult | SettledResult:
    context = ToolContext(
        RunContext("question-run", time.time()),
        _emit_progress,
        lambda: cancelled,
    )
    call = ToolCall("question-call", tool.spec.name, arguments)
    if through_registry:
        catalog = await ToolRegistry((tool,)).open_catalog()
        return await catalog.bind(call).invoke(context)
    return await tool.invoke(call, context)


def _invoke(
    tool: AskQuestionTool,
    arguments: Mapping[str, Any],
    *,
    cancelled: bool = False,
    through_registry: bool = False,
) -> WaitingResult | SettledResult:
    return asyncio.run(
        _invoke_async(
            tool,
            arguments,
            cancelled=cancelled,
            through_registry=through_registry,
        )
    )


def _all_questions() -> list[dict[str, Any]]:
    return [
        {
            "id": "confirmed",
            "kind": "confirm",
            "prompt": "Proceed?",
            "description": "Confirm the complete operation",
            "required": True,
            "default": False,
        },
        {
            "id": "database",
            "kind": "single_choice",
            "prompt": "Choose a database",
            "required": True,
            "options": [
                {
                    "value": "postgres",
                    "label": "PostgreSQL",
                    "description": "Server database",
                },
                {"value": "sqlite", "label": "SQLite"},
            ],
            "allow_custom": True,
            "default": "postgres",
        },
        {
            "id": "features",
            "kind": "multiple_choice",
            "prompt": "Choose features",
            "required": True,
            "options": [
                {"value": "cache", "label": "Cache"},
                {"value": "audit", "label": "Audit"},
                {"value": "search", "label": "Search"},
            ],
            "allow_custom": False,
            "min_selections": 1,
            "max_selections": 2,
            "default": ["cache"],
        },
        {
            "id": "notes",
            "kind": "text",
            "prompt": "Add notes",
            "required": True,
            "multiline": True,
            "placeholder": "Implementation constraints",
            "min_length": 2,
            "max_length": 80,
            "default": "Durable",
        },
        {
            "id": "retries",
            "kind": "number",
            "prompt": "Maximum retries",
            "required": True,
            "minimum": 0,
            "maximum": 10,
            "step": 2,
            "integer_only": True,
            "default": 2,
        },
        {
            "id": "deadline",
            "kind": "date",
            "prompt": "Choose a deadline",
            "required": True,
            "minimum": "2026-07-01",
            "maximum": "2026-12-31",
            "default": "2026-08-15",
        },
        {
            "id": "confidence",
            "kind": "scale",
            "prompt": "Rate confidence",
            "required": True,
            "minimum": 1,
            "maximum": 5,
            "step": 0.5,
            "minimum_label": "Low",
            "maximum_label": "High",
            "default": 3,
        },
        {
            "id": "priorities",
            "kind": "ranking",
            "prompt": "Rank priorities",
            "required": True,
            "options": [
                {"value": "correctness", "label": "Correctness"},
                {"value": "speed", "label": "Speed"},
                {"value": "simplicity", "label": "Simplicity"},
            ],
            "min_ranked": 2,
            "max_ranked": 3,
            "default": ["correctness", "simplicity"],
        },
    ]


def _valid_answers() -> dict[str, Any]:
    return {
        "confirmed": True,
        "database": "postgres",
        "features": ["cache", "audit"],
        "notes": "Keep it durable",
        "retries": 4,
        "deadline": "2026-08-15",
        "confidence": 4.5,
        "priorities": ["correctness", "simplicity", "speed"],
    }


def _question_request(
    questions: Collection[Mapping[str, Any]] | None = None,
    *,
    max_answer_chars: int = 16_384,
) -> Any:
    selected = _all_questions() if questions is None else questions
    result = _invoke(
        AskQuestionTool(max_answer_chars=max_answer_chars),
        {"questions": list(selected)},
    )
    assert isinstance(result, WaitingResult)
    payload = thaw_json_value(result.outcome.structured_content)
    assert isinstance(payload, dict)
    request_id = payload["request_id"]
    text_limit = payload["max_text_chars"]
    enabled_kinds = payload["enabled_kinds"]
    max_questions = payload["max_questions"]
    max_options = payload["max_options"]
    max_prompt_chars = payload["max_prompt_chars"]
    normalized_questions = payload["questions"]
    assert isinstance(request_id, str)
    assert isinstance(text_limit, int)
    assert isinstance(enabled_kinds, list)
    assert all(isinstance(kind, str) for kind in enabled_kinds)
    assert isinstance(max_questions, int)
    assert isinstance(max_options, int)
    assert isinstance(max_prompt_chars, int)
    assert isinstance(normalized_questions, list)
    assert all(isinstance(question, dict) for question in normalized_questions)
    typed_questions = tuple(cast(dict[str, Any], question) for question in normalized_questions)
    return QuestionRequest(
        request_id,
        text_limit,
        typed_questions,
        tuple(cast(list[str], enabled_kinds)),
        max_questions,
        max_options,
        max_prompt_chars,
    )


def _failure(result: WaitingResult | SettledResult, code: str) -> ToolFailure:
    assert isinstance(result, SettledResult)
    outcome = result.outcome
    assert isinstance(outcome, ToolFailure)
    assert outcome.error.code == code
    assert outcome.structured_content is None
    return outcome


def test_ask_question_public_contract_constructor_registry_and_schema() -> None:
    tool = tools.AskQuestionTool(
        enabled_kinds={"confirm", "text", "number"},
        max_questions=3,
        max_options=4,
        max_prompt_chars=101,
        max_answer_chars=202,
    )

    assert "AskQuestionTool" in tools.__all__
    assert isinstance(tool, Tool)
    assert tool.enabled_kinds == frozenset({"confirm", "text", "number"})
    assert tool.max_questions == 3
    assert tool.max_options == 4
    assert tool.max_prompt_chars == 101
    assert tool.max_answer_chars == 202
    assert tool.spec.name == "AskQuestion"
    assert tool.spec.execution.concurrency == "serial"
    assert tool.spec.execution.read_only is True
    assert tool.spec.execution.idempotent is True
    assert tool.spec.parallel_safe is False
    assert tool.spec.risk.filesystem == "none"
    assert tool.spec.risk.network == "none"
    assert tool.spec.risk.subprocess is False
    assert tool.spec.risk.destructive is False
    assert tool.spec.risk.requires_approval is False

    schema = thaw_json_value(tool.spec.input_schema)
    assert isinstance(schema, dict)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["type"] == "object"
    assert schema["required"] == ["questions"]
    assert schema["additionalProperties"] is False
    questions_schema = cast(dict[str, Any], schema["properties"])["questions"]
    assert questions_schema["minItems"] == 1
    assert questions_schema["maxItems"] == 3
    variants = questions_schema["items"]["oneOf"]
    assert {variant["properties"]["kind"]["const"] for variant in variants} == {
        "confirm",
        "text",
        "number",
    }
    for variant in variants:
        assert variant["additionalProperties"] is False
        assert variant["properties"]["prompt"]["maxLength"] == 101

    output_schema = thaw_json_value(tool.spec.output_schema)
    assert isinstance(output_schema, dict)
    one_of = output_schema["oneOf"]
    assert isinstance(one_of, list)
    waiting = one_of[0]
    assert isinstance(waiting, dict)
    waiting_properties = waiting["properties"]
    assert isinstance(waiting_properties, dict)
    assert waiting_properties["status"] == {"const": "waiting"}
    assert waiting_properties["schema_version"] == {"const": 1}
    assert waiting_properties["enabled_kinds"] == {"const": ["confirm", "text", "number"]}
    assert waiting_properties["max_questions"] == {"const": 3}
    assert waiting_properties["max_options"] == {"const": 4}
    assert waiting_properties["max_prompt_chars"] == {"const": 101}
    assert waiting_properties["max_text_chars"] == {"const": 202}
    assert one_of[1] == {"type": "null"}

    async def registry_names() -> tuple[str, ...]:
        catalog = await ToolRegistry((tool,)).open_catalog()
        return tuple(spec.name for spec in catalog.specs())

    assert asyncio.run(registry_names()) == ("AskQuestion",)


def test_model_schema_does_not_publish_context_dependent_defaults() -> None:
    schema = thaw_json_value(AskQuestionTool().spec.input_schema)
    assert isinstance(schema, dict)
    questions_schema = cast(dict[str, Any], schema["properties"])["questions"]
    variants = questions_schema["items"]["oneOf"]
    by_kind = {
        variant["properties"]["kind"]["const"]: variant["properties"] for variant in variants
    }

    assert "default" not in by_kind["multiple_choice"]["min_selections"]
    assert "default" not in by_kind["text"]["min_length"]
    assert "default" not in by_kind["ranking"]["min_ranked"]
    assert by_kind["text"]["max_length"]["default"] == 16_384
    assert by_kind["scale"]["step"]["default"] == 1


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_questions", 0),
        ("max_options", 1),
        ("max_prompt_chars", 0),
        ("max_answer_chars", 0),
    ],
)
def test_ask_question_constructor_rejects_invalid_limits(keyword: str, value: int) -> None:
    with pytest.raises(ValueError, match=keyword):
        AskQuestionTool(**cast(Any, {keyword: value}))


@pytest.mark.parametrize(
    "keyword",
    ["max_questions", "max_options", "max_prompt_chars", "max_answer_chars"],
)
def test_ask_question_constructor_rejects_unserializable_huge_limits(keyword: str) -> None:
    with pytest.raises(ValueError, match="JSON-serializable schemas"):
        AskQuestionTool(**cast(Any, {keyword: 10**5000}))


@pytest.mark.parametrize(
    ("enabled_kinds", "error"),
    [
        (set[str](), ValueError),
        ({"text", "unsupported"}, ValueError),
        ("text", TypeError),
        (["text", 1], TypeError),
        (["text", "text"], ValueError),
    ],
)
def test_ask_question_constructor_rejects_invalid_enabled_kinds(
    enabled_kinds: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match=r"enabled_kinds|unsupported"):
        AskQuestionTool(enabled_kinds=cast(Any, enabled_kinds))


def test_all_eight_question_types_normalize_and_wait_without_mutating_input() -> None:
    tool = AskQuestionTool()
    questions = _all_questions()
    original = deepcopy(questions)
    result = _invoke(tool, {"questions": questions}, through_registry=True)

    assert questions == original
    assert isinstance(result, WaitingResult)
    outcome = result.outcome
    assert isinstance(outcome, ToolWaiting)
    assert len(outcome.parts) == 1
    assert outcome.parts[0].text is not None
    structured = thaw_json_value(outcome.structured_content)
    assert structured == {
        "status": "waiting",
        "schema_version": 1,
        "request_id": "ask-question:12:question-run:13:question-call",
        "enabled_kinds": [
            "confirm",
            "single_choice",
            "multiple_choice",
            "text",
            "number",
            "date",
            "scale",
            "ranking",
        ],
        "max_questions": 8,
        "max_options": 12,
        "max_prompt_chars": 2_000,
        "max_text_chars": 16_384,
        "questions": original,
    }
    assert result.suspension.reason == "human_input"
    assert result.suspension.source == "AskQuestion"
    assert result.suspension.wait_id == "ask-question:12:question-run:13:question-call"
    assert result.suspension.metadata == {
        "contract_id": question_schema.build_contract_id(
            question_schema.SUPPORTED_QUESTION_KINDS,
            max_questions=8,
            max_options=12,
            max_prompt_chars=2_000,
            max_answer_chars=16_384,
        ),
        "tool_call_id": "question-call",
        "schema_version": 1,
    }


def test_request_id_length_prefix_prevents_run_and_call_colon_collisions() -> None:
    first = question_schema.build_request_id("a:b", "c")
    second = question_schema.build_request_id("a", "b:c")

    assert first == "ask-question:3:a:b:1:c"
    assert second == "ask-question:1:a:3:b:c"
    assert first != second


def test_question_defaults_are_ui_hints_and_omitted_common_defaults_normalize() -> None:
    question = {
        "id": "notes",
        "kind": "text",
        "prompt": "Notes?",
        "default": "suggested only",
    }
    result = _invoke(AskQuestionTool(), {"questions": [question]})
    assert isinstance(result, WaitingResult)
    structured = thaw_json_value(result.outcome.structured_content)
    assert isinstance(structured, dict)
    normalized_questions = structured["questions"]
    assert isinstance(normalized_questions, list)
    normalized = normalized_questions[0]
    assert isinstance(normalized, dict)
    assert normalized["required"] is True
    assert normalized["default"] == "suggested only"


def test_cancel_requested_precedes_argument_validation() -> None:
    outcome = _failure(
        _invoke(AskQuestionTool(), {"not_questions": True}, cancelled=True),
        "cancelled",
    )
    assert "cancel" in outcome.error.message.lower()


@pytest.mark.parametrize(
    "arguments",
    [
        {"questions": "not-an-array"},
        {"questions": []},
        {"questions": [1]},
        {"questions": [{"id": "q", "prompt": "Missing kind"}]},
        {"questions": [{"id": "q", "kind": 1, "prompt": "Bad kind"}]},
        {"questions": [{"id": "q", "kind": "unknown", "prompt": "Bad kind"}]},
        {"questions": [{"id": "bad id", "kind": "confirm", "prompt": "Bad id"}]},
        {"questions": [{"id": "q", "kind": "confirm"}]},
        {"questions": [{"id": "q", "kind": "confirm", "prompt": "Extra", "extra": True}]},
        {"questions": [{"id": "q", "kind": "confirm", "prompt": 1, "required": True}]},
        {"questions": [{"id": "q", "kind": "confirm", "prompt": "Required", "required": 1}]},
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "single_choice",
                    "prompt": "Options",
                    "options": "not-an-array",
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "single_choice",
                    "prompt": "Options",
                    "options": [{"value": "a", "label": "A"}],
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "single_choice",
                    "prompt": "Options",
                    "options": [1, {"value": "b", "label": "B"}],
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "single_choice",
                    "prompt": "Options",
                    "options": [{"value": "a"}, {"value": "b", "label": "B"}],
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "single_choice",
                    "prompt": "Options",
                    "options": [
                        {"value": "a", "label": "A", "extra": True},
                        {"value": "b", "label": "B"},
                    ],
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "single_choice",
                    "prompt": "Default",
                    "options": [
                        {"value": "a", "label": "A"},
                        {"value": "b", "label": "B"},
                    ],
                    "default": "unknown",
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "multiple_choice",
                    "prompt": "Default",
                    "options": [
                        {"value": "a", "label": "A"},
                        {"value": "b", "label": "B"},
                    ],
                    "min_selections": 2,
                    "max_selections": 2,
                    "default": ["a"],
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "multiple_choice",
                    "prompt": "Default",
                    "options": [
                        {"value": "a", "label": "A"},
                        {"value": "b", "label": "B"},
                    ],
                    "default": "a",
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "text",
                    "prompt": "Default",
                    "min_length": 2,
                    "max_length": 3,
                    "default": "x",
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "number",
                    "prompt": "Step",
                    "step": 0,
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "number",
                    "prompt": "Integer",
                    "integer_only": True,
                    "default": 1.5,
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "number",
                    "prompt": "Minimum",
                    "minimum": 2,
                    "default": 1,
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "number",
                    "prompt": "Maximum",
                    "maximum": 2,
                    "default": 3,
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "number",
                    "prompt": "Step",
                    "step": 2,
                    "default": 3,
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "date",
                    "prompt": "Date",
                    "minimum": "2026-08-02",
                    "maximum": "2026-08-01",
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "date",
                    "prompt": "Date",
                    "minimum": "2026-08-02",
                    "default": "2026-08-01",
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "date",
                    "prompt": "Date",
                    "maximum": "2026-08-01",
                    "default": "2026-08-02",
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "scale",
                    "prompt": "Scale",
                    "minimum": 1,
                    "maximum": 5,
                    "step": 0,
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "ranking",
                    "prompt": "Rank",
                    "options": [
                        {"value": "a", "label": "A"},
                        {"value": "b", "label": "B"},
                    ],
                    "default": ["unknown"],
                }
            ]
        },
        {
            "questions": [
                {
                    "id": "q",
                    "kind": "ranking",
                    "prompt": "Rank",
                    "options": [
                        {"value": "a", "label": "A"},
                        {"value": "b", "label": "B"},
                    ],
                    "min_ranked": 2,
                    "max_ranked": 2,
                    "default": ["a"],
                }
            ]
        },
    ],
)
def test_direct_invocation_rejects_malformed_and_invalid_defaults(
    arguments: Mapping[str, Any],
) -> None:
    _failure(_invoke(AskQuestionTool(), arguments), "invalid_question")


def test_direct_invocation_rejects_a_disabled_kind() -> None:
    _failure(
        _invoke(
            AskQuestionTool(enabled_kinds={"text"}),
            {"questions": [{"id": "q", "kind": "confirm", "prompt": "Go?"}]},
        ),
        "invalid_question",
    )


def test_minimal_numeric_date_and_scale_questions_cover_normalized_defaults() -> None:
    questions: list[dict[str, Any]] = [
        {"id": "number", "kind": "number", "prompt": "Number"},
        {
            "id": "date_min",
            "kind": "date",
            "prompt": "Minimum date",
            "minimum": "2026-01-01",
        },
        {
            "id": "date_max",
            "kind": "date",
            "prompt": "Maximum date",
            "maximum": "2026-12-31",
        },
        {
            "id": "scale",
            "kind": "scale",
            "prompt": "Scale",
            "minimum": 1,
            "maximum": 5,
        },
    ]
    result = _invoke(AskQuestionTool(), {"questions": questions})
    assert isinstance(result, WaitingResult)
    payload = thaw_json_value(result.outcome.structured_content)
    assert isinstance(payload, dict)
    normalized = payload["questions"]
    assert isinstance(normalized, list)
    assert all(isinstance(question, dict) for question in normalized)
    typed = [cast(dict[str, Any], question) for question in normalized]
    assert typed[0]["integer_only"] is False
    assert "minimum" not in typed[0]
    assert "maximum" not in typed[0]
    assert typed[3]["step"] == 1


def test_core_rejects_multiple_custom_defaults_and_overlong_custom_default() -> None:
    multiple = {
        "id": "multiple",
        "kind": "multiple_choice",
        "prompt": "Multiple",
        "options": [
            {"value": "known-a", "label": "A"},
            {"value": "known-b", "label": "B"},
        ],
        "allow_custom": True,
        "max_selections": 3,
        "default": ["custom-one", "custom-two"],
    }
    _failure(
        _invoke(AskQuestionTool(), {"questions": [multiple]}),
        "invalid_question",
    )

    single = {
        "id": "single",
        "kind": "single_choice",
        "prompt": "Single",
        "options": [
            {"value": "known-a", "label": "A"},
            {"value": "known-b", "label": "B"},
        ],
        "allow_custom": True,
        "default": "custom-too-long",
    }
    _failure(
        _invoke(AskQuestionTool(max_answer_chars=2), {"questions": [single]}),
        "invalid_question",
    )

    single["default"] = "ok"
    accepted = _invoke(
        AskQuestionTool(max_answer_chars=2),
        {"questions": [single]},
    )
    assert isinstance(accepted, WaitingResult)


def test_core_accepts_numeric_default_without_step_and_rejects_scalar_types() -> None:
    accepted = _invoke(
        AskQuestionTool(),
        {
            "questions": [
                {
                    "id": "number",
                    "kind": "number",
                    "prompt": "Number",
                    "default": 2,
                }
            ]
        },
    )
    assert isinstance(accepted, WaitingResult)

    invalid_questions = (
        {
            "id": "text",
            "kind": "text",
            "prompt": "Txt",
            "min_length": "one",
        },
        {
            "id": "number",
            "kind": "number",
            "prompt": "Number",
            "minimum": "zero",
        },
        {
            "id": "date",
            "kind": "date",
            "prompt": "Date",
            "minimum": "2026/01/01",
        },
    )
    for question in invalid_questions:
        _failure(
            _invoke(AskQuestionTool(), {"questions": [question]}),
            "invalid_question",
        )


def test_core_rejects_noncanonical_date_defensively(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NonCanonicalDate:
        @classmethod
        def fromisoformat(cls, _value: str) -> _NonCanonicalDate:
            return cls()

        def isoformat(self) -> str:
            return "different"

    monkeypatch.setattr(question_schema, "date", _NonCanonicalDate)
    with pytest.raises(question_schema.QuestionValidationError, match="ISO date"):
        question_schema._iso_date("2026-01-01", "date")


def test_registry_rejects_structurally_invalid_arguments() -> None:
    tool = AskQuestionTool(max_questions=1, max_options=2, max_prompt_chars=8)
    invalid: tuple[Mapping[str, Any], ...] = (
        {},
        {"questions": []},
        {
            "questions": [
                {"id": "one", "kind": "confirm", "prompt": "First?"},
                {"id": "two", "kind": "confirm", "prompt": "Second?"},
            ]
        },
        {"questions": [{"id": "bad id", "kind": "confirm", "prompt": "Go?"}]},
        {"questions": [{"id": "valid\n", "kind": "confirm", "prompt": "Go?"}]},
        {"questions": [{"id": "x", "kind": "unknown", "prompt": "Go?"}]},
        {"questions": [{"id": "x", "kind": "text", "prompt": "123456789"}]},
        {"questions": [{"id": "x", "kind": "text", "prompt": "Go?", "extra": 1}]},
        {"questions": [{"id": "x", "kind": "single_choice", "prompt": "Go?"}]},
        {
            "questions": [
                {
                    "id": "x",
                    "kind": "single_choice",
                    "prompt": "Go?",
                    "options": [
                        {"value": "a", "label": "A"},
                        {"value": "b", "label": "B"},
                        {"value": "c", "label": "C"},
                    ],
                }
            ]
        },
        {"questions": [{"id": "x", "kind": "number", "prompt": "Go?", "step": 0}]},
        {"questions": [{"id": "x", "kind": "date", "prompt": "Go?", "default": 1}]},
        {"questions": [{"id": "x", "kind": "scale", "prompt": "Go?"}]},
        {"questions": [{"id": "x", "kind": "ranking", "prompt": "Go?"}]},
        {"questions": [{"id": "x", "kind": "confirm", "prompt": "Go?", "default": 1}]},
        {"questions": [{"id": "x", "kind": "text", "prompt": "Go?"}], "extra": 1},
    )

    async def validate() -> None:
        catalog = await ToolRegistry((tool,)).open_catalog()
        for index, arguments in enumerate(invalid):
            with pytest.raises(ToolError, match="do not match input_schema"):
                catalog.bind(ToolCall(f"invalid-{index}", "AskQuestion", arguments))

    asyncio.run(validate())


@pytest.mark.parametrize(
    "questions",
    [
        [
            {"id": "same", "kind": "confirm", "prompt": "One?"},
            {"id": "same", "kind": "confirm", "prompt": "Two?"},
        ],
        [
            {
                "id": "choice",
                "kind": "single_choice",
                "prompt": "Choose",
                "options": [
                    {"value": "same", "label": "A"},
                    {"value": "same", "label": "B"},
                ],
            }
        ],
        [
            {
                "id": "multi",
                "kind": "multiple_choice",
                "prompt": "Choose",
                "options": [
                    {"value": "a", "label": "A"},
                    {"value": "b", "label": "B"},
                ],
                "min_selections": 2,
                "max_selections": 1,
            }
        ],
        [
            {
                "id": "text",
                "kind": "text",
                "prompt": "Text",
                "min_length": 5,
                "max_length": 4,
            }
        ],
        [
            {
                "id": "number",
                "kind": "number",
                "prompt": "Number",
                "minimum": 2,
                "maximum": 1,
            }
        ],
        [
            {
                "id": "date",
                "kind": "date",
                "prompt": "Date",
                "minimum": "2026-02-30",
            }
        ],
        [
            {
                "id": "scale",
                "kind": "scale",
                "prompt": "Scale",
                "minimum": 5,
                "maximum": 1,
            }
        ],
        [
            {
                "id": "ranking",
                "kind": "ranking",
                "prompt": "Rank",
                "options": [
                    {"value": "a", "label": "A"},
                    {"value": "b", "label": "B"},
                ],
                "max_ranked": 3,
            }
        ],
    ],
)
def test_direct_invocation_reports_semantic_failures(
    questions: list[dict[str, Any]],
) -> None:
    _failure(_invoke(AskQuestionTool(), {"questions": questions}), "invalid_question")


def test_question_response_classmethods_are_immutable_and_validate_identifiers() -> None:
    answers = {"confirmed": True, "features": ["cache"]}
    response = QuestionResponse.answered("request-1", "response-1", answers)
    answers["confirmed"] = False

    assert response.request_id == "request-1"
    assert response.response_id == "response-1"
    assert response.status == "answered"
    assert thaw_json_value(response.answers) == {
        "confirmed": True,
        "features": ["cache"],
    }
    assert response.reason is None

    cancelled = QuestionResponse.cancelled("request-1", "response-2", "not now")
    assert cancelled.status == "cancelled"
    assert cancelled.answers == {}
    assert cancelled.reason == "not now"

    for request_id, response_id in (("", "response"), ("request", "")):
        with pytest.raises(ValueError):
            QuestionResponse.answered(request_id, response_id, {})

    with pytest.raises(ValueError, match="status"):
        QuestionResponse("request", "response", cast(Any, "deferred"))
    with pytest.raises(ValueError, match="reason"):
        QuestionResponse("request", "response", "answered", reason="not allowed")
    with pytest.raises(ValueError, match="answers"):
        QuestionResponse(
            "request",
            "response",
            "cancelled",
            answers={"question": True},
        )


@pytest.mark.parametrize(
    ("questions", "max_text_chars"),
    [
        ([], 10),
        ([1], 10),
        (
            [
                {"id": "same", "kind": "confirm", "prompt": "One", "required": True},
                {"id": "same", "kind": "confirm", "prompt": "Two", "required": True},
            ],
            10,
        ),
        ([{"kind": "confirm", "prompt": "Missing id", "required": True}], 10),
        ([{"id": "bad id", "kind": "confirm", "prompt": "Bad", "required": True}], 10),
        ([{"id": "q", "kind": "other", "prompt": "Bad", "required": True}], 10),
        (
            [
                {
                    "id": "q",
                    "kind": "confirm",
                    "prompt": "Extra",
                    "required": True,
                    "extra": True,
                }
            ],
            10,
        ),
        (
            [
                {
                    "id": "q",
                    "kind": "text",
                    "prompt": "Text",
                    "required": True,
                    "multiline": False,
                    "min_length": 2,
                    "max_length": 1,
                }
            ],
            10,
        ),
        (
            [
                {
                    "id": "q",
                    "kind": "text",
                    "prompt": "Text",
                    "required": False,
                    "multiline": False,
                    "min_length": 0,
                    "max_length": 0,
                }
            ],
            10,
        ),
        (
            [
                {
                    "id": "q",
                    "kind": "text",
                    "prompt": "Text",
                    "required": True,
                    "multiline": False,
                    "min_length": 1,
                    "max_length": 11,
                }
            ],
            10,
        ),
        (
            [
                {
                    "id": "q",
                    "kind": "date",
                    "prompt": "Date",
                    "required": True,
                    "minimum": "2026-02-02",
                    "maximum": "2026-02-01",
                }
            ],
            10,
        ),
        (
            [
                {
                    "id": "q",
                    "kind": "scale",
                    "prompt": "Scale",
                    "required": True,
                    "maximum": 5,
                    "step": 1,
                }
            ],
            10,
        ),
        (
            [
                {
                    "id": "q",
                    "kind": "scale",
                    "prompt": "Scale",
                    "required": True,
                    "minimum": 5,
                    "maximum": 5,
                    "step": 1,
                }
            ],
            10,
        ),
        (
            [
                {
                    "id": "q",
                    "kind": "number",
                    "prompt": "Number",
                    "required": True,
                    "integer_only": False,
                    "minimum": 2,
                    "maximum": 1,
                }
            ],
            10,
        ),
        (
            [
                {
                    "id": "q",
                    "kind": "number",
                    "prompt": "Number",
                    "required": True,
                    "integer_only": False,
                    "step": 0,
                }
            ],
            10,
        ),
        (
            [
                {
                    "id": "q",
                    "kind": "multiple_choice",
                    "prompt": "Multiple",
                    "required": True,
                    "options": [{"value": "a", "label": "A"}],
                    "allow_custom": False,
                    "min_selections": 1,
                    "max_selections": 1,
                }
            ],
            10,
        ),
        (
            [
                {
                    "id": "q",
                    "kind": "multiple_choice",
                    "prompt": "Multiple",
                    "required": True,
                    "options": [
                        {"value": "same", "label": "A"},
                        {"value": "same", "label": "B"},
                    ],
                    "allow_custom": False,
                    "min_selections": 1,
                    "max_selections": 2,
                }
            ],
            10,
        ),
        (
            [
                {
                    "id": "q",
                    "kind": "multiple_choice",
                    "prompt": "Multiple",
                    "required": True,
                    "options": [
                        {"value": "a", "label": "A"},
                        {"value": "b", "label": "B"},
                    ],
                    "allow_custom": False,
                    "min_selections": -1,
                    "max_selections": 2,
                }
            ],
            10,
        ),
        (
            [
                {
                    "id": "q",
                    "kind": "multiple_choice",
                    "prompt": "Multiple",
                    "required": True,
                    "options": [
                        {"value": "a", "label": "A"},
                        {"value": "b", "label": "B"},
                    ],
                    "allow_custom": False,
                    "min_selections": 2,
                    "max_selections": 1,
                }
            ],
            10,
        ),
        (
            [
                {
                    "id": "q",
                    "kind": "multiple_choice",
                    "prompt": "Multiple",
                    "required": False,
                    "options": [
                        {"value": "a", "label": "A"},
                        {"value": "b", "label": "B"},
                    ],
                    "allow_custom": False,
                    "min_selections": 0,
                    "max_selections": 0,
                }
            ],
            10,
        ),
        (
            [
                {
                    "id": "q",
                    "kind": "multiple_choice",
                    "prompt": "Multiple",
                    "required": True,
                    "options": [
                        {"value": "a", "label": "A"},
                        {"value": "b", "label": "B"},
                    ],
                    "allow_custom": False,
                    "min_selections": 1,
                    "max_selections": 3,
                }
            ],
            10,
        ),
    ],
)
def test_question_request_rejects_malformed_normalized_contract(
    questions: object,
    max_text_chars: int,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        QuestionRequest("request", max_text_chars, cast(Any, questions))


def test_question_request_validates_its_root_values_and_optional_constraints() -> None:
    for request_id, maximum, questions in (
        ("", 10, tuple(_all_questions())),
        ("request", 0, tuple(_all_questions())),
        ("request", True, tuple(_all_questions())),
        ("request", 10, "not-an-array"),
    ):
        with pytest.raises((TypeError, ValueError)):
            QuestionRequest(cast(Any, request_id), cast(Any, maximum), cast(Any, questions))

    minimal = QuestionRequest(
        "request",
        10,
        (
            {
                "id": "number",
                "kind": "number",
                "prompt": "Number",
                "required": True,
                "integer_only": False,
            },
            {
                "id": "date",
                "kind": "date",
                "prompt": "Date",
                "required": True,
            },
            {
                "id": "scale",
                "kind": "scale",
                "prompt": "Scale",
                "required": True,
                "minimum": 1,
                "maximum": 5,
                "step": 1,
            },
        ),
    )
    assert len(minimal.questions) == 3


def test_question_request_preserves_and_enforces_host_capability_limits() -> None:
    request = QuestionRequest(
        "request",
        20,
        (
            {
                "id": "confirm",
                "kind": "confirm",
                "prompt": "Go?",
                "required": True,
            },
        ),
        enabled_kinds=("confirm", "text"),
        max_questions=1,
        max_options=2,
        max_prompt_chars=4,
    )

    assert request.enabled_kinds == ("confirm", "text")
    assert request.max_questions == 1
    assert request.max_options == 2
    assert request.max_prompt_chars == 4

    invalid_kinds: tuple[object, ...] = (
        (),
        ("confirm", "confirm"),
        ("unsupported",),
        ("text", "confirm"),
        "confirm",
    )
    for enabled_kinds in invalid_kinds:
        with pytest.raises((TypeError, ValueError), match="enabled_kinds"):
            QuestionRequest(
                "request",
                20,
                request.questions,
                enabled_kinds=cast(Any, enabled_kinds),
            )

    for keyword, value in (
        ("max_questions", 0),
        ("max_options", 1),
        ("max_prompt_chars", 0),
    ):
        with pytest.raises(ValueError, match=keyword):
            QuestionRequest(
                "request",
                20,
                request.questions,
                **cast(Any, {keyword: value}),
            )


def test_question_request_rechecks_question_count_kind_and_display_limits() -> None:
    confirm = {
        "id": "confirm",
        "kind": "confirm",
        "prompt": "Go?",
        "required": True,
    }
    with pytest.raises(ValueError, match="max_questions"):
        QuestionRequest(
            "request",
            20,
            (confirm, {**confirm, "id": "again"}),
            max_questions=1,
        )
    with pytest.raises(ValueError, match="not enabled"):
        QuestionRequest(
            "request",
            20,
            (confirm,),
            enabled_kinds=("text",),
        )

    display_cases = (
        {**confirm, "prompt": "long"},
        {**confirm, "description": "long"},
        {
            "id": "text",
            "kind": "text",
            "prompt": "Text",
            "required": True,
            "multiline": False,
            "placeholder": "long",
            "min_length": 0,
            "max_length": 20,
        },
        {
            "id": "scale",
            "kind": "scale",
            "prompt": "R",
            "required": True,
            "minimum": 1,
            "maximum": 5,
            "step": 1,
            "minimum_label": "long",
        },
        {
            "id": "choice",
            "kind": "single_choice",
            "prompt": "P",
            "required": True,
            "options": (
                {"value": "a", "label": "long"},
                {"value": "b", "label": "B"},
            ),
            "allow_custom": False,
        },
        {
            "id": "choice",
            "kind": "single_choice",
            "prompt": "P",
            "required": True,
            "options": (
                {"value": "a", "label": "A", "description": "long"},
                {"value": "b", "label": "B"},
            ),
            "allow_custom": False,
        },
    )
    for question in display_cases:
        with pytest.raises(ValueError, match=r"length|max_prompt_chars"):
            QuestionRequest(
                "request",
                20,
                (question,),
                max_prompt_chars=3,
            )

    choice = {
        "id": "choice",
        "kind": "single_choice",
        "prompt": "Pick",
        "required": True,
        "options": (
            {"value": "a", "label": "A"},
            {"value": "b", "label": "B"},
            {"value": "c", "label": "C"},
        ),
        "allow_custom": False,
    }
    with pytest.raises(ValueError, match="max_options"):
        QuestionRequest("request", 20, (choice,), max_options=2)

    overlong_value = {
        **choice,
        "options": (
            {"value": "x" * 129, "label": "A"},
            {"value": "b", "label": "B"},
        ),
    }
    with pytest.raises(ValueError, match=r"options\[0\]\.value"):
        QuestionRequest("request", 20, (overlong_value,))


def test_validate_question_response_accepts_every_kind_and_custom_choice() -> None:
    request = _question_request()
    response = QuestionResponse.answered(
        request.request_id,
        "response-all",
        _valid_answers(),
    )
    assert validate_question_response(request, response) is None

    custom = _valid_answers()
    custom["database"] = "cockroachdb"
    assert (
        validate_question_response(
            request,
            QuestionResponse.answered(request.request_id, "response-custom", custom),
        )
        is None
    )


def _invalid_response_cases() -> list[tuple[str, Callable[[dict[str, Any]], None]]]:  # noqa: C901
    def remove_required(answers: dict[str, Any]) -> None:
        del answers["confirmed"]

    def add_unknown(answers: dict[str, Any]) -> None:
        answers["unknown"] = "value"

    def confirm_string(answers: dict[str, Any]) -> None:
        answers["confirmed"] = "yes"

    def bad_choice(answers: dict[str, Any]) -> None:
        answers["features"] = ["unknown"]

    def multi_scalar(answers: dict[str, Any]) -> None:
        answers["features"] = "cache"

    def multi_duplicate(answers: dict[str, Any]) -> None:
        answers["features"] = ["cache", "cache"]

    def multi_too_few(answers: dict[str, Any]) -> None:
        answers["features"] = []

    def multi_too_many(answers: dict[str, Any]) -> None:
        answers["features"] = ["cache", "audit", "search"]

    def text_wrong_type(answers: dict[str, Any]) -> None:
        answers["notes"] = 1

    def text_too_short(answers: dict[str, Any]) -> None:
        answers["notes"] = "x"

    def text_too_long(answers: dict[str, Any]) -> None:
        answers["notes"] = "x" * 81

    def number_bool(answers: dict[str, Any]) -> None:
        answers["retries"] = True

    def number_fraction(answers: dict[str, Any]) -> None:
        answers["retries"] = 2.5

    def number_below(answers: dict[str, Any]) -> None:
        answers["retries"] = -2

    def number_above(answers: dict[str, Any]) -> None:
        answers["retries"] = 12

    def number_off_step(answers: dict[str, Any]) -> None:
        answers["retries"] = 3

    def date_format(answers: dict[str, Any]) -> None:
        answers["deadline"] = "2026/08/15"

    def date_impossible(answers: dict[str, Any]) -> None:
        answers["deadline"] = "2026-02-30"

    def date_below(answers: dict[str, Any]) -> None:
        answers["deadline"] = "2026-06-30"

    def date_above(answers: dict[str, Any]) -> None:
        answers["deadline"] = "2027-01-01"

    def scale_bool(answers: dict[str, Any]) -> None:
        answers["confidence"] = False

    def scale_below(answers: dict[str, Any]) -> None:
        answers["confidence"] = 0.5

    def scale_above(answers: dict[str, Any]) -> None:
        answers["confidence"] = 5.5

    def scale_off_step(answers: dict[str, Any]) -> None:
        answers["confidence"] = 4.25

    def ranking_scalar(answers: dict[str, Any]) -> None:
        answers["priorities"] = "speed"

    def ranking_unknown(answers: dict[str, Any]) -> None:
        answers["priorities"] = ["correctness", "unknown"]

    def ranking_duplicate(answers: dict[str, Any]) -> None:
        answers["priorities"] = ["correctness", "correctness"]

    def ranking_too_few(answers: dict[str, Any]) -> None:
        answers["priorities"] = ["correctness"]

    return [
        ("required", remove_required),
        ("unknown", add_unknown),
        ("confirmed", confirm_string),
        ("features", bad_choice),
        ("features", multi_scalar),
        ("features", multi_duplicate),
        ("features", multi_too_few),
        ("features", multi_too_many),
        ("notes", text_wrong_type),
        ("notes", text_too_short),
        ("notes", text_too_long),
        ("retries", number_bool),
        ("retries", number_fraction),
        ("retries", number_below),
        ("retries", number_above),
        ("retries", number_off_step),
        ("deadline", date_format),
        ("deadline", date_impossible),
        ("deadline", date_below),
        ("deadline", date_above),
        ("confidence", scale_bool),
        ("confidence", scale_below),
        ("confidence", scale_above),
        ("confidence", scale_off_step),
        ("priorities", ranking_scalar),
        ("priorities", ranking_unknown),
        ("priorities", ranking_duplicate),
        ("priorities", ranking_too_few),
    ]


@pytest.mark.parametrize(("message", "mutate"), _invalid_response_cases())
def test_validate_question_response_rejects_every_answer_boundary(
    message: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    request = _question_request()
    answers = _valid_answers()
    mutate(answers)
    response = QuestionResponse.answered(request.request_id, "response-invalid", answers)
    with pytest.raises((TypeError, ValueError), match=message):
        validate_question_response(request, response)


def test_required_default_does_not_supply_an_answer_and_optional_can_be_omitted() -> None:
    request = _question_request(
        [
            {
                "id": "required",
                "kind": "text",
                "prompt": "Required",
                "default": "hint only",
            },
            {
                "id": "optional",
                "kind": "confirm",
                "prompt": "Optional",
                "required": False,
            },
        ]
    )
    response = QuestionResponse.answered(request.request_id, "response-empty", {})
    with pytest.raises(ValueError, match="required"):
        validate_question_response(request, response)

    accepted = QuestionResponse.answered(
        request.request_id,
        "response-required",
        {"required": "provided"},
    )
    assert validate_question_response(request, accepted) is None


def test_global_answer_character_limit_applies_to_text_and_custom_values() -> None:
    request = _question_request(
        [
            {
                "id": "text",
                "kind": "text",
                "prompt": "Text",
            },
            {
                "id": "choice",
                "kind": "single_choice",
                "prompt": "Choice",
                "options": [
                    {"value": "a", "label": "A"},
                    {"value": "b", "label": "B"},
                ],
                "allow_custom": True,
            },
        ],
        max_answer_chars=4,
    )
    for answer in (
        {"text": "12345", "choice": "a"},
        {"text": "ok", "choice": "12345"},
    ):
        with pytest.raises(ValueError, match=r"text|choice"):
            validate_question_response(
                request,
                QuestionResponse.answered(request.request_id, "response-long", answer),
            )


def test_multiple_choice_allows_at_most_one_custom_value() -> None:
    request = _question_request(
        [
            {
                "id": "choice",
                "kind": "multiple_choice",
                "prompt": "Choose",
                "options": [
                    {"value": "known-a", "label": "A"},
                    {"value": "known-b", "label": "B"},
                ],
                "allow_custom": True,
                "min_selections": 1,
                "max_selections": 3,
            }
        ]
    )
    accepted = QuestionResponse.answered(
        request.request_id,
        "one-custom",
        {"choice": ["known-a", "custom-one"]},
    )
    assert validate_question_response(request, accepted) is None

    rejected = QuestionResponse.answered(
        request.request_id,
        "two-custom",
        {"choice": ["custom-one", "custom-two"]},
    )
    with pytest.raises(ValueError, match=r"choice|custom"):
        validate_question_response(request, rejected)


def test_known_option_values_are_not_limited_by_custom_text_limit() -> None:
    questions = [
        {
            "id": "single",
            "kind": "single_choice",
            "prompt": "Single",
            "options": [
                {"value": "known-long-a", "label": "A"},
                {"value": "known-long-b", "label": "B"},
            ],
            "default": "known-long-a",
        },
        {
            "id": "multiple",
            "kind": "multiple_choice",
            "prompt": "Multiple",
            "options": [
                {"value": "known-long-a", "label": "A"},
                {"value": "known-long-b", "label": "B"},
            ],
            "default": ["known-long-a"],
        },
    ]
    request = _question_request(questions, max_answer_chars=2)
    response = QuestionResponse.answered(
        request.request_id,
        "known-values",
        {"single": "known-long-b", "multiple": ["known-long-a"]},
    )
    assert validate_question_response(request, response) is None


def test_cancelled_response_needs_no_answers_and_message_is_model_visible_json() -> None:
    request = _question_request()
    response = QuestionResponse.cancelled(
        request.request_id,
        "response-cancel",
        "user declined",
    )
    assert validate_question_response(request, response) is None

    message = question_response_message(request, response)
    assert message.role == "external"
    assert message.metadata == {
        "kind": "question_response",
        "request_id": request.request_id,
        "response_id": "response-cancel",
        "status": "cancelled",
    }
    assert len(message.parts) == 1
    assert message.parts[0].text is not None
    prefix = "AskQuestion response:\n"
    assert message.parts[0].text.startswith(prefix)
    assert json.loads(message.parts[0].text.removeprefix(prefix)) == {
        "answers": {},
        "request_id": request.request_id,
        "response_id": "response-cancel",
        "status": "cancelled",
        "reason": "user declined",
    }


def test_answered_message_contains_validated_answers_in_text_and_metadata() -> None:
    request = _question_request()
    response = QuestionResponse.answered(
        request.request_id,
        "response-answer",
        _valid_answers(),
    )
    message = question_response_message(request, response)

    assert message.role == "external"
    assert message.metadata["kind"] == "question_response"
    assert message.metadata["request_id"] == request.request_id
    assert message.metadata["response_id"] == "response-answer"
    assert message.metadata["status"] == "answered"
    assert len(message.parts) == 1
    assert message.parts[0].text is not None
    prefix = "AskQuestion response:\n"
    assert message.parts[0].text.startswith(prefix)
    assert json.loads(message.parts[0].text.removeprefix(prefix)) == {
        "request_id": request.request_id,
        "response_id": "response-answer",
        "status": "answered",
        "answers": _valid_answers(),
    }


def test_response_message_canonical_json_escapes_lone_surrogates_for_utf8() -> None:
    request = QuestionRequest(
        "request",
        4,
        (
            {
                "id": "text",
                "kind": "text",
                "prompt": "Text",
                "required": True,
                "multiline": False,
                "min_length": 0,
                "max_length": 4,
            },
        ),
    )
    response = QuestionResponse.answered(
        "request",
        "response",
        {"text": "\ud800"},
    )

    message = question_response_message(request, response)
    text = message.parts[0].text
    assert text is not None
    assert text.encode("utf-8")
    assert "\\ud800" in text
    assert json.loads(text.removeprefix("AskQuestion response:\n"))["answers"] == {"text": "\ud800"}


def _forged_checkpoint(
    state: object,
    history: Sequence[object] = (),
    *,
    context: RunContext | None = None,
    fact: object | None = None,
) -> Checkpoint:
    checkpoint = object.__new__(Checkpoint)
    snapshot = SimpleNamespace(
        state=state,
        history=history,
        context=RunContext("run", time.time()) if context is None else context,
    )
    object.__setattr__(checkpoint, "id", "forged")
    object.__setattr__(checkpoint, "snapshot", snapshot)
    object.__setattr__(checkpoint, "fact", SimpleNamespace() if fact is None else fact)
    return checkpoint


def _question_checkpoint(
    *,
    arguments: Mapping[str, Any] | None = None,
    tool: AskQuestionTool | None = None,
    run_id: str = "run",
    tool_call_id: str = "call",
) -> Checkpoint:
    selected_arguments: Mapping[str, Any] = (
        {"questions": [{"id": "confirm", "kind": "confirm", "prompt": "Go?"}]}
        if arguments is None
        else arguments
    )
    selected_tool = AskQuestionTool() if tool is None else tool
    context = RunContext(run_id, time.time())
    call = ToolCall(tool_call_id, "AskQuestion", selected_arguments)
    result = asyncio.run(
        selected_tool.invoke(call, ToolContext(context, _emit_progress, lambda: False))
    )
    assert isinstance(result, WaitingResult)
    suspension = result.suspension
    fact = ToolBatchFact(
        time.time(),
        "question-batch",
        (tool_call_id,),
        False,
        (ToolOutcomeKind.WAITING,),
        SuspensionView(
            suspension.reason,
            suspension.source,
            suspension.wait_id,
            tuple(suspension.metadata),
        ),
    )
    history = (
        Message.user("Ask"),
        Message.assistant(tool_calls=(call,)),
        Message.tool(tool_call_id, result.outcome),
    )
    return _forged_checkpoint(
        Suspended(Planning(), suspension),
        history,
        context=context,
        fact=fact,
    )


def _ask_suspension(*, schema_version: int = 1) -> Suspension:
    return Suspension(
        "human_input",
        "AskQuestion",
        "request",
        {
            "contract_id": question_schema.build_contract_id(
                question_schema.SUPPORTED_QUESTION_KINDS,
                max_questions=8,
                max_options=12,
                max_prompt_chars=2_000,
                max_answer_chars=16_384,
            ),
            "tool_call_id": "call",
            "schema_version": schema_version,
        },
    )


def test_extract_question_request_rejects_wrong_type_suspension_and_resume_state() -> None:
    with pytest.raises(TypeError, match="Checkpoint"):
        extract_question_request(cast(Any, None))

    wrong_source = _forged_checkpoint(
        Suspended(Planning(), Suspension("human_input", "Other", "request"))
    )
    with pytest.raises(ValueError, match="human_input"):
        extract_question_request(wrong_source)

    wrong_resume = _forged_checkpoint(
        Suspended(
            ToolsPending(PendingToolCalls((ToolCall("pending", "AskQuestion"),))),
            _ask_suspension(),
        )
    )
    with pytest.raises(ValueError, match="Planning"):
        extract_question_request(wrong_resume)

    wrong_version = _forged_checkpoint(Suspended(Planning(), _ask_suspension(schema_version=2)))
    with pytest.raises(ValueError, match="schema_version"):
        extract_question_request(wrong_version)


def test_extract_question_request_rejects_missing_result_and_corrupt_payload() -> None:
    empty = _forged_checkpoint(Planning())
    with pytest.raises(ValueError, match="waiting tool batch"):
        question_response_module._current_waiting_outcome(empty, "call")

    settled_message = SimpleNamespace(
        role="tool",
        tool_call_id="call",
        outcome=ToolSuccess((ContentPart.text_part("done"),)),
    )
    settled = _forged_checkpoint(Planning(), (settled_message,))
    with pytest.raises(ValueError, match="waiting tool batch"):
        question_response_module._current_waiting_outcome(settled, "call")

    base: dict[str, Any] = {
        "status": "waiting",
        "schema_version": 1,
        "request_id": "request",
        "enabled_kinds": list(question_schema.SUPPORTED_QUESTION_KINDS),
        "max_questions": 8,
        "max_options": 12,
        "max_prompt_chars": 2_000,
        "max_text_chars": 10,
        "questions": [],
    }
    for field, value, pattern in (
        ("status", "done", "status"),
        ("schema_version", 2, "schema_version"),
        ("request_id", "other", "request_id"),
    ):
        payload = dict(base)
        payload[field] = value
        with pytest.raises(ValueError, match=pattern):
            question_response_module._validate_waiting_payload(payload, "request")


def test_extract_question_request_binds_snapshot_context_contract_and_assistant_arguments() -> None:
    checkpoint = _question_checkpoint(
        tool=AskQuestionTool(
            enabled_kinds={"confirm", "text"},
            max_questions=2,
            max_options=3,
            max_prompt_chars=20,
            max_answer_chars=40,
        ),
        run_id="run:one",
        tool_call_id="call:one",
    )
    request = extract_question_request(checkpoint)
    assert request.request_id == "ask-question:7:run:one:8:call:one"
    assert request.enabled_kinds == ("confirm", "text")
    assert request.max_questions == 2
    assert request.max_options == 3
    assert request.max_prompt_chars == 20
    assert request.max_text_chars == 40

    with pytest.raises(ValueError, match="run and tool call"):
        extract_question_request(
            _forged_checkpoint(
                checkpoint.snapshot.state,
                checkpoint.snapshot.history,
                context=RunContext("different", time.time()),
                fact=checkpoint.fact,
            )
        )

    state = cast(Suspended, checkpoint.snapshot.state)
    wrong_metadata = dict(state.suspension.metadata)
    wrong_metadata["contract_id"] = "sha256:wrong"
    wrong_contract = Suspension(
        state.suspension.reason,
        state.suspension.source,
        state.suspension.wait_id,
        wrong_metadata,
    )
    with pytest.raises(ValueError, match=r"contract.*suspension"):
        extract_question_request(
            _forged_checkpoint(
                Suspended(Planning(), wrong_contract),
                checkpoint.snapshot.history,
                context=checkpoint.snapshot.context,
                fact=checkpoint.fact,
            )
        )

    user_message, _, tool_message = checkpoint.snapshot.history
    malformed_call = ToolCall("call:one", "AskQuestion", {"questions": []})
    malformed_history = (
        user_message,
        Message.assistant(tool_calls=(malformed_call,)),
        tool_message,
    )
    with pytest.raises(ValueError, match="assistant call"):
        extract_question_request(
            _forged_checkpoint(
                checkpoint.snapshot.state,
                malformed_history,
                context=checkpoint.snapshot.context,
                fact=checkpoint.fact,
            )
        )

    changed_call = ToolCall(
        "call:one",
        "AskQuestion",
        {"questions": [{"id": "confirm", "kind": "confirm", "prompt": "Different?"}]},
    )
    changed_history = (
        user_message,
        Message.assistant(tool_calls=(changed_call,)),
        tool_message,
    )
    with pytest.raises(ValueError, match=r"questions.*assistant call"):
        extract_question_request(
            _forged_checkpoint(
                checkpoint.snapshot.state,
                changed_history,
                context=checkpoint.snapshot.context,
                fact=checkpoint.fact,
            )
        )


def test_current_waiting_outcome_rejects_stale_or_unproven_tool_messages() -> None:
    checkpoint = _question_checkpoint()
    fact = checkpoint.fact
    assert isinstance(fact, ToolBatchFact)
    user_message, assistant_message, tool_message = checkpoint.snapshot.history
    waiting, assistant_call = question_response_module._current_waiting_outcome(checkpoint, "call")
    assert waiting is tool_message.outcome
    assert assistant_call is assistant_message.tool_calls[0]

    bad_fact = replace(fact, call_ids=("other",))
    with pytest.raises(ValueError, match="current waiting tool call"):
        question_response_module._current_waiting_outcome(
            _forged_checkpoint(
                checkpoint.snapshot.state,
                checkpoint.snapshot.history,
                fact=bad_fact,
            ),
            "call",
        )

    bad_outcome_fact = replace(fact, outcome_kinds=(ToolOutcomeKind.SUCCESS,))
    with pytest.raises(ValueError, match="current waiting tool call"):
        question_response_module._current_waiting_outcome(
            _forged_checkpoint(
                checkpoint.snapshot.state,
                checkpoint.snapshot.history,
                fact=bad_outcome_fact,
            ),
            "call",
        )

    with pytest.raises(ValueError, match="missing its assistant and tool"):
        question_response_module._current_waiting_outcome(
            _forged_checkpoint(checkpoint.snapshot.state, (tool_message,), fact=fact),
            "call",
        )

    with pytest.raises(ValueError, match="trailing tool"):
        question_response_module._current_waiting_outcome(
            _forged_checkpoint(
                checkpoint.snapshot.state,
                (*checkpoint.snapshot.history, Message.external("newer history")),
                fact=fact,
            ),
            "call",
        )

    wrong_tool_id = Message.tool("other", cast(ToolWaiting, tool_message.outcome))
    with pytest.raises(ValueError, match="trailing tool"):
        question_response_module._current_waiting_outcome(
            _forged_checkpoint(
                checkpoint.snapshot.state,
                (user_message, assistant_message, wrong_tool_id),
                fact=fact,
            ),
            "call",
        )

    with pytest.raises(ValueError, match="assistant tool call"):
        question_response_module._current_waiting_outcome(
            _forged_checkpoint(
                checkpoint.snapshot.state,
                (user_message, Message.external("not an assistant"), tool_message),
                fact=fact,
            ),
            "call",
        )

    empty_assistant = Message.assistant((ContentPart.text_part("no call"),))
    with pytest.raises(ValueError, match="assistant tool call"):
        question_response_module._current_waiting_outcome(
            _forged_checkpoint(
                checkpoint.snapshot.state,
                (user_message, empty_assistant, tool_message),
                fact=fact,
            ),
            "call",
        )

    wrong_identity = Message.assistant(
        tool_calls=(ToolCall("call", "Other", assistant_call.arguments),)
    )
    with pytest.raises(ValueError, match="wrong identity"):
        question_response_module._current_waiting_outcome(
            _forged_checkpoint(
                checkpoint.snapshot.state,
                (user_message, wrong_identity, tool_message),
                fact=fact,
            ),
            "call",
        )

    wrong_call_id = Message.assistant(
        tool_calls=(ToolCall("other", "AskQuestion", assistant_call.arguments),)
    )
    with pytest.raises(ValueError, match="wrong identity"):
        question_response_module._current_waiting_outcome(
            _forged_checkpoint(
                checkpoint.snapshot.state,
                (user_message, wrong_call_id, tool_message),
                fact=fact,
            ),
            "call",
        )

    settled_message = Message.tool(
        "call",
        ToolSuccess((ContentPart.text_part("done"),)),
    )
    with pytest.raises(ValueError, match="ToolWaiting"):
        question_response_module._current_waiting_outcome(
            _forged_checkpoint(
                checkpoint.snapshot.state,
                (user_message, assistant_message, settled_message),
                fact=fact,
            ),
            "call",
        )


def test_response_helpers_reject_wrong_types_and_overlong_cancellation() -> None:
    request = QuestionRequest(
        "request",
        4,
        (
            {
                "id": "confirm",
                "kind": "confirm",
                "prompt": "Confirm",
                "required": True,
            },
        ),
    )
    response = QuestionResponse.cancelled("request", "response")
    with pytest.raises(TypeError, match="QuestionRequest"):
        validate_question_response(cast(Any, None), response)
    with pytest.raises(TypeError, match="QuestionResponse"):
        validate_question_response(request, cast(Any, None))
    with pytest.raises(ValueError, match="request_id"):
        validate_question_response(
            request,
            QuestionResponse.cancelled("other", "response"),
        )
    with pytest.raises(ValueError, match="max_text_chars"):
        validate_question_response(
            request,
            QuestionResponse.cancelled("request", "response", "12345"),
        )

    with pytest.raises(TypeError, match="Runtime"):
        question_response_module.resume_question(
            cast(Any, None),
            cast(Any, None),
            response,
        )
    runtime = Runtime(model=_FinalModel())
    with pytest.raises(TypeError, match="stream"):
        question_response_module.resume_question(
            runtime,
            cast(Any, None),
            response,
            stream=cast(Any, 1),
        )


def test_numeric_answer_without_step_and_defensive_private_value_checks() -> None:
    request = QuestionRequest(
        "request",
        10,
        (
            {
                "id": "number",
                "kind": "number",
                "prompt": "Number",
                "required": True,
                "integer_only": False,
            },
        ),
    )
    assert (
        validate_question_response(
            request,
            QuestionResponse.answered("request", "response", {"number": 2}),
        )
        is None
    )

    with pytest.raises(ValueError, match="minimum and maximum"):
        question_response_module._validate_numeric_constraints(
            {},
            "scale",
            require_bounds=True,
        )
    with pytest.raises(TypeError, match="keys"):
        question_response_module._mapping({1: "value"}, "mapping")
    with pytest.raises(ValueError, match="finite"):
        question_response_module._number(float("inf"), "number")


def test_integer_only_questions_require_a_satisfiable_integer_domain() -> None:
    impossible = {
        "id": "integer",
        "kind": "number",
        "prompt": "Integer",
        "minimum": 0.1,
        "step": 1,
        "integer_only": True,
    }
    failure = _failure(
        _invoke(AskQuestionTool(), {"questions": [impossible]}),
        "invalid_question",
    )
    assert "integer" in failure.error.message

    normalized = {
        **impossible,
        "required": True,
    }
    with pytest.raises(ValueError, match="integer"):
        QuestionRequest("request", 20, (normalized,))

    bounded = {
        "id": "integer",
        "kind": "number",
        "prompt": "Integer",
        "minimum": 0.1,
        "maximum": 0.9,
        "integer_only": True,
        "required": True,
    }
    with pytest.raises(ValueError, match="integer"):
        QuestionRequest("request", 20, (bounded,))


def test_integral_float_integer_default_is_normalized_to_an_integer() -> None:
    result = _invoke(
        AskQuestionTool(),
        {
            "questions": [
                {
                    "id": "integer",
                    "kind": "number",
                    "prompt": "Integer",
                    "integer_only": True,
                    "default": 1.0,
                }
            ]
        },
        through_registry=True,
    )
    assert isinstance(result, WaitingResult)
    payload = thaw_json_value(result.outcome.structured_content)
    assert isinstance(payload, dict)
    questions = payload["questions"]
    assert isinstance(questions, list)
    default = cast(dict[str, Any], questions[0])["default"]
    assert default == 1
    assert type(default) is int


def test_integer_domain_math_and_integral_schema_counts_cover_fractional_edges() -> None:
    assert question_schema.integer_answer_exists(0.5, None, 0.5) is True
    assert question_schema.integer_answer_exists(2, 1, 1) is False
    assert question_schema.integer_answer_exists(0.5, 0.75, 0.5) is False
    assert question_schema._optional_int({"count": 1.0}, "count", 0, "value") == 1
    with pytest.raises(question_schema.QuestionValidationError, match="integer"):
        question_schema._optional_int({"count": 1.5}, "count", 0, "value")
    with pytest.raises(question_schema.QuestionValidationError, match="integer"):
        question_schema._optional_int({"count": float("inf")}, "count", 0, "value")


def test_unbounded_huge_integer_default_and_answer_do_not_overflow() -> None:
    huge = 10**1000
    question = {
        "id": "huge",
        "kind": "number",
        "prompt": "Huge integer",
        "integer_only": True,
        "default": huge,
    }

    core_result = _invoke(
        AskQuestionTool(),
        {"questions": [question]},
        through_registry=True,
    )
    assert isinstance(core_result, WaitingResult)

    request = _question_request([question])
    response = QuestionResponse.answered(
        request.request_id,
        "huge-response",
        {"huge": huge},
    )
    assert validate_question_response(request, response) is None


def test_unserializable_huge_numeric_values_fail_without_leaking_json_errors() -> None:
    huge = 10**5000
    raw_question = {
        "id": "huge",
        "kind": "number",
        "prompt": "Huge integer",
        "integer_only": True,
        "default": huge,
    }
    failure = _failure(
        _invoke(AskQuestionTool(), {"questions": [raw_question]}),
        "invalid_question",
    )
    assert failure.error.message == "questions[0].default must be JSON-serializable"

    normalized_question = {
        **raw_question,
        "required": True,
    }
    with pytest.raises(ValueError, match="JSON-serializable"):
        QuestionRequest("request", 20, (normalized_question,))

    request = QuestionRequest(
        "request",
        20,
        (
            {
                "id": "huge",
                "kind": "number",
                "prompt": "Huge integer",
                "required": True,
                "integer_only": True,
            },
        ),
    )
    response = QuestionResponse.answered("request", "response", {"huge": huge})
    with pytest.raises(ValueError, match="JSON-serializable"):
        validate_question_response(request, response)


def test_huge_integer_with_float_step_uses_exact_overflow_fallback() -> None:
    huge = 10**1000
    question = {
        "id": "huge",
        "kind": "number",
        "prompt": "Huge stepped integer",
        "step": 1.0,
        "default": huge,
    }

    core_result = _invoke(
        AskQuestionTool(),
        {"questions": [question]},
        through_registry=True,
    )
    assert isinstance(core_result, WaitingResult)

    request = _question_request([question])
    response = QuestionResponse.answered(
        request.request_id,
        "huge-stepped-response",
        {"huge": huge},
    )
    assert validate_question_response(request, response) is None


def test_extreme_finite_step_alignment_does_not_overflow() -> None:
    question = {
        "id": "extreme",
        "kind": "number",
        "prompt": "Extreme finite number",
        "minimum": -1e308,
        "maximum": 1e308,
        "step": 1,
        "default": 1e308,
    }

    core_result = _invoke(
        AskQuestionTool(),
        {"questions": [question]},
        through_registry=True,
    )
    assert isinstance(core_result, WaitingResult)

    request = _question_request([question])
    response = QuestionResponse.answered(
        request.request_id,
        "extreme-response",
        {"extreme": 1e308},
    )
    assert validate_question_response(request, response) is None


def test_numeric_fallback_failures_are_controlled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenFraction:
        def __init__(self, _value: object) -> None:
            raise ValueError("forced fallback failure")

    monkeypatch.setattr(question_schema, "Fraction", _BrokenFraction)
    monkeypatch.setattr(question_response_module, "Fraction", _BrokenFraction)

    assert question_schema._is_step_aligned(1e308, 1.0, -1e308) is False
    assert question_response_module._is_step_aligned(1e308, 1.0, -1e308) is False
    with pytest.raises(question_schema.QuestionValidationError, match="finite"):
        question_schema._number(float("inf"), "number")


def test_step_alignment_uses_exact_decimal_semantics() -> None:
    base = {
        "id": "number",
        "kind": "number",
        "prompt": "Precisely stepped number",
        "minimum": 0,
        "step": 1,
    }
    invalid_default = {**base, "default": 1_000_000_000.5}
    _failure(
        _invoke(AskQuestionTool(), {"questions": [invalid_default]}),
        "invalid_question",
    )

    request = _question_request([base])
    with pytest.raises(ValueError, match="step"):
        validate_question_response(
            request,
            QuestionResponse.answered(
                request.request_id,
                "off-step-response",
                {"number": 1_000_000_000.5},
            ),
        )

    decimal_step = {**base, "step": 0.1, "default": 0.3}
    accepted = _invoke(
        AskQuestionTool(),
        {"questions": [decimal_step]},
        through_registry=True,
    )
    assert isinstance(accepted, WaitingResult)
    decimal_request = _question_request(
        [{key: value for key, value in decimal_step.items() if key != "default"}]
    )
    assert (
        validate_question_response(
            decimal_request,
            QuestionResponse.answered(
                decimal_request.request_id,
                "decimal-step-response",
                {"number": 0.3},
            ),
        )
        is None
    )


def test_extract_question_request_rejects_non_suspended_checkpoint() -> None:
    checkpoint = asyncio.run(
        Runtime(
            model=_FinalModel(),
            tools=ToolRegistry((AskQuestionTool(),)),
        )
        .start((Message.user("Do not ask"),))
        .result()
    )
    with pytest.raises(ValueError, match=r"suspended|question"):
        extract_question_request(checkpoint)


class _FinalModel(Model):
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
        del request, context, stream, emit_delta
        return ModelResponse((ContentPart.text_part("done"),), finish_reason="stop")
