"""Host-side validation and resume helpers for ``AskQuestion``."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from fractions import Fraction
from math import isfinite
from typing import Any, Literal, cast

from jharness.kernel import (
    Checkpoint,
    Invocation,
    Message,
    Planning,
    Runtime,
    Suspended,
    SuspensionSelector,
    ToolBatchFact,
    ToolCall,
    ToolOutcomeKind,
    ToolWaiting,
    freeze_json_value,
    thaw_json_value,
)
from jharness.tools.interaction._schema import (
    DEFAULT_MAX_OPTIONS,
    DEFAULT_MAX_PROMPT_CHARS,
    DEFAULT_MAX_QUESTIONS,
    OPTION_VALUE_CHARS,
    SCHEMA_VERSION,
    SUPPORTED_QUESTION_KINDS,
    QuestionValidationError,
    build_contract_id,
    build_request_id,
    integer_answer_exists,
    normalize_questions,
    validate_enabled_kinds,
)

QuestionStatus = Literal["answered", "cancelled"]

_QUESTION_KINDS = frozenset(
    {
        "confirm",
        "single_choice",
        "multiple_choice",
        "text",
        "number",
        "date",
        "scale",
        "ranking",
    }
)
_COMMON_FIELDS = frozenset({"id", "kind", "prompt", "description", "required"})
_KIND_FIELDS: Mapping[str, frozenset[str]] = {
    "confirm": frozenset({"default"}),
    "single_choice": frozenset({"options", "allow_custom", "default"}),
    "multiple_choice": frozenset(
        {"options", "allow_custom", "min_selections", "max_selections", "default"}
    ),
    "text": frozenset({"multiline", "placeholder", "min_length", "max_length", "default"}),
    "number": frozenset({"minimum", "maximum", "step", "integer_only", "default"}),
    "date": frozenset({"minimum", "maximum", "default"}),
    "scale": frozenset(
        {
            "minimum",
            "maximum",
            "step",
            "minimum_label",
            "maximum_label",
            "default",
        }
    ),
    "ranking": frozenset({"options", "min_ranked", "max_ranked", "default"}),
}
_KIND_REQUIRED_FIELDS: Mapping[str, frozenset[str]] = {
    "confirm": frozenset(),
    "single_choice": frozenset({"options", "allow_custom"}),
    "multiple_choice": frozenset({"options", "allow_custom", "min_selections", "max_selections"}),
    "text": frozenset({"multiline", "min_length", "max_length"}),
    "number": frozenset({"integer_only"}),
    "date": frozenset(),
    "scale": frozenset({"minimum", "maximum", "step"}),
    "ranking": frozenset({"options", "min_ranked", "max_ranked"}),
}
_DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_QUESTION_ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")


@dataclass(frozen=True, slots=True)
class QuestionRequest:
    """One immutable, validated question request extracted from a checkpoint."""

    request_id: str
    max_text_chars: int
    questions: tuple[Mapping[str, Any], ...]
    enabled_kinds: tuple[str, ...] = SUPPORTED_QUESTION_KINDS
    max_questions: int = DEFAULT_MAX_QUESTIONS
    max_options: int = DEFAULT_MAX_OPTIONS
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS

    def __post_init__(self) -> None:
        request_id = _non_empty_string(self.request_id, "request_id")
        max_text_chars = _positive_int(self.max_text_chars, "max_text_chars")
        enabled_kinds = _enabled_kinds(self.enabled_kinds)
        max_questions = _positive_int(self.max_questions, "max_questions")
        max_options = _positive_int(self.max_options, "max_options")
        if max_options < 2:
            raise ValueError("max_options must be >= 2")
        max_prompt_chars = _positive_int(self.max_prompt_chars, "max_prompt_chars")
        questions = _freeze_questions(
            self.questions,
            max_text_chars,
            enabled_kinds=enabled_kinds,
            max_questions=max_questions,
            max_options=max_options,
            max_prompt_chars=max_prompt_chars,
        )
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "max_text_chars", max_text_chars)
        object.__setattr__(self, "questions", questions)
        object.__setattr__(self, "enabled_kinds", enabled_kinds)
        object.__setattr__(self, "max_questions", max_questions)
        object.__setattr__(self, "max_options", max_options)
        object.__setattr__(self, "max_prompt_chars", max_prompt_chars)


@dataclass(frozen=True, slots=True)
class QuestionResponse:
    """One immutable Host response to an ``AskQuestion`` request."""

    request_id: str
    response_id: str
    status: QuestionStatus
    answers: Mapping[str, Any] = field(default_factory=dict[str, Any])
    reason: str | None = None

    def __post_init__(self) -> None:
        request_id = _non_empty_string(self.request_id, "request_id")
        response_id = _non_empty_string(self.response_id, "response_id")
        if self.status not in {"answered", "cancelled"}:
            raise ValueError("status must be 'answered' or 'cancelled'")
        answers = _freeze_mapping(self.answers, "answers")
        reason = self.reason
        if reason is not None:
            reason = _non_empty_string(reason, "reason")
        if self.status == "answered" and reason is not None:
            raise ValueError("an answered response cannot include a cancellation reason")
        if self.status == "cancelled" and answers:
            raise ValueError("a cancelled response cannot include answers")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "response_id", response_id)
        object.__setattr__(self, "answers", answers)
        object.__setattr__(self, "reason", reason)

    @classmethod
    def answered(
        cls,
        request_id: str,
        response_id: str,
        answers: Mapping[str, Any],
    ) -> QuestionResponse:
        """Construct an answered response."""

        return cls(request_id, response_id, "answered", answers)

    @classmethod
    def cancelled(
        cls,
        request_id: str,
        response_id: str,
        reason: str | None = None,
    ) -> QuestionResponse:
        """Construct a response that explicitly cancels the interaction."""

        return cls(request_id, response_id, "cancelled", reason=reason)


def extract_question_request(checkpoint: Checkpoint) -> QuestionRequest:
    """Extract and validate the ``AskQuestion`` request in a suspended checkpoint."""

    if not isinstance(cast(object, checkpoint), Checkpoint):
        raise TypeError("checkpoint must be a Checkpoint")
    state = checkpoint.snapshot.state
    if not isinstance(state, Suspended):
        raise ValueError("checkpoint must be suspended for AskQuestion")
    suspension = state.suspension
    if suspension.reason != "human_input" or suspension.source != "AskQuestion":
        raise ValueError("checkpoint is not an AskQuestion human_input suspension")
    if not isinstance(state.resume_to, Planning):
        raise ValueError("AskQuestion must resume to Planning")
    request_id = _non_empty_string(suspension.wait_id, "AskQuestion suspension wait_id")
    _require_exact_keys(
        suspension.metadata,
        "AskQuestion suspension metadata",
        required=frozenset({"contract_id", "tool_call_id", "schema_version"}),
        allowed=frozenset({"contract_id", "tool_call_id", "schema_version"}),
    )
    contract_id = _non_empty_string(
        suspension.metadata["contract_id"], "AskQuestion suspension contract_id"
    )
    tool_call_id = _non_empty_string(
        suspension.metadata["tool_call_id"], "AskQuestion suspension tool_call_id"
    )
    schema_version = _exact_int(
        suspension.metadata["schema_version"], "AskQuestion suspension schema_version"
    )
    if schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported AskQuestion suspension schema_version")
    expected_request_id = build_request_id(checkpoint.snapshot.context.run_id, tool_call_id)
    if request_id != expected_request_id:
        raise ValueError("AskQuestion suspension wait_id does not match its run and tool call")
    waiting, assistant_call = _current_waiting_outcome(checkpoint, tool_call_id)
    payload = _waiting_payload(waiting)
    _validate_waiting_payload(payload, request_id)
    enabled_kinds = _string_sequence(payload["enabled_kinds"], "AskQuestion enabled_kinds")
    max_questions = _positive_int(payload["max_questions"], "AskQuestion max_questions")
    max_options = _positive_int(payload["max_options"], "AskQuestion max_options")
    max_prompt_chars = _positive_int(payload["max_prompt_chars"], "AskQuestion max_prompt_chars")
    max_text_chars = _positive_int(payload["max_text_chars"], "AskQuestion max_text_chars")
    expected_contract_id = build_contract_id(
        enabled_kinds,
        max_questions=max_questions,
        max_options=max_options,
        max_prompt_chars=max_prompt_chars,
        max_answer_chars=max_text_chars,
    )
    if contract_id != expected_contract_id:
        raise ValueError("AskQuestion payload contract does not match its suspension")
    try:
        expected_questions = normalize_questions(
            thaw_json_value(assistant_call.arguments, label="AskQuestion arguments"),
            enabled_kinds=validate_enabled_kinds(enabled_kinds),
            max_questions=max_questions,
            max_options=max_options,
            max_prompt_chars=max_prompt_chars,
            max_answer_chars=max_text_chars,
        )
    except QuestionValidationError as exc:
        raise ValueError("AskQuestion payload contract does not match its assistant call") from exc
    if expected_questions != payload["questions"]:
        raise ValueError("AskQuestion payload questions do not match its assistant call")
    questions = _mapping_sequence(payload["questions"], "AskQuestion questions")
    return QuestionRequest(
        request_id=request_id,
        max_text_chars=max_text_chars,
        questions=tuple(questions),
        enabled_kinds=enabled_kinds,
        max_questions=max_questions,
        max_options=max_options,
        max_prompt_chars=max_prompt_chars,
    )


def validate_question_response(
    request: QuestionRequest,
    response: QuestionResponse,
) -> None:
    """Validate one response against its exact immutable question request."""

    if not isinstance(cast(object, request), QuestionRequest):
        raise TypeError("request must be a QuestionRequest")
    if not isinstance(cast(object, response), QuestionResponse):
        raise TypeError("response must be a QuestionResponse")
    if response.request_id != request.request_id:
        raise ValueError("response request_id does not match the question request")
    if response.status == "cancelled":
        if response.reason is not None and len(response.reason) > request.max_text_chars:
            raise ValueError("cancellation reason exceeds max_text_chars")
        return

    questions = {cast(str, question["id"]): question for question in request.questions}
    answer_ids = frozenset(response.answers)
    unknown = answer_ids - questions.keys()
    if unknown:
        raise ValueError(f"answers contain unknown question ids: {_joined(unknown)}")
    missing = {
        question_id
        for question_id, question in questions.items()
        if cast(bool, question["required"]) and question_id not in answer_ids
    }
    if missing:
        raise ValueError(f"answers are missing required question ids: {_joined(missing)}")
    for question_id, answer in response.answers.items():
        _validate_answer(questions[question_id], answer, request.max_text_chars)


def question_response_message(
    request: QuestionRequest,
    response: QuestionResponse,
) -> Message:
    """Create the deterministic, model-visible external response message."""

    validate_question_response(request, response)
    body: dict[str, Any] = {
        "answers": thaw_json_value(response.answers, label="answers"),
        "request_id": response.request_id,
        "response_id": response.response_id,
        "status": response.status,
    }
    if response.reason is not None:
        body["reason"] = response.reason
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return Message.external(
        f"AskQuestion response:\n{encoded}",
        metadata=_response_metadata(response),
    )


def resume_question(
    runtime: Runtime,
    checkpoint: Checkpoint,
    response: QuestionResponse,
    *,
    stream: bool = False,
) -> Invocation:
    """Validate and resume a suspended ``AskQuestion`` invocation."""

    if not isinstance(cast(object, runtime), Runtime):
        raise TypeError("runtime must be a Runtime")
    if not isinstance(cast(object, stream), bool):
        raise TypeError("stream must be bool")
    request = extract_question_request(checkpoint)
    validate_question_response(request, response)
    state = cast(Suspended, checkpoint.snapshot.state)
    suspension = state.suspension
    selector = SuspensionSelector(
        reason=suspension.reason,
        source=suspension.source,
        wait_id=suspension.wait_id,
        metadata=suspension.metadata,
    )
    return runtime.resume(
        checkpoint,
        selector=selector,
        append_messages=(question_response_message(request, response),),
        metadata=_response_metadata(response),
        stream=stream,
    )


def _freeze_questions(
    value: object,
    max_text_chars: int,
    *,
    enabled_kinds: tuple[str, ...],
    max_questions: int,
    max_options: int,
    max_prompt_chars: int,
) -> tuple[Mapping[str, Any], ...]:
    frozen = freeze_json_value(
        value,
        label="questions",
        error_message="questions are immutable",
    )
    items = _sequence(frozen, "questions")
    if not items:
        raise ValueError("questions must not be empty")
    if len(items) > max_questions:
        raise ValueError("questions exceed max_questions")
    questions: list[Mapping[str, Any]] = []
    identifiers: set[str] = set()
    for index, item in enumerate(items):
        question = _mapping(item, f"questions[{index}]")
        _validate_question(
            question,
            f"questions[{index}]",
            max_text_chars,
            enabled_kinds=enabled_kinds,
            max_options=max_options,
            max_prompt_chars=max_prompt_chars,
        )
        question_id = cast(str, question["id"])
        if question_id in identifiers:
            raise ValueError(f"duplicate question id: {question_id}")
        identifiers.add(question_id)
        questions.append(question)
    return tuple(questions)


def _validate_question(
    question: Mapping[str, Any],
    label: str,
    max_text_chars: int,
    *,
    enabled_kinds: tuple[str, ...],
    max_options: int,
    max_prompt_chars: int,
) -> None:
    _require_keys(question, label, frozenset({"id", "kind", "prompt", "required"}))
    question_id = _non_empty_string(question["id"], f"{label}.id")
    if _QUESTION_ID_PATTERN.fullmatch(question_id) is None:
        raise ValueError(f"{label}.id has an invalid format")
    kind = _non_empty_string(question["kind"], f"{label}.kind")
    if kind not in _QUESTION_KINDS:
        raise ValueError(f"{label}.kind is unsupported: {kind}")
    if kind not in enabled_kinds:
        raise ValueError(f"{label}.kind is not enabled: {kind}")
    _bounded_string(
        question["prompt"],
        f"{label}.prompt",
        minimum=1,
        maximum=max_prompt_chars,
    )
    _boolean(question["required"], f"{label}.required")
    if "description" in question:
        _bounded_string(
            question["description"],
            f"{label}.description",
            minimum=0,
            maximum=max_prompt_chars,
        )
    allowed = _COMMON_FIELDS | _KIND_FIELDS[kind]
    required = frozenset({"id", "kind", "prompt", "required"}) | _KIND_REQUIRED_FIELDS[kind]
    _require_exact_keys(question, label, required=required, allowed=allowed)
    _validate_display_limits(
        question,
        kind,
        label,
        max_options=max_options,
        max_prompt_chars=max_prompt_chars,
    )
    validator = _QUESTION_VALIDATORS[kind]
    validator(question, f"question {question_id}", max_text_chars)
    if "default" in question:
        _ANSWER_VALIDATORS[kind](question, question["default"], max_text_chars)


def _validate_display_limits(
    question: Mapping[str, Any],
    kind: str,
    label: str,
    *,
    max_options: int,
    max_prompt_chars: int,
) -> None:
    if kind in {"single_choice", "multiple_choice", "ranking"}:
        _option_values(
            question,
            label,
            max_options=max_options,
            max_prompt_chars=max_prompt_chars,
        )
    if kind == "text" and "placeholder" in question:
        _bounded_string(
            question["placeholder"],
            f"{label}.placeholder",
            minimum=0,
            maximum=max_prompt_chars,
        )
    if kind == "scale":
        for field_name in ("minimum_label", "maximum_label"):
            if field_name in question:
                _bounded_string(
                    question[field_name],
                    f"{label}.{field_name}",
                    minimum=0,
                    maximum=max_prompt_chars,
                )


def _validate_confirm_question(
    question: Mapping[str, Any], label: str, max_text_chars: int
) -> None:
    del question, label, max_text_chars


def _validate_single_question(question: Mapping[str, Any], label: str, max_text_chars: int) -> None:
    del max_text_chars
    _option_values(question, label)
    _boolean(question["allow_custom"], f"{label}.allow_custom")


def _validate_multiple_question(
    question: Mapping[str, Any], label: str, max_text_chars: int
) -> None:
    del max_text_chars
    values = _option_values(question, label)
    allow_custom = _boolean(question["allow_custom"], f"{label}.allow_custom")
    minimum = _nonnegative_int(question["min_selections"], f"{label}.min_selections")
    maximum = _nonnegative_int(question["max_selections"], f"{label}.max_selections")
    allowed_count = len(values) + int(allow_custom)
    _validate_count_bounds(minimum, maximum, allowed_count, label, "selections")


def _validate_text_question(question: Mapping[str, Any], label: str, max_text_chars: int) -> None:
    _boolean(question["multiline"], f"{label}.multiline")
    minimum = _nonnegative_int(question["min_length"], f"{label}.min_length")
    maximum = _nonnegative_int(question["max_length"], f"{label}.max_length")
    if minimum > maximum:
        raise ValueError(f"{label}.min_length must be <= max_length")
    if maximum < 1:
        raise ValueError(f"{label}.max_length must be >= 1")
    if maximum > max_text_chars:
        raise ValueError(f"{label}.max_length exceeds request max_text_chars")
    if "placeholder" in question:
        _string(question["placeholder"], f"{label}.placeholder")


def _validate_number_question(question: Mapping[str, Any], label: str, max_text_chars: int) -> None:
    del max_text_chars
    integer_only = _boolean(question["integer_only"], f"{label}.integer_only")
    _validate_numeric_constraints(question, label, require_bounds=False)
    if integer_only:
        minimum = _optional_number(question, "minimum", label)
        maximum = _optional_number(question, "maximum", label)
        step = _optional_number(question, "step", label)
        if not integer_answer_exists(minimum, maximum, step):
            raise ValueError(f"{label} has no valid integer answer")


def _validate_date_question(question: Mapping[str, Any], label: str, max_text_chars: int) -> None:
    del max_text_chars
    minimum = _optional_date(question, "minimum", label)
    maximum = _optional_date(question, "maximum", label)
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{label}.minimum must be <= maximum")


def _validate_scale_question(question: Mapping[str, Any], label: str, max_text_chars: int) -> None:
    del max_text_chars
    _validate_numeric_constraints(question, label, require_bounds=True)
    if cast(int | float, question["minimum"]) >= cast(int | float, question["maximum"]):
        raise ValueError(f"{label}.minimum must be < maximum")
    if "minimum_label" in question:
        _string(question["minimum_label"], f"{label}.minimum_label")
    if "maximum_label" in question:
        _string(question["maximum_label"], f"{label}.maximum_label")


def _validate_ranking_question(
    question: Mapping[str, Any], label: str, max_text_chars: int
) -> None:
    del max_text_chars
    values = _option_values(question, label)
    minimum = _nonnegative_int(question["min_ranked"], f"{label}.min_ranked")
    maximum = _nonnegative_int(question["max_ranked"], f"{label}.max_ranked")
    _validate_count_bounds(minimum, maximum, len(values), label, "ranked items")


def _validate_numeric_constraints(
    question: Mapping[str, Any], label: str, *, require_bounds: bool
) -> None:
    minimum = _optional_number(question, "minimum", label)
    maximum = _optional_number(question, "maximum", label)
    if require_bounds and (minimum is None or maximum is None):
        raise ValueError(f"{label} requires minimum and maximum")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{label}.minimum must be <= maximum")
    if "step" in question:
        step = _number(question["step"], f"{label}.step")
        if step <= 0:
            raise ValueError(f"{label}.step must be > 0")


def _validate_count_bounds(
    minimum: int,
    maximum: int,
    available: int,
    label: str,
    noun: str,
) -> None:
    if minimum > maximum:
        raise ValueError(f"{label} minimum {noun} must be <= maximum")
    if maximum < 1:
        raise ValueError(f"{label} maximum {noun} must be >= 1")
    if maximum > available:
        raise ValueError(f"{label} maximum {noun} exceeds available values")


def _option_values(
    question: Mapping[str, Any],
    label: str,
    *,
    max_options: int | None = None,
    max_prompt_chars: int | None = None,
) -> tuple[str, ...]:
    options = _mapping_sequence(question["options"], f"{label}.options")
    if len(options) < 2:
        raise ValueError(f"{label}.options must contain at least two options")
    if max_options is not None and len(options) > max_options:
        raise ValueError(f"{label}.options exceed max_options")
    values: list[str] = []
    for index, option in enumerate(options):
        option_label = f"{label}.options[{index}]"
        _require_exact_keys(
            option,
            option_label,
            required=frozenset({"value", "label"}),
            allowed=frozenset({"value", "label", "description"}),
        )
        values.append(
            _bounded_string(
                option["value"],
                f"{option_label}.value",
                minimum=1,
                maximum=OPTION_VALUE_CHARS,
            )
        )
        option_text = _non_empty_string(option["label"], f"{option_label}.label")
        if max_prompt_chars is not None and len(option_text) > max_prompt_chars:
            raise ValueError(f"{option_label}.label exceeds max_prompt_chars")
        if "description" in option:
            description = _string(option["description"], f"{option_label}.description")
            if max_prompt_chars is not None and len(description) > max_prompt_chars:
                raise ValueError(f"{option_label}.description exceeds max_prompt_chars")
    if len(values) != len(set(values)):
        raise ValueError(f"{label}.options values must be unique")
    return tuple(values)


def _validate_answer(question: Mapping[str, Any], answer: object, max_text_chars: int) -> None:
    kind = cast(str, question["kind"])
    _ANSWER_VALIDATORS[kind](question, answer, max_text_chars)


def _validate_confirm_answer(
    question: Mapping[str, Any], answer: object, max_text_chars: int
) -> None:
    del max_text_chars
    _boolean(answer, _answer_label(question))


def _validate_single_answer(
    question: Mapping[str, Any], answer: object, max_text_chars: int
) -> None:
    value = _non_empty_string(answer, _answer_label(question))
    _validate_choice_value(question, value, max_text_chars)


def _validate_multiple_answer(
    question: Mapping[str, Any], answer: object, max_text_chars: int
) -> None:
    label = _answer_label(question)
    values = _string_sequence(answer, label)
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicate values")
    allowed = frozenset(_option_values(question, _question_label(question)))
    if sum(value not in allowed for value in values) > 1:
        raise ValueError(f"{label} must not contain more than one custom value")
    minimum = cast(int, question["min_selections"])
    maximum = cast(int, question["max_selections"])
    if not minimum <= len(values) <= maximum:
        raise ValueError(f"{label} must contain between {minimum} and {maximum} values")
    for value in values:
        _validate_choice_value(question, value, max_text_chars)


def _validate_text_answer(question: Mapping[str, Any], answer: object, max_text_chars: int) -> None:
    value = _string(answer, _answer_label(question))
    minimum = cast(int, question["min_length"])
    maximum = min(cast(int, question["max_length"]), max_text_chars)
    if not minimum <= len(value) <= maximum:
        raise ValueError(
            f"{_answer_label(question)} length must be between {minimum} and {maximum}"
        )


def _validate_number_answer(
    question: Mapping[str, Any], answer: object, max_text_chars: int
) -> None:
    del max_text_chars
    label = _answer_label(question)
    value = _number(answer, label)
    if (
        cast(bool, question["integer_only"])
        and not isinstance(value, int)
        and not value.is_integer()
    ):
        raise TypeError(f"{label} must be an integer")
    _validate_numeric_value(question, value, label, default_base=0)


def _validate_date_answer(question: Mapping[str, Any], answer: object, max_text_chars: int) -> None:
    del max_text_chars
    label = _answer_label(question)
    value = _iso_date(answer, label)
    minimum = _optional_date(question, "minimum", _question_label(question))
    maximum = _optional_date(question, "maximum", _question_label(question))
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be on or after {minimum.isoformat()}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be on or before {maximum.isoformat()}")


def _validate_scale_answer(
    question: Mapping[str, Any], answer: object, max_text_chars: int
) -> None:
    del max_text_chars
    label = _answer_label(question)
    value = _number(answer, label)
    minimum = cast(int | float, question["minimum"])
    _validate_numeric_value(question, value, label, default_base=minimum)


def _validate_ranking_answer(
    question: Mapping[str, Any], answer: object, max_text_chars: int
) -> None:
    del max_text_chars
    label = _answer_label(question)
    values = _string_sequence(answer, label)
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicate values")
    allowed = frozenset(_option_values(question, _question_label(question)))
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"{label} contains unknown option values: {_joined(unknown)}")
    minimum = cast(int, question["min_ranked"])
    maximum = cast(int, question["max_ranked"])
    if not minimum <= len(values) <= maximum:
        raise ValueError(f"{label} must contain between {minimum} and {maximum} values")


def _validate_choice_value(question: Mapping[str, Any], value: str, max_text_chars: int) -> None:
    allowed = frozenset(_option_values(question, _question_label(question)))
    if value in allowed:
        return
    if not cast(bool, question["allow_custom"]):
        raise ValueError(f"{_answer_label(question)} contains an unknown option value")
    if len(value) > max_text_chars:
        raise ValueError(f"{_answer_label(question)} custom value exceeds max_text_chars")


def _validate_numeric_value(
    question: Mapping[str, Any],
    value: int | float,
    label: str,
    *,
    default_base: int | float,
) -> None:
    minimum = _optional_number(question, "minimum", _question_label(question))
    maximum = _optional_number(question, "maximum", _question_label(question))
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    if "step" not in question:
        return
    step = cast(int | float, question["step"])
    base = minimum if minimum is not None else default_base
    if not _is_step_aligned(value, step, base):
        raise ValueError(f"{label} must align to step {step} from base {base}")


def _is_step_aligned(value: int | float, step: int | float, base: int | float) -> bool:
    try:
        exact_distance = (_number_fraction(value) - _number_fraction(base)) / _number_fraction(step)
    except (OverflowError, ValueError, ZeroDivisionError):
        return False
    return exact_distance.denominator == 1


def _number_fraction(value: int | float) -> Fraction:
    return Fraction(value) if isinstance(value, int) else Fraction(str(value))


def _current_waiting_outcome(
    checkpoint: Checkpoint,
    tool_call_id: str,
) -> tuple[ToolWaiting, ToolCall]:
    fact = checkpoint.fact
    if not isinstance(fact, ToolBatchFact):
        raise ValueError("AskQuestion checkpoint must end at its waiting tool batch")
    if fact.call_ids != (tool_call_id,) or fact.outcome_kinds != (ToolOutcomeKind.WAITING,):
        raise ValueError("AskQuestion suspension does not match the current waiting tool call")
    history = checkpoint.snapshot.history
    if len(history) < 2:
        raise ValueError("AskQuestion checkpoint is missing its assistant and tool messages")
    tool_message = history[-1]
    assistant_message = history[-2]
    if tool_message.role != "tool" or tool_message.tool_call_id != tool_call_id:
        raise ValueError("AskQuestion checkpoint has no matching trailing tool message")
    if assistant_message.role != "assistant" or len(assistant_message.tool_calls) != 1:
        raise ValueError("AskQuestion checkpoint has no matching assistant tool call")
    assistant_call = assistant_message.tool_calls[0]
    if assistant_call.id != tool_call_id or assistant_call.name != "AskQuestion":
        raise ValueError("AskQuestion checkpoint assistant call has the wrong identity")
    outcome = tool_message.outcome
    if not isinstance(outcome, ToolWaiting):
        raise ValueError("current AskQuestion tool message must contain ToolWaiting")
    return outcome, assistant_call


def _waiting_payload(waiting: ToolWaiting) -> Mapping[str, Any]:
    payload = thaw_json_value(
        waiting.structured_content,
        label="AskQuestion waiting structured_content",
    )
    return _mapping(payload, "AskQuestion waiting structured_content")


def _validate_waiting_payload(payload: Mapping[str, Any], request_id: str) -> None:
    expected = frozenset(
        {
            "status",
            "schema_version",
            "request_id",
            "enabled_kinds",
            "max_questions",
            "max_options",
            "max_prompt_chars",
            "max_text_chars",
            "questions",
        }
    )
    _require_exact_keys(payload, "AskQuestion waiting payload", required=expected, allowed=expected)
    if payload["status"] != "waiting":
        raise ValueError("AskQuestion waiting payload status must be 'waiting'")
    if (
        _exact_int(payload["schema_version"], "AskQuestion payload schema_version")
        != SCHEMA_VERSION
    ):
        raise ValueError("unsupported AskQuestion waiting payload schema_version")
    payload_request_id = _non_empty_string(payload["request_id"], "AskQuestion payload request_id")
    if payload_request_id != request_id:
        raise ValueError("AskQuestion payload request_id does not match suspension wait_id")


def _response_metadata(response: QuestionResponse) -> Mapping[str, Any]:
    return {
        "kind": "question_response",
        "request_id": response.request_id,
        "response_id": response.response_id,
        "status": response.status,
    }


def _freeze_mapping(value: object, label: str) -> Mapping[str, Any]:
    frozen = freeze_json_value(value, label=label, error_message=f"{label} are immutable")
    return _mapping(frozen, label)


def _enabled_kinds(value: object) -> tuple[str, ...]:
    kinds = _string_sequence(value, "enabled_kinds")
    if not kinds:
        raise ValueError("enabled_kinds must not be empty")
    if len(kinds) != len(set(kinds)):
        raise ValueError("enabled_kinds must not contain duplicates")
    unknown = set(kinds) - set(SUPPORTED_QUESTION_KINDS)
    if unknown:
        raise ValueError(f"enabled_kinds contain unsupported values: {_joined(unknown)}")
    expected_order = tuple(kind for kind in SUPPORTED_QUESTION_KINDS if kind in kinds)
    if kinds != expected_order:
        raise ValueError("enabled_kinds must use canonical order")
    return kinds


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in cast(Mapping[object, object], value)):
        raise TypeError(f"{label} keys must be strings")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{label} must be an array")
    return cast(Sequence[Any], value)


def _mapping_sequence(value: object, label: str) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        _mapping(item, f"{label}[{index}]") for index, item in enumerate(_sequence(value, label))
    )


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    return tuple(
        _non_empty_string(item, f"{label}[{index}]")
        for index, item in enumerate(_sequence(value, label))
    )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _bounded_string(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> str:
    text = _string(value, label)
    if not minimum <= len(text) <= maximum:
        raise ValueError(f"{label} length must be between {minimum} and {maximum}")
    return text


def _non_empty_string(value: object, label: str) -> str:
    text = _string(value, label)
    if not text:
        raise ValueError(f"{label} must not be empty")
    return text


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _exact_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    number = _exact_int(value, label)
    if number < 0:
        raise ValueError(f"{label} must be >= 0")
    return number


def _positive_int(value: object, label: str) -> int:
    number = _exact_int(value, label)
    if number < 1:
        raise ValueError(f"{label} must be >= 1")
    return number


def _number(value: object, label: str) -> int | float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{label} must be a number")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{label} must be finite")
    try:
        json.dumps(value, allow_nan=False)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} must be JSON-serializable") from error
    return value


def _optional_number(value: Mapping[str, Any], key: str, label: str) -> int | float | None:
    if key not in value:
        return None
    return _number(value[key], f"{label}.{key}")


def _iso_date(value: object, label: str) -> date:
    text = _string(value, label)
    if _DATE_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{label} must use YYYY-MM-DD")
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{label} must be a valid calendar date") from error


def _optional_date(value: Mapping[str, Any], key: str, label: str) -> date | None:
    if key not in value:
        return None
    return _iso_date(value[key], f"{label}.{key}")


def _require_keys(value: Mapping[str, Any], label: str, required: frozenset[str]) -> None:
    missing = required - value.keys()
    if missing:
        raise ValueError(f"{label} is missing fields: {_joined(missing)}")


def _require_exact_keys(
    value: Mapping[str, Any],
    label: str,
    *,
    required: frozenset[str],
    allowed: frozenset[str],
) -> None:
    _require_keys(value, label, required)
    extra = value.keys() - allowed
    if extra:
        raise ValueError(f"{label} contains unexpected fields: {_joined(extra)}")


def _question_label(question: Mapping[str, Any]) -> str:
    return f"question {cast(str, question['id'])}"


def _answer_label(question: Mapping[str, Any]) -> str:
    return f"answer for question {cast(str, question['id'])}"


def _joined(values: Sequence[str] | set[str] | frozenset[str]) -> str:
    return ", ".join(sorted(values))


_QUESTION_VALIDATORS = {
    "confirm": _validate_confirm_question,
    "single_choice": _validate_single_question,
    "multiple_choice": _validate_multiple_question,
    "text": _validate_text_question,
    "number": _validate_number_question,
    "date": _validate_date_question,
    "scale": _validate_scale_question,
    "ranking": _validate_ranking_question,
}

_ANSWER_VALIDATORS = {
    "confirm": _validate_confirm_answer,
    "single_choice": _validate_single_answer,
    "multiple_choice": _validate_multiple_answer,
    "text": _validate_text_answer,
    "number": _validate_number_answer,
    "date": _validate_date_answer,
    "scale": _validate_scale_answer,
    "ranking": _validate_ranking_answer,
}


__all__ = [
    "QuestionRequest",
    "QuestionResponse",
    "extract_question_request",
    "question_response_message",
    "resume_question",
    "validate_question_response",
]
