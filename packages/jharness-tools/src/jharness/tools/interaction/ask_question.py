"""The Host-mediated AskQuestion preset."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass, field

from jharness.kernel import (
    ContentPart,
    SettledResult,
    Suspension,
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolFailure,
    ToolResult,
    ToolRisk,
    ToolSpec,
    ToolWaiting,
    WaitingResult,
    thaw_json_value,
)
from jharness.tools.interaction._schema import (
    DEFAULT_MAX_ANSWER_CHARS,
    DEFAULT_MAX_OPTIONS,
    DEFAULT_MAX_PROMPT_CHARS,
    DEFAULT_MAX_QUESTIONS,
    SCHEMA_VERSION,
    SUPPORTED_QUESTION_KINDS,
    QuestionKind,
    QuestionValidationError,
    build_contract_id,
    build_request_id,
    input_schema,
    normalize_questions,
    output_schema,
    validate_enabled_kinds,
)


@dataclass(frozen=True, slots=True, init=False)
class AskQuestionTool:
    """Suspend a run while the Host collects structured user input."""

    enabled_kinds: frozenset[QuestionKind]
    max_questions: int
    max_options: int
    max_prompt_chars: int
    max_answer_chars: int
    spec: ToolSpec = field(repr=False)

    def __init__(
        self,
        *,
        enabled_kinds: Collection[str] = SUPPORTED_QUESTION_KINDS,
        max_questions: int = DEFAULT_MAX_QUESTIONS,
        max_options: int = DEFAULT_MAX_OPTIONS,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
        max_answer_chars: int = DEFAULT_MAX_ANSWER_CHARS,
    ) -> None:
        kinds = validate_enabled_kinds(enabled_kinds)
        max_questions = _positive_int(max_questions, "max_questions")
        max_options = _positive_int(max_options, "max_options")
        if max_options < 2:
            raise ValueError("max_options must be at least 2")
        max_prompt_chars = _positive_int(max_prompt_chars, "max_prompt_chars")
        max_answer_chars = _positive_int(max_answer_chars, "max_answer_chars")
        object.__setattr__(self, "enabled_kinds", kinds)
        object.__setattr__(self, "max_questions", max_questions)
        object.__setattr__(self, "max_options", max_options)
        object.__setattr__(self, "max_prompt_chars", max_prompt_chars)
        object.__setattr__(self, "max_answer_chars", max_answer_chars)
        object.__setattr__(
            self,
            "spec",
            _spec(
                kinds,
                max_questions=max_questions,
                max_options=max_options,
                max_prompt_chars=max_prompt_chars,
                max_answer_chars=max_answer_chars,
            ),
        )

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        if context.cancel_requested:
            return _failure("cancelled", "AskQuestion was cancelled.")
        try:
            questions = normalize_questions(
                thaw_json_value(call.arguments, label="AskQuestion arguments"),
                enabled_kinds=self.enabled_kinds,
                max_questions=self.max_questions,
                max_options=self.max_options,
                max_prompt_chars=self.max_prompt_chars,
                max_answer_chars=self.max_answer_chars,
            )
        except QuestionValidationError as exc:
            return _failure("invalid_question", str(exc))

        request_id = build_request_id(context.run.run_id, call.id)
        contract_id = build_contract_id(
            self.enabled_kinds,
            max_questions=self.max_questions,
            max_options=self.max_options,
            max_prompt_chars=self.max_prompt_chars,
            max_answer_chars=self.max_answer_chars,
        )
        structured = {
            "status": "waiting",
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "enabled_kinds": [
                kind for kind in SUPPORTED_QUESTION_KINDS if kind in self.enabled_kinds
            ],
            "max_questions": self.max_questions,
            "max_options": self.max_options,
            "max_prompt_chars": self.max_prompt_chars,
            "max_text_chars": self.max_answer_chars,
            "questions": questions,
        }
        return WaitingResult(
            ToolWaiting(
                (ContentPart.text_part("Waiting for the user to answer the questions."),),
                structured_content=structured,
            ),
            Suspension(
                reason="human_input",
                source="AskQuestion",
                wait_id=request_id,
                metadata={
                    "contract_id": contract_id,
                    "tool_call_id": call.id,
                    "schema_version": SCHEMA_VERSION,
                },
            ),
        )


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _failure(code: str, message: str) -> ToolResult:
    return SettledResult(ToolFailure.from_error(code, message))


def _spec(
    enabled_kinds: frozenset[QuestionKind],
    *,
    max_questions: int,
    max_options: int,
    max_prompt_chars: int,
    max_answer_chars: int,
) -> ToolSpec:
    kinds = ", ".join(kind for kind in SUPPORTED_QUESTION_KINDS if kind in enabled_kinds)
    model_input_schema = input_schema(
        enabled_kinds,
        max_questions=max_questions,
        max_options=max_options,
        max_prompt_chars=max_prompt_chars,
        max_answer_chars=max_answer_chars,
    )
    model_output_schema = output_schema(
        enabled_kinds,
        max_questions=max_questions,
        max_options=max_options,
        max_prompt_chars=max_prompt_chars,
        max_answer_chars=max_answer_chars,
    )
    try:
        json.dumps((model_input_schema, model_output_schema), allow_nan=False)
    except (OverflowError, ValueError) as exc:
        raise ValueError("AskQuestion limits must produce JSON-serializable schemas") from exc
    return ToolSpec(
        name="AskQuestion",
        description=(
            "Ask the user one or more structured questions and wait for the Host to collect "
            f"their answers. Supported interaction kinds: {kinds}. Use stable question ids and "
            "option values. This tool does not answer questions on the user's behalf."
        ),
        input_schema=model_input_schema,
        output_schema=model_output_schema,
        execution=ToolExecution(concurrency="serial", read_only=True, idempotent=True),
        risk=ToolRisk(
            filesystem="none",
            network="none",
            subprocess=False,
            destructive=False,
            requires_approval=False,
        ),
    )
