"""Schema construction and normalization for the AskQuestion preset."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection, Mapping
from datetime import date
from fractions import Fraction
from hashlib import sha256
from typing import Any, Literal, TypeAlias, cast

from jharness.kernel import JsonValue

QuestionKind: TypeAlias = Literal[
    "confirm",
    "single_choice",
    "multiple_choice",
    "text",
    "number",
    "date",
    "scale",
    "ranking",
]

SUPPORTED_QUESTION_KINDS: tuple[QuestionKind, ...] = (
    "confirm",
    "single_choice",
    "multiple_choice",
    "text",
    "number",
    "date",
    "scale",
    "ranking",
)
SCHEMA_VERSION = 1
DEFAULT_MAX_QUESTIONS = 8
DEFAULT_MAX_OPTIONS = 12
DEFAULT_MAX_PROMPT_CHARS = 2_000
DEFAULT_MAX_ANSWER_CHARS = 16_384

_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{0,63}(?![\s\S])"
_DATE_PATTERN = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}(?![\s\S])"
OPTION_VALUE_CHARS = 128
_COMMON_FIELDS = frozenset({"id", "kind", "prompt", "description", "required"})
_KIND_FIELDS: Mapping[QuestionKind, frozenset[str]] = {
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


class QuestionValidationError(ValueError):
    """A recoverable validation error in model-provided question data."""


def build_request_id(run_id: str, tool_call_id: str) -> str:
    """Build an unambiguous stable identifier for one question request."""

    return f"ask-question:{len(run_id)}:{run_id}:{len(tool_call_id)}:{tool_call_id}"


def build_contract_id(
    enabled_kinds: Collection[str],
    *,
    max_questions: int,
    max_options: int,
    max_prompt_chars: int,
    max_answer_chars: int,
) -> str:
    """Fingerprint the Host-owned request contract stored beside a suspension."""

    body = {
        "enabled_kinds": [kind for kind in SUPPORTED_QUESTION_KINDS if kind in enabled_kinds],
        "max_answer_chars": max_answer_chars,
        "max_options": max_options,
        "max_prompt_chars": max_prompt_chars,
        "max_questions": max_questions,
        "schema_version": SCHEMA_VERSION,
    }
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return f"sha256:{sha256(encoded).hexdigest()}"


def validate_enabled_kinds(value: object) -> frozenset[QuestionKind]:
    """Return a validated immutable set of enabled interaction kinds."""

    if isinstance(value, str | bytes) or not isinstance(value, Collection):
        raise TypeError("enabled_kinds must be a collection of question kind strings")
    items = list(cast(Collection[object], value))
    if not items:
        raise ValueError("enabled_kinds must not be empty")
    if any(not isinstance(item, str) for item in items):
        raise TypeError("enabled_kinds must contain only strings")
    names = cast(list[str], items)
    if len(names) != len(set(names)):
        raise ValueError("enabled_kinds must not contain duplicates")
    supported = frozenset(SUPPORTED_QUESTION_KINDS)
    unknown = sorted(set(names) - supported)
    if unknown:
        raise ValueError(f"unsupported question kinds: {', '.join(unknown)}")
    return frozenset(cast(Collection[QuestionKind], names))


def input_schema(
    enabled_kinds: frozenset[QuestionKind],
    *,
    max_questions: int,
    max_options: int,
    max_prompt_chars: int,
    max_answer_chars: int,
) -> dict[str, Any]:
    """Build the strict model-facing Draft 2020-12 input schema."""

    return {
        "$schema": _DRAFT_2020_12,
        "type": "object",
        "required": ["questions"],
        "properties": {
            "questions": _questions_schema(
                enabled_kinds,
                max_questions=max_questions,
                max_options=max_options,
                max_prompt_chars=max_prompt_chars,
                max_answer_chars=max_answer_chars,
            )
        },
        "additionalProperties": False,
    }


def output_schema(
    enabled_kinds: frozenset[QuestionKind],
    *,
    max_questions: int,
    max_options: int,
    max_prompt_chars: int,
    max_answer_chars: int,
) -> dict[str, Any]:
    """Build the strict waiting-or-null Draft 2020-12 output schema."""

    waiting = {
        "type": "object",
        "required": [
            "status",
            "schema_version",
            "request_id",
            "enabled_kinds",
            "max_questions",
            "max_options",
            "max_prompt_chars",
            "max_text_chars",
            "questions",
        ],
        "properties": {
            "status": {"const": "waiting"},
            "schema_version": {"const": SCHEMA_VERSION},
            "request_id": {"type": "string", "minLength": 1},
            "enabled_kinds": {
                "const": [kind for kind in SUPPORTED_QUESTION_KINDS if kind in enabled_kinds]
            },
            "max_questions": {"const": max_questions},
            "max_options": {"const": max_options},
            "max_prompt_chars": {"const": max_prompt_chars},
            "max_text_chars": {"const": max_answer_chars},
            "questions": _questions_schema(
                enabled_kinds,
                max_questions=max_questions,
                max_options=max_options,
                max_prompt_chars=max_prompt_chars,
                max_answer_chars=max_answer_chars,
            ),
        },
        "additionalProperties": False,
    }
    return {
        "$schema": _DRAFT_2020_12,
        "oneOf": [waiting, {"type": "null"}],
    }


def normalize_questions(
    arguments: object,
    *,
    enabled_kinds: frozenset[QuestionKind],
    max_questions: int,
    max_options: int,
    max_prompt_chars: int,
    max_answer_chars: int,
) -> list[dict[str, JsonValue]]:
    """Validate and normalize a complete AskQuestion argument object."""

    root = _object(arguments, "arguments")
    _keys(root, frozenset({"questions"}), frozenset({"questions"}), "arguments")
    raw_questions = root["questions"]
    if not isinstance(raw_questions, list):
        raise QuestionValidationError("questions must be an array")
    if not 1 <= len(raw_questions) <= max_questions:
        raise QuestionValidationError(f"questions must contain between 1 and {max_questions} items")

    normalized: list[dict[str, JsonValue]] = []
    seen_ids: set[str] = set()
    for index, raw_question in enumerate(raw_questions):
        label = f"questions[{index}]"
        question = _object(raw_question, label)
        kind_value = question.get("kind")
        if not isinstance(kind_value, str) or kind_value not in SUPPORTED_QUESTION_KINDS:
            raise QuestionValidationError(f"{label}.kind is not supported")
        kind = kind_value
        if kind not in enabled_kinds:
            raise QuestionValidationError(f"{label}.kind is not enabled: {kind}")
        item = _normalize_question(
            question,
            kind,
            label=label,
            max_options=max_options,
            max_prompt_chars=max_prompt_chars,
            max_answer_chars=max_answer_chars,
        )
        question_id = cast(str, item["id"])
        if question_id in seen_ids:
            raise QuestionValidationError(f"duplicate question id: {question_id}")
        seen_ids.add(question_id)
        normalized.append(item)
    return normalized


def _questions_schema(
    enabled_kinds: frozenset[QuestionKind],
    *,
    max_questions: int,
    max_options: int,
    max_prompt_chars: int,
    max_answer_chars: int,
) -> dict[str, Any]:
    variants = _question_variants(
        max_options=max_options,
        max_prompt_chars=max_prompt_chars,
        max_answer_chars=max_answer_chars,
    )
    return {
        "type": "array",
        "minItems": 1,
        "maxItems": max_questions,
        "items": {
            "oneOf": [variants[kind] for kind in SUPPORTED_QUESTION_KINDS if kind in enabled_kinds]
        },
    }


def _question_variants(
    *,
    max_options: int,
    max_prompt_chars: int,
    max_answer_chars: int,
) -> dict[QuestionKind, dict[str, Any]]:
    options = {
        "type": "array",
        "minItems": 2,
        "maxItems": max_options,
        "items": _option_schema(max_prompt_chars),
    }
    common = _common_properties(max_prompt_chars)
    return {
        "confirm": _variant(common, "confirm", {"default": {"type": "boolean"}}),
        "single_choice": _variant(
            common,
            "single_choice",
            {
                "options": options,
                "allow_custom": {"type": "boolean", "default": False},
                "default": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": max(OPTION_VALUE_CHARS, max_answer_chars),
                },
            },
            required=("options",),
        ),
        "multiple_choice": _variant(
            common,
            "multiple_choice",
            {
                "options": options,
                "allow_custom": {"type": "boolean", "default": False},
                "min_selections": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": max_options + 1,
                },
                "max_selections": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_options + 1,
                },
                "default": {
                    "type": "array",
                    "maxItems": max_options + 1,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": max(OPTION_VALUE_CHARS, max_answer_chars),
                    },
                },
            },
            required=("options",),
        ),
        "text": _variant(
            common,
            "text",
            {
                "multiline": {"type": "boolean", "default": False},
                "placeholder": {"type": "string", "maxLength": max_prompt_chars},
                "min_length": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": max_answer_chars,
                },
                "max_length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_answer_chars,
                    "default": max_answer_chars,
                },
                "default": {"type": "string", "maxLength": max_answer_chars},
            },
        ),
        "number": _variant(
            common,
            "number",
            {
                "minimum": {"type": "number"},
                "maximum": {"type": "number"},
                "step": {"type": "number", "exclusiveMinimum": 0},
                "integer_only": {"type": "boolean", "default": False},
                "default": {"type": "number"},
            },
        ),
        "date": _variant(
            common,
            "date",
            {
                "minimum": _date_schema(),
                "maximum": _date_schema(),
                "default": _date_schema(),
            },
        ),
        "scale": _variant(
            common,
            "scale",
            {
                "minimum": {"type": "number"},
                "maximum": {"type": "number"},
                "step": {"type": "number", "exclusiveMinimum": 0, "default": 1},
                "minimum_label": {"type": "string", "maxLength": max_prompt_chars},
                "maximum_label": {"type": "string", "maxLength": max_prompt_chars},
                "default": {"type": "number"},
            },
            required=("minimum", "maximum"),
        ),
        "ranking": _variant(
            common,
            "ranking",
            {
                "options": options,
                "min_ranked": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": max_options,
                },
                "max_ranked": {"type": "integer", "minimum": 1, "maximum": max_options},
                "default": {
                    "type": "array",
                    "maxItems": max_options,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": OPTION_VALUE_CHARS,
                    },
                },
            },
            required=("options",),
        ),
    }


def _common_properties(max_prompt_chars: int) -> dict[str, Any]:
    return {
        "id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 64,
            "pattern": _ID_PATTERN,
        },
        "prompt": {"type": "string", "minLength": 1, "maxLength": max_prompt_chars},
        "description": {"type": "string", "maxLength": max_prompt_chars},
        "required": {"type": "boolean", "default": True},
    }


def _variant(
    common: Mapping[str, Any],
    kind: QuestionKind,
    extra: Mapping[str, Any],
    *,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["id", "kind", "prompt", *required],
        "properties": {**common, "kind": {"const": kind}, **extra},
        "additionalProperties": False,
    }


def _option_schema(max_prompt_chars: int) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["value", "label"],
        "properties": {
            "value": {"type": "string", "minLength": 1, "maxLength": OPTION_VALUE_CHARS},
            "label": {"type": "string", "minLength": 1, "maxLength": max_prompt_chars},
            "description": {"type": "string", "maxLength": max_prompt_chars},
        },
        "additionalProperties": False,
    }


def _date_schema() -> dict[str, Any]:
    return {"type": "string", "pattern": _DATE_PATTERN, "format": "date"}


def _normalize_question(
    question: dict[str, JsonValue],
    kind: QuestionKind,
    *,
    label: str,
    max_options: int,
    max_prompt_chars: int,
    max_answer_chars: int,
) -> dict[str, JsonValue]:
    required_fields: set[str] = (
        {"options"} if kind in {"single_choice", "multiple_choice", "ranking"} else set()
    )
    if kind == "scale":
        required_fields.update({"minimum", "maximum"})
    _keys(
        question,
        _COMMON_FIELDS | _KIND_FIELDS[kind],
        frozenset({"id", "kind", "prompt", *required_fields}),
        label,
    )
    result = _normalize_common(question, kind, label=label, max_prompt_chars=max_prompt_chars)
    if kind == "confirm":
        _normalize_confirm(question, result, label)
    elif kind == "single_choice" or kind == "multiple_choice":
        _normalize_choice(
            question,
            result,
            kind,
            label=label,
            max_options=max_options,
            max_prompt_chars=max_prompt_chars,
            max_answer_chars=max_answer_chars,
        )
    elif kind == "text":
        _normalize_text(
            question,
            result,
            label=label,
            max_prompt_chars=max_prompt_chars,
            max_answer_chars=max_answer_chars,
        )
    elif kind == "number":
        _normalize_number(question, result, label)
    elif kind == "date":
        _normalize_date(question, result, label)
    elif kind == "scale":
        _normalize_scale(question, result, label=label, max_prompt_chars=max_prompt_chars)
    else:
        _normalize_ranking(
            question,
            result,
            label=label,
            max_options=max_options,
            max_prompt_chars=max_prompt_chars,
        )
    return result


def _normalize_common(
    question: dict[str, JsonValue],
    kind: QuestionKind,
    *,
    label: str,
    max_prompt_chars: int,
) -> dict[str, JsonValue]:
    question_id = _text(question.get("id"), f"{label}.id", minimum=1, maximum=64)
    if re.fullmatch(_ID_PATTERN, question_id) is None:
        raise QuestionValidationError(f"{label}.id has an invalid format")
    result: dict[str, JsonValue] = {
        "id": question_id,
        "kind": kind,
        "prompt": _text(
            question.get("prompt"),
            f"{label}.prompt",
            minimum=1,
            maximum=max_prompt_chars,
        ),
        "required": _optional_bool(question, "required", True, label),
    }
    if "description" in question:
        result["description"] = _text(
            question["description"],
            f"{label}.description",
            maximum=max_prompt_chars,
        )
    return result


def _normalize_confirm(
    question: dict[str, JsonValue],
    result: dict[str, JsonValue],
    label: str,
) -> None:
    if "default" in question:
        result["default"] = _bool(question["default"], f"{label}.default")


def _normalize_choice(
    question: dict[str, JsonValue],
    result: dict[str, JsonValue],
    kind: Literal["single_choice", "multiple_choice"],
    *,
    label: str,
    max_options: int,
    max_prompt_chars: int,
    max_answer_chars: int,
) -> None:
    options, option_values = _normalize_options(
        question.get("options"),
        label=f"{label}.options",
        max_options=max_options,
        max_prompt_chars=max_prompt_chars,
    )
    result["options"] = cast(JsonValue, options)
    allow_custom = _optional_bool(question, "allow_custom", False, label)
    result["allow_custom"] = allow_custom
    if kind == "single_choice":
        if "default" in question:
            default = _text(
                question["default"],
                f"{label}.default",
                minimum=1,
                maximum=max(OPTION_VALUE_CHARS, max_answer_chars),
            )
            _validate_choice_value(
                default,
                option_values,
                allow_custom,
                max_custom_chars=max_answer_chars,
                label=f"{label}.default",
            )
            result["default"] = default
        return

    required = cast(bool, result["required"])
    capacity = len(options) + int(allow_custom)
    minimum = _optional_int(question, "min_selections", 1 if required else 0, label)
    maximum = _optional_int(question, "max_selections", capacity, label)
    if minimum < 0 or maximum < 1 or maximum > capacity or minimum > maximum:
        raise QuestionValidationError(f"{label} has invalid selection bounds")
    result["min_selections"] = minimum
    result["max_selections"] = maximum
    if "default" in question:
        defaults = _text_array(
            question["default"],
            f"{label}.default",
            maximum=max(OPTION_VALUE_CHARS, max_answer_chars),
        )
        _unique(defaults, f"{label}.default")
        custom_values = [value for value in defaults if value not in option_values]
        if len(custom_values) > 1:
            raise QuestionValidationError(f"{label}.default contains more than one custom value")
        for value in defaults:
            _validate_choice_value(
                value,
                option_values,
                allow_custom,
                max_custom_chars=max_answer_chars,
                label=f"{label}.default",
            )
        if not minimum <= len(defaults) <= maximum:
            raise QuestionValidationError(f"{label}.default violates selection bounds")
        result["default"] = cast(JsonValue, defaults)


def _normalize_text(
    question: dict[str, JsonValue],
    result: dict[str, JsonValue],
    *,
    label: str,
    max_prompt_chars: int,
    max_answer_chars: int,
) -> None:
    result["multiline"] = _optional_bool(question, "multiline", False, label)
    required = cast(bool, result["required"])
    minimum = _optional_int(question, "min_length", 1 if required else 0, label)
    maximum = _optional_int(question, "max_length", max_answer_chars, label)
    if minimum < 0 or maximum < 1 or maximum > max_answer_chars or minimum > maximum:
        raise QuestionValidationError(f"{label} has invalid text length bounds")
    result["min_length"] = minimum
    result["max_length"] = maximum
    if "placeholder" in question:
        result["placeholder"] = _text(
            question["placeholder"],
            f"{label}.placeholder",
            maximum=max_prompt_chars,
        )
    if "default" in question:
        default = _text(question["default"], f"{label}.default", maximum=max_answer_chars)
        if not minimum <= len(default) <= maximum:
            raise QuestionValidationError(f"{label}.default violates text length bounds")
        result["default"] = default


def _normalize_number(
    question: dict[str, JsonValue],
    result: dict[str, JsonValue],
    label: str,
) -> None:
    minimum = _optional_number(question, "minimum", label)
    maximum = _optional_number(question, "maximum", label)
    step = _optional_number(question, "step", label)
    if minimum is not None:
        result["minimum"] = minimum
    if maximum is not None:
        result["maximum"] = maximum
    if minimum is not None and maximum is not None and minimum > maximum:
        raise QuestionValidationError(f"{label}.minimum cannot exceed maximum")
    if step is not None:
        if step <= 0:
            raise QuestionValidationError(f"{label}.step must be positive")
        result["step"] = step
    integer_only = _optional_bool(question, "integer_only", False, label)
    result["integer_only"] = integer_only
    if integer_only and not integer_answer_exists(minimum, maximum, step):
        raise QuestionValidationError(f"{label} has no valid integer answer")
    if "default" in question:
        default = _normalize_number_default(question["default"], integer_only, label)
        _validate_numeric_default(default, minimum, maximum, step, f"{label}.default")
        result["default"] = default


def _normalize_number_default(value: object, integer_only: bool, label: str) -> int | float:
    default = _number(value, f"{label}.default")
    if not integer_only or isinstance(default, int):
        return default
    if default.is_integer():
        return int(default)
    raise QuestionValidationError(f"{label}.default must be an integer")


def _normalize_date(
    question: dict[str, JsonValue],
    result: dict[str, JsonValue],
    label: str,
) -> None:
    minimum = _optional_date(question, "minimum", label)
    maximum = _optional_date(question, "maximum", label)
    if minimum is not None:
        result["minimum"] = minimum.isoformat()
    if maximum is not None:
        result["maximum"] = maximum.isoformat()
    if minimum is not None and maximum is not None and minimum > maximum:
        raise QuestionValidationError(f"{label}.minimum cannot exceed maximum")
    if "default" in question:
        default = _iso_date(question["default"], f"{label}.default")
        if minimum is not None and default < minimum:
            raise QuestionValidationError(f"{label}.default is below minimum")
        if maximum is not None and default > maximum:
            raise QuestionValidationError(f"{label}.default exceeds maximum")
        result["default"] = default.isoformat()


def _normalize_scale(
    question: dict[str, JsonValue],
    result: dict[str, JsonValue],
    *,
    label: str,
    max_prompt_chars: int,
) -> None:
    minimum = _number(question.get("minimum"), f"{label}.minimum")
    maximum = _number(question.get("maximum"), f"{label}.maximum")
    if minimum >= maximum:
        raise QuestionValidationError(f"{label}.minimum must be less than maximum")
    step = _optional_number(question, "step", label)
    if step is None:
        step = 1
    if step <= 0:
        raise QuestionValidationError(f"{label}.step must be positive")
    result.update({"minimum": minimum, "maximum": maximum, "step": step})
    for field_name in ("minimum_label", "maximum_label"):
        if field_name in question:
            result[field_name] = _text(
                question[field_name],
                f"{label}.{field_name}",
                maximum=max_prompt_chars,
            )
    if "default" in question:
        default = _number(question["default"], f"{label}.default")
        _validate_numeric_default(default, minimum, maximum, step, f"{label}.default")
        result["default"] = default


def _normalize_ranking(
    question: dict[str, JsonValue],
    result: dict[str, JsonValue],
    *,
    label: str,
    max_options: int,
    max_prompt_chars: int,
) -> None:
    options, option_values = _normalize_options(
        question.get("options"),
        label=f"{label}.options",
        max_options=max_options,
        max_prompt_chars=max_prompt_chars,
    )
    result["options"] = cast(JsonValue, options)
    required = cast(bool, result["required"])
    minimum = _optional_int(question, "min_ranked", 1 if required else 0, label)
    maximum = _optional_int(question, "max_ranked", len(options), label)
    if minimum < 0 or maximum < 1 or maximum > len(options) or minimum > maximum:
        raise QuestionValidationError(f"{label} has invalid ranking bounds")
    result["min_ranked"] = minimum
    result["max_ranked"] = maximum
    if "default" in question:
        defaults = _text_array(
            question["default"],
            f"{label}.default",
            maximum=OPTION_VALUE_CHARS,
        )
        _unique(defaults, f"{label}.default")
        if any(value not in option_values for value in defaults):
            raise QuestionValidationError(f"{label}.default contains an unknown option")
        if not minimum <= len(defaults) <= maximum:
            raise QuestionValidationError(f"{label}.default violates ranking bounds")
        result["default"] = cast(JsonValue, defaults)


def _normalize_options(
    value: object,
    *,
    label: str,
    max_options: int,
    max_prompt_chars: int,
) -> tuple[list[dict[str, JsonValue]], frozenset[str]]:
    if not isinstance(value, list):
        raise QuestionValidationError(f"{label} must be an array")
    raw_options = cast(list[object], value)
    if not 2 <= len(raw_options) <= max_options:
        raise QuestionValidationError(f"{label} must contain between 2 and {max_options} options")
    normalized: list[dict[str, JsonValue]] = []
    values: list[str] = []
    for index, raw_option in enumerate(raw_options):
        option_label = f"{label}[{index}]"
        option = _object(raw_option, option_label)
        _keys(
            option,
            frozenset({"value", "label", "description"}),
            frozenset({"value", "label"}),
            option_label,
        )
        item: dict[str, JsonValue] = {
            "value": _text(
                option.get("value"),
                f"{option_label}.value",
                minimum=1,
                maximum=OPTION_VALUE_CHARS,
            ),
            "label": _text(
                option.get("label"),
                f"{option_label}.label",
                minimum=1,
                maximum=max_prompt_chars,
            ),
        }
        if "description" in option:
            item["description"] = _text(
                option["description"],
                f"{option_label}.description",
                maximum=max_prompt_chars,
            )
        values.append(cast(str, item["value"]))
        normalized.append(item)
    _unique(values, label)
    return normalized, frozenset(values)


def _validate_choice_value(
    value: str,
    options: frozenset[str],
    allow_custom: bool,
    *,
    max_custom_chars: int,
    label: str,
) -> None:
    if value in options:
        return
    if not allow_custom:
        raise QuestionValidationError(f"{label} is not an available option")
    if len(value) > max_custom_chars:
        raise QuestionValidationError(f"{label} custom value exceeds max_answer_chars")


def _validate_numeric_default(
    value: int | float,
    minimum: int | float | None,
    maximum: int | float | None,
    step: int | float | None,
    label: str,
) -> None:
    if minimum is not None and value < minimum:
        raise QuestionValidationError(f"{label} is below minimum")
    if maximum is not None and value > maximum:
        raise QuestionValidationError(f"{label} exceeds maximum")
    if step is not None:
        origin = minimum if minimum is not None else 0
        if not _is_step_aligned(value, step, origin):
            raise QuestionValidationError(f"{label} is not aligned to step")


def _is_step_aligned(
    value: int | float,
    step: int | float,
    origin: int | float,
) -> bool:
    try:
        exact_distance = (_number_fraction(value) - _number_fraction(origin)) / _number_fraction(
            step
        )
    except (OverflowError, ValueError, ZeroDivisionError):
        return False
    return exact_distance.denominator == 1


def _number_fraction(value: int | float) -> Fraction:
    return Fraction(value) if isinstance(value, int) else Fraction(str(value))


def integer_answer_exists(
    minimum: int | float | None,
    maximum: int | float | None,
    step: int | float | None,
) -> bool:
    """Return whether integer-only numeric constraints have at least one answer."""

    lower = None if minimum is None else _number_fraction(minimum)
    upper = None if maximum is None else _number_fraction(maximum)
    if step is None:
        lower_integer = None if lower is None else _fraction_ceiling(lower)
        upper_integer = None if upper is None else _fraction_floor(upper)
        return lower_integer is None or upper_integer is None or lower_integer <= upper_integer

    origin = lower if lower is not None else Fraction(0)
    stride = _number_fraction(step)
    common_denominator = math.lcm(origin.denominator, stride.denominator)
    origin_units = origin.numerator * (common_denominator // origin.denominator)
    stride_units = stride.numerator * (common_denominator // stride.denominator)
    divisor = math.gcd(stride_units, common_denominator)
    if origin_units % divisor != 0:
        return False

    modulus = common_denominator // divisor
    if modulus == 1:
        first_solution = 0
    else:
        inverse = pow(stride_units // divisor, -1, modulus)
        first_solution = (-origin_units // divisor * inverse) % modulus

    lower_step = None if lower is None else _fraction_ceiling((lower - origin) / stride)
    upper_step = None if upper is None else _fraction_floor((upper - origin) / stride)
    if lower_step is None or upper_step is None:
        return True
    if lower_step > upper_step:
        return False
    multiplier = -(-(lower_step - first_solution) // modulus)
    return first_solution + multiplier * modulus <= upper_step


def _fraction_floor(value: Fraction) -> int:
    return value.numerator // value.denominator


def _fraction_ceiling(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _object(value: object, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise QuestionValidationError(f"{label} must be an object")
    return cast(dict[str, JsonValue], value)


def _keys(
    value: Mapping[str, JsonValue],
    allowed: frozenset[str],
    required: frozenset[str],
    label: str,
) -> None:
    present = frozenset(value)
    missing = sorted(required - present)
    if missing:
        raise QuestionValidationError(f"{label} is missing required field: {missing[0]}")
    unexpected = sorted(present - allowed)
    if unexpected:
        raise QuestionValidationError(f"{label} has unexpected field: {unexpected[0]}")


def _text(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int,
) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise QuestionValidationError(
            f"{label} must be text between {minimum} and {maximum} characters"
        )
    return value


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise QuestionValidationError(f"{label} must be a boolean")
    return value


def _optional_bool(
    value: Mapping[str, JsonValue],
    key: str,
    default: bool,
    label: str,
) -> bool:
    return default if key not in value else _bool(value[key], f"{label}.{key}")


def _optional_int(
    value: Mapping[str, JsonValue],
    key: str,
    default: int,
    label: str,
) -> int:
    raw = value.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise QuestionValidationError(f"{label}.{key} must be an integer")
    if isinstance(raw, float):
        if not math.isfinite(raw) or not raw.is_integer():
            raise QuestionValidationError(f"{label}.{key} must be an integer")
        return int(raw)
    return raw


def _number(value: object, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise QuestionValidationError(f"{label} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise QuestionValidationError(f"{label} must be a finite number")
    try:
        json.dumps(value, allow_nan=False)
    except (OverflowError, ValueError) as exc:
        raise QuestionValidationError(f"{label} must be JSON-serializable") from exc
    return value


def _optional_number(
    value: Mapping[str, JsonValue],
    key: str,
    label: str,
) -> int | float | None:
    return None if key not in value else _number(value[key], f"{label}.{key}")


def _iso_date(value: object, label: str) -> date:
    if not isinstance(value, str) or re.fullmatch(_DATE_PATTERN, value) is None:
        raise QuestionValidationError(f"{label} must be an ISO date in YYYY-MM-DD form")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise QuestionValidationError(f"{label} must be an ISO date in YYYY-MM-DD form") from exc
    if parsed.isoformat() != value:
        raise QuestionValidationError(f"{label} must be an ISO date in YYYY-MM-DD form")
    return parsed


def _optional_date(
    value: Mapping[str, JsonValue],
    key: str,
    label: str,
) -> date | None:
    return None if key not in value else _iso_date(value[key], f"{label}.{key}")


def _text_array(value: object, label: str, *, maximum: int) -> list[str]:
    if not isinstance(value, list):
        raise QuestionValidationError(f"{label} must be an array")
    values = cast(list[object], value)
    return [
        _text(item, f"{label}[{index}]", minimum=1, maximum=maximum)
        for index, item in enumerate(values)
    ]


def _unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise QuestionValidationError(f"{label} contains duplicate values")
