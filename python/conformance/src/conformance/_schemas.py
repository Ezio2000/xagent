"""Schema loading and validator construction for conformance cases."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

REQUIRED_SCHEMA_FILES = {
    "events.schema.json",
    "model-request.schema.json",
    "run-snapshot.schema.json",
    "run-trace.schema.json",
    "runtime-context.schema.json",
    "runtime-extensions.schema.json",
    "resume-input.schema.json",
    "messages.schema.json",
    "model-response.schema.json",
    "model-error.schema.json",
    "tools.schema.json",
    "tool-result.schema.json",
    "state.schema.json",
    "limits.schema.json",
}
REGISTRY_CLS: Any = Registry
RESOURCE_CLS: Any = Resource
DRAFT_2020_12_SPEC: Any = DRAFT202012


@dataclass(slots=True)
class ConformanceValidators:
    event: Any
    run_snapshot: Any
    run_trace: Any
    runtime_context: Any
    resume_input: Any
    message: Any
    model_error: Any
    model_request: Any
    model_response: Any
    tool_result: Any
    state: Any
    limits: Any
    approval_request: Any
    approval_decision: Any
    checkpoint_summary: Any
    stored_checkpoint: Any
    journal_record: Any


def load_json_schema(path: Path) -> dict[str, Any]:
    schema = load_json_object(path, "schema")
    schema_id = schema.get("$id")
    if not isinstance(schema_id, str) or not schema_id:
        raise ValueError(f"{path}: schema must define a non-empty $id")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"{path}: invalid JSON schema: {exc.message}") from exc
    return schema


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_text()
    except OSError as exc:
        raise OSError(f"failed to read {label} {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}: invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise TypeError(f"{path}: {label} must contain an object")
    return cast(dict[str, Any], value)


def load_contract_schemas(spec_dir: Path) -> dict[str, dict[str, Any]]:
    if not spec_dir.is_dir():
        raise FileNotFoundError(f"spec directory not found: {spec_dir}")
    schemas = {path.name: load_json_schema(path) for path in sorted(spec_dir.glob("*.schema.json"))}
    missing = REQUIRED_SCHEMA_FILES - set(schemas)
    if missing:
        raise ValueError(f"spec directory missing schema file(s): {', '.join(sorted(missing))}")
    return schemas


def build_schema_registry(schemas: Mapping[str, Mapping[str, Any]]) -> Any:
    registry: Any = REGISTRY_CLS().with_resources(
        [
            (
                cast(str, schema["$id"]),
                RESOURCE_CLS.from_contents(
                    cast(Any, schema), default_specification=DRAFT_2020_12_SPEC
                ),
            )
            for schema in schemas.values()
        ]
    )
    return registry


def build_case_validator(spec_dir: Path, case_schema_path: Path) -> Draft202012Validator:
    schemas = load_contract_schemas(spec_dir)
    registry = build_schema_registry(schemas)
    return Draft202012Validator(load_json_schema(case_schema_path), registry=registry)


def build_validators(spec_dir: Path) -> ConformanceValidators:
    schemas = load_contract_schemas(spec_dir)
    registry = build_schema_registry(schemas)
    runtime_extensions_ref = "https://agent-runtime.local/spec/v0/runtime-extensions.schema.json"

    def runtime_extension_validator(def_name: str) -> Draft202012Validator:
        return Draft202012Validator(
            {"$ref": f"{runtime_extensions_ref}#/$defs/{def_name}"},
            registry=registry,
        )

    return ConformanceValidators(
        event=Draft202012Validator(schemas["events.schema.json"], registry=registry),
        run_snapshot=Draft202012Validator(schemas["run-snapshot.schema.json"], registry=registry),
        run_trace=Draft202012Validator(schemas["run-trace.schema.json"], registry=registry),
        runtime_context=Draft202012Validator(
            schemas["runtime-context.schema.json"], registry=registry
        ),
        resume_input=Draft202012Validator(schemas["resume-input.schema.json"], registry=registry),
        message=Draft202012Validator(schemas["messages.schema.json"], registry=registry),
        model_error=Draft202012Validator(schemas["model-error.schema.json"], registry=registry),
        model_request=Draft202012Validator(schemas["model-request.schema.json"], registry=registry),
        model_response=Draft202012Validator(
            schemas["model-response.schema.json"], registry=registry
        ),
        tool_result=Draft202012Validator(schemas["tool-result.schema.json"], registry=registry),
        state=Draft202012Validator(schemas["state.schema.json"], registry=registry),
        limits=Draft202012Validator(schemas["limits.schema.json"], registry=registry),
        approval_request=runtime_extension_validator("approval_request"),
        approval_decision=runtime_extension_validator("approval_decision"),
        checkpoint_summary=runtime_extension_validator("checkpoint_summary"),
        stored_checkpoint=runtime_extension_validator("stored_checkpoint"),
        journal_record=runtime_extension_validator("journal_record"),
    )


def assert_validator_matches(label: str, validator: Any, instance: Mapping[str, Any]) -> None:
    errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if not errors:
        return
    error = errors[0]
    path = ".".join(str(part) for part in error.absolute_path) or "$"
    raise AssertionError(f"{label} schema violation at {path}: {error.message}") from error
