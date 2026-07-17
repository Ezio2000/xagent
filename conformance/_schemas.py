"""Offline Draft 2020-12 schema loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.protocols import Validator
from referencing import Registry
from referencing.jsonschema import DRAFT202012, Schema, SchemaRegistry, SchemaResource

from conformance._values import load_object

_SCHEMA_BASE = "https://jharness.invalid/spec/v0"
_REQUIRED = {
    "approval.schema.json",
    "checkpoint.schema.json",
    "events.schema.json",
    "limits.schema.json",
    "messages.schema.json",
    "model-error.schema.json",
    "model-request.schema.json",
    "model-response.schema.json",
    "run-context.schema.json",
    "run-request.schema.json",
    "run-snapshot.schema.json",
    "run-trace.schema.json",
    "state.schema.json",
    "tool-result.schema.json",
    "tools.schema.json",
}


class SchemaValidationError(ValueError):
    pass


class SchemaSuite:
    """All portable schemas in one retrieval-disabled registry."""

    def __init__(self, spec_dir: Path, case_schema_path: Path) -> None:
        if not spec_dir.is_dir():
            raise FileNotFoundError(f"spec directory not found: {spec_dir}")
        schemas: dict[str, Schema] = {}
        resources: list[tuple[str, SchemaResource]] = []
        for path in sorted(spec_dir.glob("*.schema.json")):
            schema = _schema(path)
            schema_id = _schema_id(schema, path)
            expected = f"{_SCHEMA_BASE}/{path.name}"
            if schema_id != expected:
                raise ValueError(f"{path}: expected $id {expected!r}, got {schema_id!r}")
            schemas[path.name] = schema
            resources.append((schema_id, DRAFT202012.create_resource(schema)))
        missing = _REQUIRED - set(schemas)
        if missing:
            raise ValueError(f"missing contract schemas: {', '.join(sorted(missing))}")
        registry: SchemaRegistry = Registry[Schema]().with_resources(resources).crawl()
        self._validators: dict[str, Validator] = {
            name: Draft202012Validator(schema, registry=registry)
            for name, schema in schemas.items()
        }
        self._registry = registry
        self._case_validator: Validator = Draft202012Validator(
            _schema(case_schema_path),
            registry=registry,
        )

    def validate_case(self, value: object) -> None:
        _validate(self._case_validator, value, "conformance case")

    def validate(self, schema_name: str, value: object) -> None:
        try:
            validator = self._validators[schema_name]
        except KeyError as exc:
            raise KeyError(f"unknown contract schema: {schema_name}") from exc
        _validate(validator, value, schema_name)

    def validate_ref(self, reference: str, value: object) -> None:
        validator: Validator = Draft202012Validator(
            cast(Schema, {"$ref": reference}),
            registry=self._registry,
        )
        _validate(validator, value, reference)

    def validate_document(self, schema_path: Path, value: object) -> None:
        validator: Validator = Draft202012Validator(
            _schema(schema_path),
            registry=self._registry,
        )
        _validate(validator, value, schema_path.name)


def _schema(path: Path) -> Schema:
    value = load_object(path, "JSON Schema")
    try:
        Draft202012Validator.check_schema(value)
    except SchemaError as exc:
        raise ValueError(f"{path}: invalid JSON Schema: {exc.message}") from exc
    return cast(Schema, value)


def _schema_id(schema: Schema, path: Path) -> str:
    if isinstance(schema, bool):
        raise ValueError(f"{path}: schema must be an object")
    schema_id = schema.get("$id")
    if not isinstance(schema_id, str) or not schema_id:
        raise ValueError(f"{path}: schema must define a non-empty $id")
    return schema_id


def _validate(validator: Validator, value: object, label: str) -> None:
    try:
        validator.validate(cast(Any, value))
    except ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path) or "$"
        raise SchemaValidationError(f"{label} violation at {path}: {exc.message}") from exc
