"""Validate the complete JHarness specification without network retrieval."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin

from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator
from referencing import Registry
from referencing.jsonschema import DRAFT202012, Schema, SchemaRegistry, SchemaResource

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts" / "v0"
CONFORMANCE = ROOT / "conformance"
SCHEMA_BASE = "https://jharness.invalid/spec/v0"


def _object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return cast(dict[str, Any], value)


def _references(value: object) -> Iterator[str]:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        reference = mapping.get("$ref")
        if isinstance(reference, str):
            yield reference
        for item in mapping.values():
            yield from _references(item)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in cast(Sequence[object], value):
            yield from _references(item)


def _schemas() -> tuple[dict[str, Schema], SchemaRegistry]:
    paths = sorted(CONTRACTS.glob("*.schema.json"))
    paths.extend((CONFORMANCE / "case.schema.json", CONFORMANCE / "tools.contract.schema.json"))
    schemas: dict[str, Schema] = {}
    resources: list[tuple[str, SchemaResource]] = []
    for path in paths:
        schema_object = _object(path)
        schema = cast(Schema, schema_object)
        Draft202012Validator.check_schema(schema)
        schema_id = cast(str, schema_object["$id"])
        if schema_id in schemas:
            raise ValueError(f"duplicate schema id: {schema_id}")
        if path.parent == CONTRACTS and schema_id != f"{SCHEMA_BASE}/{path.name}":
            raise ValueError(f"{path}: unexpected schema id {schema_id!r}")
        schemas[schema_id] = schema
        resources.append((schema_id, DRAFT202012.create_resource(schema)))
    registry: SchemaRegistry = Registry[Schema]().with_resources(resources).crawl()
    known = set(schemas)
    for schema_id, schema in schemas.items():
        for reference in _references(schema):
            base = reference.split("#", 1)[0]
            if not base:
                continue
            resolved = urljoin(schema_id, base)
            if resolved not in known and resolved != "https://json-schema.org/draft/2020-12/schema":
                raise ValueError(f"{schema_id}: unresolved offline reference {resolved}")
    return schemas, registry


def _validator(
    schemas: Mapping[str, Schema],
    registry: SchemaRegistry,
    schema_id: str,
) -> Validator:
    return Draft202012Validator(schemas[schema_id], registry=registry)


def _validate_documents(schemas: Mapping[str, Schema], registry: SchemaRegistry) -> int:
    case_validator = _validator(
        schemas,
        registry,
        "https://jharness.invalid/conformance/case.schema.json",
    )
    tool_validator = _validator(
        schemas,
        registry,
        "https://jharness.invalid/conformance/tools.contract.schema.json",
    )
    tool_validator.validate(_object(CONFORMANCE / "tools.contract.json"))

    names: set[str] = set()
    case_paths = sorted((CONFORMANCE / "cases").glob("*.json"))
    if sorted((CONFORMANCE / "cases").rglob("*.json")) != case_paths:
        raise ValueError("conformance/cases must remain flat")
    for path in case_paths:
        case = _object(path)
        case_validator.validate(case)
        name = cast(str, case["name"])
        if name != path.stem:
            raise ValueError(f"{path}: case name must match the filename")
        if name in names:
            raise ValueError(f"duplicate case name: {name}")
        names.add(name)

    coverage = (CONFORMANCE / "coverage.md").read_text()
    missing = sorted(name for name in names if f"`{name}`" not in coverage)
    if missing:
        raise ValueError(f"cases missing from conformance/coverage.md: {missing}")
    return len(case_paths)


def _validate_links() -> int:
    excluded = {".git", ".venv"}
    markdown_paths = sorted(
        path for path in ROOT.rglob("*.md") if not excluded.intersection(path.parts)
    )
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    broken: list[str] = []
    for path in markdown_paths:
        for target in pattern.findall(path.read_text()):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = target.split("#", 1)[0]
            if relative and not (path.parent / relative).resolve().exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    if broken:
        raise ValueError(f"broken local Markdown links: {broken}")
    return len(markdown_paths)


def main() -> None:
    """Validate schemas, cases, coverage, and local documentation links."""

    schemas, registry = _schemas()
    case_count = _validate_documents(schemas, registry)
    markdown_count = _validate_links()
    print(f"specification ok: schemas={len(schemas)} cases={case_count} markdown={markdown_count}")


if __name__ == "__main__":
    main()
