from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin

from jsonschema import Draft202012Validator

from conformance import ConformanceRunner

ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = ROOT / "contracts" / "v0"
CONFORMANCE_DIR = ROOT / "conformance"


def json_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text())
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def references(value: object) -> Iterator[str]:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        for keyword in ("$ref", "$dynamicRef"):
            ref = mapping.get(keyword)
            if isinstance(ref, str):
                yield ref
        for item in mapping.values():
            yield from references(item)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in cast(Sequence[object], value):
            yield from references(item)


def test_all_contract_schemas_are_valid_unique_and_offline_resolvable() -> None:
    paths = sorted(SPEC_DIR.glob("*.schema.json"))
    paths.append(CONFORMANCE_DIR / "case.schema.json")
    paths.append(CONFORMANCE_DIR / "tools.contract.schema.json")
    schemas = [json_object(path) for path in paths]
    ids = {str(schema["$id"]) for schema in schemas}

    assert len(paths) == 17
    assert len(ids) == len(paths)
    for schema in schemas:
        Draft202012Validator.check_schema(schema)
        for ref in references(schema):
            base = ref.split("#", 1)[0]
            if base:
                resolved = urljoin(str(schema["$id"]), base)
                assert resolved in ids or resolved == "https://json-schema.org/draft/2020-12/schema"


def test_tool_manifest_and_all_cases_validate() -> None:
    manifest = json_object(CONFORMANCE_DIR / "tools.contract.json")
    assert manifest["schema_version"] == "v0"

    runner = ConformanceRunner(
        cases_dir=CONFORMANCE_DIR / "cases",
        spec_dir=SPEC_DIR,
    )
    runner.schemas.validate_document(
        CONFORMANCE_DIR / "tools.contract.schema.json",
        manifest,
    )
    assert len(runner.load_cases()) == 72


def test_conformance_cases_use_the_flat_portable_layout() -> None:
    cases_dir = CONFORMANCE_DIR / "cases"
    direct_cases = sorted(cases_dir.glob("*.json"))
    assert sorted(cases_dir.rglob("*.json")) == direct_cases


def test_contract_ids_use_one_canonical_v0_namespace() -> None:
    for path in SPEC_DIR.glob("*.schema.json"):
        schema_id = str(json_object(path)["$id"])
        assert schema_id == f"https://jharness.invalid/spec/v0/{path.name}"
