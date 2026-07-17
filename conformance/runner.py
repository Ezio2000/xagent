"""Portable case loader and Python reference implementation runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conformance._execution import run_invocation
from conformance._expectations import assert_invocation
from conformance._schemas import SchemaSuite
from conformance._tools import load_standard_tools
from conformance._validation import run_validation_case
from conformance._values import load_object, mapping, sequence, string
from jharness.kernel import Checkpoint
from jharness.kernel.wire import decode_checkpoint


@dataclass(frozen=True, slots=True)
class ConformanceCaseResult:
    name: str
    case_type: str
    invocation_count: int = 0


class ConformanceRunner:
    """Run sorted v0 fixtures with an offline schema registry."""

    def __init__(
        self,
        *,
        cases_dir: Path,
        spec_dir: Path,
        case_schema_path: Path | None = None,
    ) -> None:
        self.cases_dir = cases_dir
        conformance_dir = cases_dir.parent
        self.case_schema_path = case_schema_path or conformance_dir / "case.schema.json"
        self.schemas = SchemaSuite(spec_dir, self.case_schema_path)
        self.tools = load_standard_tools(
            conformance_dir / "tools.contract.json",
            conformance_dir / "tools.contract.schema.json",
            self.schemas,
        )

    def load_cases(self) -> tuple[dict[str, Any], ...]:
        if not self.cases_dir.is_dir():
            raise FileNotFoundError(f"cases directory not found: {self.cases_dir}")
        cases: list[dict[str, Any]] = []
        names: set[str] = set()
        for path in sorted(self.cases_dir.glob("*.json")):
            case = load_object(path, "conformance case")
            self.schemas.validate_case(case)
            name = string(case["name"], "case name")
            if name in names:
                raise ValueError(f"duplicate conformance case name: {name}")
            names.add(name)
            cases.append(case)
        if not cases:
            raise ValueError(f"no conformance cases found in {self.cases_dir}")
        return tuple(cases)

    async def run_case(self, case: Mapping[str, Any]) -> ConformanceCaseResult:
        name = string(case["name"], "case name")
        kind = string(case["kind"], "case kind")
        if kind == "validation":
            run_validation_case(case, self.schemas)
            return ConformanceCaseResult(name, kind)
        if kind != "scenario":
            raise ValueError(f"unsupported conformance case kind: {kind}")

        raw_seed = case.get("seed_checkpoint")
        seed = None if raw_seed is None else decode_checkpoint(raw_seed)
        previous: Checkpoint | None = None
        invocations = sequence(case["invocations"], "case invocations")
        for index, raw_invocation in enumerate(invocations, start=1):
            invocation = mapping(raw_invocation, f"invocation {index}")
            try:
                outcome = await run_invocation(
                    invocation,
                    seed=seed,
                    previous=previous,
                    tools=self.tools,
                    schemas=self.schemas,
                )
                assert_invocation(
                    outcome,
                    mapping(invocation["expected"], "invocation expected"),
                )
            except Exception as exc:
                raise AssertionError(f"{name} invocation {index}: {exc}") from exc
            previous = outcome.checkpoint
        return ConformanceCaseResult(name, kind, len(invocations))
