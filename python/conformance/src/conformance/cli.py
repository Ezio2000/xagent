"""Command line entry point for the Python conformance runner."""

from __future__ import annotations

import argparse
import asyncio
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from conformance._case import expect_case_str
from conformance.runner import ConformanceRunner


def infer_spec_dir(cases_dir: Path) -> Path:
    if cases_dir.name == "cases" and cases_dir.parent.name == "conformance":
        return cases_dir.parent.parent / "contracts" / "v0"
    return Path.cwd() / "contracts" / "v0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run runtime conformance cases.")
    parser.add_argument("cases_dir", type=Path, help="Directory containing conformance case JSON")
    parser.add_argument(
        "--spec-dir",
        type=Path,
        help="Directory containing contracts/v0 schema JSON. Defaults to the repository layout.",
    )
    parser.add_argument(
        "--case-schema",
        type=Path,
        help="Path to conformance/case.schema.json. Defaults to the repository layout.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print failures and the final summary.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Print Python tracebacks for failing cases.",
    )
    return parser


async def _run_cli_cases(runner: ConformanceRunner, *, quiet: bool, show_tracebacks: bool) -> int:
    cases = runner.load_cases()
    passed = 0
    failed = 0
    for case in cases:
        name = expect_case_str(case["name"], "case name")
        try:
            result = await runner.run_case(case)
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
            if show_tracebacks:
                traceback.print_exception(exc)
            continue
        passed += 1
        if not quiet:
            print(f"PASS {result.name} [{result.case_type}]")
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cases_dir = args.cases_dir.resolve()
    spec_dir = args.spec_dir.resolve() if args.spec_dir is not None else infer_spec_dir(cases_dir)
    case_schema_path = args.case_schema.resolve() if args.case_schema is not None else None
    try:
        runner = ConformanceRunner(
            cases_dir=cases_dir,
            spec_dir=spec_dir,
            case_schema_path=case_schema_path,
        )
        return asyncio.run(
            _run_cli_cases(
                runner,
                quiet=cast(bool, args.quiet),
                show_tracebacks=cast(bool, args.traceback),
            )
        )
    except Exception as exc:
        print(f"FAIL load: {exc}")
        if cast(bool, args.traceback):
            traceback.print_exception(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
