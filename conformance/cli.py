"""Command-line entry point for the Python conformance runner."""

from __future__ import annotations

import argparse
import asyncio
import traceback
from collections.abc import Sequence
from pathlib import Path

from conformance.runner import ConformanceRunner


def infer_spec_dir(cases_dir: Path) -> Path:
    if cases_dir.name == "cases" and cases_dir.parent.name == "conformance":
        return cases_dir.parent.parent / "contracts" / "v0"
    return Path.cwd() / "contracts" / "v0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run JHarness v0 conformance cases.")
    parser.add_argument("cases_dir", type=Path)
    parser.add_argument("--spec-dir", type=Path)
    parser.add_argument("--case-schema", type=Path)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--traceback", action="store_true")
    return parser


async def _run(runner: ConformanceRunner, *, quiet: bool, show_tracebacks: bool) -> int:
    passed = 0
    failed = 0
    for case in runner.load_cases():
        name = str(case["name"])
        try:
            result = await runner.run_case(case)
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
            if show_tracebacks:
                traceback.print_exception(exc)
        else:
            passed += 1
            if not quiet:
                print(f"PASS {result.name} [{result.case_type}]")
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases_dir = args.cases_dir.resolve()
    spec_dir = args.spec_dir.resolve() if args.spec_dir is not None else infer_spec_dir(cases_dir)
    try:
        runner = ConformanceRunner(
            cases_dir=cases_dir,
            spec_dir=spec_dir,
            case_schema_path=(None if args.case_schema is None else args.case_schema.resolve()),
        )
        return asyncio.run(
            _run(
                runner,
                quiet=bool(args.quiet),
                show_tracebacks=bool(args.traceback),
            )
        )
    except Exception as exc:
        print(f"FAIL load: {exc}")
        if bool(args.traceback):
            traceback.print_exception(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
