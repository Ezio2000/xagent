from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import cast

import yaml

ROOT = Path(__file__).resolve().parents[2]
_PINNED_ACTION = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _mapping(value: object, label: str) -> dict[str, object]:
    assert isinstance(value, dict), f"{label} must be a mapping"
    mapping = cast(dict[object, object], value)
    assert all(isinstance(key, str) for key in mapping), f"{label} keys must be strings"
    return cast(dict[str, object], mapping)


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    steps = job.get("steps")
    assert isinstance(steps, list)
    return [_mapping(item, "CI step") for item in cast(list[object], steps)]


def _named_step(job: dict[str, object], name: str) -> dict[str, object]:
    matching = [step for step in _steps(job) if step.get("name") == name]
    assert len(matching) == 1
    return matching[0]


def run_example(name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "examples" / name)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_offline_examples_execute_with_uv_selected_python() -> None:
    basic = run_example("basic_tool_loop.py")
    resume = run_example("pause_resume_trace.py")

    assert basic.returncode == 0, basic.stderr
    assert "Tool said: hello" in basic.stdout
    assert "status=completed" in basic.stdout
    assert resume.returncode == 0, resume.stderr
    assert "paused=suspended" in resume.stdout
    assert "resumed=completed" in resume.stdout


def test_every_local_markdown_link_resolves() -> None:
    broken: list[str] = []
    markdown_files = [
        ROOT / name
        for name in (
            "README.md",
            "AGENTS.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
        )
    ]
    for directory in ("contracts", "conformance", "docs", "tests"):
        markdown_files.extend((ROOT / directory).rglob("*.md"))
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in markdown_files:
        for target in pattern.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            relative = target.split("#", 1)[0]
            if relative and not (path.parent / relative).resolve().exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    assert broken == []


def test_ci_references_existing_sources_and_required_commands() -> None:
    workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
    workflow = _mapping(
        cast(object, yaml.load(workflow_path.read_text(), Loader=yaml.BaseLoader)),
        "CI workflow",
    )
    raw_jobs = _mapping(workflow.get("jobs"), "CI jobs")
    jobs = {name: _mapping(job, f"CI job {name}") for name, job in raw_jobs.items()}
    assert set(jobs) == {"quality", "runtime"}

    quality = jobs["quality"]
    services = _mapping(quality.get("services"), "quality services")
    assert set(services) == {"mysql", "redis"}
    quality_runs: dict[str, str] = {}
    for step in _steps(quality):
        uses = step.get("uses")
        if uses is not None:
            assert isinstance(uses, str)
            assert _PINNED_ACTION.fullmatch(uses)
        run = step.get("run")
        if run is not None:
            assert isinstance(run, str)
            name = step.get("name")
            assert isinstance(name, str)
            quality_runs[name] = run

    assert "uv run pytest" in quality_runs["Test with branch coverage"]
    test_environment = _mapping(
        _named_step(quality, "Test with branch coverage").get("env"),
        "quality test environment",
    )
    assert "JHARNESS_TEST_MYSQL_URL" in test_environment
    assert "JHARNESS_TEST_REDIS_URL" in test_environment
    assert quality_runs["Run runtime smoke benchmark"] == (
        "uv run python benchmarks/runtime_smoke.py"
    )
    assert quality_runs["Validate contracts, cases, and documentation"] == (
        "uv run python scripts/validate_spec.py"
    )
    distribution_checks = quality_runs["Verify distribution set and isolated imports"]
    assert "uv run python scripts/verify_distribution.py" in distribution_checks
    assert "find_spec('pymysql') is None" in distribution_checks
    assert "find_spec('redis') is None" in distribution_checks
    assert '"${repository_wheels[0]}[mysql,redis]"' in distribution_checks
    assert all("packages/" not in run for run in quality_runs.values())

    runtime = jobs["runtime"]
    strategy = _mapping(runtime.get("strategy"), "runtime strategy")
    matrix = _mapping(strategy.get("matrix"), "runtime matrix")
    assert matrix["python-version"] == ["3.11", "3.12", "3.13", "3.14"]
    assert matrix["os"] == ["ubuntu-latest", "windows-latest"]
    runtime_steps = _steps(runtime)
    for step in runtime_steps:
        uses = step.get("uses")
        if uses is not None:
            assert isinstance(uses, str)
            assert _PINNED_ACTION.fullmatch(uses)
    assert _named_step(runtime, "Test complete runtime")["run"] == (
        "uv run pytest -q -p no:cacheprovider"
    )

    workflow_commands = "\n".join(quality_runs.values())
    referenced = set(
        re.findall(
            r"(?:benchmarks|conformance|examples|scripts|tests)/[A-Za-z0-9_./-]+\.py",
            workflow_commands,
        )
    )
    missing = sorted(path for path in referenced if not (ROOT / path).is_file())
    assert missing == []
