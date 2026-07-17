from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "uv run pytest" in workflow
    assert "benchmarks/runtime_smoke.py" in workflow
    assert "scripts/verify_distribution.py" in workflow
    assert "scripts/validate_spec.py" in workflow
    assert "packages/" not in workflow

    referenced = set(
        re.findall(
            r"(?:benchmarks|conformance|examples|scripts|tests)/[A-Za-z0-9_./-]+\.py",
            workflow,
        )
    )
    missing = sorted(path for path in referenced if not (ROOT / path).is_file())
    assert missing == []
