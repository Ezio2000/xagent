"""Install the coordinated JHarness release from TestPyPI and run smoke examples."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import time
from pathlib import Path

_VERSION = re.compile(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?")
_PUBLIC_IMPORTS = (
    "from jharness.kernel import Runtime; "
    "from jharness.models.openai import OpenAIChatCompletionsModel; "
    "from jharness.toolkit import ToolRegistry; "
    "from jharness.tools import ReadTool; "
    "assert Runtime and OpenAIChatCompletionsModel and ToolRegistry and ReadTool"
)
_PROJECT = """\
[project]
name = "jharness-testpypi-smoke"
version = "0"
requires-python = ">=3.11"
dependencies = [
  "jharness-kernel=={version}",
  "jharness-toolkit=={version}",
  "jharness-models=={version}",
  "jharness-tools=={version}",
]

[tool.uv.sources]
jharness-kernel = {{ index = "testpypi" }}
jharness-toolkit = {{ index = "testpypi" }}
jharness-models = {{ index = "testpypi" }}
jharness-tools = {{ index = "testpypi" }}

[[tool.uv.index]]
name = "testpypi"
url = "https://test.pypi.org/simple"
explicit = true
"""


def project_document(version: str) -> str:
    """Return an isolated uv project that pins JHarness to TestPyPI."""

    if _VERSION.fullmatch(version) is None:
        raise ValueError(f"invalid release version: {version!r}")
    return _PROJECT.format(version=version)


def _run_example(project: Path, example: Path) -> None:
    subprocess.run(
        ["uv", "run", "--project", str(project), "--refresh", "python", str(example)],
        check=True,
    )


def _verify_public_imports(project: Path) -> None:
    subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(project),
            "--refresh",
            "python",
            "-I",
            "-c",
            _PUBLIC_IMPORTS,
        ],
        check=True,
    )


def main() -> int:
    """Retry index propagation, then execute offline public smoke examples."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--delay", type=float, default=15)
    parser.add_argument("--examples", type=Path, default=Path("examples"))
    args = parser.parse_args()
    if args.attempts < 1 or args.delay < 0:
        parser.error("attempts must be positive and delay must be non-negative")
    try:
        document = project_document(args.version)
    except ValueError as exc:
        parser.error(str(exc))
    with tempfile.TemporaryDirectory(prefix="jharness-testpypi-") as temporary:
        project = Path(temporary)
        (project / "pyproject.toml").write_text(document, encoding="utf-8")
        for attempt in range(1, args.attempts + 1):
            try:
                _verify_public_imports(project)
                _run_example(project, args.examples / "basic_tool_loop.py")
            except subprocess.CalledProcessError:
                if attempt == args.attempts:
                    return 1
                time.sleep(args.delay)
                continue
            _run_example(project, args.examples / "pause_resume_trace.py")
            return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
