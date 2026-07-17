from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DISTRIBUTIONS = (
    "jharness-kernel",
    "jharness-toolkit",
    "jharness-models",
    "jharness-tools",
)
MODULES = ("jharness.kernel", "jharness.toolkit", "jharness.models", "jharness.tools")


def test_release_metadata_is_coordinated() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_release.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "distribution set" not in result.stdout
    assert "distributions=jharness-kernel" in result.stdout


def test_release_workflow_builds_and_publishes_four_distributions() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "verify_release.py --tag" in workflow
    assert "uv build --all-packages" in workflow
    assert "scripts/verify_distribution.py dist" in workflow
    assert workflow.count("pypa/gh-action-pypi-publish@") == 2
    assert "testpypi-jharness" in workflow
    assert "pypi-jharness" in workflow
    assert "sha256sum --check dist/SHA256SUMS" in workflow
    assert 'test "$(find publish -maxdepth 1 -type f | wc -l)" -eq 8' in workflow
    assert "source_run_id:" in workflow
    assert "--clobber" in workflow
    for distribution in DISTRIBUTIONS:
        assert distribution in workflow


def test_repository_ownership_and_dependency_updates_are_explicit() -> None:
    owners = (ROOT / ".github" / "CODEOWNERS").read_text()
    assert "* @Ezio2000" in owners
    for protected_path in (
        "/.github/workflows/**",
        "/packages/**/pyproject.toml",
        "/pyproject.toml",
        "/uv.lock",
        "/scripts/**",
    ):
        assert protected_path in owners

    dependabot = (ROOT / ".github" / "dependabot.yml").read_text()
    assert "package-ecosystem: uv" in dependabot
    assert "directory: /" in dependabot
    assert "package-ecosystem: github-actions" in dependabot


def test_readme_documents_distributions_and_public_modules() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    for distribution in DISTRIBUTIONS:
        assert f"uv add {distribution}" in readme
    for module in MODULES:
        assert module in readme


def test_testpypi_smoke_project_pins_all_distributions() -> None:
    script = (ROOT / "scripts" / "verify_testpypi.py").read_text()
    for distribution in DISTRIBUTIONS:
        assert f'"{distribution}=={{version}}"' in script
        assert f'{distribution} = {{{{ index = "testpypi" }}}}' in script
    for module in MODULES:
        assert module in script
    assert script.count('{{ index = "testpypi" }}') == 4


def test_testpypi_smoke_project_rejects_invalid_versions() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_testpypi.py"), "0.2.0; unsafe"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid release version" in result.stderr
