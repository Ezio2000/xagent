from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from email.message import Message
from pathlib import Path
from typing import cast

import pytest
import yaml

from scripts import verify_distribution, verify_testpypi

ROOT = Path(__file__).resolve().parents[2]
DISTRIBUTIONS = (
    "jharness-kernel",
    "jharness-toolkit",
    "jharness-models",
    "jharness-repository",
    "jharness-tools",
)
MODULES = (
    "jharness.kernel",
    "jharness.toolkit",
    "jharness.models",
    "jharness.repository",
    "jharness.tools",
)
_PINNED_ACTION = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
_MYSQL_IMAGE = (
    "mysql:8.4.10@sha256:c592c15aaf4a1961e15d82eb31ea5987dda862d1c4b1e93424438c0e91dc1f8d"
)
_REDIS_IMAGE = "redis:8.4.4@sha256:c44528447fa07ed62bdb0c1944cba54f8cad6a4e4a49ada9d4843b5b07d03227"


def _mapping(value: object, label: str) -> dict[str, object]:
    assert isinstance(value, dict), f"{label} must be a mapping"
    mapping = cast(dict[object, object], value)
    assert all(isinstance(key, str) for key in mapping), f"{label} keys must be strings"
    return cast(dict[str, object], mapping)


def _workflow(name: str) -> dict[str, object]:
    loaded = cast(
        object,
        yaml.load(
            (ROOT / ".github" / "workflows" / name).read_text(),
            Loader=yaml.BaseLoader,
        ),
    )
    return _mapping(loaded, name)


def _jobs(workflow: dict[str, object]) -> dict[str, dict[str, object]]:
    jobs = _mapping(workflow.get("jobs"), "jobs")
    return {name: _mapping(job, f"job {name}") for name, job in jobs.items()}


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    raw_steps = job.get("steps")
    assert isinstance(raw_steps, list)
    return [_mapping(step, "workflow step") for step in cast(list[object], raw_steps)]


def _step(job: dict[str, object], name: str) -> dict[str, object]:
    matching = [step for step in _steps(job) if step.get("name") == name]
    assert len(matching) == 1, f"expected exactly one step named {name!r}"
    return matching[0]


def _run(job: dict[str, object], name: str) -> str:
    run = _step(job, name).get("run")
    assert isinstance(run, str)
    return run


def _assert_actions_are_commit_pinned(jobs: dict[str, dict[str, object]]) -> None:
    for job_name, job in jobs.items():
        for step in _steps(job):
            uses = step.get("uses")
            if uses is not None:
                assert isinstance(uses, str)
                assert _PINNED_ACTION.fullmatch(uses), f"unpinned action in {job_name}: {uses}"


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


def test_repository_integration_images_are_immutable_and_consistent() -> None:
    quality = _jobs(_workflow("ci.yml"))["quality"]
    services = _mapping(quality.get("services"), "CI services")
    ci_images = {
        str(_mapping(services.get(name), f"{name} service").get("image"))
        for name in ("mysql", "redis")
    }
    expected_images = {_MYSQL_IMAGE, _REDIS_IMAGE}
    assert ci_images == expected_images

    release_build = _jobs(_workflow("release.yml"))["build"]
    integration_run = _run(release_build, "Start repository integration services")
    release_images = set(re.findall(r"(?:mysql|redis):[^\s\\]+", integration_run))
    assert release_images == expected_images
    assert all(":latest" not in image for image in ci_images | release_images)
    assert all(
        re.fullmatch(r"[^:]+:\d+\.\d+\.\d+@sha256:[0-9a-f]{64}", image) for image in ci_images
    )


@pytest.mark.parametrize(
    "mysql_requirement",
    (
        "pymysql>=1.2.0",
        "pymysql[rsa]",
        "pymysql[rsa]>=1.1.0",
        "pymysql[other]>=1.2.0",
    ),
)
def test_distribution_verifier_rejects_inexact_repository_extras(
    mysql_requirement: str,
) -> None:
    metadata = Message()
    metadata["Requires-Dist"] = "jharness-kernel==0.3.0"
    metadata["Requires-Dist"] = f'{mysql_requirement}; extra == "mysql"'
    metadata["Requires-Dist"] = 'redis>=8.0.1; extra == "redis"'
    metadata["Provides-Extra"] = "mysql"
    metadata["Provides-Extra"] = "redis"

    with pytest.raises(ValueError, match="optional dependencies differ"):
        verify_distribution._verify_dependencies(  # pyright: ignore[reportPrivateUsage]
            metadata, "jharness-repository", "0.3.0"
        )


def test_release_workflow_builds_and_publishes_five_distributions() -> None:
    jobs = _jobs(_workflow("release.yml"))
    assert set(jobs) == {
        "build",
        "publish-testpypi",
        "verify-testpypi",
        "publish-pypi",
        "verify-pypi",
        "github-release",
    }
    _assert_actions_are_commit_pinned(jobs)

    assert jobs["publish-testpypi"]["needs"] == "build"
    assert jobs["verify-testpypi"]["needs"] == "publish-testpypi"
    assert jobs["publish-pypi"]["needs"] == "verify-testpypi"
    assert jobs["verify-pypi"]["needs"] == "publish-pypi"
    assert jobs["github-release"]["needs"] == "verify-pypi"

    build = jobs["build"]
    integration_step = _step(build, "Start repository integration services")
    assert integration_step["if"] == "github.event_name == 'push'"
    integration_run = _run(build, "Start repository integration services")
    assert _MYSQL_IMAGE in integration_run
    assert _REDIS_IMAGE in integration_run
    quality_environment = _mapping(
        _step(build, "Run release quality gate").get("env"),
        "release quality environment",
    )
    assert "JHARNESS_TEST_MYSQL_URL" in quality_environment
    assert "JHARNESS_TEST_REDIS_URL" in quality_environment
    assert 'verify_release.py --tag "$RELEASE_TAG"' in _run(build, "Verify tag and metadata")
    assert _run(build, "Build immutable artifact set") == ("uv build --all-packages --out-dir dist")
    artifact_checks = _run(build, "Verify artifacts, imports, and checksums")
    assert "scripts/verify_distribution.py dist" in artifact_checks
    assert "find_spec('pymysql') is None" in artifact_checks
    assert "find_spec('redis') is None" in artifact_checks
    assert '"${repository_wheels[0]}[mysql,redis]"' in artifact_checks
    assert "sha256sum --check dist/SHA256SUMS" in artifact_checks
    assert "-name '*.whl' -o -name '*.tar.gz'" in artifact_checks
    assert '| wc -l)" -eq 10' in artifact_checks
    assert "test -f dist/SHA256SUMS" in artifact_checks
    recovery_checks = _run(build, "Verify recovered run identity and artifacts")
    assert 'test "$run_path" = ".github/workflows/release.yml"' in recovery_checks
    assert "scripts/verify_distribution.py dist" in recovery_checks
    assert "-name '*.whl' -o -name '*.tar.gz'" in recovery_checks
    assert '| wc -l)" -eq 10' in recovery_checks
    assert "test -f dist/SHA256SUMS" in recovery_checks

    test_publish = _step(jobs["publish-testpypi"], "Publish with trusted publishing")
    pypi_publish = _step(jobs["publish-pypi"], "Publish with trusted publishing")
    expected_action = "pypa/gh-action-pypi-publish@cef221092ed1bacb1cc03d23a2d87d1d172e277b"
    assert test_publish["uses"] == expected_action
    assert pypi_publish["uses"] == expected_action
    test_options = _mapping(test_publish.get("with"), "TestPyPI publish options")
    pypi_options = _mapping(pypi_publish.get("with"), "PyPI publish options")
    assert test_options["repository-url"] == "https://test.pypi.org/legacy/"
    assert test_options["packages-dir"] == "publish/"
    assert pypi_options["packages-dir"] == "publish/"
    assert "repository-url" not in pypi_options
    for job_name, environment_name, environment_url in (
        (
            "publish-testpypi",
            "testpypi-${{ matrix.project }}",
            "https://test.pypi.org/p/${{ matrix.project }}",
        ),
        (
            "publish-pypi",
            "pypi-${{ matrix.project }}",
            "https://pypi.org/p/${{ matrix.project }}",
        ),
    ):
        environment = _mapping(jobs[job_name].get("environment"), f"{job_name} environment")
        permissions = _mapping(jobs[job_name].get("permissions"), f"{job_name} permissions")
        assert environment["name"] == environment_name
        assert environment["url"] == environment_url
        assert permissions == {"id-token": "write"}

        strategy = _mapping(jobs[job_name].get("strategy"), f"{job_name} strategy")
        assert strategy["fail-fast"] == "false"
        matrix = _mapping(strategy.get("matrix"), f"{job_name} matrix")
        include = matrix.get("include")
        assert isinstance(include, list)
        assert include == [
            {"project": "jharness-kernel", "artifact_prefix": "jharness_kernel"},
            {"project": "jharness-models", "artifact_prefix": "jharness_models"},
            {"project": "jharness-repository", "artifact_prefix": "jharness_repository"},
            {"project": "jharness-toolkit", "artifact_prefix": "jharness_toolkit"},
            {"project": "jharness-tools", "artifact_prefix": "jharness_tools"},
        ]

    test_stage = _run(jobs["publish-testpypi"], "Verify checksums and stage distributions")
    pypi_stage = _run(jobs["publish-pypi"], "Verify checksums and stage distributions")
    expected_count = 'test "$(find publish -maxdepth 1 -type f | wc -l)" -eq 2'
    assert expected_count in test_stage
    assert expected_count in pypi_stage
    assert "dist/${{ matrix.artifact_prefix }}-*.whl" in test_stage
    assert "dist/${{ matrix.artifact_prefix }}-*.tar.gz" in test_stage
    assert "dist/${{ matrix.artifact_prefix }}-*.whl" in pypi_stage
    assert "dist/${{ matrix.artifact_prefix }}-*.tar.gz" in pypi_stage
    github_publish = _run(jobs["github-release"], "Publish GitHub Release")
    assert "--clobber" in github_publish
    pypi_verification = _run(jobs["verify-pypi"], "Install and import from PyPI")
    for distribution in DISTRIBUTIONS:
        assert f'--with "{distribution}==$version"' in pypi_verification


def test_every_distribution_declares_and_copies_the_repository_license() -> None:
    expected = (ROOT / "LICENSE").read_bytes()
    for distribution in DISTRIBUTIONS:
        package_root = ROOT / "packages" / distribution
        assert (package_root / "LICENSE").read_bytes() == expected
        project = tomllib.loads((package_root / "pyproject.toml").read_text(encoding="utf-8"))
        metadata = _mapping(cast(object, project.get("project")), f"{distribution} project")
        assert metadata["license"] == "MIT"
        assert metadata["license-files"] == ["LICENSE"]


def test_test_suite_has_structural_workflow_dependencies_and_a_global_timeout() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    groups = _mapping(cast(object, project.get("dependency-groups")), "dependency groups")
    dev = groups.get("dev")
    assert isinstance(dev, list)
    dependencies = cast(list[object], dev)
    assert "pytest-timeout>=2.4.0" in dependencies
    assert "pymysql[rsa]>=1.2.0" in dependencies
    assert "pyyaml>=6.0.3" in dependencies
    assert "redis>=8.0.1" in dependencies
    assert "types-pyyaml>=6.0.12.20260518" in dependencies
    tool = _mapping(cast(object, project.get("tool")), "tool settings")
    pytest_settings = _mapping(
        _mapping(tool.get("pytest"), "pytest tool settings").get("ini_options"),
        "pytest ini options",
    )
    assert pytest_settings["timeout"] == 60
    assert pytest_settings["timeout_method"] == "thread"


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
    assert script.count('{{ index = "testpypi" }}') == 5


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


def test_testpypi_retries_only_index_propagation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_attempts = 0
    examples: list[str] = []
    sleeps: list[float] = []

    def verify_imports(_project: Path) -> None:
        nonlocal import_attempts
        import_attempts += 1
        if import_attempts < 3:
            raise subprocess.CalledProcessError(1, "uv")

    def run_example(_project: Path, example: Path) -> None:
        examples.append(example.name)

    monkeypatch.setattr(verify_testpypi, "_verify_public_imports", verify_imports)
    monkeypatch.setattr(verify_testpypi, "_run_example", run_example)
    monkeypatch.setattr(verify_testpypi.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_testpypi.py", "0.2.0", "--attempts", "3", "--delay", "2"],
    )

    assert verify_testpypi.main() == 0
    assert import_attempts == 3
    assert sleeps == [2.0, 2.0]
    assert examples == ["basic_tool_loop.py", "pause_resume_trace.py"]


def test_testpypi_does_not_retry_a_smoke_example_defect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples: list[str] = []
    sleeps: list[float] = []

    def run_example(_project: Path, example: Path) -> None:
        examples.append(example.name)
        raise subprocess.CalledProcessError(1, str(example))

    def verify_imports(_project: Path) -> None:
        return None

    monkeypatch.setattr(verify_testpypi, "_verify_public_imports", verify_imports)
    monkeypatch.setattr(verify_testpypi, "_run_example", run_example)
    monkeypatch.setattr(verify_testpypi.time, "sleep", sleeps.append)
    monkeypatch.setattr(sys, "argv", ["verify_testpypi.py", "0.2.0", "--attempts", "3"])

    assert verify_testpypi.main() == 1
    assert examples == ["basic_tool_loop.py"]
    assert sleeps == []
