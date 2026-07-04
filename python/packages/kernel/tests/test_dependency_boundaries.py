from __future__ import annotations

import ast
import tomllib
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = REPO_ROOT / "python" / "packages"

PACKAGE_IMPORTS = {"conformance", "kernel"}
PROJECT_NAMES = {"conformance": "conformance", "kernel": "kernel"}
RETIRED_PACKAGE_IMPORTS = {
    "agent_runtime",
    "agent_runtime_conformance",
    "engine",
    "extensions",
    "protocol",
    "run_state",
    "tracing",
}
RETIRED_PACKAGE_DIRS = {"engine", "extensions", "protocol", "run-state", "tracing"}
RETIRED_SOURCE_PATHS = (REPO_ROOT / "sdks",)
FORBIDDEN_PROJECT_NAME_FRAGMENTS = ("xagent", "agent_", "agent-", "runtime_", "runtime-")

ALLOWED_IMPORTS = {
    "kernel": set[str](),
    "conformance": {"kernel"},
}


def project_name(pyproject: Path) -> str:
    data = tomllib.loads(pyproject.read_text())
    name = data.get("project", {}).get("name")
    if not isinstance(name, str):
        raise TypeError(f"{pyproject} project.name must be a string")
    return name


def imported_packages(path: Path, package_names: set[str]) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in package_names:
                    imports.add(root)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in package_names:
                imports.add(root)
    return imports


def test_retired_runtime_packages_are_removed() -> None:
    existing = {path.name for path in PACKAGE_ROOT.iterdir() if path.is_dir()}
    retired = existing & RETIRED_PACKAGE_DIRS
    assert not retired, f"retired package directories remain: {', '.join(sorted(retired))}"


def test_retired_sdk_source_trees_are_removed() -> None:
    remaining = [path.relative_to(REPO_ROOT) for path in RETIRED_SOURCE_PATHS if path.exists()]
    assert not remaining, f"retired SDK source trees remain: {remaining}"


def test_python_package_set_is_explicit() -> None:
    existing = {path.name for path in PACKAGE_ROOT.iterdir() if path.is_dir()}
    expected = set(ALLOWED_IMPORTS)
    assert existing == expected


def test_python_package_dependencies_follow_declared_boundaries() -> None:
    violations: list[str] = []
    for package_dir in sorted(PACKAGE_ROOT.iterdir()):
        if not package_dir.is_dir():
            continue
        package_name = package_dir.name
        allowed = ALLOWED_IMPORTS[package_name]
        src_dir = package_dir / "src"
        for path in sorted(src_dir.rglob("*.py")):
            package_import = path.relative_to(src_dir).parts[0]
            imports = imported_packages(path, PACKAGE_IMPORTS) - {package_import}
            forbidden = imports - allowed
            if forbidden:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} imports forbidden package(s): "
                    f"{', '.join(sorted(forbidden))}"
                )
    assert not violations, "\n".join(violations)


def test_retired_runtime_imports_do_not_remain() -> None:
    violations: list[str] = []
    for path in sorted((REPO_ROOT / "python" / "packages").rglob("*.py")):
        retired_imports = imported_packages(path, RETIRED_PACKAGE_IMPORTS)
        if retired_imports:
            violations.append(
                f"{path.relative_to(REPO_ROOT)} imports retired package(s): "
                f"{', '.join(sorted(retired_imports))}"
            )
    assert not violations, "\n".join(violations)


def test_project_names_do_not_use_retired_prefixes() -> None:
    violations: list[str] = []
    pyprojects = [REPO_ROOT / "pyproject.toml", *sorted(PACKAGE_ROOT.glob("*/pyproject.toml"))]
    for pyproject in pyprojects:
        name = project_name(pyproject)
        forbidden = [fragment for fragment in FORBIDDEN_PROJECT_NAME_FRAGMENTS if fragment in name]
        if forbidden:
            violations.append(
                f"{pyproject.relative_to(REPO_ROOT)} has retired project name fragment(s): "
                f"{', '.join(forbidden)}"
            )
    assert not violations, "\n".join(violations)


def project_dependencies(package_dir: Path) -> set[str]:
    data = tomllib.loads((package_dir / "pyproject.toml").read_text())
    raw_dependencies = data.get("project", {}).get("dependencies", [])
    if not isinstance(raw_dependencies, list):
        raise TypeError(f"{package_dir.name} dependencies must be a list")
    dependencies: set[str] = set()
    for dependency in cast(list[object], raw_dependencies):
        if not isinstance(dependency, str):
            raise TypeError(f"{package_dir.name} dependency must be a string")
        dependencies.add(dependency.split(" ", 1)[0].split(">=", 1)[0])
    return dependencies


def test_python_package_manifests_declare_direct_runtime_imports() -> None:
    violations: list[str] = []
    for package_dir in sorted(PACKAGE_ROOT.iterdir()):
        if not package_dir.is_dir():
            continue
        src_dir = package_dir / "src"
        direct_imports: set[str] = set()
        for path in sorted(src_dir.rglob("*.py")):
            package_import = path.relative_to(src_dir).parts[0]
            direct_imports.update(imported_packages(path, PACKAGE_IMPORTS) - {package_import})
        required_dependencies = {PROJECT_NAMES[name] for name in direct_imports}
        missing = required_dependencies - project_dependencies(package_dir)
        if missing:
            violations.append(
                f"{package_dir.relative_to(REPO_ROOT)} missing direct dependency/dependencies: "
                f"{', '.join(sorted(missing))}"
            )
    assert not violations, "\n".join(violations)
