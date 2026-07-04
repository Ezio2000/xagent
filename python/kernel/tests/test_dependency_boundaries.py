from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "python"

PACKAGE_IMPORTS = {
    "conformance",
    "diagnostics",
    "harness",
    "kernel",
    "modelkit",
    "prompting",
    "toolkit",
}
PROJECT_NAMES = {
    "conformance": "conformance",
    "diagnostics": "diagnostics",
    "harness": "harness",
    "kernel": "kernel",
    "modelkit": "modelkit",
    "prompting": "prompting",
    "toolkit": "toolkit",
}
RETIRED_PACKAGE_IMPORTS = {
    "agent_runtime",
    "agent_runtime_conformance",
    "engine",
    "extensions",
    "protocol",
    "run_state",
    "tracing",
}
RETIRED_PACKAGE_DIRS = {
    "agent_runtime",
    "agent_runtime_conformance",
    "engine",
    "extensions",
    "protocol",
    "run_state",
    "run-state",
    "tracing",
}
RETIRED_SOURCE_PATHS = (REPO_ROOT / "sdks",)
FORBIDDEN_PROJECT_NAME_FRAGMENTS = ("xagent", "agent_", "agent-", "runtime_", "runtime-")

ALLOWED_IMPORTS = {
    "kernel": set[str](),
    "diagnostics": {"kernel"},
    "harness": {"kernel"},
    "modelkit": {"kernel"},
    "prompting": {"kernel"},
    "toolkit": {"kernel"},
    "conformance": {"diagnostics", "kernel", "prompting", "toolkit"},
}
EXPECTED_PROJECT_DEPENDENCIES = {
    "kernel": set[str](),
    "diagnostics": {"kernel"},
    "harness": {"kernel"},
    "modelkit": {"kernel"},
    "prompting": {"kernel"},
    "toolkit": {"jsonschema", "kernel"},
    "conformance": {"diagnostics", "jsonschema", "kernel", "prompting", "referencing", "toolkit"},
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


def cross_package_private_imports(path: Path, package_name: str) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                root = parts[0]
                if root in PACKAGE_IMPORTS and root != package_name and len(parts) > 1:
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            root = parts[0]
            if root in PACKAGE_IMPORTS and root != package_name and len(parts) > 1:
                violations.append(node.module)
    return violations


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


def test_python_src_layout_matches_package_name() -> None:
    for package_dir in sorted(PACKAGE_ROOT.iterdir()):
        if not package_dir.is_dir():
            continue
        src_dir = package_dir / "src"
        package_src = src_dir / package_dir.name
        assert package_src.is_dir(), f"{package_dir.name} must use src/{package_dir.name}"
        assert (package_src / "__init__.py").is_file()
        assert (package_src / "py.typed").is_file()
        import_roots = {
            path.name
            for path in src_dir.iterdir()
            if path.is_dir() and (path / "__init__.py").is_file()
        }
        assert import_roots == {package_dir.name}


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


def test_cross_package_imports_use_public_package_api() -> None:
    violations: list[str] = []
    for package_dir in sorted(PACKAGE_ROOT.iterdir()):
        if not package_dir.is_dir():
            continue
        package_name = package_dir.name
        for path in sorted(package_dir.rglob("*.py")):
            forbidden = cross_package_private_imports(path, package_name)
            if forbidden:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)} imports non-root package API: "
                    f"{', '.join(sorted(forbidden))}"
                )
    assert not violations, "\n".join(violations)


def test_kernel_does_not_define_diagnostics_replay_api() -> None:
    violations: list[str] = []
    forbidden = {
        "def replay_trace": re.compile(r"^def replay_trace\b", re.MULTILINE),
        "class ReplayError": re.compile(r"^class ReplayError\b", re.MULTILINE),
        "class ReplayResult": re.compile(r"^class ReplayResult\b", re.MULTILINE),
        "class RunTrace": re.compile(r"^class RunTrace\b", re.MULTILINE),
        "class TraceStep": re.compile(r"^class TraceStep\b", re.MULTILINE),
    }
    for path in sorted((PACKAGE_ROOT / "kernel" / "src" / "kernel").rglob("*.py")):
        text = path.read_text()
        matches = [name for name, pattern in forbidden.items() if pattern.search(text)]
        if matches:
            violations.append(
                f"{path.relative_to(REPO_ROOT)} defines diagnostics API: {', '.join(matches)}"
            )
    assert not violations, "\n".join(violations)


def test_retired_runtime_imports_do_not_remain() -> None:
    violations: list[str] = []
    source_roots = [
        package_dir / "src"
        for package_dir in sorted(PACKAGE_ROOT.iterdir())
        if package_dir.is_dir() and (package_dir / "src").is_dir()
    ]
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*.py")):
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


def test_kernel_and_diagnostics_frozen_helpers_stay_identical() -> None:
    kernel_helper = PACKAGE_ROOT / "kernel" / "src" / "kernel" / "_frozen.py"
    diagnostics_helper = PACKAGE_ROOT / "diagnostics" / "src" / "diagnostics" / "_frozen.py"

    assert kernel_helper.read_bytes() == diagnostics_helper.read_bytes()


def dependency_name(dependency: str) -> str:
    return re.split(r"[\s<>=!~;\[]", dependency, maxsplit=1)[0]


def project_dependencies(package_dir: Path) -> set[str]:
    data = tomllib.loads((package_dir / "pyproject.toml").read_text())
    raw_dependencies = data.get("project", {}).get("dependencies", [])
    if not isinstance(raw_dependencies, list):
        raise TypeError(f"{package_dir.name} dependencies must be a list")
    dependencies: set[str] = set()
    for dependency in cast(list[object], raw_dependencies):
        if not isinstance(dependency, str):
            raise TypeError(f"{package_dir.name} dependency must be a string")
        dependencies.add(dependency_name(dependency))
    return dependencies


def test_python_package_manifests_match_expected_dependencies() -> None:
    violations: list[str] = []
    for package_dir in sorted(PACKAGE_ROOT.iterdir()):
        if not package_dir.is_dir():
            continue
        expected = EXPECTED_PROJECT_DEPENDENCIES[package_dir.name]
        actual = project_dependencies(package_dir)
        if actual != expected:
            violations.append(
                f"{package_dir.relative_to(REPO_ROOT)} dependencies mismatch: "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )
    assert not violations, "\n".join(violations)


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
