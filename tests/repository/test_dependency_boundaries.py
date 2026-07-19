from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = {
    "kernel": "jharness-kernel",
    "toolkit": "jharness-toolkit",
    "models": "jharness-models",
    "repository": "jharness-repository",
    "tools": "jharness-tools",
}
IMPORT_ROOTS = {name: f"jharness.{name}" for name in COMPONENTS}
ALLOWED: dict[str, set[str]] = {
    "kernel": set(),
    "toolkit": {"jharness.kernel"},
    "models": {"jharness.kernel"},
    "repository": {"jharness.kernel"},
    "tools": {"jharness.kernel"},
}
PUBLIC_SUBMODULES = {("repository", "jharness.kernel.wire")}
SOURCE_ROOTS = {
    component: ROOT / "packages" / distribution / "src" / "jharness" / component
    for component, distribution in COMPONENTS.items()
}


def import_target(module: str) -> str:
    parts = module.split(".")
    if parts[0] == "jharness" and len(parts) > 1:
        return ".".join(parts[:2])
    return parts[0]


def imports(path: Path) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((import_target(alias.name), alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.extend(
                (import_target(module), module) for module in _import_from_modules(path, node)
            )
    return found


def _import_from_modules(path: Path, node: ast.ImportFrom) -> tuple[str, ...]:
    if node.level == 0:
        return () if node.module is None else (node.module,)

    relative = path.relative_to(ROOT)
    source_index = relative.parts.index("src")
    module_parts = list(relative.with_suffix("").parts[source_index + 1 :])
    package_parts = module_parts[:-1]
    keep = len(package_parts) - (node.level - 1)
    if keep < 0:
        return ("<relative-import-outside-package>",)
    base = package_parts[:keep]
    if node.module is not None:
        return (".".join((*base, *node.module.split("."))),)
    return tuple(".".join((*base, alias.name)) for alias in node.names)


def source_files(owner: str) -> list[Path]:
    return sorted(SOURCE_ROOTS[owner].rglob("*.py"))


def requirement_name(requirement: str) -> str:
    return re.split(r"[\s\[\]();<>=!~@]", requirement, maxsplit=1)[0].lower()


def distribution_dependencies(owner: str) -> set[str]:
    project_file = ROOT / "packages" / COMPONENTS[owner] / "pyproject.toml"
    document = tomllib.loads(project_file.read_text(encoding="utf-8"))
    project = cast(dict[str, Any], document["project"])
    return {requirement_name(item) for item in cast(list[str], project.get("dependencies", []))}


def distribution_optional_dependencies(owner: str) -> dict[str, set[str]]:
    project_file = ROOT / "packages" / COMPONENTS[owner] / "pyproject.toml"
    document = tomllib.loads(project_file.read_text(encoding="utf-8"))
    project = cast(dict[str, Any], document["project"])
    groups = cast(dict[str, list[str]], project.get("optional-dependencies", {}))
    return {
        extra: {requirement_name(requirement) for requirement in requirements}
        for extra, requirements in groups.items()
    }


def test_source_dependency_graph_is_one_way_and_public_rooted() -> None:
    violations: list[str] = []
    internal_roots = set(IMPORT_ROOTS.values())
    for owner in sorted(COMPONENTS):
        own_root = IMPORT_ROOTS[owner]
        for path in source_files(owner):
            for target, module in imports(path):
                if target not in internal_roots or target == own_root:
                    continue
                if target not in ALLOWED[owner]:
                    violations.append(f"{path.relative_to(ROOT)} imports forbidden {module}")
                elif module != target and (owner, module) not in PUBLIC_SUBMODULES:
                    violations.append(
                        f"{path.relative_to(ROOT)} bypasses public root with {module}"
                    )
    assert violations == []


def test_each_distribution_declares_exactly_its_used_dependencies() -> None:
    internal_distributions = {"jharness-kernel"}
    for owner in COMPONENTS:
        used = {
            target
            for path in source_files(owner)
            for target, _ in imports(path)
            if target not in IMPORT_ROOTS.values() and target not in sys.stdlib_module_names
        }
        declared = distribution_dependencies(owner) - internal_distributions
        assert used == declared


def test_kernel_has_only_standard_library_dependencies() -> None:
    imported = {
        target
        for path in source_files("kernel")
        for target, _ in imports(path)
        if target != IMPORT_ROOTS["kernel"]
    }
    assert imported <= sys.stdlib_module_names


def test_repository_remote_drivers_are_exact_optional_extras() -> None:
    assert distribution_dependencies("repository") == {"jharness-kernel"}
    assert distribution_optional_dependencies("repository") == {
        "mysql": {"pymysql"},
        "redis": {"redis"},
    }
    for owner in set(COMPONENTS) - {"repository"}:
        assert distribution_optional_dependencies(owner) == {}


def test_workspace_has_five_coordinated_distributions() -> None:
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = cast(dict[str, Any], document["project"])
    assert project["name"] == "jharness-workspace"
    assert cast(dict[str, Any], document["tool"])["uv"]["package"] is False
    members = set(cast(dict[str, Any], document["tool"])["uv"]["workspace"]["members"])
    assert members == {f"packages/{name}" for name in COMPONENTS.values()}


def test_distributions_own_non_overlapping_namespace_portions() -> None:
    assert not (ROOT / "src").exists()
    for component, distribution in COMPONENTS.items():
        namespace = ROOT / "packages" / distribution / "src" / "jharness"
        assert not (namespace / "__init__.py").exists()
        source = namespace / component
        assert (source / "__init__.py").is_file()
        assert (source / "py.typed").is_file()
        children = {path.name for path in namespace.iterdir() if path.is_dir()}
        assert children == {component}


def test_conformance_runner_is_development_only() -> None:
    assert (ROOT / "conformance" / "__init__.py").is_file()
    assert all(not (source / "conformance").exists() for source in SOURCE_ROOTS.values())


def test_repository_has_no_obsolete_package_layout() -> None:
    for relative in (
        "sdks",
        "python",
        "spec.lock",
        ".jharness-spec",
        "scripts/sync_spec.py",
    ):
        assert not (ROOT / relative).exists()
