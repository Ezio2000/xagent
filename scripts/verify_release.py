"""Validate coordinated JHarness distribution versions and release metadata."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG_FILE = ROOT / "CHANGELOG.md"
VERSION = re.compile(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?")
RELEASE_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
COMPONENTS = {
    "jharness-kernel": "kernel",
    "jharness-toolkit": "toolkit",
    "jharness-models": "models",
    "jharness-repository": "repository",
    "jharness-tools": "tools",
}
FORBIDDEN_PATHS = (
    ROOT / "VERSION",
    ROOT / "src",
    ROOT / "spec.lock",
    ROOT / ".jharness-spec",
    ROOT / "scripts" / "sync_spec.py",
)


def _document(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _project(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], _document(path)["project"])


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in COMPONENTS:
        project_file = ROOT / "packages" / distribution / "pyproject.toml"
        project = _project(project_file)
        if project.get("name") != distribution:
            raise ValueError(f"unexpected project name in {project_file.relative_to(ROOT)}")
        version = project.get("version")
        if not isinstance(version, str) or VERSION.fullmatch(version) is None:
            raise ValueError(f"unsupported {distribution} version: {version!r}")
        versions[distribution] = version
    if len(set(versions.values())) != 1:
        raise ValueError(f"distribution versions differ: {versions}")
    return versions


def _verify_workspace(version: str) -> None:
    document = _document(ROOT / "pyproject.toml")
    project = cast(dict[str, Any], document["project"])
    if project.get("name") != "jharness-workspace":
        raise ValueError("root project must be the non-published jharness-workspace")
    expected_dependencies = {f"{name}=={version}" for name in COMPONENTS}
    if set(cast(list[str], project.get("dependencies", []))) != expected_dependencies:
        raise ValueError("root dependencies must pin all coordinated distributions")

    tool = cast(dict[str, Any], document["tool"])
    uv = cast(dict[str, Any], tool["uv"])
    if uv.get("package") is not False:
        raise ValueError("root workspace must not be published")
    workspace = cast(dict[str, Any], uv["workspace"])
    members = set(cast(list[str], workspace["members"]))
    expected_members = {f"packages/{name}" for name in COMPONENTS}
    if members != expected_members:
        raise ValueError(f"unexpected workspace members: {sorted(members)}")


def _verify_components(version: str) -> None:
    for distribution, component in COMPONENTS.items():
        root = ROOT / "packages" / distribution
        project = _project(root / "pyproject.toml")
        _verify_component_dependencies(distribution, project, version)

        source = root / "src" / "jharness" / component
        if not (source / "__init__.py").is_file():
            raise ValueError(f"{distribution} is missing its public package")
        if not (source / "py.typed").is_file():
            raise ValueError(f"{distribution} is missing its nested py.typed marker")
        if (root / "src" / "jharness" / "__init__.py").exists():
            raise ValueError(f"{distribution} must use the implicit jharness namespace")


def _verify_component_dependencies(
    distribution: str,
    project: dict[str, Any],
    version: str,
) -> None:
    dependencies = cast(list[str], project.get("dependencies", []))
    kernel_pin = f"jharness-kernel=={version}"
    if distribution == "jharness-kernel":
        if dependencies:
            raise ValueError("jharness-kernel must have no runtime dependencies")
    elif kernel_pin not in dependencies:
        raise ValueError(f"{distribution} must pin {kernel_pin}")

    optional = cast(dict[str, list[str]], project.get("optional-dependencies", {}))
    if distribution == "jharness-repository":
        expected_optional = {
            "mysql": ["pymysql[rsa]>=1.2.0"],
            "redis": ["redis>=8.0.1"],
        }
        if dependencies != [kernel_pin]:
            raise ValueError("jharness-repository base install must depend only on kernel")
        if optional != expected_optional:
            raise ValueError("jharness-repository optional driver extras differ")
    elif optional:
        raise ValueError(f"{distribution} must not declare optional dependencies")


def _verify_changelog(version: str, *, released: bool) -> None:
    changelog = CHANGELOG_FILE.read_text(encoding="utf-8")
    if "## [Unreleased]" not in changelog:
        raise ValueError("CHANGELOG.md must contain an [Unreleased] section")
    match = re.search(rf"^## \[{re.escape(version)}\] - (.+)$", changelog, re.MULTILINE)
    if match is None:
        raise ValueError(f"CHANGELOG.md has no section for version {version}")
    marker = match.group(1).strip()
    if released and RELEASE_DATE.fullmatch(marker) is None:
        raise ValueError(f"release {version} must have a YYYY-MM-DD date, got {marker!r}")
    if not released and marker != "Unreleased" and RELEASE_DATE.fullmatch(marker) is None:
        raise ValueError(f"invalid changelog marker for {version}: {marker!r}")


def _verify_no_obsolete_layout() -> None:
    stale = [str(path.relative_to(ROOT)) for path in FORBIDDEN_PATHS if path.exists()]
    if stale:
        raise ValueError(f"obsolete paths remain: {stale}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag, exactly v<coordinated-version>")
    args = parser.parse_args()
    try:
        versions = _versions()
        version = next(iter(versions.values()))
        if args.tag is not None and args.tag != f"v{version}":
            raise ValueError(f"tag must be v{version}, got {args.tag!r}")
        _verify_workspace(version)
        _verify_components(version)
        _verify_changelog(version, released=args.tag is not None)
        _verify_no_obsolete_layout()
    except (KeyError, OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"release verification failed: {exc}", file=sys.stderr)
        return 1
    names = ",".join(COMPONENTS)
    print(f"release metadata ok: distributions={names} version={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
