"""Verify the coordinated JHarness wheels and source distributions."""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import tomllib
import zipfile
from dataclasses import dataclass
from email import message_from_bytes
from email.message import Message
from pathlib import Path, PurePosixPath
from typing import cast

VERSION = re.compile(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?")
REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9_.-]+)")
EXTRA_MARKER = re.compile(r"\(?\s*extra\s*==\s*['\"]([A-Za-z0-9_.-]+)['\"]\s*\)?")
COMPONENTS = {
    "jharness-kernel": "kernel",
    "jharness-toolkit": "toolkit",
    "jharness-models": "models",
    "jharness-repository": "repository",
    "jharness-tools": "tools",
}
DEPENDENCIES: dict[str, set[str]] = {
    "jharness-kernel": set(),
    "jharness-toolkit": {"jharness-kernel", "jsonschema", "referencing"},
    "jharness-models": {"jharness-kernel", "httpx"},
    "jharness-repository": {"jharness-kernel"},
    "jharness-tools": {"jharness-kernel", "regex"},
}
OPTIONAL_REQUIREMENTS: dict[str, dict[str, set[str]]] = {
    "jharness-repository": {
        "mysql": {"pymysql[rsa]>=1.2.0"},
        "redis": {"redis>=8.0.1"},
    }
}
ROOT_LICENSE = (Path(__file__).resolve().parents[1] / "LICENSE").read_bytes()


@dataclass(frozen=True)
class Wheel:
    path: Path
    distribution: str
    version: str
    files: frozenset[str]


def _normalized(distribution: str) -> str:
    return re.sub(r"[-_.]+", "_", distribution)


def _safe_archive_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive path: {name!r}")
    return path


def _metadata(archive: zipfile.ZipFile) -> Message:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("wheel contains duplicate archive paths")
    for name in names:
        _safe_archive_path(name)
    paths = [name for name in names if name.endswith(".dist-info/METADATA")]
    if len(paths) != 1:
        raise ValueError("wheel must contain exactly one METADATA file")
    return message_from_bytes(archive.read(paths[0]))


def _requirement_name(requirement: str) -> str:
    match = REQUIREMENT_NAME.match(requirement)
    if match is None:
        raise ValueError(f"invalid Requires-Dist: {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(1).lower())


def _normalized_requirement(requirement: str) -> str:
    name_match = REQUIREMENT_NAME.match(requirement)
    if name_match is None:
        raise ValueError(f"invalid Requires-Dist: {requirement!r}")
    name = _requirement_name(name_match.group(1))
    remainder = requirement[name_match.end() :].strip()
    extras = ""
    if remainder.startswith("["):
        closing = remainder.find("]")
        if closing < 0:
            raise ValueError(f"invalid Requires-Dist extras: {requirement!r}")
        raw_extras = remainder[1:closing].split(",")
        normalized_extras = sorted(
            _requirement_name(extra.strip()) for extra in raw_extras if extra.strip()
        )
        if not normalized_extras or len(normalized_extras) != len(raw_extras):
            raise ValueError(f"invalid Requires-Dist extras: {requirement!r}")
        extras = f"[{','.join(normalized_extras)}]"
        remainder = remainder[closing + 1 :].strip()
    specifier = re.sub(r"\s+", "", remainder)
    if specifier and re.fullmatch(r"(?:===|==|!=|~=|<=|>=|<|>)[^;]+", specifier) is None:
        raise ValueError(f"unsupported Requires-Dist specification: {requirement!r}")
    return f"{name}{extras}{specifier}"


def _requirement_groups(message: Message) -> dict[str | None, set[str]]:
    groups: dict[str | None, set[str]] = {}
    requirements = message.get_all("Requires-Dist", [])
    for requirement in requirements:
        specification, separator, marker = requirement.partition(";")
        extra: str | None = None
        if separator:
            match = EXTRA_MARKER.fullmatch(marker.strip())
            if match is None:
                raise ValueError(f"unsupported Requires-Dist marker: {requirement!r}")
            extra = re.sub(r"[-_.]+", "-", match.group(1).lower())
        groups.setdefault(extra, set()).add(_normalized_requirement(specification.strip()))
    return groups


def _verify_dependencies(message: Message, distribution: str, version: str) -> None:
    requirement_groups = _requirement_groups(message)
    required = requirement_groups.pop(None, set())
    actual_dependencies = {_requirement_name(requirement) for requirement in required}
    if actual_dependencies != DEPENDENCIES[distribution]:
        raise ValueError(f"{distribution} dependencies differ: {sorted(actual_dependencies)}")
    expected_optional = {
        extra: {_normalized_requirement(requirement) for requirement in requirements}
        for extra, requirements in OPTIONAL_REQUIREMENTS.get(distribution, {}).items()
    }
    if requirement_groups != expected_optional:
        raise ValueError(f"{distribution} optional dependencies differ: {requirement_groups}")
    provided_extras = {
        re.sub(r"[-_.]+", "-", extra.lower()) for extra in message.get_all("Provides-Extra", [])
    }
    if provided_extras != set(expected_optional):
        raise ValueError(f"{distribution} extras differ: {sorted(provided_extras)}")
    if distribution != "jharness-kernel" and f"jharness-kernel=={version}" not in required:
        raise ValueError(f"{distribution} does not pin the coordinated kernel")


def _verify_wheel(path: Path) -> Wheel:
    with zipfile.ZipFile(path) as archive:
        message = _metadata(archive)
        distribution = str(message.get("Name", "")).lower()
        version = str(message.get("Version", ""))
        if distribution not in COMPONENTS or VERSION.fullmatch(version) is None:
            raise ValueError(
                f"unexpected wheel identity: name={distribution!r} version={version!r}"
            )
        normalized = _normalized(distribution)
        if path.name != f"{normalized}-{version}-py3-none-any.whl":
            raise ValueError(f"unexpected wheel filename: {path.name!r}")
        component = COMPONENTS[distribution]
        info = f"{normalized}-{version}.dist-info"
        names = frozenset(archive.namelist())
        required = {
            f"jharness/{component}/__init__.py",
            f"jharness/{component}/py.typed",
            f"{info}/METADATA",
            f"{info}/RECORD",
            f"{info}/WHEEL",
            f"{info}/licenses/LICENSE",
        }
        if missing := sorted(required - names):
            raise ValueError(f"{distribution} wheel is missing files: {missing}")
        if "jharness/__init__.py" in names:
            raise ValueError(f"{distribution} must not own jharness/__init__.py")
        if archive.read(f"{info}/licenses/LICENSE") != ROOT_LICENSE:
            raise ValueError(f"{distribution} wheel LICENSE differs from the repository")
        allowed = (f"jharness/{component}/", f"{info}/")
        if unexpected := sorted(name for name in names if not name.startswith(allowed)):
            raise ValueError(f"{distribution} wheel contains unexpected files: {unexpected[:5]}")
        _verify_dependencies(message, distribution, version)
    return Wheel(path, distribution, version, names)


def _verify_sdist(path: Path, *, distribution: str, version: str) -> None:
    normalized = _normalized(distribution)
    expected_root = f"{normalized}-{version}"
    if path.name != f"{expected_root}.tar.gz":
        raise ValueError(f"unexpected sdist filename: {path.name!r}")
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        raw_names = [member.name for member in members]
        if len(raw_names) != len(set(raw_names)):
            raise ValueError(f"{distribution} sdist contains duplicate paths")
        if any(member.issym() or member.islnk() for member in members):
            raise ValueError(f"{distribution} sdist must not contain links")
        names = {_safe_archive_path(name) for name in raw_names}
        roots = {name.parts[0] for name in names}
        if roots != {expected_root}:
            raise ValueError(f"unexpected {distribution} sdist roots: {roots}")
        component = COMPONENTS[distribution]
        required = {
            PurePosixPath(expected_root, "LICENSE"),
            PurePosixPath(expected_root, "pyproject.toml"),
            PurePosixPath(expected_root, "README.md"),
            PurePosixPath(expected_root, "src", "jharness", component, "__init__.py"),
            PurePosixPath(expected_root, "src", "jharness", component, "py.typed"),
        }
        if missing := sorted(str(name) for name in required - names):
            raise ValueError(f"{distribution} sdist is missing files: {missing}")
        license_file = archive.extractfile(f"{expected_root}/LICENSE")
        if license_file is None or license_file.read() != ROOT_LICENSE:
            raise ValueError(f"{distribution} sdist LICENSE differs from the repository")
        project_file = archive.extractfile(f"{expected_root}/pyproject.toml")
        if project_file is None:
            raise ValueError(f"{distribution} sdist has no regular pyproject.toml")
        project = tomllib.loads(project_file.read().decode())
        metadata = cast(dict[str, object], project["project"])
        if metadata.get("name") != distribution or metadata.get("version") != version:
            raise ValueError(f"{distribution} sdist metadata differs from its wheel")
        raw_optional = cast(
            dict[str, list[str]],
            metadata.get("optional-dependencies", {}),
        )
        optional = {extra: set(requirements) for extra, requirements in raw_optional.items()}
        if optional != OPTIONAL_REQUIREMENTS.get(distribution, {}):
            raise ValueError(f"{distribution} sdist optional dependencies differ: {optional}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory containing build artifacts")
    args = parser.parse_args()
    try:
        wheel_paths = sorted(args.directory.glob("*.whl"))
        sdist_paths = sorted(args.directory.glob("*.tar.gz"))
        if len(wheel_paths) != 5 or len(sdist_paths) != 5:
            raise ValueError(
                "expected five wheels and five sdists, "
                f"got {len(wheel_paths)} and {len(sdist_paths)}"
            )
        wheels = [_verify_wheel(path) for path in wheel_paths]
        by_distribution = {wheel.distribution: wheel for wheel in wheels}
        if set(by_distribution) != set(COMPONENTS):
            raise ValueError(f"unexpected wheel set: {sorted(by_distribution)}")
        versions = {wheel.version for wheel in wheels}
        if len(versions) != 1:
            raise ValueError(f"wheel versions differ: {sorted(versions)}")
        version = next(iter(versions))

        owned_paths: set[str] = set()
        for wheel in wheels:
            package_paths = {name for name in wheel.files if name.startswith("jharness/")}
            if overlap := sorted(owned_paths & package_paths):
                raise ValueError(f"wheels overlap namespace files: {overlap}")
            owned_paths.update(package_paths)

        for distribution in COMPONENTS:
            normalized = _normalized(distribution)
            path = args.directory / f"{normalized}-{version}.tar.gz"
            _verify_sdist(path, distribution=distribution, version=version)
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        tarfile.TarError,
        tomllib.TOMLDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"distribution verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"distribution set ok: count=5 version={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
