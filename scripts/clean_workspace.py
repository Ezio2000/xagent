"""Safely preview or remove generated repository artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_DIRECTORY_NAMES = frozenset({"build", "coverage", "dist", "htmlcov"})
_CACHE_DIRECTORY_NAMES = frozenset(
    {"__pycache__", ".mypy_cache", ".pyright", ".pytest_cache", ".ruff_cache"}
)
_GENERATED_FILE_NAMES = frozenset({".coverage", ".DS_Store"})
_GENERATED_FILE_SUFFIXES = frozenset({".pyc", ".pyo"})
_PROTECTED_DIRECTORY_NAMES = frozenset({".git", ".venv", "venv", "ENV"})


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    """Generated targets that may be removed safely."""

    targets: tuple[Path, ...]


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _is_protected(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return any(part in _PROTECTED_DIRECTORY_NAMES for part in relative.parts)


def _minimal_targets(paths: set[Path], root: Path) -> tuple[Path, ...]:
    ordered = sorted(paths, key=lambda path: (len(path.relative_to(root).parts), str(path)))
    selected: list[Path] = []
    for candidate in ordered:
        if candidate == root or not _is_within(candidate, root):
            raise ValueError(f"cleanup target escapes repository root: {candidate}")
        if _is_protected(candidate, root):
            raise ValueError(f"cleanup target is protected: {candidate}")
        if any(_is_within(candidate, parent) for parent in selected):
            continue
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda path: str(path.relative_to(root))))


def _prune_directory(candidate: Path, name: str, root: Path, targets: set[Path]) -> bool:
    if _is_protected(candidate, root):
        return True
    if any(candidate == target or _is_within(candidate, target) for target in targets):
        return True
    is_root_output = candidate.parent == root and name in _OUTPUT_DIRECTORY_NAMES
    if name in _CACHE_DIRECTORY_NAMES or is_root_output or name.endswith(".egg-info"):
        targets.add(candidate)
        return True
    return False


def build_cleanup_plan(root: Path) -> CleanupPlan:
    """Collect generated targets without traversing protected environments."""

    root = root.resolve()
    package_root = root / "packages"
    if not (root / "pyproject.toml").is_file() or not all(
        (package_root / distribution / "src" / "jharness" / component).is_dir()
        for distribution, component in (
            ("jharness-kernel", "kernel"),
            ("jharness-toolkit", "toolkit"),
            ("jharness-models", "models"),
            ("jharness-repository", "repository"),
            ("jharness-tools", "tools"),
        )
    ):
        raise ValueError(f"not a recognized JHarness repository root: {root}")
    targets: set[Path] = set()
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = [
            name
            for name in directories
            if not _prune_directory(current_path / name, name, root, targets)
        ]
        for name in files:
            candidate = current_path / name
            if not _is_protected(candidate, root) and (
                name in _GENERATED_FILE_NAMES or candidate.suffix in _GENERATED_FILE_SUFFIXES
            ):
                targets.add(candidate)
    return CleanupPlan(_minimal_targets(targets, root))


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def main() -> int:
    """Preview cleanup by default and mutate only with explicit confirmation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="remove displayed targets")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        plan = build_cleanup_plan(root)
    except (OSError, ValueError) as exc:
        print(f"workspace cleanup failed: {exc}", file=sys.stderr)
        return 1
    if not plan.targets:
        print("workspace is clean")
        return 0
    action = "removing" if args.apply else "would remove"
    try:
        for path in plan.targets:
            print(f"{action} {path.relative_to(root).as_posix()}")
            if args.apply:
                _remove(path)
    except OSError as exc:
        print(f"workspace cleanup failed: {exc}", file=sys.stderr)
        return 1
    print(f"{'removed' if args.apply else 'previewed'} {len(plan.targets)} artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
