"""Validate the JHarness specification version, tag, and changelog."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def main() -> int:
    """Require a matching tag and dated changelog for a specification release."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag, exactly v<VERSION>")
    args = parser.parse_args()
    try:
        version = (ROOT / "VERSION").read_text().strip()
        if not version:
            raise ValueError("VERSION must not be empty")
        if args.tag is not None and args.tag != f"v{version}":
            raise ValueError(f"tag must be v{version}, got {args.tag!r}")
        changelog = (ROOT / "CHANGELOG.md").read_text()
        match = re.search(rf"^## \[{re.escape(version)}\] - (.+)$", changelog, re.MULTILINE)
        if match is None:
            raise ValueError(f"CHANGELOG.md has no section for version {version}")
        marker = match.group(1).strip()
        if args.tag is not None and RELEASE_DATE.fullmatch(marker) is None:
            raise ValueError(f"release {version} requires a YYYY-MM-DD changelog date")
        if marker != "Unreleased" and RELEASE_DATE.fullmatch(marker) is None:
            raise ValueError(f"invalid changelog marker for {version}: {marker!r}")
    except (OSError, ValueError) as exc:
        print(f"release verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"specification release metadata ok: version={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
