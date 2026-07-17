# Release Process

JHarness releases `jharness-kernel`, `jharness-toolkit`, `jharness-models`, and
`jharness-tools` from one version and one immutable tag. A successful release contains
four wheels and four source distributions built once and published together.

## One-Time Repository Setup

For each of the four projects, configure a Pending Trusted Publisher on TestPyPI and
PyPI for repository `Ezio2000/jharness`, workflow `release.yml`, and the matching
GitHub environment:

- TestPyPI: `testpypi-jharness`
- PyPI: `pypi-jharness`

The same environment may authorize all four project publishers. Protect `main` and
`v*` tags, require CI, pin allowed Actions, and enable GitHub dependency and secret
scanning. Publication uses OIDC; do not store long-lived package-index credentials.

## Prepare a Release

1. Set the same PEP 440 version in all four `packages/*/pyproject.toml` files.
2. Pin that version in the root workspace dependencies and every component's kernel
   dependency.
3. Update `CHANGELOG.md` and comparison links.
4. Refresh the lock file with `uv lock` and install it with `uv sync --locked`.
5. Run:

   ```bash
   uv run python scripts/verify_release.py
   uv run ruff check .
   uv run ruff format --check .
   uv run pyright
   uv run pytest -q -p no:cacheprovider --cov
   uv run python scripts/validate_spec.py
   uv run python -m conformance.cli conformance/cases --spec-dir contracts/v0 --quiet
   uv run python benchmarks/runtime_smoke.py
   uv build --all-packages --out-dir dist
   uv run python scripts/verify_distribution.py dist
   uv run twine check dist/*
   ```

## Publish

Create and push the reviewed `vX.Y.Z` tag. The release workflow:

1. verifies all four versions, dependency pins, tag, changelog, and lock file;
2. reruns the complete quality gate;
3. builds all eight archives exactly once;
4. verifies archive ownership, non-overlap, metadata, dependencies, and checksums;
5. installs all local wheels together in isolation;
6. publishes the same eight archives to TestPyPI;
7. installs all four exact TestPyPI versions and runs smoke examples;
8. publishes the same files to PyPI and verifies each component plus the full set;
9. creates one GitHub Release containing all archives and `SHA256SUMS`.

The workflow must never rebuild between TestPyPI, PyPI, and GitHub Release. Publishing
only a subset is a failed release, even if an individual PyPI upload succeeded.

## Failure and Recovery

Workflow dispatch may recover only the verified artifact set from a previous release
run for the same immutable tag and commit. It verifies hashes and all eight archives,
then uses `skip-existing` to complete missing uploads. Never move a published tag or
overwrite a version; publish a patch release when correction is required.

Normal PEP 440 prereleases such as `0.3.0a1` and `0.3.0rc1` are supported when every
manifest, dependency pin, tag, and changelog entry agrees.
