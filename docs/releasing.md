# Release Process

JHarness releases `jharness-kernel`, `jharness-toolkit`, `jharness-models`,
`jharness-repository`, and `jharness-tools` from one version and one immutable tag. A
successful release contains five wheels and five source distributions built once and
published together.

## One-Time Repository Setup

For each of the five projects, configure a Trusted Publisher on TestPyPI and PyPI for
repository `Ezio2000/jharness` and workflow `release.yml`. Use a dedicated GitHub
environment for each project:

| Project | TestPyPI environment | PyPI environment |
| --- | --- | --- |
| `jharness-kernel` | `testpypi-jharness-kernel` | `pypi-jharness-kernel` |
| `jharness-models` | `testpypi-jharness-models` | `pypi-jharness-models` |
| `jharness-repository` | `testpypi-jharness-repository` | `pypi-jharness-repository` |
| `jharness-toolkit` | `testpypi-jharness-toolkit` | `pypi-jharness-toolkit` |
| `jharness-tools` | `testpypi-jharness-tools` | `pypi-jharness-tools` |

The distinct environments allow all not-yet-created projects to register pending
publishers at the same time and scope each OIDC credential to one distribution.
Protect `main` and `v*` tags, require CI, pin allowed Actions, and enable GitHub
dependency and secret scanning. Publication uses OIDC; do not store long-lived
package-index credentials.

## Prepare a Release

1. Set the same PEP 440 version in all five `packages/*/pyproject.toml` files.
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

1. verifies all five versions, dependency pins, tag, changelog, and lock file;
2. reruns the complete quality gate;
3. builds all ten archives exactly once;
4. verifies archive ownership, non-overlap, metadata, dependencies, checksums, and an
   exact copy of the repository `LICENSE` in every wheel and source distribution;
5. installs all local wheels together without repository drivers, verifies base
   imports, then verifies the repository driver extras;
6. publishes the same ten archives to TestPyPI in five parallel, project-scoped jobs;
7. installs all five exact TestPyPI versions and runs smoke examples;
8. publishes the same files to PyPI in five parallel, project-scoped jobs and verifies
   each component plus the full set;
9. creates one GitHub Release containing all archives and `SHA256SUMS`.

The workflow must never rebuild between TestPyPI, PyPI, and GitHub Release. Publishing
only a subset is a failed release, even if an individual PyPI upload succeeded.
TestPyPI verification retries only the exact-version installation/import check while
the index propagates new files. Once imports succeed, each offline smoke example runs
once; a smoke failure is a release defect and must fail immediately rather than being
hidden by propagation retries.

## Failure and Recovery

Workflow dispatch may recover only the verified artifact set from a previous release
run for the same immutable tag and commit. It verifies hashes and all ten archives,
then uses `skip-existing` to complete missing uploads. Never move a published tag or
overwrite a version; publish a patch release when correction is required.

Normal PEP 440 prereleases such as `0.3.0a1` and `0.3.0rc1` are supported when every
manifest, dependency pin, tag, and changelog entry agrees.
