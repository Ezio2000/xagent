# ADR 0010: Coordinated Python Component Distributions

Status: Accepted
Date: 2026-07-19
Supersedes: [ADR 0009](0009-python-single-project.md)

## Context

The kernel owns a narrow `RunRepository` protocol and portable checkpoint codecs, but
database drivers and backend-specific concurrency behavior do not belong in the
dependency-free runtime. Applications also need a supported implementation without
copying the revision-CAS and checkpoint-idempotency rules.

## Decision

Keep one repository, lock file, coordinated version, immutable tag, and release
workflow. Publish five distributions:

- `jharness-kernel` owns `jharness.kernel`;
- `jharness-toolkit` owns `jharness.toolkit`;
- `jharness-models` owns `jharness.models`;
- `jharness-repository` owns `jharness.repository`;
- `jharness-tools` owns `jharness.tools`.

`jharness-repository` pins the exact kernel version and owns the supported memory,
SQLite, MySQL, and Redis adapters. It consumes the public kernel protocol and wire
codec; it does not redefine checkpoints or persistence semantics. Database clients are
strict opt-in extras: the base distribution supports Memory and standard-library
SQLite, `mysql` selects PyMySQL, and `redis` selects redis-py. Remote adapter modules
load only the selected client at initialization. No implicit driver, compatibility
alias, or fallback path is retained, and database clients never become kernel
dependencies.

The implicit PEP 420 `jharness` namespace, non-overlapping wheel ownership, and
one-way dependencies from each component to kernel remain unchanged. One tag builds,
verifies, and publishes five wheels and five source distributions together.

## Consequences

- Applications can install official storage adapters independently of model and tool
  integrations.
- Kernel remains dependency-free and keeps its lightweight invocation-local default.
- MySQL and Redis users supply service endpoints; development integration tests use
  disposable containers rather than host-installed middleware.
- Applications install remote database clients only through the explicit backend
  extras they select.
- Five PyPI Trusted Publishers and artifact pairs must be configured.
