# ADR 0009: Single Repository and Coordinated Python Distributions

Status: Accepted
Date: 2026-07-18

## Context

Kernel, toolkit, model adapters, and reusable tools evolve with the same contracts and
conformance suite, so they belong in one repository. They have different dependency
costs and useful standalone installation modes, so forcing all code into one wheel
unnecessarily installs HTTP, schema-validation, and tool dependencies.

Multiple wheels sharing a regular `jharness` root package would also overlap files and
make installation and uninstallation order-dependent.

## Decision

Maintain one repository, one lock file, one coordinated version, one immutable tag,
and one release workflow. Publish four distributions from that tag:

- `jharness-kernel` owns `jharness.kernel`;
- `jharness-toolkit` owns `jharness.toolkit`;
- `jharness-models` owns `jharness.models`;
- `jharness-tools` owns `jharness.tools`.

The `jharness` root is a PEP 420 namespace. No distribution contains
`jharness/__init__.py`, and no archive path is owned by more than one wheel. Every
component includes its own `py.typed` marker.

Kernel has no runtime dependency. Each other distribution pins the exact coordinated
kernel version and declares only its own third-party dependencies. Toolkit, models,
and tools never depend on one another.

The root project is a non-published uv workspace. Contracts, conformance, tests,
examples, benchmarks, and scripts remain repository assets. One tag builds, verifies,
and publishes all eight archives as one immutable artifact set. Partial publication is
not considered a successful release.

No umbrella distribution, obsolete package, forwarding module, or compatibility alias
is published.

## Consequences

- Users install only the components they need.
- Installing toolkit, models, or tools also installs the exact matching kernel.
- Python imports remain under the cohesive `jharness.*` namespace.
- Four PyPI Trusted Publishers and artifact pairs must be configured.
- Version metadata is repeated in four standard project manifests and enforced equal
  by repository verification.
- A source distribution remains independently buildable without repository-local
  version injection.
