# ADR 0007: Explicit Wire Codecs

Status: Accepted
Date: 2026-07-13

## Context

Serialization methods on every domain value mix validation, portable naming,
copying, and state behavior. Generic reflection frameworks would add runtime
weight and obscure cross-field invariants.

## Decision

Place explicit top-level portable codecs in `jharness.kernel.wire`. Nested codecs are
private. Domain values have no generic reflective serialization methods.

Decode validates the complete untrusted aggregate once. Internal frozen values
are reused directly. Schema version belongs to top-level wire documents, not
domain objects. Provider codecs and tool JSON Schema validation stay in their owning
`jharness.models` and `jharness.toolkit` packages.

## Consequences

- Domain code and wire evolution have visible separate ownership.
- Unknown fields, discriminators, numeric rules, and aggregate invariants remain
  strict.
- Serialization is not used as defensive copying.
- Kernel gains no reflection serializer or validation-framework dependency.
