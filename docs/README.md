# Documentation

Use the shortest path that matches your role:

- Users start with the repository [`README`](../README.md) and the README for the
  installed package.
- Host integrators read [`architecture.md`](architecture.md), then the protocol or
  adapter document for the boundary they implement.
- Contributors follow [`CONTRIBUTING.md`](../CONTRIBUTING.md), the relevant contract,
  and its conformance coverage.

## Sources of Truth

- [`contracts/v0`](../contracts/v0/README.md) defines portable behavior and wire
  compatibility.
- This directory explains architecture, public integration, and operations without
  redefining contracts.
- [`decisions`](decisions/README.md) records why major constraints were chosen.
- [`conformance`](../conformance/README.md) explains how implementations prove the
  contracts.

## Guide

| Need | Read |
| --- | --- |
| System overview | [`architecture.md`](architecture.md) |
| Lifecycle and controls | [`state-machine.md`](state-machine.md), [`event-stream.md`](event-stream.md) |
| Model or tool integration | [`model-protocol.md`](model-protocol.md), [`tool-protocol.md`](tool-protocol.md) |
| Provider adapters | [`model-adapters.md`](model-adapters.md) |
| Persistence and decoding | [`wire-protocol.md`](wire-protocol.md) |
| Traces and verification | [`diagnostics.md`](diagnostics.md) |
| Package ownership | [`python-package-boundaries.md`](python-package-boundaries.md) |
| Performance constraints | [`performance.md`](performance.md) |
| Publishing | [`releasing.md`](releasing.md) |

Normative details should appear once under `contracts/v0`; explanatory documents link
to them rather than copying their field lists or transition tables.
