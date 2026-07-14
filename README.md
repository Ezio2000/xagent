# JHarness Specification

JHarness defines a model-neutral runtime contract for durable model and tool
execution. This repository is the single source of truth for portable schemas,
behavior cases, and language-neutral architecture decisions.

Implementations live in independent repositories and pin a released version of
this specification.

## Repository Index

| Area | Path |
| --- | --- |
| Architecture | [`docs/architecture.md`](docs/architecture.md) |
| Architecture decisions | [`docs/decisions/`](docs/decisions/README.md) |
| State machine | [`docs/state-machine.md`](docs/state-machine.md) |
| Model protocol | [`docs/model-protocol.md`](docs/model-protocol.md) |
| Tool protocol | [`docs/tool-protocol.md`](docs/tool-protocol.md) |
| Event stream | [`docs/event-stream.md`](docs/event-stream.md) |
| Wire boundary | [`docs/wire-protocol.md`](docs/wire-protocol.md) |
| Diagnostics | [`docs/diagnostics.md`](docs/diagnostics.md) |
| Portable contracts | [`contracts/v0/`](contracts/v0/README.md) |
| Conformance cases | [`conformance/cases/`](conformance/cases/) |

## Implementations

| Language | Repository | Status |
| --- | --- | --- |
| Python | [`jharness-python`](https://github.com/Ezio2000/jharness-python) | Active |
| Go | [`jharness-go`](https://github.com/Ezio2000/jharness-go) | Planned; not implemented |

Repository names are implementation-management boundaries, not umbrella packages.
Python users install `jharness-kernel`, `jharness-toolkit`, and
`jharness-providers`; future Go users will consume the Go module directly.

## Versioning

Specification releases use repository tags such as `v0.1.0`. The wire contract
family remains `contracts/v0`; a specification release identifies one immutable
combination of documentation, schemas, and conformance cases within that family.

Implementations version independently and declare the exact specification tag and
commit they implement. A portable behavior change lands and releases here before an
implementation updates its pin.

## Validation

Development tooling is managed with `uv`; it is not a runtime implementation or a
published package.

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run python scripts/validate_spec.py
```

The validator resolves every schema offline, validates all portable cases and the
standard tool catalog, verifies coverage inventory, and checks local documentation
links.
