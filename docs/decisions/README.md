# Architecture Decisions

These records define accepted runtime architecture. They are normative design
inputs for contracts and code.

| ADR | Decision |
| --- | --- |
| [0001](0001-algebraic-run-state.md) | Use a flat algebraic lifecycle state. |
| [0002](0002-atomic-checkpoint.md) | Persist one atomic checkpoint containing snapshot and fact. |
| [0003](0003-single-tool-invocation.md) | Use one tool invocation protocol and a closed result union. |
| [0004](0004-narrow-extension-ports.md) | Extend through narrow ports, pure policies, and decorators. |
| [0005](0005-runtime-invocation.md) | Use one runtime and one single-use invocation execution. |
| [0006](0006-single-model-invoke.md) | Use one complete/streaming model operation. |
| [0007](0007-explicit-wire-codecs.md) | Separate domain values from explicit portable codecs. |
| [0008](0008-trace-verification.md) | Verify traces with shared pure durable rules. |

Changing an accepted decision requires a replacement ADR plus an atomic update
to documentation, portable contracts, and conformance cases. Implementations then
adopt the released specification through their pinned revision. The repository keeps
only one active protocol shape within a contract family.
