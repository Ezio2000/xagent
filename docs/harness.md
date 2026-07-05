# Harness Package

`harness` is the controlled test environment for the Python runtime workspace.
Its job is to make runtime behavior easy to run, observe, and assert in
repeatable scenarios without moving production semantics out of `kernel`,
`toolkit`, `prompting`, `modelkit`, or `diagnostics`.

The package is test infrastructure, not a production SDK layer. It may provide
model drivers, fake runtime ports, tool stubs and registry doubles, scenario
builders, message fixtures, event collectors, timeline helpers, trace
assertions, and behavior assertions. It may compose public APIs from sibling
packages that are needed to build realistic tests. It must not define
production state transitions, scheduling rules, checkpoint semantics, portable
conformance contracts, JSON Schema validation rules, provider adapters, or app
infrastructure.

## Layers

| Layer | Responsibility | Must Not Own |
| --- | --- | --- |
| `scenarios` | High-level harness entry points that assemble runtime package inputs, run controlled scenarios, and return observed results. | Kernel runtime semantics, conformance case parsing, trace replay implementation. |
| `drivers` | Model and stream drivers for deterministic responses, errors, timeouts, cancellation, and control actions. | Provider adapters or model protocol semantics. |
| `environment` | Fake runtime ports and policies: stores, journals, approval policies, hooks, and context helpers used to simulate host infrastructure. | Concrete production stores, approval UIs, queues, or deployment runtime. |
| `tools` | Tool stubs, registry doubles, invocation records, and reusable controlled tool behaviors, including toolkit-facing fixtures when needed by tests. | Production tool implementations or JSON Schema validation rules. |
| `observation` | Event collection, filtering, timeline helpers, and trace payload access for observing a run. | Diagnostics trace object semantics or replay implementation. |
| `assertions` | Reusable behavior assertions for events, state, checkpoints, pause/resume, ordering, and traces. | Portable conformance contract definitions or fixture schema validation. |

## Package Boundaries

`harness` may import public package roots from `kernel`, `toolkit`, `prompting`,
and `diagnostics`. That makes it a workspace-level test harness that can compose
the same public APIs a host or runner would compose. It must not import package-
private submodules from sibling packages.

Runtime source packages must not import `harness`. `kernel`, `toolkit`,
`prompting`, `modelkit`, and `diagnostics` own production behavior; `harness`
only owns controlled test equipment around that behavior.

`conformance` may use `harness` to build deterministic runs, but fixture loading,
case-field interpretation, schema validation, and expected portable behavior
remain in `conformance`, `contracts/v0`, and `conformance/cases`.

Tests should prefer importing harness APIs through the package root:

```python
from harness import MemoryRunStore, ScriptedModel, collect_events
```

Subpackages exist to keep implementation ownership clear. They are not an
invitation for cross-package consumers to bypass the root API.
