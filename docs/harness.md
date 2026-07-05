# Harness Package

`harness` is the controlled kernel assembly and scenario support package for
the Python runtime workspace. Its job is to compose `kernel.AgentLoop` with
model clients, tool registries, stores, journals, approval policies, hooks,
controllers, messages, observation, and assertions so runtime scenarios are easy
to run, inspect, and reuse.

The package provides thin runtime scenario assembly and reusable controlled
components. It is not a full scenario framework, a replacement for the kernel,
a provider adapter, or an app framework. It may provide controlled model
drivers, runtime port implementations and fakes, tool fixtures and registry
doubles, scenario builders, message fixtures, event collectors, timeline
helpers, trace assertions, and behavior assertions. It may compose public APIs
from sibling packages that are needed to build realistic runtime scenarios. It
must not define kernel state transitions, scheduling rules, checkpoint
semantics, portable conformance contracts, JSON Schema validation rules,
provider adapters, app infrastructure, or application-specific scenario
semantics.

## Layers

| Layer | Responsibility | Must Not Own |
| --- | --- | --- |
| `scenarios` | Thin harness entry points that assemble `kernel` inputs, run controlled runtime scenarios, and return observed results. | Kernel runtime semantics, conformance case parsing, trace replay implementation, application-specific scenario semantics. |
| `drivers` | Model and stream drivers for deterministic responses, errors, timeouts, cancellation, and control actions. | Provider adapters or model protocol semantics. |
| `environment` | Runtime port implementations and fakes: stores, journals, approval policies, hooks, and context helpers used to assemble host-like infrastructure. | Concrete production stores, approval UIs, queues, or deployment runtime. |
| `tools` | Tool fixtures, registry doubles, invocation records, and reusable controlled tool behaviors, including toolkit-facing fixtures when needed by scenarios. | Production tool implementations or JSON Schema validation rules. |
| `observation` | Event collection, filtering, timeline helpers, and trace payload access for observing a run. | Diagnostics trace object semantics or replay implementation. |
| `assertions` | Reusable behavior assertions for events, state, checkpoints, pause/resume, ordering, and traces. | Portable conformance contract definitions or fixture schema validation. |

## Package Boundaries

`harness` may import public package roots from `kernel`, `toolkit`, `prompting`,
and `diagnostics`. That makes it a workspace-level assembly and scenario
support package that can compose the same public APIs a host, scenario runner,
example, or conformance runner would compose. It must not import package-private
submodules from sibling packages.

Runtime source packages must not import `harness`. `kernel`, `toolkit`,
`prompting`, `modelkit`, and `diagnostics` own lower-level runtime behavior;
`harness` owns reusable assembly patterns and controlled components around that
behavior.

`conformance` may use `harness` to assemble deterministic runs, but fixture
loading, case-field interpretation, schema validation, and expected portable
behavior remain in `conformance`, `contracts/v0`, and `conformance/cases`.

Consumers should prefer importing harness APIs through the package root:

```python
from harness import MemoryRunStore, ScriptedModel, collect_events
```

Subpackages exist to keep implementation ownership clear. They are not an
invitation for cross-package consumers to bypass the root API.
