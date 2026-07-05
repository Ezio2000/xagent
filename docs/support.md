# Support Package

`support` owns controlled runtime support components used by tests, examples,
and conformance runners. It is intentionally separate from `harness` so the
scenario assembly layer does not become a mock and fixture surface.

## Layers

| Layer | Responsibility | Must Not Own |
| --- | --- | --- |
| `drivers` | Deterministic model and stream drivers for scripted responses, errors, timeouts, cancellation, and control actions. | Provider adapters or model protocol semantics. |
| `environment` | Runtime port fakes for stores, journals, approval policies, hooks, and related controlled host infrastructure. | Concrete production stores, approval UIs, queues, or deployment runtime. |
| `tools` | Tool fixtures, registry doubles, invocation records, and reusable controlled tool behaviors. | Production tool implementations, tool scheduling semantics, or JSON Schema validation rules. |
| `messages` | Message fixtures that delegate to `prompting` helpers. | Runtime message semantics. |
| `assertions` | Reusable behavior assertions for events, state, checkpoints, ordering, and traces. | Portable conformance contract definitions or fixture schema validation. |

## Package Boundaries

`support` may import public package roots from `kernel`, `toolkit`,
`prompting`, `diagnostics`, and `harness`. It must not be imported by lower-level
runtime source packages. `kernel`, `toolkit`, `prompting`, `modelkit`,
`diagnostics`, and `harness` must stay usable without `support`.

Consumers should prefer importing support APIs through the package root:

```python
from support import ScriptedModel, MemoryRunStore, FixtureToolRegistry
```
