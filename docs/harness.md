# Harness Package

`harness` is the thin controlled kernel scenario assembly and observation
package for the Python runtime workspace. Its job is to assemble
`kernel.AgentLoop` from already-provided runtime inputs, run scenarios, and
collect or label runtime events.

It is not a mock library, fixture package, assertion package, provider adapter,
or application framework. Controlled model drivers, runtime port fakes, tool
fixtures, registry doubles, message fixtures, and reusable behavior assertions
belong in `support`.

## Layers

| Layer | Responsibility | Must Not Own |
| --- | --- | --- |
| `scenarios` | Thin entry points that assemble `kernel` inputs, run controlled runtime scenarios, and return observed results. | Kernel runtime semantics, model drivers, tool fixtures, conformance case parsing, trace replay implementation, or application-specific scenario semantics. |
| `observation` | Event collection and timeline helpers for observing a run. | Diagnostics trace object semantics, replay implementation, model/tool fakes, or behavior assertions. |

## Package Boundaries

`harness` may import public package roots from `kernel`. Runtime source packages
must not import `harness`; host applications and higher-level helpers may use it
when they want repeatable scenario assembly around the kernel.

`support` may use `harness` to run scenarios with controlled fakes and fixtures.
`conformance` may use `support`, but fixture loading, case-field interpretation,
schema validation, and expected portable behavior remain in `conformance`,
`contracts/v0`, and `conformance/cases`.

Consumers should prefer importing harness APIs through the package root:

```python
from harness import KernelScenario, collect_events
```
