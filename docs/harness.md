# Harness Package

`harness` is the open agent workflow and controlled scenario package for the
Python runtime workspace. Its job is to provide reusable workflow facades around
public SDK ports, assemble `kernel.AgentLoop` from already-provided runtime
inputs, and collect or label runtime events.

It is not a mock library, fixture package, assertion package, provider adapter,
tool pack, storage backend, UI, deployment runtime, or product application.
Controlled model drivers, runtime port fakes, tool fixtures, registry doubles,
message fixtures, and reusable behavior assertions belong in `support`.

## Layers

| Layer | Responsibility | Must Not Own |
| --- | --- | --- |
| `workflows` | High-level, provider-neutral agent workflows such as common input normalization, tool-loop execution, event streaming, pause/resume helpers, trace replay wiring, and waiting/background-task extraction. | Kernel runtime semantics, concrete provider adapters, production tools, storage implementations, UI policy, or application-specific workflow semantics. |
| `scenarios` | Low-level entry points that assemble `kernel` inputs, run controlled runtime scenarios, and return observed results. | Kernel runtime semantics, model drivers, tool fixtures, conformance case parsing, trace replay implementation, or application-specific scenario semantics. |
| `observation` | Event collection and timeline helpers for observing a run. | Diagnostics trace object semantics, replay implementation, model/tool fakes, or behavior assertions. |

## Workflow API

The high-level entry point is `AgentHarness`. It accepts public kernel ports and
toolkit tools, then creates a fresh `kernel.AgentLoop` per operation:

```python
from harness import AgentHarness

agent = AgentHarness(model=model, tools=[SearchTool(), FetchTool()])

result = await agent.run("search docs and summarize")
events = await agent.events("search docs", stream=True)
paused = await agent.run_until_pause("start external job")
resumed = await agent.resume(paused, "job completed")
traced = await agent.run_with_trace("summarize current state")
```

The workflow layer is intentionally open: hosts can inject custom model clients,
tool registries, approval policies, stores, journals, hooks, limits, and tool
schedulers through the same public ports consumed by `kernel.AgentLoop`.
`AgentHarness` may adapt a sequence of toolkit `Tool` implementations into a
default `ToolRegistry`, but custom `ToolRegistryProtocol` implementations remain
first-class.

`KernelScenario` remains available for lower-level controlled scenario assembly
when tests, conformance helpers, or debugging scripts need direct message-level
control.

## Package Boundaries

`harness` may import public package roots from `kernel`, `toolkit`,
`prompting`, and `diagnostics`. Runtime source packages must not import
`harness`; host applications and higher-level helpers may use it when they want
repeatable workflow assembly around the kernel.

`support` may use `harness` to run scenarios with controlled fakes and fixtures.
`conformance` may use `support`, but fixture loading, case-field interpretation,
schema validation, and expected portable behavior remain in `conformance`,
`contracts/v0`, and `conformance/cases`.

Consumers should prefer importing harness APIs through the package root:

```python
from harness import AgentHarness, KernelScenario, collect_events
```
