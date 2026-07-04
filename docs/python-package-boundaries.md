# Python Package Boundaries

Python packages live directly under `python` and use the standard `src` layout.
For example, `python/kernel` is the workspace/distribution package root, while
`python/kernel/src/kernel` is the import package that users see as
`import kernel`. The repeated package name is intentional: the outer directory
belongs to packaging metadata and tests; the inner directory is the Python module
namespace. Do not treat it as two kernels.

## Target Layout

| Package | Import Surface | Responsibility | Must Not Own |
| --- | --- | --- | --- |
| `kernel` | `import kernel` | Execution loop, scheduler, protocol/value types, state, snapshots, resume, limits, events, hooks, store/journal/approval ports, tool/model protocols, canonical stream accumulation, capability normalization, and immutable trace payload emission. | Prompt engineering helpers, default registries, concrete providers, replay tooling, public trace object helpers, test harnesses, UI, queues, deployment, or concrete stores. |
| `toolkit` | `import toolkit` | Default tool registry, tool schema validation, concrete invocation adapter from `Tool` protocols to `ToolOutput`. | Agent loop, state machine, model adapter code, conformance runner. |
| `prompting` | `import prompting` | Message and prompt construction conveniences built on kernel message types. | Runtime state, scheduling, model/provider clients. |
| `modelkit` | `import modelkit` | Model adapter helper facade re-exporting kernel stream accumulation and capability helper functions. | Runtime loop, tool execution, prompt helpers, independent copies of kernel model helper logic. |
| `diagnostics` | `import diagnostics` | Public trace objects, trace-from-events helpers, deterministic replay validation, and diagnostics-only error types. | Running agents, invoking tools, provider adapters, persistence backends. |
| `harness` | `import harness` | Reusable tests and examples support such as scripted models and event collection. | Production runtime semantics, conformance contracts, public kernel ports. |
| `conformance` | `uv run conformance ...` | Cross-SDK fixture runner and JSON Schema validation around `contracts/v0`. | Kernel internals or package-private APIs. |

## Dependency Rules

| From | May Import | Rule |
| --- | --- | --- |
| `kernel` | External libraries from its own manifest only | No internal runtime package imports. |
| `toolkit` | `kernel` | Import only `kernel` package root, not `kernel.tools` or other private modules. |
| `prompting` | `kernel` | Import only `kernel` package root. |
| `modelkit` | `kernel` | Import only `kernel` package root. |
| `diagnostics` | `kernel` | Import only `kernel` package root. |
| `harness` | `kernel` | Import only `kernel` package root. |
| `conformance` | `kernel`, `toolkit`, `prompting`, `diagnostics` | Use public package roots only. |

The boundary tests in `python/kernel/tests/test_dependency_boundaries.py`
enforce the allowed package set, declared dependencies, retired package names,
and public-root-only cross-package imports.

These dependency rules apply to production import packages under
`python/*/src`. Package tests and examples may include workspace integration
smoke tests that combine multiple public package roots, but package-local self
tests should prefer only that package's declared runtime dependencies.

## Placement Checklist

| Question | Put It In |
| --- | --- |
| Does it change status transitions, checkpoint placement, resume semantics, tool scheduling, model call orchestration, event ordering, or host extension ports? | `kernel` |
| Does it validate or invoke a concrete collection of host tools through `ToolRegistryProtocol`? | `toolkit` |
| Does it make messages easier to construct or shape prompt text without changing runtime semantics? | `prompting` |
| Does it define canonical stream accumulation or inspect optional model capabilities used by the runtime? | `kernel` |
| Does it expose adapter-friendly imports for kernel model helpers without adding behavior? | `modelkit` |
| Does it inspect, replay, summarize, or validate traces after a run? | `diagnostics` |
| Is it a reusable fake, scripted model, collector, or test-only convenience? | `harness` |
| Is it portable behavior that every SDK should satisfy? | `contracts/v0` and `conformance/cases` |

## Public API Rule

Each package should be consumed through one package root. Cross-package imports
such as `from toolkit import ToolRegistry` are allowed; imports such as
`from toolkit.registry import ToolRegistry` are package-private and rejected by
tests outside that package. This keeps the boundary small without forcing every
package to communicate through a single oversized protocol class.
