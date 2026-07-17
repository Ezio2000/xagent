# Python Distribution and Package Boundaries

JHarness is one versioned uv workspace containing four independently installable PyPI
distributions. The repository is the unit of development and release; a distribution
is the unit of installation.

## Distribution Map

| PyPI distribution | Public import | Responsibility | Runtime dependencies |
| --- | --- | --- | --- |
| `jharness-kernel` | `jharness.kernel` | Runtime, immutable values, ports, checkpoints, codecs, diagnostics | None |
| `jharness-toolkit` | `jharness.toolkit` | Registry, function adaptation, schema validation, retry, circuit breaking | exact `jharness-kernel`, `jsonschema`, `referencing` |
| `jharness-models` | `jharness.models` | OpenAI, Anthropic, and DeepSeek models, HTTP/SSE, profiles, codecs | exact `jharness-kernel`, `httpx` |
| `jharness-tools` | `jharness.tools` | Filesystem, shell, interaction, and child-agent tools | exact `jharness-kernel`, `regex` |

`jharness.kernel.wire` and `jharness.kernel.diagnostics` are public kernel APIs.
Model-specific public entry points are `jharness.models.openai`,
`jharness.models.anthropic`, and `jharness.models.deepseek`.

## Dependency Graph

```text
jharness-kernel
├── jharness-toolkit
├── jharness-models
└── jharness-tools

conformance (development only) -> jharness-kernel + jharness-toolkit
```

Arrows point from a dependency to its dependents. Toolkit, models, and tools cannot
import one another. In particular, ready-to-use tools implement kernel contracts;
applications may compose them with `ToolRegistry` without making `jharness-tools`
depend on `jharness-toolkit`.

## Installation

Install only what the application uses:

```bash
uv add jharness-kernel
uv add jharness-toolkit
uv add jharness-models
uv add jharness-tools
```

Install the complete coordinated set explicitly:

```bash
uv add jharness-kernel jharness-toolkit jharness-models jharness-tools
```

Python imports use dots even though PyPI distribution names use hyphens:

```python
from jharness.kernel import Runtime
from jharness.models.openai import OpenAIChatCompletionsModel
from jharness.toolkit import ToolRegistry
from jharness.tools import ReadTool
```

## Namespace Ownership

The four wheels contribute non-overlapping portions to the implicit PEP 420
`jharness` namespace:

```text
packages/
  jharness-kernel/src/jharness/kernel/
  jharness-toolkit/src/jharness/toolkit/
  jharness-models/src/jharness/models/
  jharness-tools/src/jharness/tools/
```

There is no `jharness/__init__.py`. Each component owns its own `__init__.py` and
`py.typed`; this keeps editable workspace installs and installed wheels composable and
prevents overlapping archive files.

## Placement Rules

- Provider-neutral runtime semantics and portable wire codecs belong in kernel.
- Tool registration and JSON Schema validation belong in toolkit.
- External model transports, profiles, request/response codecs, and endpoint errors
  belong in models.
- Ready-to-use tool behavior belongs in tools.
- Conformance, contracts, tests, examples, benchmarks, and release scripts are not
  included in runtime wheels.

## Build and Release Gates

CI proves that:

- all four projects use one valid version;
- every non-kernel project pins that exact kernel version;
- four wheels and four source distributions are built;
- every wheel owns exactly one namespace portion and one nested `py.typed` marker;
- no two wheels contain the same path;
- each wheel installs and imports alone with its declared dependencies;
- the four wheels install and import together;
- internal and third-party imports match the declared dependency graph;
- one tag publishes the same verified eight archives to TestPyPI and PyPI.

No compatibility package or old import path is retained.
