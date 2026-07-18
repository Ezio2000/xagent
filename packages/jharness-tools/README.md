# jharness-tools

Ready-to-use filesystem, shell, interaction, and child-agent tools implementing the
public JHarness kernel tool contracts.

```bash
pip install jharness-tools
```

```python
from jharness.tools import GlobTool, GrepTool, ReadTool
```

Installing this distribution installs the matching `jharness-kernel` version.

## Security and Lifecycle Defaults

`BashTool` starts Bash with `--noprofile --norc`, a bounded command, bounded output,
and a minimal environment. The default environment copies only available platform
keys needed for executable lookup, locale, home, and temporary directories. The
explicit `environment` mapping overlays those values. Set `inherit_environment=True`
only when exposing all host variables—including credentials and shell startup
controls—is intended. Bash commands remain capable of every filesystem and network
operation allowed by the operating system; the workspace working directory is not a
sandbox.

Process cleanup and workspace containment use the strongest supported platform
primitives but do not create an operating-system sandbox. Deployment limitations for
process trees, hard links, containers, and mutually untrusted writers are documented in
the project [security policy](https://github.com/Ezio2000/jharness/blob/main/SECURITY.md).

Filesystem tools reject path escapes and hide their private atomic-write temporary
names. Search tools skip those names.

`GrepTool` bounds files, bytes read, matches, per-match text, and total serialized
output. `Agent` requires approval because it delegates a child run with host-selected
capabilities. The host-owned `AgentBackend` owns authorization, idempotency, depth,
supervision, and telemetry.
