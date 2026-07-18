# Security Policy

## Reporting a Vulnerability

Use GitHub private vulnerability reporting for `Ezio2000/jharness`. Do not place
credentials, user data, exploit details, or an unpatched proof of concept in a public
issue.

Reports may cover any part of the single project, including:

- runtime lifecycle, approval, cancellation, deadlines, and checkpoint integrity;
- provider authentication, HTTP/SSE transport, error handling, and data exposure;
- tool validation, filesystem containment, command execution, and child-agent control;
- wire decoding, schema validation, persisted state, traces, and repository adapters;
- packaging, dependencies, release provenance, and build automation.

Include the affected JHarness version, operating system and Python version, a minimal
reproduction, expected and observed behavior, and an assessment of impact. Remove all
real credentials and sensitive payloads.

## Supported Releases

The latest published minor line receives security fixes. Pre-releases may be
superseded without a backport. A fix is released as one coordinated `jharness` version
covering every affected subpackage, contract, test, and advisory.

Published artifacts and tags are immutable. A compromised or materially unsafe
version is documented, yanked when appropriate, and replaced with a new version.

## Deployment Responsibility

JHarness exposes capabilities; it is not an operating-system sandbox. Hosts remain
responsible for least-privilege credentials, authorization, approval policy, workspace
boundaries, process and network isolation, durable queue ownership, fencing, secret
management, and audit retention appropriate to their threat model.

`BashTool` does not inherit the complete host environment by default. It copies only a
small platform allowlist needed to locate programs, select locale, and use temporary
directories, then applies the explicit `environment` overlay. Enabling
`inherit_environment=True` exposes every host variable, including credentials and shell
startup controls, and therefore requires the same review as passing a secret-bearing
credential set to an untrusted subprocess.

Filesystem path containment is a validation boundary, not a mount namespace. Hosts
that accept writes from mutually distrusting principals must provide a dedicated
filesystem or sandbox and must control hard-link creation. Process-tree cleanup is
best-effort within operating-system primitives: POSIX uses a new session and process
group, while Windows uses a Job Object with a tree-aware fallback. Containers must
provide signal and reaping behavior suitable for child processes, especially when the
application is PID 1.
