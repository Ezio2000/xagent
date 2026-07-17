# Test layout

Tests follow the public module boundaries of the single Python project:

- [`kernel`](kernel/) covers runtime, value, diagnostic, and wire behavior;
- [`toolkit`](toolkit/) covers registries and function-tool adapters;
- [`models`](models/) covers model profiles, codecs, streams, and HTTP behavior;
- [`tools`](tools/) covers the reusable filesystem, shell, interaction, and agent tools;
- [`conformance`](conformance/) validates the local contracts and runs every portable case;
- [`repository`](repository/) guards the single-project layout, public API, examples, and
  release automation.

Portable behavior lives in [`conformance/cases`](../conformance/cases/) and persistence
schemas live in [`contracts/v0`](../contracts/v0/). Tests read both trees directly; there
is no synchronized specification mirror.
