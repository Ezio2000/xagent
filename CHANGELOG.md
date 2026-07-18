# Changelog

All notable changes to the JHarness Python project are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-18

### Changed

- Consolidated runtime code, model adapters, reusable tools, portable contracts,
  conformance cases, tests, examples, benchmarks, documentation, and release automation
  into one Python repository.
- Established one uv workspace that publishes the coordinated `jharness-kernel`,
  `jharness-toolkit`, `jharness-models`, and `jharness-tools` distributions.
- Established `jharness.models` as the sole model-adapter package across source paths,
  imports, documentation, tests, and release metadata, without compatibility aliases.
- Established one-way component dependencies on the kernel and exact coordinated
  kernel version pins in the other three distributions.
- Made contracts and conformance fixtures local project inputs, removing external
  synchronization and revision-pin workflows.

### Added

- OpenAI Chat Completions, Anthropic Messages, and DeepSeek model profiles.
- Ready-to-use filesystem, shell, structured interaction, and child-agent tools.
- Four-distribution namespace ownership, non-overlap, isolated-install verification,
  and coordinated release documentation.

## [0.1.0] - 2026-07-14

### Added

- Portable v0 JSON schemas and normative runtime documentation.
- Sixty-six deterministic conformance cases and a standard tool catalog.
- Provider-neutral lifecycle, model, tool, event, wire, and trace contracts.

[Unreleased]: https://github.com/Ezio2000/jharness/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Ezio2000/jharness/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Ezio2000/jharness/releases/tag/v0.1.0
