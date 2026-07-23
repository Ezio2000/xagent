# Changelog

All notable changes to the JHarness Python project are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-07-23

### Added

- Added OpenAI Chat Completions profile controls for seed support, reasoning-content
  round trips, required reasoning on tool calls, and non-null assistant tool-call
  content, plus an Anthropic profile control for redacted-thinking replay.

### Fixed

- Enabled DeepSeek OpenAI-format thinking models to use tools while omitting the
  unsupported `tool_choice` parameter and preserving the required
  `reasoning_content` and non-null assistant `content` across tool-call turns.
- Rejected unsupported DeepSeek `seed` requests and mapped its top-level
  `prompt_cache_hit_tokens` usage field to `cache_read_tokens`.
- Prevented DeepSeek Anthropic-format history from replaying unsupported
  `redacted_thinking` blocks.

## [0.3.0] - 2026-07-19

### Changed

- Replaced tuple-backed run history with structurally shared `RunHistory`, including
  persistent tool-call linkage proofs and cursor-based pending tool calls.
- Replaced `RunRepository.commit(checkpoint)` with validated incremental
  `DurableCommit` values and run-scoped checkpoint idempotency.
- Replaced full-checkpoint repository writes with shared in-memory values and v2
  incremental history chunks for SQLite, MySQL, and Redis; obsolete v1 storage is not
  read or migrated.

### Performance

- Fixed-size append runs now perform linear cumulative history, persistence, and
  pending-tool work instead of repeatedly scanning or encoding old state. Model requests
  intentionally continue to contain the complete current history.

## [0.2.2] - 2026-07-19

### Added

- Added the coordinated `jharness-repository` distribution with memory, SQLite,
  MySQL, and Redis implementations of the kernel checkpoint repository protocol.

### Changed

- Made MySQL and Redis repository drivers strict opt-in extras; the base repository
  install now depends only on the coordinated kernel and supports Memory and SQLite.

### Fixed

- Closed MySQL and SQLite repository executors when asynchronous context initialization
  fails, and made real MySQL and Redis integration tests remove their generated data.

## [0.2.1] - 2026-07-18

### Fixed

- Made release artifact counting ignore non-package files created by the build tool.

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

[Unreleased]: https://github.com/Ezio2000/jharness/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/Ezio2000/jharness/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Ezio2000/jharness/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/Ezio2000/jharness/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Ezio2000/jharness/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Ezio2000/jharness/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Ezio2000/jharness/releases/tag/v0.1.0
