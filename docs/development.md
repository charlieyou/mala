# Development

## Dependencies

```bash
uv sync
```

## Type Checking

mala uses strict type checking with both `ty` and `ruff`:

```bash
uvx ty check         # Type check with strict rules
uvx ruff check .     # Lint with type annotation rules
uvx ruff format .    # Format code
```

All ty rules are set to `error` level in `pyproject.toml` for maximum strictness.

## Test Coverage

Tests require 72% minimum coverage (enforced via `--cov-fail-under=72`):

```bash
uv run pytest                              # Unit + integration tests (default, excludes e2e, no coverage)
uv run pytest -m unit                      # Unit tests only
uv run pytest -m integration -n auto       # Integration tests in parallel
uv run pytest -m e2e                       # End-to-end tests (requires CLI auth)
uv run pytest -m "unit or integration or e2e"  # All tests
uv run pytest --reruns 2                   # Auto-retry flaky tests (2 retries)
uv run pytest -m integration -n auto --reruns 2   # Parallel + auto-retry
uv run pytest --cov=src --cov-fail-under=72       # Manual coverage check
```

- **Parallel execution**: Use `-n auto` for parallel test runs (pytest-xdist)
- **Flaky test retries**: Use `--reruns N` to auto-retry failed tests (pytest-rerunfailures)
- **Unit/Integration/E2E**: Use markers `unit`, `integration`, `e2e` to select categories
- **Coverage**: Not run by default; quality gate adds coverage flags automatically

## Package Structure

The codebase is organized into layered packages with enforced import boundaries (via import-linter):

```
src/
├── core/           # Models, protocols, log events (no internal dependencies)
├── domain/         # Business logic: lifecycle, evidence_check, validation, prompts
├── infra/          # Infrastructure: clients/, io/, tools/, hooks/
├── pipeline/       # Agent session pipeline, gate/review runners, run coordinator
├── orchestration/  # Orchestrator, factory, CLI support
├── cli/            # CLI entry point
├── prompts/        # Prompt templates
└── scripts/        # Utility scripts
```

**Layer dependencies** (enforced by import-linter contracts):
- `core` → (none)
- `domain` → `core`
- `infra` → `core`
- `pipeline` → `core`, `domain`, `infra`
- `orchestration` → `core`, `domain`, `infra`, `pipeline`
- `cli` → all layers

For detailed architecture documentation, see [architecture.md](architecture.md).

## Adding a New Coder Backend

Mala's per-issue implementation agent runs behind the
[`AgentProvider`](architecture.md#8-agent-provider-abstraction-coder-backend-selection)
protocol (`src/core/protocols/agent_provider.py`). To add a new coder (e.g.,
Codex, Aider) symmetric to Claude and Amp, implement the three concerns:

1. **Client** — produce an `SDKClientProtocol`-conforming client
   (`src/core/protocols/sdk.py`). The pipeline reads message events as
   duck-typed Anthropic-shaped objects (`AssistantMessage`, `ResultMessage`,
   `TextBlock`, `ToolUseBlock`, `ToolResultBlock`); synthetic dataclasses with
   matching class names + field names work without touching
   `MessageStreamProcessor`. See `src/infra/clients/amp_messages.py` for the
   pattern.
2. **Runtime builder** — implement `CoderRuntimeBuilder.build()` returning an
   opaque coder-shaped runtime (CLI args, env, config). The pipeline never
   inspects the runtime; only the matching `client_factory.create(runtime, …)`
   knows its shape. See `src/infra/clients/amp_runtime.py`.
3. **Log provider** — implement `LogProvider` (`src/core/protocols/log.py`)
   for evidence parsing. Reuse `FileSystemLogProvider` if your coder writes
   JSONL, or build a tee-based provider like
   `src/infra/clients/amp_log_provider.py` if the native log format is
   undocumented or unstable.

Bundle them in an `AgentProvider` implementation
(`src/infra/clients/<coder>_provider.py`). Add an `install_prerequisites()`
method that performs any idempotent setup (plugin install, runtime self-test);
return early if no setup is needed.

**Wiring:**

- Extend the `coder` enum in `src/infra/io/config.py` (`parse_coder`,
  `MalaConfig.coder` literal) and add yaml validation in
  `src/domain/validation/config.py`.
- Add a CLI flag if a coder-specific option is needed (e.g.,
  `--<coder>-mode`); follow the `--amp-mode` shape: a Typer callback that
  parses-and-validates at parse time, then plumbs through `MalaConfig`.
- Add the provider selection branch in `src/orchestration/factory.py` —
  this is the **only** place that reads `coder` and selects between
  providers. Pipeline modules consume only `AgentProvider`.
- Mirror the existing CLI > env > yaml > default precedence pattern for any
  new options.

**Tests:** add unit tests for the client, runtime builder, and log provider
(see `tests/unit/infra/clients/test_amp_*`). Add a fake-binary integration
test on `PATH` (`tests/integration/test_amp_provider.py` is the template).
A real-CLI smoke job is recommended for any provider whose stream format is
upstream and may drift.

## Amp Plugin Development

The Amp safety plugin (`plugins/amp/mala-safety.ts`) is the safety surface
under `--dangerously-allow-all`. It mirrors a subset of the Claude-path hooks
(`src/infra/hooks/dangerous_commands.py`,
`src/infra/hooks/locking.py::make_lock_enforcement_hook`) in TypeScript. See
[`plugins/amp/README.md`](../plugins/amp/README.md) for the full code-review
checklist.

### WIP-API Caveat

The Amp plugin API is officially **experimental** ("expect many breaking
changes"). Plugin files must start with the verbatim acknowledgment header:

```
// @i-know-the-amp-plugin-api-is-wip-and-very-experimental-right-now
```

This line **must be preserved character-for-character** through any installer
or packaging step. The installer (`AmpPluginInstaller`) ships the `.ts` file
unchanged; tests assert the header in
`tests/unit/infra/clients/test_amp_plugin_installer.py` and in the CI smoke
job.

The plugin API only loads under:

- The **official binary install** of Amp (npm install does not load plugins).
- `PLUGINS=all` in the environment (mala sets this for you in
  `AmpRuntimeBuilder`).
- A working Bun runtime (provided by the binary install).

Mala's runtime self-test detects all three failure modes — see
[architecture.md §9](architecture.md#9-safety-critical-plugin-self-test-amp).

### Local Verification

Run a Bun type/syntax check before committing plugin changes:

```bash
bun build plugins/amp/mala-safety.ts --target=bun --outfile=/tmp/mala-safety-check.js
```

A non-zero exit (or a TypeScript diagnostic) is a failure. The orchestrator
does **not** rely on a build artifact; Amp executes the `.ts` file directly
from `~/.config/amp/plugins/mala-safety/`.

### Spiking Upstream Changes

When the upstream Amp plugin API changes (it will), the path-gated CI smoke
job (`tests/smoke/test_amp_real_cli.py`) catches drift on a real Amp install.
To spike a fix locally:

1. Bump the pinned Amp CLI version in your local install.
2. Run the smoke job locally with `AMP_API_KEY` set:
   `uv run pytest -m smoke tests/smoke/test_amp_real_cli.py`.
3. Update `mala-safety.ts` and adjust the matching Python mirrors in
   `src/infra/hooks/dangerous_commands.py` /
   `src/infra/hooks/file_cache.py` in the **same commit**. The lock-file
   format and the file-write-tool name list are
   [cross-language contracts](architecture.md#lock-file-format-cross-language-contract)
   — drift between Python and TypeScript is a safety bug.
4. The plugin's content hash is the marker `version` field on
   `session.start`; mala's self-test computes the same hash from the
   installed file. Any byte-level edit therefore changes the version
   automatically — no manual semver bump needed.

### What's Out of Scope (MVP)

- `MALA_DISALLOWED_TOOLS` parity (token-saver tool denylist).
- File-write surfaces beyond `edit_file`, `create_file`, `undo_edit`,
  `apply_patch`.
- Read-cache redundancy blocking (the Claude-side `FileReadCache` hook).
- Shell-redirect / `mv` / `cp` gating — too many legitimate uses to block
  reliably without parsing target paths.

These are tracked as follow-ups in the
[Amp provider plan](../plans/2026-04-29-amp-provider-plan.md).
