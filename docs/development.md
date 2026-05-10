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
uv run pytest -m e2e                       # End-to-end tests (requires real CLI prerequisites)
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
protocol (`src/core/protocols/agent_provider.py`). Three coders ship today
— Claude, Amp, and Codex. To add a fourth (e.g., Aider) symmetric to the
existing three, implement the three concerns:

1. **Client** — produce an `SDKClientProtocol`-conforming client
   (`src/core/protocols/sdk.py`) whose `receive_response()` yields
   `AgentEvent` values (`text` / `tool_use` / `tool_result` / `result`)
   defined in `src/core/protocols/agent_event.py`. The pipeline
   (`MessageStreamProcessor`) branches on `event.kind` only and silently
   drops messages without a recognised `kind`, so producing the wrong
   shape is a quiet failure. See
   `src/infra/clients/amp_client.py::AmpClient.receive_response` for
   the canonical pattern (Amp emits `AgentEvent`s directly). New
   coders should emit `AgentEvent`s from the client; if the underlying
   SDK yields Anthropic-shaped `AssistantMessage` / `ResultMessage`
   objects, run them through `src.core.protocols.agent_event.to_agent_events`
   inside the client (or wrap the client) so consumers always see the
   uniform event stream.
2. **Runtime builder** — implement `CoderRuntimeBuilder.build()` returning an
   opaque coder-shaped runtime (CLI args, env, config). The pipeline never
   inspects the runtime; only the matching `client_factory.create(runtime, …)`
   knows its shape. See `src/infra/clients/amp_runtime.py`.
3. **Evidence provider** — implement `EvidenceProvider`
   (`src/core/protocols/evidence.py`) for evidence parsing. Reuse
   `FileSystemLogProvider` if your coder writes JSONL, or build a tee-based
   provider like `src/infra/clients/amp_log_provider.py` if the native log
   format is undocumented or unstable.

Bundle them in an `AgentProvider` implementation
(`src/infra/clients/<coder>_provider.py`). Add an `install_prerequisites()`
method that performs any idempotent setup (plugin install, runtime self-test);
return early if no setup is needed.

**Wiring:**

- Extend the `coder` enum in `src/infra/io/config.py` (`parse_coder`,
  `MalaConfig.coder` literal) and add yaml validation in
  `src/domain/validation/config.py`.
- Add a CLI flag if a coder-specific option is needed (e.g.,
  `--<coder>-mode`); follow the `--amp-mode` / `--codex-model` shape: a
  Typer callback that parses-and-validates at parse time, then plumbs
  through `MalaConfig`. For richer per-coder option blocks, add a nested
  dataclass under `CoderOptions` (`AmpOptions`, `CodexOptions` are the
  templates) and wire env / CLI / yaml resolution in `MalaConfig.from_env`.
- Add the provider selection branch in `src/orchestration/factory.py` —
  this is the **only** place that reads `coder` and selects between
  providers. Pipeline modules consume only `AgentProvider`. The orchestrator
  no longer branches on `provider.name`; provider-owned `mcp_server_factory()`
  handles MCP factory selection.
- Mirror the existing CLI > env > yaml > default precedence pattern for any
  new options.

**Tests:** add unit tests for the client, runtime builder, and evidence
provider (see `tests/unit/infra/clients/test_amp_*` and
`tests/unit/infra/clients/test_codex_*`). Add a fake-SDK integration test
(`tests/integration/test_amp_provider.py` and
`tests/integration/test_codex_provider.py` are the templates). A real-CLI
or real-SDK e2e test is recommended for any provider whose stream format
is upstream and may drift.

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
`tests/unit/infra/clients/test_amp_plugin_installer.py`.

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

When the upstream Amp plugin API changes (it will), the real-Amp e2e test
(`tests/e2e/test_amp_real_cli.py`) catches stream-json drift on a real Amp
install. To spike a fix locally:

1. Bump the pinned Amp CLI version in your local install.
2. Run the e2e check locally:
   `uv run pytest -m e2e tests/e2e/test_amp_real_cli.py`.
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

## Codex Provider Development

The Codex provider lives under `src/infra/clients/codex_*` and the bundled
plugin under `plugins/codex/mala-safety/`. Unlike Amp, Codex's safety
surface is a Python `PreToolUse` command hook (not a TypeScript runtime
plugin), and the SDK manages the `codex app-server` subprocess in-process
— there is no CLI subprocess wrapping inside `CodexClient`.

### Experimental SDK — Pin the Version

The `codex_app_server` Python SDK is officially **experimental** ("expect
breaking changes"). Mala pins to a known-good upstream tag in
`pyproject.toml` under the `codex` extra. When the upstream release shifts
the notification or item schema:

1. Bump the pinned tag in `pyproject.toml` (`[project.optional-dependencies].codex`).
2. Run the real-Codex e2e test (gated on SDK + runtime + auth):
   `uv run pytest -m e2e tests/e2e/test_codex_real_sdk.py`.
3. Update `src/infra/clients/codex_event_adapter.py` (notification → `AgentEvent`
   mapping) and `src/infra/clients/codex_evidence_provider.py` (tee parsing)
   in lockstep with the schema change.
4. Run the fake-SDK integration suite to catch regressions in the unchanged
   mappings: `uv run pytest tests/integration/test_codex_provider.py`.

### Bundled Hook + Plugin Trust UX

The bundled `mala-safety` plugin is the only safety surface under
`sandbox: danger-full-access` + `approval_policy: never`. Two failure
modes drove the design:

- **"Hook installed but Codex didn't load it."**
  `CodexAgentProvider.install_prerequisites()` runs a runtime hook self-test
  on every Codex run; failures raise `CodexHookNotActiveError` with a
  structured `CodexHookNotActiveReason` enum
  (`HOOK_MARKER_MISSING`, `VERSION_MISMATCH`, `SCRIPT_MISSING`,
  `PLUGIN_DISABLED`, `TRUSTED_HASH_MISMATCH`, `CODEX_BINARY_MISSING`).
  The self-test drives both Codex hook events the bundled plugin declares:
  `SessionStart` proves the plugin dispatch path is enabled, and a real
  tool-call turn proves the safety-critical `PreToolUse` hook fires before
  unattended tool execution.
  See [architecture.md §10a](architecture.md#10a-codex-safety-model--bundled-hook--plugin-self-test).
- **"User skipped the trust step."** Codex's hook framework requires
  `trusted_hash` matching for the hook to fire. The provider creates a
  temporary `CODEX_HOME` for mala-launched Codex sessions and
  `CodexPluginInstaller` writes the expected hash into that isolated home's
  hook state automatically, so there is no manual trust step and no mutation
  of the user's normal `~/.codex`. If a future Codex UX change requires
  user-interactive trust acceptance, the self-test catches the resulting
  silent-dormant state and surfaces a `TRUSTED_HASH_MISMATCH` reason.

### Lock-Path Reuse — In-Process

The Codex hook (`src/infra/hooks/codex_pre_tool_use.py`) imports
`src/infra/tools/locking.lock_path` directly. This is the **advantage**
over the Amp TS plugin (which had to reimplement SHA-256 hashing in
TypeScript and tracks a cross-language lock-format contract): Codex
lock-key derivation is byte-identical to the rest of mala by construction.
Any change to the lock format in `src/infra/tools/locking.py`
automatically applies to the Codex hook on the next install.

### Per-Process Env Isolation

The hook reads `MALA_AGENT_ID`, `MALA_LOCK_DIR`, `MALA_REPO_NAMESPACE`,
and `MALA_DISALLOWED_TOOLS` from its process env. Mala MUST NOT mutate
`os.environ` in the parent process — under `--max-agents > 1`, concurrent
agents would leak each other's `MALA_AGENT_ID` to subprocesses.
`CodexRuntimeBuilder.build()` constructs a per-subprocess env dict
explicitly (`env_extra` overlays on a fresh copy of `os.environ` plus the
mandatory `MALA_AGENT_ID`, resolved `MALA_LOCK_DIR`, and
`MALA_REPO_NAMESPACE` values); the SDK's env-injection support plumbs this
through to the spawned `codex app-server` and its hook subprocess.

Env injection is the only supported transport. The previous
`~/.config/mala/agent-state/{session_id}.env` fallback was removed because
it could let hooks read stale per-session state; missing env now fails
closed at decision time.

### Local Verification

Before committing Codex changes, run the unit + integration suites:

```bash
uv run pytest tests/unit/infra/clients/test_codex_*.py
uv run pytest tests/unit/infra/hooks/test_codex_pre_tool_use.py
uv run pytest tests/integration/test_codex_*.py
```

The real-Codex e2e gate (`tests/e2e/test_codex_real_sdk.py`) is skipped
when the SDK / runtime / auth aren't available; install the `codex`
extra and configure auth locally to enable it.

### What's Out of Scope (MVP)

- `ReasoningThreadItem` content surfaced as user-visible `AgentEvent`s
  (parity with Amp's stripped-thinking stance — content is tee'd to disk
  for diagnostics).
- Codex `thread_fork` / `thread_archive` (resume-only MVP).
- Cross-coder session resume (Codex `thr_*` IDs are not interchangeable
  with Claude session IDs or Amp `T-*` thread IDs).
- Native `Thread.read(include_turns=True)` evidence (Phase F1 spike
  disconfirmed; using F3 tee fallback as the primary path. The spike
  test stays in-tree so a future SDK release that re-enables Extended
  persistence can be validated cheaply).
- Codex DevContainer baking — Codex install/auth is a user prerequisite
  (the existing DevContainer mounts `~/.codex` so an authed local install
  carries through).

These are tracked as follow-ups in the
[Codex provider plan](../plans/2026-05-07-codex-provider-plan.md).
