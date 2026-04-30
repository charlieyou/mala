# Implementation Plan: Amp as an Alternative Main Agent Provider

## Context & Goals

- **Spec**: N/A — derived from user request "implement amp as an option for the main agent instead of claude".
- **Prior art**: `plans/2025-12-30-codex-provider.md` — a Codex provider plan that was *never implemented*. We do not assume any of its scaffolding exists; the codebase is currently 100% Claude.
- **Why Amp**: Sourcegraph's Amp is an alternative coding agent CLI with three agent modes (`smart` = Claude Opus 4.7, `rush` = Haiku 4.5, `deep` = GPT-5.4 reasoning). Including Amp diversifies coder choice without coupling model selection to the orchestration layer, and gives users a different cost/latency profile.
- **Audience**: mala operators who want their per-issue implementation agent on Amp instead of Claude, with everything else (gating, review, beads workflow, lifecycle policy) unchanged.

## Scope & Non-Goals

### In Scope
- Add Sourcegraph **Amp** (`amp --execute --stream-json`) as an alternative to Claude as the **main implementation agent** spawned per beads issue.
- Three-layer selection with the same precedence as the existing `claude_settings_sources` resolver:
  - CLI: `--coder claude|amp` and `--amp-mode smart|rush|deep`
  - Env: `MALA_CODER`, `MALA_AMP_MODE`
  - `mala.yaml`: top-level `coder:` and `coder_options.amp.mode`
  - Default: `coder: claude`, `mode: smart`
- Routes through the same `MalaOrchestrator` → `RunCoordinator` → `AgentSessionRunner` pipeline via a new `AgentProvider` protocol abstraction.
- **Fixer agents follow the main agent**: when `coder: amp` is selected, `FixerService` also spawns Amp (symmetric pipeline; one coder end-to-end).
- Bundle a TypeScript Amp plugin (`plugins/amp/mala-safety.ts`) installed idempotently to `~/.config/amp/plugins/` and loaded via `PLUGINS=all`. Plugin enforces dangerous-command blocking and `lock_acquire`-based lock-enforcement parity.
- Reuse `locking_mcp` MCP server through Amp's `--mcp-config`.
- Documentation: `README.md` Prerequisites + Usage, `docs/cli-reference.md`, `docs/project-config.md`, `docs/architecture.md`, `docs/development.md`.

### Out of Scope (Non-Goals)
- Replacing Claude as the default — `coder: claude` remains the default.
- Per-issue agent selection — every issue in a single run uses the same coder.
- Hot-swap mid-run.
- Modifying the **Cerberus / agent_sdk reviewer** path. `reviewer_type` is independent of `coder`.
- Modifying the **Epic verifier** (`src/infra/epic_verifier.py`); it uses the direct Anthropic API, not the agent SDK.
- Provider fallback / retry-with-other-coder.
- Cross-coder session resume (Amp's thread IDs ≠ Claude's session IDs).
- Surfacing Amp `thinking` / `redacted_thinking` content blocks (stripped in MVP, parity with Claude SDK default).
- `MALA_DISALLOWED_TOOLS` enforcement on Amp (deferred to follow-up; documented as a known gap).
- Devcontainer changes — Amp install/auth is documented as a user prerequisite.

## Assumptions & Constraints

### Assumptions
- Amp is invoked as a CLI subprocess: `amp --execute --stream-json --dangerously-allow-all --mcp-config '{...}' [-x | --stream-json-input]` with `AMP_API_KEY` in env.
  - `--dangerously-allow-all` matches Claude's `permission_mode=bypassPermissions`; safety relies entirely on the bundled mala-safety plugin (loaded via `PLUGINS=all`). Because the safety surface collapses to the plugin under this flag, **the plugin must actually load** — this is the safety-critical invariant.
  - Per the [Amp plugin API docs](https://ampcode.com/manual/plugin-api), plugins **only load under the official binary install**, **only with `PLUGINS=all` set**, and require a working Bun runtime. npm-installed Amp will silently never load the plugin. The orchestrator therefore performs a runtime plugin-load self-test before spawning real issue agents and fails closed if the plugin is not active.
- Amp's stream-json output emits `system | user | assistant | result` events whose content blocks (`text`, `tool_use`, `tool_result`, `thinking`) are structurally close to the Anthropic message schema. The existing `MessageStreamProcessor` already keys off duck-typed class names (`AssistantMessage`, `ResultMessage`, `TextBlock`, `ToolUseBlock`, `ToolResultBlock`) and `getattr` field reads (`text`, `id`, `name`, `input`, `tool_use_id`, `is_error`, `content`, `session_id`, `result`), so synthetic dataclasses with the same names work without touching the processor.
- Amp accepts MCP servers via `--mcp-config` (stdio launch shape) so `locking_mcp` can be reused unchanged.
- Amp has a TypeScript plugin API (`session.start`, `agent.start`, `tool.call` ≈ PreToolUse with `allow|reject-and-continue|modify|synthesize|error`, `tool.result` ≈ PostToolUse, `agent.end`). API is officially **experimental** ("expect many breaking changes"), only works in the Amp CLI (not the editor extension), and requires the `// @i-know-the-amp-plugin-api-is-wip-...` acknowledgment header.
- Tool events in stream-json output are *observational* (Amp has already executed them). Pre-tool *blocking* requires a plugin's `tool.call` handler.
- Amp's native session/thread log location and on-disk format are **undocumented** as of 2026-04. The reliable evidence source is the stream-json stdout we already capture.
- `amp threads continue` exists for resume; exact shape (flag on `--execute` vs separate subcommand) is undocumented and resolved during the implementation spike.
- `MalaConfig` lives in `src/infra/io/config.py` and uses an InitVar / private-tuple pattern for `claude_settings_sources` (`src/infra/io/config.py:159+`). New `coder` / `coder_options` fields follow the same `MalaConfig.from_env()` plumbing and overrides flow.
- `MalaEventSink` lives behind `src/core/protocols/events` with concrete impls in `src/infra/io/{base_sink,console_sink}.py`. Adding a `coder` span attribute is a non-breaking extension.

### Implementation Constraints
- **No backward-compatibility shims** (per `CLAUDE.md`): when modules are renamed or moved, update all callers; no re-export wrappers.
- **No re-exports**: Amp adapter lives in its own module space under `src/infra/clients/`; no aliasing modules for convenience.
- **Lazy SDK imports preserved**: `claude_agent_sdk` imports stay local to the Claude provider so an Amp-only run doesn't require the package. Symmetrically, the Amp adapter must not be imported when running Claude.
- **Layered architecture** (per `docs/architecture.md` and grimp rules):
  - Protocol contracts live in `src/core/protocols/`.
  - Concrete providers live under `src/infra/clients/`.
  - Pipeline-layer code (`src/pipeline/`) consumes only the protocol — no `if coder == "amp"` branches.
  - Orchestration (`src/orchestration/factory.py`) is the only place that selects between providers.
- **`agent_session_runner.py` and `fixer_service.py` stay coder-agnostic**: all coder-specific behavior is hidden behind `AgentProvider`.
- **Selection resolver mirrors `claude_settings_sources`**: same CLI > env > yaml > default precedence and same code shape (so contributors only learn one pattern).
- **Existing `mala.yaml` files remain valid**: new fields are optional with defaults; schema additions never break older configs.

### Testing Constraints
- New code includes unit tests; `tests/CLAUDE.md` rules apply (no over-mocking integration paths).
- **Path-gated CI smoke job** required: a CI job that calls real `amp --execute --stream-json` against a one-line prompt to catch upstream stream-json schema drift. Job is path-gated on `src/infra/clients/amp_*`, `plugins/amp/`, and this plan path; gated on `AMP_API_KEY` repository secret.
- **Regression coverage for Claude path**: after the `ClaudeAgentProvider` refactor, the existing Claude-path test suite must continue to pass with no behavior changes. Existing Claude integration tests are **not rewritten** — they are run in CI as the regression guard.
- **Fake-Amp integration suite**: a Python stand-in `amp` binary on `PATH` that emits canned stream-json. Drives the full per-issue lifecycle without real network/cost.

## Integration Analysis

### Existing Mechanisms Considered

| Existing Mechanism | Could Serve Feature? | Decision | Rationale |
|---|---|---|---|
| `SDKClientProtocol` (`src/core/protocols/sdk.py:28`) — async ctx mgr + `query()` + `receive_response()` + `disconnect()` | Yes | **Reuse** | Already the abstraction line between pipeline and Claude SDK. `AmpClient` conforms directly. |
| `SDKClientFactoryProtocol` (`src/core/protocols/sdk.py:75`) — leaks Claude vocab (`hooks`, `setting_sources`, `mcp_servers`, `resume`) | Partial | **Replace via `AgentProvider`** | Introduce a clean `AgentProvider` protocol bundling client + runtime + log provider. Claude vocab stays inside `ClaudeAgentProvider`; Amp vocab stays inside `AmpAgentProvider`. |
| `AgentRuntimeBuilder` (`src/infra/agent_runtime.py:73`) — fluent builder for SDK options + hooks dict + MCP | Partial | **Wrap and extend** | Hooks dict is Claude-SDK-specific. Factor a `CoderRuntimeBuilder` protocol so Claude builds a Claude-shaped runtime and Amp builds an `AmpRuntime` (CLI args, env, `--mcp-config` JSON, log path). The pipeline forwards the runtime opaquely to the matching client factory. |
| `LogProvider` protocol (`src/core/protocols/log.py:37`) + `FileSystemLogProvider` (`src/infra/io/session_log_parser.py:387`) | Yes | **Reuse + extend** | Already abstracted from SDK log format. Add `AmpLogProvider` (probe-then-tee strategy). `WAIT_FOR_LOG` lifecycle effect is reused. |
| `McpServerFactory` (`src/core/protocols/sdk.py:24`) + `locking_mcp` MCP server (`src/infra/tools/locking_mcp.py`) | Yes | **Reuse** | Amp accepts `--mcp-config`. Same stdio launch shape; same Python server. |
| `make_*_hook` family (`src/infra/hooks/`) — `dangerous_commands`, `locking`, `precompact`, `commit_guard`, etc. | Partial | **Replicate critical surfaces via TS plugin** | Bundle `plugins/amp/mala-safety.ts` to mirror **dangerous-command blocking** and **lock-enforcement** via `tool.call`. Disallowed-tools enforcement (`MALA_DISALLOWED_TOOLS`) is **out of MVP scope**. |
| `MessageStreamProcessor` (`src/pipeline/message_stream_processor.py`) — duck-typed by class name | Yes | **Reuse unchanged** | Synthetic Anthropic-shaped dataclasses ensure zero touch. |
| `mala.yaml` schema (`src/domain/validation/config.py`) | Yes | **Extend** | Add `coder:` and `coder_options:` fields with strict-enum validation. |
| `MalaConfig` + CLI in `src/cli/cli.py` (resolver pattern at `src/infra/io/config.py:90-121`, `:591-617`) | Yes | **Extend** | Add `--coder`, `--amp-mode` flags; env parsing for `MALA_CODER`, `MALA_AMP_MODE`. Mirror the existing override pattern. |
| `IdleTimeoutRetryPolicy` (`src/pipeline/idle_retry_policy.py`) and session-resume wiring | Partial | **Reuse** | Idle timeout still applies. Session-resume goes through `AgentProvider.client_factory.with_resume(session_id)` — Claude maps to SDK `resume=`, Amp maps to thread-continue. |
| `MalaEventSink` (`src/core/protocols/events`, impls in `src/infra/io/`) | Yes | **Extend** | Add `coder` attribute to relevant spans so dashboards can split by coder. Non-breaking extension. |
| `claude_settings_sources` resolver (`src/infra/io/config.py:90-121`) | Yes | **Mirror pattern** | New `coder` resolver follows the same CLI > env > yaml > default code shape. When `coder=amp`, `claude_settings_sources` is logged as ignored at info-level. When `coder=claude`, `--amp-mode` is logged as ignored. No errors — keeps the CLI safe across coder switches. |
| `src/pipeline/review_runner.py`, review clients (`src/infra/clients/agent_sdk_review.py`, `cerberus_review.py`) | No | **Leave unchanged** | Reviewer selection is independent from main coder. |
| `src/infra/epic_verifier.py` | No | **Leave unchanged** | Direct Anthropic API; not part of the agent SDK pipeline. |

### Integration Approach

Introduce an `AgentProvider` protocol bundling three pluggable concerns:

1. **`client_factory`** — produces an `SDKClientProtocol`-conforming client (Claude SDK client or `AmpClient` subprocess wrapper). Owns `with_resume(session_id)`.
2. **`runtime_builder`** — produces a coder-shaped runtime. Claude builds hooks dict + MCP servers + setting sources; Amp builds CLI args + `--mcp-config` JSON + plugin install side-effect. The runtime object is opaque to the pipeline; only the matching client knows its shape.
3. **`log_provider`** — already a protocol; Claude returns `FileSystemLogProvider`; Amp returns `AmpLogProvider`.

Selection happens once at orchestrator construction (`src/orchestration/factory.py`). The chosen provider is injected into `OrchestratorDependencies`, then threaded into `AgentSessionRunner` and `FixerService`. The Claude code path moves behind the new protocol but its observable behavior is unchanged.

## Prerequisites

- [ ] `amp` CLI installed via the **official binary install** documented at <https://ampcode.com/manual> (the install path that ships the plugin runtime). The npm package `@sourcegraph/amp` is **not supported** for `coder=amp` because the [Amp plugin API](https://ampcode.com/manual/plugin-api) only loads plugins under the binary install. Documented in `README.md` Prerequisites; **no devcontainer changes** in this scope.
- [ ] **Hard prerequisite for `coder=amp`**: the binary install path. `AmpAgentProvider.install_prerequisites()` performs a runtime self-test (see Technical Design) that fails closed with an actionable error if the binary install is missing — npm-installed Amp users get an explicit error, not a silent unguarded run.
- [ ] `AMP_API_KEY` exported in the user's shell / CI secrets.
- [ ] Bun runtime present for plugin execution (provided by the Amp binary install; mala does not install Bun separately). The self-test surfaces a Bun-missing failure as a fail-closed prerequisite error.
- [ ] `PLUGINS=all` is set on every `amp` invocation by `AmpRuntimeBuilder` (not user-managed), and the self-test verifies the plugin actually loaded under that env.
- [ ] Minimum Amp version pinned in `README.md` (documentation-level pin chosen at impl time against the schema we validate). No automatic version check unless a one-line `amp --version` parse is trivially reliable.
- [ ] `~/.config/amp/plugins/` writable (Amp's standard global plugin dir).
- [ ] CI: `AMP_API_KEY` added as a repository secret for the path-gated smoke job.
- [ ] Existing Claude test suite green before the refactor lands, so provider-abstraction regressions are easy to isolate.

## High-Level Approach

1. **Introduce `AgentProvider` and `CoderRuntimeBuilder` protocols** in `src/core/protocols/agent_provider.py`, bundling client_factory + runtime_builder + log_provider, plus an idempotent `install_prerequisites()` hook.
2. **Refactor Claude path behind the protocol**: existing `SDKClientFactory`, `AgentRuntimeBuilder`, and `FileSystemLogProvider` are repackaged as `ClaudeAgentProvider`. Behavior unchanged. Lazy imports preserved. The existing Claude integration suite is the regression guard.
3. **Wire selection**: `--coder`, `--amp-mode`, `MALA_CODER`, `MALA_AMP_MODE`, `mala.yaml coder:` and `coder_options.amp.mode:` with CLI > env > yaml > default precedence. Plumbed via `MalaConfig` → `src/orchestration/factory.py` → `OrchestratorDependencies.agent_provider` → `AgentSessionRunner` and `FixerService`.
4. **Implement `AmpAgentProvider`**: synthetic message dataclasses, `AmpClient` (subprocess + stream-json adapter conforming to `SDKClientProtocol`), `AmpRuntimeBuilder` (CLI args + `--mcp-config` + plugin-install side-effect), and `AmpLogProvider` (probe-then-tee).
5. **Bundle `plugins/amp/mala-safety.ts`** mirroring `block_dangerous_commands` and `lock_enforcement`. Installed to `~/.config/amp/plugins/` on first Amp run via `AmpPluginInstaller`. Installer creates the plugins directory with `mkdir -p` semantics if it doesn't exist (a fresh Amp install on a new machine may not have it yet) and writes the plugin file via write-temp-then-rename for concurrency safety. Plugin emits a sentinel marker on `session.start` (a JSON line on stderr like `{"mala_plugin":"loaded","version":"<hash>"}`) so the runtime self-test can prove the plugin actually loaded — content-hash on disk alone is insufficient. The plugin also emits a marker on a sentinel `tool.call` as a fallback signal, but the primary detection path is `session.start`.
6. **Plugin self-test (fail-closed, safety-critical)**: `AmpAgentProvider.install_prerequisites(repo_path, mcp_server_factory)` is invoked by the orchestrator after the `runtime_builder(...)` is available, so the self-test can spawn Amp using **the same runtime configuration real sessions use** — the same `--mcp-config`, the same `MALA_*` env injection, the same `PLUGINS=all`. The self-test passes the moment the `session.start` sentinel marker arrives on stderr (with version hash matching the installed plugin); the runtime then terminates the `amp` process before any model call, bounding self-test latency to plugin-load time and avoiding LLM cost on every `mala run --coder amp`. The `tool.call` observation path is fallback-only (kept for robustness if `session.start` stderr is buffered or suppressed in some Amp version). The result is cached in-memory keyed on `(amp_version, plugin_hash)` for the duration of the run; mala does not persist the cache across runs since the cost of a fresh self-test is bounded. If the marker is missing, the version mismatches, Bun is unavailable, `PLUGINS=all` failed to take effect, or the binary install isn't present, **the orchestrator refuses to spawn any issue agent** and exits with a clear actionable error pointing at the binary-install docs. This is the hard safety invariant under `--dangerously-allow-all`.
7. **Native-log spike during impl**: probe `~/.config/amp/` and `~/.local/share/amp/` for stable JSONL. If found, `AmpLogProvider` reads native; otherwise tee stream-json to a **per-thread, append-only** path at `~/.config/mala/amp-sessions/{thread_id}.jsonl`. The tee path is the safe default and is implemented first. Each `amp --execute --stream-json` invocation for a given thread (initial run or any resume) **appends** to the same thread-scoped file, so `AmpLogProvider.iter_events()` always sees the full event history regardless of whether Amp's resume mode emits delta-only or full-history events. Because the thread ID is not known until the first `system(init)` event arrives, the tee initially writes to a temp file and is renamed to `{thread_id}.jsonl` once the ID is captured.
8. **Resume via thread-id**: `AmpClient.with_resume(thread_id)` stores the ID; the next `query()` invokes the appropriate Amp continue mechanism (flag on `--execute` or `amp threads continue` subcommand, decided at impl time). The resumed invocation tees into the existing `{thread_id}.jsonl` file in append mode so evidence from earlier invocations remains observable to validation gates.
9. **Tests**: unit tests with mocked subprocess; integration tests with a fake `amp` binary on `PATH`; path-gated CI smoke job calling real `amp` against a one-line prompt; self-test fail-closed cases (plugin missing, version mismatch, Bun missing, `PLUGINS=all` unset, binary install absent).

## Technical Design

### Architecture

```
                       ┌──────────────────────────────┐
   CLI / env / yaml ───┤ MalaConfig.coder + coder_opts │
                       └──────────────┬───────────────┘
                                      ▼
                       ┌──────────────────────────────┐
                       │ src/orchestration/factory.py  │
                       │  picks AgentProvider          │
                       │  (calls install_prerequisites)│
                       └──────────────┬───────────────┘
                                      │
              ┌───────────────────────┴────────────────────────┐
              ▼                                                ▼
   ┌──────────────────────────┐                ┌──────────────────────────┐
   │ ClaudeAgentProvider      │                │ AmpAgentProvider         │
   │  - client_factory        │                │  - client_factory        │
   │  - runtime_builder       │                │  - runtime_builder       │
   │  - log_provider (FS)     │                │  - log_provider          │
   │  - install_prereq() noop │                │  - install_prereq()      │
   └─────────────┬────────────┘                │     copies plugin        │
                 │                             └─────────────┬────────────┘
                 ▼                                           ▼
       ClaudeSDKClient                                   AmpClient (subprocess)
       ├ hooks dict                                     ├ amp --execute --stream-json
       ├ MCP servers                                    ├ --dangerously-allow-all
       └ ~/.claude/projects/...                         ├ --mcp-config '{...}'
                                                        ├ env: PLUGINS=all, AMP_API_KEY,
                                                        │      MALA_AGENT_ID,
                                                        │      MALA_LOCK_DIR,
                                                        │      MALA_REPO_NAMESPACE
                                                        └ tee → ~/.config/mala/amp-sessions/{thread_id}.jsonl
                                                          (per-thread, append-only across resumes)

  Both feed into the same AgentSessionRunner / MessageStreamProcessor / gate / review
  / lifecycle / FixerService pipeline. FixerService receives the same provider, so
  fixers spawn whichever coder the main run uses.
```

- **Selection switch**: `src/orchestration/factory.py::create_orchestrator()` reads `MalaConfig.coder`, instantiates the appropriate `AgentProvider`, and injects it into `OrchestratorDependencies`. `AgentProvider.install_prerequisites()` is invoked once before the first session of a run.
- **Pipeline is coder-agnostic**: `AgentSessionRunner`, `RunCoordinator`, and `FixerService` consume `AgentProvider` (and via it `SDKClientProtocol` + `LogProvider`) without knowing which coder is active.
- **Claude code path** remains functionally identical: same `ClaudeSDKClient`, same hooks, same log paths. Only the wrapping has changed.
- **Plugin invariant (safety-critical, fail-closed)**: orchestrator startup verifies (a) `mala-safety.ts` exists at `~/.config/amp/plugins/` with expected content hash *and* (b) a one-shot **runtime plugin self-test** confirms the plugin actually loaded under `PLUGINS=all` (sentinel marker observed within timeout, version hash matches installed plugin). If either check fails — npm-installed Amp, missing Bun, `PLUGINS=all` unset, mismatched version, or any other silent-load failure — the Amp run aborts with a clear actionable error before any real issue agent is spawned. Hash-only verification is insufficient and is explicitly rejected as the sole gate.

### Data Model

```python
# src/infra/io/config.py
@dataclass(frozen=True)
class AmpOptions:
    mode: Literal["smart", "rush", "deep"] = "smart"


@dataclass(frozen=True)
class CoderOptions:
    amp: AmpOptions = field(default_factory=AmpOptions)


@dataclass(frozen=True)
class MalaConfig:
    # ... existing fields ...
    coder: Literal["claude", "amp"] = "claude"
    coder_options: CoderOptions = field(default_factory=CoderOptions)
```

**Resolution precedence** (mirrors the existing `claude_settings_sources` resolver pattern at `src/infra/io/config.py:90-121` and `:591-617`):

| Setting | CLI | Env | YAML | Default |
|---|---|---|---|---|
| Coder | `--coder amp` | `MALA_CODER=amp` | `coder: amp` | `claude` |
| Amp mode | `--amp-mode deep` | `MALA_AMP_MODE=deep` | `coder_options.amp.mode: deep` | `smart` |

YAML shape:

```yaml
coder: amp
coder_options:
  amp:
    mode: smart
```

- CLI > env > yaml > default.
- Invalid values fail validation (`src/domain/validation/config.py`) **before** any agent process starts.
- `coder_options.amp.mode` is **only consulted** when `coder == "amp"`. If set with `coder == "claude"`, log info-level "ignored — coder is claude". No error.
- Likewise, when `coder == "amp"`, `claude_settings_sources` is logged as ignored at info-level. No error.
- Existing `mala.yaml` files without `coder:` remain valid and default to Claude.

**Amp runtime state** (`src/infra/clients/amp_runtime.py`):

```python
@dataclass(frozen=True)
class AmpRuntime:
    cwd: Path
    env: Mapping[str, str]                    # PLUGINS=all, AMP_API_KEY passthrough,
                                              # MALA_AGENT_ID, MALA_LOCK_DIR,
                                              # MALA_REPO_NAMESPACE (consumed by
                                              # mala-safety.ts for lock-ownership check)
    argv: tuple[str, ...]                     # complete argv including --execute --stream-json
    mcp_config: dict[str, object]             # forwarded to --mcp-config
    mode: Literal["smart", "rush", "deep"]
    log_path: Path                            # where mala tees stream-json
    resume_thread_id: str | None = None
```

**Amp client options** (`src/infra/clients/amp_client.py`):

```python
@dataclass(frozen=True)
class AmpClientOptions:
    cwd: Path
    env: Mapping[str, str]
    argv: Sequence[str]
    log_path: Path
    thread_id: str | None
    extra_cli_args: tuple[str, ...]           # always includes --dangerously-allow-all
```

### API/Interface Design

**`AgentProvider` protocol** (new, `src/core/protocols/agent_provider.py`):

```python
@runtime_checkable
class CoderRuntimeBuilder(Protocol):
    def build(self) -> object:
        """Return an opaque coder-shaped runtime; consumed only by the matching client_factory."""
        ...


@runtime_checkable
class AgentProvider(Protocol):
    """Encapsulates a coder backend (Claude or Amp).

    Bundles three pluggable concerns:
      - client_factory:  produces SDKClientProtocol-conforming clients
      - runtime_builder: produces a coder-shaped runtime
      - log_provider:    produces a LogProvider for evidence parsing
    """

    name: Literal["claude", "amp"]
    client_factory: SDKClientFactoryProtocol
    log_provider: LogProvider

    def runtime_builder(
        self,
        repo_path: Path,
        agent_id: str,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> CoderRuntimeBuilder: ...

    def install_prerequisites(
        self,
        repo_path: Path,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> None:
        """Idempotent setup invoked once per run before first session.

        Takes the same context the runtime_builder takes (minus agent_id, which
        is per-session) so the Amp self-test exercises the *same* runtime path
        real sessions use — same --mcp-config, same MALA_* env, same PLUGINS=all.
        The orchestrator constructs a synthetic agent_id ("amp-selftest") for
        the self-test runtime; it never reaches a real beads issue.

        Claude impl: no-op (signature still required by the protocol).
        Amp impl:
          1. mkdir -p ~/.config/amp/plugins/ (idempotent; first-time users may
             not have the directory yet).
          2. Copy plugins/amp/mala-safety.ts there using write-temp-then-rename
             for concurrent-run safety.
          3. Verify installed file's content hash matches the bundled copy.
          4. Run a runtime plugin-load self-test (fail-closed, safety-critical):
             build a self-test AmpRuntime using runtime_builder(repo_path,
             "amp-selftest", mcp_server_factory=...). Spawn `amp --execute
             --stream-json --dangerously-allow-all` with that runtime's env
             (PLUGINS=all + AMP_API_KEY + MALA_* lock vars + os.environ
             passthrough — see AmpRuntimeBuilder env composition) and the
             runtime's --mcp-config. Pass when the session.start sentinel
             marker `{"mala_plugin":"loaded","version":"<hash>"}` arrives on
             stderr within a bounded timeout AND the version matches the
             installed plugin's hash; the runtime terminates `amp` immediately
             on detection (before any LLM call) to bound self-test latency to
             plugin-load time. The tool.call observation path is fallback-only
             if session.start stderr is buffered. Otherwise raise a fail-closed
             AmpPluginNotActiveError naming the most likely cause:
               - npm-installed Amp (binary install required)
               - Bun runtime missing
               - PLUGINS=all not honored
               - plugin version mismatch
               - amp binary missing from PATH
             The orchestrator refuses to spawn any issue agent in this case.
        """
```

`CoderRuntimeBuilder.build()` returns an opaque object; the pipeline never inspects it — it forwards it to the matching `client_factory.create(runtime, ...)`. Concrete return types are `ClaudeRuntime` (wraps existing `AgentRuntimeBuilder` output) and `AmpRuntime`.

**`AmpRuntimeBuilder` env injection (lock-ownership plumbing)**: `AmpRuntime.env`
is built as a **copy of `os.environ` with overlays applied** — it is *never* a
bare dict, because the spawned `amp` subprocess must inherit `PATH` (to find
`bun`, `git`, etc.), `HOME` (for `~/.config/amp/plugins/` discovery), `TMPDIR`,
locale vars, and any other ambient configuration. The builder mirrors the
Claude path's `with_env()` shape (`src/infra/agent_runtime.py:180-202`, which
itself does `{**os.environ, ...overlays}`) and overlays:

| Env var | Value | Consumer |
|---|---|---|
| `PATH` | `f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}"` (same prepend the Claude path uses) | Amp CLI to find `bun`, plus mala-shipped scripts |
| `PLUGINS` | `all` | Amp CLI (loads bundled plugin) |
| `AMP_API_KEY` | passthrough from `os.environ` (already inherited; called out for emphasis) | Amp CLI auth |
| `MALA_AGENT_ID` | the per-issue `agent_id` (same value the Claude path passes as `AGENT_ID`) | `mala-safety.ts` lock-ownership check |
| `MALA_LOCK_DIR` | `str(get_lock_dir())` (same accessor the Claude path uses for `LOCK_DIR`) | `mala-safety.ts` lock-file lookup |
| `MALA_REPO_NAMESPACE` | `str(repo_path)` (same value the Claude path passes as `REPO_NAMESPACE`) | `mala-safety.ts` lock-key computation |
| `MCP_TIMEOUT` | `"300000"` (same as the Claude path) | MCP server bootstrap timeout |

The `MALA_*` prefix is intentional — it disambiguates from the unprefixed Claude-path
names (`AGENT_ID`, `LOCK_DIR`, `REPO_NAMESPACE`), since both Claude and Amp may
co-exist in the same shell during dev/test, and the Amp plugin runs in Amp's Bun
process where it could otherwise see stale Claude-side env. The Python locking
module's source of truth for the lock dir (`MALA_LOCK_DIR` accessor at
`src/infra/tools/env.py`) is unchanged; the new env name aligns with that accessor.

**`AmpClient` sketch** (`src/infra/clients/amp_client.py`, conforms to `SDKClientProtocol`):

```python
class AmpClient:
    """Implements SDKClientProtocol via amp --execute --stream-json subprocess."""

    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, *exc) -> None:
        # SIGTERM → grace → SIGKILL (mirrors existing subprocess pattern)
        ...

    async def query(self, prompt: str, session_id: str | None = None) -> None:
        # Build cmd: amp --execute --stream-json --dangerously-allow-all
        #            --mcp-config '<json>'
        #            [--thread-id <id>  OR  via amp threads continue <id>]
        #            [--mode <smart|rush|deep>]   (exact flag confirmed during spike)
        # Spawn subprocess with PLUGINS=all in env. Write prompt to stdin and close.
        ...

    def receive_response(self) -> AsyncIterator[object]:
        # For each line of stdout JSON:
        #   system(init)         → capture self._session_id (T-...); tee header to log_path
        #   assistant.message    → synthesize AssistantMessage(content=[TextBlock|ToolUseBlock])
        #                          (thinking blocks stripped in MVP)
        #   user.message         → synthesize AssistantMessage with ToolResultBlock(s)
        #                          (MessageStreamProcessor reads ToolResultBlock by class
        #                           name regardless of which message contains it)
        #   result(success|error)→ synthesize ResultMessage(session_id, result)
        # Capture stderr to bounded buffer; classify auth errors.
        # Tee every event line to log_path so AmpLogProvider can parse evidence.
        # Unknown event types: log warn-level, do not crash.
        ...

    async def disconnect(self) -> None: ...

    # Coder-specific accessors used by AmpAgentProvider:
    @property
    def session_id(self) -> str | None: ...    # Amp's "T-..." thread id
    @property
    def thread_id(self) -> str | None: ...     # alias for session_id when Amp
    def get_stderr(self) -> str: ...
    def is_auth_error(self) -> bool: ...
```

**Synthetic message objects** (`src/infra/clients/amp_messages.py`): lightweight dataclasses named exactly `AssistantMessage` / `ResultMessage` / `TextBlock` / `ToolUseBlock` / `ToolResultBlock`. Field names match what `MessageStreamProcessor` reads (`text`, `id`, `name`, `input`, `tool_use_id`, `is_error`, `content`, `session_id`, `result`). This is the keystone that lets the existing pipeline consume Amp output without modification.

**Amp stream-json mapping**:

| Amp Event | Adapter Output | Notes |
|---|---|---|
| `system` / `init` | Capture session/thread id; tee raw event | No emitted message unless lifecycle expects one. |
| `assistant` w/ `text` | `AssistantMessage(content=[TextBlock])` | Preserve text for transcript. |
| `assistant` w/ `tool_use` | `AssistantMessage(content=[ToolUseBlock])` | Preserve `id`, `name`, `input`. |
| `assistant` w/ `thinking` / `redacted_thinking` | No user-visible block in MVP | Tee for diagnostics; not surfaced. |
| `user` w/ `tool_result` | `AssistantMessage(content=[ToolResultBlock])` | Matches processor class-name behavior. |
| `result(success)` | `ResultMessage(session_id=..., result=...)` | Normal completion. |
| `result(error_during_execution)` / `result(error_max_turns)` | `ResultMessage(...)` + error classification | Maps to existing failure/retry policy. |
| Malformed JSON | Provider error with stderr/stdout context | Fail per existing fatal error policy. |

**Resume**: `AmpClient.with_resume(thread_id)` stores the ID; the next `query()` adds whatever continuation argv shape the spike confirms. If thread continuation is unstable, the implementation logs a warning and falls back to a fresh thread plus prompt accumulation (parity with the Codex plan's stance), which preserves liveness at the cost of context. Tests are parametrized across both candidate argv shapes.

**Plugin behavior** (`plugins/amp/mala-safety.ts`):

```text
plugins/amp/mala-safety.ts
  - session.start:
      Read MALA_AGENT_ID, MALA_LOCK_DIR, MALA_REPO_NAMESPACE from process.env
      and cache for the session. If any are missing, the plugin enters fail-closed
      mode: every tool.call returning reject-and-continue with an explanatory message.
      (Fail-closed parity with the Claude path's hook, which is wired only when the
      orchestrator owns these values.)
  - tool.call for shell/bash-like commands:
      reject dangerous commands matching existing src/infra/hooks/dangerous_commands.py policy
  - tool.call for file-write tools (Amp's edit_file, create_file, and any
    Edit/Write/MultiEdit-equivalent — exact tool name list confirmed during the
    plugin spike, with parity to Python's FILE_WRITE_TOOLS / FILE_PATH_KEYS in
    src/infra/hooks/file_cache.py):
      LOCK OWNERSHIP CHECK (mirrors make_lock_enforcement_hook in
      src/infra/hooks/locking.py):
        1. Extract the target file path from the tool input using the same
           tool-name → path-key mapping as Python's FILE_PATH_KEYS.
        2. Build the lock key: "<MALA_REPO_NAMESPACE>:<canonical_path>", where
           canonical_path is realpath(path) when path is absolute, or
           realpath(MALA_REPO_NAMESPACE / path) when relative — matching
           src/infra/tools/locking._lock_key + _canonicalize_path. For non-existent
           paths, walk up to the first existing ancestor and resolve symlinks there
           (mirrors _resolve_with_parents).
        3. Compute SHA-256(lock_key) and take the first 16 hex chars (matches
           lock_path() in src/infra/tools/locking.py).
        4. Read MALA_LOCK_DIR/<hash>.lock. The file format is:
              line 1: agent_id (single line, trailing newline; produced by try_lock)
           If the file does not exist → reject-and-continue (no lock held).
           If the file exists and its first line == MALA_AGENT_ID → allow.
           If the file exists and the first line ≠ MALA_AGENT_ID → reject-and-continue
           (held by another agent).
        5. The companion .meta file (canonical filepath) is for diagnostics only;
           the .lock file's agent_id is the source of truth for ownership.
      Reject reasons mirror the Python hook's deny messages so user-visible UX is
      consistent across coders.
  - tool.result:
      optional diagnostics only
  - intentionally omits MALA_DISALLOWED_TOOLS support in MVP (tracked as follow-up)
  - includes the // @i-know-the-amp-plugin-api-is-wip-... acknowledgment header
```

**Tools write-protected in MVP**: Amp's file-write tool names (e.g. `edit_file`,
`create_file`, plus any tools Amp exposes that map to Anthropic-shaped
`Edit`/`Write`/`MultiEdit` semantics) — the exact name list is confirmed during the
plugin spike against a real Amp install and mirrors Python's `FILE_WRITE_TOOLS`. Any
file-write surface not enumerated in this list is **not** blocked by the plugin in
MVP and is deferred to the same follow-up that adds `MALA_DISALLOWED_TOOLS` parity.

**Lock-file format is a cross-language contract**: the plugin parses the same
`<hash>.lock` file format that `src/infra/tools/locking.py::try_lock` writes
(SHA-256-prefix filename + `<agent_id>\n` body). Any change to that format must be
made in both the Python locking module and the TS plugin in the same commit. The
plugin treats the format as read-only — it never writes lock files; lock acquisition
goes through the `lock_acquire` MCP tool exactly as in the Claude path.

**`AmpLogProvider` sketch** (`src/infra/clients/amp_log_provider.py`, conforms to
`LogProvider`):

```python
class AmpLogProvider:
    """LogProvider for Amp; aggregates evidence across all invocations of a thread.

    Tee path is per-thread and append-only: every `amp --execute --stream-json`
    invocation for the same thread (initial run + any resumes triggered by idle
    timeout, gate failure, or review-issue retries) appends its events to
    `~/.config/mala/amp-sessions/{thread_id}.jsonl`. Validation evidence — Bash
    tool_use events for lint/test/typecheck commands — therefore persists across
    Amp thread continuations regardless of whether Amp's resume mode emits
    delta-only or full-history events on stream-json.
    """

    def get_log_path(self, thread_id: str) -> Path:
        """Return the thread-scoped tee path (or the discovered native path).

        For the tee implementation: `~/.config/mala/amp-sessions/{thread_id}.jsonl`.
        Path is stable across resume invocations; AmpClient opens it in append
        mode every time it spawns the amp subprocess.
        """
        ...

    def iter_events(self, thread_id: str) -> Iterator[LogEvent]:
        """Read the full JSONL file (all invocations) and yield events in order.

        Tolerates malformed / partial trailing lines (existing FileSystemLogProvider
        tolerance pattern), so a crash mid-event in one invocation does not
        corrupt evidence parsing on resume.
        """
        ...
```

**Tee bootstrap (first-run thread ID unknown)**: `AmpClient` does not know the
thread ID until the first `system(init)` event arrives on stdout. The tee
therefore starts in a temp file (e.g.,
`~/.config/mala/amp-sessions/.pending-{pid}.jsonl`) and is renamed to
`{thread_id}.jsonl` atomically once the `system(init)` event is observed and
the ID captured. If `{thread_id}.jsonl` already exists (resume case), the temp
file is appended to it and unlinked. The temp prefix (e.g., `.pending-`) is
ignored by `AmpLogProvider`'s thread-id-keyed scan so half-written first-run
files never collide with thread-keyed reads.

**Cross-invocation evidence guarantee**: because the tee path is keyed by
`thread_id` and opened in append mode, the second (resumed) invocation does not
overwrite the first. `AmpLogProvider.iter_events()` reads the full file and
therefore observes Bash tool_use events from *every* invocation of the thread,
which is the contract validation gates rely on. This holds whether Amp emits
delta-only events or full-history events on resume; the plan tolerates either
shape.

### File Impact Summary

```
# Wiring & config
src/cli/cli.py                                            Exists  Add --coder, --amp-mode flags; pass through CLI overrides
src/infra/io/config.py                                    Exists  MalaConfig.coder + CoderOptions.amp; env parse; resolver mirroring claude_settings_sources
src/domain/validation/config.py                           Exists  mala.yaml schema for coder + coder_options + strict enum
src/orchestration/factory.py                              Exists  Instantiate AgentProvider, call install_prerequisites(), inject into deps
src/orchestration/orchestration_wiring.py                 Exists  Plumb provider through dependencies
src/orchestration/run_config.py                           Exists  Carry coder choice into per-run config
src/orchestration/types.py                                Exists  OrchestratorDependencies includes agent_provider

# Pipeline (kept coder-agnostic)
src/pipeline/agent_session_runner.py                      Exists  Consume AgentProvider instead of SDKClientFactory
src/pipeline/fixer_service.py                             Exists  Consume AgentProvider so fixers follow main coder
src/pipeline/run_coordinator.py                           Exists  Update if it constructs clients directly
src/pipeline/idle_retry_policy.py                         Exists  No code change expected; tested for both coders

# Telemetry
src/infra/io/base_sink.py                                 Exists  Add `coder` span attribute (low-risk extension)
src/infra/io/console_sink.py                              Exists  Surface `coder` in console output where helpful

# Claude path (refactor behind protocol; behavior unchanged)
src/infra/agent_runtime.py                                Exists  Wrap inside ClaudeAgentProvider; expose CoderRuntimeBuilder
src/infra/sdk_adapter.py                                  Exists  Expose ClaudeSDKClient via ClaudeAgentProvider.client_factory
src/infra/io/session_log_parser.py                        Exists  FileSystemLogProvider exposed via ClaudeAgentProvider

# Amp path (new)
src/infra/clients/amp_client.py                           New     AmpClient + AmpClientOptions
src/infra/clients/amp_messages.py                         New     Synthetic Anthropic-shaped message dataclasses
src/infra/clients/amp_runtime.py                          New     AmpRuntime + AmpRuntimeBuilder
src/infra/clients/amp_log_provider.py                     New     Tee'd stream-json log reader (MVP) + native-log probe
src/infra/clients/amp_provider.py                         New     AmpAgentProvider (binds the four above)
src/infra/clients/claude_provider.py                      New     ClaudeAgentProvider (binds existing pieces)
src/infra/clients/amp_plugin_installer.py                 New     Idempotent copy to ~/.config/amp/plugins/ + content-hash verify

# Plugin (TypeScript, shipped in repo)
plugins/amp/mala-safety.ts                                New     tool.call hook for dangerous-cmd + lock-enforce
plugins/amp/package.json                                  New     Plugin metadata; declares Bun runtime
plugins/amp/README.md                                     New     Plugin docs + version note + WIP-API caveat

# Protocols (new contracts in core)
src/core/protocols/agent_provider.py                      New     AgentProvider, CoderRuntimeBuilder protocols
src/core/protocols/sdk.py                                 Exists  Trim Claude-vocab leak from SDKClientFactoryProtocol post-refactor if cleanup is straightforward

# Tests
tests/unit/infra/clients/test_amp_client.py               New     Subprocess mocking, JSONL parsing, resume, lifecycle, auth errors
tests/unit/infra/clients/test_amp_messages.py             New     Synthetic message shape verification + processor round-trip
tests/unit/infra/clients/test_amp_runtime.py              New     CLI args + env + mcp_config assembly
tests/unit/infra/clients/test_amp_log_provider.py         New     Tee read + native-log probe fallback
tests/unit/infra/clients/test_amp_plugin_installer.py     New     Idempotency, write-temp-then-rename, concurrent install, stale replace
tests/unit/infra/clients/test_amp_plugin_self_test.py     New     Plugin-load self-test fail-closed cases (npm-install, Bun missing, PLUGINS=all unset, version mismatch, amp missing)
tests/unit/infra/clients/test_amp_provider.py             New     AgentProvider conformance (isinstance + protocol shape)
tests/unit/infra/clients/test_claude_provider.py          New     ClaudeAgentProvider conformance + lazy-import regression
tests/unit/orchestration/test_factory_provider_selection.py New   CLI/env/yaml precedence wiring at factory level
tests/unit/cli/test_coder_flag.py                         New     --coder, --amp-mode parsing + env precedence
tests/unit/infra/io/test_coder_config.py                  New     MalaConfig resolver: CLI > env > yaml > default; invalid values
tests/unit/domain/validation/test_coder_schema.py         New     mala.yaml validation for coder + coder_options
tests/integration/test_amp_provider.py                    New     Fake `amp` binary on PATH end-to-end (per-issue lifecycle)
tests/integration/test_coder_selection.py                 New     CLI > env > yaml > default precedence end-to-end
tests/integration/test_fixer_follows_coder.py             New     Fixer spawns Amp when coder=amp
tests/integration/test_lock_mcp_via_amp.py                New     locking_mcp round-trip through Amp's --mcp-config
tests/integration/test_amp_lock_enforcement.py            New     mala-safety.ts lock-ownership check: allow/no-lock/wrong-agent/wrong-namespace/env-missing/format-drift/canonicalization parity
tests/smoke/test_amp_real_cli.py                          New     CI smoke job (path-gated, AMP_API_KEY-gated)

# Docs
README.md                                                 Exists  Prerequisites: add Amp subsection; Usage: --coder examples
docs/cli-reference.md                                     Exists  Document --coder, --amp-mode, MALA_CODER, MALA_AMP_MODE
docs/project-config.md                                    Exists  Document `coder:` and `coder_options.amp.mode:`
docs/architecture.md                                      Exists  AgentProvider abstraction + provider triple diagram
docs/development.md                                       Exists  How to develop / extend providers + plugin dev notes
```

> `.devcontainer/devcontainer.json` is intentionally **not** in this list — Amp install/auth is documented as a user prerequisite per the interview decision.

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes

- **Plugin API is officially WIP** ("expect many breaking changes"). Mitigations:
  - Pin Amp version in `README.md`; document the tested-against version range.
  - Path-gated CI smoke job runs against a real Amp install to catch breakage early.
  - Required acknowledgment header (`// @i-know-the-amp-plugin-api-is-wip-...`) included verbatim and pinned in `plugins/amp/README.md`.
  - Plugin scope deliberately small (only dangerous-cmd + lock-enforce) so it is easier to maintain across upstream changes.
  - On startup, `AmpAgentProvider.install_prerequisites()` checks plugin file presence + content hash; if mismatch, refresh.

- **Plugin missing or tampered at runtime**. If `~/.config/amp/plugins/` is purged or modified between runs, file-write protection collapses to MCP-only. Mitigations:
  - `install_prerequisites()` runs every run (idempotent).
  - **Startup invariant check**: orchestrator refuses to start an Amp run if `mala-safety.ts` is missing or content hash mismatches after install. Raises a clear actionable error rather than running `--dangerously-allow-all` with no replacement gating.

- **Silent plugin-load failure under `--dangerously-allow-all` (safety-critical)**. The Amp plugin API only loads under the binary install with `PLUGINS=all` set and a working Bun runtime. npm-installed Amp, Bun missing, or `PLUGINS=all` unset all cause the plugin to silently never load — leaving an unguarded `--dangerously-allow-all` happy path with no replacement gating. **A content-hash check on the installed `.ts` file does not prove the plugin loaded.** Mitigations:
  - Binary install is a documented hard prerequisite for `coder=amp`.
  - `AmpAgentProvider.install_prerequisites()` runs a **runtime plugin-load self-test** before any issue agent is spawned: a one-shot `amp --execute --stream-json` invocation with a sentinel prompt; pass only if a sentinel marker (with version hash) is observed within a bounded timeout.
  - **Fail closed**: any self-test failure (missing marker, version mismatch, binary missing, Bun missing, `PLUGINS=all` ignored) aborts the run with `AmpPluginNotActiveError` naming the likely cause and pointing at the binary-install docs.
  - This is documented as the safety-critical invariant for `coder=amp`.

- **Lock-file format is a cross-language contract**. The `<hash>.lock` file
  format (SHA-256-prefix filename + `<agent_id>\n` body, paired with a `<hash>.meta`
  diagnostics file) is now consumed by *two* code paths: Python
  (`src/infra/tools/locking.py` writes; `src/infra/hooks/locking.py::make_lock_enforcement_hook`
  reads) and TypeScript (`plugins/amp/mala-safety.ts` reads). Any change to the
  filename hash, the lock-key derivation, the body format, or the lock-dir layout
  must be made in lockstep across both languages in the same commit. Mitigations:
  - Lock-file format documented in `docs/architecture.md` as a stable contract.
  - TS plugin parses the file with a single small helper that mirrors the Python
    `parse_lock_file` + `lock_path` shape; the helper is unit-tested against
    fixtures generated by the real Python `try_lock` so format drift is caught.
  - Open Question / follow-up: whether to extract the lock-key + filename hashing
    into a tiny shared spec file (e.g., a JSON-described format under `docs/`)
    that both implementations cite, to reduce drift risk further.

- **Lock-ownership env vars missing or stale in Amp's process**. If
  `MALA_AGENT_ID`, `MALA_LOCK_DIR`, or `MALA_REPO_NAMESPACE` are not propagated
  into the Amp subprocess (e.g., a builder bug, or env stripping by Amp itself),
  the plugin would have no way to compute lock keys. Mitigations:
  - Plugin enters fail-closed mode on `session.start` if any of the three vars
    are unset, rejecting all subsequent file-write `tool.call`s with a clear
    "lock-ownership env missing" message rather than silently allowing writes.
  - Unit test in `test_amp_runtime.py` asserts these three vars are always
    present in `AmpRuntime.env`.
  - Plugin self-test (existing fail-closed gate) exercises a sentinel write
    path that depends on the env, indirectly proving the env reaches the plugin.

- **Concurrent runs sharing `~/.config/amp/plugins/`**. Two mala runs writing the plugin simultaneously is benign with write-temp-then-rename. Tests cover concurrent install.

- **Stream-json schema drift**. Mitigations:
  - Tests assert exact field paths (`event.type`, `event.message.content[].type`, `event.session_id`, etc.) so failures are loud.
  - CI smoke job catches drift on a real Amp install.
  - `AmpClient` logs unrecognized event types at warn-level rather than crashing on unknown `type:`.

- **Native log undocumented**. Implementation commits to *probe-then-tee*: probe at install time; if no stable JSONL emerges, tee stream-json to `~/.config/mala/amp-sessions/{thread_id}.jsonl`. Tee path is the safe default and is implemented first.

- **Log stitching across Amp thread continuation (validation-evidence persistence)**. When a session is resumed (idle timeout, gate failure, review-issue retry) via the captured thread ID, Amp may stream only the *delta* of new events on the resumed invocation rather than re-emitting the full thread history. If the tee'd JSONL were keyed per-invocation, the resumed file would lack earlier tool calls (lint/test/typecheck) and the gate would fail to find required evidence. Mitigations:
  - Tee path is **per-thread, append-only** at `~/.config/mala/amp-sessions/{thread_id}.jsonl`. Every `amp --execute --stream-json` invocation for the same thread appends to the same file. `AmpLogProvider.iter_events()` reads the whole file regardless of which invocation produced which events.
  - The strategy tolerates both possible Amp behaviors: full-history-on-resume (events get duplicated in the file but parsing is still correct since evidence is presence-based, not count-based) and delta-only-on-resume (the original events remain in the file from the first invocation).
  - The implementer-time spike confirms which shape Amp emits and documents the observed behavior in `docs/architecture.md`. The plan does not block on the answer.

- **First-run tee bootstrap (thread ID unknown until `system(init)`)**. The tee target path depends on the thread ID, but the ID is not known until the first `system(init)` event arrives. Mitigations:
  - `AmpClient` initially writes to a temp file (e.g., `.pending-{pid}.jsonl` under `~/.config/mala/amp-sessions/`) and atomically renames to `{thread_id}.jsonl` once the ID is captured.
  - If `{thread_id}.jsonl` already exists (resume case), the pending file's contents are appended to it and the pending file unlinked.
  - If the process crashes before `system(init)`, the orphan `.pending-*` file is harmless: `AmpLogProvider`'s thread-id-keyed scan ignores the `.pending-` prefix. A periodic cleanup of stale pending files is optional and tracked as a follow-up.

- **Result / assistant events arrive before `system(init)` (malformed or drifted streams)**. Amp is *expected* to always emit `system(init)` first, but drift or upstream bugs could produce a stream where `assistant`/`user`/`result` events arrive without an init. In that case the thread ID is never captured and the pending file cannot be renamed. Failure-path rule:
  - The pending file is closed (events still on disk for diagnostics) and left under `~/.config/mala/amp-sessions/.pending-{pid}.jsonl` with its contents intact — **not** auto-deleted — so the user/operator can inspect it. The orphan-pending cleanup follow-up may delete files older than a documented TTL (e.g., 7 days).
  - `AmpClient` raises `AmpStreamMissingInitError` carrying the pending-file path and a bounded slice of stderr/stdout (truncated to e.g. 4 KiB to avoid unbounded memory) so the orchestrator can fail the issue with diagnostics.
  - Because no thread ID is captured, no resume is possible for that issue; the orchestrator treats it as a per-issue fatal and continues to the next issue (matches the "issue fails, run continues" policy for transient per-issue failures).
  - Test in `test_amp_log_provider.py`: synthesize a stream with `assistant` + `result` only (no `system(init)`); assert `AmpStreamMissingInitError` is raised, the pending file is preserved at the documented path, the error carries truncated stderr/stdout, and `AmpLogProvider.iter_events({thread_id})` returns nothing for any thread (the events were never indexed under a thread).

- **User deletes `~/.config/mala/amp-sessions/` between attempts**. On the next resume, mala has no prior history. Mitigations:
  - Evidence parsing degrades gracefully — missing files are equivalent to "no events"; the gate may then legitimately fail and is treated as a normal gate-fail-then-retry.
  - The orchestrator does not crash on a missing tee file; this is the existing `FileSystemLogProvider` behavior pattern.

- **Log file corruption / partial trailing line on crash**. `AmpLogProvider` reuses the existing `FileSystemLogProvider` tolerance for malformed JSON lines: a partial trailing line is skipped with a warn-level log, not raised.

- **Concurrent runs sharing a thread ID**. Two mala processes simultaneously appending to the same `{thread_id}.jsonl` is theoretically possible if a thread ID collision occurs. In practice Amp generates unique thread IDs (`T-...`), so this is treated as a documented assumption rather than an actively defended-against case. If a collision is ever observed, the symptom is interleaved events in one file; treated as a follow-up rather than an MVP blocker.

- **`amp threads continue` shape unknown**. Spike during impl tries:
  - `--thread-id <id>` on `amp --execute`.
  - `amp threads continue <id>` as separate subcommand.
  - If neither is reliable, fall back to fresh thread + prompt-accumulation per retry. Documented as a known limitation; tracked as follow-up issue.

- **Exact Amp mode flag** (`--mode` vs `--agent-mode` vs settings). Resolved during spike; `AmpRuntimeBuilder` updated accordingly.

- **`--claude-settings-sources` flag passed when `coder=amp`**. Logged as ignored (info-level). Not an error — keeps the CLI safe across coder switches.

- **`--amp-mode` passed when `coder=claude`**. Logged as ignored (info-level). Same rationale.

- **`MALA_DISALLOWED_TOOLS` set when `coder=amp`**. Has no effect on Amp in MVP. Logged once at warn-level on run start so users know. Tracked as a follow-up.

- **Auth errors**. `AmpClient` inspects stderr for `AMP_API_KEY`, `unauthorized`, `401`, `forbidden`. Maps to existing fatal-vs-per-issue policy: missing key → fatal exit; transient → fail issue and continue.

- **Amp CLI missing from `PATH`**. Fail before starting issue execution with a clear prerequisite error that names `amp` and points to docs.

- **Amp process hangs without output**. Existing `IdleTimeoutRetryPolicy` applies; subprocess is terminated during cleanup.

- **Amp process exits before a `result` event**. Treat as provider failure; include exit code and bounded stderr in diagnostics.

- **Amp returns `error_max_turns` / `error_during_execution`**. Map into existing issue failure/retry policy as a result error, not as a parser crash.

- **MCP launch parity**. `locking_mcp` is a stdio Python server. Integration test `test_lock_mcp_via_amp.py` verifies a `lock_acquire` round-trip succeeds through Amp's `--mcp-config`. If MCP config is rejected by Amp at startup, fail the run and surface the generated MCP config path/shape in diagnostics.

- **Bun missing**. If Amp's plugin runtime can't load (Bun missing), Amp itself fails to start. `AmpClient` surfaces stderr with a clear "install Bun via Amp" hint.

- **Cancellation / SIGINT**. `AmpClient.__aexit__` and the existing `SigintGuard` (`src/infra/sigint_guard.py`) terminate the spawned `amp` process cleanly (SIGTERM → grace → SIGKILL); pattern mirrors existing subprocess termination.

- **Stderr unbounded growth**. Capture into a bounded ring buffer for diagnostics rather than letting memory grow without limit.

- **Cost / rate limits**. Amp's smart=Opus, rush=Haiku, deep=GPT-5.4. Documented in `README.md`. CI smoke job uses a one-line prompt to bound cost.

### Breaking Changes & Compatibility

- **Potential breaking changes (internal — no public API)**:
  - `SDKClientFactoryProtocol` may shed Claude-specific vocabulary (`hooks`, `setting_sources`, `mcp_servers`, `resume`) once it is wrapped by `AgentProvider`. All in-tree call sites are updated directly per the no-shim rule.
  - `AgentSessionRunner`, `FixerService`, and `RunCoordinator` constructor signatures change to take `AgentProvider`. All in-tree call sites updated.
  - Generalizing factory/runtime APIs could affect existing Claude-only call sites; covered by the existing Claude integration suite as the regression guard.

- **External user-facing surface**:
  - **Default behavior unchanged**: omitting `--coder` selects `claude` and produces byte-equivalent runs.
  - New CLI flags are additive (`--coder`, `--amp-mode`).
  - New `coder:` / `coder_options:` mala.yaml fields are optional with defaults; existing configs remain valid.
  - Existing telemetry attribute names unchanged; new `coder` attribute added.
  - **npm-installed Amp users get an explicit error, not silent unsafety.** Selecting `coder=amp` against an `npm install -g @sourcegraph/amp` install will fail the runtime plugin-load self-test and abort the run with an actionable error pointing at the official binary install. This is an intentional fail-closed posture: under `--dangerously-allow-all`, running without the safety plugin is unsafe. The documented fix is to install Amp via the binary install path.

- **Mitigations**:
  - Feature-flag-style default (`coder: claude`) is the safe-by-default behavior.
  - Comprehensive Claude-path regression coverage (existing test suite untouched).
  - Path-gated CI smoke catches Amp drift without affecting Claude-path PRs.
  - Document Claude-only and Amp-only options clearly in `docs/cli-reference.md` and `docs/project-config.md`.

## Testing & Validation Strategy

- **Unit tests (Amp client)** — `tests/unit/infra/clients/test_amp_client.py`:
  - `system(init)` → captures `session_id` (`T-...`); tee log header written.
  - `assistant` events with `text` / `tool_use` / `thinking` → emits synthetic `AssistantMessage` (thinking stripped in MVP).
  - `user` events with `tool_result` → emits synthetic `AssistantMessage` containing `ToolResultBlock`.
  - `result(success | error_during_execution | error_max_turns)` → emits synthetic `ResultMessage` with right `session_id` + `result`.
  - Stderr capture and auth-error classification (`unauthorized`, `401`, `AMP_API_KEY`).
  - Subprocess lifecycle: spawn, write prompt to stdin, terminate on `__aexit__` (SIGTERM → grace → SIGKILL).
  - Resume: `with_resume(thread_id)` produces the expected argv (assertion stable across the two candidate shapes via parametrization).
  - Tee'd log produced exactly once per session, idempotent on retry.
  - Cancellation mid-stream cleans up subprocess and tee file handle.
  - Malformed JSON, premature exit, and missing `result` events handled deterministically.

- **Unit tests (synthetic messages)** — `test_amp_messages.py`:
  - Class names match what `MessageStreamProcessor` keys off (`AssistantMessage`, `ResultMessage`, `TextBlock`, `ToolUseBlock`, `ToolResultBlock`).
  - Fields read by the processor are present and have correct types (`text`, `id`, `name`, `input`, `tool_use_id`, `is_error`, `content`, `session_id`, `result`).
  - **Round-trip**: feeding a constructed `AssistantMessage` into the real `MessageStreamProcessor` produces the same observable output as a Claude SDK `AssistantMessage` would.

- **Unit tests (runtime builder)** — `test_amp_runtime.py`:
  - CLI args include `--execute --stream-json --dangerously-allow-all --mcp-config <json>`.
  - Env includes `PLUGINS=all` and `AMP_API_KEY` passthrough.
  - **Env includes `MALA_AGENT_ID`, `MALA_LOCK_DIR`, `MALA_REPO_NAMESPACE`** with
    values matching what the Claude path's `AgentRuntimeBuilder.with_env()` would
    inject for `AGENT_ID`, `LOCK_DIR`, `REPO_NAMESPACE` for the same `agent_id`
    and `repo_path`. Asserted as a positive case **and** as a regression guard:
    if any of the three is missing from `AmpRuntime.env`, the test fails loudly
    so a future env-shape refactor cannot silently disable the plugin's
    lock-ownership check.
  - `--mode <smart|rush|deep>` (or settings-equivalent) wired from `AmpOptions.mode`.
  - `mcp_config` JSON contains the `locking_mcp` stdio launch spec.

- **Unit / integration tests (plugin lock-ownership check)** —
  `tests/integration/test_amp_lock_enforcement.py` (new) drives the bundled
  `mala-safety.ts` plugin **via real Amp** (which provides Bun) against a real
  `MALA_LOCK_DIR` populated by the real Python `try_lock`, so the plugin's
  reader and the Python writer share the same on-disk fixture. **Bun is not a
  separate CI prerequisite**: this test file is gated on the same conditions
  as the smoke job (`AMP_API_KEY` secret + path-gated trigger on
  `src/infra/clients/amp_*` and `plugins/amp/`), and Bun arrives transitively
  with the binary Amp install. A small, optional pure-TypeScript unit test
  (`tests/unit/plugins/test_lock_check.ts`) that exercises only the lock-key
  hashing/parsing helper functions (no Amp dependency, runs under whatever
  Bun/Deno/Node is locally available) is **out of MVP scope** and tracked as a
  follow-up:
  - **Allow case**: agent A acquires a lock for `repo/foo.py` via `lock_acquire`
    MCP; the plugin observes a `tool.call` for an Amp file-write tool targeting
    `repo/foo.py` from agent A; result is `allow`.
  - **No-lock case**: no lock file exists for `repo/foo.py`; plugin observes
    a write `tool.call` from agent A; result is `reject-and-continue` with the
    no-lock message.
  - **Wrong-agent case**: agent B holds the lock; agent A's plugin observes a
    write `tool.call` for the same file; result is `reject-and-continue`.
  - **Wrong-namespace case**: a lock exists for the same path under a different
    `MALA_REPO_NAMESPACE`; the plugin (running with this run's namespace) does
    not see a matching `<hash>.lock` and rejects.
  - **Env-missing case (fail closed)**: `MALA_LOCK_DIR` (or `MALA_AGENT_ID` or
    `MALA_REPO_NAMESPACE`) is unset on the Amp process; plugin
    `session.start` puts it into fail-closed mode and any subsequent file-write
    `tool.call` is rejected with "lock-ownership env missing".
  - **Format-drift resilience**: a fixture lock file written by the real Python
    `try_lock` is parsed correctly by the plugin (i.e., the plugin and Python
    agree on the SHA-256-prefix-filename + `<agent_id>\n`-body format). If the
    Python format changes, this test fails — the cross-language contract is
    test-enforced.
  - **Path canonicalization parity**: a lock acquired with a relative path
    under the same `MALA_REPO_NAMESPACE` matches a `tool.call` whose input
    contains the absolute path of the same file (and vice versa), mirroring
    `_canonicalize_path` semantics. Symlinked-parent and non-existent-target
    cases are exercised against the same fixtures the Python locking tests use.
  - **End-to-end through real Amp**: covered as part of `test_lock_mcp_via_amp.py`
    — a `lock_acquire` round-trip followed by a write `tool.call` for the
    locked file is allowed; a write to an unlocked file is rejected.

- **Unit tests (plugin installer)** — `test_amp_plugin_installer.py`:
  - First call writes `~/.config/amp/plugins/mala-safety.ts`.
  - Second call is a no-op when content hash matches.
  - When content differs (mala upgraded), file is replaced via temp-then-rename.
  - Concurrent calls don't corrupt the file.
  - Acknowledgment header preserved verbatim in installed copy.
  - Missing or unwritable plugin directory fails with a clear error.

- **Unit / integration tests (plugin self-test, fail-closed)** — `test_amp_plugin_self_test.py`:
  - **Pass case**: fake `amp` binary on `PATH` emits the sentinel marker with the matching version hash → self-test passes; orchestrator proceeds to spawn issue agents.
  - **Fail closed: plugin missing** — fake `amp` emits no sentinel marker within timeout → `AmpPluginNotActiveError` raised; orchestrator refuses to spawn any issue agent; error message names the binary-install docs.
  - **Fail closed: Bun unavailable** — fake `amp` emits a Bun-missing stderr signature → `AmpPluginNotActiveError` raised with Bun-specific guidance.
  - **Fail closed: `PLUGINS=all` not set / not honored** — self-test detects the plugin did not load and fails closed with the env-var hint.
  - **Fail closed: version mismatch** — sentinel marker carries an unexpected version hash → fail closed with a stale-plugin hint (suggests rerunning to refresh).
  - **Fail closed: amp binary missing** — `amp` not on `PATH` → fail closed before any other check, with binary-install docs reference.
  - **Fail closed: npm-install fingerprint** — sentinel is absent and stderr matches the npm-install signature → error message explicitly states the binary install is required (the npm path is unsupported).
  - Self-test timeout is bounded (e.g., 10s) and uses a sentinel prompt that does not require network/model calls beyond the minimum needed to trigger `tool.call`.
  - Self-test is invoked exactly once per run (idempotent within a run).

- **Unit tests (log provider)** — `test_amp_log_provider.py`:
  - Reads tee'd JSONL and emits the same Bash evidence shape `FileSystemLogProvider` produces.
  - Native-log probe-then-fallback: returns deterministic "no native log" when the probe finds nothing, then uses tee.
  - When native log discovered (mock fixture), prefers it.
  - **First-invocation tee bootstrap**: `AmpClient` opens a `.pending-{pid}.jsonl` temp file; on the first `system(init)` event the file is renamed to `{thread_id}.jsonl`. After rename, `AmpLogProvider.get_log_path(thread_id)` resolves to the renamed file.
  - **Resume appends to existing thread file**: a second `AmpClient` invocation for the same `thread_id` opens `{thread_id}.jsonl` in append mode; events from both invocations are present in the file in order.
  - **`iter_events()` reads across invocations**: feeding a fixture file containing events written by two simulated invocations (initial run + resume) yields events in file order spanning both invocations.
  - **Cross-resume validation evidence (regression for the log-stitching finding)**: the first invocation logs Bash tool_use events for `pytest` / `ruff check` / `ty check`; the second (resumed) invocation logs only new events. `AmpLogProvider.iter_events(thread_id)` still surfaces the original lint/test/typecheck Bash tool_use events so the gate's evidence parser observes them — i.e., a delta-only resume does not lose validation evidence.
  - **Missing tee file (user deleted `~/.config/mala/amp-sessions/`)**: `iter_events()` returns an empty iterator without raising; gate-fail-then-retry is the documented downstream behavior.
  - **Malformed trailing line tolerance**: a JSONL file with a truncated last line yields all preceding events successfully and logs the partial line at warn-level.
  - **Orphan `.pending-*` file ignored**: a leftover pending temp file in `~/.config/mala/amp-sessions/` does not interfere with thread-keyed reads.

- **Unit tests (selection wiring)** — `test_factory_provider_selection.py`, `test_coder_flag.py`, `test_coder_config.py`, `test_coder_schema.py`:
  - `--coder amp` → `AmpAgentProvider` selected by factory.
  - `MALA_CODER=amp` honored when CLI absent.
  - `mala.yaml` `coder: amp` honored when CLI + env absent.
  - Default remains `claude`.
  - Precedence: CLI > env > yaml > default.
  - `--amp-mode` / `MALA_AMP_MODE` / `coder_options.amp.mode` follow same precedence.
  - mala.yaml strict-enum validation rejects `coder: foo`, `mode: bar`.
  - `--amp-mode rush --coder claude` logs ignored info-level.
  - Existing `mala.yaml` files without `coder:` remain valid and default to Claude.
  - `install_prerequisites()` invoked once per run before the first Amp session.

- **Unit tests (provider protocol)** — `test_amp_provider.py`, `test_claude_provider.py`:
  - Both providers pass `isinstance(p, AgentProvider)` runtime check.
  - `install_prerequisites()` is idempotent for both.
  - `ClaudeAgentProvider` returns objects compatible with existing pipeline (regression coverage for the refactor).
  - **Lazy-import guard**: importing `AmpAgentProvider` modules does not import `claude_agent_sdk`; importing `ClaudeAgentProvider` modules does not import Amp adapter modules.

- **Integration tests**:
  - `tests/integration/test_amp_provider.py`: fake `amp` binary on `PATH` (small Python script returning canned stream-json) — full per-issue lifecycle: query, parse, gate, review hand-off, close.
  - `tests/integration/test_coder_selection.py`: end-to-end CLI/env/yaml precedence flowing through orchestration.
  - `tests/integration/test_fixer_follows_coder.py`: with `coder=amp`, `FixerService` spawns Amp (verified by counting calls to a stub `AgentProvider`); with `coder=claude`, fixers stay on Claude.
  - `tests/integration/test_lock_mcp_via_amp.py`: `locking_mcp` reachable through Amp's `--mcp-config`; `lock_acquire` round-trip succeeds with the bundled plugin loaded.
  - Verify `MessageStreamProcessor` consumes Amp synthetic messages without provider-specific branches.
  - Verify `--coder claude` path is byte-equivalent to today (regression).
  - Verify `--claude-settings-sources` is ignored (info log) under Amp; still applied under Claude.

- **Regression tests**:
  - Existing Claude integration suite runs unchanged after the `ClaudeAgentProvider` refactor. Any behavior diff is a refactor bug, not a feature.
  - Existing config files without `coder` remain valid and default to Claude.
  - Existing reviewer configuration and Epic verifier behavior remain unchanged.
  - Existing locking behavior remains enforced for Claude and is mirrored for Amp through plugin + MCP.
  - Existing idle timeout policy behavior unchanged for Claude.

- **CI smoke job** — `tests/smoke/test_amp_real_cli.py`:
  - Path-gated on `src/infra/clients/amp_*`, `plugins/amp/`, `tests/smoke/test_amp_real_cli.py`, and this plan path.
  - Gated on `AMP_API_KEY` repository secret.
  - Calls real `amp --execute --stream-json` with a one-line prompt; asserts the schema fields we depend on are present (`event.type`, `event.message.content[].type`, `session_id`).
  - Reports observed Amp version and minimal schema fields.
  - Catches upstream stream-json schema drift early.

- **Manual verification** before declaring done:
  - `mala run --coder amp` against a real beads issue end-to-end; verify commit produced, gate passes, review runs, issue closes.
  - Verify `MALA_CODER=amp` works without `--coder` flag.
  - Verify `mala.yaml coder: amp` works without env or flag.
  - Verify `mala run --coder claude` is byte-equivalent to today (manual diff of an existing run output).
  - Verify a deliberately dangerous shell prompt is blocked by the bundled plugin.
  - Verify file edits without lock acquisition are rejected before execution.
  - Verify validation evidence is available from the Amp log provider.
  - Verify fixer flow uses Amp if a review/gate fix is needed.

- **Monitoring / observability**:
  - `MalaEventSink` spans carry `coder=claude|amp`.
  - Include Amp session/thread id in provider diagnostics when available.
  - Log plugin install/verify status at startup for Amp runs.
  - Log when Claude-only settings are ignored under Amp (info-level).
  - Log when tee'd Amp logs are used instead of native logs.
  - Dashboards can split per-coder success rate / duration; not part of MVP delivery but enabled by the attribute.

### Acceptance Criteria Coverage

| AC | Approach |
|---|---|
| AC #1: `mala run --coder amp` invokes Amp instead of Claude | New `--coder` CLI flag → `MalaConfig.coder` → `AmpAgentProvider` selected by factory. Tests: `test_coder_flag.py`, `test_factory_provider_selection.py`, integration `test_amp_provider.py`. |
| AC #2: Default behavior remains Claude when `--coder` is absent | `coder` defaults to `claude`; `ClaudeAgentProvider` is byte-equivalent to today's path. Tests: existing Claude integration suite + `test_claude_provider.py`. |
| AC #3: Selection precedence is CLI > env > yaml > default | Mirrors `claude_settings_sources` resolver. Tests: `test_coder_flag.py`, `test_coder_config.py`, `test_coder_selection.py`. |
| AC #4: Amp mode is configurable as `smart`, `rush`, or `deep` (default `smart`) | `CoderOptions.amp.mode`, Amp runtime builder tests, docs updates. |
| AC #5: Fixer agents follow main coder | `FixerService` consumes `AgentProvider` from deps. Test: `test_fixer_follows_coder.py`. |
| AC #6: Per-issue lifecycle (gate + review + close) works for both Claude and Amp | `AgentSessionRunner` is coder-agnostic; only the provider differs. Test: integration `test_amp_provider.py` + Claude regression. |
| AC #7: Validation evidence is available for Amp runs | `AmpLogProvider` reads tee'd or native log; same Bash tool_use shape. Tests: `test_amp_log_provider.py`, integration. |
| AC #7a: Validation evidence persists across Amp thread continuations | Tee path is per-thread (`~/.config/mala/amp-sessions/{thread_id}.jsonl`) and append-only across resumes; `AmpLogProvider.iter_events()` reads the full file regardless of which invocation produced which events. Tolerates Amp emitting either delta-only or full-history events on resume. Tests: cross-resume tests in `test_amp_log_provider.py` (first-invocation bootstrap, resume-appends, `iter_events()` across invocations, lint/test/typecheck evidence preserved across an idle-retry boundary). |
| AC #8: Idle / review retries work for Amp via thread continuation (or documented fallback) | `AmpClient.with_resume(thread_id)` continues the thread. Test: `test_amp_client.py::test_resume`. |
| AC #9: Critical safety controls (dangerous-cmd, lock-enforce) hold for Amp | Bundled `mala-safety.ts` plugin loaded via `PLUGINS=all`. Tests: plugin installer unit tests + manual verification of dangerous prompt + integration lock test. |
| AC #9a: An Amp file-write is allowed only when the current Amp agent holds the lock for that file in this repo namespace (parity with the Claude `make_lock_enforcement_hook`) | `mala-safety.ts` reads `MALA_AGENT_ID` / `MALA_LOCK_DIR` / `MALA_REPO_NAMESPACE` injected by `AmpRuntimeBuilder`, derives the same SHA-256-prefix lock-file path as `src/infra/tools/locking.lock_path`, and rejects any file-write `tool.call` whose target lacks a `<hash>.lock` whose body equals `MALA_AGENT_ID`. Tests: `test_amp_lock_enforcement.py` (allow / no-lock / wrong-agent / wrong-namespace / env-missing / format-drift / canonicalization-parity) plus the real-Amp `test_lock_mcp_via_amp.py` round-trip. |
| AC #10: `--coder claude` runs unchanged | Regression suite passes; CI smoke unchanged. Tests: existing Claude integration suite + lazy-import guard. |
| AC #11: Amp prerequisites and limitations are documented | `README.md`, `docs/cli-reference.md`, `docs/project-config.md`, `docs/architecture.md`, `docs/development.md`. |
| AC #12: Real Amp schema drift is detectable before merge for Amp changes | `tests/smoke/test_amp_real_cli.py`, path-filtered CI job, `AMP_API_KEY` gating. |
| AC #13: `mala.yaml` validation rejects unknown `coder` / `mode` values before any process starts | Strict-enum schema validation in `src/domain/validation/config.py`. Test: `test_coder_schema.py`. |
| AC #14: Plugin install is idempotent and concurrent-safe; absent plugin aborts the Amp run | `AmpPluginInstaller` write-temp-then-rename + content-hash invariant check at startup. Tests: `test_amp_plugin_installer.py`. |
| AC #15: `AmpLogProvider` uses native log if discovered at install time, otherwise tees stream-json | Probe-then-tee strategy; tee is the safe default. Tests: `test_amp_log_provider.py`. |
| AC #16: `coder=amp` requires the binary Amp install; npm-installed Amp fails closed before any issue agent runs | Documented binary-install prerequisite + runtime plugin-load self-test in `AmpAgentProvider.install_prerequisites()` raising `AmpPluginNotActiveError`. Tests: `test_amp_plugin_self_test.py` covering npm-install, Bun missing, `PLUGINS=all` unset, version mismatch, and amp-missing cases. |
| AC #17: Under `--dangerously-allow-all`, the orchestrator refuses to spawn issue agents if the safety plugin is not active at runtime (fail-closed safety invariant) | Self-test gate before first session; hash-only check is explicitly insufficient. Tests: `test_amp_plugin_self_test.py` fail-closed cases + manual verification with a deliberately disabled plugin. |

## Spec/Legacy Fidelity

This plan is derived from a user request, not an existing spec. The 2025-12-30 Codex provider plan is referenced for shape only — its decisions are not binding here, and the codebase currently has no provider abstraction.

### Deviation Log

| Source | Deviation | Rationale | Approved? |
|---|---|---|---|
| Skeleton draft | `AgentProvider` exposes `client_factory` and `log_provider` as attributes (rather than properties) and `runtime_builder` as a method | Clean and consistent: factory/log are values; runtime is constructed per session. | Implicit (Python protocol convention) |
| Skeleton draft | Native-log probe deferred to impl spike; tee is the MVP default | Reliability — undocumented Amp log format means tee is the only safe baseline. | Implicit |
| Codex plan (2025-12-30) | Different abstraction (this plan introduces `AgentProvider` triple) | Codex plan was never implemented and used a different abstraction shape; revisiting from scratch is correct. | N/A — not binding |
| User interview | `block_mala_disallowed_tools` parity deferred to follow-up | User explicitly out-of-scoped this for MVP. | Yes |
| User interview | Devcontainer changes excluded | User explicitly out-of-scoped. | Yes |
| User interview | Plugin shipped in mala repo, copied to `~/.config/amp/plugins/` on first run | User decision; global install for cross-repo coverage. | Yes |

## Open Questions

These are deferred to implementation-time spikes, **not blockers** for the plan:

- **Exact `amp threads continue` invocation**: confirm whether Amp accepts `--thread-id <id>` on `amp --execute` or requires `amp threads continue <id>`. `AmpClient.with_resume` is built to accept either argv shape via parametrized tests; if neither is reliable, fall back to fresh-thread + prompt-accumulation per documented limitation.
- **Amp resume emit shape (delta vs. full history)**: while running the resume spike, also verify whether `amp --execute --stream-json [--thread-id <id>]` (or the equivalent `amp threads continue <id>` invocation) emits *only the new events* (delta) or re-emits the *full thread history* on resume. The plan tolerates either shape because the tee is per-thread and append-only — so this is a documentation/observability item, not a correctness blocker. Document the observed behavior in `docs/architecture.md` so reviewers and future contributors do not have to re-derive it.
- **Native log location**: probe `~/.config/amp/` and `~/.local/share/amp/` during the install spike. If a stable JSONL emerges, prefer it. Otherwise tee. Either is satisfied by `AmpLogProvider`.
- **Exact Amp mode flag**: verify whether agent-mode is `--mode`, `--agent-mode`, or via settings/plugin. Pin during impl and update `AmpRuntimeBuilder` accordingly.
- **Plugin acknowledgment header drift**: track upstream changes to the WIP-API string. Pinned in `plugins/amp/README.md`.
- **Exact Amp tool names and payload shapes for file edits and shell commands**: affects plugin enforcement coverage *and* the lock-ownership tool-name → path-key mapping in `mala-safety.ts` (parity with `FILE_WRITE_TOOLS` / `FILE_PATH_KEYS` in `src/infra/hooks/file_cache.py`); verified during plugin spike.
- **Lock-file format spec extraction**: should the `<hash>.lock` filename hashing + body format be lifted out of Python and into a tiny shared format spec (e.g., a JSON description under `docs/`) that both `src/infra/tools/locking.py` and `plugins/amp/mala-safety.ts` cite? Deferred — for MVP, the contract is documented in the plan and asserted by `test_amp_lock_enforcement.py`'s format-drift test.
- **Amp version pin range**: chosen at impl time after the smoke job is green; documented in `README.md`.

## Next Steps

After this plan is approved, run `/cerberus:create-tasks` to generate execution artifacts:
- `--beads` → Beads issues with dependencies for multi-agent execution
- `--agent-team` → cerberus team-tasks.md for implementer/reviewer pairs
- (default) → TODO.md checklist for simpler tracking

A reasonable task ordering (dependencies in parens):

1. **`AgentProvider` + `CoderRuntimeBuilder` protocols** (foundation, no behavior change)
2. **`ClaudeAgentProvider` refactor** behind the new protocol (depends on 1; existing Claude tests must pass as regression guard)
3. **CLI / env / yaml selection wiring + tests** (depends on 1)
4. **Synthetic Amp message dataclasses + tests** (independent)
5. **`AmpClient` (subprocess + stream-json parsing) + unit tests with mocked subprocess** (depends on 4)
6. **`mala-safety.ts` plugin + `AmpPluginInstaller` + tests** (independent of 5)
7. **`AmpLogProvider` (tee MVP, native-log probe) + tests** (depends on 5)
8. **`AmpRuntimeBuilder` + `AmpAgentProvider` wiring + tests** (depends on 5, 6, 7)
9. **`FixerService` consumes `AgentProvider`** (depends on 8)
10. **Resume via thread continuation + tests** (depends on 5; spike)
11. **Integration tests with fake `amp` binary on PATH** (depends on 8, 9)
12. **CI smoke job + path filter config** (depends on 11)
13. **Telemetry: add `coder` span attribute** (independent, low-risk)
14. **Docs** (`README.md`, `docs/cli-reference.md`, `docs/project-config.md`, `docs/architecture.md`, `docs/development.md`)
