# Implementation Plan: Codex as a Third Coder Provider (Multi-Epic)

## Context & Goals

- **Spec**: N/A — derived from user request "add a Codex coder provider; reference how Amp was added; structure as multi-phase epics where Phase A cleans up architecture friction surfaced by the Amp work and the rest implement Codex".
- **Prior art**:
  - `plans/2026-04-29-amp-provider-plan.md` — the Amp provider plan, now implemented (`src/infra/clients/amp_*.py`, `plugins/amp/`, etc.). The `AgentProvider` / `CoderRuntimeBuilder` protocol abstraction it introduced is the foundation we extend.
  - Codex Python SDK at `../codex/sdk/python/` (package `codex_app_server`, distributed as `openai-codex-app-server-sdk`). It is a JSON-RPC v2 over stdio client wrapping the `codex app-server` runtime; the runtime ships separately as `openai-codex-cli-bin` (pinned exact version per SDK release).
- **Why Codex**: Adds a third backend so users can pick OpenAI's `codex app-server` agent (gpt-5.5 family) alongside Claude (Anthropic SDK) and Amp (Sourcegraph CLI subprocess). Diversifies model/cost/safety profiles and exercises the `AgentProvider` abstraction across a fundamentally different shape (in-process Python SDK + JSON-RPC, not a CLI subprocess wrapper).
- **What Codex actually exposes for safety/integration** (correcting an earlier assumption — Codex has *both* hooks and plugins):
  - **Claude-Code-style hooks** (`codex-rs/hooks/schema/generated/*.json`): events `PreToolUse | PermissionRequest | PostToolUse | PreCompact | PostCompact | SessionStart | UserPromptSubmit | Stop`. Configured in TOML under `[hooks]` with `MatcherGroup` arrays per event. Three handler types: `command` (shell command, JSON stdin/stdout, async-optional, with `trusted_hash` security gate), `prompt`, `agent`. `PreToolUse` output supports `decision: "approve"|"block"` and `hookSpecificOutput.permissionDecision: "allow"|"deny"|"ask"` with reason. **This is the Claude-SDK hook shape, not the Amp-style WIP TypeScript plugin API.**
  - **Codex plugin manifest** (`codex-rs/skills/src/assets/samples/plugin-creator/references/plugin-json-spec.md`): a `plugin.json` with `name/version/description`, plus relative paths to `skills/`, `hooks` (a `hooks.json` file), `mcpServers` (a `.mcp.json` file), `apps`, and an `interface` UX block. Plugins live at `.codex-plugin/plugin.json`. So a "Codex plugin" is a packaging unit for hooks + skills + MCP, not a runtime API.
  - **MCP** is configured via `mcpServers` (either inline in TOML config or in a plugin's `.mcp.json`).
  - **Sandbox + approval_policy** layer atop hooks: hooks run before sandbox/approval evaluation for `PreToolUse`, so a hook `deny` short-circuits without hitting the sandbox. Defense-in-depth.
- **Implication for safety design**: Codex's hook surface is structurally close to the *Claude* path (`make_lock_enforcement_hook` etc., `src/infra/hooks/`), not the Amp path (TS plugin). The Mala safety hook is a single Python command-hook script (`mala-codex-pre-tool-use`) that reuses the same `MALA_*` env contract Amp's plugin already establishes. We bundle the hook config as a Mala-shipped Codex plugin (`plugins/codex/mala-safety/.codex-plugin/{plugin.json, hooks.json, .mcp.json}`, mirroring the spec-canonical `.codex-plugin/` layout); the plugin packaging is the right shape because it co-locates the hook with the locking MCP launcher and is reversible (versus writing into the user's `~/.codex/config.toml` `[hooks]` block, which is rejected because it touches user config).
- **Why a refactoring epic first**: The Amp work landed under time pressure and left several architectural seams that Codex would either have to copy verbatim (re-cementing them as multi-coder smell) or work around. Phase A cleans those up *before* any Codex code is written, so Codex code lands clean and the Claude path benefits too. Specifically (with file:line evidence):
  1. **Anthropic-class-name duck typing in the message stream** (`src/pipeline/message_stream_processor.py:230-292`) — the processor branches on `type(message).__name__ == "AssistantMessage"` / `"ToolUseBlock"` / `"ToolResultBlock"` and `getattr`-reads Anthropic-shaped fields. Amp coped by fabricating Anthropic-shaped synthetic dataclasses (`src/infra/clients/amp_messages.py`). Codex's items (`AgentMessageThreadItem`, `CommandExecutionThreadItem`, `FileChangeThreadItem`, `McpToolCallThreadItem`, `ReasoningThreadItem`) do not align with the Anthropic content-block model, so synthesizing them into Anthropic shape would be a fragile lossy adapter inside an already-fragile contract.
  2. **`SDKClientFactoryProtocol` carries Claude vocabulary** (`src/core/protocols/sdk.py:74-165`) — `create_options(permission_mode=, hooks=, mcp_servers=, setting_sources=, disallowed_tools=, ...)` plus `create_hook_matcher(...)`. Amp's `_AmpClientFactory` raises `NotImplementedError` on most of these (`src/infra/clients/amp_provider.py:527-560`). Codex would add a third NotImplementedError block — protocol bloat.
  3. **`runtime.options` field is mandatory but coder-shaped** — `AgentSessionRunner` reads `runtime.options` directly (`src/pipeline/agent_session_runner.py:445`), forcing Amp's `AmpRuntimeBuilder.build()` to construct an `AmpClientOptions` eagerly even though no real Anthropic options object exists (`src/infra/clients/amp_runtime.py:354-368`). Codex doesn't have an "options" object in the same sense; we'd be inventing a third synthetic options shape.
  4. **MCP factory dispatch is a hardcoded `if agent_provider.name == "amp"` branch** (`src/orchestration/factory.py:1210` and `src/orchestration/orchestrator.py:404`). Adding Codex makes this an N-way switch in two places — should be owned by the provider.
  5. **`LogProvider.iter_thread_events` was added as an Amp-shaped escape hatch** (`src/core/protocols/log.py` — the docstring explicitly says the offset is honored on Claude but ignored on Amp). Codex stores threads in its own data dir indexed by SDK calls (`Thread.read(include_turns=True)` rather than tailed JSONL), so the LogProvider contract continues to bend.
  6. **`AgentRuntimeBuilder` fluent surface is Claude-shaped** (`with_setting_sources`, `with_hooks(deadlock_monitor, include_*hook=...)`, etc.). Amp's `AmpRuntimeBuilder` implements most as recorded-but-unused no-ops (`src/infra/clients/amp_runtime.py:164-252`). Codex would add yet another no-op layer.
  7. **Lifecycle effects assume an on-disk session log to wait for** (`WAIT_FOR_LOG`-style coupling) — Codex provides session/thread state directly via SDK calls, so the wait step is at best vacuous and at worst a hard stall.
  8. **Selection / config types are explicitly two-valued**: `MalaConfig.coder: Literal["claude", "amp"]` (`src/infra/io/config.py:364`), `ValidationConfig.coder: Literal["claude", "amp"] | None` (`src/domain/validation/config.py:895`), CLI typer enum, `AgentProvider.name: Literal["claude", "amp"]` (`src/core/protocols/agent_provider.py:54`). Adding Codex requires touching all of them and the resolver in `MalaConfig.from_env`.
- **Audience**: mala operators who want their per-issue implementation agent (and the fixer agents that follow it) on `codex app-server` instead of Claude or Amp, with everything else (gating, review, beads workflow, lifecycle policy) unchanged.

## Scope & Non-Goals

### In Scope

**Multi-phase epic delivery** (each phase is a coherent epic; later phases depend on earlier ones):

- **Phase A — Provider-agnostic architecture cleanup** (no new feature, no behavior change for Claude or Amp). Address the eight friction points listed above so Codex lands cleanly:
  - A1. Replace Anthropic-class-name duck typing in `MessageStreamProcessor` with a coder-agnostic event protocol (`AgentEvent` / `AgentTextEvent` / `AgentToolUseEvent` / `AgentToolResultEvent` / `AgentResultEvent`). Rewire Claude and Amp adapters to emit the new events; processor branches on event `kind`. **Single big-bang PR** (decision #6).
  - A2. Slim `SDKClientFactoryProtocol` to its actual cross-coder surface (`create(runtime)` + `with_resume(runtime, resume)`) and move Claude-only knobs (`create_options`, `create_hook_matcher`) to a Claude-private factory class. Amp's `NotImplementedError` walls go away; Codex never inherits them.
  - A3. Make `runtime.options` opaque again — pipeline calls `provider.client_factory.create(runtime)` rather than `create(runtime.options)`. Each provider unpacks its own runtime privately.
  - A4. Move MCP factory selection onto `AgentProvider.mcp_server_factory()` (decision #14) so the orchestrator no longer branches on `provider.name`.
  - A5. Generalize `LogProvider`: replace the `iter_thread_events` escape hatch with a typed `EvidenceProvider` protocol that providers implement directly (Claude reads JSONL, Amp reads tee'd JSONL, Codex reads via `Thread.read(include_turns=True)`).
  - A6. Generalize `AgentRuntimeBuilder` fluent surface — keep cross-coder calls (`with_resume`, `with_agent_timeout`, `with_env`, `with_lint_tools`, `with_mcp`); move Claude-only calls (`with_setting_sources`, the disabled-hook flags) onto `ClaudeAgentRuntimeBuilder` exclusively.
  - A7. Decouple session-id-based filesystem assumptions in lifecycle effects — define `WAIT_FOR_LOG` semantics in terms of the provider's `EvidenceProvider`, not file existence at a hardcoded path.
  - A8. Generalize selection types: `MalaConfig.coder`, `ValidationConfig.coder`, CLI enum, `AgentProvider.name` all become `Literal["claude", "amp", "codex"]`.
- **Phase B — Codex selection + scaffolding (no real Codex execution yet)**:
  - B1. Add `coder=codex` to selection precedence (CLI `--coder codex`, env `MALA_CODER=codex`, `mala.yaml coder: codex`). All three plumb through to `_create_agent_provider` (`src/orchestration/factory.py:110`).
  - B2. Add `coder_options.codex.*`: `model`, `effort`, `approval_policy`, `sandbox`. Defaults: `model=gpt-5.5` (decision #9), `approval_policy=never` (decision #3), `sandbox=danger-full-access` (decision #2). Effort = pass-through with strict validation against Codex's `ReasoningEffort` enum (decision #13).
  - B3. Add `CodexAgentProvider` stub conforming to the (Phase A-cleaned) `AgentProvider` protocol. Stub raises a clear "not yet implemented" until C+ land, but selection wiring works end-to-end (so smoke tests can prove the new branch is reached).
- **Phase C — Codex client adapter (`CodexClient`)**:
  - C1. `CodexClient` conforms to (cleaned) `SDKClientProtocol`. Backed by `codex_app_server.AsyncCodex` + `AsyncThread` + `AsyncTurnHandle` — async-native, no subprocess wrapping ourselves (the SDK manages the `codex app-server` subprocess internally).
  - C2. `query()` opens `AsyncCodex` (lazily on first call), starts/resumes a thread, starts a turn, and exposes the notification stream.
  - C3. `receive_response()` consumes `TurnHandle.stream()` and emits coder-agnostic `AgentEvent`s (Phase A1 protocol).
  - C4. `with_resume(thread_id)` maps to `AsyncCodex.thread_resume(thread_id)` (decision #10).
  - C5. Cancellation maps to `AsyncTurnHandle.interrupt()` and `AsyncCodex.close()`; SIGINT pathway integrates with existing `SigintGuard`.
- **Phase D — Codex item → AgentEvent mapping**:
  - D1. `AgentMessageDeltaNotification` / `AgentMessageThreadItem` → `AgentTextEvent`.
  - D2. `CommandExecutionThreadItem` (status `started|in_progress|completed|failed`) → `AgentToolUseEvent("bash", {command, cwd, ...})` plus a paired `AgentToolResultEvent(is_error=...)` on completion. Used by `LintCacheProtocol.detect_lint_command` so the cache works for Codex too.
  - D3. `FileChangeThreadItem` → `AgentToolUseEvent("file_change", {changes})` + paired result event tied to `PatchApplyStatus`.
  - D4. `McpToolCallThreadItem` → `AgentToolUseEvent(<server>.<tool>, arguments)` + paired `AgentToolResultEvent` from `McpToolCallStatus` / `McpToolCallResult`.
  - D5. `ReasoningThreadItem` / `ReasoningTextDeltaNotification` → diagnostic-only (mirroring Amp's stripped-thinking decision). Tee'd to evidence log; not surfaced as `AgentEvent` in MVP (decision #12).
  - D6. `TurnCompletedNotification` → `AgentResultEvent(session_id=thread.id, is_error=<from status>, result=<TurnStatus value>)`.
  - D7. `ErrorNotification` → `AgentResultEvent(is_error=True, ...)`; classify auth / rate-limit / overload via existing classification helpers (extend if needed).
- **Phase E — Codex safety model** (Codex has Claude-Code-style command hooks; safety is hook-based with sandbox/approval-policy as defense-in-depth):
  - E1. **Default `sandbox`** = `danger-full-access` (decision #2). Implication: safety relies entirely on the bundled Mala hook + lock-MCP, mirroring Amp's `--dangerously-allow-all` posture. No silent unguarded path: if the hook isn't loaded, the run aborts (parity with Amp's plugin self-test).
  - E2. **Default `approval_policy`** = `never` (decision #3). Combined with `danger-full-access`, the only gate is the bundled `PreToolUse` command hook.
  - E3. **Lock-ownership + dangerous-cmd + `MALA_DISALLOWED_TOOLS` enforcement** via a bundled `PreToolUse` command hook (decision #4): a Python script `mala-codex-pre-tool-use`, registered as a `[project.scripts]` entry point.
    - Reads `cwd, tool_name, tool_input, session_id, turn_id` from JSON stdin (`pre-tool-use.command.input` schema).
    - Reads `MALA_AGENT_ID, MALA_LOCK_DIR, MALA_REPO_NAMESPACE, MALA_DISALLOWED_TOOLS` from env (parity with Amp's plugin contract; ensures the hook can be developed and unit-tested with the same fixtures that `tests/integration/test_amp_lock_enforcement.py` produces).
    - For shell-like tools (`bash`, Codex's local-shell tool name TBD by spike): mirror `src/infra/hooks/dangerous_commands.py` and reject dangerous commands.
    - For file-edit tools (Codex's `apply_patch` / file-write equivalent — exact tool name set confirmed in Phase D spike): compute the lock key + SHA-256 hash by **calling `src/infra/tools/locking.py::lock_path` directly** in-process (existing Python module — no cross-language reimpl needed; this is the *advantage* over Amp's TS plugin which had to reimplement SHA-256 hashing in TypeScript). Accept if the lock body equals `MALA_AGENT_ID`; otherwise reject with `permissionDecision: "deny"`.
    - `MALA_DISALLOWED_TOOLS`: enforce inside the same hook (advantage over Amp, where it's a known gap).
    - Returns `permissionDecision: "allow" | "deny" | "ask"` per the `pre-tool-use.command.output` schema, with `permissionDecisionReason` carrying the existing Mala deny messages.
  - E4. **Hook + locking-MCP packaging** = bundled Mala-shipped Codex plugin (decision #5): `plugins/codex/mala-safety/.codex-plugin/{plugin.json, hooks.json, .mcp.json}` installed idempotently to `~/.codex/plugins/mala-safety/.codex-plugin/` (or wherever Codex discovers user plugins; spike-confirmed). Plugin's `hooks.json` declares the `PreToolUse` command pointing at `mala-codex-pre-tool-use`. Plugin's `.mcp.json` references the `mala-locking` MCP server (so MCP and hook ship together — single install target for both safety hook and locking server). Alternative (writing into `~/.codex/config.toml`) is rejected because it touches user config; plugin packaging is reversible.
  - E5. **Hook self-test** (parity with Amp's `_run_selftest_subprocess`, `src/infra/clients/amp_provider.py`): on every Codex run, `CodexAgentProvider.install_prerequisites()` exercises a one-shot Codex turn that triggers the bundled hook (sentinel `PreToolUse` event) and verifies the hook script ran with the expected version hash. **Fail-closed** on any of: hook not installed, version mismatch, `mala-codex-pre-tool-use` script missing from PATH, plugin disabled, `trusted_hash` mismatch. Error class `CodexHookNotActiveError` with `Reason` enum: `HOOK_MARKER_MISSING | VERSION_MISMATCH | SCRIPT_MISSING | PLUGIN_DISABLED | TRUSTED_HASH_MISMATCH | CODEX_BINARY_MISSING`.
  - E6. **Trusted-hash auto-trust** (decision #16): `install_prerequisites()` writes the expected `trusted_hash` into Codex's hook-state file (location confirmed by Phase E spike against `codex-rs/core/config.schema.json:1016` `HookStateToml`). If Codex's UX requires user-interactive trust acceptance, fall back to a documented one-time prerequisite step (Open Question).
- **Phase F — Codex evidence provider** (decision #11: native `Thread.read(include_turns=True)`; tee fallback only if F1 spike disconfirms):
  - F1. **Spike-first**: confirm `Thread.read(include_turns=True)` returns command output (`CommandExecutionThreadItem.aggregated_output`) and tool-call results in a stable shape suitable for `extract_bash_commands` / `extract_tool_results` / `extract_assistant_text_blocks` semantics; confirm cross-resume invariant (full thread history regardless of resume count); confirm read cost is bounded for repeated gate calls. F1 lands before C/D/E/F implementation completes so the decision is validated early.
  - F2. `CodexEvidenceProvider` (Phase A5 protocol) calls `AsyncCodex.thread_read(thread_id, include_turns=True)` (or sync `Codex.thread_read` from a bounded executor) on every evidence query. Returns generator that yields `AgentEvent`-shaped entries (extracted from `ThreadItem`s).
  - F3. **Fallback** if F1 fails: implement tee strategy mirroring Amp (`~/.config/mala/codex-sessions/{thread_id}.jsonl`, append-mode, per-thread keying). Tracked as contingency, not the primary path. Namespaced separately from Amp's `~/.config/mala/amp-sessions/` to avoid thread-id collision.
  - F4. Evidence parsing — extract Bash commands from `CommandExecutionThreadItem.command` + `aggregated_output`; mirror `extract_bash_commands` / `extract_tool_results` / `extract_assistant_text_blocks` semantics.
- **Phase G — Codex MCP integration**:
  - G1. Bundle `mala-locking` MCP launcher inside the Mala-shipped Codex plugin (`plugins/codex/mala-safety/.codex-plugin/.mcp.json`), so the plugin is the single install target for both the safety hook *and* the locking server.
  - G2. Introduce a new `mala-codex-mcp-locking` console script alongside the existing `mala-amp-mcp-locking`; both are thin entry points backed by the same shared module (`src/infra/tools/locking_mcp_stdio.py`), so there is no code duplication. The Codex plugin's `.mcp.json` references `mala-codex-mcp-locking`; the existing Amp script is left untouched to avoid breaking already-installed Amp plugins (whose bundled `.mcp.json` references the old name). The shared module is the right factoring for cross-coder reuse — Phase A4 (provider-owned MCP factory) is the cleanup that removes per-coder dispatch logic without renaming either underlying script. Codex consumes the same stdio launch shape Amp does (stdio JSON-RPC over `command`+`args`+`env`).
  - G3. If the user explicitly sets a different `coder_options.codex.mcp_servers`, **merge** with the bundled `mala-locking` rather than replacing — bundled is mandatory.
- **Phase H — Resume, fixer-follows-main-coder, idle retry**:
  - H1. `AsyncCodex.thread_resume(thread_id)` integration. Validate that resume preserves prior context (no prompt-accumulation fallback needed).
  - H2. Fixer agents follow main coder (decision #7, parity with Amp AC#5): when `coder=codex`, `FixerService` spawns Codex.
  - H3. `IdleTimeoutRetryPolicy` works for Codex unchanged — Codex is async-native and `asyncio.wait_for` already wraps the iteration.
- **Phase I — Prerequisites, telemetry, tests, docs, ship**:
  - I1. `CodexAgentProvider.install_prerequisites()` — verify `codex_app_server` SDK importable + `openai-codex-cli-bin` runtime resolvable + Codex auth state present. Fail-closed on missing deps with actionable error pointing at `uv add openai-codex-app-server-sdk` and Codex auth docs (decision #8).
  - I2. Telemetry — extend `coder` span attribute to include `"codex"` (Phase A generalization makes this trivial).
  - I3. Unit tests with mocked `AsyncCodex`; integration tests with a fake `AsyncAppServerClient`; real-Codex e2e gated on the binary + auth.
  - I4. Docs: `README.md` (new Prerequisites subsection), `docs/cli-reference.md`, `docs/project-config.md`, `docs/architecture.md`, `docs/development.md`.
  - I5. Existing Claude/Amp regression suites must remain green throughout (Phase A landing point in particular gates this).

### Out of Scope (Non-Goals)

- Replacing Claude as the default — `coder: claude` (or whatever `DEFAULT_CODER` resolves to today) remains default.
- Per-issue agent selection — every issue in a single run uses the same coder (parity with Amp).
- Per-fixer coder selection — fixer always follows main coder (decision #7).
- Hot-swap mid-run.
- Modifying the **Cerberus / agent_sdk reviewer** path (`reviewer_type` is independent of `coder`).
- Modifying the **Epic verifier** (`src/infra/epic_verifier.py`); it uses the direct Anthropic API and is out of the agent-SDK pipeline.
- Provider fallback / retry-with-other-coder.
- Cross-coder session resume (Codex `thr_*` ids ≠ Claude session ids ≠ Amp `T-*` thread ids).
- Surfacing `ReasoningThreadItem` content blocks as user-visible `AgentEvent`s in MVP (decision #12, parity with Amp's stripped-thinking stance).
- Codex `thread_fork` / `thread_archive` (decision #10 — resume-only MVP).
- Devcontainer changes — Codex install/auth is documented as a user prerequisite.
- Plan-mode / structured-output / image-input Codex features — MVP is text-only single-turn-per-prompt.
- Codex `ApprovalsReviewer.guardian_subagent` integration — exotic, not needed for our use case.
- Pre-emptively refactoring Claude/Amp call sites of `iter_thread_events` on a separate PR — the migration is part of A5 itself.

## Assumptions & Constraints

### Assumptions

- The `codex_app_server` Python SDK at `../codex/sdk/python/` is the supported entry point. It owns its own `codex app-server` subprocess (we do not spawn `codex` ourselves the way the Amp adapter spawns `amp`). Implication: most adapter complexity from Amp (subprocess lifecycle, stderr ring buffer, SIGTERM/SIGKILL, tee'd JSONL bootstrap) does not recur — Codex is much closer to the Claude SDK in shape.
- Both `Codex` (sync) and `AsyncCodex` (async) clients expose the same surface (`thread_start`, `thread_resume`, `thread_fork`, `thread.run/turn`, `TurnHandle.stream/run/steer/interrupt`). Mala uses `AsyncCodex` exclusively — the orchestrator is async-native. (`Codex.thread_start()` would block the event loop.)
- `AsyncCodex` is initialized lazily on context entry; `async with AsyncCodex() as codex:` is the documented golden path. Implication: `CodexClient.__aenter__` should `async with` the codex client so start/shutdown is paired explicitly.
- Notification-stream `event.method` strings (`turn/started`, `item/agentMessage/delta`, `turn/completed`, etc.) and payload types (`AgentMessageDeltaNotification`, `ItemCompletedNotification`, `TurnCompletedNotification`, `ErrorNotification`, ...) are stable enough to key adapter logic on. SDK is **experimental**; pin the version in `pyproject.toml` and add a real-Codex e2e gate so drift surfaces in CI. (Same posture as the Amp plan's WIP-API stance.)
- `codex_app_server` imports stay lazy so Claude-only and Amp-only users do not need Codex dependencies installed.
- Missing SDK, runtime, or auth fails closed through `CodexNotInstalledError` with actionable setup guidance.
- The `openai-codex-cli-bin` runtime package is platform-specific (mac/linux/windows wheels) and pins to an exact version matching the SDK release. Mala does not vendor the runtime; the user installs it.
- Default Codex model is `gpt-5.5`, subject to the model-id confirmation spike.
- Default Codex `sandbox` is `danger-full-access`.
- Default Codex `approval_policy` is `never`.
- With those defaults, the bundled hook is safety-critical. If it is missing, disabled, untrusted, stale, or not executed, Codex runs abort.
- Resume uses `AsyncCodex.thread_resume(thread_id)` only.
- Evidence uses native `Thread.read(include_turns=True)` unless the Phase F spike proves it insufficient.
- Reasoning items are stripped from runtime events in MVP.
- The Codex SDK exposes structured `Notification` events synchronously enough that idle-timeout detection (`asyncio.wait_for`) works without instrumentation. Implication: `IdleTimeoutStream` (`src/pipeline/message_stream_processor.py:59`) wraps the Codex notification iterator unchanged.
- `MalaConfig` resolution and the orchestration factory accept new `coder` values via the same `Literal` widening + parser update pattern Amp followed; no shape change beyond enum widening is required.
- Codex's hook input/output schemas (`pre-tool-use.command.{input,output}.schema.json`) are stable enough to drive the bundled hook script.
- Codex's plugin discovery directory is `~/.codex/plugins/` (verified by Phase E spike — listed as Open Question if otherwise).

### Implementation Constraints

- **No backward-compatibility shims** (per `CLAUDE.md`): Phase A renames/moves modules; update all callers; no re-export wrappers.
- **No re-exports**: Codex adapter lives under `src/infra/clients/`; no aliasing modules.
- **Lazy SDK imports preserved**: `codex_app_server` imports stay local to the Codex adapter so a Claude-only or Amp-only run does not require the Codex SDK / runtime. Symmetric to existing Claude / Amp lazy-import contracts (`src/infra/clients/amp_provider.py:34-37`, `src/infra/clients/claude_provider.py:11-15`).
- **Layered architecture** (per `docs/architecture.md` and grimp rules):
  - Protocol contracts live in `src/core/protocols/`. Phase A enriches `agent_provider.py`, replaces `sdk.py`'s factory protocol, and replaces `log.py`'s `iter_thread_events` overload (renaming to `evidence.py` per CLAUDE.md no-shim rule).
  - Concrete providers live under `src/infra/clients/`.
  - Pipeline-layer code (`src/pipeline/`) consumes only the protocol — no `if coder == "codex"` branches in pipeline modules. The `agent_provider.name == "amp"` branch in `factory.py:1210` is replaced by provider-owned dispatch in Phase A4.
- **`agent_session_runner.py`, `fixer_service.py`, `run_coordinator.py` stay coder-agnostic**: all coder-specific behavior is hidden behind `AgentProvider`. After Phase A, even `runtime.options` reads are gone.
- **Selection resolver mirrors `claude_settings_sources` / `coder` precedence**: same CLI > env > yaml > default precedence.
- **Existing `mala.yaml` files remain valid**: new fields (`coder_options.codex.*`) are optional with defaults; schema additions never break older configs. Phase A8 generalizes the `coder` Literal but the *default* value is unchanged.
- **Existing Claude and Amp paths are byte-equivalent** after Phase A (regression contract). Phase A's review gate runs the existing Claude integration suite + Amp integration suite as the regression guard, **per PR** (decision #15).
- Extend existing config, validation, orchestration, event, evidence, MCP, and telemetry mechanisms before adding new infrastructure.
- Do not make Codex a hard dependency for non-Codex runs.
- A1 is a single big-bang PR: introduce `AgentEvent`, rewire Claude and Amp adapters, update `MessageStreamProcessor`, and delete `amp_messages.py` together (decision #6).

### Testing Constraints

- Use repo test commands from `tests/AGENTS.md`.
- New code includes unit tests; `tests/CLAUDE.md` rules apply (no over-mocking integration paths).
- Existing Claude and Amp tests remain unchanged where possible and serve as Phase A regression guards.
- Prefer fakes over mocks for protocol boundaries.
- **Real-Codex e2e coverage** required: an e2e test that runs against real `AsyncCodex` with a one-line prompt to catch SDK schema drift. Gated only on (a) `codex_app_server` importable and (b) `openai-codex-cli-bin` runtime available + auth configured.
- **Per-PR cadence** (decision #15): each Phase A PR runs the full Claude + Amp integration suites in CI before merge. If CI cost becomes a bottleneck, downgrade to fast tests per PR + full integration on phase merge via a follow-up.
- **Fake-Codex integration suite**: a `monkeypatch`-injected fake `AsyncAppServerClient` returning canned notifications. Drives the full per-issue lifecycle without real network/cost.
- **Lock-enforcement parity tests** mirror `tests/integration/test_amp_lock_enforcement.py`: allow / no-lock / wrong-agent / wrong-namespace / env-missing / canonicalization-parity. Plus disallowed-tools parity (Codex enforces in-hook; Amp gap closed).
- Coverage threshold remains the repo default, currently enforced by pytest config at 72%.
- Validation must include `uvx ruff check .`, `uvx ty check`, and appropriate pytest subsets before ship.

### Decision Log

Decisions made during Phase 2 interview (2026-05-07). Items marked **(R)** were filled by author recommendation when the interview was cut short; all other items are explicit user choices.

| # | Decision | Rationale | Evidence | Tradeoff / Risk / Follow-up |
|---|----------|-----------|----------|------------------------------|
| 1 | **Phase A scope = full A1–A8.** All eight cleanups land before Codex code is written. | User chose "Full A1-A8". Cleanest result; Codex code lands without inheriting Amp-era compromises. | User answer to "Phase A scope" question. | Bigger pre-Codex blast radius; gated by Claude+Amp regression suites per A* sub-task. |
| 2 | **Default `sandbox` = `danger-full-access`.** Codex runs unsandboxed; safety relies entirely on the bundled `PreToolUse` command hook (parity with Amp's `--dangerously-allow-all` posture). | User explicit override. | User answer to "Sandbox default" question. | The bundled hook is now safety-critical: if it's not loaded, the run must abort. Phase E5 self-test mirrors Amp's plugin-load self-test exactly. **Shell-command write-path enforcement is required to close the lock-bypass gap** (normal shell writes via `>`, `sed -i`, `python -c "open(...,'w')"`, `cp`, `mv`, etc. would otherwise mutate unlocked files without a lock check); see Phase E "Shell-command write-path enforcement" sub-section. The heuristic is defense-in-depth and has acknowledged residual risk (sufficiently obfuscated `python -c` bypasses); follow-up work to tighten enforcement is tracked there. |
| 3 | **Default `approval_policy` = `never`.** No interactive prompts; the hook is the only gate. | Standard for unattended automation. | User answer to "Approval default" question. | Combined with #2, the hook is the single point of safety enforcement; `CodexHookNotActiveError` must be fail-closed. |
| 4 | **Lock-ownership + dangerous-cmd enforcement = bundled `PreToolUse` command hook (Python entry-point), with the hook also enforcing `MALA_DISALLOWED_TOOLS`.** Codex has Claude-Code-style hooks (`codex-rs/hooks/schema/generated/pre-tool-use.command.{input,output}.schema.json` with `decision: approve\|block` / `permissionDecision: allow\|deny\|ask` / `permissionDecisionReason`), so we get pre-tool gating without WIP plugin APIs. The hook script reuses `src/infra/tools/locking.py::lock_path` directly (in-process Python — advantage over Amp's TS plugin which had to reimplement SHA-256 hashing in TypeScript). | User pointed out Codex has both hooks and plugins; verified in `/Users/cyou/code/codex/codex-rs/hooks/schema/generated/`. | `pre-tool-use.command.input.schema.json` carries `cwd, tool_name, tool_input, session_id, turn_id`; output supports `permissionDecision: "allow"\|"deny"\|"ask"` with reason. | Trusted-hash mechanism (Codex's `HookStateToml.trusted_hash`) needs Phase E spike to confirm install path; addressed in #16. |
| 5 | **Hook + locking-MCP packaging = bundled Codex plugin** (`plugins/codex/mala-safety/.codex-plugin/{plugin.json, hooks.json, .mcp.json}`) installed idempotently to `~/.codex/plugins/mala-safety/.codex-plugin/`. | Single install target; reversible; co-locates hook + MCP. Plugin manifest spec confirmed in `codex-rs/skills/src/assets/samples/plugin-creator/references/plugin-json-spec.md`. | User answer to "Hook packaging" question. | Spike confirms the actual Codex plugin discovery path (could be `~/.codex/plugins/`, `~/.codex-plugin/plugins/`, or another location). Tracked as Open Question. |
| 6 | **Phase A1 rollout = single big-bang PR.** Replace Anthropic-class duck typing in `MessageStreamProcessor` + delete `amp_messages.py` + rewire Claude+Amp adapters to emit `AgentEvent`s + add the new event protocol — all in one PR. | User explicit choice. | User answer to "A1 rollout" question. | Higher review burden; harder to bisect; mitigated by running both Claude *and* Amp full integration suites on the PR before merge. |
| 7 | **Fixer agents always follow main coder.** No per-fixer coder selection. | Parity with Amp AC#5; one mental model. | User answer to "Fixer coder" question. | If users later want main=codex/fixer=claude split, it can be added behind a coder_options field. |
| 8 | **Missing-dep behavior = fail-closed with actionable error.** `CodexAgentProvider.install_prerequisites()` raises `CodexNotInstalledError` (parity with `AmpPluginNotActiveError`) pointing at `uv add openai-codex-app-server-sdk` and Codex auth docs. | Parity with Amp's binary-install fail-closed posture. | User answer to "Missing deps" question. | No silent fallback to claude. |
| 9 | **Default model = `gpt-5.5`.** Used when neither `--codex-model` nor `coder_options.codex.model` is set. | User explicit choice (newer than gpt-5.4 which the SDK examples reference). | User answer to "Default model" question. | Spike confirms the exact model id Codex accepts (e.g., `gpt-5.5` vs `gpt-5.5-codex` vs another tag); failure surfaces during real-Codex e2e. |
| 10 | **Resume = `thread_resume` only.** `CodexClient.with_resume(thread_id)` maps to `AsyncCodex.thread_resume(thread_id)`; no fork, no archive in MVP. | Parity with Amp's resume-only stance. | User answer to "Resume strategy" question. | Out-of-MVP fork/archive can be added without protocol change. |
| 11 | **Evidence = native `Thread.read(include_turns=True)`.** Codex SDK's `Thread.read` is the canonical evidence source; no tee'd JSONL infrastructure. | User explicit choice; cleaner than Amp's tee strategy. | User answer to "Evidence source" question; SDK API at `../codex/sdk/python/docs/api-reference.md`. | Phase F1 spike must confirm: (a) `read(include_turns=True)` returns command output for `CommandExecutionThreadItem.aggregated_output` so lint evidence is observable; (b) read cost is bounded enough for repeated gate calls; (c) the call works after `thread_resume` and reflects events from both invocations. **Risk**: if `read()` is paginated or doesn't expose what we need, we have to reverse this decision and add tee fallback, costing a phase. Mitigated by F1 spike landing before C/D/E/F so the choice is validated early. |
| 12 | **Reasoning items stripped in MVP.** `ReasoningTextDelta`, `ReasoningSummary*Notification`, `ReasoningThreadItem` are tee'd to disk for diagnostics but not surfaced as `AgentEvent`s. | Parity with Amp's stripped-thinking decision. | User answer to "Reasoning display" question. | Can add `AgentReasoningEvent` later without protocol break. |
| 13 | **Effort mapping = pass-through with strict validation.** `MalaConfig.effort` flows to Codex's `ReasoningEffort` enum; values validated against the Codex enum at config-parse time; invalid values fail fast with the supported list. | User explicit choice. | User answer to "Effort mapping" question. | Phase B spike confirms the exact `ReasoningEffort` enum values; `validate_codex_effort` parser added next to `validate_amp_effort_for_mode` in `src/core/constants.py`. |
| 14 | **A4 = `AgentProvider.mcp_server_factory()` method.** `factory.py:1210` and `orchestrator.py:404` lose their `if agent_provider.name == "amp"` branches; orchestrator calls the provider method directly. | User explicit choice; matches Phase A's "no `if name`" goal. | User answer to "A4 ownership" question. | After A4, `create_mcp_server_factory` / `create_amp_mcp_server_factory` / new Codex factory in `orchestration_wiring.py` are referenced *only* through the providers, not by name dispatch. |
| 15 | **(R) Regression cadence = per-PR.** Each A1–A8 PR runs the full Claude + Amp integration suites in CI before merge. | Catches regressions where bisection is cheap (one PR back); aligned with mala's no-shim refactor posture; avoids "the suite was broken three PRs ago" debugging. | Author recommendation; user did not choose explicitly. | Per-PR CI cost is non-trivial; if it becomes a bottleneck, Phase A can downgrade to "fast tests per PR, full integration on phase merge" via a follow-up. |
| 16 | **(R) Hook trust = auto-trust via `install_prerequisites`.** Mala writes the expected `trusted_hash` into Codex's hook-state file during plugin install. | User-side ergonomics: a manual trust step is easily skipped, leaving the hook silently dormant — same failure mode the Amp plugin self-test was designed to prevent. Auto-trust + self-test catches "hook installed but Codex didn't load it" together. | Author recommendation; `HookStateToml.trusted_hash` confirmed in `codex-rs/core/config.schema.json:1016`. | Phase E spike must reverse-engineer Codex's hook-state file location and format. **Risk**: if Codex's trust UX requires user-interactive acceptance, auto-trust is impossible — fall back to a documented one-time step. Tracked as Open Question. |

## Integration Analysis

### Existing Mechanisms Considered

| Existing Mechanism | Could Serve Feature? | Decision | Rationale |
|---|---|---|---|
| `AgentProvider` protocol (`src/core/protocols/agent_provider.py`) | Yes | **Reuse + extend in Phase A** | Already the abstraction line. Phase A8 widens `name: Literal["claude", "amp"]` to include `"codex"`; Phase A4 adds `mcp_server_factory()`; Phase A5 renames `log_provider` to `evidence_provider`. |
| `CoderRuntimeBuilder` protocol (same file) | Yes | **Reuse, narrow fluent API in Phase A6** | Already opaque-payload-shaped. Phase A6 narrows the cross-coder set so Codex doesn't inherit Claude-only methods as no-ops. |
| `SDKClientProtocol` (`src/core/protocols/sdk.py:28-71`) | Yes | **Reuse** | `__aenter__/__aexit__/query/receive_response/disconnect` maps cleanly onto `AsyncCodex` lifecycle. |
| `SDKClientFactoryProtocol` (`src/core/protocols/sdk.py:74-165`) | No (in current shape) | **Replace via slimmer protocol in Phase A2** | `create_options(...)` and `create_hook_matcher(...)` are Claude-only. Move them off the protocol; protocol keeps only `create(runtime)` + `with_resume(runtime, resume)`. |
| `MessageStreamProcessor` (`src/pipeline/message_stream_processor.py`) | Partial | **Replace duck-typing in Phase A1** | Currently keys off Anthropic class names. Replace with `AgentEvent` discriminator. Claude/Amp adapters emit the new events; Codex adapter follows directly. |
| `LogProvider` (`src/core/protocols/log.py`) | Partial | **Generalize in Phase A5** | `iter_thread_events` was an Amp escape hatch. Phase A5 introduces an `EvidenceProvider` typed protocol (renamed to `evidence.py`) that providers implement directly. Claude and Amp implementations are renamed but unchanged behaviorally; Codex adds the third. |
| `AgentRuntimeBuilder` (`src/infra/agent_runtime.py:73`) | Partial | **Narrow cross-coder surface in Phase A6** | Move Claude-only fluent methods off the cross-coder shape onto `ClaudeAgentRuntimeBuilder`. Mechanical refactor; no behavior change. |
| `McpServerFactory` callable (`src/core/protocols/sdk.py:24`) + `locking_mcp` (`src/infra/tools/locking_mcp.py`) | Yes | **Reuse via Phase A4 ownership** | Codex adapter uses the same `mala-locking` server. Phase A4 makes provider-owned dispatch the standard. |
| `make_*_hook` family (`src/infra/hooks/`) + `src/infra/tools/locking.py` | Partial | **Codex inherits Claude-Code-style hook semantics; reuse `lock_path`** | Codex has its own `PreToolUse` command-hook surface. The bundled `mala-codex-pre-tool-use` script reuses `src/infra/tools/locking.lock_path` and `src/infra/hooks/dangerous_commands` logic in-process — advantage over Amp's TS plugin which had to reimplement SHA-256 hashing. |
| `mala.yaml` schema (`src/domain/validation/config.py:846+`) | Yes | **Extend** | Add `coder: codex` to enum; add `coder_options.codex.*` block. Strict-enum validation as today. |
| `MalaConfig` (`src/infra/io/config.py`) | Yes | **Extend** | Widen `coder` Literal; add `CoderOptions.codex` companion to `CoderOptions.amp`; add `parse_codex_*` parsers; widen resolver chain. |
| CLI (`src/cli/cli.py`) | Yes | **Extend** | Widen `--coder` typer enum; add `--codex-model` / `--codex-effort`. Other codex_options yaml-only for MVP. |
| `IdleTimeoutRetryPolicy` (`src/pipeline/idle_retry_policy.py`) | Yes | **Reuse** | Idle timeout still applies to async iteration; Codex notification stream is wrapped by `IdleTimeoutStream` unchanged. |
| `MalaEventSink` + `coder` span attribute | Yes | **Extend** | Already exists; widen the attribute domain to include `"codex"`. Non-breaking. |
| Amp plugin env contract and lock tests | Partial | **Reuse patterns** | Codex hook should honor `MALA_AGENT_ID`, `MALA_LOCK_DIR`, `MALA_REPO_NAMESPACE`, and `MALA_DISALLOWED_TOOLS`. |
| Amp plugin install pattern (`src/infra/clients/amp_plugin_installer.py`) | Yes | **Reuse the pattern, not the code** | Codex plugin installer mirrors the idempotent install + version-marker + selftest pattern, but installs to Codex's plugin dir with Codex's plugin manifest shape. |
| `IssueCoordinator`, `IssueExecutionCoordinator`, `RunCoordinator`, `FixerService`, `AgentSessionRunner` | Yes | **Reuse unchanged after Phase A** | All consume `AgentProvider` already. Phase A removes the residual `runtime.options` direct-read; afterwards these are fully coder-agnostic. |
| Reviewer / epic verifier (`src/pipeline/review_runner.py`, `src/infra/epic_verifier.py`) | No | **Leave unchanged** | Independent of `coder` choice. |

### Integration Approach

The existing `AgentProvider` triple (`client_factory` + per-session `runtime_builder` + cross-run `evidence_provider`) plus the Phase A cleanups give Codex a clean fit:

1. **Phase A** removes the Anthropic-shape duck-typing, the Claude-vocab leaks in `SDKClientFactoryProtocol`, the per-coder `if name == "amp"` branches, and the `iter_thread_events` escape hatch. After Phase A, the Claude path and the Amp path are byte-equivalent to today; the protocol surface area is smaller and provider-owned.
2. **Phase B** widens the `coder` enum end-to-end (CLI / env / yaml / `MalaConfig` / `ValidationConfig` / `_create_agent_provider`) and adds a `CodexAgentProvider` stub.
3. **Phases C–F** fill in the stub:
   - C: `CodexClient` wraps `AsyncCodex` + thread/turn lifecycle behind `SDKClientProtocol`.
   - D: notification → `AgentEvent` adapter (Phase A1's protocol).
   - E: safety model — bundled plugin `PreToolUse` hook + locking MCP, sandbox + approval_policy as defense-in-depth.
   - F: evidence provider — native `Thread.read(include_turns=True)` preferred, tee fallback only if F1 spike disconfirms.
4. **Phases G–H** wire MCP (G — bundled in the same plugin as the safety hook), resume (H1), fixer-follows-main (H2), and idle retry (H3, no-op).
5. **Phase I** ships: prerequisites, telemetry, real-Codex e2e gate, docs.

The only new infrastructure is Codex-specific where no existing component can serve the SDK or runtime contract: `CodexClient`, `CodexRuntimeBuilder`, `CodexAgentProvider`, `CodexEvidenceProvider`, the bundled Codex plugin, and the `mala-codex-pre-tool-use` hook script. Existing locking, dangerous-command detection, config resolution, validation, MCP factory wiring, event processing, idle retry, fixer, telemetry, and docs systems are extended.

## Prerequisites

- [ ] Phase A has landed before any real Codex execution work begins.
- [ ] Existing Claude **and** Amp test suites green before each Phase A PR (decision #15).
- [ ] `openai-codex-app-server-sdk` Python package installable via `uv add openai-codex-app-server-sdk` (or pinned via `pyproject.toml` extras `mala[codex]`). Lazy-imported only on `coder=codex` runs.
- [ ] `openai-codex-cli-bin` runtime package installed (the SDK pins an exact version; we follow the pin).
- [ ] User's local Codex auth/session configured (per Codex docs). Codex will fail at `Codex()` initialization with an actionable error if auth is missing; `CodexAgentProvider.install_prerequisites()` surfaces this as a Mala-shaped error pointing at Codex docs.
- [ ] Maintainers accept that `CodexAgentProvider.install_prerequisites()` installs or updates the mala Codex plugin under the Codex user plugin directory.
- [ ] Maintainers accept auto-writing Codex hook trust state once the hook-state location is confirmed (decision #16).
- [ ] Real Codex e2e tests have skip conditions for missing SDK, runtime, or auth.
- [ ] `mala.yaml` files validated as backwards-compatible (defaults preserve Claude/Amp behavior).
- [ ] No other in-flight refactoring touches `SDKClientFactoryProtocol`, `MessageStreamProcessor`, `LogProvider`, or `AgentRuntimeBuilder` in a way that would conflict with Phase A. (Quick branch survey before A1 starts.)

## High-Level Approach

The plan delivers Codex as a third coder over nine phases. **Phase A is a no-feature refactor** that addresses eight specific architecture friction points the Amp work surfaced (A1–A8); each sub-task lands as its own PR gated by the full Claude+Amp integration suites in CI (decision #15). After Phase A, the pipeline consumes coder-agnostic `AgentEvent`s instead of Anthropic-shaped duck-typed messages (A1, decision #6 — single big-bang PR), the `SDKClientFactoryProtocol` no longer leaks Claude vocabulary (A2), MCP factory dispatch lives on `AgentProvider.mcp_server_factory()` (A4, decision #14), evidence is surfaced through a typed `EvidenceProvider` protocol instead of the `iter_thread_events` overload (A5), and the runtime fluent surface only carries cross-coder methods (A6).

**Phases B–I then implement Codex on top of the cleaner abstraction.** Codex's safety story is hook-shaped, not Amp-shaped: it has Claude-Code-style `PreToolUse` command hooks (`decision: approve|block` / `permissionDecision: allow|deny|ask`) configured via TOML or via plugin packaging. Mala ships `plugins/codex/mala-safety/.codex-plugin/{plugin.json, hooks.json, .mcp.json}` (decision #5), installed idempotently to Codex's user-plugin directory. The hook is a Python entry-point (`mala-codex-pre-tool-use`) that reuses `src/infra/tools/locking.lock_path` directly — no cross-language hashing reimpl needed. With `sandbox=danger-full-access` and `approval_policy=never` (decisions #2 and #3), the hook is the single safety gate; a plugin-load self-test on every Codex run mirrors Amp's `AmpPluginNotActiveError` pattern and aborts fail-closed if the hook didn't actually load (decision #16 — auto-trust via `install_prerequisites`). Evidence is read natively via `Thread.read(include_turns=True)` (decision #11), with a tee fallback only if the F1 spike disconfirms native viability. Default coder model is `gpt-5.5` (decision #9). Resume uses `AsyncCodex.thread_resume(thread_id)` only (decision #10). Reasoning items are stripped from the event stream (decision #12, parity with Amp). Fixers always follow the main coder (decision #7). Missing SDK / runtime / auth fails closed with an actionable error (decision #8).

## Phase Dependencies

```
                 ┌──────────────────┐
                 │ Phase A (A1–A8)  │   eight PRs, each gated by Claude+Amp regression
                 └─────────┬────────┘
                           ▼
                 ┌──────────────────┐
                 │ Phase B (B1–B3)  │   selection + provider stub
                 └─────────┬────────┘
                           ▼
                 ┌──────────────────┐    ┌──────────────────┐
                 │ Phase C (C1–C5)  │    │ Phase F1 (spike) │   may run in parallel with C
                 │ CodexClient      │    │ Thread.read()    │   F1 result feeds F2/F3 path
                 └────┬─────────┬───┘    └────┬─────────────┘
                      ▼         ▼             ▼
              ┌───────────┐ ┌──────────────┐ ┌──────────────┐
              │ Phase D   │ │ Phase E      │ │ Phase F2/F3  │
              │ events    │ │ safety plugin│ │ evidence prov│
              └─────┬─────┘ └──────┬───────┘ └──────┬───────┘
                    │              ▼                │
                    │       ┌──────────────┐        │
                    │       │ Phase G       │       │
                    │       │ MCP integ     │       │
                    │       └──────┬────────┘       │
                    └──────────────┼────────────────┘
                                   ▼
                          ┌──────────────────┐
                          │ Phase H (H1–H3)  │   resume + fixer + idle
                          └────────┬─────────┘
                                   ▼
                          ┌──────────────────┐
                          │ Phase I (I1–I5)  │   prereqs + telemetry + tests + docs + ship
                          └──────────────────┘
```

| Phase | Blocks | Dependency Notes |
|-------|--------|------------------|
| Phase A | B-I | All Codex phases depend on the cleaned provider/event/evidence/runtime contracts. |
| A1 | C, D, F | Codex event mapping depends on `AgentEvent`. |
| A2 | C | `CodexClient` factory cannot inherit Claude knobs. |
| A3 | C, D | Pipeline calls `provider.client_factory.create(runtime)` rather than `create(runtime.options)`. |
| A4 | G | Codex MCP integration depends on provider-owned MCP factory selection. |
| A5 | F, H, I | Codex evidence and validation gates depend on `EvidenceProvider`. |
| A6 | C | Cross-coder `CoderRuntimeBuilder` shape must be Claude-free before Codex builds on it. |
| A7 | H, I | Lifecycle effects must be provider-neutral before Codex resume/retry/ship. |
| A8 | B | Codex selection depends on generalized coder naming and validation infrastructure. |
| Phase B | C-I | Later phases need `CodexAgentProvider` scaffold and config objects. |
| Phase C | D, E, F, H | Safety, evidence, mapping, and resume need a Codex client/runtime. |
| Phase D | F, I | Evidence extraction and e2e validation depend on stable item-to-event mapping. |
| Phase E | G, I | Real Codex execution should not ship until safety is fail-closed. |
| F1 spike | F2/F3 | Spike result decides native vs tee fallback. |
| Phase F | H, I | Resume/retry and validation gates require evidence across turns/resumes. |
| Phase G | H, I | Locking MCP must be available before full lifecycle runs are declared ready. |
| Phase H | I | Ship readiness depends on resume, fixer, and idle retry behavior. |
| Phase I | Release | Final docs, telemetry, prerequisites, and complete validation. |

Soft ordering (parallelizable):
- D and E can land in parallel after C1+C2.
- F1 spike can run in parallel with C/D scaffolding.
- Documentation (I4) can be drafted in parallel with E/F/G but merges last.

## Technical Design

### Architecture (post-Phase A)

```
                  ┌──────────────────────────────────────┐
   CLI / env / yaml ─→  MalaConfig.coder ∈ {claude, amp, codex}
                       MalaConfig.coder_options.{amp,codex}
                  └────────────────────┬─────────────────┘
                                       ▼
                  ┌──────────────────────────────────────┐
                  │ src/orchestration/factory.py          │
                  │  picks AgentProvider                  │
                  │  calls install_prerequisites()        │
                  │  picks MCP factory via provider hook  │  ← Phase A4
                  └────────────────────┬─────────────────┘
                                       │
        ┌──────────────────┬───────────┼───────────────────┐
        ▼                  ▼           ▼                   ▼
   ClaudeAgentProvider  AmpAgentProvider  CodexAgentProvider
   ─ client_factory     ─ client_factory  ─ client_factory
   ─ runtime_builder    ─ runtime_builder ─ runtime_builder
   ─ evidence_provider  ─ evidence_provider ─ evidence_provider   ← Phase A5 (renamed/typed)
   ─ install_prereq()   ─ install_prereq() ─ install_prereq()
   ─ mcp_server_factory ─ mcp_server_factory ─ mcp_server_factory ← Phase A4
        │                  │                  │
        ▼                  ▼                  ▼
   ClaudeSDKClient     AmpClient (subprocess)   CodexClient (AsyncCodex wrapper)
        │                  │                  │
        └──────────────────┴───────────┬──────┘
                                       ▼
                  ┌──────────────────────────────────────┐
                  │ MessageStreamProcessor (consumes      │
                  │  AgentEvent, not Anthropic classes)   │  ← Phase A1
                  └──────────────────────────────────────┘
                  │ AgentSessionRunner / FixerService /   │
                  │ RunCoordinator (coder-agnostic)       │
                  └──────────────────────────────────────┘
```

Codex-specific behavior stays in `src/infra/clients/codex_*` and the bundled `plugins/codex/mala-safety/` package. Shared orchestration and pipeline code consumes only provider protocols.

### Data Model

```python
# Phase A1 — coder-agnostic event protocol (replaces Anthropic-class-name duck typing)
# src/core/protocols/agent_event.py (NEW)

@runtime_checkable
class AgentEvent(Protocol):
    kind: Literal["text", "tool_use", "tool_result", "result"]

@dataclass(frozen=True)
class AgentTextEvent:
    kind: Literal["text"] = "text"
    text: str = ""

@dataclass(frozen=True)
class AgentToolUseEvent:
    kind: Literal["tool_use"] = "tool_use"
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AgentToolResultEvent:
    kind: Literal["tool_result"] = "tool_result"
    tool_use_id: str = ""
    is_error: bool = False
    content: object = None

@dataclass(frozen=True)
class AgentResultEvent:
    kind: Literal["result"] = "result"
    session_id: str = ""
    is_error: bool = False
    subtype: str = ""
    result: object = None
```

```python
# Phase B — Codex coder options
# src/infra/io/config.py

@dataclass(frozen=True)
class CodexOptions:
    model: str = "gpt-5.5"  # decision #9
    effort: str | None = None  # validated against ReasoningEffort enum at parse time (decision #13)
    approval_policy: Literal["never", "on-request", "on-failure", "untrusted"] = "never"  # decision #3
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "danger-full-access"  # decision #2
    mcp_servers: Mapping[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class CoderOptions:
    amp: AmpOptions = field(default_factory=AmpOptions)
    codex: CodexOptions = field(default_factory=CodexOptions)

# coder widened to include "codex"
@dataclass(frozen=True)
class MalaConfig:
    coder: Literal["claude", "amp", "codex"] = DEFAULT_CODER
    coder_options: CoderOptions = field(default_factory=CoderOptions)
    # ...
```

```python
# Phase C — Codex runtime (per-session)
# src/infra/clients/codex_runtime.py (NEW)

@dataclass(frozen=True)
class CodexRuntime:
    cwd: Path
    agent_id: str
    model: str
    effort: str | None
    approval_policy: AskForApproval | None  # mapped from CodexOptions
    sandbox: SandboxMode | None
    base_instructions: str | None
    mcp_servers: dict[str, object]  # bundled mala-locking + user merges (Phase G3)
    env: Mapping[str, str]
    resume_thread_id: str | None = None
    lint_cache: LintCache  # parity with Phase A6 cross-coder runtime contract
    # NOT carrying argv/env/log_path the way AmpRuntime does — Codex
    # SDK manages the subprocess; we configure via thread_start params.
```

**Resolution precedence** (mirrors `claude_settings_sources` / `amp_mode` resolver pattern):

| Setting | CLI | Env | YAML | Default |
|---|---|---|---|---|
| Coder | `--coder codex` | `MALA_CODER=codex` | `coder: codex` | unchanged from current default |
| Codex model | `--codex-model gpt-5.5` | `MALA_CODEX_MODEL=gpt-5.5` | `coder_options.codex.model: gpt-5.5` | `gpt-5.5` |
| Codex effort | `--codex-effort high` | `MALA_CODEX_EFFORT=high` | `coder_options.codex.effort: high` | None (SDK default) |
| Codex approval policy | yaml-only for MVP | yaml-only | `coder_options.codex.approval_policy: never` | `never` |
| Codex sandbox | yaml-only for MVP | yaml-only | `coder_options.codex.sandbox: danger-full-access` | `danger-full-access` |

YAML shape:

```yaml
coder: codex
coder_options:
  codex:
    model: gpt-5.5
    effort: high
    approval_policy: never
    sandbox: danger-full-access
```

### API/Interface Design

**Phase A1 — `MessageStreamProcessor` consumes `AgentEvent`**:

```python
# Updated processing loop (no class-name duck typing)
async for event in stream:
    if event.kind == "text":
        on_agent_text(issue_id, event.text)
    elif event.kind == "tool_use":
        record_tool_use(event.id, event.name, event.input)
    elif event.kind == "tool_result":
        record_tool_result(event.tool_use_id, event.is_error)
    elif event.kind == "result":
        record_result(event)
```

Adapters:
- `ClaudeSDKClient.receive_response()` translates Anthropic SDK messages → `AgentEvent`s.
- `AmpClient.receive_response()` translates Amp tee'd JSONL events → `AgentEvent`s directly (synthesized Anthropic dataclasses in `amp_messages.py` deleted in A1).
- `CodexClient.receive_response()` translates Codex notifications/items → `AgentEvent`s.

**Phase A2 — `SDKClientFactoryProtocol` slimmed**:

```python
@runtime_checkable
class SDKClientFactoryProtocol(Protocol):
    def create(self, runtime: object) -> SDKClientProtocol: ...
    def with_resume(self, runtime: object, resume: str | None) -> object: ...

# Claude-only knobs move onto a Claude-private factory class:
class ClaudeSDKClientFactory:  # private to claude_provider.py
    def create_options(self, *, cwd, permission_mode, ...) -> ClaudeAgentOptions: ...
    def create_hook_matcher(self, ...) -> HookMatcher: ...
    # plus the slim protocol surface
```

**Phase A4 — provider-owned MCP factory**:

```python
class AgentProvider(Protocol):
    name: Literal["claude", "amp", "codex"]
    client_factory: SDKClientFactoryProtocol
    evidence_provider: EvidenceProvider

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
    ) -> None: ...

    def mcp_server_factory(self) -> McpServerFactory:
        """Return the coder-shaped MCP factory.

        Replaces the orchestrator-side `if provider.name == "amp"` branch.
        Each provider knows whether it consumes in-process Server objects
        (Claude) or stdio launch specs (Amp / Codex).
        """
```

**Phase A5 — `EvidenceProvider` replaces `LogProvider.iter_thread_events`**:

```python
@runtime_checkable
class EvidenceProvider(Protocol):
    def iter_session_events(self, session_id: str, offset: int = 0) -> Iterator[ParsedEvent]:
        """Per-attempt events; offset is honored."""
    def iter_thread_evidence(self, thread_id: str) -> Iterator[ParsedEvent]:
        """Cross-attempt evidence for validation gates; reads from session start."""
    def extract_bash_commands(self, events: Iterable[ParsedEvent]) -> list[BashEvidence]: ...
    def extract_tool_results(self, events: Iterable[ParsedEvent]) -> list[ToolResultEvidence]: ...
    def extract_assistant_text_blocks(self, events: Iterable[ParsedEvent]) -> list[str]: ...
    def extract_tool_result_content(self, ev: ToolResultEvidence) -> str: ...
```

**Phase C — `CodexClient`**:

```python
class CodexClient:
    """SDKClientProtocol implementation backed by codex_app_server.AsyncCodex."""

    def __init__(self, runtime: CodexRuntime) -> None: ...

    async def __aenter__(self) -> Self:
        # Lazy-import codex_app_server; construct AsyncCodex; enter context.
        ...

    async def query(self, prompt: str, session_id: str | None = None) -> None:
        # 1. If runtime.resume_thread_id and not yet resumed: thread_resume(resume_thread_id)
        #    else: thread_start(model=, effort=, approval_policy=, sandbox=, mcp_servers=, ...)
        # 2. thread.turn(TextInput(prompt))  (returns AsyncTurnHandle)
        # 3. self._stream = turn.stream()
        ...

    def receive_response(self) -> AsyncIterator[AgentEvent]:
        # async-for over self._stream:
        #   ItemStarted/Completed for AgentMessage   -> AgentTextEvent (D1)
        #   ItemStarted for CommandExecution         -> AgentToolUseEvent("bash", ...) (D2)
        #   ItemCompleted for CommandExecution       -> AgentToolResultEvent
        #   ItemStarted for FileChange               -> AgentToolUseEvent("file_change", ...) (D3)
        #   ItemCompleted for FileChange             -> AgentToolResultEvent
        #   ItemStarted for McpToolCall              -> AgentToolUseEvent(<tool>, ...) (D4)
        #   ItemCompleted for McpToolCall            -> AgentToolResultEvent
        #   ReasoningTextDelta                       -> diagnostic-only (tee, no event) (D5)
        #   TurnCompleted                            -> AgentResultEvent (D6)
        #   ErrorNotification                        -> AgentResultEvent(is_error=True) (D7)
        ...

    async def disconnect(self) -> None:
        # await TurnHandle.interrupt()  (if a turn is active)
        # await AsyncCodex.close()
        ...

    @property
    def session_id(self) -> str | None:  # Codex thread id ("thr_...")
        ...
```

`CodexClient.query()` starts or resumes a Codex thread, creates a turn from `TextInput(prompt)`, and stores the active turn handle. `CodexClient.receive_response()` consumes `TurnHandle.stream()` and yields `AgentEvent`s. `disconnect()` interrupts the active turn if needed and closes `AsyncCodex`.

### Phase A Sub-Task Details

Each sub-task lands as its own PR (decision #15: per-PR Claude+Amp regression suite gate; A1 is the exception — single big-bang PR per decision #6).

#### A1 — Replace Anthropic-class-name duck typing with `AgentEvent`

**Rationale.** `MessageStreamProcessor` (`src/pipeline/message_stream_processor.py:230-292`) currently keys on `type(message).__name__ == "AssistantMessage" | "ToolUseBlock" | "ToolResultBlock"` and `getattr`-reads Anthropic-shaped fields. To make Amp work, `src/infra/clients/amp_messages.py` fabricates synthetic dataclasses that imitate the Anthropic shape so the duck typing falls through. Codex's items (`AgentMessageThreadItem`, `CommandExecutionThreadItem`, `FileChangeThreadItem`, `McpToolCallThreadItem`, `ReasoningThreadItem`) don't fit that shape; another synthetic adapter would be lossy. A1 introduces a coder-agnostic event protocol with explicit `kind` discriminators; both adapters emit the new events directly; the synthetic Anthropic-shape dataclasses are deleted in the same PR.

**Files touched.**
- New: `src/core/protocols/agent_event.py` (protocol + dataclasses).
- Edit: `src/pipeline/message_stream_processor.py` (branch on `event.kind`).
- Edit: `src/infra/sdk_adapter.py` (`ClaudeSDKClient.receive_response()` emits `AgentEvent`s).
- Edit: `src/infra/clients/amp_client.py` (`AmpClient.receive_response()` emits `AgentEvent`s directly).
- **Delete**: `src/infra/clients/amp_messages.py` (synthetic Anthropic shape no longer needed).
- New: `tests/unit/core/protocols/test_agent_event.py`, `tests/unit/pipeline/test_message_stream_processor_kinds.py`.

**Regression contract.** Full Claude integration suite + full Amp integration suite must pass on the PR. Behavior diff for either path = refactor bug. Single big-bang PR (decision #6); no "old + new in parallel" intermediate state. No `type(message).__name__` Anthropic checks remain in the processor; lint command detection still works against the new event stream.

#### A2 — Slim `SDKClientFactoryProtocol`

**Rationale.** `src/core/protocols/sdk.py:74-165` carries Claude-only methods (`create_options(permission_mode=, hooks=, mcp_servers=, setting_sources=, disallowed_tools=, ...)`, `create_hook_matcher(...)`). Amp's `_AmpClientFactory` raises `NotImplementedError` on most of them (`amp_provider.py:527-560`). Codex would add a third `NotImplementedError` block. A2 keeps only `create(runtime)` + `with_resume(runtime, resume)` on the cross-coder protocol; Claude-only knobs move onto a Claude-private factory class.

**Files touched.**
- Edit: `src/core/protocols/sdk.py` (slim protocol).
- Edit: `src/infra/sdk_adapter.py` (Claude-private factory class).
- Edit: `src/infra/clients/claude_provider.py` (private Claude factory wiring).
- Edit: `src/infra/clients/amp_provider.py` (drop `NotImplementedError` walls).
- New: `tests/unit/core/protocols/test_sdk_factory_slim.py`.

**Regression contract.** Claude + Amp suites pass. Protocol-shape test asserts no Claude-only methods on the cross-coder protocol. All in-tree call sites compile against the slim protocol.

#### A3 — Make `runtime.options` opaque

**Rationale.** `src/pipeline/agent_session_runner.py:445` reads `runtime.options` directly, forcing `AmpRuntimeBuilder.build()` to construct a synthetic `AmpClientOptions` (`amp_runtime.py:354-368`). Codex doesn't have an "options" object in this sense; A3 makes the pipeline call `provider.client_factory.create(runtime)`, where each provider unpacks its own runtime privately.

**Files touched.**
- Edit: `src/pipeline/agent_session_runner.py` (remove direct `runtime.options` read).
- Edit: `src/pipeline/fixer_service.py` (same).
- Edit: `src/infra/clients/amp_runtime.py` (drop pre-built `options` field).
- Edit: `src/infra/clients/amp_provider.py` (factory unpacks runtime privately).
- Edit: `src/infra/clients/claude_provider.py` (factory `create(runtime)` unpacks runtime → options internally).

**Regression contract.** Claude + Amp suites pass; `runtime.options` field is no longer referenced outside provider-private code. Amp no longer constructs fake Anthropic-style options just to satisfy the pipeline.

#### A4 — Provider-owned MCP factory dispatch

**Rationale.** `src/orchestration/factory.py:1210` and `src/orchestration/orchestrator.py:404` branch on `if agent_provider.name == "amp"`. Adding Codex would make this an N-way switch in two places. A4 moves dispatch onto `AgentProvider.mcp_server_factory()` (decision #14); orchestrator just calls the method.

**Files touched.**
- Edit: `src/core/protocols/agent_provider.py` (add `mcp_server_factory()` method).
- Edit: `src/orchestration/factory.py:1210` (delete `name == "amp"` branch; call `provider.mcp_server_factory()`).
- Edit: `src/orchestration/orchestrator.py:404` (same).
- Edit: `src/orchestration/orchestration_wiring.py` (provider-owned wiring).
- Edit: `src/infra/clients/claude_provider.py`, `src/infra/clients/amp_provider.py` (implement `mcp_server_factory()`).
- New: `tests/unit/orchestration/test_factory_no_name_branch.py` (asserts no `provider.name` checks remain in factory/orchestrator).

**Regression contract.** Claude + Amp suites pass; grep-based test fails CI if any `provider.name == "amp"` branch resurfaces outside provider construction.

#### A5 — `EvidenceProvider` replaces `LogProvider.iter_thread_events`

**Rationale.** `src/core/protocols/log.py`'s `iter_thread_events` was added as an Amp escape hatch (offset honored on Claude, ignored on Amp). Codex stores threads via `Thread.read(include_turns=True)` rather than tailed JSONL. A5 introduces a typed `EvidenceProvider` protocol that providers implement directly: Claude reads JSONL, Amp reads tee'd JSONL, Codex reads via `Thread.read(include_turns=True)`. File renamed `log.py` → `evidence.py` per CLAUDE.md no-shim rule.

**Files touched.**
- New: `src/core/protocols/evidence.py`.
- **Delete**: `src/core/protocols/log.py` (no re-export shim).
- Edit: `src/core/protocols/agent_provider.py` (rename `log_provider` → `evidence_provider`).
- Edit: `src/infra/clients/claude_provider.py`, `src/infra/clients/amp_provider.py` (rename + conform).
- Edit: `src/infra/clients/amp_log_provider.py`, `src/infra/io/session_log_parser.py` (conform to new protocol).
- Edit: all consumers of `log_provider` / `iter_thread_events` (audit via Grep before PR).
- New: `tests/unit/core/protocols/test_evidence_provider.py`.

**Regression contract.** Claude + Amp suites pass. Claude offset behavior remains honored; Amp evidence behavior remains unchanged; validation extracts the same bash commands, tool results, assistant text, and tool-result content as before. Cross-resume invariant for `iter_thread_evidence` test mirrors current behavior.

#### A6 — Narrow `AgentRuntimeBuilder` cross-coder fluent surface

**Rationale.** `AgentRuntimeBuilder` (`src/infra/agent_runtime.py:73`) exposes Claude-only fluent calls (`with_setting_sources`, `with_hooks(deadlock_monitor, include_*hook=...)`); Amp's builder records most as no-ops (`amp_runtime.py:164-252`). A6 narrows the cross-coder shape to `with_resume`, `with_agent_timeout`, `with_env`, `with_lint_tools`, `with_mcp`; Claude-only calls move onto `ClaudeAgentRuntimeBuilder`.

**Files touched.**
- Edit: `src/core/protocols/agent_provider.py` (`CoderRuntimeBuilder` shape).
- Edit: `src/infra/agent_runtime.py` (split `ClaudeAgentRuntimeBuilder` from cross-coder builder).
- Edit: `src/infra/clients/amp_runtime.py` (drop no-op fluent calls).
- Edit: Claude runtime builder implementation and all runtime builder call sites.

**Regression contract.** Claude + Amp suites pass. No `with_setting_sources` / `with_hooks(...)` calls remain on the cross-coder protocol. Claude still receives setting sources and hook configuration through `ClaudeAgentRuntimeBuilder`; Amp no-op methods disappear; pipeline code only calls cross-coder builder methods.

#### A7 — Decouple `WAIT_FOR_LOG` from filesystem assumptions

**Rationale.** Lifecycle effects assume an on-disk session log to wait for. Codex provides session/thread state via SDK calls (`Thread.read()`); the wait step is at best vacuous, at worst a hard stall. A7 redefines `WAIT_FOR_LOG` semantics in terms of the provider's `EvidenceProvider` (e.g., "wait until `iter_session_events(session_id)` returns at least one event"), not file existence at a hardcoded path.

**Files touched.**
- Edit: `src/domain/lifecycle.py:198` — `Effect.WAIT_FOR_LOG` enum definition site (no rename; add a doc-comment cross-ref to the new `EvidenceProvider` readiness contract). Emission sites at `src/domain/lifecycle.py:343` (debug log) and `:346` (`effect=Effect.WAIT_FOR_LOG` construction) are unchanged structurally.
- Edit: `src/pipeline/agent_session_runner.py:852` — replace the `_handle_wait_for_log()` implementation (currently waits for the log file to become available on disk; consumed at `src/pipeline/agent_session_runner.py:541-542`) with a call into `provider.evidence_provider.wait_for_session_ready(session_id)` (Phase A5's typed protocol). Claude impl polls the JSONL file's existence as today; Codex impl returns immediately because `Thread.read()` is always available once the thread is started; Amp impl polls the tee'd JSONL existence as today.
- Edit (if A5 renames `LogProvider` → `EvidenceProvider`): `src/pipeline/lifecycle_effect_handler.py:260` (`LifecycleEffectHandler` definition) plus its wiring at `src/pipeline/agent_session_runner.py:351,367` — type-only churn from the protocol rename.
- Edit: `src/infra/clients/claude_provider.py`, `src/infra/clients/amp_provider.py` (provider expresses readiness via `EvidenceProvider`).
- Audit (no edits unless on-disk semantics leak): `session_log_path` references at `src/pipeline/issue_result.py:34`, `src/pipeline/session_callback_factory.py:86,97,160,1134-1135,1188`, `src/orchestration/orchestrator.py:422,775,785,805,836,967,1059,1128,1136`, `src/orchestration/orchestrator_state.py:28,37`, `src/pipeline/review_runner.py:96,101,348,350,362`, `src/core/protocols/review.py:154`. Most are diagnostic-only ("where is the log on disk if I want to debug") and remain as-is; document which subset becomes optional vs required for Codex (which has no on-disk log) so reviewers can confirm the audit is exhaustive.

**Acceptance criterion (AC#A7).** `_handle_wait_for_log` (`src/pipeline/agent_session_runner.py:852`) no longer probes the filesystem directly; it calls `provider.evidence_provider.wait_for_session_ready(session_id)`. A unit test asserts a stub provider with a slow readiness signal causes the runner to block, and a fast readiness signal causes it to proceed.

**Regression contract.** Claude + Amp suites pass; Amp's existing tee'd JSONL polling still works because the new readiness check semantically equals "first event observed". Codex will not stall waiting for a nonexistent JSONL file.

#### A8 — Generalize selection types to `Literal["claude", "amp", "codex"]`

**Rationale.** `MalaConfig.coder` (`src/infra/io/config.py:364`), `ValidationConfig.coder` (`src/domain/validation/config.py:895`), CLI typer enum, `AgentProvider.name` (`src/core/protocols/agent_provider.py:54`) are all explicitly two-valued. A8 widens them to three values. No new Codex behavior — just the enum widening.

**Files touched.**
- Edit: `src/infra/io/config.py:364` (`MalaConfig.coder` Literal widens; resolver still rejects unknown values).
- Edit: `src/domain/validation/config.py:846-1271` (validation widens).
- Edit: `src/cli/cli.py` (typer enum widens).
- Edit: `src/core/protocols/agent_provider.py:54` (`name` Literal widens).
- Edit: `src/infra/clients/claude_provider.py`, `src/infra/clients/amp_provider.py` (no-op type-only changes).

**Regression contract.** Claude + Amp suites pass; existing `mala.yaml` files remain valid; selecting `coder: codex` resolves to `MalaConfig.coder == "codex"` but `_create_agent_provider` still raises (no Codex provider yet — that's Phase B). Existing `coder: claude` and `coder: amp` configs remain valid; omitted coder still resolves to the current default; unknown coder values remain rejected.

### Phase B–I Per-Phase Detail

Each phase entry: **Goals / Files-touched / Acceptance Criteria / Risks / Test list.**

#### Phase B — Codex Selection and Scaffolding

**Goals.** End-to-end selection of `coder=codex` works (CLI > env > yaml > default); `coder_options.codex.*` validates strictly; `CodexAgentProvider` stub exists and is selected by `_create_agent_provider`, raising a clear "not yet implemented" error when invoked.

**Files touched.** `src/infra/io/config.py` (`CodexOptions`, `CoderOptions.codex`, parsers, resolver), `src/domain/validation/config.py` (schema), `src/cli/cli.py` (`--codex-model`, `--codex-effort`), `src/core/constants.py` (`validate_codex_effort` parser next to `validate_amp_effort_for_mode`), `src/orchestration/factory.py:110` (Codex branch), New `src/infra/clients/codex_provider.py` (stub).

**AC.**
- AC #1: `mala run --coder codex` reaches the `CodexAgentProvider` branch (verified by smoke test that asserts the stub's "not yet implemented" error).
- AC #3: precedence is CLI > env > yaml > default for both `coder` and `coder_options.codex.*`.
- AC #4: `coder_options.codex.{model, effort, approval_policy, sandbox}` configurable; defaults `gpt-5.5`, `None`, `never`, `danger-full-access`.
- AC #13: invalid `coder_options.codex.*` values are rejected with the supported list.

**Risks.** Importing the stub must not import `codex_app_server`; widening schemas must not make existing configs invalid. `validate_codex_effort` requires the exact `ReasoningEffort` enum values, which are SDK-version-dependent (Open Question). Risk: drift between Mala's parser list and the SDK enum. Mitigation: source the enum at parser-build time (lazy-import) so the parser is data-driven.

**Test list.**
- `tests/unit/orchestration/test_factory_codex_selection.py` (CLI/env/yaml selects codex).
- `tests/unit/cli/test_codex_flags.py` (flag parsing + precedence).
- `tests/unit/infra/clients/test_codex_provider.py` (stub conformance + lazy-import guard).
- `tests/unit/domain/validation/test_codex_config.py` (strict-enum validation; invalid values rejected).
- Existing Claude/Amp regression suites pass.

#### Phase C — Codex Client Adapter

**Goals.** `CodexClient` conforms to `SDKClientProtocol` (Phase-A-cleaned). Backed by `codex_app_server.AsyncCodex` + `AsyncThread` + `AsyncTurnHandle`. `query()`, `receive_response()`, `with_resume()`, `disconnect()`, `session_id` property all functional against a mocked `AsyncCodex`.

**Files touched.** New `src/infra/clients/codex_client.py`, New `src/infra/clients/codex_runtime.py`, `src/infra/clients/codex_provider.py` (extends B3 stub; wires `client_factory` + `runtime_builder`), `src/core/protocols/sdk.py` tests.

**AC.**
- C1–C5 all pass against mocked `AsyncCodex`.
- AC #8 (resume): `with_resume(thread_id)` triggers `AsyncCodex.thread_resume(thread_id)` on next `query()`.
- Cancellation: SIGINT mid-turn calls `AsyncTurnHandle.interrupt()` and `AsyncCodex.close()` exactly once.
- Lazy-import: `import src.infra.clients.codex_client` does not transitively import `codex_app_server`.

**Risks.** Async context leaks; SDK subprocess close semantics; `ErrorNotification` without `TurnCompletedNotification`; `AsyncCodex` lifecycle subtleties (interaction with `SigintGuard`, `IdleTimeoutStream`); `thread_start` config shape for MCP not yet confirmed (Open Question — lands in Phase G). **Per-process env isolation** for the Phase E hook contract (`MALA_AGENT_ID`, `MALA_LOCK_DIR`, `MALA_REPO_NAMESPACE`, `MALA_DISALLOWED_TOOLS`) is a Phase B/C spike blocker — Mala MUST NOT mutate `os.environ` in the parent process to pass these vars (concurrent agents under `--max-agents > 1` would clobber each other's `MALA_AGENT_ID`). Parity target is the Amp path's posture: `AmpRuntimeBuilder.build()` constructs a per-subprocess `env: dict[str, str] = {**os.environ, ...overlays}` explicitly (`src/infra/clients/amp_runtime.py:315-327`). Spike outcome (see Open Questions) determines whether vars are plumbed via an explicit `env=` constructor argument on `AsyncCodex` / `AppServerConfig`, via per-thread `thread_start` config, or via a per-agent state file fallback. Mitigation: keep MCP wiring stubbed in C; finalize in G.

**Test list.**
- `tests/unit/infra/clients/test_codex_client.py` (mocked `AsyncCodex` + `AsyncTurnHandle`; lifecycle, cancellation, lazy-import guard).
- `tests/unit/infra/clients/test_codex_runtime.py` (runtime assembly, resume thread id, model/effort/approval_policy/sandbox + mcp_servers assembly).
- `tests/integration/test_codex_provider.py` (fake `AsyncAppServerClient` end-to-end smoke).

#### Phase D — Codex Item to `AgentEvent` Mapping

**Goals.** Map Codex notifications and thread items into Phase A `AgentEvent`s. All 6 Codex notification/item types map to expected `AgentEvent`s per D1–D7. Cancellation and error notifications surface as `AgentResultEvent(is_error=...)`. Reasoning is stripped.

Concrete mapping table:

| Codex Item / Notification | `AgentEvent` Emitted | Notes |
|---|---|---|
| `AgentMessageDeltaNotification` | `AgentTextEvent(text=delta_text)` | D1: streamed deltas while item in-flight. |
| `AgentMessageThreadItem` (completed) | `AgentTextEvent(text=full_text)` | D1: completed item; deduplicate against deltas if processor expects single emit. |
| `CommandExecutionThreadItem` (started) | `AgentToolUseEvent(name="bash", id=item_id, input={"command", "cwd"})` | D2: tool-use start. |
| `CommandExecutionThreadItem` (completed) | `AgentToolResultEvent(tool_use_id=item_id, is_error=<status==failed>, content=aggregated_output)` | D2: paired tool-result with command output and error status. |
| `FileChangeThreadItem` (started) | `AgentToolUseEvent(name="file_change", id=item_id, input={"changes": [...]})` | D3: file edit start. |
| `FileChangeThreadItem` (completed) | `AgentToolResultEvent(tool_use_id=item_id, is_error=<PatchApplyStatus!=success>)` | D3: paired result tied to `PatchApplyStatus`. |
| `McpToolCallThreadItem` (started) | `AgentToolUseEvent(name=f"{server}.{tool}", id=item_id, input=arguments)` | D4: MCP tool-use. |
| `McpToolCallThreadItem` (completed) | `AgentToolResultEvent(tool_use_id=item_id, is_error=<McpToolCallStatus>, content=McpToolCallResult)` | D4: paired MCP result. |
| `ReasoningTextDeltaNotification` | (none) | D5: tee'd to evidence log; no `AgentEvent` (decision #12). |
| `ReasoningSummary*Notification` | (none) | D5: tee only. |
| `ReasoningThreadItem` | (none) | D5: tee only. |
| `TurnCompletedNotification` | `AgentResultEvent(session_id=thread.id, is_error=<from TurnStatus>, subtype="completed", result=TurnStatus)` | D6: terminal result. |
| `ErrorNotification` | `AgentResultEvent(session_id=thread.id, is_error=True, subtype="error", result=error_payload)` | D7: classify auth / rate-limit / overload via existing helpers; coerced into result-event so lifecycle treats it as failure not idle hang. |

**Files touched.** New `src/infra/clients/codex_event_adapter.py` or equivalent code inside `codex_client.py`; final placement decided during implementation. New `tests/unit/infra/clients/test_codex_event_adapter.py`.

**AC.**
- D1–D7 mapping correctness as above.
- AC #15 (Phase A reuse): `MessageStreamProcessor` consumes Codex `AgentEvent`s without provider-specific branches.
- Each supported Codex notification/item produces the expected `AgentEvent`; reasoning notifications produce no runtime event; command execution evidence includes command, cwd, output, and error status.

**Risks.** Duplicate text when both delta and completed item arrive; missing IDs for paired results; SDK schema drift on item payload shape; experimental SDK. Mitigation: version pin + real-Codex e2e gate (Phase I).

**Test list.**
- `tests/unit/infra/clients/test_codex_event_adapter.py` parametrized over all 6 notification types + edge cases (delta-then-completed without started, error mid-turn).
- Lint-cache test: `LintCacheProtocol.detect_lint_command` works against the Codex-emitted `AgentToolUseEvent("bash", ...)`.
- Table-driven unit tests for `AgentMessage*`, `CommandExecutionThreadItem`, `FileChangeThreadItem`, `McpToolCallThreadItem`, `TurnCompletedNotification`, `ErrorNotification`, and reasoning notifications.

#### Phase E — Codex Safety Model (bundled plugin: hook + locking MCP)

**Goals.** `mala-codex-pre-tool-use` hook script enforces lock ownership + dangerous-cmd + `MALA_DISALLOWED_TOOLS`. Bundled Codex plugin (`plugins/codex/mala-safety/.codex-plugin/`) installs idempotently to `~/.codex/plugins/mala-safety/.codex-plugin/` (exact target path TBD by Phase E spike). `install_prerequisites()` writes `trusted_hash` + runs selftest. Fail-closed on any safety degradation.

**Hook stdin schema (input)** — what Codex sends to `mala-codex-pre-tool-use` on stdin (subset of `pre-tool-use.command.input.schema.json`):

```json
{
  "cwd": "/abs/path/to/repo",
  "tool_name": "bash" | "apply_patch" | "<server>.<tool>",
  "tool_input": { ... },
  "session_id": "thr_...",
  "turn_id": "trn_..."
}
```

**Hook stdout schema (output)** — what `mala-codex-pre-tool-use` writes to stdout:

```json
{
  "decision": "approve" | "block",
  "hookSpecificOutput": {
    "permissionDecision": "allow" | "deny" | "ask",
    "permissionDecisionReason": "<short string>"
  }
}
```

**Hook env contract**:

| Env var | Source | Used for |
|---|---|---|
| `MALA_AGENT_ID` | injected by Mala when starting Codex turn | identifies which agent owns locks |
| `MALA_LOCK_DIR` | injected by Mala | filesystem location of lock state |
| `MALA_REPO_NAMESPACE` | injected by Mala | namespace for cross-repo lock isolation |
| `MALA_DISALLOWED_TOOLS` | optional, comma-separated | tool names the hook should always deny |

**Concurrency invariant (per-process env isolation).** Mala MUST pass the `MALA_*` vars to the Codex subprocess via an explicit per-process env dict, NOT by mutating `os.environ` in the parent — parity with the Amp path (`AmpRuntimeBuilder.build()` constructs `env: dict[str, str] = {**os.environ, ...overlays}` explicitly, `src/infra/clients/amp_runtime.py:315-327`). Otherwise, when `--max-agents > 1` is in effect, concurrent Codex agents would clobber each other's `MALA_AGENT_ID` in the shared parent `os.environ` and the inherited subprocess env would identify the wrong owner to the bundled `PreToolUse` hook. Whether the SDK actually supports per-process env injection is a Phase B/C spike blocker (see Open Questions).

**Fallback transport: per-session state file** (used only if the Phase B/C spike confirms the Codex SDK does not support explicit per-process env injection). The hook input always carries `session_id` and (for in-turn calls) `turn_id` per `pre-tool-use.command.input.schema.json`. The fallback transport is therefore keyed by `session_id` — the discriminator the hook receives natively — not by `agent_id`, which the hook never sees:

  1. **Write phase.** Immediately after `AsyncCodex.thread_start(...)` returns a `Thread`, before the first turn is started, `CodexClient` writes `~/.config/mala/agent-state/{session_id}.env` containing the `MALA_*` key-value pairs for the current agent (one `KEY=VALUE` per line, sorted, fsync'd before the first turn fires). The file is written via the same write-temp-then-rename pattern Amp uses for `~/.config/amp/plugins/mala-safety.ts` (`src/infra/clients/amp_plugin_installer.py`), so concurrent agents creating their own state files cannot tear each other.
  2. **Read phase.** The hook reads `os.environ` first; if `MALA_AGENT_ID` is set there, the hook uses env values directly (the env-injection path). Otherwise, the hook falls back to reading `~/.config/mala/agent-state/{session_id}.env` (where `session_id` is taken from the hook stdin JSON payload). Missing file or missing required keys → the hook denies with `permissionDecisionReason: "lock-ownership state missing for session {session_id}"` (parity with Amp's plugin fail-closed mode when the env vars are unset, plan `L633` of `2026-04-29-amp-provider-plan.md`).
  3. **No `agent_id`-keyed index needed.** Because `session_id` is one-to-one with the spawning agent (and `CodexClient.with_resume(thread_id)` reuses the original `session_id`, so resumed turns hit the same state file), there is no separate `{session_id → agent_id}` index — the file *is* the index.
  4. **Cleanup.** `CodexClient.disconnect()` unlinks `{session_id}.env` after the turn completes (best-effort; missing-file errors are tolerated). A periodic cleanup of files older than a documented TTL (e.g., 7 days) is tracked as a follow-up so a crashed agent doesn't leak state files indefinitely.
  5. **Nested-agent caveat.** If a future Codex feature spawns sub-agents that emit `PreToolUse` hooks under a different `session_id` than the parent (e.g., guardian sub-agents), the parent's state file is not directly consulted — each sub-agent must have its own state file written by whichever `CodexClient` (or upstream wrapper) spawned it. Documented as an Open Question; not an MVP concern because mala does not currently nest Codex agents.

The hook implementation MUST accept either source (env or state file), with env taking precedence when both are present, so that the same hook binary works whether or not the spike confirms env injection. Tests cover both paths (env-only, state-file-only, env-overrides-state-file, state-file-missing → deny).

**Tool-name → path-key mapping table** (used by hook for lock-ownership lookup):

| Codex tool name | Hook branch | Lock path source |
|---|---|---|
| `bash` (or local-shell, [TBD: confirm exact name in Phase D spike]) | dangerous-cmd path **+ shell-command write-path enforcement** | `dangerous_commands.py` for command-pattern denial; **plus** parsed/heuristic write-path extraction → `lock_path` for each detected write target (see "Shell-command write-path enforcement" below) |
| `apply_patch` ([TBD: confirm exact name(s) in Phase D spike]) | file-edit path | derive from `tool_input.path` (or equivalent) via `lock_path` |
| `<server>.<tool>` (MCP) | tool-name allowlist | check `MALA_DISALLOWED_TOOLS`; allow otherwise |

**Shell-command write-path enforcement.** Because `sandbox=danger-full-access` + `approval_policy=never` (decisions #2, #3) make the hook the only safety gate, the hook's bash/local-shell branch must also gate writes performed by ordinary shell commands — not just by `apply_patch`. Without this, a turn could legally run `printf X > file`, `sed -i ...`, `python -c "open('f','w').write(...)"`, `cp src dst`, `mv a b`, `git checkout -- file`, etc. and silently mutate unlocked or wrong-owner paths, bypassing AC #9's lock-enforcement guarantee.

The hook's bash branch performs both checks before allowing the command:

1. **Source the parsed actions if available.** Codex provides best-effort parsed actions for shell commands (`CommandExecutionThreadItem.command_actions`, schema in `pre-tool-use.command.input.schema.json`). [TBD: confirm during Phase E spike whether `command_actions` is delivered in the hook's stdin payload or only on the surfaced Item; if it is *not* in the hook input, the hook must parse `tool_input.command` (the raw command string / argv) itself.] Treat parsed actions as advisory — the hook always also runs the heuristic below, so a missing or sparse `command_actions` field never silently weakens enforcement.
2. **Heuristic write-path detection.** Apply the table below to the command string. For every detected write path, run `src/infra/tools/locking.lock_path` (the same call the `apply_patch` branch uses) and deny if the path has no lock or is owned by a different agent.

| # | Pattern | Example | Lock-key extraction strategy |
|---|---------|---------|------------------------------|
| 1 | Shell redirection | `cmd > file`, `cmd >> file`, `cmd &> file`, `cmd 2> file`, `cmd \| tee file`, `cmd \|& tee file` | RHS of `>`, `>>`, `&>`, `2>`, etc. after redirection-operator tokenization; arg(s) to `tee` / `tee -a` |
| 2 | In-place edits | `sed -i ... file`, `sed -i'' ... file`, `perl -i ... file`, `perl -pi -e ... file`, `awk -i inplace ... file`, `gawk -i inplace ... file` | Trailing positional file arg(s) after `-i`/inplace flag |
| 3 | Language one-liners that open files for writing | `python -c '...open("f","w")...'`, `python3 -c ...`, `node -e ...`, `ruby -e ...`, `perl -e ...`, `bash -c '... > f'`, `sh -c '... > f'` | Recursively re-apply heuristic to the embedded script body; deny if any write-path expression is parseable but cannot be resolved (conservative). Best-effort string-level scan for `open(<lit>, ('w'\|'a'\|'x'\|...))`, `Path(<lit>).write_text/write_bytes`, `fs.writeFile(<lit>,...)`, `File.write(<lit>,...)`. |
| 4 | File-creation / mutation utilities | `touch f`, `cp src dst`, `mv a b`, `rm f`, `mkdir d`, `rmdir d`, `ln -s a b`, `chmod 755 f`, `chown user f`, `install -m 0644 src dst`, `dd of=f` | Destination arg(s) (last positional for `cp`/`mv`/`install`/`ln`; all positionals for `rm`/`mkdir`/`rmdir`/`touch`/`chmod`/`chown`; `of=` arg for `dd`) |
| 5 | Commit/checkout operations that write tracked files | `git checkout -- f`, `git checkout HEAD f`, `git restore f`, `git apply patch`, `git stash apply`, `git stash pop`, `patch < file`, `patch -p1 < file` | All paths the operation will write: explicit positional path args for `git checkout/restore`; for `git apply`/`patch`, parse the diff for `+++ b/<path>` / `*** <path>` headers (best-effort); for `git stash apply/pop`, conservatively require a fresh whole-repo lock or deny if any tracked file would be modified |

3. **Deny semantics.** If any extracted write path is missing a lock for `MALA_AGENT_ID` (or is locked by another agent), emit `permissionDecision: "deny"` with `permissionDecisionReason` reusing the existing strings (`path '<rel>' has no active lock for agent '<me>'` / `path '<rel>' is locked by agent '<other>'; current agent is '<me>'`). Heuristic-detected paths use the same canonicalization as `lock_path` so messages are byte-identical to the `apply_patch` branch.
4. **Residual-risk acknowledgment.** This is a heuristic, not a sandbox: a sufficiently obfuscated command (e.g., `python -c "$(echo b3BlbignZicsJ3cnKQ== \| base64 -d)"`, indirection through environment variables, dynamically-built filenames, fully-quoted compound commands invoked via `eval`, or a child process that fork-execs a writer) can bypass it. The plan accepts this as defense-in-depth; tighter enforcement (e.g., ptrace/eBPF-based write-syscall monitoring, or a constrained sandbox per Approach B in the plan-review issue) is recorded as **follow-up work** and out of MVP scope.
5. **Parity note vs Amp.** Amp's `mala-safety.ts` plugin has the **same fundamental gap** — it gates `edit_file`/`create_file`-style structured tool calls but cannot fully enforce locks against arbitrary shell-command writes for the same heuristic-coverage reason. [TBD: confirm during Phase E spike by re-reading `mala-safety.ts` whether it currently implements any shell-command write-path heuristic; document parity (Codex matches Amp's heuristic table) or divergence (Codex strictly broader/narrower) in Phase E5 selftest docs.] This documentation should live next to the same maintainer-facing notes the Amp plugin uses, so the two providers' residual-risk story is read in one place.

**Deny-message strings** (shown via `permissionDecisionReason`):

- Lock owned by another agent: `"path '<rel>' is locked by agent '<other>'; current agent is '<me>'"`
- No lock found: `"path '<rel>' has no active lock for agent '<me>'"`
- Dangerous command: `"command matches dangerous pattern: <pattern>"`
- Disallowed tool: `"tool '<name>' is in MALA_DISALLOWED_TOOLS"`
- Lock env missing: `"MALA_AGENT_ID/MALA_LOCK_DIR/MALA_REPO_NAMESPACE missing; refusing to evaluate lock"`

**Plugin install layout** (sketches; exact spec resolved during E spike):

`plugins/codex/mala-safety/.codex-plugin/plugin.json`:
```json
{
  "name": "mala-safety",
  "version": "<auto-derived from mala version>",
  "description": "Mala safety hook + locking MCP for unattended Codex runs",
  "hooks": "hooks.json",
  "mcpServers": ".mcp.json"
}
```

`plugins/codex/mala-safety/.codex-plugin/hooks.json`:
```json
{
  "hooks": [
    {
      "event": "PreToolUse",
      "type": "command",
      "command": "mala-codex-pre-tool-use",
      "trusted_hash": "<sha256 of mala-codex-pre-tool-use entry-point + plugin.json version>"
    }
  ]
}
```

`plugins/codex/mala-safety/.codex-plugin/.mcp.json`:
```json
{
  "mcpServers": {
    "mala-locking": {
      "command": "mala-codex-mcp-locking",
      "args": [],
      "env": {}
    }
  }
}
```

[TBD: confirm exact `plugin.json` field names and structure during E spike against `codex-rs/skills/src/assets/samples/plugin-creator/references/plugin-json-spec.md`.]

**Files touched.** New `plugins/codex/mala-safety/.codex-plugin/plugin.json`, New `plugins/codex/mala-safety/.codex-plugin/hooks.json`, New `plugins/codex/mala-safety/.codex-plugin/.mcp.json`, New `src/infra/hooks/codex_pre_tool_use.py` (entry-point), New `src/infra/clients/codex_plugin_installer.py` (idempotent install + trusted-hash write), `src/infra/clients/codex_provider.py` (extends with `install_prerequisites()` + selftest), `pyproject.toml` (register `mala-codex-pre-tool-use` console script), tests.

**AC.**
- E1: default sandbox = `danger-full-access`.
- E2: default approval_policy = `never`.
- E3: hook denies on (a) lock owned by different agent (file-edit branch), (b) dangerous command (mirrors `dangerous_commands.py`), (c) tool in `MALA_DISALLOWED_TOOLS`, (d) **shell command writing to a path that is not locked by the current agent or is owned by another agent (heuristic table in "Shell-command write-path enforcement" sub-section).** Hook allows otherwise. Lock canonicalization parity with `lock_path` for both the `apply_patch` branch and the bash write-path branch. Valid locked edits / valid locked shell writes are allowed.
- E4: plugin installs idempotently; reinstall doesn't duplicate state.
- E5: selftest fails closed on `HOOK_MARKER_MISSING | VERSION_MISMATCH | SCRIPT_MISSING | PLUGIN_DISABLED | TRUSTED_HASH_MISMATCH | CODEX_BINARY_MISSING`.
- E6: `install_prerequisites()` writes `trusted_hash` to Codex hook-state file; if Codex requires interactive trust (Open Question), fall back to documented one-time prompt.
- AC #9 (safety): sandbox + approvals + lock-enforce hold for Codex; unowned file edits are denied; dangerous shell commands are denied; disallowed tools are denied. **Lock-enforce parity covers both `apply_patch` and bash shell-command writes (AC #19).**
- AC #19 (shell-command write-path enforcement): every heuristic-table pattern (redirection, `sed -i`, `perl -i`, `awk -i inplace`, `python -c`/`node -e`/`ruby -e`/`bash -c` write-open, `touch`/`cp`/`mv`/`rm`/`mkdir`/`ln`/`chmod`/`chown`/`install`/`dd`, `git checkout`/`restore`/`apply`/`stash apply/pop`, `patch`) is denied when the target is unlocked or owned by another agent. Residual obfuscation gap (e.g., base64-encoded `python -c`) is acknowledged as defense-in-depth limitation; tighter enforcement is follow-up work.
- AC #14 (fail-closed on missing deps): missing SDK/runtime/auth/plugin → `CodexNotInstalledError` or `CodexHookNotActiveError`.

**Risks.** Plugin discovery path differs from expected; hook trust state format changes; Codex tool names for shell/file-edit tools differ from assumptions. **Codex plugin discovery path** (Open Question — `~/.codex/plugins/`?). **Codex hook-state file location and format** for trusted-hash auto-trust (Open Question — risk that interactive trust is mandatory). Mitigations: spike before E4 finalizes; documented one-time-step fallback; E5 selftest catches both failure modes uniformly.

**Test list.**
- `tests/unit/infra/hooks/test_codex_pre_tool_use.py` (allow / deny / dangerous-cmd / disallowed-tool / lock-mismatch / env-missing).
- `tests/unit/infra/clients/test_codex_plugin_installer.py` (idempotent install, marker/hash write, uninstall).
- `tests/integration/test_codex_lock_enforcement.py` (parity matrix with `tests/integration/test_amp_lock_enforcement.py`).
- `tests/integration/test_codex_shell_lock_enforcement.py` (shell-command write-path enforcement: every row of the heuristic table — redirection `>`/`>>`/`tee`, `sed -i`, `perl -i`, `awk -i inplace`, `python -c`/`node -e`/`ruby -e`/`bash -c` write-open patterns, `touch`/`cp`/`mv`/`rm`/`mkdir`/`ln`/`chmod`/`chown`/`install`/`dd`, `git checkout`/`restore`/`apply`/`stash apply`/`pop`, `patch`. Each pattern is asserted to be denied when the target path is unlocked or owned by a different agent, and allowed when locked by the current agent. Includes the residual-risk regression case: a deliberately-obfuscated `python -c "$(...)"` write that the heuristic does not detect — assert it is documented behavior, not silently denied or silently allowed by mistake).
- `tests/unit/infra/clients/test_codex_provider.py` (extends): selftest pass/fail modes.

#### Phase F — Codex Evidence Provider

**Goals.** `CodexEvidenceProvider` returns `ParsedEvent` stream from `Thread.read(include_turns=True)`. Cross-resume invariant: full thread history regardless of resume count. F1 spike validates this before F2/F3 lock in (decision #11).

**Files touched.** New `src/infra/clients/codex_evidence_provider.py`, `src/infra/clients/codex_provider.py` (wires `evidence_provider`), evidence-provider tests.

**AC.**
- F1 spike: `Thread.read(include_turns=True)` returns command output (`CommandExecutionThreadItem.aggregated_output`); read cost bounded; works after `thread_resume`. **If F1 disconfirms any of these, switch to F3 tee fallback** (`~/.config/mala/codex-sessions/{thread_id}.jsonl`, append-mode, per-thread keying).
- AC #7: validation evidence available for Codex runs (extract Bash commands, tool results, assistant text blocks).
- AC #18 (Phase A reuse): `EvidenceProvider` is the unified evidence surface — `CodexEvidenceProvider` conforms to the same protocol Claude and Amp do.
- Evidence includes history across resume.

**Risks.** F1 spike disconfirms — costs a phase to add tee fallback. `Thread.read` omits aggregated command output, is paginated, or is too expensive for repeated gate reads. Mitigation: F1 lands first; tee implementation is well-understood (Amp parity).

**Test list.**
- `tests/unit/infra/clients/test_codex_evidence_provider.py` (Thread.read fixture + extraction).
- `tests/integration/test_codex_evidence_cross_resume.py` (resume preserves evidence stream).
- Validation extractor tests; contingency tee test only if fallback is implemented.

#### Phase G — Codex MCP Integration

**Goals.** `mala-locking` MCP launcher bundled inside the Codex plugin. User-supplied `coder_options.codex.mcp_servers` merges with bundled, never replaces.

**Files touched.** `plugins/codex/mala-safety/.codex-plugin/.mcp.json` (extends E4 with `mala-locking` launch spec referencing `mala-codex-mcp-locking`), `src/orchestration/orchestration_wiring.py` (Codex MCP factory), `src/infra/clients/codex_runtime.py`, `src/infra/tools/locking_mcp_stdio.py` (reused unchanged; shared by both Amp and Codex entry points), `pyproject.toml` (add new `mala-codex-mcp-locking` console script alongside the existing `mala-amp-mcp-locking`).

**AC.**
- G1: `mala-locking` launchable via Codex's stdio MCP shape.
- G2: Codex round-trips `lock_acquire` / `lock_release` calls through the same locking server Amp uses.
- G3: User-specified MCP servers merge with bundled; bundled is never overridden.
- AC #17 (Phase A reuse): MCP factory dispatch is provider-owned (no `if name == "amp"` branches).
- Orchestration does not branch on provider name for MCP dispatch.

**Risks.** **Codex MCP wire shape** (Open Question — `Codex.thread_start(config={...})` vs config-file vs runtime parameter, or via plugin `.mcp.json` only). Plugin `.mcp.json` loading may not apply to app-server turns. Mitigation: Phase G spike before implementation; falls back to in-config-file approach if inline launch spec unsupported.

**Test list.**
- `tests/integration/test_codex_mcp_round_trip.py` (lock_acquire / lock_release via Codex).
- `tests/unit/orchestration/test_codex_mcp_merge.py` (user MCP servers merge with bundled).
- MCP config assembly tests; fake Codex runtime test verifying merged MCP specs.

#### Phase H — Resume, Fixer, and Idle Retry

**Goals.** `AsyncCodex.thread_resume(thread_id)` integration; fixer agents follow main coder; idle retry works unchanged.

**Files touched.** `src/infra/clients/codex_client.py`, `src/infra/clients/codex_runtime.py` (extends C with `with_resume`), `src/pipeline/fixer_service.py` if Phase A did not already remove coder-specific assumptions; otherwise no code change, test added (H2).

**AC.**
- H1: resume preserves prior thread context (verified by integration test that issues 2 turns across a resume boundary and asserts continuity).
- H2: when `coder=codex`, `FixerService` spawns Codex (AC #5 parity with Amp).
- H3: `IdleTimeoutRetryPolicy` triggers `IdleTimeoutError` → `thread_resume` retry (no Codex-specific code; `IdleTimeoutStream` wraps Codex notification iterator).
- Fixer runs instantiate Codex provider when main coder is Codex; idle timeout retries resume the same Codex thread.

**Risks.** Interrupted turns may not be resumable immediately; retry classification may need Codex-specific error mapping. Idle-timeout under SDK-managed subprocess: if `AsyncTurnHandle.stream()` blocks past idle threshold without surfacing an error, idle stream raises and retry kicks in. Test parametrizes fast/slow streams.

**Test list.**
- `tests/integration/test_fixer_follows_codex.py` (H2).
- `tests/integration/test_codex_resume.py` (H1: 2 turns across resume).
- `tests/unit/pipeline/test_idle_retry_codex.py` (H3 idle-timeout simulated).
- Fake Codex resume integration; cancellation-plus-resume test.

#### Phase I — Prerequisites, Telemetry, Tests, Docs, Ship

**Goals.** `install_prerequisites()` complete; `coder` telemetry attribute extended; real-Codex e2e gate; docs updated; existing Claude/Amp suites green.

**Files touched.** `src/infra/clients/codex_provider.py` (final `install_prerequisites()`), `src/infra/io/base_sink.py` (telemetry attribute), `pyproject.toml` (`mala[codex]` extras + version pin), `README.md`, `docs/cli-reference.md`, `docs/project-config.md`, `docs/architecture.md`, `docs/development.md`, e2e tests.

**`pyproject.toml` extras sketch**:

```toml
[project.optional-dependencies]
codex = [
    "openai-codex-app-server-sdk>=[TBD: lower],<[TBD: upper]",
    # openai-codex-cli-bin is a separate platform-specific runtime install; not pip-managed
]

[project.scripts]
mala-codex-pre-tool-use = "src.infra.hooks.codex_pre_tool_use:main"
mala-amp-mcp-locking = "src.infra.tools.locking_mcp_stdio:main"  # existing; left unchanged
mala-codex-mcp-locking = "src.infra.tools.locking_mcp_stdio:main"  # new sibling for Codex; same module
```

[TBD: pin exact SDK version range once SDK release notes are reviewed.]

**AC.**
- I1 (AC #14): missing SDK / runtime / auth / plugin → `CodexNotInstalledError` or `CodexHookNotActiveError` with actionable message.
- I2: telemetry `coder` attribute domain includes `"codex"`.
- I3 (AC #12): `tests/e2e/test_codex_real_sdk.py` runs against real `AsyncCodex` (gated on import + binary + auth env-var).
- I4 (AC #11): docs updated.
- I5 (AC #10): existing Claude/Amp suites pass unchanged.

**Risks.** Real-Codex e2e flakiness (network, model availability). Platform-specific runtime packaging; auth setup differences; docs drifting from actual option names. Mitigation: tight prompt + retry-on-rate-limit + skip-if-auth-missing.

**Test list.**
- `tests/e2e/test_codex_real_sdk.py` (one-line prompt; gated on SDK + runtime + auth).
- All previously-listed unit/integration tests pass.
- Existing Claude + Amp regression suites pass.

### File Impact Summary

#### Phase A — refactoring (no behavior change)

| Path | Status | Description |
|------|--------|-------------|
| `src/core/protocols/agent_event.py` | **New** | `AgentEvent` + concrete event dataclasses (A1). |
| `src/core/protocols/sdk.py` | Exists | Slim `SDKClientFactoryProtocol`; remove Claude-only methods (A2). |
| `src/core/protocols/agent_provider.py` | Exists | Add `mcp_server_factory()`; rename `log_provider` to `evidence_provider`; widen `name` Literal (A4, A5, A8). |
| `src/core/protocols/log.py` | Exists → **Delete** | Replaced by `src/core/protocols/evidence.py` (A5); no re-export shim. |
| `src/core/protocols/evidence.py` | **New** | `EvidenceProvider` typed protocol (A5). |
| `src/pipeline/message_stream_processor.py` | Exists | Branch on `AgentEvent.kind`, not class name (A1). |
| `src/pipeline/agent_session_runner.py` | Exists | Stop reading `runtime.options` directly; pass runtime to provider's client factory (A3). Lifecycle effect alignment (A7). |
| `src/pipeline/fixer_service.py` | Exists | Same `runtime.options` removal (A3); coder-agnostic post-Phase A (also tested in H2). |
| `src/infra/agent_runtime.py` | Exists | Move Claude-only fluent methods onto `ClaudeAgentRuntimeBuilder` subclass; cross-coder builder protocol shrinks (A6). |
| `src/infra/sdk_adapter.py` | Exists | `SDKClientFactory` becomes Claude-private; `ClaudeSDKClient.receive_response()` emits `AgentEvent`s (A1, A2). |
| `src/infra/clients/claude_provider.py` | Exists | Wire `mcp_server_factory()`; rename `log_provider` → `evidence_provider`; private Claude factory wiring (A1, A2, A4, A5). |
| `src/infra/clients/amp_provider.py` | Exists | Same wiring as Claude; `_AmpClientFactory.NotImplementedError` walls deleted (A2, A3, A4, A5). |
| `src/infra/clients/amp_client.py` | Exists | `receive_response()` emits `AgentEvent`s directly (A1). |
| `src/infra/clients/amp_runtime.py` | Exists | Drop pre-built `options` field once A3 lands; drop no-op fluent calls covered by A6. |
| `src/infra/clients/amp_messages.py` | Exists → **Delete** | Synthesized Anthropic shape no longer needed (A1). |
| `src/infra/clients/amp_log_provider.py` | Exists | Conform to new `EvidenceProvider` (A5); behavior unchanged. |
| `src/infra/io/session_log_parser.py` | Exists | `FileSystemLogProvider` conforms to new `EvidenceProvider` (A5); behavior unchanged. |
| `src/orchestration/factory.py` | Exists | Delete `if agent_provider.name == "amp"` MCP branch at line 1210 (A4); provider owns dispatch. Widen coder selection at line 110 (A8). Codex provider selection later wired in B. |
| `src/orchestration/orchestrator.py` | Exists | Delete `name == "amp"` branch at line 404 (A4). |
| `src/orchestration/orchestration_wiring.py` | Exists | Provider-owned MCP factory wiring (A4); add Codex MCP factory in G. |
| `src/domain/validation/config.py` | Exists | Widen `coder` Literal (A8); add Codex schema and strict option validation in B. |
| `src/cli/cli.py` | Exists | Widen `--coder` typer enum (A8); add `--codex-model` / `--codex-effort` flags in B. |
| `src/domain/lifecycle.py` | Exists | `Effect.WAIT_FOR_LOG` definition site at line 198 — doc-comment cross-ref to `EvidenceProvider` readiness contract; emission sites at lines 343, 346 unchanged structurally (A7). |
| `src/pipeline/agent_session_runner.py` | Exists | `_handle_wait_for_log()` at line 852 rewritten to call `provider.evidence_provider.wait_for_session_ready(session_id)`; consumer at lines 541–542 unchanged (A7). |
| `src/pipeline/lifecycle_effect_handler.py` | Exists | Type-only churn from A5's `LogProvider` → `EvidenceProvider` rename (`LifecycleEffectHandler` at line 260; wired at `agent_session_runner.py:351,367`). |

#### Phase B — selection wiring + provider scaffold

| Path | Status | Description |
|------|--------|-------------|
| `src/infra/io/config.py` | Exists | Widen `coder` Literal; add `CodexOptions` + `CoderOptions.codex`; add `parse_codex_*`; widen `from_env` resolver (B1, B2). |
| `src/core/constants.py` | Exists | Add `validate_codex_effort` parser next to `validate_amp_effort_for_mode` (B2, decision #13). |
| `src/infra/clients/codex_provider.py` | **New** | `CodexAgentProvider` stub conforming to Phase-A-cleaned `AgentProvider` (B3); extended in C, E, I. |

#### Phase C — CodexClient + runtime

| Path | Status | Description |
|------|--------|-------------|
| `src/infra/clients/codex_client.py` | **New** | `CodexClient` adapter (C1–C5), extended in H1 with resume integration. |
| `src/infra/clients/codex_runtime.py` | **New** | `CodexRuntime` + `CodexRuntimeBuilder` (C1); extended in H1 with `with_resume(thread_id)`. |

#### Phase D — item → AgentEvent mapping

| Path | Status | Description |
|------|--------|-------------|
| `src/infra/clients/codex_event_adapter.py` | **New** | Notification → `AgentEvent` translation (D1–D7). May live inside `codex_client.py`; final placement decided during implementation. |

#### Phase E — safety model

| Path | Status | Description |
|------|--------|-------------|
| `plugins/codex/mala-safety/.codex-plugin/plugin.json` | **New** | Plugin manifest (E4, E5). |
| `plugins/codex/mala-safety/.codex-plugin/hooks.json` | **New** | `PreToolUse` command hook config pointing at `mala-codex-pre-tool-use` (E3, E4). |
| `plugins/codex/mala-safety/.codex-plugin/.mcp.json` | **New** | `mala-locking` MCP server launch spec (E4, G1). |
| `src/infra/hooks/codex_pre_tool_use.py` | **New** | `mala-codex-pre-tool-use` script entry point: reads JSON stdin, calls `src/infra/tools/locking.lock_path`, returns `permissionDecision` JSON (E3). |
| `src/infra/clients/codex_plugin_installer.py` | **New** | Idempotent straight-copy install of the bundled Codex plugin (`plugins/codex/mala-safety/.codex-plugin/` source tree) to Codex's user-plugin directory (target path TBD by Phase E spike — assumed `~/.codex/plugins/mala-safety/.codex-plugin/`); writes `trusted_hash` to Codex hook-state file (E4, E6). |
| `pyproject.toml` | Exists | Register `mala-codex-pre-tool-use` console script (E3); add `mala[codex]` extras + version pin in I1; add new `mala-codex-mcp-locking` console script (sibling of existing `mala-amp-mcp-locking`, same module) in G2. |

#### Phase F — evidence provider

| Path | Status | Description |
|------|--------|-------------|
| `src/infra/clients/codex_evidence_provider.py` | **New** | Codex `EvidenceProvider` impl: native `Thread.read(include_turns=True)` (F2); tee fallback at `~/.config/mala/codex-sessions/{thread_id}.jsonl` (F3 if F1 disconfirms). |

#### Phase G — MCP integration

| Path | Status | Description |
|------|--------|-------------|
| `src/infra/tools/locking_mcp_stdio.py` | Exists | Reuse for Codex (Amp shape compatible); shared module backing both `mala-amp-mcp-locking` (existing) and the new `mala-codex-mcp-locking` console-script entry points. |
| `src/orchestration/orchestration_wiring.py` | Exists | Add `create_codex_mcp_server_factory()`; wired through `provider.mcp_server_factory()` (A4). |

#### Phase H — resume / fixer / idle

(`codex_client.py` and `codex_runtime.py` extended; `fixer_service.py` already coder-agnostic after Phase A.)

#### Phase I — prerequisites + telemetry + docs

| Path | Status | Description |
|------|--------|-------------|
| `src/infra/io/base_sink.py` | Exists | `coder` attribute domain widens to `"codex"` (I2). |
| `pyproject.toml` | Exists | Add `openai-codex-app-server-sdk` as optional extra (`mala[codex]`); pin version (I1). |
| `README.md` | Exists | Codex Prerequisites + Usage subsections (I4). |
| `docs/cli-reference.md` | Exists | `--coder codex`, `--codex-*` flags, env vars (I4). |
| `docs/project-config.md` | Exists | `coder: codex` and `coder_options.codex.*` (I4). |
| `docs/architecture.md` | Exists | Phase A diagrams + Codex provider triple (I4). |
| `docs/development.md` | Exists | Provider-development + Codex-specific notes (I4). |

#### Tests

| Path | Status | Description |
|------|--------|-------------|
| `tests/unit/core/protocols/test_agent_event.py` | **New** | A1 protocol shape + adapter conformance. |
| `tests/unit/pipeline/test_message_stream_processor_kinds.py` | **New** | A1 processor accepts `AgentEvent`s without class-name branches. |
| `tests/unit/core/protocols/test_evidence_provider.py` | **New** | A5 `EvidenceProvider` conformance shared across providers. |
| `tests/unit/core/protocols/test_sdk_factory_slim.py` | **New** | A2 protocol-shape test (no Claude-only methods on cross-coder protocol). |
| `tests/unit/orchestration/test_factory_no_name_branch.py` | **New** | A4 asserts `factory.py`/`orchestrator.py` no longer branch on `provider.name`. |
| `tests/unit/orchestration/test_factory_codex_selection.py` | **New** | B1: CLI/env/yaml selects Codex. |
| `tests/unit/cli/test_codex_flags.py` | **New** | B2: `--coder codex` + `--codex-*` parsing. |
| `tests/unit/domain/validation/test_codex_config.py` | **New** | B2: strict-enum validation; invalid values rejected. |
| `tests/unit/infra/clients/test_codex_provider.py` | **New** | B3 + E5 + I1: provider conformance + lazy-import guard + selftest pass/fail modes. |
| `tests/unit/infra/clients/test_codex_client.py` | **New** | C: mocked `AsyncCodex`; lifecycle, parse, resume. |
| `tests/unit/infra/clients/test_codex_runtime.py` | **New** | C: model/effort/approval_policy/sandbox + mcp_servers assembly. |
| `tests/unit/infra/clients/test_codex_event_adapter.py` | **New** | D: All 6 notification/item types map to expected `AgentEvent`s (D1–D7). |
| `tests/unit/infra/hooks/test_codex_pre_tool_use.py` | **New** | E3: hook script unit tests: allow / deny / dangerous-cmd / disallowed-tool / lock-mismatch / env-missing. |
| `tests/unit/infra/clients/test_codex_plugin_installer.py` | **New** | E4: idempotent install; trusted-hash write; selftest pass/fail modes. |
| `tests/unit/infra/clients/test_codex_evidence_provider.py` | **New** | F: `Thread.read()` evidence parsing + tee fallback. |
| `tests/unit/orchestration/test_codex_mcp_merge.py` | **New** | G3: user MCP servers merge with bundled. |
| `tests/unit/pipeline/test_idle_retry_codex.py` | **New** | H3: idle-timeout simulated. |
| `tests/integration/test_codex_provider.py` | **New** | C: Fake `AsyncAppServerClient` end-to-end. |
| `tests/integration/test_codex_lock_enforcement.py` | **New** | E: parity matrix with `tests/integration/test_amp_lock_enforcement.py`: allow / no-lock / wrong-agent / wrong-namespace / env-missing / canonicalization-parity / disallowed-tool. |
| `tests/integration/test_codex_shell_lock_enforcement.py` | **New** | E (AC #19): shell-command write-path enforcement — every heuristic-table row (redirection, `sed -i`, `python -c` write-open, `cp`/`mv`/`rm`/`touch`/`ln`/`chmod`/`chown`/`install`/`dd`, `git checkout/restore/apply/stash`, `patch`) is denied when the target is unlocked or wrong-owner. |
| `tests/integration/test_codex_evidence_cross_resume.py` | **New** | F: resume preserves evidence stream. |
| `tests/integration/test_codex_mcp_round_trip.py` | **New** | G: lock_acquire / lock_release via Codex. |
| `tests/integration/test_fixer_follows_codex.py` | **New** | H2: Fixer follows main coder when codex. |
| `tests/integration/test_codex_resume.py` | **New** | H1: 2 turns across resume. |
| `tests/e2e/test_codex_real_sdk.py` | **New** | I3: Real Codex; gated on SDK + runtime + auth. |
| Existing Claude tests | Exists | Run unchanged after Phase A; behavioral diff is a refactor bug. |
| Existing Amp tests | Exists | Same. |
| `tests/integration/test_amp_lock_enforcement.py` | Exists | Continues to pass after E (no Amp behavior change). |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes

- **SDK is experimental**: pin `openai-codex-app-server-sdk` and `openai-codex-cli-bin` to known-good versions; document tested range; add real-Codex e2e gate.
- **Auth state ambient**: Codex auth lives in user's local config. `install_prerequisites()` surfaces missing/expired auth as an actionable error before any issue agent runs.
- **Codex SDK/runtime missing**: `install_prerequisites()` raises `CodexNotInstalledError`; no fallback provider is used.
- **Codex auth missing or expired**: fail closed before issue execution.
- **Hook plugin missing, disabled, stale, or untrusted**: self-test fails and Codex run aborts.
- **Hook receives unknown tool names**: deny if the action could write files or execute commands; otherwise log and follow explicit safe mapping.
- **Lock env missing**: deny write/tool action and report actionable reason.
- **Wrong lock owner or namespace**: deny with existing Mala lock-denial semantics.
- **Dangerous command detected**: deny through the Codex hook before sandbox/approval evaluation.
- **`MALA_DISALLOWED_TOOLS` contains the requested Codex tool**: deny through the same hook.
- **`thread_start` config shape for MCP not confirmed**: Phase G spike confirms whether `Codex.thread_start(config={...})` accepts an `mcp_servers` map or whether MCP must be configured via Codex's app-server config file. Tracked as Open Question.
- **Approval policy + sandbox interaction with the orchestrator's unattended-run model**: an approval-required policy would deadlock unattended runs. Default `never` (decision #3) avoids this; resolver/validator should reject `approval_policy=untrusted` with `sandbox=danger-full-access` if the combo is unsafe (validator's call).
- **Lock enforcement under bundled hook**: parity with Amp matrix (allow / no-lock / wrong-agent / wrong-namespace / env-missing / canonicalization-parity / disallowed-tool). Codex hook calls `lock_path` directly so canonicalization is byte-equivalent to Mala's existing path.
- **Idle timeout under SDK-managed subprocess**: if `AsyncTurnHandle.stream()` blocks past the idle threshold without surfacing an error, `IdleTimeoutStream` raises `IdleTimeoutError` and the orchestrator retries via `thread_resume`. Test parametrizes both fast and slow streams.
- **`turn/completed` may not fire on Codex-side errors**: `ErrorNotification` is coerced into `AgentResultEvent(is_error=True)` so the lifecycle layer treats it as a failure, not an idle hang.
- **`Thread.read(include_turns=True)` cost / omissions**: F1 spike. If reading native evidence is paginated/expensive/missing fields, fall back to tee (F3).
- **Thread-id namespace collision**: Codex `thr_*` ids must not collide with Amp `T-*` ids on disk. F3 fallback path `~/.config/mala/codex-sessions/{thread_id}.jsonl` is namespaced separately from Amp's `~/.config/mala/amp-sessions/`.
- **Runtime package missing at import time**: `codex_app_server` imports succeed but the bundled binary is absent → `Codex()` initialization fails with a nondescriptive error. `install_prerequisites()` probes both layers.
- **`ErrorNotification` arrives without `TurnCompletedNotification`**: emit failed `AgentResultEvent` so lifecycle treats as failure.
- **Stream stalls**: existing idle timeout raises and retry resumes the Codex thread.
- **SIGINT mid-turn**: interrupt active turn and close `AsyncCodex`. `SigintGuard` integrates without modification provided we expose the close hook.
- **SDK schema drift**: fake tests catch expected mappings; real e2e catches runtime drift when enabled.
- **Phase A regression risk** (highest impact): `MessageStreamProcessor` rewrite + `LogProvider`→`EvidenceProvider` rename + `runtime.options` removal touch core hot paths. Mitigations:
  - Existing Claude **and** Amp test suites are the regression guard. CI must run both.
  - Each Phase A sub-task ships as its own PR (decision #15), smallest viable diff per PR — A1 the exception (single big-bang per decision #6).
- **Default coder change**: changing `DEFAULT_CODER` would be a behavior change for users who currently run without `--coder`. Out of scope for this plan.
- **Lazy-import test coverage**: importing `codex_provider` must not import `codex_app_server`. Same lazy-import-guard pattern as `test_amp_provider.py`.
- **Hook-trust UX**: if Codex requires interactive trust acceptance (Open Question), auto-trust impossible — fall back to documented one-time step. E5 selftest catches the resulting "hook installed but Codex didn't load it" state.

### Breaking Changes & Compatibility

- **Internal (no public API)**:
  - Phase A1: `MessageStreamProcessor` accepts `AgentEvent`s, not Anthropic-shaped messages. All in-tree adapter call sites updated; no external callers.
  - Phase A2: `SDKClientFactoryProtocol` shrinks. In-tree call sites updated; no external callers.
  - Phase A3: `runtime.options` direct reads removed; pipeline calls `provider.client_factory.create(runtime)`. In-tree only.
  - Phase A4: `AgentProvider.mcp_server_factory()` added; orchestrator-side dispatch deleted.
  - Phase A5: `LogProvider` renamed to / replaced by `EvidenceProvider`; method names change. Renamed file. In-tree call sites updated.
  - Phase A6: `AgentRuntimeBuilder` cross-coder fluent surface narrows. In-tree only.
  - Phase A7: lifecycle effects no longer assume on-disk JSONL.
  - Phase A8: selection enums widen. Existing values still valid.
  - `src/infra/clients/amp_messages.py` is deleted.
  - All A* break tree-internal call sites; per CLAUDE.md no shims; updated as a single PR per sub-task.
- **External / user-facing**:
  - **Default behavior unchanged**: omitting `--coder` selects whatever the current default is.
  - New CLI flags are additive (`--codex-model`, `--codex-effort`).
  - New `coder_options.codex` mala.yaml fields are optional with defaults; existing configs remain valid.
  - Existing telemetry attribute names unchanged; the `coder` attribute domain widens to include `"codex"`.
  - Codex users get an explicit error if SDK / runtime / auth / plugin missing — fail-closed parity with Amp's binary-install gate.

### Mitigations

- No external/public CLI behavior changes for Claude or Amp.
- Current default coder remains unchanged.
- Existing `mala.yaml` files remain valid.
- New Codex config fields are optional.
- Existing Claude and Amp tests run on every Phase A PR.
- Static searches/tests verify provider-name MCP dispatch and Anthropic message duck typing are removed from shared paths.
- Phase A is gated by Claude + Amp regression suites as the safety net (per-PR cadence per decision #15).
- Real-Codex e2e coverage catches SDK drift before merge.
- Pin SDK + runtime versions; document tested range.
- Comprehensive unit + integration tests for each phase.

## Testing & Validation Strategy

- **Unit Tests**
  - Phase A1: `tests/unit/core/protocols/test_agent_event.py`, `tests/unit/pipeline/test_message_stream_processor_kinds.py`.
  - Phase A2: `tests/unit/core/protocols/test_sdk_factory_slim.py`.
  - Phase A4: `tests/unit/orchestration/test_factory_no_name_branch.py`.
  - Phase A5: `tests/unit/core/protocols/test_evidence_provider.py`.
  - Phase B: `tests/unit/orchestration/test_factory_codex_selection.py`, `tests/unit/cli/test_codex_flags.py`, `tests/unit/domain/validation/test_codex_config.py`.
  - Phase C: `tests/unit/infra/clients/test_codex_client.py`, `tests/unit/infra/clients/test_codex_runtime.py`.
  - Phase D: `tests/unit/infra/clients/test_codex_event_adapter.py`.
  - Phase E: `tests/unit/infra/hooks/test_codex_pre_tool_use.py`, `tests/unit/infra/clients/test_codex_plugin_installer.py`, `tests/unit/infra/clients/test_codex_provider.py`.
  - Phase F: `tests/unit/infra/clients/test_codex_evidence_provider.py`.
  - Phase G: `tests/unit/orchestration/test_codex_mcp_merge.py`.
  - Phase H: `tests/unit/pipeline/test_idle_retry_codex.py`.
  - General: `AgentEvent` protocol usage; slim SDK factory protocol conformance; config parsing and validation for `coder_options.codex`; Codex runtime builder defaults (`model=gpt-5.5`, `sandbox=danger-full-access`, `approval_policy=never`); Codex event adapter mappings for all supported notification/item types; Codex hook allow/deny behavior for lock ownership, dangerous commands, disallowed tools; Codex evidence extraction from fake `Thread.read(include_turns=True)` payloads; provider lazy-import and fail-closed prerequisite errors.
- **Integration / End-to-End Tests**
  - Phase C: `tests/integration/test_codex_provider.py` (fake `AsyncAppServerClient`).
  - Phase E: `tests/integration/test_codex_lock_enforcement.py` (parity matrix).
  - Phase F: `tests/integration/test_codex_evidence_cross_resume.py`.
  - Phase G: `tests/integration/test_codex_mcp_round_trip.py`.
  - Phase H: `tests/integration/test_fixer_follows_codex.py`, `tests/integration/test_codex_resume.py`.
  - Phase I: `tests/e2e/test_codex_real_sdk.py` (gated on SDK + runtime + auth).
  - Fake `AsyncCodex` lifecycle through provider, runtime, client, processor, and session runner; fake Codex resume and idle retry; fixer follows Codex when main coder is Codex; Codex locking MCP config assembly and round-trip; Codex lock-enforcement parity with Amp integration scenarios; gated real Codex e2e smoke (one prompt, one turn, event stream completes, evidence is readable).
- **Regression Tests**
  - Existing Claude integration suite passes unchanged after Phase A.
  - Existing Amp integration suite passes unchanged after Phase A.
  - `tests/integration/test_amp_lock_enforcement.py` continues to pass after E (no Amp behavior change).
  - Existing config validation for Claude/Amp configs.
  - Existing validation evidence extraction semantics.
  - Existing idle retry and fixer behavior for Claude/Amp.
- **Manual Verification** before declaring done:
  - `mala run --coder codex` against a real beads issue end-to-end; verify commit produced, gate passes, review runs, issue closes.
  - Verify dangerous prompt is blocked (e.g., `rm -rf` rejected).
  - Verify file edits without lock are rejected; verify file edits with correct lock allowed.
  - Verify validation evidence is available (lint/test command output reflects in gate).
  - Verify fixer flow uses Codex.
  - Verify CLI/env/yaml precedence by overlaying flags.
  - A resume after idle timeout continues the same Codex thread.
- **Monitoring / Observability**
  - Telemetry `coder` attribute includes `codex`.
  - Hook self-test emits structured failure reasons.
  - Provider prerequisite errors include SDK version/runtime/auth context where safe.
  - Evidence-provider failures identify thread ID and evidence source.
  - `CodexHookNotActiveError` and `CodexNotInstalledError` surface in logs with structured `Reason`.
  - Real-Codex e2e gate runs in CI (skipped if env vars missing).

Recommended verification commands:
```bash
uv run pytest -m unit
uv run pytest -m integration -n auto
uv run pytest -m e2e
uvx ruff check .
uvx ty check
```

### Acceptance Criteria Coverage

| AC | Approach |
|---|---|
| AC #1: `mala run --coder codex` invokes Codex instead of Claude/Amp | Phase B selection (B1, B3) — new `--coder codex` value → `MalaConfig.coder` → `CodexAgentProvider` selected by factory. Phase C client + fake Codex integration; Phase I real Codex e2e. |
| AC #2: Default behavior unchanged when `--coder` is absent | Phase B config tests; Claude/Amp regression suites; default coder unchanged; Claude/Amp behavior byte-equivalent after Phase A regression guard. |
| AC #3: Selection precedence is CLI > env > yaml > default | Phase B config and CLI precedence tests; mirrors `coder` resolver (B1). |
| AC #4: Codex model + effort + approval_policy + sandbox configurable | Phase B config/schema tests, Phase C runtime tests; `CoderOptions.codex` + Codex runtime tests (B2). |
| AC #5: Fixer agents follow main coder | Phase H fixer integration test; `FixerService` consumes `AgentProvider`; H2 test asserts. |
| AC #6: Per-issue lifecycle (gate + review + close) works for Codex | Phases C–H fake integration; `AgentSessionRunner` is coder-agnostic post-Phase A; Phase I real e2e smoke. |
| AC #7: Validation evidence is available for Codex runs | Phase F evidence provider; `CodexEvidenceProvider` reads native `Thread.read()` (F2) or tee (F3); validation extractor tests. |
| AC #8: Resume + idle/review retries work for Codex | Phase H resume and idle retry tests; `with_resume` → `AsyncCodex.thread_resume` (H1); idle retry unchanged (H3). |
| AC #9: Critical safety controls (sandbox + approvals + lock-enforce) hold for Codex | Phase E hook tests; lock parity integration; provider self-test; sandbox + approvals + lock-enforce hold for Codex. **Lock-enforcement parity is guaranteed by both the `apply_patch` file-edit branch *and* the bash branch's shell-command write-path enforcement (see AC #19); both must hold for AC #9 to be satisfied.** |
| AC #10: `--coder claude` and `--coder amp` runs unchanged | Phase A regression suites and existing integration tests; existing regression suites + Phase A behavioral guarantees. |
| AC #11: Codex prerequisites and limitations are documented | Phase I docs updates: README + cli-reference + project-config + architecture + development (I4). |
| AC #12: Real Codex SDK drift is detectable before merge | Gated `tests/e2e/test_codex_real_sdk.py` (I3). |
| AC #13: `mala.yaml` validation rejects unknown `coder_options.codex.*` values | Phase B validation tests; strict-enum validation in `src/domain/validation/config.py` (B2). |
| AC #14: Codex install/auth missing → fail-closed actionable error | Phase I prerequisite tests; `install_prerequisites()` + `CodexNotInstalledError` / `CodexHookNotActiveError` (E5, I1). |
| AC #15: `MessageStreamProcessor` consumes `AgentEvent`s without provider-specific branches (Phase A) | Phase A1 unit tests + Claude/Amp/Codex integration; static regression search. |
| AC #16: `SDKClientFactoryProtocol` is provider-agnostic (Phase A) | Phase A2 protocol-shape test; Amp stub removal. |
| AC #17: MCP factory dispatch is provider-owned (Phase A) | Phase A4 test asserts `factory.py`/`orchestrator.py` no longer branch on `provider.name`. |
| AC #18: `EvidenceProvider` is the unified evidence surface (Phase A) | Phase A5 protocol conformance test; Phase F Codex evidence tests; protocol conformance shared across Claude + Amp + Codex. |
| AC #19: Shell-command writes are gated against locks (Phase E shell-command write-path enforcement) | `tests/integration/test_codex_shell_lock_enforcement.py` — `printf X > file`, `sed -i ... file`, `python -c "open('file','w').write('X')"`, `cp src dst`, `mv a b`, `rm f`, `touch f`, `git checkout -- f`, `patch < diff`, etc. are all rejected when `file`/destination is not locked by the current agent (or is locked by another agent). Heuristic coverage table is the source of truth; residual obfuscation gap is acknowledged in Phase E "Shell-command write-path enforcement" sub-section. Required for AC #9 lock-enforcement parity to hold on the bash execution path. |

## Spec/Legacy Fidelity

This plan derives from a user request and explicitly references `plans/2026-04-29-amp-provider-plan.md` for shape. It does not bind to any prior Codex plan (the 2025-12-30 Codex provider plan is referenced only for shape; that plan was never implemented and used a different abstraction surface). The most significant departure from the Amp plan shape is the explicit Phase A refactoring epic: where the Amp plan landed against the existing protocol surface and accumulated friction, this plan addresses that friction in-flight (per user direction) before Codex implementation begins.

### Deviation Log

| Source | Deviation | Rationale | Approved? |
|---|---|---|---|
| Amp plan | Phase A refactoring epic precedes Codex implementation | User requested it explicitly; addresses architecture friction Amp surfaced. | Yes (decision #1) |
| Amp plan | Single big-bang PR for A1 | User explicit choice; mitigated by per-PR Claude+Amp regression suites. | Yes (decision #6) |
| Amp plan | Use `AsyncCodex` SDK instead of wrapping a CLI subprocess directly | Codex Python SDK owns app-server subprocess lifecycle; no subprocess wrapping in `CodexClient`. | Yes (SDK shape in context) |
| Amp plan | No `--dangerously-allow-all`-style CLI flag | Codex SDK doesn't expose one; sandbox + approval_policy + bundled hook are the safety surface. | Implicit |
| Amp plan | Safety ships as Codex plugin with `PreToolUse` command hook, not Amp TypeScript plugin | Codex has Claude-Code-style hooks and plugin packaging | Yes (decisions #4, #5) |
| Amp plan | `MALA_DISALLOWED_TOOLS` enforced for Codex (advantage over Amp gap) | Codex hook is a Python entry point that can read env vars trivially; Amp's TS plugin had a known gap. | Implicit (decision #4) |
| Amp plan | Evidence uses native `Thread.read(include_turns=True)` rather than primary tee JSONL | User selected native Codex evidence | Yes (decision #11) |
| Amp plan | Hook trust auto-write via `install_prerequisites` | Author recommendation; mitigates "hook installed but dormant" silent failure | Yes (decision #16; pending E spike result) |
| Existing implementation | Delete `amp_messages.py` instead of preserving compatibility shim | Repo rules prohibit shims/re-exports; A1 replaces the need for synthetic Anthropic messages | Yes (decision #6 and repo instructions) |

## Open Questions

These are deferred to implementation-time spikes; not blockers for the plan.

- **Codex MCP wire shape (Phase G)**: how does `codex app-server` accept stdio MCP server specs? Inline via `Codex.thread_start(config={...})` (does it accept `mcpServers` inline?), via Codex's app-server config file, via plugin `.mcp.json`, or via runtime parameter? Affects G1/G2 wiring shape but not whether MCP works.
- **Codex `ReasoningEffort` enum values (Phase B)**: exact enum members for `validate_codex_effort` parser. Source dynamically from the SDK at parser-build time so the list stays in sync. Affects B2.
- **Per-process env isolation for the Phase E hook contract (Phase B/C blocker)**: verify whether `codex_app_server.AsyncCodex` accepts an explicit `env=` dict (via `AppServerConfig` or another constructor argument) or per-thread env on `thread_start` for plumbing `MALA_AGENT_ID` / `MALA_LOCK_DIR` / `MALA_REPO_NAMESPACE` / `MALA_DISALLOWED_TOOLS` through to the bundled `PreToolUse` hook. Mala MUST NOT mutate `os.environ` in the parent process — under `--max-agents > 1`, concurrent agents would leak each other's `MALA_AGENT_ID` to subprocesses (parity target: `AmpRuntimeBuilder.build()` constructs a per-subprocess env dict explicitly, `src/infra/clients/amp_runtime.py:315-327`). If the spike confirms explicit per-process env injection is supported, plumb the `MALA_*` vars through it. If not, the documented fallback (Phase E "Fallback transport: per-session state file") writes `~/.config/mala/agent-state/{session_id}.env` immediately after `thread_start` (write-temp-then-rename, fsync'd before first turn fires) and the hook reads it keyed by the `session_id` it receives in stdin — no `agent_id` index is required because `session_id` is one-to-one with the spawning agent. Two alternatives are noted but rejected for MVP: (a) propose a `config.env` field upstream in the Codex SDK (timeline-dependent), or (b) serialize Codex agent spawning so only one agent at a time has `MALA_AGENT_ID` in `os.environ` (this defeats `--max-agents > 1` for Codex). **Phase B/C blocker** — must be resolved before Phase E hook input source is finalized; the hook implementation accepts either source so the same binary works under either spike outcome.
- **Codex plugin discovery directory (Phase E)**: `~/.codex/plugins/` is the assumption. Spike confirms exact path before E4's installer copies `plugins/codex/mala-safety/.codex-plugin/` into place. The source-tree layout (`.codex-plugin/{plugin.json, hooks.json, .mcp.json}`) is fixed regardless of spike outcome, so the installer is a straight copy with no path translation. No design impact.
- **Codex hook-state file location and format (Phase E, decision #16)**: where does Codex persist `HookStateToml.trusted_hash`? Schema is at `codex-rs/core/config.schema.json:1016` but the on-disk path needs reverse-engineering. **Risk**: if Codex's trust UX requires user-interactive acceptance, auto-trust is impossible — fall back to a documented one-time step (E6).
- **Codex model id confirmation (Phase B/I)**: `gpt-5.5` is the user-stated default (decision #9); confirm the exact tag Codex's API accepts (`gpt-5.5` vs `gpt-5.5-codex` etc.). Surfaces during real-Codex e2e (I3).
- **`Thread.read(include_turns=True)` viability for evidence (Phase F1 spike)**: validates decision #11. Confirm whether `Thread.read(include_turns=True)` exposes `aggregated_output` for `CommandExecutionThreadItem`. If the spike disconfirms (paginated, expensive, missing fields, or stale across resume), reverse to F3 tee fallback.
- **Local-shell tool name in Codex (Phase D/E)**: `bash` is the assumption for the dangerous-cmd hook branch. Confirm Codex's actual local-shell tool name during Phase D event-mapping spike.
- **File-edit tool name(s) in Codex (Phase D/E)**: `apply_patch` is the assumption for the lock-enforce hook branch. Confirm full set during Phase D.
- **Codex auth detection (Phase I)**: how do we probe Codex auth state from `install_prerequisites()` without triggering an interactive prompt? Likely by attempting `Codex().__enter__()` and catching the auth-error class.
- **Real-Codex e2e gate env var (Phase I)**: which env var indicates Codex is installable in CI? Mirrors `tests/e2e/test_amp_real_cli.py` gating; for Codex it's `import codex_app_server` succeeding + binary check + auth-detection helper from above.
- **`pyproject.toml` extras vs hard dep (Phase I)**: `mala[codex]` extras vs always-installed dependency. Lazy-import + auth-check pattern allows extras; preferred for keeping Claude/Amp-only installs lean.
- **Future cross-coder console-script consolidation (out of scope for this plan)**: this plan adds `mala-codex-mcp-locking` as a new sibling to the existing `mala-amp-mcp-locking` (both backed by `src/infra/tools/locking_mcp_stdio.py`), and explicitly does NOT rename or remove the Amp script — already-installed Amp plugins reference the old name in their bundled `.mcp.json`, and renaming here would break them on first post-upgrade run unless plugin re-install + version-marker bump are handled in lockstep. A unified `mala-mcp-locking` name is desirable long-term but is a separate dedicated migration plan that must coordinate the `AmpPluginInstaller` `(amp_version, plugin_hash)` cache-key bump, the bundled Amp `.mcp.json` update, and a release-notes entry — not bundled with "add Codex".

## Next Steps

After this plan is approved, run `/cerberus:create-tasks` to generate execution artifacts:
- `--beads` → Beads issues with dependencies for multi-agent execution (preferred for the multi-epic shape).
- (default) → TODO.md checklist for simpler tracking.

Phase ordering is partially linear with the following rough breakdown (each phase is its own epic; see **Phase Dependencies** above):

1. **Phase A — Provider-agnostic architecture cleanup** (sub-tasks A1–A8; each its own PR; Claude+Amp regression guard per PR; A1 single big-bang).
2. **Phase B — Codex selection wiring + provider stub** (depends on A8).
3. **Phase C — `CodexClient`** (depends on A1, A2, A3, A6, B).
4. **Phase D — Item → AgentEvent mapping** (depends on A1, C).
5. **Phase E — Codex safety model** (depends on B, C; spike-heavy).
6. **Phase F — Codex evidence provider** (depends on A5, C; F1 spike-first).
7. **Phase G — Codex MCP integration** (depends on A4, C, E; spike-heavy).
8. **Phase H — Resume + fixer follows main coder + idle retry** (depends on C, F).
9. **Phase I — Prerequisites + telemetry + tests + docs + ship** (depends on B–H all landed).
