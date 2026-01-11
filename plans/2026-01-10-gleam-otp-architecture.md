# Gleam/OTP Rewrite: Comprehensive Technical Design

Date: 2026-01-10
Owner: mala
Status: draft

> **Note on code examples:** All Gleam code blocks in this document are **pseudocode** illustrating design intent. They do not represent final, compilable Gleam syntax. Actual implementation will use `gleam_otp` and `gleam_erlang` libraries with different APIs.

## Executive Summary

This document proposes a full rewrite of mala on Gleam/OTP to reduce tech debt, improve concurrency safety, and make agent backends truly pluggable (Claude, Codex, AMP, OpenCode, etc.). The design adopts OTP-native supervision, state machines, and event-driven orchestration to replace Python asyncio + SDK-specific glue. The result is a smaller, more reliable, and more extensible system built for orchestration workloads.

## Goals

- Replace Python implementation with Gleam/OTP (clean break, no backward compatibility).
- Make agent backends swappable without changing orchestration logic.
- Use OTP semantics for supervision, retries, timeouts, and failure isolation.
- Centralize configuration into a typed schema with layered overrides.
- Leverage Erlang/OTP libraries instead of bespoke infrastructure.
- Improve observability with structured events and telemetry.

## Non-Goals

- Multi-repo orchestration (out of scope for v1 rewrite).
- Distributed clustering (not needed for v1).
- UI/GUI or web console.
- Automatic issue decomposition (still driven by Beads).

## Assumptions & Prerequisites

### Required Binaries

| Binary | Min Version | Purpose | Detection | Required When |
|--------|-------------|---------|-----------|---------------|
| `git` | 2.20+ | Version control | `git --version` | Always |
| `bd` | 0.1.0+ | Beads issue provider | `bd --version` | Always |
| `claude` | 1.0.0+ | Claude CLI adapter | `claude --version` | `agents.profile` uses claude |
| `erlang` | OTP 26+ | Runtime | Built-in | Always |

**Startup validation (profile-driven):** On `mala run`, validate only the binaries required for the configured agent profile and enabled subsystems:
- Always required: `git`, `bd`, `erlang`
- Claude profile: additionally requires `claude`
- Deferred backends (Codex, AMP, OpenCode): validation added when adapters ship in v1.1+

Fail with clear error message listing missing/outdated binaries for the selected profile only.

### Supported Operating Systems

- **Linux:** Primary target, fully supported (Ubuntu 20.04+, Debian 11+, Fedora 38+)
- **macOS:** Supported (12.0+ Monterey)
- **Windows:** Not supported in v1 (WSL2 recommended)

### Filesystem Requirements

| Path | Purpose | Required Permissions |
|------|---------|---------------------|
| `~/.config/mala/` | User config, runs, logs | Read/Write |
| `/tmp/mala-*` | Lock files, temp worktrees | Read/Write |
| Project directory | Code, git operations | Read/Write |

**Disk space:** Minimum 1 GB free for run logs and worktrees.

### Environment Requirements

- Git repository initialized in project directory
- Network access for agent backends (Claude API, Codex API)
- API keys configured for agent backends (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`)

### v1 Platform Constraints

This subsection consolidates OS and tooling constraints for v1 implementation.

**Operating system:**
- **Linux:** Primary target (Ubuntu 20.04+, Debian 11+, Fedora 38+)
- **macOS:** Supported (12.0+ Monterey)
- **Windows:** Not supported (WSL2 recommended)

**Unix socket constraints:**

| Platform | Max Socket Path | Mitigation |
|----------|-----------------|------------|
| Linux | 108 chars | Hash run_id to keep paths short |
| macOS | 104 chars | Hash run_id to keep paths short |
| Windows | N/A | Not supported (use WSL2) |

Socket path mitigation is **always enabled** (not configurable). See "Socket path and cleanup" for implementation.

**Runtime versions (pinned):**

| Component | Minimum Version | Notes |
|-----------|-----------------|-------|
| Erlang/OTP | 26.0 | Required for modern gen_statem APIs |
| Gleam | 1.0.0 | Stable release |
| `erlexec` | 1.18.0 | Process execution library |
| `disk_log` | OTP 26 built-in | Append-only log |
| `dets` | OTP 26 built-in | Disk-based ETS |
| `digraph` | OTP 26 built-in | Deadlock detection graph |

**External binary versions:**

| Binary | Min Version | Detection |
|--------|-------------|-----------|
| `git` | 2.20+ | `git --version` |
| `bd` | 0.1.0+ | `bd --version` |
| `claude` | 1.0.0+ (for hooks) | `claude --version` |

**Library dependencies:**

| Library | Purpose | Version |
|---------|---------|---------|
| `telemetry` | Observability | ^1.0 |
| `opentelemetry` | Tracing export | ^1.0 |
| `erlexec` | Process execution | ^1.18 |
| `tom` | TOML parsing | ^2.0 |
| `jiffy` | JSON encoding/decoding | ^1.1 |

### Assumptions

1. **Single-user:** One mala process per repository at a time (no multi-user coordination).
2. **Clean git state:** Repository should have no uncommitted changes in tracked files at run start (configurable via `require_clean_git`).
3. **Agent cooperation:** Agents are expected to use MCP lock tools when available (enforcement is best-effort).
4. **Network stability:** Transient network failures are retried, but prolonged outages fail the run.

### `require_clean_git` Check Points

The `require_clean_git` setting controls when git cleanliness is validated:

| Check Point | Behavior | Failure Action |
|-------------|----------|----------------|
| **Run start** | Check once at `mala run` startup | Abort run with error |
| Issue attempt start | NOT checked | N/A |
| Gate/review | NOT checked | N/A |
| Resume | NOT checked | N/A |

**Rationale:** Checking only at run start allows the natural workflow where agents make uncommitted changes during their work. Mid-run checks would fail constantly as agents edit files.

**Multi-issue concurrency:** With multiple agents working concurrently, the repo will have uncommitted changes from all active agents. This is expected and allowed. Each agent is responsible for committing their own changes.

**Failure behavior:**
```gleam
pub fn validate_git_clean(config: Config, repo: String) -> Result(Nil, Error) {
  case config.run.require_clean_git {
    False -> Ok(Nil)
    True -> {
      case git.status(repo) |> has_uncommitted_changes() {
        False -> Ok(Nil)
        True -> Error(GitNotClean(
          "Repository has uncommitted changes. " <>
          "Commit or stash changes before running, or set require_clean_git = false."
        ))
      }
    }
  }
}
```

## Required Behavior (Parity With Current System)

### v1 Required (Must Ship)

- CLI commands: `run`, `status`, `clean`, `logs`
- Beads issue provider (bd CLI): filtering, claiming, closing, epic relationships
- Parallel issue processing with max concurrency and max issue caps
- Ordering modes: focus, epic-priority, issue-priority, input
- **Shared-repo mode** with lock enforcement via MCP tools + PreToolUse hooks
- Per-issue agent sessions with:
  - idle timeout recovery
  - resume with same session on retry
  - context pressure handling
  - no-progress detection
- Quality gate (orchestrator-executed):
  - require commit with bd-<issue_id>
  - require evidence for all validation commands
- Validation (orchestrator-executed):
  - spec-driven from config
  - per-session + global scope
  - optional coverage + e2e
- Locking:
  - lock acquire/release MCP tools
  - file-write enforcement via PreToolUse hooks (best-effort)
  - deadlock detection
- Event logging, run metadata, and status reporting

### v1.1 Deferred

- CLI command: `epic-verify`
- Per-issue worktrees (for non-hook backends)
- Codex, AMP, OpenCode adapters

### v1 Stubs (Present but Non-Functional)

The following components appear in the OTP topology and config schema but are **no-op stubs in v1**:

| Component | Why Stub | v1 Behavior |
|-----------|----------|-------------|
| `epic_sup` | `epic-verify` deferred to v1.1 | Not spawned in v1 |
| `epic_verification.enabled` config | Epic verify deferred | Silently ignored in v1 (always treated as `false`) |

> **Note:** Code review (Cerberus) IS included in v1. Only epic verification is deferred.

**v1 config validation rule:**
- Config field `epic_verification.enabled` is accepted but **silently ignored** in v1
- v1 always behaves as if `epic_verification.enabled = false` regardless of config value
- `review.enabled` IS respected in v1 - code review is fully functional

**Sample config (v1):**
```toml
[review]
enabled = true  # v1: fully functional

[epic_verification]
enabled = false  # v1: silently ignored, always treated as false
```

## Architecture Options (OTP-native)

1) Per-Issue gen_statem + supervised workers (recommended)
   - Pros: explicit lifecycle, clean retry semantics, strong isolation.
   - Cons: more state machine code than a pipeline.

2) GenStage pipeline
   - Pros: backpressure and throughput are built-in.
   - Cons: awkward for feedback loops (gate/review retry).

3) Event-sourced orchestrator
   - Pros: full auditability and replay.
   - Cons: more infrastructure, likely overkill for v1.

4) Lock-centric actor model
   - Pros: best for lock contention.
   - Cons: harder to reason about issue lifecycle.

Chosen: Option 1 with a light event log for observability.

## High-Level Topology

mala_app
  mala_sup
    config_server        (gen_server)
    event_bus            (pg/gproc + telemetry)
    lock_sup
      lock_server        (gen_server + ETS)
      deadlock_monitor   (gen_server + digraph)
      mcp_lock_server    (stdio JSON-RPC)
    issue_provider_sup
      beads_provider     (gen_server, port)
    run_sup (dynamic)
      run_manager        (gen_server)
      scheduler          (gen_server)
      worker_sup         (dynamic supervisor)
      validation_sup     (supervisor)
      review_sup         (supervisor)
      epic_sup           (supervisor)
      log_sink           (disk_log)

## Module Layout (Gleam)

src/mala/
  app.gleam
  cli.gleam
  config/
    loader.gleam
    types.gleam
    validate.gleam
  domain/
    events.gleam
    issue.gleam
    epic.gleam
    run.gleam
    session.gleam
    lifecycle.gleam
    validation_spec.gleam
    validation_config.gleam
    review.gleam
    resolution.gleam
  orchestrator/
    run_manager.gleam
    scheduler.gleam
    issue_worker.gleam
    session_runner.gleam
    worker_supervisor.gleam
    event_router.gleam
    run_state.gleam
  agents/
    adapter.gleam
    claude_cli.gleam
    codex_app.gleam
    amp_cli.gleam
    open_code_cli.gleam
  validation/
    runner.gleam
    cache.gleam
    evidence.gleam
    coverage.gleam
    e2e.gleam
    # worktree.gleam - v1.1 only (not in v1)
  review/
    adapter.gleam
    cerberus_cli.gleam
  epic/
    verifier.gleam
    scope.gleam
  infra/
    command_runner.gleam
    git.gleam
    lock_server.gleam
    deadlock.gleam
    mcp_lock_server.gleam
    log_sink.gleam
    telemetry.gleam

## Beads Provider Interface

The Beads provider wraps the `bd` CLI to fetch, claim, and close issues.

### bd CLI Commands Used

| Operation | Command | Output Format |
|-----------|---------|---------------|
| List issues | `bd list --json --status ready` | JSON array |
| List by epic | `bd list --json --epic <id>` | JSON array |
| Get issue | `bd show <id> --json` | JSON object |
| Claim issue | `bd claim <id>` | Exit code |
| Close issue | `bd close <id> -m "<message>"` | Exit code |
| Create issue | `bd create --json -t "<title>" -d "<desc>"` | JSON object |

### Issue JSON Schema (bd output)

```json
{
  "id": "bd-123",
  "title": "Fix login bug",
  "description": "Users cannot log in when...",
  "status": "ready",
  "priority": "high",
  "labels": ["bug", "auth"],
  "parent_epic": "bd-100",
  "created_at": "2026-01-10T12:00:00Z",
  "updated_at": "2026-01-10T14:30:00Z"
}
```

### Provider Implementation

```gleam
pub type BeadsProvider {
  bd_path: String,  // Path to bd binary
  workdir: String,  // Repo root
}

pub fn list_issues(provider: BeadsProvider, filter: IssueFilter) -> Result(List(Issue), Error) {
  let args = ["list", "--json", "--status", filter.status]
  let args = case filter.epic {
    Some(epic) -> list.append(args, ["--epic", epic])
    None -> args
  }
  use output <- result.try(command_runner.run(Command(provider.bd_path, args), ...))
  json.decode_list(output.stdout, issue_decoder)
}

pub fn close_issue(provider: BeadsProvider, issue_id: String, message: String) -> Result(Nil, Error) {
  let cmd = Command(provider.bd_path, ["close", issue_id, "-m", message])
  use output <- result.try(command_runner.run(cmd, ...))
  case output.exit_code {
    0 -> Ok(Nil)
    _ -> Error(CloseError(output.stderr))
  }
}
```

## Core Data Models

- AgentEvent
  - tool_use, tool_result, assistant_text, usage, checkpoint, final
  - **schema_version:** int (v1 = 1)
- Issue
  - id, title, status, priority, parent_epic
- SessionState
  - session_id, agent_backend, log_offset, attempt counters
- GateResult
  - passed, failure_reasons, evidence, no_progress
- ReviewResult
  - passed, findings, severity
- ValidationSpec
  - commands, scope, coverage, e2e, patterns

### Event Schema Versioning

All persisted events include a `schema_version` field for forward compatibility.

```gleam
pub type PersistedEvent {
  PersistedEvent(
    schema_version: Int,  // Current: 1
    timestamp: Timestamp,
    event: AgentEvent,
  )
}

pub fn read_event(data: BitArray) -> Result(AgentEvent, Error) {
  use persisted <- result.try(decode_persisted(data))
  case persisted.schema_version {
    1 -> Ok(persisted.event)
    v -> Error(UnsupportedSchemaVersion(v, "upgrade mala to read this log"))
  }
}
```

**Versioning policy:**
- v1: Initial schema (this design)
- Schema changes increment version
- Old versions: Reject with clear error message directing user to upgrade
- No automatic migration in v1 (logs are append-only, not rewritten)

### ResumeHandle (Session Resume Semantics)

Resume is **best-effort**: the orchestrator persists what it can, but backends may not support true continuation. The `ResumeHandle` model defines what's persisted and what guarantees each backend provides.

```gleam
pub type ResumeHandle {
  ResumeHandle(
    backend: String,              // "claude_cli" | "codex_app" | etc.
    session_id: Option(String),   // Backend-specific session identifier
    resume_token: Option(String), // Opaque token for resume (Claude: --resume value)
    checkpoint_token: Option(String), // Mid-session checkpoint (context pressure)
    last_commit: Option(String),  // Git commit SHA at session start
    // NOTE: No per-issue cursor - see "Global Cursor Model" section
  )
}

// EventCursor is an opaque continuation from disk_log:chunk/2
// See "Global Cursor Model" section for authoritative definition
pub type EventCursor {
  EventCursor(continuation: Dynamic)
}
```

**Invariants per backend:**

| Backend | session_id | resume_token | checkpoint_token | Resume Behavior |
|---------|------------|--------------|------------------|-----------------|
| Claude CLI | Yes (from Final) | Yes (--resume) | Yes | True continuation: same context, picks up from last message |
| Codex App | Yes (conversation_id) | No | No | New turn in same conversation, may lose recent context |
| AMP CLI | No | No | No | Fresh session with prompt containing prior summary |
| OpenCode | TBD | TBD | TBD | TBD |

**What "same session" means:**
- **True continuation (Claude):** Agent resumes with full prior context, can reference previous tool calls/results.
- **Conversation continuation (Codex):** Agent sees conversation history, but may summarize/compress older turns.
- **Fresh with summary (AMP, OpenCode):** New session, orchestrator injects a summary prompt with: issue description, previous attempts, what was tried, what failed.

**Orchestrator responsibilities:**
1. Persist `ResumeHandle` to DETS after each checkpoint/final event.
2. On retry, check if `resume_token` exists and backend `supports_resume`.
3. If resumable: call `adapter.resume(handle.resume_token)`.
4. If not resumable: call `adapter.start_session()` with injected summary prompt.

**Event replay on resume:**
- A single global cursor tracks the last synced position in the event log.
- On resume, events are replayed from the global cursor and routed by `issue_id`.
- See "Global Cursor Model" section for the authoritative replay algorithm.

```gleam
pub fn resume_or_start(handle: Option(ResumeHandle), adapter: Adapter, prompt: String) -> Session {
  case handle, adapter.capabilities().supports_resume {
    Some(h), True -> adapter.resume(h.resume_token |> option.unwrap(""))
    _, _ -> adapter.start_session(build_summary_prompt(handle, prompt))
  }
}
```

## Agent Backend Abstraction

Behavior: agents/adapter.gleam

- start_session(profile, prompt, env) -> session_ref
- send(session_ref, prompt)
- resume(session_ref, resume_token)
- stop(session_ref)
- stream(session_ref) -> Stream(AgentEvent)
- capabilities() -> {supports_resume, supports_hooks, supports_tool_restrictions}

### Tool Execution Model

**mala is an observer, not a driver.** The agent backend (Claude CLI, Codex, etc.) executes tools autonomously. mala observes the event stream and enforces policies via hooks and configuration.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Backend                             │
│  ┌─────────┐    ┌──────────┐    ┌─────────────┐                 │
│  │  LLM    │───►│  Tools   │───►│  Filesystem │                 │
│  │ (Claude)│    │ (Edit,   │    │  (actual    │                 │
│  └─────────┘    │  Bash)   │    │   writes)   │                 │
│       │         └────┬─────┘    └─────────────┘                 │
│       │              │                                           │
│       ▼              ▼                                           │
│  ┌─────────────────────────────────────────┐                    │
│  │         Event Stream (stdout)            │                    │
│  │  ToolUse → ToolResult → AssistantText   │                    │
│  └─────────────────────┬───────────────────┘                    │
└────────────────────────│────────────────────────────────────────┘
                         │ stdio/JSON
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     mala (Orchestrator)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ SessionRunner│  │ Evidence     │  │ Progress     │          │
│  │ (observer)   │  │ Collector    │  │ Detector     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

**Enforcement mechanisms (how mala controls without driving):**

1. **PreToolUse hooks (Claude CLI):** Before agent executes a tool, mala's hook script can block it.
   - **Note:** Hooks and MCP are separate systems. Hooks are shell scripts invoked by Claude CLI; MCP exposes callable tools.

   **Hook configuration (Claude CLI `settings.json`):**
   ```json
   {
     "hooks": {
       "PreToolUse": [
         {
           "matcher": "*",
           "command": "mala internal-hook pretooluse --socket /tmp/mala-<run_id>.sock"
         }
       ]
     }
   }
   ```

   **Hook protocol:**
   - Claude CLI invokes hook command with tool info on stdin (JSON: `{tool_name, tool_input}`)
   - Hook command connects to mala orchestrator via Unix socket (same as MCP proxy)
   - Orchestrator validates lock ownership for file-writing tools
   - Hook command exits with code 0 (allow) or non-zero (deny, message on stdout)

   **Hook command implementation:**
   ```gleam
   // mala internal-hook pretooluse
   pub fn main(args: List(String)) {
     let socket = parse_socket_arg(args)
     let tool_info = json.decode(io.read_stdin())

     // Send to orchestrator for validation
     let response = socket.send(HookValidation(tool_info))

     case response {
       Allow -> os.exit(0)
       Deny(reason) -> {
         io.write_stdout(reason)
         os.exit(1)
       }
     }
   }
   ```

2. **--allowedTools (Claude CLI):** Restrict which tools the agent can use at session start.
   - Agent cannot call tools not in the allowed list
   - Used to prevent Bash access for certain issues

### Run-time Wiring (Claude CLI Configuration)

This section specifies how mala configures Claude CLI for each run/session.

**Per-run configuration directory:**
```
~/.config/mala/runs/<run_id>/
  settings.json       # Claude CLI settings with hooks configured
  mcp-config.json     # MCP server configuration
  hook.sh             # Hook script (or use mala binary subcommand)
```

**settings.json generation:**

On `mala run` startup, mala generates a per-run `settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "*",
        "command": "mala internal-hook pretooluse --socket /tmp/mala-<short_id>.sock --run-id <run_id>"
      }
    ]
  }
}
```

**Claude CLI invocation per session:**

```gleam
pub fn spawn_claude_session(
  issue_id: String,
  run_config: RunConfig,
) -> Result(Session, Error) {
  let args = [
    "--output-format", "stream-json",
    "--verbose",
    "--print", build_prompt(issue_id),
    "--mcp-server", "mala internal-mcp-proxy --socket " <> run_config.socket_path,
  ]

  // Set CLAUDE_CONFIG_DIR to use per-run settings
  let env = [
    #("CLAUDE_CONFIG_DIR", run_config.config_dir),
    #("ANTHROPIC_API_KEY", run_config.api_key),
  ]

  exec.spawn("claude", args, env: env, workdir: run_config.repo_root)
}
```

**Hook command implementation:**

The hook is a **subcommand of the mala binary**, not a separate script:

```bash
# Hook invocation by Claude CLI
mala internal-hook pretooluse --socket /tmp/mala-abc123.sock --run-id 20260110-143052-a7b3c9d2
```

**Per-session identifiers:**

| Identifier | Scope | Used For |
|------------|-------|----------|
| `run_id` | Per mala run | Socket path, config directory, log correlation |
| `issue_id` | Per issue worker | Lock ownership, event routing |
| `session_id` | Per Claude session | Resume token, event correlation |

**Session-to-issue binding:**

The hook receives `issue_id` via environment variable set by the orchestrator:

```gleam
// When spawning Claude session for an issue
let env = [
  #("MALA_ISSUE_ID", issue_id),
  #("MALA_RUN_ID", run_id),
  // ... other env vars
]
```

The hook reads `MALA_ISSUE_ID` to determine lock ownership for validation.

3. **MCP lock tools (separate from hooks):** Agent can call `acquire_lock`/`release_lock` as MCP tools.
   - MCP proxy (`mala internal-mcp-proxy`) handles tool calls
   - Hook enforcement validates lock ownership before file writes
   - Both proxy and hook connect to same orchestrator socket

4. **Per-issue worktrees (structural):** For backends without hooks, isolation prevents conflicts.
   - Agent can write freely in its worktree
   - mala controls when changes are merged to main

**What mala observes vs controls:**

| Aspect | mala's Role |
|--------|-------------|
| Tool execution | **Observes** (via ToolResult events) |
| Lock acquisition | **Controls** (via MCP lock server) |
| Pre-tool validation | **Controls** (via hooks, where supported) |
| Evidence collection | **Observes** (from event stream) |
| Session lifecycle | **Controls** (start/stop/resume) |
| File writes | **Cannot directly control** (relies on hooks or worktrees) |

### Capability Detection

Capabilities are **adapter-derived only**, not configurable. This ensures consistency between what the orchestrator expects and what the backend can actually do.

**Detection flow:**
1. On adapter initialization, call `adapter.capabilities()`.
2. Adapter probes the backend using safe, local-only methods (see probe policy below).
3. Result is cached for the session lifetime.

#### Capability Probe Policy (Safety Rules)

Probes must be **safe, fast, and side-effect free**. This prevents hanging on network calls or accidentally starting sessions.

| Rule | Requirement |
|------|-------------|
| **Local only** | Probes must not make network requests by default |
| **No side effects** | Probes must not start sessions, authenticate, or modify state |
| **Time-bounded** | Probes must complete within 5 seconds or fail |
| **Fail closed** | On probe failure, default all capabilities to `False` |

**Claude CLI probe (v1):**

Detection uses a **multi-layer approach** for reliability:

1. **Version-based detection (primary):** Parse `claude --version` for known-good versions
2. **Self-test validation (optional):** Run a no-op hook test on startup with `--verify-backend`
3. **Fallback on detection failure:** If detection fails, set `detection_confidence = Low` and capabilities to `False`

```gleam
// v1 uses Bool + DetectionConfidence, NOT tri-state
// See "v1 Capability Rules" below for authoritative type definition

pub fn detect_claude_capabilities() -> Result(Capabilities, CapabilityError) {
  // Step 1: Get version (more reliable than parsing --help)
  use version <- result.try(get_claude_version())

  // Step 2: Check against known capability matrix
  let caps = case version {
    v if version_gte(v, "1.0.0") -> Capabilities(
      supports_resume: True,
      supports_hooks: True,
      supports_tool_restrictions: True,
      detection_confidence: High,
    )
    v if version_gte(v, "0.9.0") -> Capabilities(
      supports_resume: True,
      supports_hooks: False,
      supports_tool_restrictions: False,
      detection_confidence: High,
    )
    _ -> conservative_capabilities()
  }

  Ok(caps)
}

fn get_claude_version() -> Result(String, Error) {
  case exec.run("claude", ["--version"], timeout: 5000) {
    Ok(output) -> {
      // Parse "claude vX.Y.Z" or "X.Y.Z" format
      parse_version(output.stdout)
    }
    Error(_) -> Error(BinaryNotFound("claude"))
  }
}

fn conservative_capabilities() -> Capabilities {
  // When detection fails, assume capabilities are False with Low confidence
  Capabilities(
    supports_resume: False,
    supports_hooks: False,
    supports_tool_restrictions: False,
    detection_confidence: Low,
  )
}
```

**Version contract for Claude CLI:**

| Version | supports_resume | supports_hooks | supports_tool_restrictions |
|---------|-----------------|----------------|---------------------------|
| >= 1.0.0 | Yes | Yes | Yes |
| 0.9.x | Yes | No | No |
| < 0.9.0 | No | No | No |

**Self-test validation (`--verify-backend` flag):**

> **Network requirement:** `--verify-backend` starts a real Claude CLI session and **requires network access and valid API credentials**. This is different from normal capability detection which is local-only.

When `mala run --verify-backend` is passed, run a startup self-test:

1. Create a temporary hook script that writes a marker file
2. Start a minimal Claude session with `--max-turns 1` and a no-op prompt
3. Check if the hook was invoked (marker file exists)
4. Clean up and report result

```gleam
pub fn verify_hook_support(claude_path: String) -> Result(Bool, Error) {
  // Only run if explicitly requested (avoids session startup overhead)
  let marker = "/tmp/mala-hook-test-" <> random_id()
  let hook_script = write_test_hook(marker)

  // Configure Claude to use our test hook
  let settings = create_test_settings(hook_script)

  // Run minimal session
  let result = exec.run(claude_path, [
    "--output-format", "stream-json",
    "--max-turns", "1",
    "--print", "Say 'test'",  // Minimal prompt, no tools needed
  ], timeout: 30000, env: [("CLAUDE_SETTINGS", settings)])

  // Check if hook was invoked
  let hook_worked = file.exists(marker)

  // Cleanup
  let _ = file.delete(marker)
  let _ = file.delete(hook_script)

  Ok(hook_worked)
}
```

**--verify-backend error handling:**

| Condition | Error Message | Exit |
|-----------|---------------|------|
| No network | "Verification requires network access. Claude CLI could not connect to API." | 1 |
| No API key | "Verification requires ANTHROPIC_API_KEY. Set it or skip verification." | 1 |
| Hook not invoked | "Hook verification failed. Claude CLI may not support hooks." | 1 |
| Session timeout | "Verification session timed out. Check network connectivity." | 1 |

**Commands that do NOT require network:**

| Command | Network | Notes |
|---------|---------|-------|
| `mala run` (normal) | Only for agent sessions | Capability detection is local |
| `mala status` | No | Reads local state files |
| `mala logs` | No | Reads local log files |
| `mala clean` | No | Deletes local files |
| `mala run --verify-backend` | **Yes** | Starts real Claude session |

**v1 Capability Rules (Strict, No Worktree Fallback)**

v1 does not support worktrees, so capability failures cannot fall back to structural isolation.

**Capability type (strict bool with confidence):**

```gleam
pub type Capabilities {
  Capabilities(
    supports_resume: Bool,
    supports_hooks: Bool,
    supports_tool_restrictions: Bool,
    detection_confidence: DetectionConfidence,
    detected_version: Option(String),  // For error messages
    probe_method: ProbeMethod,
  )
}

pub type DetectionConfidence {
  High    // Version-based detection succeeded
  Medium  // Version parsed but not in known matrix
  Low     // Detection failed, using conservative defaults
}

pub type ProbeMethod {
  VersionParse   // Parsed from --version output
  SelfTest       // Verified via --verify-backend
  Assumed        // No probe, using defaults
}
```

**v1 Gating Rules:**

| Condition | `max_agents` | Action |
|-----------|--------------|--------|
| `supports_hooks == True` | Any | Proceed |
| `supports_hooks == False` | 1 | Proceed (single-agent, no lock needed) |
| `supports_hooks == False` | > 1 | **Fail startup** |
| Detection confidence `Low` | 1 | Proceed with warning |
| Detection confidence `Low` | > 1 | **Fail startup** (unless `--force`) |

**No worktree fallback in v1:**

v1 does NOT have worktree support. When hooks are unavailable and `max_agents > 1`, there is no fallback - startup fails with a clear error directing the user to either:
1. Use `max_agents = 1` (single-agent mode)
2. Upgrade Claude CLI to a version with hooks
3. Wait for v1.1 which adds worktree isolation

```gleam
pub fn validate_v1_requirements(
  caps: Capabilities,
  config: Config,
) -> Result(Nil, Error) {
  let max_agents = config.run.max_agents |> option.unwrap(1)

  case max_agents, caps.supports_hooks, caps.detection_confidence {
    // Concurrent mode requires confirmed hook support
    n, False, _ if n > 1 ->
      Error(ConcurrentModeUnavailable(
        "Concurrent mode (max_agents > 1) requires hook support. " <>
        "Claude CLI version " <> caps.detected_version <> " does not support hooks. " <>
        "Options: (1) Set max_agents = 1, (2) Upgrade Claude CLI to >= 1.0.0"
      ))

    // Low confidence with concurrency is unsafe
    n, _, Low if n > 1 ->
      Error(CapabilityDetectionFailed(
        "Cannot confirm hook support for concurrent mode. " <>
        "Run with --verify-backend to test hooks, or use max_agents = 1. " <>
        "To proceed anyway (UNSAFE): --force-concurrent"
      ))

    // Single-agent mode doesn't require hooks
    1, False, _ -> {
      log.warn("Hook support unavailable, running in single-agent mode without lock enforcement")
      Ok(Nil)
    }

    // All other cases: proceed
    _, _, _ -> Ok(Nil)
  }
}
```

**--force-concurrent flag (escape hatch):**

For expert users who understand the risks, `--force-concurrent` bypasses capability checks:
- Warning logged: "Running concurrent mode without confirmed hook support - data corruption possible"
- Not recommended for production use
- Useful for testing or when user knows their Claude CLI version supports hooks but detection failed

**For unknown/new backends:**
- Default all capabilities to `False` until explicit adapter implementation and tests
- Network-based capability detection (e.g., `/capabilities` endpoint) is **opt-in** and requires:
  1. Explicit `adapter.requires_network_probe = True` flag
  2. Time-bounded request (5 second timeout)
  3. Failure defaults to conservative capabilities (not blocking startup)

**Capability flags (v1):**
| Capability | Claude CLI | Codex App | AMP CLI | OpenCode |
|------------|------------|-----------|---------|----------|
| `supports_resume` | Yes (session_id) | Deferred | Deferred | Deferred |
| `supports_hooks` | Yes (PreToolUse) | Deferred | Deferred | Deferred |
| `supports_tool_restrictions` | Yes (--allowedTools) | Deferred | Deferred | Deferred |

**Validation at startup:**
- If orchestrator requires a capability (e.g., resume mode enabled but backend doesn't support resume), fail startup with clear error.
- This prevents silent degradation where the user expects resume but gets fresh sessions.

```gleam
pub fn validate_capabilities(config: Config, caps: Capabilities) -> Result(Nil, Error) {
  case config.run.resume, caps.supports_resume {
    True, False -> Error(CapabilityMismatch("Resume enabled but backend doesn't support it"))
    _, _ -> Ok(Nil)
  }
}
```

Rationale:
- Event schema decouples orchestration from SDK/CLI specific logs.
- Enables Claude/Codex/AMP interchangeably.

## AgentEvent Ingestion Per Backend

Each adapter must normalize backend-specific output to the canonical `AgentEvent` schema. This section defines the concrete ingestion contract for each backend.

### Claude CLI (`claude_cli.gleam`)

**Transport:** stdio (stdin/stdout)
- Launch: `claude --output-format stream-json --verbose ...`
- Communication: JSONL on stdout, prompts via stdin

**Message Framing:** Newline-delimited JSON (one JSON object per line)

**Mapping Rules:**
| Backend Event | AgentEvent Variant |
|---------------|-------------------|
| `{"type": "assistant", "message": {...}}` | `AssistantText` |
| `{"type": "tool_use", "id": "...", "name": "...", "input": {...}}` | `ToolUse(tool_id, tool_name, input)` |
| `{"type": "tool_result", "tool_use_id": "...", ...}` | `ToolResult(tool_id, status, output)` |
| `{"type": "result", "session_id": "...", ...}` | `Final(session_id, summary, resume_token)` |
| `{"type": "system", "subtype": "context_pressure"}` | `ContextPressure` |
| `{"type": "system", "subtype": "checkpoint", "resume": "..."}` | `Checkpoint(resume_token)` |

**Tool Correlation:** `tool_use.id` matches `tool_result.tool_use_id`

**Timestamps:** Derived from local wall clock at parse time (Claude CLI does not emit timestamps)

**Resume Detection:** `supports_resume = true` when `session_id` is present in `Final` events. Adapter verifies resume capability via `claude --help` output for `--resume` flag.

### Codex App (`codex_app.gleam`) - DEFERRED TO v1.1

> **v1 Decision:** Codex adapter is **deferred to v1.1**. The API contract is speculative and unconfirmed.
> v1 ships with Claude CLI only + mock adapter for testing.

**Discovery gate for v1.1:**
Before implementing Codex adapter:
1. Confirm real API transport (HTTP, WebSocket, stdio)
2. Confirm event schema and mapping
3. Confirm capability detection mechanism (if `/capabilities` exists)
4. Confirm authentication model

See **Appendix A: Codex Adapter Spike** for speculative design to be validated.

### AMP CLI (`amp_cli.gleam`)

**Transport:** stdio (stdin/stdout)
- Launch: `amp run --json-stream ...`
- Communication: JSONL on stdout

**Message Framing:** Newline-delimited JSON

**Mapping Rules:**
| Backend Event | AgentEvent Variant |
|---------------|-------------------|
| `{"kind": "text", ...}` | `AssistantText` |
| `{"kind": "tool_call", "id": "...", ...}` | `ToolUse` |
| `{"kind": "tool_output", "call_id": "...", ...}` | `ToolResult` |
| `{"kind": "complete", "run_id": "..."}` | `Final` |

**Tool Correlation:** `tool_call.id` matches `tool_output.call_id`

**Timestamps:** From `ts` field if present, else local wall clock

**Resume Detection:** `supports_resume = false` (AMP does not support session resume in v1)

### OpenCode CLI (`open_code_cli.gleam`)

**Transport:** stdio + file tailing
- Launch: `opencode run --log-file <path> ...`
- Communication: stdout for status, file tailing for events

**Message Framing:** JSONL in log file

**Mapping Rules:** Similar to Claude CLI; exact mapping TBD based on OpenCode output format.

**Resume Detection:** Capability detected via `opencode --version` for `--resume` support.

### Synthesized Events

When backend does not emit certain signals, adapter synthesizes them:
- **ContextPressure:** Synthesized when `usage.total_tokens > threshold` (configurable, default 80% of context window)
- **Final without resolution:** Synthesized when adapter process exits without explicit final event
- **Checkpoint:** Only emitted by Claude CLI; other backends synthesize via session save API or mark as unsupported

## Issue Worker State Machine

States: idle -> running -> gating -> reviewing -> finalizing -> done

### Progress Detection

Progress is a critical guard for retry decisions. An issue has made **progress** if any of the following occurred since the last attempt:

**Primary signal (always checked first):**
1. **New commits with issue tag:** Git log shows commits with message containing `bd-<issue_id>` since last attempt. **(Authoritative, unforgeable)**

**Secondary signals (checked if no commits):**
2. **Validation improvement:** Gate failure count decreased (e.g., 3 lint errors → 1 lint error). Orchestrator compares current vs previous gate run output. **(Objective, orchestrator-verified)**

**Removed (too weak or not orchestrator-verified):**
- ~~Code changes in worktree (git diff)~~ - In shared-repo mode, cannot attribute changes reliably
- ~~Relevant tool success (ToolResult ok)~~ - Agent events are not authoritative
- ~~Explicit progress marker via text patterns~~ - Agent can emit text without actual progress

**Rationale:** Progress detection must be based on orchestrator-observable state (git commits, validation output), not agent-reported events. This prevents agents from gaming retries.

**Working directory for progress checks:**

Progress checks must run in the **issue's effective working directory**, which varies by enforcement mode:

| Enforcement Mode | Workdir for Progress Checks |
|------------------|----------------------------|
| Strong (Claude CLI) | Repo root (shared) |
| Medium (Codex App) | Repo root (shared) |
| Structural (AMP, etc.) | Per-issue worktree |

**Shared-repo mode attribution:**

In shared-repo mode with concurrent workers, `git diff` at repo root cannot attribute changes to a specific issue.

**Progress detection hierarchy (for retry decisions):**

| Signal | Purpose | Used for Retry Decision? |
|--------|---------|-------------------------|
| Commits with `bd-<issue_id>` | Authoritative progress evidence | **Yes** (primary) |
| Validation improvement | Objective improvement metric | **Yes** (secondary, orchestrator-verified) |
| Lock-attributed file changes | Reporting and debugging | **No** (best-effort only) |
| Tool results with issue_id metadata | Audit trail | **No** (agent-reported, not authoritative) |

**Retry decision algorithm:**
```gleam
pub fn should_retry(issue_state: IssueState, gate_result: GateResult) -> Bool {
  // Only commit-based and validation-delta signals are used for retry decisions
  let has_progress = case detect_progress(issue_state) {
    CommitsFound(n) if n > 0 -> True
    ValidationImproved(old_failures, new_failures) if new_failures < old_failures -> True
    _ -> False
  }

  has_progress && issue_state.gate_retries < max_retries
}

pub fn detect_progress(state: IssueState) -> ProgressSignal {
  // Primary: check for commits (unforgeable)
  let commits = git.log(state.workdir, since: state.last_attempt_commit, grep: "bd-" <> state.issue_id)
  case commits {
    [_, ..] -> CommitsFound(list.length(commits))
    [] -> {
      // Secondary: check validation improvement (orchestrator-verified)
      case state.previous_gate_result, state.current_gate_result {
        Some(prev), Some(curr) if count_failures(curr) < count_failures(prev) ->
          ValidationImproved(count_failures(prev), count_failures(curr))
        _ -> NoProgress
      }
    }
  }
}
```

**Lock-attributed diffs (reporting only):**

Lock-attributed file changes are useful for debugging and reporting, but are **not used for retry decisions** because:
1. Lock ownership can be released/reacquired between checks
2. File timestamps don't prove the change was meaningful
3. Agent could touch files without making actual progress

These are logged in the run report for human review but don't affect orchestrator decisions.

**Handling multi-issue commits:**

If a commit message contains multiple issue tags (e.g., `bd-123 bd-456: shared refactor`):
1. Credit progress to **all** tagged issues
2. Each issue independently evaluates retry based on its own gate results
3. This is acceptable because shared commits are rare and the gate result is issue-specific

**`last_attempt_commit` tracking:**
```gleam
pub type IssueState {
  // ...
  last_attempt_commit: Option(String),  // Git SHA at start of attempt
  last_attempt_start: Timestamp,
}

// At attempt start
pub fn record_attempt_start(state: IssueState) -> IssueState {
  let head = git.rev_parse("HEAD", state.workdir)
  IssueState(..state, last_attempt_commit: Some(head), last_attempt_start: now())
}

// Progress check: commits since last attempt
pub fn has_new_commits(state: IssueState) -> Bool {
  case state.last_attempt_commit {
    Some(sha) -> {
      let commits = git.log(state.workdir, since: sha, grep: "bd-" <> state.issue_id)
      !list.is_empty(commits)
    }
    None -> False  // First attempt, no baseline
  }
}
```

**No-progress detection:**
- If session completes (Final event) with none of the above signals, mark as `no_progress`.
- No-progress triggers followup marking instead of retry (prevents infinite loops).

### Resolution Marker Detection

A **resolution marker** indicates the agent believes the issue is complete. **Resolution markers do NOT skip gating** - they only affect how the issue is closed after gate passes.

**Resolution is orchestrator-verified, not agent-reported.**

The orchestrator does NOT trust agent-reported resolution (detecting `bd close` commands in agent events). Instead:

1. **Agent signals intent:** Agent may run `bd close <issue_id>` or emit resolution text.
2. **Orchestrator verifies:** After session ends, orchestrator queries Beads directly via `bd show <issue_id>` to check actual issue status.
3. **Truth source:** Beads CLI is the authoritative source for issue state.

**Resolution verification flow:**
```gleam
pub fn verify_resolution(issue_id: String) -> ResolutionStatus {
  // Query Beads directly - don't trust agent events
  case beads.show(issue_id) {
    Ok(issue) if issue.status == "closed" -> Resolved(issue.close_message)
    Ok(issue) -> NotResolved  // Issue still open
    Error(_) -> Unknown  // Beads query failed, treat as not resolved
  }
}
```

**Gate requirement:** Even if issue is closed in Beads, the gate **still runs** to verify code quality:
- If gate passes → finalize as closed (commit is good)
- If gate fails → mark as followup with "closed but gate failed" (agent claimed done but broke something)

**AgentEvent resolution hints (NOT authoritative):**
- `Final` event may include `resolution_hint: Option(String)` from agent text.
- This is for logging/audit only, NOT used for decision making.

**Removed (agent-reported, not trustworthy):**
- ~~Detecting `bd close` command in ToolUse events~~ - Agent could run the command but it could fail
- ~~Commit message patterns~~ - Agent can write anything in commit message
- ~~Explicit resolution text patterns~~ - Agent can claim resolution without actual fix

**Rationale:** Resolution must be verified via Beads CLI (orchestrator-executed) because agent-reported events can be spoofed or inaccurate.

Transitions:
- idle: Start(issue) -> running (spawn session)
- running: Final event -> gating (if no resolution marker)
- running: Resolution marker -> finalizing
- running: idle timeout -> resume or fail
- gating: pass -> review or finalizing
- gating: fail -> resume (retry) or finalizing
- reviewing: pass -> finalizing
- reviewing: fail -> resume (retry) or finalizing
- finalizing: close/mark followup -> done

### IssueWorker State Transition Table (gen_statem)

| State | Event | Guard | Action | Next |
|---|---|---|---|---|
| idle | Start(issue) | - | spawn session, **persist(started)** | running |
| running | AgentEvent(Final) | has_resolution | finalize resolution | finalizing |
| running | AgentEvent(Final) | no resolution | capture log offset, **persist(gating)** | gating |
| running | AgentEvent(ContextPressure) | supports_checkpoint | request checkpoint + resume | running |
| running | Timeout(Idle) | supports_resume | resume session | running |
| running | Timeout(Idle) | !supports_resume | mark followup | finalizing |
| running | Abort | - | mark followup | finalizing |
| gating | GateResult(Pass) | review_enabled | enqueue review, **persist(reviewing)** | reviewing |
| gating | GateResult(Pass) | !review_enabled | - | finalizing |
| gating | GateResult(Fail) | retries_left && progress | **incr gate_retries, persist(retry)**, resume | running |
| gating | GateResult(Fail) | !retries_left \|\| !progress | mark followup | finalizing |
| reviewing | ReviewResult(Pass) | - | - | finalizing |
| reviewing | ReviewResult(Fail) | retries_left && progress | **incr review_retries, persist(retry)**, resume | running |
| reviewing | ReviewResult(Fail) | !retries_left \|\| !progress | mark followup | finalizing |
| finalizing | Finalized | - | emit events, close issue, **persist(done)** | done |
| done | * | - | ignore | done |

**IssueWorker Persistence Points:**
- `persist(started)`: Write issue_id, attempt=1, state=running, start_time
- `persist(gating)`: Write state=gating, log_offset, evidence collected so far
- `persist(reviewing)`: Write state=reviewing, gate_result
- `persist(retry)`: Write updated attempt counters (gate_retries/review_retries)
- `persist(done)`: Write final state, resolution method, close time

### SessionRunner State Transition Table (gen_statem)

| State | Event | Guard | Action | Next |
|---|---|---|---|---|
| spawn | Start(prompt) | - | adapter.start_session, **persist(spawned)** | streaming |
| streaming | AgentEvent(ToolUse) | - | emit event, track tool | streaming |
| streaming | AgentEvent(ToolResult) | - | emit event, update evidence | streaming |
| streaming | AgentEvent(Usage) | over_threshold | request checkpoint | checkpoint |
| streaming | AgentEvent(AssistantText) | - | emit event | streaming |
| streaming | AgentEvent(Final) | - | emit final, **persist(final)** | done |
| streaming | Timeout(Idle) | supports_resume | resume, **persist(resume)** | streaming |
| streaming | Timeout(Idle) | !supports_resume | fail | done |
| streaming | **Timeout(Session)** | - | **abort session, fail** | **done** |
| checkpoint | AgentEvent(Checkpoint) | - | resume with new prompt, **persist(checkpoint)** | streaming |
| checkpoint | Timeout(Checkpoint) | - | fail | done |
| done | * | - | ignore | done |

**Timeout types:**
- `Timeout(Idle)`: No events received for `idle_timeout_sec` (default: 300s). Triggers resume attempt.
- `Timeout(Session)`: Total session duration exceeds `timeout_sec` (default: 3600s). Hard termination.

**Persistence points (via PersistIntent to run_manager):**

> Workers NEVER write DETS directly. All persistence goes through `run_manager`.

- `send_persist(UpdateIssueState(spawned))`: Session started
- `send_persist(UpdateResumeToken(token))`: New resume token received
- `send_persist(RecordEvent(checkpoint))`: Checkpoint event appended
- `send_persist(RecordEvent(final))`: Final event appended
- `send_persist(Sync)`: Force DETS sync after critical transitions

Note: Global cursor is updated by `run_manager` periodically via `SyncCursor`. See "Global Cursor Model".

## Validation Subsystem

### Validation and Gate Contract

**Authoritative rule: Validation and quality gates are always orchestrator-executed.**

The agent cannot be trusted to run validation commands or provide gate evidence. The orchestrator runs all validation commands directly via `infra/command_runner.gleam` and evaluates pass/fail based on exit codes and output.

- **Agent events are supplemental only:** AgentEvents (tool calls, assistant text) provide context and audit trail, but are NOT used to determine gate pass/fail.
- **No spoofing risk:** Because the orchestrator executes validation commands in the working directory after the agent session ends, the agent cannot fake validation results.
- **Evidence collection:** The orchestrator collects evidence from its own command runs (stdout, stderr, exit code) and git state (commits, diff).

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Gate Flow (v1)                               │
│                                                                      │
│  Agent Session Ends → Orchestrator Runs Validation → Gate Decision  │
│         │                      │                          │         │
│         ▼                      ▼                          ▼         │
│  [AgentEvents]           [Command Results]          [Pass/Fail]     │
│  (audit trail)           (authoritative)            (orchestrator)  │
└─────────────────────────────────────────────────────────────────────┘
```

### Validation Components

- ValidationSpec built from config + presets
- Commands executed via infra/command_runner (erlexec)
- Evidence derived from orchestrator command results (NOT agent events)
- Coverage and e2e optional, scope-aware

### Worktree Strategy (v1.1 - Deferred)

> **v1 scope:** No worktrees. All validation runs in the shared repo working directory.
>
> **v1.1+ (deferred):** Per-epic worktrees with shared-repo mode within each epic. Multiple agents work on one epic's worktree, then changes are merged to main via a separate system.

**v1: Shared-repo validation**

In v1, validation commands run directly in the repository root. No worktrees are created or managed. This simplifies the execution model:
- Agents commit to the working branch
- Orchestrator runs validation in the same directory after agent session ends
- No isolation between concurrent issues (lock enforcement prevents conflicts)

## Command Execution Security Policy

This section distinguishes between two execution contexts with different security guarantees.

### (A) Orchestrator-Executed Commands (Full Control)

Commands executed by mala directly via `infra/command_runner.gleam`:
- **Validation commands:** lint, test, typecheck, etc.
- **Git operations:** worktree create/remove, rebase, merge
- **Internal utilities:** bd CLI, config migration

These commands have **full security guarantees**: argv-only, path confinement, env filtering, output limits.

### (B) Agent-Executed Commands (Observed Only)

Commands executed by agent backends (Claude, Codex, etc.):
- **Tool invocations:** Edit, Write, Bash, etc.
- **Agent-chosen commands:** Whatever the agent decides to run

For these commands, mala can only **observe and conditionally block** (if hooks supported):
- Outputs are treated as **untrusted** (may contain secrets, malicious content)
- Security depends on backend capabilities (hooks, tool restrictions)
- If backend lacks hooks: **per-issue worktree isolation required**

### Security Guarantees by Context

| Property | Orchestrator Commands | Agent Commands (with hooks) | Agent Commands (no hooks) |
|----------|----------------------|----------------------------|---------------------------|
| Argv-only execution | ✓ Guaranteed | ✗ Agent decides | ✗ Agent decides |
| Path confinement | ✓ Guaranteed | ✗ Best-effort | ✗ None (worktree isolates) |
| Env filtering | ✓ Guaranteed | ✗ Not applicable | ✗ Not applicable |
| Output capture limits | ✓ Guaranteed | ✓ Event stream | ✓ Event stream |
| Lock enforcement | N/A | ✓ Hook validates | ✗ None (worktree isolates) |

### Orchestrator Command Execution Details

### Timeout and Kill Behavior

- **Soft timeout:** Configurable per-command (default: 5 minutes for validation, 60 minutes for agent sessions).
- **Kill strategy:** SIGTERM first, wait 5 seconds, then SIGKILL.
- **Process group:** All spawned commands run in their own process group. Kill targets the entire group to prevent orphans.
- **Implementation:** Use `erlexec` with `{kill_timeout, 5000}` and `{group, true}` options.

```gleam
pub fn run_command(cmd: String, opts: CommandOpts) -> Result(Output, Error) {
  exec.run(cmd, [
    exec.timeout(opts.timeout_ms),
    exec.kill_timeout(5000),
    exec.group(True),
    exec.stdout(Capture),
    exec.stderr(Capture),
    exec.cd(opts.workdir),
    exec.env(filter_env(opts.env)),
  ])
}
```

### Output Capture Limits

- **stdout/stderr limit:** 10 MB per stream (configurable via `command.output_limit_bytes`).
- **Truncation strategy:** Keep first 5 MB and last 5 MB with "[truncated]" marker.
- **Persistence:** Full output stored in run logs; truncated output in events/evidence.

### Environment Variable Handling

- **Inheritance:** Commands inherit a filtered subset of the parent environment.
- **Allowlist:** Only explicitly allowed variables are passed through:
  - `PATH`, `HOME`, `USER`, `SHELL`, `TERM`, `LANG`, `LC_*`
  - `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (for agent backends)
  - Variables explicitly listed in config `agents.<profile>.env`
- **Denylist:** Never pass through: `AWS_*`, `GCP_*`, `AZURE_*`, `DATABASE_*`, `*_PASSWORD`, `*_SECRET`, `*_TOKEN` (except explicitly allowed).
- **Redaction in logs:** Sensitive values are redacted in event logs using pattern matching.

```gleam
const ENV_ALLOWLIST = ["PATH", "HOME", "USER", "SHELL", "TERM", "LANG"]
const ENV_DENYLIST_PATTERNS = ["AWS_", "GCP_", "AZURE_", "DATABASE_", "_PASSWORD", "_SECRET"]

pub fn filter_env(overrides: Map(String, String)) -> List(#(String, String)) {
  let base = os.env()
    |> map.filter(fn(k, _) { is_allowed(k) && !is_denied(k) })
  map.merge(base, overrides)
  |> map.to_list()
}
```

### Path Confinement

- **Working directory:** All commands execute with `cwd` set to either:
  - The repository root (for agent sessions)
  - A specific worktree path (for global validation)
- **Path validation:** Before execution, validate that `cwd` is within allowed directories:
  - Repository root or its subdirectories
  - Temporary worktree paths under `paths.worktree_dir`
- **No shell expansion:** Commands are passed as argument vectors, never shell-joined strings.

```gleam
pub fn validate_workdir(path: String, config: Config) -> Result(Nil, Error) {
  let allowed = [config.repo_root, config.paths.worktree_dir]
  case list.any(allowed, fn(a) { string.starts_with(path, a) }) {
    True -> Ok(Nil)
    False -> Error(PathConfinementError(path))
  }
}
```

### Command Representation (Argv-Only Design)

Commands are represented **exclusively** as argv vectors (`Command { exe, args }`). No string commands are accepted at runtime. This eliminates shell injection risks and platform-specific quoting issues.

**Config format (array only):**
```toml
[validation.commands]
# Commands MUST be arrays of strings (argv)
test = ["uv", "run", "pytest"]
lint = ["uvx", "ruff", "check", "."]
format = ["uvx", "ruff", "format", "."]
typecheck = ["uvx", "ty", "check"]
e2e = ["uv", "run", "pytest", "-m", "e2e"]
setup = ["uv", "sync"]
```

**No string format support:** String commands are rejected at config load time with a clear error message directing users to convert to array format.

```gleam
pub type Command {
  Command(
    exe: String,           // Executable path or name
    args: List(String),    // Arguments as separate elements
  )
}

pub fn parse_command(key: String, value: toml.Value) -> Result(Command, ConfigError) {
  case value {
    toml.Array(items) -> {
      case list.map(items, toml.as_string) |> result.all() {
        Ok([exe, ..args]) -> Ok(Command(exe, args))
        Ok([]) -> Error(EmptyCommand(key))
        Error(_) -> Error(NonStringInCommandArray(key))
      }
    }
    toml.String(_) -> {
      Error(StringCommandNotSupported(key,
        "Use array format: [\"exe\", \"arg1\", \"arg2\"]. " <>
        "Run 'mala config convert-commands' to migrate."))
    }
    _ -> Error(InvalidCommandType(key))
  }
}
```

**Command runner API (single canonical path):**
```gleam
pub fn run_command(cmd: Command, opts: CommandOpts) -> Result(Output, Error) {
  // exe and args are passed directly to erlexec, never shell-joined
  exec.run(cmd.exe, cmd.args, [
    exec.timeout(opts.timeout_ms),
    exec.kill_timeout(5000),
    exec.group(True),
    exec.stdout(Capture),
    exec.stderr(Capture),
    exec.cd(opts.workdir),
    exec.env(build_env(opts.base_env, opts.command_env)),
  ])
}
```

**Dynamic arguments:**
- Any dynamic values (file paths, issue IDs) are appended to `args` as separate elements.
- Never string-interpolated: `Command("pytest", ["--file", path])` not `Command("pytest", ["--file=" <> path])`.

**No migration tool:** This is a clean break. Users manually convert their configs to TOML with array commands.

## Locking + Deadlock

- lock_server: ETS-backed lock table.
- lock keys are canonicalized file paths + repo namespace.
- mcp_lock_server: MCP JSON-RPC tools for agents.
- deadlock_monitor: digraph cycle detection on lock waits.

### Lock Key Canonicalization

Lock keys are canonicalized file paths to prevent two different path strings from referring to the same file.

**Canonicalization rules:**

1. **Resolve to absolute path:** Convert relative paths to absolute using repo root as base
2. **Resolve symlinks:** Use `realpath` equivalent to get canonical path (no symlinks)
3. **Normalize separators:** Use `/` on all platforms
4. **Resolve `.` and `..`:** Eliminate path traversal components
5. **Reject paths outside repo:** Paths resolving outside repo root are rejected

```gleam
pub fn canonicalize_lock_key(path: String, repo_root: String) -> Result(String, Error) {
  // 1. Make absolute
  let abs_path = case string.starts_with(path, "/") {
    True -> path
    False -> repo_root <> "/" <> path
  }

  // 2. Resolve symlinks and normalize (realpath equivalent)
  use canonical <- result.try(file.realpath(abs_path))

  // 3. Verify within repo root
  case string.starts_with(canonical, repo_root) {
    True -> Ok(canonical)
    False -> Error(PathOutsideRepo(path, canonical))
  }
}
```

**Symlink handling:**

| Scenario | Behavior | Rationale |
|----------|----------|-----------|
| Lock `link.txt` (symlink to `real.txt`) | Lock key is `/repo/real.txt` | Prevents bypass via symlink |
| Lock `../outside.txt` | **Rejected** | Path resolves outside repo |
| Lock `./subdir/../file.txt` | Lock key is `/repo/file.txt` | Normalized |

**Case sensitivity:**

| Platform | Behavior |
|----------|----------|
| Linux (ext4) | Case-sensitive - `File.txt` ≠ `file.txt` |
| macOS (APFS default) | Case-insensitive - `File.txt` = `file.txt` |

On case-insensitive filesystems, canonicalization preserves the case from `realpath` output, ensuring consistent lock keys regardless of how the path was requested.

**Lock key format:**

```
<repo_root>/<canonical_relative_path>
```

Example: `/home/user/myrepo/src/main.py`

### MCP Lock Server IPC Architecture

External agent processes (e.g., `claude` CLI) run in separate OS processes and cannot directly access the main `mala` ETS tables. The MCP lock server uses an IPC mechanism to bridge this gap.

**Architecture:**
```
┌─────────────────┐     stdio/JSON-RPC     ┌─────────────────┐
│  claude CLI     │ ◄──────────────────────► │  mala-mcp-proxy │
│  (agent)        │                          │  (subprocess)   │
└─────────────────┘                          └────────┬────────┘
                                                      │ Unix socket
                                                      ▼
                                             ┌─────────────────┐
                                             │  mala           │
                                             │  (orchestrator) │
                                             │  - lock_server  │
                                             │  - ETS tables   │
                                             └─────────────────┘
```

**IPC Mechanism:** Unix domain socket

1. **Socket server:** When `mala run` starts, it creates a Unix socket at `/tmp/mala-<run_id>.sock`.
2. **Proxy command:** `mala internal-mcp-proxy --socket /tmp/mala-<run_id>.sock` is spawned by agents as their MCP tool server.
3. **Protocol:** JSON-RPC 2.0 over the socket, same as MCP stdio protocol.
4. **Discovery:** Agents are configured with `--mcp-server "mala internal-mcp-proxy --socket /tmp/mala-<run_id>.sock"` via orchestrator.

**Proxy implementation:**
```gleam
// mala internal-mcp-proxy
pub fn main(args: List(String)) {
  let socket_path = parse_socket_arg(args)
  let conn = unix_socket.connect(socket_path)

  // Bridge stdio JSON-RPC to socket
  loop {
    case io.read_line(stdin) {
      Ok(line) -> {
        // Forward request to orchestrator
        unix_socket.send(conn, line)
        // Read response
        let response = unix_socket.recv(conn)
        io.write(stdout, response)
      }
      Error(Eof) -> break
    }
  }
}
```

**Lock server socket handler (in orchestrator):**
```gleam
pub fn handle_socket_request(request: JsonRpc, state: LockState) -> JsonRpc {
  case request.method {
    "tools/call" -> {
      case request.params.name {
        "acquire_lock" -> handle_acquire(request.params, state)
        "release_lock" -> handle_release(request.params, state)
        "check_lock" -> handle_check(request.params, state)
        _ -> error_response("Unknown tool")
      }
    }
    "tools/list" -> list_lock_tools()
    _ -> error_response("Unknown method")
  }
}
```

**Security considerations:**
- Socket file permissions set to `0600` (owner only)
- Socket path includes run_id to prevent cross-run interference
- Proxy validates JSON-RPC format before forwarding

**run_id generation:**

run_id is a unique identifier for each mala run, used for:
- Persistence directory naming (`~/.config/mala/runs/<run_id>/`)
- Socket path derivation
- Log correlation

```gleam
pub fn generate_run_id() -> String {
  // Format: <timestamp>-<random>
  // Example: "20260110-143052-a7b3c9d2"
  let timestamp = time.now() |> time.format("YYYYMMDD-HHmmss")
  let random = crypto.random_bytes(4) |> base16.encode() |> string.lowercase()
  timestamp <> "-" <> random
}
```

**Properties:**
- **Unique:** Timestamp + random ensures uniqueness even with concurrent starts
- **Sortable:** Timestamp prefix allows chronological ordering
- **Short:** Fits in path constraints when hashed for socket

**Cross-platform constraints:**

| Platform | Socket Support | Path Constraints | Notes |
|----------|---------------|------------------|-------|
| Linux | Full | Max 108 chars | Use `/tmp/mala-<short_id>.sock` |
| macOS | Full | Max 104 chars | Same as Linux |
| Windows | Not supported | N/A | Use WSL2 (Linux behavior) |

**Socket path and cleanup:**
```gleam
pub fn socket_path(run_id: String) -> String {
  // Use short hash to stay within path limits
  let short_id = crypto.sha256(run_id) |> string.slice(0, 8)
  "/tmp/mala-" <> short_id <> ".sock"
}

pub fn ensure_socket_available(path: String, run_id: String) -> Result(Nil, Error) {
  case file.exists(path) {
    False -> Ok(Nil)  // Path is free
    True -> {
      // Socket exists - check if it's ours or orphaned
      case try_connect(path) {
        Ok(conn) -> {
          // Someone is listening - check run_id via handshake
          case handshake(conn, run_id) {
            Ok(Same) -> Ok(Nil)  // It's us (restart), reuse
            Ok(Different) -> Error(SocketInUse(path))  // Another run
            Error(_) -> {
              // Handshake failed - assume orphaned, remove
              let _ = file.delete(path)
              Ok(Nil)
            }
          }
        }
        Error(_) -> {
          // Can't connect - orphaned socket, safe to remove
          let _ = file.delete(path)
          Ok(Nil)
        }
      }
    }
  }
}
```

**Handshake protocol:**
- On connect, client sends `{"type": "handshake", "run_id": "<id>"}`
- Server responds `{"type": "handshake_ack", "run_id": "<server_id>"}`
- If run_ids match, connection proceeds; otherwise, connection closed

### IPC Protocol Specification

This section defines the wire protocol for Unix socket communication between mala components.

**Message framing:** Newline-delimited JSON (NDJSON)

Each message is a single JSON object followed by a newline (`\n`). This matches the MCP stdio protocol and is simple to implement.

```
<json-object>\n<json-object>\n...
```

**Why NDJSON (not length-prefixed):**
- Compatible with MCP stdio protocol (no translation needed)
- Easy to debug with standard tools (`nc`, `socat`)
- Claude CLI hook expects stdout/stdin as plain text with newlines

**Request/response format:** JSON-RPC 2.0

```json
// Request
{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "acquire_lock", "arguments": {"path": "file.txt"}}, "id": 1}

// Response (success)
{"jsonrpc": "2.0", "result": {"acquired": true}, "id": 1}

// Response (error)
{"jsonrpc": "2.0", "error": {"code": -32600, "message": "Lock conflict"}, "id": 1}
```

**Timeout configuration:**

| Operation | Timeout | On Timeout |
|-----------|---------|------------|
| Hook validation | 250ms | Deny (fail closed) |
| MCP tool call (acquire_lock) | 2s | Error response |
| MCP tool call (release_lock) | 1s | Error response |
| Socket connect | 500ms | Error |
| Handshake | 1s | Close connection |

**Hook fallback behavior (socket failure):**

If the hook cannot connect to the orchestrator socket:

| Failure Mode | Single-Agent | Concurrent Mode |
|--------------|--------------|-----------------|
| Socket not found | Allow with warning | **Deny** |
| Connection timeout | Allow with warning | **Deny** |
| Handshake failed | Allow with warning | **Deny** |
| Server error | Allow with warning | **Deny** |

Rationale: In concurrent mode, lock enforcement is critical for safety. Failing open would allow agents to write without locks, causing conflicts.

```gleam
pub fn hook_validate_with_fallback(
  tool_info: ToolInfo,
  socket_path: String,
  config: Config,
) -> HookResult {
  case connect_and_validate(socket_path, tool_info) {
    Ok(result) -> result
    Error(err) -> {
      // Fallback depends on concurrency mode
      case config.run.max_agents {
        Some(n) if n > 1 -> {
          log.error("Hook validation failed in concurrent mode: " <> err)
          Deny("Lock server unavailable, blocking in concurrent mode")
        }
        _ -> {
          log.warn("Hook validation failed, allowing in single-agent mode: " <> err)
          AllowWithWarning("Lock server unavailable")
        }
      }
    }
  }
}
```

**Concurrency model:**

The socket server handles multiple simultaneous connections (one per agent session). Each connection is handled in a separate Erlang process.

**Single in-flight request per connection:**

The v1 protocol enforces **one in-flight request per connection**. The handler blocks on response before reading the next request line. This simplifies implementation and matches typical MCP tool call patterns.

```gleam
pub fn socket_acceptor(listen_socket: Socket, state: ServerState) {
  case socket.accept(listen_socket, timeout: infinity) {
    Ok(client) -> {
      // Spawn handler process for this connection
      spawn(fn() { handle_connection(client, state) })
      // Continue accepting
      socket_acceptor(listen_socket, state)
    }
    Error(err) -> log.error("Accept failed: " <> err)
  }
}

fn handle_connection(client: Socket, state: ServerState) {
  // One request at a time - blocks until response sent
  case read_line(client, timeout: 5000) {
    Ok(line) -> {
      let response = handle_request(json.decode(line), state)
      socket.send(client, json.encode(response) <> "\n")
      // Only after response sent do we read next request
      handle_connection(client, state)
    }
    Error(Eof) -> socket.close(client)
    Error(Timeout) -> socket.close(client)
  }
}
```

**JSON-RPC notifications:** Not supported in v1. All messages must have an `id` field and expect a response. Notifications (messages without `id`) are ignored with a warning.

**Protocol versioning:**

The handshake includes a protocol version for future compatibility:

```json
{"type": "handshake", "run_id": "xxx", "protocol_version": 1}
{"type": "handshake_ack", "run_id": "xxx", "protocol_version": 1, "supported_versions": [1]}
```

Version negotiation:
- Client sends its preferred version
- Server responds with its version and supported range
- If no overlap, connection closed with error
- v1: Initial protocol (this design)

### Deadlock Detection Algorithm

**Trigger:** Detection runs on every `acquire_lock` request that would block (lock held by another issue).

**Wait-for graph construction:**
- Nodes: Issue IDs
- Edges: Issue A → Issue B means "A is waiting for a lock held by B"
- Graph stored in `digraph` (OTP module)

**Algorithm:**
1. On blocking `acquire_lock(issue_id, lock_key)`:
   - Find current holder: `holder = lock_table[lock_key].owner`
   - Add edge: `issue_id → holder`
   - Run cycle detection from `issue_id`
2. Cycle detection: DFS from the waiting issue
   - If path returns to starting node, deadlock detected
   - Complexity: O(V + E) where V = active issues, E = waiting edges

**Action on deadlock detection:**
1. **Victim selection:** Choose the issue with:
   - Fewer completed tool invocations (less work to lose)
   - Higher issue ID (deterministic tie-breaker)
2. **Abort victim:** Send `Abort("deadlock detected")` message to victim worker
3. **Release victim's locks:** All locks held by victim are released
4. **Log event:** Emit `DeadlockDetected` event with cycle path for debugging

```gleam
pub fn detect_cycle(graph: Digraph, start: IssueId) -> Option(List(IssueId)) {
  dfs(graph, start, [start], set.new())
}

fn dfs(graph: Digraph, current: IssueId, path: List(IssueId), visited: Set(IssueId)) -> Option(List(IssueId)) {
  case digraph.out_neighbours(graph, current) {
    [] -> None
    neighbors -> {
      list.find_map(neighbors, fn(next) {
        case next == list.first(path) |> result.unwrap("") {
          True -> Some(path)  // Cycle found
          False -> case set.contains(visited, next) {
            True -> None  // Already visited, no cycle this way
            False -> dfs(graph, next, [next, ..path], set.insert(visited, next))
          }
        }
      })
    }
  }
}
```

**Periodic sweep (optional):**
- In addition to on-demand detection, `deadlock_monitor` can run periodic sweeps every `deadlock_interval_sec` to catch missed cycles.
- Useful if lock requests are processed asynchronously.

### Lock Enforcement Mechanism

Lock enforcement varies by backend capability. **Shared-repo mode is only allowed when hooks are supported.**

#### Shared-Repo Mode Eligibility Rule (v1)

**Shared-repo mode requires:** `supports_hooks == True`

In v1, if hooks are unavailable and `max_agents > 1`, **startup fails** (no worktree fallback).

```gleam
pub fn get_enforcement_mode(caps: Capabilities, max_agents: Int) -> Result(EnforcementMode, Error) {
  case caps.supports_hooks, max_agents {
    True, _ -> Ok(SharedRepoWithLocks)  // Lock enforcement via hooks
    False, 1 -> Ok(SharedRepoNoLocks)   // Single-agent, no enforcement needed
    False, n if n > 1 -> Error(ConcurrentModeUnavailable)  // v1 has no worktree fallback
  }
}
```

> **v1.1 note:** When worktree support is added, this function will return `Ok(PerIssueWorktree)` for the `False, n > 1` case instead of failing.

#### Enforcement Tiers by Backend

| Backend | Enforcement Mode | Reason |
|---------|-----------------|--------|
| Claude CLI | **Shared-repo** | Has hooks + allowedTools |
| Codex App | **Worktree** | No hooks (speculative API, cannot confirm enforcement) |
| AMP CLI | **Worktree** | No hooks |
| OpenCode CLI | **Worktree** | No hooks |

#### Shared-Repo Enforcement (Claude CLI only in v1)

For backends with hook support:
1. **PreToolUse hook:** Validates lock ownership before every file-modifying tool. Blocks Edit, Write, MultiEdit, and Bash commands matching write patterns.
2. **MCP lock server:** Exposes `acquire_lock` and `release_lock` tools. Agent must acquire lock before writing.
3. **--allowedTools:** Restricts available tools to those that route through lock validation.

#### Hook Pattern Matching: Best-Effort Security Model

**Hook-based command pattern matching is best-effort, not bulletproof.**

Pattern matching on Bash commands can be bypassed via:
- `python -c "open('file.txt', 'w').write('data')"`
- Heredocs: `cat << EOF > file.txt`
- `tee file.txt`, `perl -e ...`, `ruby -e ...`
- Encoded payloads, subshell redirections, symlink tricks

**Why this is acceptable for v1:**

1. **Gate is orchestrator-executed:** The agent cannot spoof validation results. Even if it bypasses hooks to write arbitrary files, the orchestrator runs lint/test/typecheck and will catch broken code.

2. **Audit trail:** All agent actions are logged via AgentEvents. Malicious bypass attempts are visible in the event log and can be flagged.

3. **Cooperative model:** Agents (Claude CLI) are generally cooperative. Hook enforcement is a guardrail, not a sandbox. The goal is preventing accidental conflicts, not defending against adversarial agents.

4. **Fallback available:** If hook bypasses become a problem, per-epic worktrees (v1.1) provide structural isolation.

**Hook denial behavior:**

When a hook denies a tool invocation:
1. Hook exits with non-zero code and prints reason to stdout
2. Claude CLI receives denial and includes it in the next assistant turn
3. Agent sees: "Tool use was denied: [reason]"
4. Agent can retry with proper lock acquisition or choose different approach

```gleam
// Hook denial response to Claude CLI
fn deny_tool(reason: String) {
  io.println(reason)  // Shown to agent
  os.exit(1)
}

// Example denial reasons:
// "Lock required: file.txt is not locked by you. Use acquire_lock first."
// "Lock conflict: file.txt is locked by agent-2. Wait or work on different files."
// "Protected file: .env files cannot be modified."
```

### Concurrent Mode Lock Enforcement (v1 Invariants)

**v1 concurrent mode safety invariant:** When `max_agents > 1`, file write interference between issues is prevented by mandatory lock enforcement.

**Supported write tools (Claude CLI v1):**

| Tool | Lock Required | Hook Validates |
|------|---------------|----------------|
| `Edit` | Yes | Yes |
| `Write` | Yes | Yes |
| `MultiEdit` | Yes | Yes |
| `Bash` | See below | Yes (pattern-based) |

**Bash in concurrent mode:**

When `max_agents > 1`:
- **Deny-by-default** for unrecognized commands (as specified in "Bash handling")
- Allowlisted read-safe commands: `ls`, `pwd`, `cat`, `head`, `tail`, `grep`, `find`, `git status`, `git log`, `git diff`, `git show`
- Write commands must match known patterns and have locks for target files
- Unknown commands are **blocked**, not allowed with warning

**Session abort policy (repeated lock failures):**

| Consecutive Denials | Action |
|---------------------|--------|
| 1-3 | Agent sees denial, can retry with lock |
| 4-5 | Warning logged, session continues |
| 6+ | **Session aborted**, issue marked as followup with "repeated lock failures" |

```gleam
pub type LockDenialTracker {
  consecutive_denials: Int,
  max_before_abort: Int,  // Default: 6
}

pub fn on_hook_denial(tracker: LockDenialTracker, session: Session) -> Action {
  let tracker = LockDenialTracker(..tracker, consecutive_denials: tracker.consecutive_denials + 1)
  case tracker.consecutive_denials {
    n if n >= tracker.max_before_abort -> {
      log.error("Session aborted: " <> int.to_string(n) <> " consecutive lock denials")
      AbortSession("Repeated lock failures - agent not cooperating with lock protocol")
    }
    n if n >= 4 -> {
      log.warn("Warning: " <> int.to_string(n) <> " consecutive lock denials")
      Continue(tracker)
    }
    _ -> Continue(tracker)
  }
}

pub fn on_successful_tool(tracker: LockDenialTracker) -> LockDenialTracker {
  // Reset consecutive count on successful tool use
  LockDenialTracker(..tracker, consecutive_denials: 0)
}
```

**Agent guidance on denial:**

Hook denial messages include actionable guidance:
```
Lock required: src/main.py is not locked.
To proceed: 1) Call acquire_lock tool for src/main.py, 2) Retry your edit.
If another agent holds the lock, work on different files or wait.
```

**What does NOT require locks (even in concurrent mode):**
- Read-only tools: `Read`, `Glob`, `Grep`, `WebFetch`
- Orchestration tools: `Task`, `TodoWrite`
- Read-only Bash commands from allowlist

#### Mid-Session Capability Change

If a backend unexpectedly exposes a write-capable tool mid-session (e.g., new tool added):
1. Hook catches unknown tool → deny with error
2. Session marked as failed
3. Issue marked for followup (no automatic worktree fallback in v1)

This prevents silent capability degradation.

#### Structural Enforcement (v1.1 - Deferred)

> **v1.1+ (deferred):** Per-epic worktree isolation for non-hook backends.

For backends without hook support (Codex, AMP, OpenCode):
1. **Per-epic worktree isolation:** Each epic runs in its own git worktree, with shared-repo mode within.
2. **Merge-time coordination:** Changes are merged to main via a separate system (out of scope for v1).
3. **v1 behavior:** Non-hook backends are not supported in v1. Only Claude CLI (with hooks) is supported.

#### Git Merge Policy (v1.1 - Deferred)

> **v1.1+ (deferred):** Worktree merge policy for per-epic isolation.

In v1, all work happens in the main repo (shared-repo mode). No worktree merging is needed. Agents commit directly to the working branch.

#### v1 Tool Classification (Claude CLI)

| Tool | Category | Lock Required | Notes |
|------|----------|---------------|-------|
| `Read` | Read-only | No | File content reading |
| `Glob` | Read-only | No | File listing |
| `Grep` | Read-only | No | Content search |
| `Edit` | Write | **Yes** | File modification |
| `Write` | Write | **Yes** | File creation/overwrite |
| `MultiEdit` | Write | **Yes** | Multiple file edits |
| `Bash` | Mixed | **Best-effort** | See below |
| `Task` | Orchestration | No | Spawns subagents |
| `WebFetch` | Read-only | No | HTTP fetch |
| `TodoWrite` | Metadata | No | Internal state |

**Bash handling (concurrency-aware):**

When `max_agents == 1` (single-agent mode):
- Hook inspects command for write patterns (see "What Constitutes a File Write")
- Pattern match success → require lock for target files
- Pattern match failure (unknown command) → **allow with warning** (cooperative model)
- Primary protection: orchestrator-executed gate catches broken code

When `max_agents > 1` (concurrent mode):
- **Deny-by-default** for unrecognized Bash commands to prevent cross-issue interference
- See precise policy table below

**v1 Bash Command Policy Table (concurrent mode):**

| Category | Commands | Policy | Lock Required |
|----------|----------|--------|---------------|
| **Always allowed** | `ls`, `pwd`, `cat`, `head`, `tail`, `wc`, `which`, `env`, `echo` (no redirect) | Allow | No |
| **Git read** | `git status`, `git log`, `git diff`, `git show`, `git branch` | Allow | No |
| **Git write** | `git add`, `git commit`, `git checkout`, `git reset`, `git stash` | Allow | No (agent's responsibility) |
| **File write** | `cp`, `mv`, `rm`, `touch`, `mkdir`, `rmdir` | **Lock required** | Yes (target path) |
| **Redirect write** | Any command with `>`, `>>`, `2>` | **Lock required** | Yes (target path) |
| **Build tools** | `make`, `npm`, `pip`, `cargo`, `uv` | **Deny** | N/A |
| **Formatters** | `black`, `ruff`, `prettier`, `gofmt` | **Deny** | N/A |
| **Symlink** | `ln -s` | **Deny** | N/A |
| **Unrecognized** | Everything else | **Deny** | N/A |

**Target extraction for lock validation:**

| Pattern | Target Path |
|---------|-------------|
| `cp <src> <dst>` | `<dst>` |
| `mv <src> <dst>` | Both `<src>` and `<dst>` |
| `rm <path>` | `<path>` |
| `cmd > <path>` | `<path>` |
| `cmd >> <path>` | `<path>` |

Unrecognized commands → **deny with error message**: "Unknown command blocked in concurrent mode. Use Edit/Write tools or request specific Bash command approval."

**v1 enforcement policy:**
- Write tools (Edit, Write, MultiEdit): always require lock
- Bash (single-agent): best-effort pattern matching, allow unknown commands
- Bash (concurrent): deny-by-default, explicit allowlist of read-safe commands
- All other tools: no lock required

**Concurrency mode detection:**
```gleam
pub fn get_bash_policy(config: Config) -> BashPolicy {
  case config.run.max_agents {
    Some(1) | None -> AllowUnknownWithWarning
    Some(n) if n > 1 -> DenyUnknownCommands
    _ -> DenyUnknownCommands  // Default to safe mode
  }
}
```

#### What Constitutes a "File Write"

For Bash command pattern matching:
- Direct file operations: `>`, `>>`, `rm`, `mv`, `cp`, `mkdir`, `touch`
- In-place edits: `sed -i`, `perl -i`, `awk -i inplace`
- Permission changes: `chmod`, `chown`
- Auto-fix tools: `ruff format`, `prettier --write`, `black`
- **Excluded:** Read operations, coverage reports in temp dirs, build artifacts in `.build/`

**Known bypass vectors (accepted risk):**
- `python -c "..."`, `ruby -e "..."`, `perl -e "..."` with file writes
- Heredocs, `tee`, encoded payloads
- Symlink tricks, subshell redirections

These are accepted because gate is orchestrator-executed and will catch broken code.

#### Stale Lock Handling

- Locks have a TTL (default: 30 minutes, configurable via `locks.ttl_minutes`).
- `lock_server` runs a periodic sweep (every `locks.sweep_interval_sec`, default: 60) to expire stale locks.
- On worker crash, supervisor restart triggers explicit lock release for that worker's locks.
- On app crash, recovery flow (see Persistence section) releases all locks for the run before restarting.

```gleam
pub type Lock {
  Lock(
    key: String,
    owner: IssueId,
    acquired_at: Timestamp,
    ttl_ms: Int,
  )
}

pub fn is_expired(lock: Lock, now: Timestamp) -> Bool {
  now - lock.acquired_at > lock.ttl_ms
}
```

#### Enforcement Limitations (v1)

- **No OS-level enforcement:** Filesystem permissions, sandboxing, FUSE not implemented.
- **Bash bypass risk:** For medium-enforcement backends, Bash tool can write without lock. Mitigated by: (1) limiting Bash availability in agent config, (2) per-issue worktrees as fallback.
- **Future hardening:** Consider overlayfs or container-based sandboxing for untrusted backends.

## Logging + Telemetry

- disk_log for append-only run events.
- telemetry + opentelemetry for spans.
- event_bus uses pg/gproc for subscribers.

## Presentation Layer (CLI Output)

The CLI presentation layer subscribes to `event_bus` and renders real-time feedback to the terminal.

### Output Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| Interactive | TTY detected | Spinners, progress bars, live updates |
| Non-interactive | Pipe/redirect | Structured log lines, no ANSI codes |
| JSON | `--output json` | Newline-delimited JSON events |

### View Components

**Run overview (interactive):**
```
mala run
┌─ Issues ────────────────────────────────────────┐
│ [✓] bd-123: Fix login bug          (00:45)      │
│ [⟳] bd-124: Add dark mode          (02:15)      │
│ [⋯] bd-125: Update docs            (queued)     │
└─────────────────────────────────────────────────┘
  Running: 2/4 │ Completed: 1 │ Failed: 0
```

**Issue detail (on selection or verbose):**
- Current state (running/gating/reviewing)
- Recent tool invocations
- Validation status
- Error messages if failed

### Event Subscription

```gleam
// Presentation layer subscribes to event_bus
pub fn start_presenter(mode: OutputMode) -> Pid {
  spawn(fn() {
    event_bus.subscribe([
      IssueStarted, IssueCompleted, IssueFailed,
      GateStarted, GateCompleted,
      ReviewStarted, ReviewCompleted,
      RunCompleted,
    ])
    presenter_loop(mode, initial_state())
  })
}
```

### Interactive Permission Handling

When `max_agents > 1` and `permission_mode != "bypassPermissions"`, permission prompts from concurrent agents could conflict on shared stdio.

**Solution: Permission prompts are serialized through orchestrator.**

1. Agent backends do **not** directly prompt the user.
2. Permission requests are sent as `PermissionRequired` events to `run_manager`.
3. `run_manager` queues permission requests and presents them one at a time.
4. User response is routed back to the requesting agent.

**Implementation:**
```gleam
pub type AgentEvent {
  // ... other variants
  PermissionRequired(
    session_id: String,
    tool_name: String,
    description: String,
    options: List(String),  // e.g., ["Allow", "Deny", "Allow all"]
  )
}

// In run_manager
pub fn handle_permission_request(req: PermissionRequired, state: RunState) -> RunState {
  case state.pending_permission {
    Some(_) -> {
      // Queue this request
      RunState(..state, permission_queue: queue.push(state.permission_queue, req))
    }
    None -> {
      // Present to user
      presenter.show_permission_prompt(req)
      RunState(..state, pending_permission: Some(req))
    }
  }
}
```

**Limitation:** Not all backends support deferred permission handling. For backends that prompt directly (some CLI tools), concurrent runs with `bypassPermissions = false` are not supported. Startup validation fails if this combination is detected.

## Persistence and Recovery

This section defines what state is persisted, where it lives, and how recovery works after crashes or restarts.

### Persisted State (Storage Locations)

**Principle: Keep DETS lean.** Store only metadata and cursors in DETS. Store bulk data (events, outputs) in disk_log or files.

| State | Storage | Contents |
|-------|---------|----------|
| Run metadata | DETS (`run.dets`) | Run config, start time, status, issue list (IDs only), **global_cursor** |
| Issue state | DETS (`issues.dets`) | Per-issue: state enum, attempt count, session_id, resume_token, gate_retries, review_retries |
| Event log | disk_log (`events.log`) | Full AgentEvents with timestamps and issue_id (append-only) |
| Evidence | Files (`evidence/<issue_id>/`) | Validation output files, git diffs |
| Lock table | ETS (not persisted) | Rebuilt from issue state on recovery |

> **Note:** There is one global cursor per run, not per-issue cursors. See "Global Cursor Model" section.

**What goes where:**

| Data Type | Storage | Rationale |
|-----------|---------|-----------|
| Issue state enum (Idle, Running, etc.) | DETS | Small, frequently updated |
| Resume token (string) | DETS | Small, needed for recovery |
| Global cursor (opaque) | DETS (run.dets) | Single cursor per run, critical for crash recovery |
| Attempt counters | DETS | Small integers |
| AgentEvents (tool calls, text) | disk_log | Bulk, append-only, includes issue_id for routing |
| Validation output (stdout/stderr) | Files | Can be large (MB), not needed in DETS |
| Git diff content | Files | Can be large |

**DETS record structure (kept small):**
```gleam
pub type PersistedIssueState {
  issue_id: String,
  state: StateEnum,  // Idle, Running, Gating, etc.
  attempt: Int,
  session_id: Option(String),
  resume_token: Option(String),
  gate_retries: Int,
  review_retries: Int,
  evidence_dir: Option(String),  // Path to evidence files, not content
  // NOTE: No per-issue cursor - global cursor is in run.dets
}

// Global cursor is stored in run.dets, not per-issue
// See "Global Cursor Model" section for details
```

### Storage Location

All persistent state lives under `~/.config/mala/runs/<run_id>/`:
```
~/.config/mala/runs/
  <run_id>/
    run.dets          # Run-level metadata
    issues.dets       # Per-issue state
    events.log        # Append-only events (disk_log)
    evidence/         # Validation outputs (files)
      <issue_id>/
        check.stdout
        check.stderr
        test.stdout
        ...
```

### Crash Consistency Semantics

**Ordering guarantee:** Append to disk_log first, then update DETS with cursor.

This ensures that on crash recovery, we can scan disk_log from the last known cursor to find any events that were logged but not yet reflected in DETS state.

**Write protocol:**
```
1. Receive AgentEvent
2. Append event to disk_log → get log_position
3. Process event (update in-memory state)
4. Write updated state + log_position to DETS
5. Call dets.sync() for critical state transitions (checkpoint, final)
```

**Recovery protocol (cursor-driven, not comparison-driven):**
```
1. Open DETS, read global_cursor (single run-level cursor, not per-issue)
2. Open disk_log, read forward from global_cursor using disk_log:chunk/2
3. For each event in chunk:
   - Route to appropriate issue based on event.issue_id
   - Update in-memory state for that issue
4. Persist updated global_cursor to DETS
5. Mark run as recovered
```

See "Persistence API Contract" for the authoritative cursor model.

### Persistence API Contract

This section specifies the exact disk_log configuration, cursor semantics, and recovery rules.

**disk_log configuration:**

| Setting | Value | Rationale |
|---------|-------|-----------|
| Mode | `wrap` | Auto-rotate segments to limit disk usage |
| Format | `internal` (term_to_binary) | Efficient, supports arbitrary Gleam terms |
| Size | `{50_000_000, 10}` | 50 MB per segment, 10 segments max = 500 MB |
| Notify | `false` | No need for rotation notifications |

```gleam
pub fn open_event_log(run_id: String) -> Result(DiskLog, Error) {
  let path = runs_dir() <> "/" <> run_id <> "/events"
  disk_log.open([
    #(name, string.to_atom("events_" <> run_id)),
    #(file, path),
    #(type_, wrap),
    #(format, internal),  // term_to_binary, not JSON
    #(size, #(50_000_000, 10)),  // 50MB x 10 segments
  ])
}
```

**Event serialization:**

Events are stored as Erlang terms (via `term_to_binary`), not JSON:
- More efficient than JSON for complex structures
- Preserves Gleam types exactly
- Compatible with disk_log's `internal` format

```gleam
// What gets written to disk_log
pub type PersistedEvent {
  PersistedEvent(
    schema_version: Int,      // For forward compatibility
    timestamp: Int,           // Unix millis
    issue_id: String,
    event: AgentEvent,
  )
}

// disk_log stores term_to_binary(PersistedEvent)
```

### Event Encoding Contract

**Canonical representations:**

| Context | Format | Notes |
|---------|--------|-------|
| In-memory | `AgentEvent` (Gleam type) | Canonical runtime representation |
| On-disk (event log) | Erlang term (`PersistedEvent`) | Binary via `term_to_binary` |
| Ingestion (Claude CLI) | JSONL | Parsed into `AgentEvent` on receipt |
| Export (`mala logs --json`) | JSON | Deterministic mapping from `AgentEvent` |

**Schema versioning:**

The `schema_version` field in `PersistedEvent` enables forward compatibility:
- v1 readers can skip events with `schema_version > 1`
- New fields are added to `AgentEvent` without breaking old readers
- Version bumps only when breaking changes occur

**JSON export contract (`mala logs --json`):**

Even though v1 may not ship `mala logs --json`, the mapping is defined for future use:

```gleam
pub fn agent_event_to_json(event: AgentEvent) -> Json {
  case event {
    ToolUse(id, name, input) -> json.object([
      #("type", json.string("tool_use")),
      #("tool_id", json.string(id)),
      #("tool_name", json.string(name)),
      #("input", input),  // Already JSON
    ])
    ToolResult(id, status, output) -> json.object([
      #("type", json.string("tool_result")),
      #("tool_id", json.string(id)),
      #("status", json.string(status_to_string(status))),
      #("output", json.string(output)),
    ])
    AssistantText(text) -> json.object([
      #("type", json.string("assistant_text")),
      #("text", json.string(text)),
    ])
    // ... other event types
  }
}
```

**Why term encoding (not JSON) for persistence:**
- JSON requires schema-aware parsing; term encoding preserves types directly
- Smaller disk footprint (no repeated field names)
- Faster serialization/deserialization
- Native to OTP ecosystem (disk_log expects terms)

### Global Cursor Model (Authoritative)

**Single global cursor per run, not per-issue cursors.**

The event log is shared across all issues in a run. We use a **single global cursor** at the run level, not per-issue cursors. This simplifies recovery and avoids cursor comparison issues.

**Why single global cursor:**
- disk_log continuations are opaque - cannot compare cursors from different read sessions
- Per-issue cursors would require cursor ordering, which disk_log wrap mode doesn't provide
- Recovery is simpler: read forward from global cursor, route events by `issue_id`

**Cursor type:**

```gleam
// The ONLY cursor type - opaque continuation from disk_log:chunk/2
pub type EventCursor {
  EventCursor(
    continuation: Dynamic,  // Erlang term from disk_log, opaque to Gleam
  )
}

// Special value for "start of log"
pub const CURSOR_START = EventCursor(continuation: start)
```

**Run state with single cursor:**

```gleam
pub type PersistedRunState {
  run_id: String,
  status: RunStatus,
  global_cursor: EventCursor,  // Single cursor for the whole run
  issues: Map(String, PersistedIssueState),  // Keyed by issue_id
}

pub type PersistedIssueState {
  issue_id: String,
  state: StateEnum,
  attempt: Int,
  session_id: Option(String),
  resume_token: Option(String),
  gate_retries: Int,
  review_retries: Int,
  // NO per-issue cursor - derived from replaying events
}
```

**Reading events (cursor-driven):**

```gleam
pub fn read_from_cursor(log: DiskLog, cursor: EventCursor) -> Result(#(List(PersistedEvent), EventCursor), Error) {
  case disk_log.chunk(log, cursor.continuation) {
    #(continuation, terms) -> {
      let events = list.map(terms, decode_event)
      Ok(#(events, EventCursor(continuation)))
    }
    #(eof, []) -> Ok(#([]), cursor))
    {error, reason} -> Error(DiskLogError(reason))
  }
}
```

**Recovery algorithm (no cursor comparisons):**

```gleam
pub fn recover_run(run_id: String) -> Result(RunState, Error) {
  // 1. Load persisted state
  use persisted <- result.try(load_dets(run_id))

  // 2. Open disk_log
  use log <- result.try(open_event_log(run_id))

  // 3. Read forward from last synced cursor
  let cursor = persisted.global_cursor
  let #(events, new_cursor) = read_all_from_cursor(log, cursor)

  // 4. Replay events to rebuild in-memory state
  let state = list.fold(events, persisted_to_memory(persisted), fn(state, event) {
    // Route to issue by event.issue_id
    apply_event(state, event)
  })

  // 5. Update cursor
  Ok(RunState(..state, global_cursor: new_cursor))
}

fn read_all_from_cursor(log: DiskLog, cursor: EventCursor) -> #(List(PersistedEvent), EventCursor) {
  case read_from_cursor(log, cursor) {
    Ok(#([], cursor)) -> #([], cursor)  // EOF
    Ok(#(batch, new_cursor)) -> {
      let #(rest, final_cursor) = read_all_from_cursor(log, new_cursor)
      #(list.append(batch, rest), final_cursor)
    }
    Error(_) -> #([], cursor)  // On error, return what we have
  }
}
```

**DETS repair policy:**

| Condition | Action | Result |
|-----------|--------|--------|
| `dets.open` succeeds | Proceed | Run resumable |
| `dets.open` fails, repair succeeds | Proceed with warning | Run resumable |
| Repair fails | Log error, delete corrupted file | Run NOT resumable (start fresh) |

```gleam
pub fn open_dets_with_repair(path: String) -> Result(Dets, RunState) {
  case dets.open(path, [#(repair, true)]) {
    Ok(dets) -> Ok(dets)
    Error(#(error, _)) -> {
      log.error("DETS repair failed: " <> inspect(error))
      // Delete corrupted file
      let _ = file.delete(path)
      // Return marker indicating fresh start needed
      Error(CorruptedNotResumable(path))
    }
  }
}
```

**Unrecoverable vs resumable runs:**

| State | Resumable? | Action |
|-------|------------|--------|
| DETS valid, disk_log valid | Yes | Resume from cursor |
| DETS valid, cursor points to rotated-out segment | **No** | See below |
| DETS corrupted, disk_log valid | No | Start fresh, cannot recover state |
| Both corrupted | No | Start fresh |

### Rotation Failure Mode

In `disk_log` wrap mode, old segments are discarded when the log exceeds its size limit. If the persisted `global_cursor` points into a rotated-out segment, **the run cannot be resumed**.

**Detection:**
```gleam
pub fn validate_cursor(log: DiskLog, cursor: EventCursor) -> Result(Nil, Error) {
  case disk_log.chunk(log, cursor.continuation) {
    {error, {invalid_continuation, _}} ->
      Error(CursorRotatedOut(
        "Events have been rotated out of the log. " <>
        "Run cannot be resumed. Start fresh with: mala run --force"
      ))
    _ -> Ok(Nil)
  }
}
```

**User-facing error:**
```
Error: Cannot resume run 20260110-143052
Events written before the last checkpoint have been rotated out of the event log.
The run state cannot be recovered.

Options:
  1. Start fresh: mala run --force
  2. Increase log retention: set [telemetry] disk_log_size in mala.toml
```

**Preventing rotation during active runs:**

The default disk_log size (500 MB = 50 MB × 10 segments) is sized to avoid rotation during typical runs:
- 1000 events × ~1 KB/event = ~1 MB per run
- Retention allows ~500 concurrent runs before rotation

**v1 sizing validation:**

At startup, validate that disk_log retention is sufficient for the configured `max_agents`:

```gleam
pub fn validate_disk_log_sizing(config: Config) -> Result(Nil, Warning) {
  let max_agents = config.run.max_agents |> option.unwrap(4)
  let expected_events_per_issue = 100  // Conservative estimate
  let event_size_kb = 2
  let estimated_mb = max_agents * expected_events_per_issue * event_size_kb / 1024

  let total_retention_mb = config.telemetry.disk_log_segment_size
    * config.telemetry.disk_log_segment_count
    / 1_000_000

  case estimated_mb * 10 < total_retention_mb {  // 10x safety margin
    True -> Ok(Nil)
    False -> Warning(
      "disk_log retention may be insufficient for max_agents=" <> int.to_string(max_agents) <>
      ". Consider increasing disk_log_segment_count to ensure resume reliability."
    )
  }
}
```

**Acceptance test requirement:** The crash-resume test fixture MUST use disk_log retention that cannot rotate during the scenario (enforced in test setup).

For long-running or high-volume runs, increase retention:
```toml
[telemetry]
disk_log_segment_size = 100_000_000  # 100 MB per segment
disk_log_segment_count = 20          # 20 segments = 2 GB total
```

### Cursor Advancement Contract

**Key constraint:** The `global_cursor` is obtained **only from `disk_log:chunk/2` continuations**, not from write operations.

**Why this matters:**
- `disk_log:log/2` does not return a position/continuation
- The continuation from `chunk/2` is the only reliable way to mark "read up to here"
- Mixing write-time positions with read-time continuations causes correctness bugs

**Cursor update flow:**
```
                    ┌─────────────────────────────────────────┐
                    │           Event Processing              │
                    └─────────────────────────────────────────┘
                                      │
     ┌────────────────────────────────┼────────────────────────────────┐
     │                                │                                │
     ▼                                ▼                                ▼
┌─────────────┐              ┌─────────────────┐             ┌─────────────────┐
│ Agent emits │              │ run_manager     │             │ Periodic sync   │
│ event       │──append──────│ appends to      │             │ (every N events │
│             │              │ disk_log        │             │  or M seconds)  │
└─────────────┘              └────────┬────────┘             └────────┬────────┘
                                      │                               │
                                      │                               │
                                      ▼                               ▼
                             ┌─────────────────┐             ┌─────────────────┐
                             │ NO cursor       │             │ chunk/2 to get  │
                             │ returned here   │             │ continuation    │
                             └─────────────────┘             └────────┬────────┘
                                                                      │
                                                                      ▼
                                                             ┌─────────────────┐
                                                             │ Persist cursor  │
                                                             │ to DETS         │
                                                             └─────────────────┘
```

**Sync interval:**
- Cursor is persisted after every `sync_interval` events (default: 100)
- Or after every `sync_timeout` seconds (default: 30)
- Or on explicit `Sync` intent (checkpoint, final)

This means on crash, up to `sync_interval` events may be replayed (idempotent replay is required).

**Failure modes and recovery:**

| Failure Point | State After Crash | Recovery Action |
|--------------|-------------------|-----------------|
| After disk_log write, before DETS | Event logged, state stale | Replay event from cursor |
| After DETS write, before sync | State updated, may be lost | Replay from last synced cursor |
| After sync | Fully persisted | No replay needed |

**Sync policy:**
- `dets.sync()` called after: checkpoint, final, state transition to Gating/Reviewing/Done
- NOT called after every event (performance trade-off: may replay a few events on crash)

### Recovery Flow

**On CLI restart (new `mala run` with `--resume`):**
1. Load `run.dets` to find the last active run.
2. Load `issues.dets` to reconstruct `IssueState` for each in-progress issue.
3. Rebuild lock table in ETS from issue state (which issues hold which locks).
4. Reconcile with external reality:
   - Query Beads for current issue status (may have been manually closed).
   - Check git state for commits with `bd-<issue_id>` (may have been pushed).
   - Remove issues that are now externally resolved.
5. Resume workers for issues that are still in-progress.

**On supervisor restart (process crash within a run):**
1. Supervisor restarts the crashed worker.
2. Worker reads its state from `issues.dets` (keyed by issue_id).
3. Worker resumes from last checkpoint (session resume if supported, else fresh start).
4. No replay needed—state is persisted incrementally.

### Idempotency Guarantees

- **Issue closing:** Before closing an issue via `bd close`, check if already closed (Beads query). Skip if already closed.
- **Validation:** Evidence is keyed by `(issue_id, validation_command, attempt)`. Re-running validation overwrites evidence for that key.
- **Review:** Review results are keyed by `(issue_id, attempt)`. Duplicate review requests are deduplicated.
- **Lock release:** Lock release is idempotent—releasing an unheld lock is a no-op.

### State Hydration on Worker Startup

```gleam
pub fn init_from_persisted(issue_id: String) -> Result(IssueState, Error) {
  use dets <- result.try(dets.open("runs/current/issues.dets"))
  use record <- result.try(dets.lookup(dets, issue_id))
  case record {
    Some(state) -> Ok(state)
    None -> Ok(IssueState.new(issue_id))
  }
}
```

### DETS Limitations and Mitigations

DETS has known limitations that must be acknowledged:

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| 2 GB max file size | Large runs with many issues could exceed | Monitor size, split into shards if needed |
| Single writer | Concurrent writes block | All writes go through `run_manager` (single writer) |
| Corruption on unclean shutdown | Data loss risk | Use `dets.sync/1` after critical writes |
| Slow with many records | Performance degradation | Typical runs have <1000 issues, well within limits |

### Single-Writer Persistence Model

**All DETS writes are funneled through `run_manager`** to avoid concurrent write issues.

**Why single-writer:**
- DETS performance degrades with concurrent writers (lock contention)
- Risk of corruption if multiple processes write simultaneously during crash
- EventCursor updates must be serialized with disk_log appends

**Architecture:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  IssueWorker 1  │     │  IssueWorker 2  │     │  SessionRunner  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │ PersistIntent         │ PersistIntent         │ PersistIntent
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │     run_manager        │
                    │   (single writer)      │
                    │  - batches writes      │
                    │  - serializes access   │
                    └────────────┬───────────┘
                                 │ sync writes
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ disk_log │ │ run.dets │ │issues.dets│
              └──────────┘ └──────────┘ └──────────┘
```

**Persist intent API:**

> **Critical:** Workers NEVER write to DETS/disk_log directly. They only send `PersistIntent` messages to `run_manager`, which serializes all disk operations.

```gleam
pub type PersistIntent {
  UpdateIssueState(issue_id: String, state: StateEnum)
  UpdateResumeToken(issue_id: String, token: String)
  RecordEvent(issue_id: String, event: AgentEvent)  // Append only, no cursor
  UpdateGateRetries(issue_id: String, retries: Int)
  SyncCursor  // Advance global_cursor via chunk/2 and persist
  Sync  // Force dets.sync() for critical transitions
}

// Workers send intents, run_manager applies them
pub fn send_persist(intent: PersistIntent) {
  gen_server.cast(run_manager, Persist(intent))
}

// run_manager handles all persistence (SINGLE WRITER)
pub fn handle_cast(Persist(intent), state: RunState) -> RunState {
  case intent {
    RecordEvent(issue_id, event) -> {
      // Append to disk_log - NO cursor returned
      let _ = disk_log.log(state.event_log, wrap_event(issue_id, event))
      // Increment pending count for cursor sync
      RunState(..state, pending_events: state.pending_events + 1)
    }
    SyncCursor -> {
      // Obtain cursor from chunk/2 (the ONLY way to get a valid cursor)
      let #(_, new_cursor) = read_to_end(state.event_log, state.global_cursor)
      dets.insert(state.run_dets, "global_cursor", new_cursor)
      RunState(..state, global_cursor: new_cursor, pending_events: 0)
    }
    UpdateIssueState(issue_id, new_state) -> {
      dets.insert(state.issues_dets, issue_id, new_state)
      state
    }
    Sync -> {
      dets.sync(state.issues_dets)
      dets.sync(state.run_dets)
      state
    }
    // ... other intents
  }
}
```

**Idempotent replay contract:**

Since cursor sync is periodic, crashes may cause event replay. Events must be idempotent:
- State transitions: applying same event twice yields same state
- Evidence files: overwritten, not duplicated
- Event IDs: each event has a unique ID for dedup if needed

**Sync triggers (when `dets.sync()` is called):**
- After `Checkpoint` event (mid-session save point)
- After `Final` event (session complete)
- After state transition to `Gating`, `Reviewing`, or `Done`
- On graceful shutdown

**Backpressure:** If workers generate events faster than `run_manager` can persist, the message queue grows. This is acceptable for typical workloads (events are small). For extreme cases, add flow control via `gen_server.call` for critical persists.

**Corruption recovery:**
- On startup, validate DETS files with `dets.open/2` options `{repair, true}`
- If repair fails, log error and start fresh (run cannot be resumed)

### Data Retention and Privacy

**Event log rotation:**
- `disk_log` configured with `{size, {50_000_000, 10}}` (50 MB per segment, 10 segments max = 500 MB per run)
- Oldest segments automatically deleted when limit reached

**Sensitive data handling:**

### Event Persistence and Redaction Policy

This section defines explicit boundaries for what is persisted, redacted, and retained.

**What is persisted (by data type):**

| Data | Persisted | Location | Redaction |
|------|-----------|----------|-----------|
| Tool name and ID | Always | disk_log | Never |
| Tool input (command/path) | Conditional | disk_log | Pattern-based |
| Tool output (stdout/stderr) | Conditional | disk_log + evidence files | Pattern-based |
| Assistant text | Conditional | disk_log | Pattern-based |
| Usage metrics | Always | disk_log | Never |
| Timestamps | Always | disk_log | Never |
| Session/resume tokens | Always | DETS | Never |
| Gate evidence | Always | evidence files | Never (orchestrator-generated) |

**`telemetry.persist_content` modes:**

| Mode | Tool input | Tool output | Assistant text | Evidence |
|------|------------|-------------|----------------|----------|
| `true` (default) | Full, redacted | Full, redacted | Full, redacted | Full |
| `false` | Hash only | Hash only | Hash only | Full |
| `metadata_only` | Omitted | Omitted | Omitted | Full |

**Redaction implementation:**

```gleam
pub type RedactionConfig {
  patterns: List(Regex),
  replacement: String,
  max_content_size: Int,  // Truncate after this (default: 100KB)
}

const DEFAULT_REDACTION_PATTERNS = [
  // API keys and tokens
  regex.compile("(?i)(api[_-]?key|token|secret|password|bearer)\\s*[:=]\\s*['\"]?\\S+"),
  // AWS credentials
  regex.compile("AKIA[0-9A-Z]{16}"),
  regex.compile("(?i)aws[_-]?secret[_-]?access[_-]?key\\s*[:=]\\s*\\S+"),
  // Generic secrets
  regex.compile("(?i)(password|passwd|pwd)\\s*[:=]\\s*['\"]?\\S+"),
  // Private keys
  regex.compile("-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----"),
]

pub fn redact_content(content: String, config: RedactionConfig) -> String {
  let content = string.slice(content, 0, config.max_content_size)
  list.fold(config.patterns, content, fn(c, pattern) {
    regex.replace(pattern, c, "[REDACTED]")
  })
}
```

**Redaction timing:**

Redaction is applied **before** writing to disk_log and evidence files, not after. This ensures:
1. Secrets never touch disk in plaintext
2. No need to scrub logs post-hoc
3. Consistent view across all consumers

```gleam
pub fn persist_event(event: AgentEvent, config: RedactionConfig) -> EventCursor {
  // Redact BEFORE persistence
  let redacted_event = redact_agent_event(event, config)
  disk_log.log(redacted_event)
}

fn redact_agent_event(event: AgentEvent, config: RedactionConfig) -> AgentEvent {
  case event {
    ToolUse(id, name, input) -> ToolUse(id, name, redact_json(input, config))
    ToolResult(id, status, output) -> ToolResult(id, status, redact_content(output, config))
    AssistantText(text) -> AssistantText(redact_content(text, config))
    _ -> event  // Timestamps, usage, etc. not redacted
  }
}
```

**Content size limits:**

| Content Type | Max Size | Truncation Strategy |
|--------------|----------|---------------------|
| Tool output | 100 KB | Keep first 50KB + last 50KB with `[...truncated...]` |
| Assistant text | 50 KB | Keep first 50KB |
| Command evidence | 1 MB | Keep first 500KB + last 500KB |

**Hash-only mode (`persist_content = false`):**

When content persistence is disabled, store SHA-256 hashes for debugging:

```gleam
pub fn hash_only_event(event: AgentEvent) -> AgentEvent {
  case event {
    ToolUse(id, name, input) ->
      ToolUse(id, name, json.object([
        #("_hash", json.string(sha256(json.encode(input)))),
        #("_size", json.int(string.length(json.encode(input)))),
      ]))
    ToolResult(id, status, output) ->
      ToolResult(id, status, json.object([
        #("_hash", json.string(sha256(output))),
        #("_size", json.int(string.length(output))),
      ]))
    AssistantText(text) ->
      AssistantText("[hash:" <> sha256(text) <> ",size:" <> int.to_string(string.length(text)) <> "]")
    _ -> event
  }
}
```

**Unified redaction policy:**

**All persisted content is redacted by default**, including both agent events and orchestrator command output.

| Content Type | Source | Redaction | Rationale |
|--------------|--------|-----------|-----------|
| Agent tool input/output | Agent events | Pattern-based | May contain secrets typed by agent |
| Agent assistant text | Agent events | Pattern-based | May echo secrets |
| Validation stdout/stderr | Orchestrator commands | Pattern-based | Test logs can print env vars, stack traces with tokens |
| Command evidence | Orchestrator commands | Pattern-based | pip/uv output, HTTP calls can echo auth headers |

**Why evidence is also redacted:**

Orchestrator-executed commands are **not inherently safe**:
- Test output can print `DATABASE_URL`, `API_KEY` from env
- Stack traces may contain auth tokens
- HTTP client libraries log request headers
- Build tools echo environment during failures

**Evidence storage with redaction:**
```gleam
pub fn store_evidence(
  issue_id: String,
  command: String,
  output: CommandOutput,
  config: RedactionConfig,
) -> EvidencePath {
  // Redact BEFORE writing to disk
  let redacted_stdout = redact_content(output.stdout, config)
  let redacted_stderr = redact_content(output.stderr, config)

  let path = evidence_path(issue_id, command)
  file.write(path <> ".stdout", redacted_stdout)
  file.write(path <> ".stderr", redacted_stderr)
  file.write(path <> ".meta", json.encode(#(
    "exit_code", output.exit_code,
    "command", command,  // argv is safe, no secrets
    "duration_ms", output.duration_ms,
  )))
  path
}
```

**Raw evidence mode (opt-in, dangerous):**

For debugging, users can enable `telemetry.raw_evidence = true`:
- Evidence files are NOT redacted
- Files are stored with restrictive permissions (0600)
- Warning logged at startup: "Raw evidence mode enabled - secrets may be written to disk"
- Not recommended for production use

```toml
[telemetry]
persist_content = true
raw_evidence = false  # Default: evidence is redacted
```

**Env filtering is NOT sufficient for privacy:**

Even with env filtering (denylist of `AWS_*`, `_SECRET`, etc.), orchestrator outputs can still contain secrets:
- Secrets passed as command arguments (not env)
- Secrets echoed from config files read by commands
- Secrets in error messages from external services

Redaction is the primary defense; env filtering is defense-in-depth.

**Encryption at rest:**
- Not implemented in v1
- Future: Consider DETS alternative with encryption (e.g., SQLite with SQLCipher)

### Cleanup Policy

- Completed runs: Retained for 7 days (configurable via `run.retention_days`).
- Failed runs: Retained for 30 days.
- `mala clean` removes runs older than retention period.
- Worktrees are cleaned up on run completion or via explicit `mala clean --worktrees`.

## Configuration System

Layered config precedence:
1) CLI
2) Env
3) User config (~/.config/mala)
4) Project config (mala.toml)
5) Defaults

Config types are validated at startup. All agent profiles, validation specs,
review settings, and runtime directories live in a single typed config.

### Config Schema (Field List + Types + Defaults)

**run**
- `max_agents`: int | default: null (unlimited)
- `max_issues`: int | default: null (unlimited)
- `timeout_minutes`: int | default: 60
- `watch`: bool | default: false
- `validate_every`: int | default: 10
- `order`: string | default: "epic-priority"
- `scope`: string | default: "all"
- `resume`: bool | default: false

**paths**
- `runs_dir`: string | default: "~/.config/mala/runs"
- `lock_dir`: string | default: "/tmp/mala-locks"
- `logs_dir`: string | default: "~/.config/mala/logs"

**agents.<profile>**
- `backend`: string | default: "claude_cli"
- `model`: string | default: "opus"
- `permission_mode`: string | default: "bypassPermissions"
- `timeout_sec`: int | default: 3600
- `settings_sources`: list[string] | default: ["local", "project"]
- `env`: map[string]string | default: {}

Note: Capability flags (`supports_resume`, `supports_hooks`, `supports_tool_restrictions`) are **not** configurable. They are derived from the adapter at runtime via `adapter.capabilities()`. See "Capability Detection" section below.

**issue_provider**
- `type`: string | default: "beads"
- `bd_path`: string | default: "bd"
- `include_wip`: bool | default: false
- `env`: map[string]string | default: {}
  - Environment variables passed to `bd` CLI invocations
  - Required for CI/CD: `GITHUB_TOKEN`, `JIRA_API_TOKEN`, etc.
  - Values can reference parent env: `"${GITHUB_TOKEN}"`
  - Explicitly enumerated keys only (no wildcard passthrough)

**Example issue_provider with auth:**
```toml
[issue_provider]
type = "beads"
bd_path = "bd"
env = { GITHUB_TOKEN = "${GITHUB_TOKEN}" }
```

**locks**
- `enable`: bool | default: true
- `deadlock_detection`: bool | default: true
- `deadlock_interval_sec`: int | default: 5

**validation**
- `preset`: string | default: null
- `disable`: list[string] | default: []
- `global_enabled`: bool | default: true
- `require_clean_git`: bool | default: true
- `require_pytest_for_code_changes`: bool | default: true

**validation.commands.<cmd>** (per-command configuration)
- `cmd`: list[string] (argv) | required
  - Array format only: `["uv", "run", "pytest"]`
- `env`: map[string]string | default: {}
  - Per-command environment variables (e.g., `DATABASE_URL` for test only)
  - Explicitly enumerated keys only (no wildcard passthrough)
  - Values from `validation.commands.<cmd>.env` are redacted in logs using same patterns as denylist
- `timeout_sec`: int | default: 300
  - Per-command timeout override

**Shorthand format:** If only the command array is needed, use direct assignment:
```toml
[validation.commands]
lint = ["uvx", "ruff", "check", "."]  # Shorthand: no env, default timeout
```

**Full format with env:**
```toml
[validation.commands.test]
cmd = ["uv", "run", "pytest"]
env = { DATABASE_URL = "${DATABASE_URL}", TEST_API_KEY = "${TEST_API_KEY}" }
timeout_sec = 600
```

**validation.coverage**
- `enabled`: bool | default: false
- `format`: string | default: "xml"
- `file`: string | default: null
- `threshold`: int | default: null

**validation.code_patterns / config_files / setup_files**
- `paths`: list[string] | default: []

**validation.triggers.<trigger>**
- `failure_mode`: string | default: "continue"
- `fire_on`: string | default: "success"
- `max_retries`: int | default: 3

**validation.triggers.<trigger>.code_review**
- `enabled`: bool | default: false
- `reviewer_type`: string | default: "cerberus"
- `finding_threshold`: string | default: "none"
- `baseline`: string | default: null

**review**
- `enabled`: bool | default: true (v1: fully functional)
- `backend`: string | default: "cerberus"
- `review_timeout_sec`: int | default: 1200
- `spawn_args`: list[string] | default: []
- `wait_args`: list[string] | default: []
- `env`: map[string]string | default: {}
- `track_low_priority`: bool | default: true

**epic_verification**
- `enabled`: bool | default: false (v1: stub only, ignored)
- `max_retries`: int | default: 3
- `llm_api_key`: string | default: null
- `llm_base_url`: string | default: null

**telemetry**
- `enabled`: bool | default: false
- `exporter`: string | default: "opentelemetry"
- `service_name`: string | default: "mala"

### Proposed TOML Config (mala.toml)

```toml
[run]
max_agents = 4
max_issues = 100
timeout_minutes = 60
watch = false
validate_every = 10
order = "epic-priority"  # focus | epic-priority | issue-priority | input
scope = "all"            # all | epic:<id> | ids:<id,...> | orphans
resume = false

[paths]
runs_dir = "~/.config/mala/runs"
lock_dir = "/tmp/mala-locks"
logs_dir = "~/.config/mala/logs"

[agents.default]
backend = "claude_cli"   # claude_cli | codex_app | amp_cli | open_code_cli
model = "opus"
permission_mode = "bypassPermissions"
timeout_sec = 3600
settings_sources = ["local", "project"]
# Note: supports_resume is NOT configurable - it's detected from the adapter

[agents.codex_fast]
backend = "codex_app"
model = "gpt-4.1"
timeout_sec = 1800

[issue_provider]
type = "beads"
bd_path = "bd"
include_wip = false

[locks]
enable = true
deadlock_detection = true
deadlock_interval_sec = 5

[validation]
preset = "python-uv"
disable = ["e2e"]
global_enabled = true
require_clean_git = true
require_pytest_for_code_changes = true

[validation.commands]
# Shorthand format (no env needed)
setup = ["uv", "sync"]
lint = ["uvx", "ruff", "check", "."]
format = ["uvx", "ruff", "format", "."]
typecheck = ["uvx", "ty", "check"]

# Full format with per-command env
[validation.commands.test]
cmd = ["uv", "run", "pytest"]
env = { DATABASE_URL = "${DATABASE_URL}", TEST_API_KEY = "${TEST_API_KEY}" }
timeout_sec = 600

[validation.commands.e2e]
cmd = ["uv", "run", "pytest", "-m", "e2e"]
env = { DATABASE_URL = "${DATABASE_URL}" }
timeout_sec = 1200

[validation.coverage]
enabled = true
format = "xml"
file = "coverage.xml"
threshold = 80

[validation.code_patterns]
paths = ["**/*.py", "pyproject.toml"]

[validation.config_files]
paths = ["pyproject.toml", "ruff.toml"]

[validation.setup_files]
paths = ["uv.lock", "pyproject.toml"]

[validation.triggers.session_end]
failure_mode = "remediate"  # abort | continue | remediate
max_retries = 3

[validation.triggers.session_end.code_review]
enabled = true
reviewer_type = "cerberus"
finding_threshold = "P1"

[validation.triggers.run_end]
fire_on = "success"
failure_mode = "continue"
max_retries = 1

[review]
enabled = true  # v1: fully functional
backend = "cerberus"
review_timeout_sec = 1200
spawn_args = []
wait_args = []
env = {}
track_low_priority = true

[epic_verification]
enabled = false  # v1: stub only, setting true has no effect
max_retries = 3
llm_api_key = "${LLM_API_KEY}"
llm_base_url = "${LLM_BASE_URL}"

[telemetry]
enabled = false
exporter = "opentelemetry"
service_name = "mala"
```

Notes:
- Environment variable interpolation: config loader resolves `${VAR}` syntax.
- TOML only - no YAML support.

## External Dependencies (Erlang/OTP)

- OTP: supervisor, gen_server, gen_statem, ets, dets, disk_log, digraph
- telemetry + opentelemetry
- erlexec (process execution)
- ~~yaml_erl or yamerl~~ (not needed - TOML only, no YAML support)
- tom (TOML parsing)
- jiffy or jsx (JSON)
- hackney or finch (HTTP client for Codex API adapter and HTTP-based backends)
- gun (WebSocket client for Codex streaming)
- cowboy (HTTP server, optional for future MCP server mode)

## Error Handling Policy

- Process crashes are isolated and restarted by supervisors.
- Issue worker failures mark the issue as follow-up and continue run.
- Agent adapter failures trigger retry with backoff.
- Any unhandled failure in run_manager ends run with nonzero exit.

### Graceful Shutdown (Signal Handling)

**SIGINT/SIGTERM handling:**

1. **Signal received:** OTP application receives shutdown signal.
2. **Shutdown initiated:** `run_manager` enters shutdown mode.
3. **No new issues:** Scheduler stops spawning new issue workers.
4. **Grace period:** Allow in-flight issues up to 30 seconds to reach a checkpoint.
5. **Persist state:** Force persist current state for all workers to DETS.
6. **Terminate workers:** Send `Abort("shutdown")` to all workers.
7. **Cleanup:** Remove transient resources (sockets, but NOT worktrees).
8. **Exit:** Application exits with code 0 (graceful) or 130 (SIGINT).

```gleam
pub fn handle_shutdown(state: RunState) -> RunState {
  // Stop accepting new work
  scheduler.pause(state.scheduler)

  // Give workers grace period
  process.sleep(30_000)

  // Force persist and abort all
  list.each(state.workers, fn(w) {
    issue_worker.force_persist(w)
    issue_worker.send(w, Abort("shutdown"))
  })

  RunState(..state, shutdown: True)
}
```

**Resume after shutdown:**
- `mala run --resume` picks up from last persisted state.
- Issues interrupted mid-session restart from last checkpoint (if supported).

## Cutover Strategy

> **Note:** This is a clean break from Python. No backward compatibility with Python mala is required.

### Clean Break Model

The Gleam implementation replaces Python entirely:
- **New binary name:** `mala` (Gleam) replaces Python version
- **No coexistence:** Python version is deprecated immediately
- **Config breaking changes:** Array-only command format enforced
- **State incompatibility:** DETS files are not compatible with Python JSON files

### Config Format

**TOML only, no YAML support**

This is a clean break. YAML is not supported and will not be added.

| Format | Support | Notes |
|--------|---------|-------|
| TOML (`mala.toml`) | **Yes** | Only supported format |
| YAML (`mala.yaml`) | **No** | Not supported, no migration |

**Breaking changes:**
- Commands must be arrays: `cmd = ["uv", "run", "pytest"]`
- String commands rejected with error
- `mala.yaml` files are **ignored** - users must manually convert to TOML

**No migration tooling:** This is a full rewrite. Users convert their configs manually.

**Env interpolation:** `${VAR}` syntax in TOML values.

**Env interpolation rules:**
- Interpolation happens at config load time from process environment
- Missing `${VAR}` → **config validation error** with clear message: "Environment variable VAR not set (referenced in config)"
- Empty `${VAR}` (var exists but empty) → empty string substituted
- Interpolated values containing secrets are redacted if they match redaction patterns

**Precedence:** CLI > Env > User config (`~/.config/mala/mala.toml`) > Project config (`mala.toml`) > Defaults.

### Acceptance Test Matrix

Gleam version must pass these scenarios before release (standalone, no Python comparison):

| Scenario | Validation |
|----------|------------|
| Simple run (single issue, gate pass) | Issue closed, commits present |
| Multi-issue run (concurrency) | All issues processed, max_agents respected |
| Gate failure + retry | Retry occurs, progress detection works |
| Resume after crash | State recovered from DETS |
| Lock contention | Deadlock detected, resolved |
| Config loading | TOML config loads correctly |

### Documentation Updates

| Document | Changes |
|----------|---------|
| `docs/cli-reference.md` | Full rewrite for Gleam |
| `docs/project-config.md` | Document array-only commands |
| `README.md` | Update installation |

### State Files

Gleam uses DETS for state persistence. Old Python JSON state files are not compatible and will be ignored.

**On resume:**
- If `run.dets` exists: resume from DETS state
- If only old JSON files exist: warn user that state is incompatible, start fresh

## Implementation Plan (TDD Approach)

> **Note:** This section is intentionally detailed with phase breakdowns and test examples. Task extraction for execution will be done separately via `/create-tasks` or equivalent tooling. The detail here serves as a reference for implementers, not a task tracker.

### TDD Methodology

Each module follows: **Write test → See it fail → Implement → See it pass → Refactor**

### Phase 0: Project Scaffold

**Goal:** Gleam project compiles with type definitions.

**Tasks:**
1. Initialize Gleam project with OTP dependencies
2. Define type stubs for all domain models
3. Set up test infrastructure

**Verification:** `gleam build` succeeds, `gleam test` runs (empty tests pass).

### Phase 1: Domain Models (TDD)

**Test first:**
```gleam
// test/domain/events_test.gleam
pub fn agent_event_json_roundtrip_test() {
  let event = ToolUse("toolu_01", "bash", json.object([
    #("command", json.string("uv run pytest")),
  ]))
  let encoded = events.encode(event)
  let decoded = events.decode(encoded)
  assert decoded == Ok(event)
}

// test/domain/progress_test.gleam
// Progress detection uses orchestrator-observable state, NOT agent events
pub fn progress_detection_commit_test() {
  // Mock git.log output showing new commits since baseline
  let git_log_output = "abc123 bd-issue-1: fix typo\ndef456 bd-issue-1: add test"
  let baseline_sha = "000000"
  assert progress.detect_from_git(git_log_output, baseline_sha, "issue-1") == HasProgress(CommitsFound(2))
}

pub fn progress_detection_no_commits_test() {
  let git_log_output = ""  // No commits since baseline
  let baseline_sha = "abc123"
  assert progress.detect_from_git(git_log_output, baseline_sha, "issue-1") == NoProgress
}

// test/domain/resolution_test.gleam
// Resolution verification uses orchestrator-executed bd CLI, NOT agent events
pub fn resolution_verification_closed_test() {
  // Mock bd show output
  let bd_show_output = json.object([
    #("id", json.string("issue-1")),
    #("status", json.string("closed")),
    #("close_message", json.string("Fixed in abc123")),
  ])
  assert resolution.verify_from_beads(bd_show_output) == Resolved("Fixed in abc123")
}

pub fn resolution_verification_still_open_test() {
  let bd_show_output = json.object([
    #("id", json.string("issue-1")),
    #("status", json.string("open")),
  ])
  assert resolution.verify_from_beads(bd_show_output) == NotResolved
}
```

**Then implement:**
- `domain/events.gleam` - AgentEvent types + JSON codec
- `domain/progress.gleam` - Progress detection from git state (NOT agent events)
- `domain/resolution.gleam` - Resolution verification via Beads CLI (NOT agent events)
- `domain/issue.gleam` - Issue type

**Verification:** All domain unit tests pass.

### Phase 2: Infrastructure (TDD)

**Test first:**
```gleam
// test/infra/command_runner_test.gleam
pub fn run_command_captures_output_test() {
  let cmd = Command("echo", ["hello"])
  let result = command_runner.run(cmd, timeout: 5000)
  assert result == Ok(Output(stdout: "hello\n", stderr: "", exit_code: 0))
}

pub fn run_command_timeout_test() {
  let cmd = Command("sleep", ["10"])
  let result = command_runner.run(cmd, timeout: 100)
  assert result == Error(Timeout)
}

// test/infra/lock_server_test.gleam
pub fn acquire_release_test() {
  let server = lock_server.start()
  assert lock_server.acquire(server, "file.txt", "agent-1") == Ok(Acquired)
  assert lock_server.release(server, "file.txt", "agent-1") == Ok(Released)
}

pub fn acquire_conflict_test() {
  let server = lock_server.start()
  assert lock_server.acquire(server, "file.txt", "agent-1") == Ok(Acquired)
  assert lock_server.acquire(server, "file.txt", "agent-2") == Error(Conflict("agent-1"))
}

pub fn deadlock_detection_test() {
  let server = lock_server.start()
  lock_server.acquire(server, "a.txt", "agent-1")
  lock_server.acquire(server, "b.txt", "agent-2")
  // agent-1 wants b.txt, agent-2 wants a.txt → deadlock
  let result = lock_server.acquire(server, "b.txt", "agent-1")
  assert result == Error(DeadlockRisk(["agent-1", "agent-2"]))
}
```

**Then implement:**
- `infra/command_runner.gleam` - Command execution with argv-only
- `infra/lock_server.gleam` - ETS-based lock with deadlock detection
- `infra/mcp_socket.gleam` - Unix domain socket server for MCP proxy
- `infra/persistence.gleam` - DETS wrapper for state persistence

**Verification:** All infrastructure tests pass.

### Phase 3: Agent Adapters (TDD)

**Test first:**
```gleam
// test/agents/mock_adapter_test.gleam
pub fn mock_adapter_streams_events_test() {
  let events = [
    ToolUse("t1", "bash", json.object([])),
    ToolResult("t1", Ok("output")),
    Final("sess_1", None),
  ]
  let adapter = mock_adapter.new(events)
  let received = adapter |> stream.to_list()
  assert received == events
}

// test/agents/claude_adapter_test.gleam
pub fn claude_jsonl_parse_test() {
  let line = "{\"type\":\"tool_use\",\"tool_name\":\"bash\",\"tool_id\":\"t1\",\"input\":{\"command\":\"ls\"}}"
  let event = claude_adapter.parse_line(line)
  assert event == Ok(ToolUse("t1", "bash", json.object([#("command", json.string("ls"))])))
}

pub fn claude_spawn_creates_session_test() {
  // Integration test with mock filesystem
  let config = claude_adapter.Config(
    prompt: "test",
    worktree: "/tmp/test",
    settings_path: "/tmp/settings.json",
  )
  let result = claude_adapter.spawn(config)
  assert result.is_ok()
}
```

**Then implement:**
- `agents/adapter.gleam` - Adapter trait/interface
- `agents/mock_adapter.gleam` - Mock for testing
- `agents/claude_adapter.gleam` - Claude CLI adapter with JSONL parsing
- `agents/capability.gleam` - Capability detection (supports_hooks, supports_resume, etc.)

**Verification:** All adapter tests pass, mock adapter e2e works.

### Phase 4: State Machines (TDD)

**Test first:**
```gleam
// test/orchestrator/issue_worker_test.gleam
pub fn idle_to_running_on_start_test() {
  let worker = issue_worker.init(issue, config)
  assert worker.state == Idle
  let worker = issue_worker.handle(worker, Start(issue))
  assert worker.state == Running
}

pub fn running_to_gating_on_final_test() {
  let worker = issue_worker.init(issue, config) |> set_state(Running)
  let worker = issue_worker.handle(worker, AgentEvent(Final(session_id: "s1", resolution: None)))
  assert worker.state == Gating
}

pub fn gating_retry_on_fail_with_progress_test() {
  let worker = issue_worker.init(issue, config) |> set_state(Gating)
  let result = GateResult(Fail("tests failed"), progress: True)
  let worker = issue_worker.handle(worker, GateResult(result))
  assert worker.state == Running  // Retry
  assert worker.data.gate_retries == 1
}

pub fn gating_to_finalizing_on_no_progress_test() {
  let worker = issue_worker.init(issue, config) |> set_state(Gating)
  let result = GateResult(Fail("no changes"), progress: False)
  let worker = issue_worker.handle(worker, GateResult(result))
  assert worker.state == Finalizing
  assert worker.data.followup_reason == Some("no changes")
}

pub fn session_timeout_terminates_test() {
  let worker = issue_worker.init(issue, config) |> set_state(Running)
  let worker = issue_worker.handle(worker, Timeout(Session))
  assert worker.state == Finalizing
}

// test/orchestrator/session_runner_test.gleam
pub fn spawn_to_streaming_test() {
  let runner = session_runner.init(session_config)
  let runner = session_runner.handle(runner, Start("issue-1"))
  assert runner.state == Streaming
}

pub fn checkpoint_on_idle_timeout_test() {
  let runner = session_runner.init(session_config) |> set_state(Streaming)
  let runner = session_runner.handle(runner, Timeout(Idle))
  assert runner.state == Checkpoint
}
```

**Then implement:**
- `orchestrator/issue_worker.gleam` - gen_statem for issue lifecycle
- `orchestrator/session_runner.gleam` - gen_statem for agent session
- `orchestrator/scheduler.gleam` - gen_server for concurrency control
- `orchestrator/run_manager.gleam` - gen_server for run coordination

**Verification:** All state transition tests pass.

### Phase 5: Validation (TDD)

**Test first:**
```gleam
// test/validation/runner_test.gleam
pub fn validation_runs_commands_test() {
  let spec = ValidationSpec(commands: [
    #("check", Command("uvx", ["ruff", "check", "."])),
    #("test", Command("uv", ["run", "pytest"])),
  ])
  let runner = validation_runner.new(spec)
  let result = runner.run("/tmp/repo")  // v1: shared repo, not worktree
  assert result.commands_run == ["check", "test"]
}

pub fn validation_stops_on_first_failure_test() {
  let spec = ValidationSpec(commands: [
    #("fail", Command("false", [])),
    #("never_runs", Command("echo", ["hi"])),
  ])
  let runner = validation_runner.new(spec)
  let result = runner.run("/tmp/repo")
  assert result.commands_run == ["fail"]
  assert result.status == Fail("fail")
}

// test/validation/evidence_test.gleam
pub fn evidence_redaction_test() {
  let output = CommandOutput(
    stdout: "Password: secret123\nTest passed",
    stderr: "",
    exit_code: 0,
  )
  let evidence = evidence.store("issue-1", "test", output, default_redaction())
  let stored = file.read(evidence.path <> ".stdout")
  assert string.contains(stored, "[REDACTED]")
  assert !string.contains(stored, "secret123")
}
```

**Then implement:**
- `validation/runner.gleam` - Command runner for validation spec
- `validation/evidence.gleam` - Evidence storage with redaction
- `validation/spec.gleam` - ValidationSpec type with per-command env

> **Note:** `validation/worktree.gleam` is **v1.1 only**. v1 runs validation in the shared repo directory.

**Verification:** Validation commands execute correctly in shared repo, evidence is properly redacted.

### Phase 6: CLI + Config (TDD)

**Test first:**
```gleam
// test/config/loader_test.gleam
pub fn load_toml_config_test() {
  let toml = "
[run]
max_agents = 5
idle_timeout = 300

[validation]
commands = [\"check\", \"test\"]
"
  let config = config.load_toml(toml)
  assert config.run.max_agents == Some(5)
  assert config.run.idle_timeout == Some(300)
}

pub fn reject_yaml_config_test() {
  // YAML files are not supported - should error clearly
  let result = config.load("mala.yaml")
  assert result == Error(UnsupportedFormat("YAML not supported, use mala.toml"))
}

// test/cli/run_test.gleam
pub fn cli_run_parses_args_test() {
  let args = ["run", "--profile", "fast", "--max-agents", "2"]
  let cmd = cli.parse(args)
  assert cmd == Run(profile: "fast", max_agents: Some(2))
}
```

**Then implement:**
- `config/loader.gleam` - TOML config loading (YAML not supported)
- `config/types.gleam` - Config types
- `cli.gleam` - CLI argument parsing
- `cli/run.gleam` - Run command implementation
- `cli/status.gleam` - Status command
- `cli/logs.gleam` - Logs command
- `cli/clean.gleam` - Clean command

**Verification:** Config loads correctly, CLI commands work end-to-end.

### Phase 7: Acceptance Testing

**Test first:** Run acceptance test suite.

**Scenarios:**
1. Single issue succeeds first try
2. Issue fails gate, retries with progress, succeeds
3. Issue fails gate without progress → followup
4. Issue times out (idle) → checkpoint and resume
5. Issue times out (session) → abort and followup
6. Multi-issue concurrent run respects max_agents

**Then:** Fix any discrepancies until all acceptance tests pass.

**Verification:** All 6 scenarios pass, benchmark fixture runs successfully.

### TDD Workflow Per Module

1. **Write failing test** - Test the behavior you want
2. **Run test, see it fail** - Confirms test is valid
3. **Write minimal implementation** - Just enough to pass
4. **Run test, see it pass** - Implementation works
5. **Refactor** - Clean up while tests stay green
6. **Repeat** - Next behavior

### Milestone Checkpoints

| Milestone | Criteria | Gate |
|-----------|----------|------|
| M1: Types compile | Phase 0 complete | `gleam build` passes |
| M2: Domain logic | Phase 1 complete | Domain tests pass |
| M3: Infra works | Phase 2 complete | Lock + command tests pass |
| M4: Agent streams | Phase 3 complete | Mock adapter e2e works |
| M5: State machines | Phase 4 complete | All transitions tested |
| M6: Full loop | Phase 5-6 complete | Single issue e2e works |
| M7: Release | Phase 7 complete | Acceptance suite 100% |

### Key Test Categories

- **Unit tests:** JSON codecs, progress detection, state transitions
- **Integration tests:** Command runner, lock server, DETS persistence
- **E2E tests:** Full run with mock adapter, parity scenarios

### Gleam Type Sketches (State Machines)

```gleam
pub type IssueWorkerState {
  Idle(IssueState)
  Running(IssueState)
  Gating(IssueState)
  Reviewing(IssueState)
  Finalizing(IssueState)
  Done(IssueState)
}

pub type IssueWorkerMsg {
  Start(Issue)
  AgentEvent(AgentEvent)
  GateResult(GateResult)
  ReviewResult(ReviewResult)
  Timeout(TimeoutKind)
  Abort(String)
}

pub type IssueState {
  issue: Issue,
  attempt: Int,
  session_id: Option(String),
  last_log_offset: Int,
  gate_retries: Int,
  review_retries: Int,
  agent_profile: String,
  config: Config,
}
```

```gleam
pub type SessionRunnerState {
  Spawn(SessionState)
  Streaming(SessionState)
  Checkpoint(SessionState)
  Done(SessionState)
}

pub type SessionRunnerMsg {
  Start(String)
  AgentEvent(AgentEvent)
  Timeout(TimeoutKind)
  Stop(String)
}
```

### Agent Event Schema (JSON Example)

```json
{
  "type": "tool_use",
  "tool_name": "bash",
  "tool_id": "toolu_01",
  "input": {"command": "uv run pytest"},
  "timestamp": "2026-01-10T12:34:56Z"
}
```

```json
{
  "type": "tool_result",
  "tool_id": "toolu_01",
  "status": "ok",
  "output": "tests passed",
  "timestamp": "2026-01-10T12:35:10Z"
}
```

```json
{
  "type": "final",
  "session_id": "sess_abc",
  "summary": "Implemented feature X",
  "resolution": null,
  "timestamp": "2026-01-10T12:40:00Z"
}
```

### IssueWorker Handler Pseudocode (Gleam Style)

```gleam
pub fn handle(
  state: IssueWorkerState,
  msg: IssueWorkerMsg,
) -> otp.Next(IssueWorkerState) {
  case state, msg {
    Idle(s), Start(issue) ->
      let s = start_session(s, issue)
      otp.next(Running(s))

    Running(s), AgentEvent(Final(final)) ->
      case final.resolution {
        Some(res) -> otp.next(Finalizing(apply_resolution(s, res)))
        None -> otp.next(Gating(capture_log_offset(s)))
      }

    Running(s), Timeout(IdleTimeout) ->
      if supports_resume(s) {
        otp.next(Running(resume_session(s)))
      } else {
        otp.next(Finalizing(mark_followup(s, "idle timeout")))
      }

    Gating(s), GateResult(Pass) ->
      if review_enabled(s) { otp.next(Reviewing(s)) }
      else { otp.next(Finalizing(s)) }

    Gating(s), GateResult(Fail(reason, progress)) ->
      if progress && retries_left_gate(s) {
        otp.next(Running(resume_session(incr_gate_retry(s))))
      } else {
        otp.next(Finalizing(mark_followup(s, reason)))
      }

    Reviewing(s), ReviewResult(Fail(reason, progress)) ->
      if progress && retries_left_review(s) {
        otp.next(Running(resume_session(incr_review_retry(s))))
      } else {
        otp.next(Finalizing(mark_followup(s, reason)))
      }

    Finalizing(s), _ ->
      let _ = close_issue(s)
      otp.next(Done(s))

    Done(_), _ ->
      otp.next(state)

    _, Abort(reason) ->
      otp.next(Finalizing(mark_followup(s, reason)))
  }
}
```

### TOML Decoding Sketch (Gleam)

```gleam
import toml
import gleam/result

pub fn decode_config(src: String) -> Result(Config, ConfigError) {
  use doc <- result.try(toml.parse(src))
  use run <- result.try(toml.get_table(doc, ["run"]))
  let max_agents =
    toml.get_int(run, ["max_agents"])
    |> result.unwrap_or(None)

  Ok(Config(..))
}
```

### TOML Decoding Outline (Full Section Map)

```gleam
pub fn decode_config(src: String) -> Result(Config, ConfigError) {
  use doc <- result.try(toml.parse(src))
  let run = decode_run(doc)
  let paths = decode_paths(doc)
  let agents = decode_agents(doc)
  let provider = decode_issue_provider(doc)
  let locks = decode_locks(doc)
  let validation = decode_validation(doc)
  let review = decode_review(doc)
  let epic = decode_epic(doc)
  let telemetry = decode_telemetry(doc)
  Ok(Config(
    run: run,
    paths: paths,
    agents: agents,
    issue_provider: provider,
    locks: locks,
    validation: validation,
    review: review,
    epic_verification: epic,
    telemetry: telemetry,
  ))
}
```

### GenStatem Skeleton (Gleam + OTP)

```gleam
import gleam/otp
import gleam/otp/statem

pub type State {
  Idle(IssueState)
  Running(IssueState)
  Gating(IssueState)
  Reviewing(IssueState)
  Finalizing(IssueState)
  Done(IssueState)
}

pub type Msg {
  Start(Issue)
  AgentEvent(AgentEvent)
  GateResult(GateResult)
  ReviewResult(ReviewResult)
  Timeout(TimeoutKind)
  Abort(String)
}

pub fn init(init_state: IssueState) -> statem.Init(State, Msg) {
  statem.init(Idle(init_state))
}

pub fn handle(
  state: State,
  msg: Msg,
  data: IssueState,
) -> statem.Next(State, Msg) {
  case state, msg {
    Idle(_), Start(issue) -> {
      let s = start_session(data, issue)
      statem.next(Running(s))
    }

    Running(_), AgentEvent(Final(final)) -> {
      case final.resolution {
        Some(res) -> statem.next(Finalizing(apply_resolution(data, res)))
        None -> statem.next(Gating(capture_log_offset(data)))
      }
    }

    Gating(_), GateResult(Pass) ->
      if review_enabled(data) { statem.next(Reviewing(data)) }
      else { statem.next(Finalizing(data)) }

    _ -> statem.next(state)
  }
}
```

### GenServer Skeleton (RunManager)

```gleam
import gleam/otp
import gleam/otp/actor

pub type RunMsg {
  StartRun
  StopRun
  SpawnIssue(String)
  IssueFinished(String, IssueResult)
}

pub fn init(config: Config) -> actor.Init(RunState, RunMsg) {
  actor.init(RunState(config: config, running: false))
}

pub fn handle(
  msg: RunMsg,
  state: RunState,
) -> actor.Next(RunState, RunMsg) {
  case msg {
    StartRun -> actor.next(state |> start_scheduler)
    StopRun -> actor.next(state |> stop_run)
    SpawnIssue(id) -> actor.next(state |> spawn_issue(id))
    IssueFinished(id, result) -> actor.next(state |> finalize_issue(id, result))
  }
}
```

### Minimal “Hello Run” Flow

1) `mala run` loads config (TOML) and starts `mala_app`.\n
2) `run_manager` starts `scheduler` and `worker_sup`.\n
3) `scheduler` asks `beads_provider` for ready issues.\n
4) For each issue, `worker_sup` spawns `issue_worker` (gen_statem).\n
5) `issue_worker` starts `session_runner` with agent adapter.\n
6) `session_runner` streams `AgentEvent`s to `issue_worker`.\n
7) On completion, `issue_worker` runs gate + review + finalize.\n

## Testing Strategy

### Unit Tests

- State machine transitions: `IssueWorker` and `SessionRunner` state transitions with mocked events
- Validation spec construction: Config parsing, preset loading, command validation
- Progress/resolution detection: Pattern matching, git diff parsing
- Lock server: Acquire/release/TTL/deadlock detection logic

### Integration Tests

- Agent adapter streams: Mock backend processes, verify event parsing and correlation
- Lock server behavior: Concurrent lock requests, deadlock scenarios
- DETS persistence: Write/read/recovery cycles
- Command runner: Process execution, timeout, output capture

### End-to-End Tests

- Run loop with fake agent + fake Beads provider
- Acceptance scenario suite (see below)
- Chaos tests: Random process kills during runs

### Acceptance Scenario Suite

The acceptance scenario suite verifies that the Gleam implementation correctly handles all required behaviors.

> **Note:** This is a clean-break implementation. There is no Python comparison - scenarios are validated against expected outcomes, not against another implementation.

**Test harness:**
1. **Deterministic fixtures:** Test repos with pre-defined issues, agent responses (via mock adapter)
2. **Expected outcomes:** Each scenario defines expected final state (issues closed, commits present, etc.)
3. **Output normalization:** Strip timestamps, PIDs, and other non-deterministic elements before validation

**Test execution flow:**
```bash
# Acceptance scenario runner (pseudocode)
for scenario in scenarios:
    setup_fixture(scenario)

    # Run Gleam mala
    output = run("mala", scenario.args, env=scenario.env)

    # Normalize and validate against expected outcome
    normalized = normalize(output)
    validate(normalized, scenario.expected_outcome)
```

**Non-deterministic handling:**
- **Timestamps:** Replaced with `<TIMESTAMP>` placeholder
- **Session IDs:** Replaced with `<SESSION_ID>` or sequence number
- **Issue ordering:** Sorted by ID before validation (unless testing order-specific behavior)

**Test scenarios (fixed corpus):**

| Scenario | Fixture | Expected Outcome |
|----------|---------|------------------|
| Simple pass | 1 issue, mock agent succeeds | Issue closed, commit with `bd-<id>` present |
| Gate failure + retry | 1 issue, first attempt fails lint | Retry occurs, pass on second, issue closed |
| Multi-issue concurrency | 3 issues, max_agents=2 | All 3 complete, max 2 concurrent at any time |
| Deadlock resolution | 2 issues, circular lock dependency | One aborted (followup), other completes |
| Resume after crash | Interrupt mid-run, restart with --resume | Continues from checkpoint, all issues complete |
| Config validation | Valid TOML config | Config loads, run starts successfully |

**CI/CD integration:**
- Acceptance scenarios run on every PR to `main`
- Full scenario suite runs on merge
- Scenario failures block release

### Performance Regression Tests

- **Benchmark suite:** Fixed scenarios with timing measurements
- **Memory tracking:** Peak RSS per scenario
- **Regression detection:** Compare against baseline, fail if >10% regression

## Success Criteria

Success is measured by **scenario-based acceptance tests** against a fixed benchmark fixture, not proxy metrics.

### Primary Acceptance Criteria (Release Gates)

| Criterion | Test | Pass Condition |
|-----------|------|----------------|
| Acceptance suite | All 6 acceptance scenarios (see Testing Strategy) | 100% pass |
| Crash-resume | Kill mala mid-run, resume with --resume | Completes successfully |
| Deadlock resolution | 2 issues with circular lock dependency | One aborted, other completes |
| Concurrent completion | 10 issues, max_agents=4 | All 10 complete without corruption |
| Config loading | Valid TOML config | Loads and validates successfully |

### Benchmark Fixture Definition

All performance measurements use this **fixed benchmark fixture**.

> **Note:** The counts below are **harness parameters only**, not product goals or acceptance criteria. They define the test environment, not release requirements.

```toml
# benchmark-fixture.toml (HARNESS PARAMETERS ONLY)
issues = 20                   # Test parameter, not a product limit
issue_complexity = "medium"   # ~5 tool calls per issue
max_agents = 4                # Test parameter
mock_agent_delay_ms = 100     # Simulated agent response time
validation_commands = 3       # lint, test, typecheck
events_per_issue = 50         # Approximate, varies by scenario
total_expected_events = 1000  # Approximate
```

**Fixture repository:** `tests/fixtures/benchmark-repo/` containing:
- 20 pre-defined issues in Beads format
- Mock agent responses for deterministic behavior
- Expected output files for comparison

### Performance Targets (Non-Acceptance, Informational Only)

> **These metrics are NOT acceptance criteria and will NOT be used to determine release readiness.**
> The scenario-based acceptance tests (see "Primary Acceptance Criteria") are the sole release gate.

These are tracked for visibility but do not block release:

| Metric | Benchmark Fixture | Target | Measurement |
|--------|-------------------|--------|-------------|
| Startup time | 20 issues | < 5s | `time mala run --dry-run` |
| Peak memory | 4 concurrent issues | < 200 MB total | `rusage` after run |
| Event throughput | 1000 events | > 500/s sustained | disk_log write timing |

**Measurement environment:**
- CPU: 4 cores (CI runner baseline)
- Memory: 8 GB available
- Disk: SSD (for disk_log)

### Quality Gates (Not Primary, But Tracked)

| Metric | Target | Notes |
|--------|--------|-------|
| Integration test pass | 100% | All integration tests pass |
| No regressions | 0 | No new failures in existing tests |

### Coverage Targets (Non-Acceptance, Informational Only)

> **These metrics are NOT acceptance criteria and will NOT be used to determine release readiness.**
> The scenario-based acceptance tests (see "Primary Acceptance Criteria") are the sole release gate.

Coverage is tracked per-package for critical modules but does not block release. Low-value tests that chase coverage numbers are worse than no tests.

| Package | Target | Rationale |
|---------|--------|-----------|
| `infra/lock_server` | > 90% | Safety-critical: lock conflicts, deadlock detection |
| `infra/persistence` | > 90% | Crash recovery depends on correct persistence |
| `orchestrator/issue_worker` | > 85% | State machine transitions must be correct |
| `orchestrator/session_runner` | > 85% | Resume and checkpoint handling |
| `agents/capability` | > 80% | Capability detection affects enforcement mode |
| `config/loader` | > 75% | Config parsing edge cases |
| Other packages | Informational | Tracked but not targeted |

**What matters more than coverage:**
- Behavior-based acceptance tests for critical paths (lock safety, resume after crash)
- Property-based tests for state machine invariants
- Chaos tests for crash recovery scenarios

**Coverage reporting:**
- `gleam test --coverage` generates report
- Report is uploaded to CI artifacts for visibility
- Coverage changes are commented on PRs for awareness, not blocking

### Verification Process

1. **PR checks:** Unit tests + integration tests + 2 parity scenarios (fast)
2. **Merge to main:** Full parity suite + benchmark fixture run
3. **Release:** Full suite + chaos tests + manual smoke test

## Open Questions

Resolved questions (addressed in this plan):
- ~~Should mala.yaml stay YAML or move to TOML?~~ **Decision: TOML only.** No YAML support - clean break.
- ~~Do we need persistent run state beyond disk_log?~~ Yes, DETS for run/issue state, disk_log for events.

Open questions:
- Which agents are required in v1? (Current answer: Claude CLI required, Codex App optional)
- OpenCode CLI: What is the exact output format for event mapping?
- Should lock TTL be per-lock or global? (Current: global via config)

## v1 Scope and Parity Contract

### v1 Backend Validation Rule

**v1 only supports `claude_cli` backend for production runs.**

```gleam
pub fn validate_backend(config: Config) -> Result(Nil, Error) {
  case config.agents.default_backend {
    "claude_cli" -> Ok(Nil)
    "mock" -> Ok(Nil)  // Allowed for testing
    other -> Error(UnsupportedBackend(
      "Backend '" <> other <> "' is not supported in v1. " <>
      "Only 'claude_cli' is available. Other backends ship in v1.1."
    ))
  }
}
```

Other backends (`codex_app`, `amp_cli`, `open_code_cli`) appear in config schema for forward compatibility but are rejected at startup in v1.

### v1 Required Features (Must Ship)

| Category | Feature | Parity With Python |
|----------|---------|-------------------|
| CLI | `run`, `status`, `logs`, `clean` | Full |
| Provider | Beads (bd CLI) | Full |
| Agent | Claude CLI only (others rejected at startup) | Full |
| Isolation | Shared-repo with lock enforcement (hooks) | Full |
| Validation | All commands + evidence | Full |
| **Review** | **Cerberus integration** | **Full** |
| Crash recovery | Resume with --resume | Full |
| Config | TOML only (no YAML) | Breaking change |
| Locking | File-level locks via MCP + PreToolUse hooks | Full |

### v1.1+ Features (Deferred)

| Feature | Reason for Deferral |
|---------|---------------------|
| Codex App adapter | API unconfirmed |
| AMP CLI adapter | Lower priority |
| OpenCode adapter | Lower priority |
| `epic-verify` command | Can ship in v1.1 |
| Per-issue worktrees | Needed for non-hook backends (Codex, AMP, OpenCode) |

### Acceptance Test Alignment

The acceptance scenario suite (6 scenarios) covers **v1 scope only**:
1. Simple pass (Claude CLI, shared-repo)
2. Gate failure + retry
3. Multi-issue concurrency with lock enforcement
4. Deadlock detection and resolution
5. Resume after crash
6. Config loading (TOML)

Epic verification and review loop tests are added in v1.1.

## Notes on Code Examples

All Gleam code blocks in this document are **pseudocode** illustrating the intended API and logic. They do not represent final, compilable Gleam syntax. Actual implementation will use the `gleam_otp` and `gleam_erlang` libraries, which have different APIs than shown here. Key differences:
- Record syntax will use Gleam's actual type constructors
- OTP APIs will use `gleam_otp/actor` or similar wrappers
- Erlang interop will use FFI bindings

The pseudocode is intended to communicate design intent clearly, not to be copied verbatim.

---

## Appendix A: Codex Adapter Spike (Speculative)

> **Status:** This appendix contains speculative design for the Codex adapter.
> Do NOT implement until the discovery gate (see main section) is complete.

### Speculative Transport

HTTP + WebSocket (TBC)
- Launch: HTTP API call to start session
- Communication: WebSocket for streaming events, HTTP for control

### Speculative Event Mapping

| Backend Event | AgentEvent Variant |
|---------------|-------------------|
| `{"event": "message", ...}` | `AssistantText` |
| `{"event": "function_call", ...}` | `ToolUse` |
| `{"event": "function_result", ...}` | `ToolResult` |
| `{"event": "done", ...}` | `Final` |

### Speculative Capability Detection

```gleam
pub fn detect_capabilities(config: CodexConfig) -> Result(Capabilities, Error) {
  // Assumes /capabilities endpoint exists - TO BE CONFIRMED
  case http.get(config.base_url <> "/capabilities", timeout: 5000) {
    Ok(response) -> Ok(parse_capabilities(response.body))
    Error(_) -> Ok(conservative_caps)  // All False, forces worktree isolation
  }
}
```

### Validation Required Before Implementation

1. Does Codex expose an HTTP/WS API or is it stdio-based?
2. What is the actual event schema?
3. Is there a capabilities endpoint?
4. What authentication is required?
5. Does it support session resume?
