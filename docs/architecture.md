# Architecture

**mala** (Multi-Agent Loop Architecture) is an orchestrator for processing issues in parallel using Claude agents. This document describes its layered architecture, module responsibilities, key flows, and design decisions.

> **Source of truth**: Configuration values (thresholds, contracts) are documented here for convenience, but the canonical source is `pyproject.toml` and code under `src/domain/validation`.

## Executive Summary

- The codebase is a **layered, protocol-driven architecture** with explicit boundaries: `cli -> orchestration -> pipeline -> domain -> infra -> core`.
- Orchestration is split into **run-level coordination**, **per-issue session execution**, **quality gate**, and **external review** stages.
- Validation is **spec-driven** and uses explicit inputs/outputs, worktrees, and evidence parsing from JSONL logs.
- Infrastructure code (clients, IO, hooks, tools) is isolated behind Protocols to keep core and domain logic testable.

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                           CLI Layer                             │
│                         (src/cli)                               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     Orchestration Layer                         │
│                    (src/orchestration)                          │
│         Factory, Orchestrator, Session Management               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                       Pipeline Layer                            │
│                      (src/pipeline)                             │
│    Agent Session Runner, Gate Runner, Review Runner             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                        Domain Layer                             │
│                       (src/domain)                              │
│       Lifecycle, Quality Gate, Validation, Prompts              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                      Infrastructure Layer                       │
│                        (src/infra)                              │
│   Clients, I/O, Tools, Hooks, Telemetry, Git Utils              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                         Core Layer                              │
│                        (src/core)                               │
│              Models, Protocols, Log Events                      │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Dependencies

Import boundaries are **enforced by import-linter** with 10 contracts. The dependency graph follows:

| Layer | Can Import |
|-------|------------|
| `cli` | all lower layers (typically uses `orchestration`, `infra`) |
| `orchestration` | `pipeline`, `domain`, `infra`, `core` |
| `pipeline` | `domain`, `infra`, `core` |
| `domain` | `infra`, `core` |
| `infra` | `core` |
| `core` | (none - leaf layer) |

External dependencies are also constrained:
- **SDK packages** (anthropic, braintrust): `pipeline`, `domain`, and `core` cannot import SDKs or their wrappers directly; only `cli`, `orchestration`, and `infra` may use them
- **typer**: confined to `cli` (enforced by `Only CLI imports typer` contract)

## Grimp Architecture Snapshot

Dependency scan (grimp) for `src`:

- Packages: `src`
- Modules: 63
- Total imports: 176

Child packages under `src`:
- `src.cli`
- `src.core`
- `src.domain`
- `src.infra`
- `src.orchestration`
- `src.pipeline`

Top fan-out (imports many modules):
- `src.orchestration.orchestrator`
- `src.orchestration.factory`
- `src.pipeline.run_coordinator`
- `src.pipeline.agent_session_runner`

Top fan-in (imported by many modules):
- `src.infra.tools.command_runner`
- `src.domain.validation.spec`
- `src.infra.tools.env`
- `src.core.protocols`
- `src.infra.io.event_sink`
- `src.infra.tools.locking`
- `src.core.models`
- `src.domain.validation.result`

Layer check via grimp: **no illegal dependencies found** for the ordered layers:
```
src.cli -> src.orchestration -> src.pipeline -> src.domain -> src.infra -> src.core
```

## Architecture Diagrams

Layered dependency view (conceptual):

```mermaid
flowchart TD
  CLI[cli] --> ORCH[orchestration]
  ORCH --> PIPE[pipeline]
  PIPE --> DOMAIN[domain]
  DOMAIN --> INFRA[infra]
  INFRA --> CORE[core]
```

Per-issue call graph (main happy path):

```mermaid
flowchart LR
  CLI[cli.py] --> Factory[create_orchestrator]
  Factory --> Orch[MalaOrchestrator.run]
  Orch --> MainLoop[_run_main_loop]
  MainLoop --> Spawn[spawn_agent]
  Spawn --> Session[AgentSessionRunner.run_session]
  Session --> Gate[GateRunner.run_per_issue_gate]
  Gate --> Review[ReviewRunner.run_review]
  Review --> Finalize[_finalize_issue_result]
  Orch --> RunValidate[RunCoordinator.run_validation]
```

Per-issue sequence (orchestration -> gate -> review):

```mermaid
sequenceDiagram
  participant CLI
  participant Orch as MalaOrchestrator
  participant Beads
  participant Session as AgentSessionRunner
  participant Gate as GateRunner/QualityGate
  participant Review as ReviewRunner

  CLI->>Orch: run()
  Orch->>Beads: claim issue
  Orch->>Session: run_session(issue)
  Session-->>Orch: AgentSessionOutput (log_path, session_id, summary)
  Orch->>Gate: run_per_issue_gate(log_path)
  Gate-->>Orch: gate_result + log_offset
  alt review enabled
    Orch->>Review: run_review(commit_sha)
    Review-->>Orch: review result
  end
  Orch->>Beads: close/mark issue
```

Run-level validation + fixer retry:

```mermaid
sequenceDiagram
  participant Orch as MalaOrchestrator
  participant RunCoord as RunCoordinator
  participant Spec as SpecValidationRunner
  participant Fixer as AgentSessionRunner

  Orch->>RunCoord: run_validation()
  RunCoord->>Spec: run_spec(RUN_LEVEL)
  alt validation failed
    RunCoord->>Fixer: run_session(fixer prompt)
    Fixer-->>RunCoord: session output
    RunCoord->>Spec: run_spec(RUN_LEVEL) retry
  end
  RunCoord-->>Orch: passed/failed
```

Epic verification loop:

```mermaid
sequenceDiagram
  participant Orch as MalaOrchestrator
  participant Epic as EpicVerifier
  participant Beads
  participant Model as ClaudeEpicVerificationModel

  Orch->>Epic: verify_and_close_eligible()
  Epic->>Beads: fetch epic + child issues
  Epic->>Epic: compute scoped commits
  Epic->>Model: verify(criteria + commits + specs)
  Model-->>Epic: EpicVerdict
  alt passed
    Epic->>Beads: close epic
  else failed
    Epic->>Beads: create remediation issues + add blockers
  end
```

## Runtime Flow

High-level flow:

1. CLI parses arguments and bootstraps environment (Braintrust, config).
2. `create_orchestrator()` builds dependencies and configuration.
3. `MalaOrchestrator.run()`:
   - Fetches ready issues (Beads).
   - Spawns per-issue agent sessions (parallel).
   - Runs per-issue quality gate.
   - Optionally runs external review.
   - On success, closes issues; on failure, records followup.
4. After all issues, run-level validation (Gate 4) occurs.
5. Epics are verified and closed when all children complete.

Per-issue pipeline sequence:
```
Issue -> AgentSessionRunner -> GateRunner -> ReviewRunner -> finalize
```

Run-level validation uses `RunCoordinator` and `SpecValidationRunner` with `ValidationScope.RUN_LEVEL`, including E2E checks and coverage when configured.

## Package Layout and Responsibilities

```
src/
  cli/              CLI entry points and CLI-only wiring
  orchestration/    Main orchestrator + factory/DI + CLI support
  pipeline/         Pipeline stages for agent sessions, gate, review, run-level validation
  domain/           Business logic: lifecycle, quality gate, validation, prompts
  infra/            External systems, IO, hooks, tools, telemetry
  core/             Minimal shared models, protocols, log event schema
  prompts/          Prompt templates
  scripts/          Shell utilities (locking helpers)
```

### `src/core` — Foundation Layer

Pure data structures and interfaces with **no internal dependencies**.

| Module | Purpose |
|--------|---------|
| `models.py` | Shared dataclasses (IssueResolution, ValidationArtifacts, EpicVerdict, RetryConfig) |
| `protocols.py` | Protocol interfaces (IssueProvider, GateChecker, CodeReviewer, LogProvider, EpicVerificationModel) |
| `log_events.py` | JSONL log schema types and parsing helpers |

### `src/domain` — Business Logic

Orchestration-agnostic business rules.

| Module | Purpose |
|--------|---------|
| `lifecycle.py` | Issue lifecycle state machine and retry policy |
| `quality_gate.py` | Gate checking: commit exists, tests passed, evidence present |
| `prompts.py` | Prompt template loading |
| `validation/` | Spec-based validation pipeline |

Validation subpackage:
- `spec.py` — Build validation specs from change classification
- `spec_runner.py` — Execute spec in worktree
- `spec_executor.py` — Run individual commands with caching
- `coverage.py` — Coverage threshold checking and baseline refresh
- `worktree.py` — Git worktree creation/cleanup
- `e2e.py` — End-to-end fixture repo tests

### `src/infra` — Infrastructure

External integrations and utilities.

| Subpackage | Purpose |
|------------|---------|
| `clients/` | SDK wrappers (Anthropic, Braintrust, Beads, Cerberus) |
| `io/` | Config loading, event sink, console output, log parsing |
| `tools/` | Command runner, file locking, environment helpers |
| `hooks/` | Agent hooks (lint cache, file cache, lock enforcement) |

Key modules:
- `epic_verifier.py` — AI-powered epic acceptance verification
- `issue_manager.py` — Issue filtering, sorting, dependency resolution
- `telemetry.py` — Span abstraction (pure protocol)
- `mcp.py` — MCP server configuration and disallowed tools

### `src/pipeline` — Agent Execution

Pipeline components for running agent sessions.

| Module | Purpose |
|--------|---------|
| `agent_session_runner.py` | Main session loop (SDK stream handling + lifecycle) |
| `gate_runner.py` | Quality gate execution |
| `review_runner.py` | External code review via Cerberus |
| `run_coordinator.py` | Run-level validation and fixer agent |

### `src/orchestration` — Coordination

High-level orchestration of the agent loop.

| Module | Purpose |
|--------|---------|
| `orchestrator.py` | Main loop: claim → spawn → gate → review → close |
| `factory.py` | Dependency injection and orchestrator construction |
| `types.py` | Orchestration-specific types |
| `cli_support.py` | CLI integration helpers |

### `src/cli` — Entry Point

| Module | Purpose |
|--------|---------|
| `cli.py` | Typer commands: `run`, `status`, `clean`, `epic-verify` |
| `main.py` | App entry point |

### `src/prompts` — Prompt Templates

Markdown template files for agent prompts (implementer, fixer, gate followup, etc.). Referenced by `domain.prompts` and used directly by agents.

### `src/scripts` — Utility Scripts

Developer-facing shell scripts bundled with the package. Not part of the core runtime.

## Major Classes and Key Responsibilities

### Orchestration

- `MalaOrchestrator` (`src/orchestration/orchestrator.py`)
  - Central coordinator for parallel issue execution.
  - Owns run lifecycle, metadata tracking, and shutdown/cleanup behavior.
  - Key methods:
    - `run()` / `run_sync()`: public entrypoints for async/sync execution.
    - `_run_main_loop()`: main scheduler loop for spawning/waiting on agents.
    - `spawn_agent()`: claims issues and launches per-issue worker tasks.
    - `run_implementer()`: per-issue pipeline (session -> gate -> review).
    - `_run_quality_gate_async()` / `_run_quality_gate_sync()`: gate checks.
    - `_finalize_issue_result()`: close/mark issue based on outcome.
    - `_check_epic_closure()`: triggers epic verification when eligible.
    - `_finalize_run()`: run-level validation + summary + cleanup.

- `OrchestratorConfig`, `OrchestratorDependencies` (`src/orchestration/types.py`)
  - Split between simple config values and injected dependencies (DI).

- `create_orchestrator()` (`src/orchestration/factory.py`)
  - Factory that assembles dependencies (beads client, gate checker, reviewer, telemetry, event sink).

### Pipeline

- `AgentSessionRunner` (`src/pipeline/agent_session_runner.py`)
  - Wraps Claude SDK session lifecycle with streaming and idle-timeout recovery.
  - Uses tool hooks for locking, dangerous command blocking, and lint cache.
  - Key methods:
    - `run_session()`: main session runner; streams SDK responses, drives lifecycle.
    - `_build_hooks()`: configures PreToolUse hooks (locking, lint cache, safety).
    - `_build_agent_env()`: per-agent env including lock/agent IDs.

- `GateRunner` (`src/pipeline/gate_runner.py`)
  - Runs gate checks using a `GateChecker` protocol.
  - Tracks and applies retry/no-progress logic.
  - Key methods:
    - `run_per_issue_gate()`: synchronous gate execution (used via `to_thread`).
    - `get_cached_spec()`: returns cached per-issue `ValidationSpec`.

- `ReviewRunner` (`src/pipeline/review_runner.py`)
  - Executes external code review via `CodeReviewer` protocol.
  - Handles retry/no-progress checks and session log tracking.
  - Key methods:
    - `run_review()`: run Cerberus review-gate for a commit diff range.
    - `check_no_progress()`: avoids review retries when no new evidence.

- `RunCoordinator` (`src/pipeline/run_coordinator.py`)
  - Performs run-level validation and fixer retries.
  - Key methods:
    - `run_validation()`: run RUN_LEVEL spec; spawns fixer on failure.

### Domain / Policy

- `ImplementerLifecycle` (`src/domain/lifecycle.py`)
  - Pure state machine; defines transitions and effects for gate/review retry policy.
  - Key methods:
    - `start()`: initialize lifecycle at `INITIAL`.
    - `on_messages_complete()`: move from streaming to gate/log wait.
    - `on_log_ready()` / `on_log_timeout()`: log availability handling.
    - `on_gate_result()`: gate pass/fail transitions + retry decisions.
    - `on_review_result()`: review pass/fail transitions + retry decisions.
    - `on_timeout()` / `on_error()`: hard failure paths.

- `QualityGate` (`src/domain/quality_gate.py`)
  - Checks for required commit + evidence of validation commands.
  - Evidence is spec-driven and parsed from JSONL logs.
  - Key methods:
    - `check_with_resolution()`: primary gate entry (commit + evidence + resolution markers).
    - `parse_validation_evidence_with_spec()`: spec-driven evidence extraction.
    - `check_commit_exists()`: verify `bd-<id>` commit in range.
    - `check_no_progress()`: detect unchanged commit + no new evidence.
    - `get_log_end_offset()`: log offset tracking for retry scoping.

- Validation subsystem (`src/domain/validation/*`)
  - `ValidationSpec` defines commands and policies for per-issue vs run-level runs.
  - `SpecValidationRunner` executes commands and assembles results.
  - Worktrees are used to keep validation isolated.
  - Key methods:
    - `build_validation_spec()` (`spec.py`): assemble commands + coverage/E2E policy.
    - `SpecValidationRunner.run_spec()`: run a spec with workspace isolation.
    - `SpecValidationRunner._run_validation_pipeline()`: commands -> coverage -> e2e.

### Infra / External Integrations

- `BeadsClient`: CLI wrapper for issue operations and filtering/sorting.
  - Key methods:
    - `get_ready_async()` / `get_ready_issues_async()`: fetch + filter ready issues.
    - `claim_async()`: claim a task for work.
    - `close_async()` / `mark_needs_followup_async()`: update issue state.
    - `close_eligible_epics_async()`: epic housekeeping at end of run.
    - `get_issue_description_async()`: fetch description for prompting/review.
- `DefaultReviewer`: wrapper for Cerberus review-gate.
- `BraintrustProvider`: tracing integration for SDK activity.
- `EpicVerifier`: runs acceptance verification across child issue commits.
  - Key methods:
    - `verify_and_close_eligible()`: main epic closure loop.
    - `verify_epic()` / `verify_epic_with_options()`: verify a specific epic.
    - `create_remediation_issues()`: open follow-up tasks for unmet criteria.
    - `add_epic_blockers()` / `request_human_review()`: enforcement actions.
- `SessionLogParser` / `FileSystemLogProvider`: parsing and offset tracking for JSONL logs.
  - Key methods:
    - `iter_jsonl_entries()` / `get_log_end_offset()`: parser-level iteration + offsets.
    - `iter_events()` / `get_end_offset()`: provider-level streaming access + offsets.
    - `extract_*()` helpers: tool use, tool result, and assistant text parsing.
- `CommandRunner`: standardized subprocess execution with timeouts and process-group handling.
  - Key methods:
    - `run()` / `run_async()`: sync/async command execution with timeout semantics.
- `MalaEventSink`: event interface for CLI/log sinks (console is default implementation).

## Key Architectural Patterns

### 1) Event Sink

Decouples orchestration from presentation:
```
Orchestrator -> EventSink -> Console/Logs/Telemetry
```

The event sink enables testable orchestration and swappable output formats.

### 2) Layered Architecture (Enforced)

Import-linter contracts prevent layer violations:
- `layers` contract enforces the 6-layer hierarchy
- `forbidden` contracts isolate SDK access and leaf modules
- `independence` contract ensures pipeline modules are acyclic

### 3) Protocol-Based Interfaces

Core protocols define contracts between orchestration and infra:
- `IssueProvider`, `GateChecker`, `CodeReviewer`
- `LogProvider`, `EpicVerificationModel`, `TelemetryProvider`

### 4) Pipeline Decomposition

Orchestrator delegates specialized stages to pipeline runners. Each runner uses explicit input/output dataclasses for clarity and testability.

### 5) State Machine for Session Policy

`ImplementerLifecycle` encapsulates retry logic and effect decisions, separating policy from side effects.

### 6) Spec-Driven Validation and Evidence Parsing

Validation commands are defined in `ValidationSpec`. The quality gate derives evidence requirements from the spec to avoid hardcoded checks.

### 7) Filesystem Locking

Prevents edit conflicts between concurrent agents:
- Atomic hardlink-based locks in `/tmp/mala-locks/`
- Path canonicalization with repo namespace support
- Lock enforcement hook blocks writes to unlocked files (`infra.hooks.locking`)
- Per-run ownership tracking for clean shutdown

### 8) Worktree Validation

Clean-room validation in isolated git worktrees:
```
/tmp/mala-worktrees/{run_id}/{issue_id}/{attempt}/
```

### 9) Telemetry

Agent sessions emit spans through a `TelemetryProvider` abstraction:
- Protocol defined in `infra.telemetry`
- Braintrust-backed implementation in `infra.clients.braintrust_integration`

## Data Flow

```
┌───────────────┐     ┌─────────────────┐     ┌──────────────┐
│  bd ready     │────▶│   Orchestrator  │────▶│  bd close    │
│  (issues)     │     │    main loop    │     │  (on pass)   │
└───────────────┘     └────────┬────────┘     └──────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Agent Session     │
                    │   (Claude Code)     │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼────────┐ ┌─────▼──────┐ ┌──────▼───────┐
     │  Quality Gate   │ │   Review   │ │    Fixer     │
     │  (tests/lint)   │ │ (Cerberus) │ │   (retry)    │
     └─────────────────┘ └────────────┘ └──────────────┘
```

## Extension Points

- **Issue provider**: Replace `BeadsClient` with any implementation of `IssueProvider`.
- **Review system**: Implement `CodeReviewer` for different review systems.
- **Event sink**: Add alternative sinks (JSON logs, telemetry).
- **Telemetry**: Implement a custom `TelemetryProvider` or span class.

## Configuration

| Source | Precedence |
|--------|------------|
| CLI flags | Highest |
| Environment vars | Medium |
| `~/.config/mala/.env` | Lowest |

Key directories:
- `~/.config/mala/logs/` — JSONL session logs
- `~/.config/mala/runs/` — Run metadata
- `/tmp/mala-locks/` — Filesystem locks
- `/tmp/mala-worktrees/` — Validation worktrees

## Testing Strategy

| Category | Marker | Purpose |
|----------|--------|---------|
| Unit | `@pytest.mark.unit` | Fast, isolated tests |
| Integration | `@pytest.mark.integration` | Multi-component tests |
| E2E | `@pytest.mark.e2e` | Full CLI + agent tests |

Coverage threshold: 85% (enforced via the validation pipeline, not default pytest runs)

## Notes on Naming

- The top-level package is `src` (installed as `mala`), so imports reference `src.*` across the codebase.
- `__init__.py` exposes `MalaOrchestrator` via lazy import for reduced import cost.

## Codebase Statistics (Optional Snapshot)

These numbers were previously captured via static analysis; re-run `lizard` and `uv run import-linter` for current values.

| Metric | Value |
|--------|-------|
| Total Files | 63 |
| Architectural Contracts | 10 |

## Future Considerations

1. **Language Support**: Currently Python-only (pytest, ruff, ty). Other languages would need validation spec extensions.
2. **Multi-Repo**: Single-repo focus; multi-repo coordination not implemented.
3. **Complexity Reduction**: `run_session` and validation coverage paths are candidates for decomposition.
