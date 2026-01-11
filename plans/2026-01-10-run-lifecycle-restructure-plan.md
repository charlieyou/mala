# Implementation Plan: Run Lifecycle Restructure

## Context & Goals
- **Spec**: `/home/cyou/mala/docs/2026-01-10-run-lifecycle-restructure-spec.md`
- Move `session_end` trigger from once-at-run-end to per-issue (after gate, before review)
- Merge `global_validation` and `run_end` into a single unified run-level stage
- Remove `RunCoordinator.run_validation()` and reject legacy `global_validation_commands` key

## Scope & Non-Goals
- **In Scope**
  - R1: Per-issue session_end placement (after gate pass, before review)
  - R2: Skip session_end on gate failure
  - R3: Unified run-level stage (merge global_validation into run_end)
  - R4: Reject legacy global_validation_commands key with migration error
  - R5: SessionEndResult evidence for review
  - R6: Allow review-only run_end
  - R7: run_end fire_on behavior
  - R8: Remove RunCoordinator.run_validation
  - R9: Session_end failure_mode and remediation
  - R10: Session_end timeout/interrupt behavior
  - R11: Session_end code review baseline (issue.base_sha)
  - R12: run_end failure_mode gating for code review
- **Out of Scope (Non-Goals)**
  - Changing per-issue gate logic or gate command configuration
  - Modifying epic_completion or periodic trigger behavior
  - Adding new trigger types
  - Changing fixer agent remediation behavior (beyond scoping session_end remediation to issue)

## Assumptions & Constraints

### Implementation Constraints
- Session_end blocks only its own issue's progression; other issues continue concurrently (per spec: "same per-issue executor sequentially")
- `issue.base_sha` captured per-issue at worktree creation (each issue runs in its own git worktree/branch), stored in IssueResult, immutable for that issue
- Session_end uses separate `SessionEndRetryState` (not shared with gate retry) for independent tracking

### Run Abort Contract (failure_mode=abort)
**Authoritative cancellation mechanism:** When session_end `failure_mode=abort` triggers, the callback sets a shared `abort_event: asyncio.Event` flag on the coordinator and returns immediately with `status=aborted`.

**Behavior for concurrent issues:**
- In-flight issues: Receive cancellation signal via shared `abort_event`; complete current command (not hard-killed mid-subprocess); proceed to review with `status=interrupted, reason=run_aborted`; finalize stage still runs for state persistence
- Pending issues (not yet started): Skipped entirely, not spawned
- Already-finalized issues: Unaffected

**Stages skipped:**
- Per-issue review/finalize: Still run (for state persistence and cleanup)
- run_end trigger: Skipped entirely, logs `[trigger] run_end skipped: reason=run_aborted`

**External SIGINT behavior:** Same as abort—sets `abort_event`, in-flight issues complete current command and finalize

### Testing Constraints
- Full pyramid + scenarios: unit tests for all new classes, integration tests for lifecycle transitions, scenario tests for complex flows (timeout, remediation exhausted, run abort)
- Integration tests exercise full orchestrator flow with realistic scenarios

## Prerequisites
- [ ] Review spec R1-R12 requirements and verification criteria
- [ ] Understand existing trigger queue pattern in run_coordinator.py
- [ ] Understand lifecycle state machine in lifecycle.py
- [ ] Understand gate retry loop pattern in gate_runner.py

## High-Level Approach

The restructure adds session_end as a new effect/state in the per-issue lifecycle state machine (in `agent_session_runner._run_lifecycle_loop()`), executing after gate passes and before review. This keeps per-issue logic together and reuses existing lifecycle patterns. Session_end has its own `SessionEndRetryState` for independent retry tracking.

For run-level consolidation, we remove `RunCoordinator.run_validation()` and the separate global_validation call in orchestrator, unifying all run-level validation into the existing `run_end` trigger. Config validation will hard-reject `global_validation_commands` with a migration error message.

**Key decisions:**
- Session_end executes in `agent_session_runner._run_lifecycle_loop()` after gate passes
- `SessionEndResult` passed to review via lifecycle context (similar to gate result)
- Session_end uses its own remediation loop (separate from trigger remediation) for per-issue isolation
- `issue.base_sha` stored in IssueResult, captured at workspace checkout

## Technical Design

### Architecture
**Per-issue lifecycle with session_end:**
- Current: agent → gate (retry) → review → finalize
- Target: agent → gate (retry) → session_end (retry) → review → finalize

**State machine changes to lifecycle.py:**
- New states: `RUNNING_SESSION_END`, `SESSION_END_REMEDIATING` (if using remediation)
- New effect: `Effect.RUN_SESSION_END`
- Transitions:
  - gate passed → `RUNNING_SESSION_END`
  - session_end passed/failed/timeout/interrupted → `RUNNING_REVIEW` (or `SUCCESS` if no review)
  - session_end remediate + can retry → `SESSION_END_REMEDIATING` → `RUNNING_SESSION_END`

**Session_end execution lives in agent_session_runner:**
- Add `Effect.RUN_SESSION_END` handling in `_run_lifecycle_loop()`
- New callback: `on_session_end_check` similar to `on_gate_check`
- Session_end result stored in lifecycle context, passed to review

### Data Model

**SessionEndResult** (new dataclass in `src/domain/validation/config.py` or new file):
```python
@dataclass
class SessionEndResult:
    status: Literal["pass", "fail", "timeout", "interrupted", "skipped"]
    started_at: datetime | None  # null when skipped
    finished_at: datetime | None  # null when skipped
    commands: list[CommandResult]
    code_review_result: CodeReviewResult | None
    reason: str | None  # e.g., "gate_failed", "not_configured", "max_retries_exhausted"
```

**SessionEndRetryState** (new dataclass):
```python
@dataclass
class SessionEndRetryState:
    attempt: int = 1
    max_retries: int = 0
    log_offset: int = 0  # For scoping command output
    previous_commit_hash: str | None = None  # For no-progress detection if needed
```

**IssueResult additions** (existing file: `src/pipeline/issue_result.py`):
```python
@dataclass
class IssueResult:
    # existing fields...
    base_sha: str | None = None              # captured at workspace checkout, immutable
    session_end_result: SessionEndResult | None = None  # populated after session_end completes
```
- `base_sha` captured at workspace checkout in `orchestrator.run_implementer()` before agent work starts
- Immutable across retries/remediation for that issue
- `session_end_result` stored after session_end execution completes

### API/Interface Design

**New Callbacks (in SessionCallbacks):**
```python
on_session_end_check: Callable[
    [str, Path, SessionEndRetryState],  # issue_id, log_path, retry_state
    Awaitable[SessionEndResult]
]
```
- Follows same pattern as `on_gate_check`
- Session_end config passed through LifecycleContext (consistent with gate)
- SessionEndResult stored in lifecycle context, passed to review stage

**Session_end callback implementation pattern:**
1. Check if session_end configured; if not, return `SessionEndResult(status="skipped", reason="not_configured")`
2. Execute commands sequentially with `asyncio.timeout()`
3. If all pass, optionally run code_review
4. If fail and `failure_mode=remediate`:
   - Call fixer agent (via injected `run_coordinator` reference)
   - Re-run commands (up to `1 + max_retries` total attempts)
   - If exhausted, return `SessionEndResult(status="fail", reason="max_retries_exhausted")`
5. If fail and `failure_mode=abort`: signal run abort, return result
6. If fail and `failure_mode=continue`: return result, proceed to review

**SessionEndResult passing to review:**
- Add `session_end_result: SessionEndResult | None` field to review input context
- Review prompt includes session_end evidence when available
- Review does NOT auto-fail on session_end failure (informational only, per spec R5)

**Remediation infrastructure (layering clarification):**
Session_end remediation uses a narrow `FixerInterface` protocol to avoid tight coupling:

```python
class FixerInterface(Protocol):
    async def run_fixer(self, failure_output: str, issue_id: str) -> FixerResult: ...
```

- `SessionCallbackFactory` receives a `FixerInterface` implementation (adapter over `run_coordinator._run_fixer_agent()`)
- Session_end callback calls `fixer.run_fixer()` without direct coordinator access
- This avoids circular dependencies: session_runner → callback → fixer_interface (not coordinator)
- Existing `run_coordinator._run_fixer_agent()` is refactored to implement `FixerInterface`

**Config validation change** (`src/domain/validation/config_loader.py`):
```python
if "global_validation_commands" in config:
    raise ConfigValidationError(
        "global_validation_commands is deprecated. Migrate to validation_triggers.run_end:\n\n"
        "validation_triggers:\n"
        "  run_end:\n"
        "    failure_mode: continue\n"
        "    commands:\n"
        "      - ref: test\n"
        "      - ref: lint\n"
    )
```

### File Impact Summary

| Path | Status | Change Type |
|------|--------|-------------|
| `src/orchestration/orchestrator.py` | Exists | Remove `_fire_session_end_trigger()` from run end; remove global_validation call; capture base_sha |
| `src/pipeline/run_coordinator.py` | Exists | Remove `run_validation()`; expose fixer agent for session_end reuse |
| `src/pipeline/agent_session_runner.py` | Exists | Add `Effect.RUN_SESSION_END` handling in lifecycle loop |
| `src/domain/lifecycle.py` | Exists | Add `RUNNING_SESSION_END` state, `Effect.RUN_SESSION_END`, transitions |
| `src/pipeline/session_end_result.py` | **New** | `SessionEndResult` and `SessionEndRetryState` dataclasses |
| `src/domain/validation/config_loader.py` | Exists | Reject `global_validation_commands` key with migration error |
| `src/pipeline/session_callback_factory.py` | Exists | Add `on_session_end_check` callback factory |
| `src/infra/io/event_sink.py` | Exists | Add session_end started/completed/skipped events |
| `src/pipeline/lifecycle_effect_handler.py` | Exists | Add session_end effect handling |
| `src/pipeline/issue_result.py` | Exists | Add `base_sha: str | None` and `session_end_result: SessionEndResult | None` fields |
| `src/pipeline/review_runner.py` | Exists | Accept `session_end_result` in review context |
| `tests/unit/pipeline/test_session_end.py` | **New** | Unit tests for session_end types and logic |
| `tests/unit/pipeline/test_session_end_lifecycle.py` | **New** | Unit tests for lifecycle state transitions |
| `tests/integration/test_run_lifecycle_restructure.py` | **New** | Integration tests for full lifecycle changes |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes

### Timeout/Interrupt Handling (Detailed)

**What is timed:** The entire session_end execution (all commands + code_review) is wrapped in a single `asyncio.timeout()`. Individual commands have their own per-command timeout (existing behavior).

**Timeout behavior:**
- `asyncio.timeout()` cancels the awaited task when session_end timeout expires
- Subprocess commands receive SIGTERM via process group signal forwarding
- Partial command output is NOT recorded in `SessionEndResult.commands` (discarded for consistency per spec R10)
- Partial output is available in logs only for debugging
- Result: `SessionEndResult(status="timeout", commands=[], reason="session_end_timeout")`

**SIGINT/interrupt behavior:**
- SIGINT sets `abort_event` flag; session_end checks flag between commands
- Current command is allowed to complete (subprocess finishes naturally or times out)
- Completed command results ARE recorded (only in-progress commands are lost)
- Result: `SessionEndResult(status="interrupted", commands=[completed], reason="SIGINT received")`

**Rationale for discarding partial results on timeout:** Spec R10 explicitly states "partial results discarded for consistency" to ensure `SessionEndResult.commands` always represents complete command executions, simplifying review evidence interpretation.

| Scenario | Expected Behavior | Test Coverage |
|----------|-------------------|---------------|
| Session_end timeout mid-command | `asyncio.timeout()` cancels task; `status=timeout, commands=[]`; proceed to review | Scenario test |
| Session_end interrupted (SIGINT) | Complete current command; record completed; `status=interrupted`; proceed to review | Scenario test |
| Run abort during session_end | In-flight session_end completes current command; issue finalizes as failed with `reason=run_aborted`; run_end skipped | Scenario test |
| Remediation exhausted | After `1 + max_retries` failures: `status=fail, reason="max_retries_exhausted"`; proceed to review | Unit + integration test |
| Gate fails | Session_end skipped with `status=skipped, reason="gate_failed"` | Unit test |
| No session_end configured | Session_end skipped with `status=skipped, reason="not_configured"` | Unit test |
| Empty commands + code_review enabled | Valid config per R6; execute code_review only; `commands: []` in result | Integration test |
| All issues fail + `fire_on: success` | run_end logs `skipped: reason=fire_on_not_met` | Integration test |
| Mixed outcomes + `fire_on: success` | run_end fires (at least one success meets condition) | Integration test |

### run_end fire_on Semantics (Explicit)

Per spec R7, `fire_on` uses **"any match"** semantics (not "all match"):
- `fire_on: success` → fires if `success_count > 0` (at least one issue succeeded)
- `fire_on: failure` → fires if `failure_count > 0` (at least one issue failed)
- `fire_on: both` → always fires (equivalent to `success_count + failure_count > 0`)

**Truth table (from spec):**
| success_count | failure_count | fire_on: success | fire_on: failure | fire_on: both |
|---------------|---------------|------------------|------------------|---------------|
| 0 | N (>0) | skip | fire | fire |
| N (>0) | 0 | fire | skip | fire |
| N (>0) | M (>0) | fire | fire | fire |

This is mandated by the spec and must be encoded in integration tests.

| Scenario | Expected Behavior | Test Coverage |
|----------|-------------------|---------------|
| Session_end pass → review | `SessionEndResult` in review context with `status=pass` | Integration test |
| Session_end fail → review | `SessionEndResult` in review context with `status=fail`; review does NOT auto-fail | Integration test |

### Breaking Changes & Compatibility
- **Breaking**: `global_validation_commands` config key now rejected (hard fail, no deprecation period per spec R4)
- **Breaking**: `session_end` trigger now fires per-issue instead of once at run end (semantic change)
- **Mitigation**: Config validation error includes complete example YAML snippet for migration
- **Mitigation**: Error message surfaced in CLI output AND logged for visibility

## Testing & Validation Strategy

### Unit Tests
- `test_session_end.py`: SessionEndResult/SessionEndRetryState creation, serialization, field validation, status transitions
- `test_session_end_lifecycle.py`: Lifecycle state machine transitions for `RUNNING_SESSION_END` state and `Effect.RUN_SESSION_END`
- `test_config_rejection.py`: Config rejection for `global_validation_commands` with migration error format verification
- `test_session_end_retry.py`: Retry loop logic, attempt counting (1 + max_retries total), exhaustion handling

### Integration Tests
- End-to-end per-issue session_end execution (happy path: gate pass → session_end pass → review)
- session_end → review evidence passing (SessionEndResult present in review context)
- run_end unified stage (verify no separate global_validation logs)
- Multi-issue concurrent execution with independent session_end progress
- run_end fire_on truth table verification (all combinations from spec)

### Scenario Tests (Full Coverage)
- **Timeout scenario:** Session_end times out mid-command → `status=timeout`, empty commands array, proceeds to review
- **Remediation exhausted:** Session_end fails 1+max_retries times → fixer runs max_retries times → `status=fail, reason=max_retries_exhausted`
- **Run abort:** Run aborts during session_end → issue finalizes with `reason=run_aborted` → run_end logs `skipped: reason=run_aborted`
- **Mixed outcomes:** Some issues pass session_end, some fail → all proceed to review with respective results

### Manual Verification Steps
- Verify log ordering: `session_end started` appears after `gate_passed`, before `review_start` for same issue_id
- Verify config migration error is actionable with correct YAML example (in CLI output and logs)
- Verify no `[run] GATE` or `global_validation` entries in run logs
- Verify `[trigger] run_end skipped: reason=fire_on_not_met` when condition not met

### Monitoring & Observability

**Event naming convention:** Session_end is treated as a **trigger** (not a lifecycle stage) in logs, consistent with existing trigger events.

**Event sink emissions (single source of truth: event_sink.py):**
- `[trigger] session_end started: issue_id=X`
- `[trigger] session_end completed: issue_id=X, result={pass|fail|timeout|interrupted}`
- `[trigger] session_end skipped: issue_id=X, reason={gate_failed|not_configured}`
- `[trigger] run_end started: success_count=N, total_count=M`
- `[trigger] run_end completed: result={pass|fail}`
- `[trigger] run_end skipped: reason={fire_on_not_met|run_aborted}`

**Evidence storage (single source of truth: IssueResult):**
- `IssueResult.session_end_result` is the authoritative storage for session_end evidence
- Lifecycle context receives a reference to this same object (no duplication)
- Review reads from `IssueResult.session_end_result` via review input context

- Config validation errors logged with migration example

### Acceptance Criteria Coverage
| Spec Requirement | Covered By |
|------------------|------------|
| R1: Per-issue session_end placement | Lifecycle state machine; `agent_session_runner` effect handling; integration test for ordering |
| R2: Skip session_end on gate failure | Lifecycle transition logic; unit test for skip reason |
| R3: Unified run-level stage | Remove global_validation; integration test for single run_end log |
| R4: Reject legacy config key | `config_loader.py` validation; unit test for error format |
| R5: SessionEndResult evidence for review | `review_runner.py` context; integration test for evidence presence |
| R6: Review-only run_end | Config validation allows empty commands; integration test |
| R7: run_end fire_on behavior | Existing run_end trigger logic; integration tests for truth table |
| R8: Remove RunCoordinator.run_validation | Delete method; verify no `[run] GATE` logs |
| R9: Session_end failure_mode/remediation | Callback implementation; retry loop tests; exhaustion scenario |
| R10: Timeout/interrupt behavior | `asyncio.timeout()` wrapper; scenario tests |
| R11: Code review baseline (base_sha) | Capture at checkout; pass to `CumulativeReviewRunner` |
| R12: run_end failure_mode gating | Existing run_end logic; integration test |

## Decisions Made
- Session_end executes in `agent_session_runner._run_lifecycle_loop()` (not orchestrator callback)
- `SessionEndResult` and `SessionEndRetryState` live in new `src/pipeline/session_end_result.py`
- Session_end uses separate retry state (not shared with gate)
- Remediation uses `FixerInterface` protocol (adapter over `run_coordinator._run_fixer_agent()`) to avoid tight coupling
- Reuse `CumulativeReviewRunner` for session_end code_review with `base_sha..HEAD` range
- Use `asyncio.timeout()` for overall session_end timeout; per-command timeouts remain separate
- Full pyramid testing with all scenario tests (timeout, remediation exhausted, run abort, mixed outcomes)
- Event sink follows trigger pattern: `[trigger] session_end started: issue_id=X`
- Evidence stored in `IssueResult.session_end_result` (single source of truth)
- Run abort uses shared `abort_event: asyncio.Event` as authoritative cancellation mechanism
- Each issue runs in its own worktree; `base_sha` captured per-issue at worktree creation
- `fire_on` uses "any match" semantics per spec (success_count > 0, not all_success)

## Open Questions
(All resolved during interview)

## Implementation Phases

**Phase order: Run-level consolidation first** (simplifies codebase before adding session_end)

### Phase 1: Run-level Consolidation
**Deliverable:** Unified run-level validation via run_end trigger only
- Remove `RunCoordinator.run_validation()` and global_validation orchestrator calls
- Add config validation rejecting `global_validation_commands` with migration error
- Verify existing run_end trigger handles all run-level validation needs

### Phase 2: Core Infrastructure
**Deliverable:** New types and interfaces for session_end execution
- `SessionEndResult`, `SessionEndRetryState` dataclasses
- `FixerInterface` protocol for remediation
- `IssueResult.base_sha` and `IssueResult.session_end_result` fields
- Lifecycle states (`RUNNING_SESSION_END`, `Effect.RUN_SESSION_END`)
- Event sink methods for session_end events

### Phase 3: Per-Issue Session_end Execution
**Deliverable:** Session_end integrated into per-issue lifecycle
- `on_session_end_check` callback implementation
- Effect handling in `agent_session_runner._run_lifecycle_loop()`
- Command execution with timeout, remediation loop, code_review integration
- `base_sha` capture at worktree creation

### Phase 4: Review Integration
**Deliverable:** Session_end evidence available to review
- `session_end_result` in review input context
- Review prompt includes evidence; does not auto-fail on session_end failure

### Phase 5: Testing & Validation
**Deliverable:** Full test coverage per Testing & Validation Strategy
- Unit, integration, and scenario tests
- Manual verification of log ordering and migration errors

## Next Steps
After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
