# Implementation Plan: Trigger Code Review Terminal Logging

## Context & Goals
- **Spec**: `docs/2026-01-12-code-review-refinement-spec.md`
- Add terminal logging for trigger-based code reviews, which currently run silently with no start/progress/result output
- Enable users and operators to observe code review lifecycle: start, progress (fixer events), and outcome
- Minimal logging: start/end lifecycle events only, no per-file progress
- Follow established MalaEventSink patterns for consistency with existing trigger logging

## Scope & Non-Goals
- **In Scope**
  - Add 5 new MalaEventSink events for trigger code review lifecycle
  - Implement ConsoleEventSink handlers for terminal output
  - Implement BaseEventSink no-op implementations
  - Implement FakeEventSink recording implementations for testing
  - Wire up event emission in RunCoordinator
  - Ensure code review remediation loop emits existing fixer events
  - Unit tests with mocked CumulativeReviewRunner
  - Integration tests in existing validation triggers test file

- **Out of Scope (Non-Goals)**
  - Modifying CumulativeReviewRunner to accept event_sink (events emitted from coordinator only)
  - Adding new fixer events (existing fixer events reused)
  - Per-file progress logging during review execution
  - Per-issue review logging (separate concern)
  - Structured logging or JSON output formats
  - Metrics/observability beyond terminal output
  - New configuration options
  - UI/UX changes beyond terminal output

## Assumptions & Constraints
- CumulativeReviewRunner returns `CumulativeReviewResult` with `status`, `findings`, and `skip_reason` fields
- Existing fixer events (`on_fixer_started`, `on_fixer_completed`, `on_fixer_failed`) are already implemented in ConsoleEventSink
- Events are emitted synchronously; no async considerations needed
- `MalaEventSink` is the canonical interface for all user-facing output
- All concrete event sinks (`ConsoleEventSink`, `NullEventSink`, `FakeEventSink`) inherit from `BaseEventSink`

### Implementation Constraints
- **Single emission boundary**: ALL code review lifecycle events are emitted within `_run_trigger_validation()` only. The `_run_trigger_code_review()` helper returns a result but does NOT emit events. This ensures a single control-flow boundary for the "exactly one end event" invariant.
- **Fixer events per attempt**: Within the remediation loop in `_run_trigger_validation()`, emit `on_fixer_started` before each attempt and `on_fixer_completed/failed` after each attempt. The loop is in the coordinator, not delegated to `_run_code_review_remediation()`.
- **Inline emission**: No helper methods; follow existing trigger event pattern
- **Single end event via try/finally**: Use a status variable initialized to `None`. Set it to `skipped|passed|failed` based on result. In `finally` block, emit the appropriate end event (or `error` if exception). This guarantees exactly one end event.
- **Started after enabled check**: No events when code review is disabled
- **Event sink null check**: Prefer `NullEventSink` in tests over `None`. If `event_sink` can be `None`, check `if self.event_sink is not None:` before emitting. Document why in test setup.
- **Use log() not log_verbose()**: For normal-mode visibility
- Extend `ConsoleEventSink` using existing styling patterns (arrows `→`, checkmarks `✓`, crosses `✗`, `◦` for neutral/skip)

### Testing Constraints
- Must use `FakeEventSink` with `_record()` pattern to verify events are emitted in correct order
- Must mock `CumulativeReviewRunner` in unit tests to avoid running actual heavy reviews
- Must verify exactly-one-end-event invariant
- Must cover all 5 event types
- Integration tests use existing `test_validation_triggers.py` infrastructure

## Integration Analysis

### Existing Mechanisms Considered

| Existing Mechanism | Could Serve Feature? | Decision | Rationale |
|--------------------|---------------------|----------|-----------|
| `MalaEventSink` protocol (`src/core/protocols.py`) | Yes | Extend | Standard protocol for all mala events; add 5 new methods |
| `ConsoleEventSink` (`src/infra/io/console_sink.py`) | Yes | Extend | Existing trigger event handlers; add matching code review handlers |
| `BaseEventSink` (`src/infra/io/base_sink.py`) | Yes | Extend | Universal base class providing no-op defaults |
| `FakeEventSink` (`tests/fakes/event_sink.py`) | Yes | Extend | Uses `_record()` pattern; add recording for new events |
| Existing fixer events | Yes | Reuse | `on_fixer_started/completed/failed` already exist; emit during remediation loop |
| `RunCoordinator` (`src/pipeline/run_coordinator.py`) | Yes | Modify | Central orchestrator that manages the trigger lifecycle |

### Call Graph (Code Review Path)

```
RunCoordinator._run_trigger_validation()
  └── calls _run_trigger_code_review()
        └── calls CumulativeReviewRunner.run_review()
              └── returns CumulativeReviewResult

  └── if remediation needed, calls _run_code_review_remediation()
        └── orchestrates fixer agent loop (in coordinator, NOT in runner)
```

Both `_run_trigger_code_review()` and `_run_code_review_remediation()` are methods on RunCoordinator, so all event emission happens in the coordinator without threading event_sink into lower layers.

### Integration Approach
All new events extend existing infrastructure. The MalaEventSink protocol gets 5 new abstract methods. Each sink implementation (Console, Base, Fake) adds corresponding handlers following their established patterns. Event emission is wired into RunCoordinator methods using the existing `self.event_sink` reference. No new infrastructure needed.

## Prerequisites

- [x] All target files exist (verified in context)
- [x] MalaEventSink protocol supports extension
- [x] ConsoleEventSink trigger event pattern established
- [x] FakeEventSink `_record()` pattern available
- [x] CumulativeReviewResult provides status and blocking count
- [x] No external dependencies or approvals needed

## High-Level Approach

1. **Define Protocol**: Add 5 new abstract methods to MalaEventSink protocol
2. **Implement Sink Handlers**: Add handlers to ConsoleEventSink (formatted output), BaseEventSink (no-op), FakeEventSink (recording)
3. **Wire Emission in Coordinator**: All emissions in `_run_trigger_validation()` with try/finally pattern:
   - After enabled check passes → `started` event
   - Call `_run_trigger_code_review()` → returns `CumulativeReviewResult`
   - Map result to status variable (see precedence rule below)
   - If remediation needed: loop with `fixer_started`/`fixer_completed|failed` per attempt
   - In `finally`: emit exactly one end event based on status variable
4. **Add Tests**: Unit tests verifying event sequences for each scenario; integration tests for real execution flow

## Technical Design

### Architecture

**Single Emission Boundary Pattern** (all events emitted in `_run_trigger_validation()`):

```
RunCoordinator._run_trigger_validation(trigger_type: TriggerType, ...)
        │
        ├─[if code_review disabled]─► return (no events)
        │
        ├─► emit on_trigger_code_review_started(trigger_type.value)
        │
        ├─► try:
        │       result = _run_trigger_code_review()  # returns CumulativeReviewResult, NO events
        │       │
        │       ├─[result.status == "skipped"]─► end_status = "skipped"
        │       │
        │       ├─[result.status == "success" AND no blocking]─► end_status = "passed"
        │       │
        │       ├─[blocking findings AND failure_mode == REMEDIATE]
        │       │         │
        │       │         └─► for attempt in 1..max_retries:
        │       │               ├─► emit on_fixer_started(attempt, max_retries)
        │       │               ├─► run fixer agent
        │       │               ├─► emit on_fixer_completed() or on_fixer_failed()
        │       │               ├─► re-run review
        │       │               └─► if no blocking: end_status = "passed"; break
        │       │
        │       └─[still blocking after loop]─► end_status = "failed"
        │
        └─► finally:
              ├─[exception caught]─► emit on_trigger_code_review_error(...)
              └─[no exception]─► emit end event based on end_status
```

**Trigger Type Source**: `trigger_type` is a `TriggerType` enum value (e.g., `TriggerType.SESSION_END`, `TriggerType.RUN_END`). Use `trigger_type.value` (the string form) for event parameters. This matches existing trigger validation events.

### Data Model

**Existing `CumulativeReviewResult` (no changes needed):**
```python
@dataclass
class CumulativeReviewResult:
    status: Literal["success", "skipped", "failed"]
    findings: tuple[ReviewFinding, ...]
    new_baseline_commit: str | None
    skip_reason: str | None = None  # ← Use this for skipped event reason
```

**Result Status → End Event Precedence Rule:**

| Condition | End Event | Notes |
|-----------|-----------|-------|
| Exception bubbles up to coordinator | `error` | Caught in try/except, emit error with exception message |
| `result.status == "skipped"` | `skipped` | Use `result.skip_reason` as reason text |
| `result.status == "success"` AND `blocking_count == 0` | `passed` | No blocking findings |
| `result.status == "success"` AND `blocking_count > 0` after remediation | `failed` | Blocking findings remain |
| `result.status == "failed"` | `error` | Review execution failed internally (treat as error) |

**Key distinction**: If `CumulativeReviewRunner` catches an exception internally and returns `status="skipped"` with `skip_reason="execution_error: {e}"`, we emit `skipped`. If an exception bubbles up to the coordinator uncaught, we emit `error`. This means `skipped` includes soft failures the runner handled, while `error` means unhandled exceptions.

**Canonical skip reasons** (from `CumulativeReviewRunner`):
- `"empty_diff"` - No changes between baseline and HEAD
- `"baseline_not_reachable"` - Baseline commit not in history (shallow clone)
- `"session_end trigger missing issue_id"` - Required context missing
- `"no commits found for issue {issue_id}"` - Issue has no commits
- `"could not determine baseline commit"` - Baseline resolution failed
- `"execution_error: {e}"` - Runner caught exception and returned gracefully

### API/Interface Design

**New MalaEventSink methods** (added to protocol):

```python
def on_trigger_code_review_started(self, trigger_type: str) -> None:
    """Emitted when trigger code review begins (after enabled check)."""

def on_trigger_code_review_skipped(self, trigger_type: str, reason: str) -> None:
    """Emitted when code review is skipped (e.g., no files to review)."""

def on_trigger_code_review_passed(self, trigger_type: str) -> None:
    """Emitted when code review completes with no blocking findings."""

def on_trigger_code_review_failed(self, trigger_type: str, blocking_count: int) -> None:
    """Emitted when code review completes with blocking findings remaining."""

def on_trigger_code_review_error(self, trigger_type: str, error: str) -> None:
    """Emitted when code review execution fails with an exception."""
```

**ConsoleEventSink output format** (following existing `on_trigger_validation_*` pattern):

Use `log(symbol, message, agent_id="trigger")` helper with consistent styling:

| Event | Symbol | Output Format | Color |
|-------|--------|---------------|-------|
| started | `→` | `[{trigger_type}] code_review_started` | default |
| skipped | `◦` | `[{trigger_type}] code_review_skipped: {reason}` | `Colors.MUTED` |
| passed | `✓` | `[{trigger_type}] code_review_passed` | `Colors.GREEN` |
| failed | `✗` | `[{trigger_type}] code_review_failed ({blocking_count} blocking)` | `Colors.RED` |
| error | `✗` | `[{trigger_type}] code_review_error: {error}` | `Colors.RED` |

**Reference pattern** (from `on_trigger_validation_skipped`):
```python
log("◦", f"{Colors.MUTED}[{trigger_type}] validation_skipped: {reason}{Colors.RESET}", agent_id="trigger")
```

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/core/protocols.py` | Exists | Add 5 new event methods to MalaEventSink protocol |
| `src/infra/io/console_sink.py` | Exists | Implement 5 new event handlers with terminal output |
| `src/infra/io/base_sink.py` | Exists | Add 5 no-op implementations |
| `tests/fakes/event_sink.py` | Exists | Add 5 `_record()` implementations |
| `src/pipeline/run_coordinator.py` | Exists | Emit events at start/end of code review execution |
| `tests/unit/pipeline/test_run_coordinator.py` | Exists | Add tests for event sequences |
| `tests/integration/test_validation_triggers.py` | Exists | Add integration test for code review logging |

## Risks, Edge Cases & Breaking Changes

### Risks
- **Double-logging risk**: Remediation loop has multiple exit points; mitigate with single end-event emission point using status variable
- Low risk overall: All changes are additive (new events only), no modification to existing behavior

### Edge Cases & Failure Modes
- **Code review disabled**: No events emitted; `started` only fires after enabled check passes
- **Empty diff (skipped)**: Emit `skipped` with reason explaining why (not silent)
- **Zero blocking findings**: Emit `passed`
- **Remediation succeeds**: Fixer events during loop, then `passed` at end
- **Remediation exhausts retries**: Fixer events during loop, then `failed` with remaining count
- **Exception during review**: Emit `error` event, re-raise exception
- **Exception before started**: No `started` emitted, no `error` needed (enabled check failure)
- **Orchestrator crash after started**: User sees "started" without "finished" - mitigated by try/except emitting `error`

### Breaking Changes & Compatibility
- **Potential Breaking Changes**:
  - Adding abstract methods to MalaEventSink protocol requires all implementations to add new methods

- **Mitigations**:
  - BaseEventSink provides no-op defaults; subclasses inheriting from it get automatic compliance
  - FakeEventSink updated simultaneously; tests continue working
  - Any external MalaEventSink implementations would need updating (unlikely in this codebase)

## Testing & Validation Strategy

- **Unit Tests** (`tests/unit/pipeline/test_run_coordinator.py`)
  - Mock CumulativeReviewRunner to return controlled results
  - Test scenarios:
    1. **Enabled with passing review**: started → passed
    2. **Enabled with failing review**: started → failed
    3. **Enabled with empty diff**: started → skipped
    4. **Enabled with exception**: started → error
    5. **Disabled/unconfigured**: no events
    6. **Remediation loop succeeds**: started → fixer events → passed
    7. **Remediation loop exhausted**: started → fixer events → failed
  - Verify exactly-one-end-event invariant for each flow

- **Integration Tests** (`tests/integration/test_validation_triggers.py`)
  - End-to-end test with real CumulativeReviewRunner (mocked AI responses)
  - Verify FakeEventSink captures expected event sequence
  - Test with validation trigger that includes code review

- **Regression Tests**
  - Existing trigger validation tests continue passing
  - Existing fixer event tests unchanged

- **Manual Verification**
  - Run mala with trigger code review enabled, observe terminal output
  - Verify output formatting matches other trigger events

- **Monitoring / Observability**
  - Terminal output provides immediate visibility
  - No additional metrics in scope

### Acceptance Criteria Coverage

| Spec AC | Covered By |
|---------|------------|
| R1: Emit start event before trigger code review | RunCoordinator wiring; unit test for started event |
| R2: Exactly one end event (passed/failed/skipped/error) | Status variable pattern; unit tests verify invariant |
| R3: Emit fixer events during remediation loop | RunCoordinator wiring; unit test for fixer event flow |
| R4: Emit error event on execution failure | Exception handling in coordinator; unit test for error flow |

## Open Questions

None — All implementation details resolved:
- Event emission: Inline (no helper methods) - follows existing pattern
- Started event timing: After enabled check (no events when disabled)
- Thread-safety: Not a concern - events are emitted synchronously
- Test approach: Mock CumulativeReviewRunner for unit tests
- Integration tests: Yes, add to test_validation_triggers.py
- Double-logging: Mitigate with single emission point using status variable

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
