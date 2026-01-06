# Implementation Plan: SIGINT Escalation and Drain-Then-Abort

## Context & Goals
- **Spec**: `plans/2026-01-06-sigint-escalation-spec.md`
- Implement three-stage Ctrl-C behavior: drain (1st) → graceful abort (2nd) → hard abort (3rd)
- Let users stop accepting new work without losing in-flight progress
- Provide predictable escalation for faster shutdown when needed
- Ensure validation subprocesses respond immediately on second Ctrl-C

## Scope & Non-Goals

### In Scope
- Three-stage SIGINT escalation state machine in `MalaOrchestrator`
- Drain mode support in `IssueExecutionCoordinator` (stop spawning, let active finish)
- `CommandRunner.kill_active_process_groups()` static method for SIGKILL
- Exit code precedence: Stage 3 → 130; Stage 2 → 1 if `_validation_failed` else 130; Stage 1 → normal semantics
- New event sink methods for drain/abort escalation messages
- Unit tests for escalation logic, drain behavior, and SIGKILL helper

### Out of Scope (Non-Goals)
- Changing issue assignment or beads semantics
- Guaranteeing commit on first Ctrl-C if agent is stuck
- Adding new CLI flags (behavior is triggered by SIGINT only)
- Full Windows signal support (Stage 2/3 signal forwarding are no-ops on Windows)

## Assumptions & Constraints

- Breaking change is acceptable for internal state machine
- `MalaOrchestrator.run()` must execute in main thread for SIGINT handler installation
- All validation subprocesses use `CommandRunner` (process group coverage guaranteed)
- 5-second escalation window between consecutive Ctrl-C presses
- 10-second grace period for Stage 2 before tasks are abandoned

### Implementation Constraints
- Use `loop.call_soon_threadsafe()` for all nontrivial work from signal handler
- State variables are instance attributes on `MalaOrchestrator`, reset per-run
- Do NOT add new services — extend existing modules only
- Follow existing dataclass patterns for any new config types

### Testing Constraints
- Unit tests must cover escalation timing (time window, count reset)
- Unit tests must verify drain mode behavior (no spawn, waits for active tasks)
- Integration tests for SIGINT forwarding/kill in `CommandRunner`
- Tests that exercise SIGINT handling must run orchestrator in main thread

## Prerequisites
- [x] Spec is complete and reviewed
- [x] No special infrastructure needed
- [x] No feature flags required (behavior is SIGINT-triggered only)

## High-Level Approach

1. **Add SIGINT state machine to `MalaOrchestrator`**: Instance attributes for `_sigint_count`, `_drain_mode_active`, `_abort_mode_active`, `_validation_failed`, `_abort_exit_code`, `_shutdown_requested`. Reset at start of each `run()`.

2. **Refactor `run()` SIGINT handler**: Replace current single-event handler with three-stage escalation handler using `loop.call_soon_threadsafe()` for thread safety.

3. **Add drain mode to `IssueExecutionCoordinator`**: Pass `drain_event` alongside `interrupt_event`. When `drain_event` is set, stop spawning new issues but let active tasks complete.

4. **Add `CommandRunner.kill_active_process_groups()`**: Static method to SIGKILL all tracked pgids.

5. **Add event sink methods**: `on_drain_started`, `on_graceful_abort_started`, `on_hard_abort_started`.

6. **Wire up exit code logic**: Stage 3 always 130; Stage 2 snapshots `_validation_failed` at entry; Stage 1 follows normal semantics.

## Technical Design

### Architecture

**Current flow**:
```
SIGINT → single interrupt_event → _handle_interrupt → abort all → exit 130
```

**New flow**:
```
SIGINT #1 → drain_mode_active=True → drain_event.set() → stop spawning, wait for active
SIGINT #2 → abort_mode_active=True → interrupt_event.set() → cancel tasks, forward SIGINT, exit 130/1
SIGINT #3 → shutdown_requested=True → SIGKILL pgids, cancel run task → exit 130
```

Key architectural decisions:
- `drain_event` and `interrupt_event` are separate events that coexist
- Signal handler only updates simple Python types and schedules work via `call_soon_threadsafe`
- State reset happens at start of each `run()`, not in `__init__`

### Data Model

**New instance attributes on `MalaOrchestrator`** (all reset per-run):
```python
self._sigint_count: int = 0
self._sigint_last_at: float = 0.0
self._drain_mode_active: bool = False
self._abort_mode_active: bool = False
self._abort_exit_code: int = 130  # Snapshot at Stage 2 entry
self._validation_failed: bool = False  # Set by validation callbacks
self._shutdown_requested: bool = False  # Signals run() to terminate
```

**New asyncio.Event**:
- `drain_event`: set on 1st Ctrl-C, signals "stop spawning but let active tasks finish"
- Existing `interrupt_event`: set on 2nd Ctrl-C, signals "cancel active tasks now"

### API/Interface Design

**`CommandRunner` additions**:
```python
@staticmethod
def kill_active_process_groups() -> None:
    """Send SIGKILL to all tracked process groups."""
```

**`MalaEventSink` additions**:
```python
def on_drain_started(self, active_task_count: int) -> None:
    """Called when drain mode starts (1st Ctrl-C)."""

def on_graceful_abort_started(self) -> None:
    """Called when graceful abort starts (2nd Ctrl-C)."""

def on_hard_abort_started(self) -> None:
    """Called when hard abort starts (3rd Ctrl-C)."""
```

**`IssueExecutionCoordinator.run_loop()` signature change**:
```python
async def run_loop(
    self,
    ...,
    drain_event: asyncio.Event | None = None,  # NEW
    interrupt_event: asyncio.Event | None = None,
    ...
) -> RunResult:
```

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/orchestration/orchestrator.py` | Exists | Add SIGINT state machine, drain_event, three-stage handler |
| `src/pipeline/issue_execution_coordinator.py` | Exists | Add drain_event handling in run_loop |
| `src/infra/tools/command_runner.py` | Exists | Add `kill_active_process_groups()` static method |
| `src/core/protocols.py` | Exists | Add new event sink methods to `MalaEventSink` protocol |
| `src/core/models.py` | Exists | No changes expected (may add constants) |
| `src/infra/io/console_sink.py` | Exists | Implement new event sink methods |
| `tests/unit/orchestration/test_orchestrator.py` | Exists | Add SIGINT escalation tests |
| `tests/unit/pipeline/test_issue_execution_coordinator.py` | Exists | Add drain mode tests |
| `tests/unit/infra/tools/test_command_runner.py` | **New** | Tests for `kill_active_process_groups` |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes
- **SIGINT during run-level validation**: Stage 1 allows completion; Stage 2 forwards SIGINT; Stage 3 kills
- **Watch mode**: Drain stops polling/spawning but keeps running tasks; graceful abort behaves as today
- **No active tasks at Stage 1**: Immediate exit with success (0) if no pending validation
- **CommandRunner has no active process groups**: Escalation proceeds without error
- **Kill failures (ProcessLookupError, PermissionError)**: No logging needed (expected conditions)
- **Stage 2 grace period timeout**: After 10s, exit anyway (Stage 3 available for force kill)

### Breaking Changes & Compatibility
- **Internal state**: New instance attributes on `MalaOrchestrator` (internal, not public API)
- **Coordinator signature**: `run_loop()` gains `drain_event` parameter (optional, backward compatible)
- **Event sink**: New methods added to `MalaEventSink` protocol (existing implementations need updates)

## Testing & Validation Strategy

### Unit Tests
- **`test_orchestrator.py`**:
  - Test escalation timing (5s window, count reset when idle)
  - Test state transitions: idle → drain → abort → hard abort
  - Test exit code precedence for each stage
  - Test `_validation_failed` flag snapshot at Stage 2 entry

- **`test_issue_execution_coordinator.py`**:
  - Test drain mode: no new spawns when `drain_event` set
  - Test drain completion: returns when active tasks finish (no cancellation)
  - Test drain + interrupt: drain_event then interrupt_event escalates correctly

- **`test_command_runner.py`** (New):
  - Test `kill_active_process_groups()` sends SIGKILL to tracked pgids
  - Test no-op on Windows
  - Test clears `_SIGINT_FORWARD_PGIDS` to avoid redundant signals

### Integration Tests
- E2E test simulating drain → abort by toggling events in orchestrator

### Regression Tests
- Ensure existing `interrupt_event` behavior for single SIGINT still works (backward compat)
- Ensure watch mode polling/idle behavior unchanged

### Manual Verification
1. Run `mala run` and press Ctrl-C once → verify drain message, active tasks complete
2. Press Ctrl-C again → verify abort, subprocesses receive SIGINT
3. Press Ctrl-C third time → verify force kill, immediate exit 130

### Acceptance Criteria Coverage

| Spec Requirement | Covered By |
|------------------|------------|
| Stage 1 stops spawning, lets active finish | Coordinator drain_event handling |
| Stage 2 cancels tasks, forwards SIGINT | Orchestrator interrupt_event + `forward_sigint()` |
| Stage 3 SIGKILL all process groups | `CommandRunner.kill_active_process_groups()` |
| Exit code 130 for Stage 3 | Orchestrator `_shutdown_requested` → 130 |
| Exit code 1/130 for Stage 2 based on validation | `_validation_failed` snapshot at Stage 2 entry |
| 5s escalation window | Signal handler timing logic |
| 10s grace period Stage 2 | [TBD: implementation detail] |
| Observability (drain/abort messages) | New event sink methods |

## Open Questions

- [TBD: 10s grace period implementation]: Should the grace period be a constant in `orchestrator.py` or configurable? Spec says 10s. Likely hardcode as constant.
- [TBD: drain_event naming]: Should we rename `WatchState` to something more generic since it now tracks drain state too? Spec says "out of scope (cosmetic)".

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
