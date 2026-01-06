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
- Exit code precedence: Stage 3 → 130; Stage 2 → 1 if `_validation_failed` else 130; Stage 1 → normal semantics (0 or 1)
- New event sink methods for drain/abort escalation observability
- Unit tests for escalation logic, drain behavior, and SIGKILL helper
- Integration tests for real SIGINT handling (POSIX-only)

### Out of Scope (Non-Goals)
- Handling signals other than SIGINT (e.g., SIGTERM, SIGQUIT default behavior remains)
- Changing issue assignment or beads semantics
- Guaranteeing commit on first Ctrl-C if agent is stuck
- Adding new CLI flags (behavior is triggered by SIGINT only)
- Complex TUI/UI implementation (logging via existing sinks only)
- Persistence of interruption state across runs
- Full Windows signal support (Stage 2/3 signal forwarding are no-ops on Windows)

## Assumptions & Constraints

- The environment is POSIX-compliant (regarding process groups and signals)
- `asyncio` is the primary concurrency model
- Breaking change acceptable for internal state machine (new instance attributes on `MalaOrchestrator`)
- External APIs remain backward compatible (coordinator signature gains optional `drain_event`)
- `MalaOrchestrator.run()` must execute in main thread for SIGINT handler installation
- All validation subprocesses use `CommandRunner` (process group coverage guaranteed)
- Existing tool implementations respect SIGINT forwarding or process group signals

### Implementation Constraints
- **Signal Handling**: Use `signal.signal()` with `loop.call_soon_threadsafe()` for thread safety and compatibility
- Signal handler only updates simple Python types and schedules work onto event loop
- State variables are instance attributes on `MalaOrchestrator`, reset per-run
- Hardcode `ABORT_GRACE_SECONDS = 10.0` as module constant (not configurable)
- Hardcode `ESCALATION_WINDOW_SECONDS = 5.0` for Ctrl-C timing window
- Do NOT add new services — extend existing modules only
- Extract handler logic into private `_handle_sigint()` method for testability

### Testing Constraints
- **Unit tests**: Cover escalation timing, state transitions, exit code logic (mocked signals)
- **Integration tests**: Subprocess-based tests for real SIGINT forwarding/kill (POSIX-only)
- Tests that exercise SIGINT handling must run orchestrator in main thread
- Integration tests may be skipped in CI if signals are unreliable
- Signal tests can be timing-sensitive; use generous timeouts and explicit synchronization primitives

## Prerequisites
- [x] Spec is complete and reviewed
- [x] No special infrastructure needed
- [x] No feature flags required (behavior is SIGINT-triggered only)
- [x] Existing `CommandRunner` accurately tracks process groups via `_SIGINT_FORWARD_PGIDS`

## High-Level Approach

The implementation follows the spec's three-stage escalation design:

1. **Foundation (Protocols & Tools)**: Update `MalaEventSink` with new lifecycle methods (`on_drain_started`, `on_abort_started`, `on_force_abort`) and implement `kill_active_process_groups()` in `CommandRunner`.

2. **Add SIGINT state machine to `MalaOrchestrator`**: New instance attributes (`_sigint_count`, `_drain_mode_active`, `_abort_mode_active`, `_validation_failed`, `_abort_exit_code`, `_shutdown_requested`, `_run_task`) reset at start of each `run()`.

3. **Refactor `run()` SIGINT handler**: Replace current single-event handler with three-stage handler using `signal.signal()` + `loop.call_soon_threadsafe()`. Extract core logic to `_handle_sigint()` for testability.

4. **Add drain mode to `IssueExecutionCoordinator`**: Pass separate `drain_event` alongside existing `interrupt_event`. When `drain_event` is set, stop spawning new issues but let active tasks complete. Drain completion triggers final validation if thresholds are met.

5. **Wire up exit code logic**: Stage 3 always 130; Stage 2 snapshots `_validation_failed` at entry; Stage 1 follows normal semantics.

6. **Verification**: Add unit tests for the killing logic and integration tests simulating user Ctrl-C usage.

## Technical Design

### Architecture

**Current SIGINT flow** (single stage):
```
SIGINT → interrupt_event.set() → _handle_interrupt → abort all → exit 130
```

**New SIGINT flow** (three stages):
```
SIGINT #1 → _sigint_count=1, drain_mode_active=True
         → drain_event.set() → stop spawning, wait for active
         → on_drain_started()

SIGINT #2 → _sigint_count=2, abort_mode_active=True
         → _abort_exit_code = 1 if _validation_failed else 130
         → interrupt_event.set(), forward_sigint()
         → cancel tasks, 10s grace period → exit 130/1
         → on_abort_started()

SIGINT #3 → _sigint_count>=3, shutdown_requested=True, _exit_code=130
         → kill_active_process_groups(), cancel run task
         → exit 130 (best-effort finally cleanup)
         → on_force_abort()
```

**Key architectural decisions:**
- `drain_event` and `interrupt_event` are separate `asyncio.Event` objects that coexist
- Signal handler only updates simple Python types and schedules work via `call_soon_threadsafe`
- State reset happens at start of each `run()`, not in `__init__`
- `_validation_failed` flag is owned by orchestrator, set by validation callbacks, read by signal handler at Stage 2 entry
- Private `_handle_sigint()` method enables unit testing without real signals
- Thread safety: Python's GIL ensures atomic reads/writes to simple types

**Stage 2 exit code timing clarification:**
- `_abort_exit_code` is snapshotted at Stage 2 entry based on `_validation_failed` at that moment
- If SIGINT #2 arrives during run-level validation: validation is interrupted, exit code is 130 (validation hadn't completed/failed yet)
- If validation fails *after* Stage 2 begins: exit code remains as snapshotted (130), not retroactively changed to 1
- This "snapshot at entry" rule ensures deterministic behavior regardless of race conditions

### Data Model

**New module constants** (in `orchestrator.py`):
```python
ESCALATION_WINDOW_SECONDS = 5.0  # Time window for Ctrl-C escalation
ABORT_GRACE_SECONDS = 10.0       # Grace period for Stage 2 task cancellation
```

**New instance attributes on `MalaOrchestrator`** (all reset per-run in `run()`):
```python
self._sigint_count: int = 0
self._sigint_last_at: float = 0.0
self._drain_mode_active: bool = False
self._abort_mode_active: bool = False
self._abort_exit_code: int = 130  # Snapshot at Stage 2 entry
self._validation_failed: bool = False  # Set by validation callbacks
self._shutdown_requested: bool = False  # Signals run() to terminate
self._run_task: asyncio.Task[Any] | None = None  # Current run task for cancellation
```

Note: These are kept as direct instance attributes (not in `OrchestratorState`) because they are signal-handler state that must be accessible from the synchronous signal handler closure. They are reset at the start of `run()` alongside `self._state`.

**New asyncio.Event**:
- `drain_event`: set on 1st Ctrl-C, signals "stop spawning but let active tasks finish"
- Existing `interrupt_event`: set on 2nd Ctrl-C, signals "cancel active tasks now"

### Stage 2 Grace Period Enforcement

The 10-second grace period is enforced in `DeadlockHandler.abort_active_tasks()`:

```python
async def abort_active_tasks(...) -> int:
    # Cancel all tasks
    for task in tasks.values():
        task.cancel()

    # Await with bounded timeout
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks.values(), return_exceptions=True),
            timeout=ABORT_GRACE_SECONDS,  # 10.0
        )
    except TimeoutError:
        # Tasks still running after grace period - they remain cancelled
        # but we don't wait further. Stage 3 (force kill) is available
        # if user presses Ctrl-C again.
        pass

    return len(tasks)
```

This ensures Stage 2 always returns within ~10 seconds, allowing Stage 3 to run if needed.

### API/Interface Design

**`CommandRunner` additions** (in `src/infra/tools/command_runner.py`):
```python
@staticmethod
def kill_active_process_groups() -> None:
    """Send SIGKILL to all tracked process groups.

    Safe to call multiple times - clears pgid set after kill.
    No-op on Windows. Silently handles ProcessLookupError/PermissionError.
    """
```

**`MalaEventSink` additions** (in `src/core/protocols.py`):
```python
def on_drain_started(self, active_task_count: int) -> None:
    """Called when drain mode starts (1st Ctrl-C).

    Args:
        active_task_count: Number of active tasks that will be drained.
    """

def on_abort_started(self) -> None:
    """Called when graceful abort starts (2nd Ctrl-C)."""

def on_force_abort(self) -> None:
    """Called when hard abort starts (3rd Ctrl-C)."""
```

**Event sink method differentiation** (vs existing methods):
- `on_abort_requested(reason)` — Called when internal error triggers abort (e.g., deadlock). Existing.
- `on_tasks_aborting(count, reason)` — Called when tasks are being cancelled. Existing.
- `on_drain_started(count)` — NEW: User-initiated drain (1st Ctrl-C), informational.
- `on_abort_started()` — NEW: User-initiated abort (2nd Ctrl-C), signals escalation.
- `on_force_abort()` — NEW: User-initiated force kill (3rd Ctrl-C), final escalation.

**`IssueExecutionCoordinator.run_loop()` signature change**:
```python
async def run_loop(
    self,
    spawn_callback: SpawnCallback,
    finalize_callback: FinalizeCallback,
    abort_callback: AbortCallback,
    watch_config: WatchConfig | None = None,
    interrupt_event: asyncio.Event | None = None,
    validation_callback: Callable[[], Awaitable[bool]] | None = None,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    drain_event: asyncio.Event | None = None,  # NEW - appended at end for backward compat
) -> RunResult:
```

Note: `drain_event` is appended at the end to avoid breaking any positional callers.

**`MalaOrchestrator` private method** for testability:
```python
def _handle_sigint(
    self,
    loop: asyncio.AbstractEventLoop,
    drain_event: asyncio.Event,
    interrupt_event: asyncio.Event,
) -> None:
    """Handle SIGINT signal with three-stage escalation.

    Called by signal handler. Updates state and schedules work
    onto the event loop. Extracted for unit testing without real signals.
    """
```

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/orchestration/orchestrator.py` | Exists | Add SIGINT state machine, `_handle_sigint()`, drain_event, three-stage handler |
| `src/pipeline/issue_execution_coordinator.py` | Exists | Add `drain_event` parameter, drain mode handling in `run_loop` |
| `src/infra/tools/command_runner.py` | Exists | Add `kill_active_process_groups()` static method |
| `src/core/protocols.py` | Exists | Add `on_drain_started`, `on_abort_started`, `on_force_abort` to `MalaEventSink` |
| `src/infra/io/console_sink.py` | Exists | Implement new event sink methods with user-facing messages |
| `src/infra/io/base_sink.py` | Exists | Add no-op stubs for new methods |
| `tests/fakes/event_sink.py` | Exists | Add recording for new event types |
| `tests/unit/orchestration/test_orchestrator.py` | Exists | Add SIGINT escalation unit tests |
| `tests/unit/pipeline/test_issue_execution_coordinator.py` | Exists | Add drain mode unit tests |
| `tests/unit/infra/tools/test_command_runner.py` | **New** | Tests for `kill_active_process_groups` |
| `tests/integration/orchestration/test_sigint_escalation.py` | **New** | Subprocess-based SIGINT integration tests |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes
- **Rapid Ctrl-C**: User mashes Ctrl-C. Logic must handle concurrent signal delivery safely (state checks are atomic/idempotent via GIL).
- **SIGINT during run-level validation**: Stage 1 allows completion; Stage 2 forwards SIGINT; Stage 3 kills.
- **Watch mode**: Drain stops polling/spawning but keeps running tasks; drain completion triggers final validation if thresholds met.
- **No active tasks at Stage 1**: Immediate exit with success (0) if no pending validation.
- **CommandRunner has no active process groups**: Escalation proceeds without error (empty set iteration).
- **Stuck subprocess**: `CommandRunner` sends signal, but process ignores it. **Mitigation**: Stage 3 (SIGKILL) handles this.
- **Kill failures (ProcessLookupError, PermissionError)**: No logging needed (expected conditions during shutdown).
- **Stage 2 grace period timeout**: After 10s, exit anyway (Stage 3 available for force kill).
- **Orphan subprocess handling**: Subprocesses spawned after `kill_active_process_groups()` snapshot may survive. This is acceptable best-effort behavior.
- **Windows support**: `killpg` is Unix-specific. Stage 2/3 signal operations are no-ops on Windows.

### Breaking Changes & Compatibility
- **Internal state**: New instance attributes on `MalaOrchestrator` (internal, not public API).
- **Coordinator signature**: `run_loop()` gains optional `drain_event` parameter (backward compatible via default `None`).
- **Event sink protocol**: New methods added (`on_drain_started`, `on_abort_started`, `on_force_abort`) — existing implementations (ConsoleEventSink, BaseEventSink, FakeEventSink) need updates.
- **Behavior change**: Single Ctrl-C no longer exits immediately (deliberate UX change per spec).
- **Mitigation**: No external API changes; CLI args remain unchanged.

## Testing & Validation Strategy

### Unit Tests

**`tests/unit/orchestration/test_orchestrator.py`**:
- Test `_handle_sigint()` directly with mock loop and events
- Test escalation timing: 5s window reset only when idle (not in drain/abort)
- Test state transitions: idle → drain → abort → hard abort
- Test exit code precedence for each stage (Stage 3: 130, Stage 2: 130/1 based on validation, Stage 1: 0/1)
- Test `_validation_failed` flag snapshot at Stage 2 entry

**`tests/unit/pipeline/test_issue_execution_coordinator.py`**:
- Test drain mode: no new spawns when `drain_event` set
- Test drain completion: returns when active tasks finish (no cancellation)
- Test drain + interrupt: `drain_event` then `interrupt_event` escalates correctly
- Test drain completion triggers validation in watch mode

**`tests/unit/infra/tools/test_command_runner.py`** (New):
- Test `kill_active_process_groups()` sends SIGKILL to tracked pgids
- Test no-op on Windows (`sys.platform == "win32"`)
- Test clears `_SIGINT_FORWARD_PGIDS` to avoid redundant signals
- Test handles empty pgid set gracefully
- Test silently handles ProcessLookupError/PermissionError

### Integration Tests

**`tests/integration/orchestration/test_sigint_escalation.py`** (New):
- Subprocess-based tests for real SIGINT handling
- Test single SIGINT → drain → exit 0 (or 1 if validation fails)
- Test double SIGINT → abort → exit 130 (or 1 if validation already failed)
- Test triple SIGINT → force kill → exit 130
- Test escalation window behavior with timing
- POSIX-only (`@pytest.mark.skipif(sys.platform == "win32")`)
- May skip in CI (`@pytest.mark.skipif(os.environ.get("CI"))`) if signals unreliable

### Regression Tests
- Ensure existing `interrupt_event` behavior for single SIGINT still works when drain_event is None (backward compat)
- Ensure watch mode polling/idle behavior unchanged when drain_event not set
- Existing tests in `test_issue_execution_coordinator.py` must still pass

### Manual Verification
1. Run `mala run` and press Ctrl-C once → verify "Ctrl-C: draining" message, active tasks complete
2. Press Ctrl-C again → verify "Ctrl-C: aborting" message, subprocesses receive SIGINT
3. Press Ctrl-C third time → verify "Ctrl-C: force killing" message, immediate exit 130

### Acceptance Criteria Coverage

| Spec Requirement | Covered By |
|------------------|------------|
| Stage 1 stops spawning, lets active finish | Coordinator `drain_event` handling + unit tests |
| Stage 2 cancels tasks, forwards SIGINT | Orchestrator `interrupt_event` + `forward_sigint()` + unit tests |
| Stage 2 10s grace period | `ABORT_GRACE_SECONDS` constant + coordinator await logic |
| Stage 3 SIGKILL all process groups | `CommandRunner.kill_active_process_groups()` + unit tests |
| Exit code 130 for Stage 3 | Orchestrator `_shutdown_requested` → 130 + unit tests |
| Exit code 1/130 for Stage 2 based on validation | `_validation_failed` snapshot at Stage 2 entry + unit tests |
| Exit code 0/1 for Stage 1 (normal semantics) | Coordinator drain completion + unit tests |
| 5s escalation window (reset only when idle) | Signal handler timing logic + unit tests |
| Drain completion triggers validation (watch mode) | Coordinator drain completion handling + unit tests |
| Observability (drain/abort messages) | New event sink methods + ConsoleEventSink implementation |

## Open Questions

None — all decisions resolved during spec/interview phase:
1. Breaking change for internal state OK, external APIs backward compatible ✓
2. Use `signal.signal()` + `loop.call_soon_threadsafe()` ✓
3. Hardcode `ABORT_GRACE_SECONDS = 10.0` ✓
4. Both unit and integration tests ✓
5. Event naming: `on_drain_started`, `on_abort_started`, `on_force_abort` ✓
6. Separate `drain_event` and `interrupt_event` asyncio.Event objects ✓
7. Private `_handle_sigint()` method for testability ✓
8. Drain completion triggers validation in watch mode ✓
9. Integration tests in `tests/integration/orchestration/` ✓

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
