# SIGINT Escalation and Drain-Then-Abort Behavior

## Summary
Introduce a three-stage Ctrl-C behavior for mala runs:
1) First Ctrl-C enters **drain mode**: stop spawning new agents, allow active agents to finish their full issue flow (implement + gate + review).
2) Second Ctrl-C triggers **graceful abort**: cancel active agent tasks, forward SIGINT to subprocesses, and exit with 130 (or 1 if validation had already failed at the moment of the 2nd Ctrl-C).
3) Third Ctrl-C triggers **hard abort**: SIGKILL active subprocess groups and exit immediately (best-effort cleanup, no grace period).

This aligns with common CLI conventions while preserving a safe, user-friendly default.

## Goals
- Let users stop accepting new work without losing in-flight progress.
- Provide predictable escalation for users who want a faster shutdown.
- Ensure validation subprocesses respond immediately on second Ctrl-C.
- Keep the behavior consistent across watch and non-watch modes.

## Non-goals
- Changing issue assignment or beads semantics.
- Guaranteeing a commit on first Ctrl-C if an agent is stuck (drain mode waits for completion; it does not intervene).
- Adding new CLI flags (behavior is triggered by SIGINT only).
- **Full Windows signal support**: Signal escalation semantics (SIGINT forwarding, SIGKILL) are POSIX-only. On Windows:
  - Stage 1 (drain) works normally (event-based, no signals).
  - Stage 2/3 signal forwarding/kill operations are no-ops (subprocesses won't receive forwarded signals).
  - Graceful exit still occurs via `sys.exit()`.
  - Future work could add Windows job object support, but this is out of scope for the initial implementation.

## Desired Behavior

### Stage 1: Drain (first Ctrl-C)
- **Stop creating new issue-level tasks** (top-level agent spawns for new issues). In-flight issue flows may continue spawning their own internal work (subprocess calls, tool invocations) as they already do today.
- Continue running existing agent tasks until they complete their full flow (including gate/review).
- Do **not** forward SIGINT to validation subprocesses.
- Emit a clear event/log: "Interrupt received: draining active tasks".
- Exit once all active tasks are finalized.
- Exit code: keep current success/validation semantics (0 or 1); do not force 130 for drain completion.

### Stage 2: Graceful Abort (second Ctrl-C)
- **Cancel active agent tasks**: Cancel the per-issue flow tasks tracked in `IssueExecutionCoordinator.active_tasks` via `asyncio.Task.cancel()`. This includes only issue-level agent tasks, not internal coordination tasks or the main run task.
- Forward SIGINT to **all active `CommandRunner` process groups** (not just validation—includes any subprocess started via `CommandRunner`, such as agent tool invocations).
- **Await cancellation with bounded grace period**: Wait up to `ABORT_GRACE_SECONDS` (10 seconds) for tasks to respond to cancellation and complete finalization. If tasks do not complete within the grace period, proceed to exit anyway (Stage 3 remains available for forceful termination).
- Finalize all tasks as "Interrupted" via normal finalization (best-effort within grace period).
- Exit code: see **Exit Code Precedence** below.

### Stage 3: Hard Abort (third Ctrl-C)
- SIGKILL all tracked subprocess groups (CommandRunner groups).
- **Exit mechanism**: Cancel the main `run()` task via `asyncio.Task.cancel()`, allowing `finally` blocks and lock cleanup to run during normal exception unwinding. The `run()` method catches `CancelledError`, checks `_shutdown_requested`, and returns early. The exit code is stored in `self._exit_code` (set to 130) for the CLI to read.
- **"Immediately" means**: No 10-second grace period like Stage 2. Best-effort `finally` cleanup runs (lock release, handler restore), then `run()` returns promptly. No waiting for task finalization.
- Exit code: always 130 (user-initiated interrupt).
- Emit a final log: "Force killing active processes".
- On kill failures (ProcessLookupError, PermissionError): do not log (these are expected conditions); do not hang or retry.
- **Orphan subprocess handling**: Subprocesses spawned after the `kill_active_process_groups()` snapshot may survive. This is acceptable best-effort behavior—Stage 3 is a last resort and the user can kill remaining processes manually or via their shell.

### Exit Code Precedence

Exit codes follow this precedence (highest priority first):

| Stage | Condition | Exit Code |
|-------|-----------|-----------|
| Stage 3 (hard abort) | Always | 130 |
| Stage 2 (graceful abort) | `validation_failed` flag was set before Stage 2 entered | 1 |
| Stage 2 (graceful abort) | `validation_failed` not set when Stage 2 entered | 130 |
| Stage 1 (drain) | Validation ran and failed | 1 |
| Stage 1 (drain) | All tasks completed successfully | 0 |

**Deterministic rule for Stage 2 exit code**: The orchestrator maintains a `_validation_failed: bool` flag on the `MalaOrchestrator` instance. This flag is:
- **Owned by**: `MalaOrchestrator` instance
- **Set by**: A single callback invoked when validation completes (e.g., in `run_validation()` result handling)
- **Read by**: Signal handler at Stage 2 entry to compute `_abort_exit_code`

At the moment Stage 2 is entered (2nd Ctrl-C), the signal handler atomically snapshots the exit code:
```python
self._abort_exit_code = 1 if self._validation_failed else 130
```

This avoids race conditions: the exit code is determined by the state *at abort entry*, not by concurrent validation completion. The `_abort_exit_code` is then used when `run()` returns.

### Timing / Reset
- Escalation window: 5 seconds between consecutive Ctrl-C presses.
- **Reset only applies when idle**: The escalation window reset (treating next Ctrl-C as fresh) only applies when the orchestrator is NOT in drain or abort mode.
- **Active mode preserves escalation**: Once `drain_mode_active` is set (Stage 1), the escalation state is preserved regardless of elapsed time. A subsequent Ctrl-C always escalates to Stage 2, even if >5s have passed. Similarly, once in Stage 2, the next Ctrl-C always escalates to Stage 3.
- Rationale: Users should always be able to escalate out of a stuck drain or abort without having to press Ctrl-C rapidly.

## Implementation Plan

### Orchestrator SIGINT Handler

#### Signal Handler Architecture

Python signal handlers run in the main thread, interrupting whatever code is executing. In an asyncio program, the handler must not perform blocking I/O or directly manipulate coroutines. The implementation uses `loop.call_soon_threadsafe()` to safely schedule **all** nontrivial work onto the event loop.

**Precondition**: `MalaOrchestrator.run()` must execute in the main thread. This is enforced by:
- The CLI entrypoint (`mala run`) always runs in the main thread via `asyncio.run()`.
- Tests that exercise SIGINT handling must run the orchestrator in the main thread (use `asyncio.run()` or ensure `threading.current_thread() is threading.main_thread()`).

**Why `signal.signal()` over `loop.add_signal_handler()`**: While `loop.add_signal_handler()` is the asyncio-native approach, `signal.signal()` is simpler and sufficient since we have the main-thread precondition. Either approach works; the implementation may choose `loop.add_signal_handler()` for cleaner asyncio integration if preferred.

**State location**: Instance attributes on `MalaOrchestrator`, accessed via closure capture of `self`.

**State reset**: All SIGINT state is reset at the start of each `run()` call (not in `__init__`). This ensures a fresh orchestrator instance or a reused instance both start with clean state.

**Constants** (module-level in `orchestrator.py`):
```python
ESCALATION_WINDOW_SECONDS = 5.0  # Time window for Ctrl-C escalation
ABORT_GRACE_SECONDS = 10.0       # Grace period for Stage 2 task cancellation
```

**Implementation**:
```python
class MalaOrchestrator:
    def __init__(self, ...):
        # ... existing init ...
        # SIGINT state declared but NOT initialized here (reset per-run)
        self._sigint_count: int
        self._sigint_last_at: float
        self._drain_mode_active: bool
        self._abort_mode_active: bool
        self._abort_exit_code: int  # Snapshot at Stage 2 entry
        self._validation_failed: bool  # Set by validation callbacks
        self._shutdown_requested: bool  # Signals run() to terminate
        self._run_task: asyncio.Task[Any] | None  # Current run task for cancellation

    async def run(self, ...):
        # Reset all SIGINT state at start of each run
        self._sigint_count = 0
        self._sigint_last_at = 0.0
        self._drain_mode_active = False
        self._abort_mode_active = False
        self._abort_exit_code = 130
        self._validation_failed = False
        self._shutdown_requested = False
        self._run_task = asyncio.current_task()

        drain_event = asyncio.Event()
        interrupt_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        # Nested function for force shutdown - captures self and run_task via closure
        def force_shutdown() -> None:
            """Hard shutdown: cancel main task and exit."""
            if self._run_task and not self._run_task.done():
                self._run_task.cancel()

        def handle_sigint(sig: int, frame: object) -> None:
            """Synchronous signal handler - minimal state update, schedule all work."""
            now = time.monotonic()

            # Reset count only if idle (not in drain/abort) and outside window.
            # Once drain_mode_active is set, subsequent SIGINTs always escalate.
            if not self._drain_mode_active and not self._abort_mode_active:
                if now - self._sigint_last_at > ESCALATION_WINDOW_SECONDS:
                    self._sigint_count = 0

            self._sigint_count += 1
            self._sigint_last_at = now

            # Schedule ALL work onto the event loop for safety
            if self._sigint_count == 1:
                self._drain_mode_active = True
                loop.call_soon_threadsafe(drain_event.set)
                loop.call_soon_threadsafe(
                    lambda: logger.info("Ctrl-C: draining (press again to abort)")
                )
            elif self._sigint_count == 2:
                self._abort_mode_active = True
                # Snapshot exit code NOW based on current validation state
                self._abort_exit_code = 1 if self._validation_failed else 130
                loop.call_soon_threadsafe(interrupt_event.set)
                loop.call_soon_threadsafe(CommandRunner.forward_sigint)
                loop.call_soon_threadsafe(
                    lambda: logger.info("Ctrl-C: aborting (press again to force)")
                )
            else:  # sigint_count >= 3
                self._shutdown_requested = True
                self._exit_code = 130  # Set exit code for CLI
                loop.call_soon_threadsafe(CommandRunner.kill_active_process_groups)
                loop.call_soon_threadsafe(force_shutdown)

        original_handler = signal.signal(signal.SIGINT, handle_sigint)
        try:
            # ... run logic ...
            pass
        except asyncio.CancelledError:
            if self._shutdown_requested:
                # Stage 3: force exit after cleanup. External cancellation
                # (not from Stage 3) re-raises to propagate normally.
                return
            raise
        finally:
            signal.signal(signal.SIGINT, original_handler)
```

**Key invariants:**
- Signal handler only updates simple Python types (`int`, `float`, `bool`) and schedules work.
- **All** nontrivial operations (event.set, logging, subprocess signaling, exit) are scheduled via `loop.call_soon_threadsafe()`.
- State variables are instance attributes on `MalaOrchestrator` for testability and reset between runs.
- Thread safety: Python's GIL ensures atomic reads/writes to simple types. Signal handlers run serially in the main thread. The event loop reads state between signal deliveries, which is safe due to GIL guarantees.

### Drain Mode Wiring

**API change**: `IssueExecutionCoordinator.run_loop` gains a new parameter:
```python
async def run_loop(
    self,
    interrupt_event: asyncio.Event | None = None,
    drain_event: asyncio.Event | None = None,  # NEW
    # ... other params
) -> RunLoopResult:
```

**Relationship with `interrupt_event`**: `drain_event` and `interrupt_event` are separate events that coexist:
- `drain_event`: set on 1st Ctrl-C, signals "stop spawning but let active tasks finish"
- `interrupt_event`: set on 2nd Ctrl-C, signals "cancel active tasks now"

**Coordinator loop logic** (sketch of modified `run_loop`):
```python
async def run_loop(self, interrupt_event, drain_event, ...):
    while True:
        # Check drain first (stop spawning), then interrupt (cancel)
        if drain_event and drain_event.is_set():
            # Drain mode: don't spawn new issues
            if not self.active_tasks:
                # All tasks done, exit cleanly
                return RunLoopResult(exit_code=0, reason="drain_complete")
            # Wait for active tasks (no new spawns)
            await self._wait_for_active_tasks(interrupt_event)
            continue

        if interrupt_event and interrupt_event.is_set():
            # Abort mode: cancel and finalize
            return await self._handle_interrupt(...)

        # Normal operation: poll and spawn
        if not (drain_event and drain_event.is_set()):
            await self._spawn_new_issues(...)

        # Interruptible sleep for watch mode
        try:
            await asyncio.wait_for(
                self._create_combined_event(drain_event, interrupt_event).wait(),
                timeout=poll_interval
            )
        except asyncio.TimeoutError:
            pass  # Normal timeout, continue loop
```

- Emit a user-visible event for drain start and drain completion.
- **Drain completion in watch mode**: When drain completes (all active tasks done), run final validation if thresholds are met (consistent with current `_handle_interrupt` behavior).

### Graceful Abort Wiring
- Existing `interrupt_event` path stays, but it should only be set on 2nd Ctrl-C.
- Keep current abort_callback behavior (cancel + finalize).
- Continue forwarding SIGINT to subprocess groups on 2nd Ctrl-C.

### Hard Abort Wiring
- Add `CommandRunner.kill_active_process_groups()` as a static method (like `forward_sigint()`) to send SIGKILL to tracked pgids.
- Implementation:
  ```python
  @staticmethod
  def kill_active_process_groups() -> None:
      """Send SIGKILL to all tracked process groups."""
      if sys.platform == "win32":
          return
      # Snapshot and clear to avoid redundant signals if called multiple times
      pgids = list(_SIGINT_FORWARD_PGIDS)
      _SIGINT_FORWARD_PGIDS.clear()
      for pgid in pgids:
          try:
              os.killpg(pgid, signal.SIGKILL)
          except (ProcessLookupError, PermissionError) as e:
              # Expected: process already exited or no permission
              # No logging needed for these expected conditions
              pass
  ```
- **No logging on expected kill failures**: `ProcessLookupError` (process already exited) and `PermissionError` are expected conditions during shutdown and do not warrant logging. This keeps shutdown output clean.
- After kill, orchestrator cancels the main run task (via `_force_shutdown`) to unwind the asyncio stack cleanly. The `finally` blocks and lock cleanup run as part of normal exception unwinding.

### Process Group Invariants

**Requirement**: All validation subprocesses must be started in their own process group to receive forwarded signals.

**Guaranteed coverage:**
- All commands executed via `CommandRunner.run()` or `CommandRunner.run_async()` with `use_process_group=True` (the default) are launched with `start_new_session=True`, creating a new process group with pgid = child pid.
- The pgid is registered in `_SIGINT_FORWARD_PGIDS` for the duration of the command.

**Not covered:**
- Subprocesses spawned directly via `subprocess.Popen` or `asyncio.create_subprocess_*` without going through `CommandRunner` will NOT receive forwarded signals.
- This is acceptable: mala's validation commands all go through `CommandRunner` (see `quality_gate.py`, `validation/*.py`).

**Verification**: Audit `src/validation/` and `src/domain/quality_gate.py` to confirm all subprocess calls use `CommandRunner`.

## Affected Components
- `src/orchestration/orchestrator.py`: SIGINT handling state machine and events.
- `src/pipeline/issue_execution_coordinator.py`: drain mode support.
- `src/infra/tools/command_runner.py`: add SIGKILL helper for tracked process groups.
- Event sink/logging: add drain/abort escalation messages.

## Edge Cases
- SIGINT during run-level validation:
  - Stage 1: allow validation to complete normally.
  - Stage 2: forward SIGINT so validation stops promptly.
  - Stage 3: kill validation subprocesses immediately.
- Watch mode:
  - Drain should stop polling/spawning but keep running tasks.
  - Graceful abort should behave as today.
- No active tasks:
  - Stage 1 should cause immediate exit with success (0) if no pending validation.
- If `CommandRunner` has no active process groups, escalation should still proceed without error.

## Observability
- Add log events at each stage with count + mode.
- Add a single consolidated message to the user explaining escalation:
  - "Ctrl-C: draining (press again to abort, three times to force)"
  - "Ctrl-C: aborting (press again to force kill)"
  - "Ctrl-C: force killing"

## Testing

### Unit Tests (in-process, mock signals)
- **Escalation state machine**: Test `sigint_count` increment, window reset, `drain_mode_active`/`abort_mode_active` transitions.
- **Exit code snapshot**: Test `_abort_exit_code` is captured correctly based on `_validation_failed` at Stage 2 entry.
- **Coordinator drain behavior**: Test `run_loop` with `drain_event` set—verify no new spawns, waits for active tasks, returns on completion.

### Integration Tests (subprocess-based, POSIX-only)
**Location**: `tests/integration/test_sigint_escalation.py`

**Strategy**: Run `mala run` (or orchestrator directly) in a subprocess, send signals with controlled timing, assert outcomes.

```python
@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT semantics are POSIX-only")
@pytest.mark.skipif(os.environ.get("CI"), reason="Signals unreliable in some CI environments")
class TestSIGINTEscalation:
    def test_single_sigint_drains(self, tmp_path):
        """1 Ctrl-C: process drains and exits 0 (if no validation failure)."""
        proc = subprocess.Popen(["mala", "run", ...], ...)
        time.sleep(1)  # Let it start
        os.kill(proc.pid, signal.SIGINT)
        proc.wait(timeout=30)
        assert proc.returncode == 0
        assert "draining" in proc.stderr.read()

    def test_double_sigint_aborts(self, tmp_path):
        """2 Ctrl-C: process aborts with exit code 130."""
        proc = subprocess.Popen(["mala", "run", ...], ...)
        time.sleep(1)
        os.kill(proc.pid, signal.SIGINT)
        time.sleep(0.5)
        os.kill(proc.pid, signal.SIGINT)
        proc.wait(timeout=15)
        assert proc.returncode == 130
        assert "aborting" in proc.stderr.read()

    def test_triple_sigint_force_kills(self, tmp_path):
        """3 Ctrl-C: process force-kills subprocesses and exits 130."""
        proc = subprocess.Popen(["mala", "run", ...], ...)
        time.sleep(1)
        for _ in range(3):
            os.kill(proc.pid, signal.SIGINT)
            time.sleep(0.2)
        proc.wait(timeout=5)
        assert proc.returncode == 130
        assert "force killing" in proc.stderr.read()

    def test_escalation_window_reset(self, tmp_path):
        """After 5+ seconds idle, next Ctrl-C is treated as fresh."""
        proc = subprocess.Popen(["mala", "run", ...], ...)
        time.sleep(1)
        os.kill(proc.pid, signal.SIGINT)  # Enter drain
        time.sleep(6)  # Window expires, but drain_mode_active keeps escalation
        os.kill(proc.pid, signal.SIGINT)  # Should still escalate to abort
        proc.wait(timeout=15)
        assert proc.returncode == 130
```

### CommandRunner Signal Tests
- Test `forward_sigint()` sends SIGINT to tracked pgids.
- Test `kill_active_process_groups()` sends SIGKILL and clears the set.
- Test both handle empty pgid set gracefully.

## Resolved Decisions
- **Escalation window duration**: 5 seconds (reset only applies when idle).
- **Hard abort exit code**: 130 (user-initiated interrupt; 137 would imply SIGKILL received by the process itself, which is misleading).
- **Drain completion validation in watch mode**: Yes, run final validation if thresholds are met (consistent with current abort behavior).
