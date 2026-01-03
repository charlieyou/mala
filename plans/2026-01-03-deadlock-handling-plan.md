# Implementation Plan: Deadlock Handling for Multi-Agent Coordination

## Context & Goals
- **Spec**: N/A - derived from user interview and codebase context
- Detect deadlocks between concurrent agents in real-time (on lock contention, not timeout-based)
- Auto-resolution: kill youngest agent, release its locks, add dependency to prevent immediate restart
- Surface deadlock details to orchestrator via event sink for observability
- Hook-based event emission from agents to orchestrator (PostToolUse for lock commands)

## Scope & Non-Goals

### In Scope
- Centralized Wait-For Graph (WFG) in orchestrator for deadlock detection
- PostToolUse hook emits lock events after lock commands (`lock-try.sh`, `lock-wait.sh`, `lock-release.sh`) complete
- DFS cycle detection on WAITING events (O(n) per event, n = active agents, typically <=10)
- Victim selection: youngest agent by start_time (simplest heuristic, can enhance later)
- Resolution: cancel task, cleanup locks, add dependency, mark needs-followup
- Deadlock-related event emission and logging via event_sink
- Feature flag / config gating via `--disable-validations=deadlock-detection`

### Out of Scope (Non-Goals)
- Changes to lock acquisition primitives or lock scripts (hardlink-based locks work correctly)
- Lock ordering / deadlock prevention (too disruptive to agent behavior)
- Agent-side deadlock detection (centralized approach chosen)
- Progress-based victim selection (youngest is simpler, can add progress tracking later)
- E2E tests for deadlock scenarios (too complex to orchestrate reliably)

## Assumptions & Constraints

### Assumptions
- Agents invoke `lock-try.sh`, `lock-wait.sh`, `lock-release.sh` via Bash tool
- Lock script exit codes are reliable: try/wait (0 success, 1 blocked/timeout, 2 error)
- Orchestrator maintains `active_tasks` and `agent_ids` mappings (already exists)
- Beads API is available for `add_dependency()` and `mark_needs_followup()` (already exists)
- Event sink can be extended for deadlock events (follows existing patterns)

### Implementation Constraints
- Must use PostToolUse hook for lock event emission (not PreToolUse - need exit code)
- Must not break single-agent workflows
- O(n) cycle detection per WAITING event (acceptable for n <= 10)
- Integrate with existing hook infrastructure (`PreToolUse`, `Stop` hooks in `agent_session_runner.py`)
- No re-export shims; update imports directly if needed

### Testing Constraints
- 85% coverage threshold (enforced by quality gate)
- Unit + integration tests required
- No E2E tests for deadlock scenarios
- Use pytest markers (`unit`, `integration`) per repo convention

## Prerequisites
- [x] Understand current lock state visibility (lock files contain agent_id, can query via `get_lock_holder`)
- [x] Verify lock files contain enough metadata (agent_id is sufficient; orchestrator tracks issue_id mapping)
- [x] Verify hook infrastructure can support PostToolUse hooks (need to add type, currently only PreToolUse and Stop)
- [ ] Confirm PostToolUse hook can access bash command and exit code from `tool_result`
- [ ] Verify canonical lock path computation in Python matches shell scripts (reuse `locking._canonicalize_path`)

## High-Level Approach

1. Add `WaitForGraph`, `DeadlockMonitor`, and `LockEvent` domain model in `src/domain/deadlock.py`
2. Add PostToolUse hook type to hooks infrastructure and emit lock events when lock scripts complete
3. Wire `DeadlockMonitor` into orchestrator - register agents on start, process events, detect cycles
4. On deadlock detection: select youngest victim, cancel task, cleanup locks, add dependency, emit event
5. Gate the feature behind config + `--disable-validations=deadlock-detection`

## Technical Design

### Architecture

**Selected Approach:** Wait-For Graph (Centralized) with PostToolUse hook-based event emission.

```
+---------------------------------------------------------------------------+
|                           ORCHESTRATOR                                      |
|  +------------------+    +------------------+    +-------------------+      |
|  | DeadlockMonitor  |<---| WaitForGraph     |    | IssueCoordinator  |      |
|  |                  |    |                  |    |                   |      |
|  | - on_lock_event  |    | - holds: dict    |    | - active_tasks    |      |
|  | - check_cycle()  |    | - waits: dict    |    | - agent_ids       |      |
|  | - resolve()      |    | - detect_cycle() |    |                   |      |
|  +--------+---------+    +------------------+    +---------+---------+      |
|           |                                                |                |
|           |  on deadlock detected                          |                |
|           +------------------------------------------------+                |
|           |  1. select_victim(youngest)                                     |
|           |  2. cancel task                                                 |
|           |  3. cleanup_agent_locks()                                       |
|           |  4. add_dependency(victim -> blocker)                           |
|           |  5. mark_needs_followup()                                       |
|           |  6. emit deadlock event                                         |
+-----------+----------------------------------------------------------------+
            ^
            | LockEvent (acquired/waiting/released)
            |
+-----------+----------------------------------------------------------------+
|                           AGENT PROCESS                                     |
|  +---------------------------------------------------------------------+   |
|  | PostToolUse Hook (event emission after lock commands complete)       |   |
|  |                                                                      |   |
|  | on Bash(lock-try.sh file) completed:                                 |   |
|  |   if exit_code == 0:                                                 |   |
|  |     emit LockEvent(type=ACQUIRED, agent_id, file)                    |   |
|  |   else:                                                              |   |
|  |     emit LockEvent(type=WAITING, agent_id, file)                     |   |
|  |                                                                      |   |
|  | on Bash(lock-wait.sh file) completed:                                |   |
|  |   if exit_code == 0:                                                 |   |
|  |     emit LockEvent(type=ACQUIRED, agent_id, file)                    |   |
|  |   else:                                                              |   |
|  |     # Timeout - may still be waiting or gave up                      |   |
|  |     clear wait status (emit RELEASED or no-op)                       |   |
|  |                                                                      |   |
|  | on Bash(lock-release.sh file) completed:                             |   |
|  |   emit LockEvent(type=RELEASED, agent_id, file)                      |   |
|  +---------------------------------------------------------------------+   |
+----------------------------------------------------------------------------+
```

**Event Flow:**
1. Agent calls `lock-try.sh file.py` via Bash tool
2. PostToolUse hook intercepts result after execution completes:
   - Success (exit 0) -> emit `LockEvent(ACQUIRED, agent_id, file.py)`
   - Failure (exit 1) -> emit `LockEvent(WAITING, agent_id, file.py)`
3. Orchestrator's `DeadlockMonitor` receives event, updates `WaitForGraph`
4. After each WAITING event, run `detect_cycle()`
5. If cycle found -> `resolve_deadlock()` with youngest agent as victim

### Data Model

```python
# src/domain/deadlock.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable
import time

class LockEventType(Enum):
    ACQUIRED = "acquired"
    WAITING = "waiting"
    RELEASED = "released"

@dataclass
class LockEvent:
    """Event emitted by hooks when lock state changes."""
    event_type: LockEventType
    agent_id: str
    lock_path: str  # Canonical absolute path
    timestamp: float = field(default_factory=time.monotonic)

@dataclass
class WaitForGraph:
    """Directed graph for deadlock detection.

    Edges:
    - agent --holds--> lock (tracked in `holds`)
    - agent --waits_for--> lock (tracked in `waits_for`)
    - lock --held_by--> agent (inverse of holds, computed via get_holder)
    """
    holds: dict[str, set[str]] = field(default_factory=dict)
    # agent_id -> set of lock paths held

    waits_for: dict[str, str | None] = field(default_factory=dict)
    # agent_id -> lock path waiting for (None if not waiting)

    def add_hold(self, agent_id: str, lock_path: str) -> None:
        """Record that agent acquired a lock."""
        ...

    def add_wait(self, agent_id: str, lock_path: str) -> None:
        """Record that agent is waiting for a lock."""
        ...

    def remove_hold(self, agent_id: str, lock_path: str) -> None:
        """Record that agent released a lock."""
        ...

    def remove_agent(self, agent_id: str) -> None:
        """Remove all state for an agent (on termination)."""
        ...

    def get_holder(self, lock_path: str) -> str | None:
        """Return agent_id holding the lock, or None."""
        ...

    def detect_cycle(self) -> list[str] | None:
        """Detect deadlock cycle using DFS. Returns agent_ids in cycle or None."""
        ...

@dataclass
class DeadlockInfo:
    """Information about a detected deadlock."""
    cycle: list[str]           # Agent IDs in the cycle
    victim_id: str             # Selected victim
    victim_issue_id: str       # Victim's issue ID
    blocked_on: str            # Lock path victim was waiting for
    blocker_id: str            # Agent holding that lock
    blocker_issue_id: str      # Blocker's issue ID

DeadlockCallback = Callable[[DeadlockInfo], Awaitable[None]]

@dataclass
class DeadlockMonitor:
    """Monitors lock events and detects deadlocks."""
    graph: WaitForGraph = field(default_factory=WaitForGraph)
    agent_start_times: dict[str, float] = field(default_factory=dict)
    agent_to_issue: dict[str, str] = field(default_factory=dict)
    on_deadlock: DeadlockCallback | None = None

    def register_agent(self, agent_id: str, issue_id: str) -> None:
        """Register an agent when it starts."""
        ...

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent when it stops."""
        ...

    async def handle_event(self, event: LockEvent) -> None:
        """Process a lock event and check for deadlocks."""
        ...
```

### API/Interface Design

```python
# src/infra/hooks/deadlock.py

from typing import Callable
from src.domain.deadlock import LockEvent, LockEventType

def make_lock_event_hook(
    agent_id: str,
    emit_event: Callable[[LockEvent], None],
    repo_namespace: str | None = None,
) -> PostToolUseHook:
    """Create a PostToolUse hook that emits lock events.

    Args:
        agent_id: The agent ID for event attribution.
        emit_event: Callback to send LockEvent to orchestrator.
        repo_namespace: Optional repo namespace for lock path canonicalization.

    Returns:
        PostToolUse hook function.
    """
    ...

def _extract_lock_path(command: str) -> str | None:
    """Extract the file path argument from a lock command."""
    ...

def _get_exit_code(tool_result: dict) -> int:
    """Extract exit code from tool result."""
    ...
```

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/domain/deadlock.py` | **New** | WaitForGraph, DeadlockMonitor, DeadlockInfo, LockEvent |
| `src/infra/hooks/deadlock.py` | **New** | make_lock_event_hook PostToolUse hook |
| `src/infra/hooks/__init__.py` | Modify | Export PostToolUseHook type, make_lock_event_hook |
| `src/infra/hooks/dangerous_commands.py` | Modify | Add PostToolUseHook type alias (or new file) |
| `src/orchestration/orchestrator.py` | Modify | Add DeadlockMonitor, wire up event handling and resolution |
| `src/orchestration/orchestration_wiring.py` | Modify | Wire DeadlockMonitor into agent session hooks |
| `src/pipeline/agent_session_runner.py` | Modify | Support PostToolUse hooks in SDK options |
| `src/infra/io/event_protocol.py` | Modify | Add on_deadlock_detected event |
| `src/cli/cli.py` | Modify | Add `deadlock-detection` to VALID_DISABLE_VALUES |
| `src/infra/io/config.py` | Modify | Add deadlock_detection_enabled config option |
| `tests/domain/test_deadlock.py` | **New** | Unit tests for WaitForGraph cycle detection |
| `tests/infra/hooks/test_deadlock_hook.py` | **New** | Unit tests for lock event hook |
| `tests/integration/test_deadlock_integration.py` | **New** | Integration tests for deadlock resolution |

## Detailed Plan

### Task 1: Add Deadlock Domain Model
- **Goal**: Implement WaitForGraph and DeadlockMonitor with cycle detection
- **Depends on**: None
- **Changes**:
  - **New**: `src/domain/deadlock.py`
    - `LockEventType` enum: ACQUIRED, WAITING, RELEASED
    - `LockEvent` dataclass: event_type, agent_id, lock_path, timestamp
    - `WaitForGraph` dataclass with methods:
      - `add_hold()`, `add_wait()`, `remove_hold()`, `remove_agent()`
      - `get_holder()`: returns agent_id holding a lock
      - `detect_cycle()`: DFS algorithm returning list of agent_ids in cycle or None
    - `DeadlockInfo` dataclass: cycle, victim_id, victim_issue_id, blocked_on, blocker_id, blocker_issue_id
    - `DeadlockMonitor` dataclass:
      - `register_agent()`, `unregister_agent()`
      - `handle_event()`: updates graph, triggers cycle detection on WAITING
      - Victim selection: youngest agent (max start_time in cycle)
- **Verification**:
  - **New**: `tests/domain/test_deadlock.py`
    - Test: `detect_cycle()` with no agents returns None
    - Test: `detect_cycle()` with single agent waiting returns None (no cycle)
    - Test: `detect_cycle()` with A<->B cycle returns [A, B] or [B, A]
    - Test: `detect_cycle()` with A->B->C->A cycle returns cycle
    - Test: `add_hold`/`remove_hold` state management
    - Test: Victim selection picks youngest (max start_time)
    - Test: `remove_agent()` clears all state for agent
  - Run: `uv run pytest tests/domain/test_deadlock.py -v -m unit`
- **Rollback**: Delete `src/domain/deadlock.py` and `tests/domain/test_deadlock.py`

### Task 2: Add PostToolUse Hook Type
- **Goal**: Enable PostToolUse hooks in the hook infrastructure
- **Depends on**: None
- **Changes**:
  - `src/infra/hooks/dangerous_commands.py`: Add `PostToolUseHook` type alias (mirror PreToolUseHook pattern)
  - `src/infra/hooks/__init__.py`: Export `PostToolUseHook`
  - `src/pipeline/agent_session_runner.py`:
    - Update `_build_hooks()` to return `post_tool_hooks` list
    - Update `_build_sdk_options()` to wire `PostToolUse` hooks into SDK HookMatcher
- **Verification**:
  - Unit tests for hook wiring (verify PostToolUse hooks are registered)
  - Run: `uv run pytest tests/pipeline/test_agent_session_runner.py -v`
- **Rollback**: Revert changes to hooks and session runner

### Task 3: Emit Lock Events from PostToolUse Hook
- **Goal**: Capture lock command outcomes and emit LockEvent
- **Depends on**: Tasks 1, 2
- **Changes**:
  - **New**: `src/infra/hooks/deadlock.py`
    - `make_lock_event_hook(agent_id, emit_event, repo_namespace)` returning PostToolUse hook
    - Detect `lock-try.sh`, `lock-wait.sh`, `lock-release.sh` in bash command
    - Map exit codes:
      - `lock-try.sh` exit 0 -> ACQUIRED
      - `lock-try.sh` exit 1 -> WAITING (lock held by another)
      - `lock-wait.sh` exit 0 -> ACQUIRED
      - `lock-wait.sh` exit 1 -> clear wait status (timeout)
      - `lock-release.sh` exit 0 -> RELEASED
    - Ignore exit code 2 (error) - log warning
    - Canonicalize lock path using `locking._canonicalize_path` (or expose as public helper)
  - `src/infra/hooks/__init__.py`: Export `make_lock_event_hook`
- **Verification**:
  - **New**: `tests/infra/hooks/test_deadlock_hook.py`
    - Test: `lock-try.sh path` with exit 0 emits ACQUIRED
    - Test: `lock-try.sh path` with exit 1 emits WAITING
    - Test: `lock-release.sh path` with exit 0 emits RELEASED
    - Test: `lock-wait.sh path` with exit 0 emits ACQUIRED
    - Test: Non-lock bash commands are ignored
    - Test: Exit code 2 logs warning, no event
  - Run: `uv run pytest tests/infra/hooks/test_deadlock_hook.py -v -m unit`
- **Rollback**: Delete `src/infra/hooks/deadlock.py` and tests

### Task 4: Wire DeadlockMonitor into Orchestrator
- **Goal**: Maintain WFG state, detect cycles, and resolve deadlocks
- **Depends on**: Tasks 1, 3
- **Changes**:
  - `src/orchestration/orchestrator.py`:
    - Add `deadlock_monitor: DeadlockMonitor | None` attribute
    - Initialize in `_init_runtime_state()` if deadlock detection enabled
    - In `run_implementer()`: call `deadlock_monitor.register_agent(agent_id, issue_id)` after agent_id assigned
    - In `_cleanup_agent_locks()`: also call `deadlock_monitor.unregister_agent(agent_id)`
    - Add `_handle_deadlock(info: DeadlockInfo)` async method:
      - Log warning: "Deadlock detected: cycle={info.cycle} victim={info.victim_id}"
      - Cancel victim task if still running: `active_tasks[victim_issue_id].cancel()`
      - Call `_cleanup_agent_locks(info.victim_id)`
      - Call `await beads.add_dependency_async(info.victim_issue_id, info.blocker_issue_id)`
      - Call `await beads.mark_needs_followup_async(info.victim_issue_id, f"Deadlock: blocked on {info.blocker_issue_id}")`
      - Emit deadlock event: `event_sink.on_deadlock_detected(info)`
    - Set `deadlock_monitor.on_deadlock = self._handle_deadlock`
    - Guard resolution with `asyncio.Lock` to prevent multiple resolutions for same cycle
- **Verification**:
  - Integration tests in Task 7
- **Rollback**: Revert orchestrator changes

### Task 5: Hook Wiring in Orchestration
- **Goal**: Ensure lock event hook is installed for all agent sessions
- **Depends on**: Tasks 2, 3, 4
- **Changes**:
  - `src/orchestration/orchestration_wiring.py`:
    - Update `WiringDependencies` to include `deadlock_monitor: DeadlockMonitor | None`
    - Update `build_session_config()` to accept `deadlock_monitor` and `deadlock_detection_enabled`
    - Create `make_lock_event_hook` with callback that calls `deadlock_monitor.handle_event()`
  - `src/pipeline/agent_session_runner.py`:
    - Accept `post_tool_hooks` in `SessionConfig`
    - Update `_build_hooks()` to include deadlock hook if enabled
- **Verification**:
  - Integration tests ensure hook executes for lock scripts
- **Rollback**: Revert wiring changes

### Task 6: Event Protocol Updates
- **Goal**: Expose deadlock detection/resolution via event sink
- **Depends on**: Task 4
- **Changes**:
  - `src/infra/io/event_protocol.py`:
    - Add `on_deadlock_detected(self, info: DeadlockInfo) -> None` to MalaEventSink protocol
    - Document fields: cycle, victim_id, victim_issue_id, blocker_id, blocker_issue_id
  - `src/infra/io/console_sink.py` (or equivalent): Implement `on_deadlock_detected` with warning log
  - `src/infra/io/null_sink.py` (if exists): Add no-op implementation
- **Verification**:
  - Update event protocol tests if present
- **Rollback**: Revert event protocol changes

### Task 7: Feature Flag + Config Integration
- **Goal**: Allow disabling deadlock detection via CLI/config
- **Depends on**: Tasks 4, 5
- **Changes**:
  - `src/cli/cli.py`: Add `"deadlock-detection"` to `VALID_DISABLE_VALUES` frozenset
  - `src/infra/io/config.py`:
    - Add `deadlock_detection_enabled: bool = True` to `MalaConfig` or relevant config class
    - Wire `--disable-validations=deadlock-detection` to set this flag to False
  - `src/orchestration/orchestrator.py`: Check config flag before initializing DeadlockMonitor
- **Verification**:
  - Unit tests for config parsing / CLI disable set
  - Test: `--disable-validations=deadlock-detection` disables monitor
- **Rollback**: Revert CLI/config changes

### Task 8: Integration Tests
- **Goal**: Validate end-to-end deadlock detection + resolution
- **Depends on**: Tasks 1-7
- **Changes**:
  - **New**: `tests/integration/test_deadlock_integration.py`
    - Test setup: Mock BeadsClient, mock agent sessions
    - Simulate 2 agents: A holds L1, B holds L2
    - A waits on L2 (WAITING event), B waits on L1 (WAITING event) -> deadlock detected
    - Assert youngest agent cancelled
    - Assert `cleanup_agent_locks()` called for victim
    - Assert `add_dependency()` called with correct args
    - Assert `mark_needs_followup()` called
    - Assert `on_deadlock_detected()` event emitted
    - Test: Single agent never triggers deadlock (no cycle possible)
    - Test: 3-agent cycle (A->B->C->A) detected and resolved
- **Verification**:
  - Run: `uv run pytest tests/integration/test_deadlock_integration.py -v -m integration`
  - Run: `uv run pytest -m "unit or integration" --cov --cov-fail-under=85`
- **Rollback**: Delete integration test file

## Risks, Edge Cases & Breaking Changes

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Incorrect lock key normalization causes missed cycles | Medium | Medium | Reuse `locking._canonicalize_path` logic; add path normalization tests |
| PostToolUse hook mis-parses bash command | Medium | Medium | Unit tests on command parsing with various formats |
| Deadlock resolution fires multiple times | Low | Medium | Guard with asyncio.Lock, check task.done() before cancel |
| Victim task already done before cancel | Low | Low | Check `task.done()`; if done, pick next youngest or no-op |
| Beads API failure blocks resolution | Low | Medium | Log error; still cancel victim + cleanup locks |
| Event lost (agent crash) | Medium | Low | Orchestrator cleanup already removes agent from tracking; add unregister_agent call |

### Edge Cases

- **Agent crashes mid-wait**: `_cleanup_agent_locks` already called; add `deadlock_monitor.unregister_agent()` to cleanup path
- **Lock-wait timeout**: Emit "clear wait" event or no-op to avoid stale waits_for entries
- **Self-deadlock (agent waits on own lock)**: Detected as cycle of length 1; should not happen due to lock ownership check in `get_lock_holder`
- **Literal keys (`__test_mutex__`)**: Ensure canonicalization preserves literal form (already handled in `_canonicalize_path`)
- **Multi-party cycles (A->B->C->A)**: DFS handles arbitrary cycle lengths
- **Lock released between detect and resolve**: Re-check graph state before resolution

### Breaking Changes

- **None expected**: All additions are new code paths
- Existing single-agent workflows unaffected (no cycle possible with 1 agent)
- Hook additions are opt-in via feature flag (enabled by default)

## Testing & Validation Strategy

### Unit Tests
- `tests/domain/test_deadlock.py`: WFG cycle detection, hold/wait updates, victim selection
- `tests/infra/hooks/test_deadlock_hook.py`: Lock command parsing, exit code mapping

### Integration Tests
- `tests/integration/test_deadlock_integration.py`: Multi-agent deadlock detection and resolution

### Regression Tests
- Ensure existing lock tests (`tests/test_lock_integration.py`) still pass
- Ensure orchestration tests (`tests/test_orchestration_helpers.py`) still pass

### Manual Verification
- Run `mala` with 2 agents on issues that edit overlapping files
- Force deadlock by having both agents acquire locks in opposite order
- Observe log output: "Deadlock detected", victim selection, dependency added

### Monitoring / Observability
- Emit `on_deadlock_detected` event with cycle and victim metadata
- Log warnings on hook parse errors or resolution failures
- Log info when deadlock is resolved successfully

## Rollback Strategy

1. **Quick Disable**: Use `--disable-validations=deadlock-detection` CLI flag
2. **Config Disable**: Set `deadlock_detection_enabled: false` in config
3. **Full Revert**: Revert code changes (no database/external state involved)

### Rollback Safety
- No persistent state: graph lives in memory only
- No beads schema changes: uses existing `add_dependency` API
- Safe to disable mid-run: existing locks/agents continue normally
- Agents fall back to timeout-based handling if deadlock detection disabled

## Open Questions

| Question | Status | Resolution |
|----------|--------|------------|
| PostToolUse hook input schema for exit code? | Open | Need to verify SDK `tool_result` dict structure |
| Should lock-wait timeout emit "clear wait" event? | Open | Recommend yes - clear wait status to avoid stale entries |
| Expose `_canonicalize_path` as public helper? | Open | Recommend yes - rename to `canonicalize_lock_path()` |
| Exact event protocol method signature? | Open | Propose `on_deadlock_detected(info: DeadlockInfo)` |
| For cycles >2, add dependency to all blockers? | Open | Current plan: dependency only to immediate blocker; can iterate |

## Appendix: Synthesis Decisions

| Decision | Resolution | Source |
|----------|------------|--------|
| Detection approach | Centralized Wait-For Graph + DFS | All drafts agree |
| Hook type | PostToolUse (not PreToolUse) - need exit code | Codex/Claude drafts |
| Victim selection | Youngest agent by start_time | All drafts agree |
| Resolution steps | Cancel task, cleanup locks, add dependency, mark followup | All drafts agree |
| Cycle detection cost | O(n) per WAITING event (acceptable) | Skeleton constraints |
| Event emission | Via PostToolUse hook to orchestrator | All drafts agree |
| Feature flag | `--disable-validations=deadlock-detection` | Codex draft + existing pattern |
| lock-try.sh exit 1 interpretation | WAITING (blocked by another holder) | Claude draft |
| Config flag name | `deadlock_detection_enabled` | Gemini draft |
