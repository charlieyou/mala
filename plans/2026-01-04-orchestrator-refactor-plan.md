# Implementation Plan: Orchestrator Refactor

## Context & Goals
- **Spec**: N/A - derived from architecture review findings (High severity: "MalaOrchestrator has excessive responsibilities and state")
- **Objective**: Extract cohesion issues from MalaOrchestrator (~1023 LOC) to improve testability and maintainability
- **Components**: Extract `DeadlockHandler` for deadlock/abort logic and `OrchestratorState` for runtime state management
- **Priority**: Medium - planned tech debt reduction

## Scope & Non-Goals

### In Scope
- Extract deadlock handling logic (`_handle_deadlock`, `_resolve_deadlock`, `_cleanup_agent_locks`, `_abort_active_tasks`) into a dedicated `DeadlockHandler` class
- Encapsulate runtime state (`agent_ids`, `completed`, `_active_session_log_paths`, `_deadlock_cleaned_agents`) into an `OrchestratorState` dataclass
- Migrate `_deadlock_resolution_lock` to be owned by `DeadlockHandler`
- Delete internal tests that mock orchestrator private methods; rewrite as tests for extracted components
- Maintain existing behavior and test coverage

### Out of Scope (Non-Goals)
- Refactoring `AgentSessionRunner` (separate high-severity issue)
- Refactoring `WiringDependencies` (separate medium-severity issue)
- Adding new features or changing behavior
- Modifying domain layer (`src/domain/deadlock.py`) - detection logic stays pure
- Changes to public `MalaOrchestrator` API

## Assumptions & Constraints

### Implementation Constraints
- **Public API unchanged**: `MalaOrchestrator.run()` signature and behavior remain identical
- **Python 3.11+**: Use `dataclasses`, `asyncio`, and type annotations
- **State isolation**: `OrchestratorState` is a pure data container (anemic dataclass)
- **Lock ownership**: `DeadlockHandler` owns the `asyncio.Lock` internally as implementation detail
- **Module location**: `DeadlockHandler` lives in `src/orchestration/` due to orchestration-layer coupling (calls beads, event_sink)
- **Callback pattern**: Follow existing `FinalizerCallbackRefs`, `EpicCallbackRefs` patterns for dependency injection
- **TYPE_CHECKING imports**: Use for circular import prevention as per existing patterns

### Testing Constraints
- **Coverage**: 85% threshold maintained for new and modified files
- **Integration unchanged**: Existing `tests/integration/domain/test_deadlock.py` must pass without modification
- **Unit test isolation**: Mock complex dependencies (BeadsClient, MalaEventSink) when testing `DeadlockHandler`
- **Test migration**: Delete internal tests that mock `_handle_deadlock` etc., rewrite for new components

## Prerequisites
- [x] Architecture review complete (source of this plan)
- [x] Codebase patterns analyzed (callback injection, types.py structure)
- [ ] No blocking dependencies - this is a pure refactor

## High-Level Approach

The refactor extracts two concerns from MalaOrchestrator via a phased PR approach:

**PR1 - Extract OrchestratorState**: Create a dataclass encapsulating per-run mutable state. The orchestrator instantiates this at the start of `run()` and passes it to methods that need state access. This makes state management explicit and supports clean reset between runs.

**PR2 - Extract DeadlockHandler**: Create a service class that receives callbacks for beads and event_sink operations. Move `_handle_deadlock`, `_resolve_deadlock`, `_cleanup_agent_locks`, and `_abort_active_tasks` logic into it. The handler owns the `asyncio.Lock` internally. This allows testing deadlock/abort behavior in isolation.

**PR3 - Cleanup/Integration**: Update remaining tests, verify integration, clean up any temporary compatibility code.

## Technical Design

### Architecture

```
MalaOrchestrator (Facade/Controller)
├── OrchestratorState (passive data holder, created per run)
├── DeadlockHandler (service class, injected at construction)
│   ├── owns asyncio.Lock for deadlock resolution serialization
│   ├── receives callbacks: BeadsClient, MalaEventSink
│   └── receives state reference for reads/writes
└── existing components (IssueCoordinator, ReviewRunner, etc.)
```

**Data Flow**:
1. `MalaOrchestrator.run()` creates `OrchestratorState` instance
2. State reference passed to `DeadlockHandler` methods as needed
3. Handler calls injected callbacks (beads, event_sink) for external effects
4. Handler modifies state (e.g., `deadlock_cleaned_agents.add()`)

### Data Model

#### OrchestratorState (`src/orchestration/orchestrator_state.py`)

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.issue_result import IssueResult

@dataclass
class OrchestratorState:
    """Encapsulates mutable state for a single orchestration run.

    Created at the start of run() and passed to methods that need state.
    Pure data container - no behavior, no locks.
    """
    agent_ids: dict[str, str] = field(default_factory=dict)
    completed: list["IssueResult"] = field(default_factory=list)
    active_session_log_paths: dict[str, Path] = field(default_factory=dict)
    deadlock_cleaned_agents: set[str] = field(default_factory=set)
```

#### DeadlockHandlerCallbacks (`src/orchestration/deadlock_handler.py`)

```python
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path
    from src.domain.deadlock import DeadlockInfo

@dataclass
class DeadlockHandlerCallbacks:
    """Callbacks injected into DeadlockHandler for external operations.

    Follows existing pattern from FinalizerCallbackRefs, EpicCallbackRefs.
    """
    add_dependency: Callable[[str, str], Awaitable[bool]]
    mark_needs_followup: Callable[[str, str, Path | None], Awaitable[None]]
    on_deadlock_detected: Callable[[DeadlockInfo], None]
    on_locks_cleaned: Callable[[str, int], None]
    on_tasks_aborting: Callable[[int, str], None]
    cleanup_agent_locks: Callable[[str], tuple[int, list[str]]]
    unregister_agent: Callable[[str], None]
```

### API/Interface Design

#### DeadlockHandler Class

```python
class DeadlockHandler:
    """Handles deadlock resolution and task abort logic.

    Owns the asyncio.Lock for serializing deadlock resolution.
    Receives callbacks for external operations to avoid direct coupling.
    """

    def __init__(self, callbacks: DeadlockHandlerCallbacks) -> None:
        self._callbacks = callbacks
        self._resolution_lock = asyncio.Lock()

    async def handle_deadlock(
        self,
        info: DeadlockInfo,
        state: OrchestratorState,
        active_tasks: dict[str, asyncio.Task],
    ) -> None:
        """Handle detected deadlock by cancelling victim and recording dependency."""
        ...

    async def abort_active_tasks(
        self,
        state: OrchestratorState,
        active_tasks: dict[str, asyncio.Task],
        abort_reason: str | None,
        finalize_callback: Callable[[str, IssueResult], Awaitable[None]],
    ) -> None:
        """Cancel active tasks and mark them as failed."""
        ...

    def cleanup_agent_locks(self, agent_id: str, state: OrchestratorState) -> None:
        """Remove locks held by agent and track cleanup in state."""
        ...
```

#### Orchestrator Integration

```python
# In MalaOrchestrator.__init__:
self._deadlock_handler = DeadlockHandler(
    callbacks=DeadlockHandlerCallbacks(
        add_dependency=self.beads.add_dependency_async,
        mark_needs_followup=self.beads.mark_needs_followup_async,
        on_deadlock_detected=self.event_sink.on_deadlock_detected,
        on_locks_cleaned=self.event_sink.on_locks_cleaned,
        on_tasks_aborting=self.event_sink.on_tasks_aborting,
        cleanup_agent_locks=cleanup_agent_locks,  # from infra layer
        unregister_agent=lambda aid: self.deadlock_monitor and self.deadlock_monitor.unregister_agent(aid),
    )
)

# In run():
state = OrchestratorState()
# ... pass state to methods that need it
```

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/orchestration/orchestrator_state.py` | **New** | OrchestratorState dataclass |
| `src/orchestration/deadlock_handler.py` | **New** | DeadlockHandler class with callbacks |
| `src/orchestration/orchestrator.py` | Modify | Remove state fields/methods; delegate to new classes |
| `src/orchestration/types.py` | Modify | Add DeadlockHandlerCallbacks if shared |
| `tests/unit/orchestration/test_orchestrator_state.py` | **New** | Unit tests for state initialization |
| `tests/unit/orchestration/test_deadlock_handler.py` | **New** | Unit tests for handler with mock callbacks |
| `tests/unit/orchestration/test_orchestrator.py` | Modify | Update mocks; delete tests for moved private methods |
| `tests/integration/domain/test_deadlock.py` | Exists | Verify unchanged - no modifications expected |

## Risks, Edge Cases & Breaking Changes

### Risks
- **Regression in deadlock handling**: Complex async logic with self-cancellation edge cases
  - *Mitigation*: Preserve exact behavior, copy existing tests first, verify line-by-line
- **State desync**: If handler modifies state but orchestrator bypasses it
  - *Mitigation*: Handler is sole writer for deadlock-related state fields
- **Circular imports**: DeadlockHandler may need types from orchestrator
  - *Mitigation*: Use TYPE_CHECKING imports; shared types go in types.py

### Edge Cases
- **Self-cancellation during deadlock**: Current task is the victim (already handled with deferred cancellation)
- **Concurrent deadlock resolution**: Multiple cycles detected simultaneously (serialized by asyncio.Lock)
- **CancelledError during shielded section**: Must re-raise after completing resolution
- **Task completes during abort**: Use real result rather than marking as aborted

### Breaking Changes & Compatibility
- **Internal API only**: Methods like `_handle_deadlock` are private; moving is safe
- **Public API unchanged**: `run()` signature and return type identical
- **Test fixtures**: Some internal test fixtures may need updating for new component boundaries

## Testing & Validation Strategy

### Unit Tests

**test_orchestrator_state.py**:
- State initializes with empty collections
- Fields are independent (modifying one doesn't affect others)
- Type annotations are correct

**test_deadlock_handler.py**:
- `handle_deadlock` acquires lock, calls callbacks in correct order
- `handle_deadlock` cleans up locks and tracks in state
- `handle_deadlock` cancels victim task (non-self case)
- `handle_deadlock` defers self-cancellation correctly
- `handle_deadlock` shields resolution from cancellation
- `abort_active_tasks` cancels running tasks, uses real results for completed
- `cleanup_agent_locks` is idempotent via state tracking

### Integration Tests
- Existing `tests/integration/domain/test_deadlock.py` must pass unchanged
- Full orchestrator run with deadlock scenario produces same outcomes

### Regression Tests
- Verify all existing `test_orchestrator.py` integration scenarios still pass
- Run e2e tests to verify end-to-end behavior

### Manual Verification
- Run orchestrator with parallel agents that trigger deadlock
- Verify logs show same sequence of events

### Acceptance Criteria Coverage

| Architecture Review Finding | Covered By |
|-----------------------------|------------|
| Extract DeadlockHandler from orchestrator | File existence, code review, unit tests for handler |
| Encapsulate runtime state in OrchestratorState | File existence, code review, unit tests for state |
| Enable isolated deadlock testing | Unit tests with mock callbacks |
| No regression in deadlock resolution | Integration tests, e2e tests |
| No public API changes | Code review of MalaOrchestrator interface |

## Open Questions

- **State reset between runs**: Should `OrchestratorState` have a `reset()` method, or should orchestrator create a new instance per run?
  - *Recommendation*: New instance per run (simpler, avoids mutation bugs)
- **active_tasks ownership**: Should `active_tasks` dict be part of `OrchestratorState` or remain on orchestrator?
  - *Recommendation*: Keep on orchestrator - it's task management, not run state

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` to create Beads issues with dependencies for multi-agent parallelization
- (default) to generate TODO.md checklist for simpler tracking

**Recommended PR sequence**:
1. **PR1**: Extract `OrchestratorState` dataclass, update orchestrator to use it
2. **PR2**: Extract `DeadlockHandler` (including abort logic), wire callbacks
3. **PR3**: Test cleanup, delete obsolete internal tests, verify coverage
