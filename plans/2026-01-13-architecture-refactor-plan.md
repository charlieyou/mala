# Implementation Plan: Architecture Refactor (High + Medium Priority Issues)

## Context & Goals

- **Spec**: `plans/architecture-review-2026-01-12.md`
- Address remaining high-priority architecture issues from multi-model review
- Complete the factory pattern migration by extracting lifecycle handling
- Replace callback dataclass with explicit Protocols
- Split oversized `protocols.py` into domain-focused modules
- Extract trigger and fixer concerns from `RunCoordinator`

## Scope & Non-Goals

### In Scope
- [High] Complete MalaOrchestrator refactor - extract lifecycle/signal handling
- [High] Replace `SessionCallbacks` dataclass with explicit Protocols
- [Medium] Extract trigger policy and fixer service from `RunCoordinator`
- [Medium] Split `protocols.py` (~2400 LOC) into <300 LOC modules

### Out of Scope (Non-Goals)
- Console globals refactor (deferred per user decision)
- LOC reduction targets (focus on cohesion, not line counts)
- [Low] Domain logic in CLI (`parse_scope`) - defer to future
- [Low] `MalaEventSink` splitting - defer to future
- Config parsing duplication - lower priority, defer
- Adding new functionality beyond extraction/reorganization
- Changing behavior of extracted components

## Assumptions & Constraints

- Layer hierarchy enforced by import-linter: `cli → orchestration → pipeline → domain → infra → core`
- Existing factory pattern (`create_orchestrator()` with `OrchestratorDependencies`) preserved
- DI via dataclasses pattern continues (`SessionRunContext`, `RuntimeDeps`, `PipelineConfig`)
- Protocol-based interfaces with `@runtime_checkable` decorator

### Implementation Constraints
- **No backward-compatibility shims**: Update all imports directly per CLAUDE.md (no re-exports)
- **Single PR delivery**: All 4 work packages in one PR
- **Preserve layer boundaries**: New modules must respect existing layer hierarchy
- **Minimal behavioral change**: Extractions should not alter runtime behavior
- **Preserve existing patterns**: Factory functions, DI via dataclasses, Protocol interfaces

### Testing Constraints
- All existing tests must pass
- Import-linter contracts must pass (`uvx import-linter`)
- Type checking must pass (`uvx ty check`)
- Linting must pass (`uvx ruff check .`)
- New classes (`TriggerEngine`, `FixerService`, `LifecycleController`) should have direct unit tests

## Integration Analysis

### Existing Mechanisms Considered

| Existing Mechanism | Could Serve Feature? | Decision | Rationale |
|--------------------|---------------------|----------|-----------|
| `src/core/protocols.py` | Partial | Split into submodules | File too large (2400 LOC), splitting preserves interface definitions |
| `SessionCallbacks` dataclass | Yes | Replace with Protocols | Callbacks naturally group into 3 responsibilities |
| `OrchestratorDependencies` | Yes | Extend | Add lifecycle controller |
| `SessionRunContext` | Yes | Keep | Already consolidates context |
| Factory pattern | Yes | Extend | Use for new services |
| `RunCoordinator` class | Partial | Refactor | Class is over-responsible (1762 LOC); logic should be delegated |
| `DeadlockHandler` pattern | Yes | Model for LifecycleController | Callback-based extraction pattern proven |
| Import-linter | Yes | Extend | Add contracts for new boundaries |

### Integration Approach
This refactor extends existing infrastructure by:
1. **Protocol split**: Create submodule directory, move protocols, update imports
2. **SessionCallbacks**: Define new Protocols in appropriate module, update AgentSessionRunner to accept them
3. **LifecycleController**: Follow DeadlockHandler pattern - extract class with callbacks, wire in orchestrator factory
4. **TriggerEngine/FixerService**: Extract classes from RunCoordinator, inject as dependencies via factory

## Prerequisites

- [x] Architecture review findings available (`plans/architecture-review-2026-01-12.md`)
- [x] User decisions on scope and approach confirmed
- [x] Existing patterns documented (factory, DI, protocols)
- [x] Current tests passing
- [x] Understanding of SIGINT handling flow
- [ ] No pending changes to affected files (clean git state before starting)
- [ ] No other active branches heavily modifying `protocols.py` (merge conflict risk)

## High-Level Approach

**Priority order**: Protocols first → Callbacks → Extract services

1. **WP1 (Protocol Split)**: Create `src/core/protocols/` directory with domain-focused modules. Move protocols to appropriate modules. Update all imports throughout codebase. This has the largest blast radius so should be done first.

2. **WP2 (SessionCallbacks)**: Define `IGateRunner`, `IReviewRunner`, `ISessionLifecycle` protocols. Update `AgentSessionRunner` to accept protocols instead of dataclass. Update callers.

3. **WP3 (LifecycleController)**: Extract `LifecycleController` class with signal/interrupt handling. Wire into orchestrator via dependency injection. Move SIGINT escalation state and handlers.

4. **WP4 (TriggerEngine + FixerService)**: Extract `TriggerEngine` class for event-based trigger decisions. Extract `FixerService` class for fixer agent spawning. Wire both into `RunCoordinator` as dependencies.

## Technical Design

### Work Package 1: Split protocols.py

**Goal**: Split 2400 LOC into domain-focused modules <300 LOC each

**Current state**:
- 30+ protocols in single file
- Mixed concerns (issue, review, validation, infra, SDK, events)

**Proposed split** (8 modules in `src/core/protocols/`):

| Module | Contents |
|--------|----------|
| `issue.py` | IssueProvider, IssueResolutionProtocol |
| `review.py` | CodeReviewer, ReviewResultProtocol, ReviewIssueProtocol, IReviewRunner |
| `validation.py` | GateChecker, ValidationSpecProtocol, ValidationEvidenceProtocol, Gate*Protocol, IGateRunner |
| `infra.py` | CommandRunnerPort, LockManagerPort, EnvConfigPort, LoggerPort |
| `sdk.py` | SDKClientProtocol, SDKClientFactoryProtocol |
| `events.py` | MalaEventSink, TriggerSummary, EventRunConfig |
| `lifecycle.py` | DeadlockInfoProtocol, DeadlockMonitorProtocol, LockEventProtocol, ISessionLifecycle |
| `log.py` | LogProvider, JsonlEntryProtocol |

**Re-export strategy**: No re-exports per repo policy. `__init__.py` will be empty. All imports must use direct paths like `from src.core.protocols.review import CodeReviewer`.

**Files**:
- `src/core/protocols/` — **New** (directory)
- `src/core/protocols/__init__.py` — **New** (empty file, no re-exports)
- `src/core/protocols/issue.py` — **New**
- `src/core/protocols/review.py` — **New**
- `src/core/protocols/validation.py` — **New**
- `src/core/protocols/infra.py` — **New**
- `src/core/protocols/sdk.py` — **New**
- `src/core/protocols/events.py` — **New**
- `src/core/protocols/lifecycle.py` — **New**
- `src/core/protocols/log.py` — **New**
- `src/core/protocols.py` — **Delete** (after migration complete)

### Work Package 2: Replace SessionCallbacks with Protocols

**Goal**: Replace the 10-field callback dataclass with 3 typed Protocol interfaces

**Current state**:
```python
@dataclass
class SessionCallbacks:
    on_gate_check: GateCheckCallback | None
    on_review_check: ReviewCheckCallback | None
    on_review_no_progress: ReviewNoProgressCallback | None
    get_log_path: Callable[[str], Path] | None
    get_log_offset: LogOffsetCallback | None
    on_abort: Callable[[str], None] | None
    on_tool_use: ToolUseCallback | None
    on_agent_text: AgentTextCallback | None
    on_session_end_check: SessionEndCheckCallback | None
    get_abort_event: Callable[[], asyncio.Event | None] | None
```

**Proposed Protocols** (coarse-grained, 2-3 protocols):

```python
# In src/core/protocols/validation.py
@runtime_checkable
class IGateRunner(Protocol):
    """Protocol for gate checking operations."""

    async def run_gate_check(
        self, issue_id: str, log_path: Path, retry_state: RetryState
    ) -> tuple[GateOutcome, int]: ...

    async def run_session_end_check(
        self, issue_id: str, log_path: Path, retry_state: SessionEndRetryState
    ) -> SessionEndResult: ...

# In src/core/protocols/review.py
@runtime_checkable
class IReviewRunner(Protocol):
    """Protocol for review operations."""

    async def run_review(
        self, issue_id: str, description: str | None, session_id: str | None,
        retry_state: RetryState, author_context: str | None,
        previous_findings: Sequence[ReviewIssueProtocol] | None,
        session_end_result: SessionEndResult | None,
    ) -> ReviewOutcome: ...

    def check_no_progress(
        self, log_path: Path, log_offset: int,
        prev_commit: str | None, curr_commit: str | None
    ) -> bool: ...

# In src/core/protocols/lifecycle.py
@runtime_checkable
class ISessionLifecycle(Protocol):
    """Protocol for session lifecycle operations."""

    def get_log_path(self, session_id: str) -> Path: ...
    def get_log_offset(self, log_path: Path, start_offset: int) -> int: ...
    def on_abort(self, reason: str) -> None: ...
    def get_abort_event(self) -> asyncio.Event | None: ...
    def on_tool_use(self, agent_id: str, tool_name: str, args: dict) -> None: ...
    def on_agent_text(self, agent_id: str, text: str) -> None: ...
```

**Files**:
- `src/pipeline/agent_session_runner.py` — Exists (modify to use Protocol interfaces)
- `src/core/protocols/validation.py` — **New** (contains IGateRunner)
- `src/core/protocols/review.py` — **New** (contains IReviewRunner)
- `src/core/protocols/lifecycle.py` — **New** (contains ISessionLifecycle)

### Work Package 3: Extract LifecycleController from Orchestrator

**Goal**: Move signal handling and lifecycle management out of `MalaOrchestrator`

**Current state**:
- `_handle_sigint`, `_interrupt_event`, lifecycle hooks in orchestrator
- Orchestrator owns too many concerns
- SIGINT has 3-stage escalation behavior (interrupt → drain → abort)

**Methods/state to extract**:
- `interrupt_event: asyncio.Event`
- `drain_event: asyncio.Event`
- `_sigint_count`, `_sigint_last_at` (escalation tracking)
- `_drain_mode_active`, `_abort_mode_active` flags
- `_shutdown_requested` flag
- `handle_sigint()` method
- `request_abort()` method
- `is_interrupted()` method
- `is_drain_mode()` method
- `reset()` method

**LifecycleController interface**:
```python
@dataclass
class LifecycleController:
    """Manages SIGINT escalation and interrupt coordination."""

    interrupt_event: asyncio.Event
    drain_event: asyncio.Event

    # State
    _sigint_count: int = 0
    _sigint_last_at: float = 0.0
    _drain_mode_active: bool = False
    _abort_mode_active: bool = False
    _abort_exit_code: int = 130
    _shutdown_requested: bool = False

    def handle_sigint(self, loop: AbstractEventLoop) -> None:
        """Handle SIGINT with 3-stage escalation."""
        ...

    def request_abort(self, reason: str) -> None:
        """Request orderly abort."""
        ...

    def is_interrupted(self) -> bool:
        """Check if interrupt requested."""
        ...

    def is_drain_mode(self) -> bool:
        """Check if in drain mode."""
        ...

    def reset(self) -> None:
        """Reset state for new run."""
        ...
```

**Signal registration ownership**: Orchestrator wires the signals (calls `loop.add_signal_handler`), LifecycleController contains the logic (methods called by signal handlers).

**Files**:
- `src/orchestration/orchestrator.py` — Exists (remove lifecycle logic, use LifecycleController)
- `src/orchestration/lifecycle_controller.py` — **New** (create)
- `src/orchestration/orchestration_wiring.py` — Exists (wire LifecycleController)

### Work Package 4: Extract TriggerEngine and FixerService

**Goal**: Separate trigger policy and fixer spawning from `RunCoordinator`

**Current state**:
- `RunCoordinator` has `queue_trigger_validation`, `run_trigger_validation`, `_run_fixer_agent`
- Mixed coordination and business logic
- 1762 LOC, over-responsible

**TriggerEngine (event-based interface)**:
```python
@dataclass
class TriggerActions:
    """Result of trigger evaluation."""
    should_run: bool
    commands: list[ResolvedCommand]
    failure_mode: FailureMode

class TriggerEngine:
    """Event-based trigger decision engine."""

    def on_issue_completed(self, issue_id: str, result: IssueResult) -> TriggerActions | None:
        """Evaluate triggers when an issue completes."""
        ...

    def on_epic_closed(self, epic_id: str) -> TriggerActions | None:
        """Evaluate triggers when an epic closes."""
        ...

    def on_run_end(self, success_count: int, total_count: int) -> TriggerActions | None:
        """Evaluate triggers at end of run."""
        ...

    def resolve_commands(self, trigger_config: BaseTriggerConfig) -> list[ResolvedCommand]:
        """Resolve command templates to concrete commands."""
        ...
```

**FixerService interface**:
```python
class FixerService:
    """Spawns and manages fixer agents for validation failures."""

    async def run_fixer(
        self,
        failure_context: FailureContext,
        interrupt_event: asyncio.Event | None = None,
    ) -> FixerResult:
        """Run a fixer agent for the given failure."""
        ...

    def cleanup_locks(self) -> None:
        """Clean up any locks held by fixer."""
        ...
```

**Delegation pattern**: RunCoordinator emits events to TriggerEngine, receives TriggerActions, executes them. RunCoordinator delegates fixer execution to FixerService.

**Service lifetime**: Stateless services constructed per run (not long-lived). Concurrency handled by caller (RunCoordinator).

**Files**:
- `src/pipeline/run_coordinator.py` — Exists (remove extracted logic; use injected services)
- `src/pipeline/trigger_engine.py` — **New** (create)
- `src/pipeline/fixer_service.py` — **New** (create)

### Architecture

```
src/core/
├── protocols/                    # NEW directory
│   ├── __init__.py              # Empty (no re-exports per CLAUDE.md)
│   ├── issue.py                 # IssueProvider, IssueResolutionProtocol
│   ├── review.py                # CodeReviewer, Review*Protocol, IReviewRunner
│   ├── validation.py            # GateChecker, Validation*Protocol, IGateRunner
│   ├── infra.py                 # CommandRunnerPort, LockManagerPort, etc.
│   ├── sdk.py                   # SDKClientProtocol, SDKClientFactoryProtocol
│   ├── events.py                # MalaEventSink, TriggerSummary, EventRunConfig
│   ├── lifecycle.py             # Deadlock*Protocol, ISessionLifecycle
│   └── log.py                   # LogProvider, JsonlEntryProtocol
└── protocols.py                 # DELETED after migration

src/pipeline/
├── agent_session_runner.py      # Uses new Protocol interfaces
├── run_coordinator.py           # Delegates to TriggerEngine, FixerService
├── trigger_engine.py            # NEW: Event-based trigger decisions
└── fixer_service.py             # NEW: Fixer agent spawning

src/orchestration/
├── orchestrator.py              # Delegates to LifecycleController
└── lifecycle_controller.py      # NEW: Signal/interrupt handling
```

### Data Model
N/A - purely code structure refactor. No new persistent data or state transitions.

### API/Interface Design
See Work Package sections above for detailed interface definitions.

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/core/protocols/` | **New** | Directory for protocol submodules |
| `src/core/protocols/__init__.py` | **New** | Empty file (no re-exports) |
| `src/core/protocols/issue.py` | **New** | Issue-related protocols |
| `src/core/protocols/review.py` | **New** | Review-related protocols, IReviewRunner |
| `src/core/protocols/validation.py` | **New** | Validation-related protocols, IGateRunner |
| `src/core/protocols/infra.py` | **New** | Infra/Port protocols |
| `src/core/protocols/sdk.py` | **New** | SDK protocols |
| `src/core/protocols/events.py` | **New** | Event protocols |
| `src/core/protocols/lifecycle.py` | **New** | Lifecycle protocols, ISessionLifecycle |
| `src/core/protocols/log.py` | **New** | Log-related protocols |
| `src/core/protocols.py` | **Delete** | Remove after migration |
| `src/pipeline/trigger_engine.py` | **New** | TriggerEngine class |
| `src/pipeline/fixer_service.py` | **New** | FixerService class |
| `src/orchestration/lifecycle_controller.py` | **New** | LifecycleController class |
| `src/pipeline/run_coordinator.py` | Modify | Remove extracted logic; use injected services |
| `src/orchestration/orchestrator.py` | Modify | Remove lifecycle logic; use LifecycleController |
| `src/pipeline/agent_session_runner.py` | Modify | Replace SessionCallbacks with Protocol interfaces |
| `src/orchestration/orchestrator_state.py` | Modify | Update imports |
| `src/orchestration/orchestration_wiring.py` | Modify | Wire new dependencies (LifecycleController, TriggerEngine, FixerService) |
| `tests/unit/pipeline/test_trigger_engine.py` | **New** | Tests for TriggerEngine |
| `tests/unit/pipeline/test_fixer_service.py` | **New** | Tests for FixerService |
| `tests/unit/orchestration/test_lifecycle_controller.py` | **New** | Tests for LifecycleController |
| ~30 other files | Modify | Update protocol imports |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes
- **Circular imports during migration**: Careful ordering of protocol splits required. Migrate leaf protocols first (infra), then dependent ones (validation, review). Use `TYPE_CHECKING` blocks aggressively. Keep `core/protocols` dependency-free (pure interfaces).
- **Missing import updates**: Grep for all `from src.core.protocols import` to ensure complete migration.
- **Protocol compatibility**: Ensure extracted Protocols maintain exact same signatures to avoid runtime errors.
- **SIGINT during migration**: LifecycleController must preserve exact 3-stage escalation timing and behavior (interrupt → drain → abort).
- **Signal handling initialization**: Moving signal handling to LifecycleController must ensure it's still initialized at the right time (main thread). Verify initialization order in `Orchestrator.run()`.
- **Test coupling**: Some tests may rely on current callback structure - update mocks and fixtures.

### Breaking Changes & Compatibility
- **Potential Breaking Changes**:
  - Import paths change from `src.core.protocols` to `src.core.protocols.<module>`
  - `SessionCallbacks` replaced with Protocol interfaces
  - `RunCoordinator` constructor changes (accepts TriggerEngine, FixerService)
  - `Orchestrator` constructor changes (accepts LifecycleController)

- **Mitigations**:
  - All imports updated in same PR (atomic migration)
  - Protocol interfaces match SessionCallbacks signatures exactly
  - Wiring functions updated to construct new dependencies
  - No public API changes (internal refactor only)
  - Import-linter contracts must pass after changes

## Testing & Validation Strategy

- **Unit Tests**
  - **New**: `tests/unit/pipeline/test_trigger_engine.py` (verify trigger logic independent of coordinator)
  - **New**: `tests/unit/pipeline/test_fixer_service.py` (verify fixer execution logic)
  - **New**: `tests/unit/orchestration/test_lifecycle_controller.py` (verify SIGINT escalation, event state)
  - **Update**: `tests/unit/pipeline/test_run_coordinator.py` (mock new dependencies)
  - **Update**: `tests/pipeline/test_agent_session_runner.py` (update for Protocol interfaces)

- **Integration / End-to-End Tests**
  - Run existing `tests/e2e` suite to ensure the system still behaves identically from the outside
  - Run full test suite to catch import errors

- **Regression Tests**
  - SIGINT behavior must match exactly (3-stage escalation, timing)
  - Trigger validation must match exactly (fire_on, failure_mode)
  - Fixer behavior must match exactly (retry logic, lock cleanup)

- **Linter/Type Check**
  - Run `import-linter` to verify the new `src/core/protocols/` structure respects layer boundaries
  - Run `uvx ty check` for type checking
  - Run `uvx ruff check .` for linting

- **Manual Verification**
  - Run `mala run` with Ctrl-C at each stage to verify escalation
  - Run `mala run` with validation failures to verify fixer spawning
  - Run `mala run --watch` to verify trigger firing

- **Monitoring / Observability**
  - Existing event_sink events continue to fire correctly
  - Log output unchanged

### Acceptance Criteria Coverage

| Spec AC | Covered By |
|---------|------------|
| protocols.py split into domain modules | WP1: 8 new protocol modules in File Impact Summary |
| Each protocol file <300 LOC | Split into 8 modules (avg ~300 LOC each) |
| No backward-compatibility shims | CLAUDE.md constraint, direct import updates, empty `__init__.py` |
| SessionCallbacks replaced by Protocols | WP2: IGateRunner, IReviewRunner, ISessionLifecycle |
| Orchestrator accepts all deps, focuses on sequencing | WP3: LifecycleController extraction |
| Only one place decides trigger fire conditions | WP4: TriggerEngine |
| LifecycleController extracted | WP3: Architecture section, Data Model |
| TriggerEngine event-based interface | WP4: Data Model (TriggerActions), API design |
| FixerService extracted | WP4: Data Model (FixerService class) |
| Import-linter passes | Testing Strategy (run contracts check) |
| Existing tests pass | Testing Strategy (full test suite) |

## Open Questions

- **Protocol grouping granularity**: The proposed 8-module split may benefit from consolidation (e.g., merge `log.py` into `infra.py` if it's small). Final grouping can be refined during implementation if natural affinities emerge.
- **Epic protocols**: Should `EpicVerifierProtocol` and `EpicVerificationModel` go in `validation.py` or a separate `epic.py`? Leaning toward `validation.py` for simplicity.

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
