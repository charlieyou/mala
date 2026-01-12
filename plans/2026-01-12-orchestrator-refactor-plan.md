# Implementation Plan: Orchestrator Refactor (High-Severity Architecture Issues)

## Context & Goals
- **Spec**: Derived from architecture review at `plans/architecture-review-2026-01-12.md`
- Address two High-severity findings: God Object orchestrator + callback lambda capture
- Complete existing factory pattern by moving `_build_runtime_deps()` to factory
- Replace 10-parameter lambda soup in callback wiring with `SessionRunContext` dataclass

## Scope & Non-Goals

### In Scope
- Move `CommandRunner`, `EnvConfig`, `LockManager` construction from orchestrator to `create_orchestrator()` factory
- Eliminate `MalaOrchestrator._build_runtime_deps()` method
- Create `SessionRunContext` frozen dataclass in `src/orchestration/types.py`
- Refactor `build_session_callback_factory()` from 10 getter params to context object
- Update `SessionCallbackFactory` to accept/use `SessionRunContext`
- Add unit tests for factory RuntimeDeps construction
- Add unit tests for `SessionRunContext`
- Update existing wiring tests to work with refactored code

### Out of Scope (Non-Goals)
- Medium/Low severity issues from architecture review (RunCoordinator mixed concerns, console globals, etc.)
- Creating a Protocol for `SessionRunContext` (user decision: dataclass sufficient)
- Major orchestrator class restructuring beyond `_build_runtime_deps` extraction
- Splitting `protocols.py` into multiple files (separate issue)
- Changes to the public CLI interface
- Performance optimization

## Assumptions & Constraints

- The factory pattern is already partially implemented; this completes it
- `RuntimeDeps` dataclass in `types.py` already has `command_runner`, `env_config`, `lock_manager` fields
- `create_orchestrator()` is the primary entry point and can assume responsibility for dependency creation
- Late-bound getter pattern (`Callable[[], T]`) is established and should be followed
- All existing tests must continue to pass

### Implementation Constraints
- **Backwards Compatibility**: `create_orchestrator()` signature must remain compatible; external callers don't need code changes
- **Architecture**: Use frozen dataclasses for configuration/context objects (matches `RuntimeDeps`, `PipelineConfig` patterns)
- **Style**: No re-export shims per repo policy; import directly from defining modules
- Extend `create_orchestrator()` in `factory.py`, don't create new factory classes
- Factory fills gaps when partial `OrchestratorDependencies` provided (some fields None)

### Testing Constraints
- **Coverage**: Maintain 72% minimum coverage threshold
- **Philosophy**: Fakes over mocks per `tests/CLAUDE.md`
- New factory tests should cover: full defaults, partial deps, all deps provided
- Update existing tests as needed for refactored signatures

## Integration Analysis

### Existing Mechanisms Considered

| Existing Mechanism | Could Serve Feature? | Decision | Rationale |
|--------------------|---------------------|----------|-----------|
| `factory.py:create_orchestrator()` | Yes | Extend | Already handles most initialization; natural home for `_build_runtime_deps()` logic |
| `types.py:RuntimeDeps` | Yes | Use as-is | Already defines the structure with `command_runner`, `env_config`, `lock_manager` |
| `types.py` | Yes | Add `SessionRunContext` here | Co-locate with other orchestration dataclasses (`RuntimeDeps`, `PipelineConfig`) |
| `orchestration_wiring.py:build_session_callback_factory()` | Yes | Refactor signature | Keep function, change 10 getter params to use context object |
| `session_callback_factory.py:SessionCallbackFactory` | Yes | Refactor constructor | Reduce constructor params via context pattern |

### Integration Approach

Both issues are addressed by extending existing infrastructure rather than creating new systems:

1. **Issue 1 (God Object)**: Move `_build_runtime_deps()` into `create_orchestrator()`, construct `RuntimeDeps` in factory. This aligns with standard Factory pattern and existing codebase conventions.

2. **Issue 2 (Callback lambda)**: Create `SessionRunContext` dataclass in `types.py`, refactor `build_session_callback_factory()` to accept it. This uses the Parameter Object pattern already established with `RuntimeDeps` and `PipelineConfig`.

No parallel systems or new infrastructure required.

## Prerequisites

- [x] Architecture review approved (completed: `plans/architecture-review-2026-01-12.md`)
- [x] `src/orchestration/types.py` exists and can be modified
- [x] User decision made on `SessionRunContext` design (frozen dataclass, single combined context)
- [x] User decision made on partial deps handling (factory fills gaps for None fields)
- [ ] No blocking changes in progress to orchestration module
- [ ] Existing test suite passes (verify before starting)

## High-Level Approach

### Phase 1: Create SessionRunContext
Define `SessionRunContext` frozen dataclass in `types.py` that bundles the 10 late-bound getters currently passed individually to `build_session_callback_factory()`.

### Phase 2: Refactor Callback Wiring
Update `build_session_callback_factory()` and `SessionCallbackFactory` to accept `SessionRunContext` instead of individual parameters. Update `_init_pipeline_runners()` in orchestrator to construct the context object.

### Phase 3: Complete Factory Pattern
Move `CommandRunner()`, `EnvConfig()`, `LockManager()` construction from `MalaOrchestrator._build_runtime_deps()` to `create_orchestrator()` in `factory.py`. Update orchestrator to receive complete `RuntimeDeps` from factory. Delete `_build_runtime_deps()`.

### Phase 4: Update Tests
Add new unit tests for factory RuntimeDeps construction. Add tests for `SessionRunContext`. Update existing wiring tests to work with refactored signatures.

## Technical Design

### Architecture

#### Issue 1: Factory Pattern Completion

**Current flow:**
```
create_orchestrator() -> MalaOrchestrator.__init__() -> _init_from_factory()
                                                          |-> _build_runtime_deps()
                                                          |   creates CommandRunner, EnvConfig, LockManager
                                                          |-> _init_pipeline_runners()
```

**Target flow:**
```
create_orchestrator() -> builds RuntimeDeps with CommandRunner, EnvConfig, LockManager
                      -> MalaOrchestrator.__init__() -> _init_from_factory()
                                                          |-> uses passed RuntimeDeps
                                                          |-> _init_pipeline_runners()
```

#### Issue 2: Callback Wiring Refactor

**Current (10 getter parameters):**
```python
orchestrator._init_pipeline_runners() -> build_session_callback_factory(
    runtime, pipeline, async_gate_runner, review_runner,
    log_provider_getter,           # getter 1
    evidence_check_getter,         # getter 2
    on_session_log_path,           # callback 3
    on_review_log_path,            # callback 4
    interrupt_event_getter,        # getter 5
    get_base_sha,                  # getter 6
    cumulative_review_runner,      # optional 7
    get_run_metadata,              # getter 8
    on_abort,                      # callback 9
    abort_event_getter,            # getter 10
)
```

**Target (context object):**
```python
orchestrator._init_pipeline_runners() -> build_session_callback_factory(
    runtime, pipeline, async_gate_runner, review_runner,
    context=SessionRunContext(...)  # Single context object
)
```

### Data Model

#### SessionRunContext (New: `src/orchestration/types.py`)

```python
@dataclass(frozen=True)
class SessionRunContext:
    """Context for session callback wiring.

    Bundles late-bound getters that bridge orchestrator state
    to session callback factory. Frozen for thread safety and
    immutability guarantees.

    All fields are Callables following the late-bound getter pattern
    established in the codebase.
    """
    log_provider_getter: Callable[[], LogProvider]
    evidence_check_getter: Callable[[], GateChecker]
    on_session_log_path: Callable[[str, Path], None]
    on_review_log_path: Callable[[str, str], None]
    interrupt_event_getter: Callable[[], asyncio.Event | None]
    get_base_sha: Callable[[str], str | None]
    get_run_metadata: Callable[[], RunMetadata | None]
    on_abort: Callable[[str], None]
    abort_event_getter: Callable[[], asyncio.Event | None]
```

No changes to `RuntimeDeps` structure needed - it already has the required fields.

### API/Interface Design

#### Factory API Changes

**Signature (unchanged for backward compatibility):**
```python
def create_orchestrator(
    config: OrchestratorConfig,
    *,
    mala_config: MalaConfig | None = None,
    deps: OrchestratorDependencies | None = None,
) -> MalaOrchestrator
```

**Behavior change**: Factory now constructs `CommandRunner`, `EnvConfig`, `LockManager` if not provided via `deps`, and passes complete `RuntimeDeps` to orchestrator.

#### build_session_callback_factory() Changes

**Before (14 total params, 10 getters/callbacks):**
```python
def build_session_callback_factory(
    runtime: RuntimeDeps,
    pipeline: PipelineConfig,
    async_gate_runner: AsyncGateRunner,
    review_runner: ReviewRunner,
    log_provider_getter: Callable,
    evidence_check_getter: Callable,
    on_session_log_path: Callable[[str, Path], None],
    on_review_log_path: Callable[[str, str], None],
    interrupt_event_getter: Callable | None = None,
    get_base_sha: Callable[[str], str | None] | None = None,
    cumulative_review_runner: CumulativeReviewRunner | None = None,
    get_run_metadata: Callable | None = None,
    on_abort: Callable[[str], None] | None = None,
    abort_event_getter: Callable | None = None,
) -> SessionCallbackFactory
```

**After (5 params + context):**
```python
def build_session_callback_factory(
    runtime: RuntimeDeps,
    pipeline: PipelineConfig,
    async_gate_runner: AsyncGateRunner,
    review_runner: ReviewRunner,
    context: SessionRunContext,
    cumulative_review_runner: CumulativeReviewRunner | None = None,
) -> SessionCallbackFactory
```

Note: `cumulative_review_runner` kept separate as it's an optional runner, not a getter.

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/orchestration/types.py` | Exists | Add `SessionRunContext` frozen dataclass |
| `src/orchestration/factory.py` | Exists | Move RuntimeDeps construction from orchestrator; fill gaps for None fields in deps |
| `src/orchestration/orchestrator.py` | Exists | Remove `_build_runtime_deps()`, receive complete RuntimeDeps, update `_init_pipeline_runners()` to build context |
| `src/orchestration/orchestration_wiring.py` | Exists | Refactor `build_session_callback_factory()` signature to accept `SessionRunContext` |
| `src/pipeline/session_callback_factory.py` | Exists | Update `__init__()` to accept context object, update internal usage |
| `tests/unit/orchestration/test_factory.py` | Exists | Add tests for RuntimeDeps construction (full defaults, partial deps, all deps provided) |
| `tests/unit/orchestration/test_orchestration_wiring.py` | Exists | Update tests for new `build_session_callback_factory()` signature |
| `tests/unit/pipeline/test_session_callback_factory.py` | Exists | Update tests for new constructor signature |

## Risks, Edge Cases & Breaking Changes

### Risks

- **Test breakage**: Existing tests may construct `RuntimeDeps` without command_runner/env_config/lock_manager. **Mitigation**: Factory fills gaps with defaults; tests continue to work.
- **Runtime behavior change**: Unlikely since we're moving construction, not changing behavior. **Mitigation**: Run full test suite before/after each phase.
- **Circular imports**: Adding `SessionRunContext` to `types.py` may introduce import cycles. **Mitigation**: Use `TYPE_CHECKING` imports as established in existing code.

### Edge Cases & Failure Modes

- **Partial OrchestratorDependencies**: Factory must handle `None` fields gracefully by constructing defaults. Logic: `deps.field if deps and deps.field else DefaultClass()`.
- **Late-bound getter failures**: Getters in `SessionRunContext` may raise if called before initialization complete. This is existing behavior, no change - let exceptions propagate.
- **Optional fields in context**: Some getters may be None (e.g., `interrupt_event_getter`). Handle at usage sites as currently done.

### Breaking Changes & Compatibility

- **Potential Breaking Changes**:
  - `build_session_callback_factory()` signature changes (internal API)
  - `SessionCallbackFactory.__init__()` signature changes (internal API)
  - `MalaOrchestrator` internal initialization flow changes

- **Mitigations**:
  - All changes are to internal APIs, not public CLI interface
  - `create_orchestrator()` public signature unchanged - maintains backward compatibility
  - `_build_runtime_deps()` is private method (single underscore) - removal is internal refactor
  - Update all call sites in single PR to ensure atomicity

## Testing & Validation Strategy

### Unit Tests

- **`test_factory.py` (expand existing)**:
  - Test `create_orchestrator()` constructs `RuntimeDeps` with defaults when `deps=None`
  - Test factory fills gaps when `deps` has some fields `None`
  - Test factory respects provided values when `deps` fields are set

- **`test_types.py` or `test_factory.py`**:
  - Test `SessionRunContext` creation and immutability (frozen)
  - Test field access patterns

- **`test_orchestration_wiring.py` (update existing)**:
  - Update for new `build_session_callback_factory()` signature
  - Test context passed correctly to `SessionCallbackFactory`

- **`test_session_callback_factory.py` (update existing)**:
  - Update to construct `SessionRunContext` and pass it
  - Verify factory extracts getters correctly from context

### Integration Tests

- Existing orchestrator integration tests should pass unchanged
- Verify end-to-end session runs work with refactored code

### Regression Tests

- All existing tests in `tests/unit/orchestration/` must pass
- All existing tests in `tests/integration/` must pass
- Verify `main.py` entry point still works

### Manual Validation Steps

- Run `uv run mala run --max-issues 1` to verify orchestrator initializes correctly
- Verify SIGINT handling still works (Ctrl+C during run)
- Verify callback wiring produces same behavior as before

### Acceptance Criteria Coverage

| Architecture Review AC | Approach | Covered By |
|------------------------|----------|------------|
| `_build_runtime_deps` eliminated from Orchestrator | Factory constructs RuntimeDeps directly | Factory tests, code review |
| Orchestrator accepts all deps via injection | RuntimeDeps passed to `_init_from_factory()` | Factory tests, integration tests |
| `build_session_callback_factory()` takes context object | SessionRunContext replaces 10 getter params | Wiring tests, signature verification |
| Callbacks don't read orchestrator fields directly | All access via SessionRunContext getters | Code review, wiring tests |
| Backward compatible `create_orchestrator()` | Public signature unchanged | Integration tests pass |
| 72% test coverage maintained | New tests + updated existing tests | Coverage report |

## Open Questions

All key design decisions have been resolved:

- ~~Should `SessionRunContext` be frozen or mutable?~~ **Decided: Frozen** (matches RuntimeDeps pattern, thread-safe)
- ~~Should we add Protocol for `SessionRunContext` or is dataclass sufficient?~~ **Decided: Dataclass only** (structural typing sufficient, avoids protocol proliferation)
- ~~What's the right granularity for the context object (one vs multiple)?~~ **Decided: Single combined context** (simpler API, single place to update)
- ~~Should factory fill gaps for partial deps?~~ **Decided: Yes** (factory constructs defaults for any None fields)

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` -> Beads issues with dependencies for multi-agent execution
- (default) -> TODO.md checklist for simpler tracking
