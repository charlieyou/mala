# Implementation Plan: Split WiringDependencies

## Context & Goals
- **Spec**: N/A - derived from architecture review finding (Medium severity: "WiringDependencies is a large parameter bag mixing runtime deps, config values, and filter flags")
- **Objective**: Split the 27-field `WiringDependencies` dataclass into 3 focused, cohesive dataclasses
- **Source**: Architecture review identified mixing of runtime deps, config values, and filter flags
- **Priority**: Medium - planned tech debt reduction
- **User Decision**: 3-way split into `RuntimeDeps`, `PipelineConfig`, `IssueFilterConfig`

## Scope & Non-Goals

### In Scope
- Create 3 new dataclasses in `src/orchestration/types.py`: `RuntimeDeps`, `PipelineConfig`, `IssueFilterConfig`
- Update `src/orchestration/orchestration_wiring.py` to use the new types in build function signatures
- Update `src/orchestration/orchestrator.py` to construct the new dataclasses
- Remove the monolithic `WiringDependencies` from `orchestration_wiring.py`
- Add unit tests for the new wiring logic

### Out of Scope (Non-Goals)
- Changing any runtime behavior
- Refactoring `MalaOrchestrator` itself (separate plan in progress)
- Refactoring `AgentSessionRunner` (separate high-severity issue)
- Modifying `OrchestratorConfig` or `OrchestratorDependencies` in types.py
- Adding validation logic to the dataclasses

## Assumptions & Constraints

### Implementation Constraints
- **Python 3.11+**: Use `dataclasses` and type annotations
- **No behavior changes**: All build functions must produce identical outputs
- **Existing pattern alignment**: Follow existing `OrchestratorConfig`/`OrchestratorDependencies` pattern in types.py
- **TYPE_CHECKING imports**: Use for circular import prevention
- **User Decision**: New dataclasses go in `types.py` (alongside OrchestratorConfig)

### Testing Constraints
- **Coverage**: 85% threshold maintained for modified files
- **Integration unchanged**: Orchestrator integration tests must pass unchanged
- New test file `tests/unit/orchestration/test_orchestration_wiring.py` must be created

## Prerequisites
- [x] Architecture review complete (source of this plan)
- [x] User decisions on grouping confirmed (3-way split)
- [x] No external dependencies or approvals needed

## High-Level Approach

1. Define the 3 new dataclasses in `src/orchestration/types.py`
2. Update `src/orchestration/orchestration_wiring.py` to use the new dataclasses in function signatures, replacing `WiringDependencies`
3. Update `src/orchestration/orchestrator.py` to create instances of the new dataclasses instead of `WiringDependencies`
4. Remove `WiringDependencies` from `src/orchestration/orchestration_wiring.py`
5. Create unit tests for the wiring module

## Technical Design

### Architecture

The monolithic `WiringDependencies` parameter object is split into three:

```
orchestrator.py
├── RuntimeDeps        # Protocol implementations (injected at runtime)
├── PipelineConfig     # Configuration for pipeline execution
└── IssueFilterConfig  # Issue selection and filtering criteria
```

Build functions receive only the subset they need:
- `build_gate_runner(runtime: RuntimeDeps, pipeline: PipelineConfig)`
- `build_issue_coordinator(filters: IssueFilterConfig, runtime: RuntimeDeps)`

### Data Model

**RuntimeDeps** (8 fields) - protocol implementations injected at runtime:
```python
@dataclass(frozen=True)
class RuntimeDeps:
    """Runtime dependencies (protocol implementations)."""
    quality_gate: GateChecker
    code_reviewer: CodeReviewer
    beads: IssueProvider
    event_sink: MalaEventSink
    command_runner: CommandRunnerPort
    env_config: EnvConfigPort
    lock_manager: LockManagerPort
    mala_config: MalaConfig
```

**PipelineConfig** (11 fields) - configuration values controlling pipeline behavior:
```python
@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for pipeline execution."""
    repo_path: Path  # User decision: repo_path belongs in PipelineConfig
    timeout_seconds: int
    max_gate_retries: int
    max_review_retries: int
    coverage_threshold: float | None
    disabled_validations: set[str] | None
    context_restart_threshold: float
    context_limit: int
    prompts: PromptProvider
    prompt_validation_commands: PromptValidationCommands
    deadlock_monitor: DeadlockMonitor | None = None
```

**IssueFilterConfig** (8 fields) - issue selection and filtering criteria:
```python
@dataclass(frozen=True)
class IssueFilterConfig:
    """Configuration for issue selection."""
    max_agents: int | None
    max_issues: int | None
    epic_id: str | None
    only_ids: set[str] | None
    prioritize_wip: bool
    focus: bool
    orphans_only: bool
    epic_override_ids: set[str]
```

### API/Interface Design

**Modified Functions in `src/orchestration/orchestration_wiring.py`**:

| Function | Current | After |
|----------|---------|-------|
| `build_gate_runner` | `deps: WiringDependencies` | `runtime: RuntimeDeps, pipeline: PipelineConfig` |
| `build_review_runner` | `deps: WiringDependencies` | `runtime: RuntimeDeps, pipeline: PipelineConfig` |
| `build_run_coordinator` | `deps: WiringDependencies` | `runtime: RuntimeDeps, pipeline: PipelineConfig` |
| `build_issue_coordinator` | `deps: WiringDependencies` | `filters: IssueFilterConfig, runtime: RuntimeDeps` |
| `build_session_callback_factory` | `deps: WiringDependencies` | `runtime: RuntimeDeps, pipeline: PipelineConfig` |
| `build_session_config` | `deps: WiringDependencies` | `pipeline: PipelineConfig` |

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/orchestration/types.py` | Modify | Add `RuntimeDeps`, `PipelineConfig`, `IssueFilterConfig` dataclasses |
| `src/orchestration/orchestration_wiring.py` | Modify | Remove `WiringDependencies`, update build function signatures |
| `src/orchestration/orchestrator.py` | Modify | Update `_init_pipeline_runners` to build the 3 new config objects |
| `tests/unit/orchestration/test_orchestrator.py` | Modify | Update test fixtures that construct `WiringDependencies` directly |
| `tests/unit/orchestration/test_orchestration_wiring.py` | **New** | Unit tests for new dataclasses and wiring functions |

## Risks, Edge Cases & Breaking Changes

### Risks
- **Import cycles**: New dataclasses may create import dependencies
  - *Mitigation*: Use TYPE_CHECKING imports in types.py for protocol types
- **Incomplete field mapping**: Missing a field during split
  - *Mitigation*: Verify all 27 fields are accounted for in new structure; type checker will catch misses

### Edge Cases & Failure Modes
- **None critical**: This is a refactor of internal wiring. The main risk is passing the wrong object or missing a field, which static analysis (ty/mypy) should catch.

### Breaking Changes & Compatibility
- **Internal API only**: `WiringDependencies` is private to orchestration module
- **No external consumers**: Only orchestrator.py imports it
- **Build function signatures change**: But these are internal
- **Constructor change**: `WiringDependencies` signature changes (now requires 3 composed objects) - internal only

## Testing & Validation Strategy

### Unit Tests
- **New file**: `tests/unit/orchestration/test_orchestration_wiring.py`
- Test each new dataclass can be instantiated with valid values
- Test immutability (frozen=True) of new dataclasses
- Verify each build function correctly reads from the new dataclasses and instantiates the target component

### Integration Tests
- Run existing `tests/unit/orchestration/test_orchestrator.py` to ensure the orchestrator still initializes and runs correctly
- Existing orchestrator integration tests must pass unchanged

### Regression Tests
- `uv run pytest -m "unit or integration"` must pass
- Type checking with `uvx ty check` must pass

### Manual Verification
- None required - pure refactor with no user-visible changes

### Acceptance Criteria Coverage

| Spec AC | Covered By |
|---------|------------|
| Split WiringDependencies into 3 dataclasses | `src/orchestration/types.py` |
| `repo_path` belongs in `PipelineConfig` | `src/orchestration/types.py` |
| New dataclasses go in `types.py` | `src/orchestration/types.py` |
| No behavior changes | `tests/unit/orchestration/test_orchestrator.py` |
| 85% coverage | `tests/unit/orchestration/test_orchestration_wiring.py` |

## Open Questions

None - all decisions resolved in user context.

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` to create Beads issues with dependencies for multi-agent parallelization
- (default) to generate TODO.md checklist for simpler tracking
