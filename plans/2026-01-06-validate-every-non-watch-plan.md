# Implementation Plan: --validate-every in Non-Watch Mode

## Context & Goals
- **Spec**: N/A — derived from beads issue `mala-mkow`
- **Issue**: `mala-mkow` — "refactor --validate-every command -- should be more general, including mode to run only at end"
- Make `--validate-every N` trigger periodic validation every N issues regardless of watch mode
- Split `WatchConfig` into separate `WatchConfig` and `ValidationConfig` for cleaner separation of concerns
- Decouple validation configuration from watch mode to support batch-mode periodic validation

## Scope & Non-Goals

### In Scope
- Make `--validate-every` an `Optional[int]` in CLI to detect explicit usage vs default
- Split `WatchConfig` into `WatchConfig` (enabled, poll_interval) and `ValidationConfig` (validate_every)
- Enable periodic validation every N issues when `--validate-every` is explicitly provided (regardless of `--watch`)
- Preserve existing watch mode default behavior (validate every 10 issues when `--watch` is set)
- Update CLI help text to remove "(watch mode only)" qualification
- Update documentation to reflect new behavior

### Out of Scope (Non-Goals)
- Changes to how validation itself works (gate checking, fixer logic)
- Changes to watch mode polling behavior (idle detection, poll intervals)
- New validation types or scopes
- Adding new CLI options like `--validate-mode`
- External config file migration (strictly CLI-only wiring)
- Renaming `WatchState` (cosmetic, can be done in follow-up)

## Assumptions & Constraints
- Breaking change is acceptable — `--validate-every` being ignored without `--watch` was unintentional/unused
- Current watch mode behavior must be preserved (default 10 when `--watch` is set)
- Validation semantics (periodic at N, 2N, 3N + final) are consistent across modes
- No external configs/serialized schemas (YAML/JSON/env) materialize `WatchConfig`; this is strictly CLI-only wiring

### Implementation Constraints
- Must preserve existing watch mode behavior (periodic + final validation with default 10)
- `WatchConfig.enabled` should still control watch-specific behavior (polling, idle detection)
- `ValidationConfig.validate_every` controls validation triggering; `None` means disabled
- Use dataclasses for config; maintain existing wiring patterns in `orchestration_wiring.py`
- Extend existing modules rather than adding new services

### Testing Constraints
- Must verify validation occurs in non-watch mode at N, 2N, 3N intervals when `--validate-every` is explicit
- Must verify watch mode still validates periodically with default 10
- Must verify plain `mala run` (no flags) does NOT trigger periodic validation
- Maintain existing watch mode test coverage

## Prerequisites
- [x] Context and requirements are clear (interview complete)
- [x] No special infrastructure needed
- [x] No external config migration required

## High-Level Approach

The implementation involves four main changes:

1. **CLI changes**: Make `--validate-every` an `Optional[int]` (default `None`) so the CLI can detect whether it was explicitly provided. Derive the effective value:
   - If `--validate-every N` is explicit → use N
   - If `--watch` is set but `--validate-every` is not → use 10 (preserve existing behavior)
   - Otherwise → `None` (no periodic validation)

2. **Config refactoring**: Split `WatchConfig` in `src/core/models.py` into two dataclasses:
   - `WatchConfig`: `enabled`, `poll_interval_seconds` (watch-specific settings)
   - `ValidationConfig`: `validate_every: int | None` (None means disabled)

3. **Coordinator changes**: Only pass `validation_callback` to `IssueExecutionCoordinator` when periodic validation is enabled. Remove `watch_enabled` guards from validation callback invocations since the callback presence itself indicates enablement.

4. **Documentation updates**: Update CLI help text and docs to reflect that `--validate-every` works in both modes.

## Technical Design

### Architecture

**Current flow**:
```
CLI (--validate-every N, default=10) → WatchConfig → IssueExecutionCoordinator
                                                          ↓
                                                if watch_enabled: call validation_callback
```

**New flow**:
```
CLI (--validate-every N, default=None) → derive effective value → ValidationConfig
                                                                       ↓
                                                          if validate_every is not None:
                                                              pass validation_callback
                                                                       ↓
                                                          IssueExecutionCoordinator
                                                              calls callback if present
```

The key insight is that:
1. CLI must distinguish "user specified `--validate-every`" from "using default"
2. Watch mode defaults to 10 if `--validate-every` is not explicit
3. Callback presence (not `watch_enabled`) gates validation triggering

### Data Model

**Current `WatchConfig`** (`src/core/models.py` lines 185-197):
```python
@dataclass
class WatchConfig:
    enabled: bool = False
    validate_every: int = 10
    poll_interval_seconds: float = 60.0
```

**New split**:
```python
@dataclass
class WatchConfig:
    """Configuration for watch mode polling behavior."""
    enabled: bool = False
    poll_interval_seconds: float = 60.0

@dataclass
class ValidationConfig:
    """Configuration for periodic validation triggering.

    Attributes:
        validate_every: Run validation after this many issues complete.
            None means periodic validation is disabled.
    """
    validate_every: int | None = None  # None means disabled
```

**`WatchState`** (`src/core/models.py` lines 200-219) remains unchanged — it tracks runtime validation state. Update docstring to reference `ValidationConfig.validate_every` instead of `WatchConfig.validate_every` for `next_validation_threshold` initialization.

**WatchState initialization**: Currently initialized in `IssueExecutionCoordinator.run_loop()` at line 219-223:
```python
watch_state = WatchState(
    next_validation_threshold=(
        watch_config.validate_every if watch_config else 10
    )
)
```
After config split, change to:
```python
watch_state = WatchState(
    next_validation_threshold=(
        validation_config.validate_every if validation_config and validation_config.validate_every is not None else 10
    )
)
```
Note: WatchState is always constructed (for tracking `completed_count`), but `next_validation_threshold` only matters when `validation_callback` is passed.

### Enablement Logic (Critical)

The CLI determines whether periodic validation is enabled:

```python
# In CLI (src/cli/cli.py)
validate_every: Annotated[
    int | None,
    typer.Option("--validate-every", min=1, help="..."),
] = None  # Changed from default=10 to default=None

# Derive effective validate_every value
effective_validate_every: int | None
if validate_every is not None:
    # User explicitly specified --validate-every N
    effective_validate_every = validate_every
elif watch:
    # Watch mode enabled, use default 10
    effective_validate_every = 10
else:
    # No watch, no explicit --validate-every → disabled
    effective_validate_every = None

# Only construct ValidationConfig and pass callback if enabled
# Use explicit `is not None` check (not truthiness) to prevent future bugs if 0 ever becomes valid
validation_config = ValidationConfig(validate_every=effective_validate_every) if effective_validate_every is not None else None
```

### Final Validation Behavior (Clarification)

There are TWO places where validation runs:

1. **Periodic validation in `IssueExecutionCoordinator`** (lines 548-585): Runs when `completed_count >= next_validation_threshold`. This is gated by `watch_config.enabled` today.

2. **Final validation in `MalaOrchestrator.run()`** (lines 988-996): Runs at end of run when `success_count > 0`. This already runs regardless of watch mode.

**Plan**:
- Remove `watch_enabled` guards from coordinator periodic validation (use callback presence instead)
- Keep orchestrator final validation as-is (already mode-agnostic)
- The `last_validation_at` tracking prevents double-runs when periodic threshold coincides with exit

**Double-run prevention**: The orchestrator's final validation (lines 988-996) does NOT currently check `last_validation_at` — it runs unconditionally when `success_count > 0`. This is acceptable because:
- The coordinator's periodic validation updates `watch_state.last_validation_at`, but this state is local to `run_loop()` and not shared with the orchestrator's final validation
- The orchestrator's final validation serves a different purpose: ensuring validation runs even when periodic thresholds weren't hit
- **Regression test required**: Add explicit test for exact-boundary case (`--validate-every 5 --max-issues 5`) to verify both validations run and this is acceptable behavior (or decide to skip final validation when `completed_count == last_validation_at`)

**SIGINT handling**: Verify during implementation that final validation runs on SIGINT. Current code path: `MalaOrchestrator.run()` catches `asyncio.CancelledError` from interrupt and still calls final validation at lines 988-996 before `_finalize_run()`. If SIGINT bypasses this path, add final validation to exception handler.

### API/Interface Design

**CLI** (`src/cli/cli.py`):
- Change `--validate-every` from `int` with default `10` to `int | None` with default `None`
- Add logic to derive effective value based on `--watch` flag
- Update help text to remove "(only in watch mode)"

**Internal APIs**:
- `IssueExecutionCoordinator.run_loop()`: Accept `validation_config: ValidationConfig | None`
- Only pass `validation_callback` when `validation_config is not None`
- Remove `watch_enabled` guards; use `if validation_callback:` instead
- `MalaOrchestrator.run()` and `_run_main_loop()`: Accept split configs

**Coordinator threshold block gating**: The entire threshold check and validation block (lines 548-585) currently checks `watch_config and watch_config.enabled`. After refactoring:
- Gate the entire block on `validation_config is not None` (or equivalently, callback presence)
- The threshold check (line 549-554) uses `validation_config.validate_every`
- The agent wait logic (lines 556-564) runs before callback invocation
- The callback invocation (lines 567-574) is an inner guard: `if validation_callback:`
- The threshold advancement (lines 576-585) references `validation_config.validate_every` (was `watch_config.validate_every`)

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/core/models.py` | Exists | Split `WatchConfig` into `WatchConfig` + `ValidationConfig`; update `WatchState` docstring |
| `src/cli/cli.py` | Exists | Change `--validate-every` to `Optional[int]`, add enablement logic, update help text |
| `src/pipeline/issue_execution_coordinator.py` | Exists | Remove `watch_enabled` guards (4 locations), use callback presence check |
| `src/orchestration/orchestrator.py` | Exists | Update to use split configs, conditionally pass validation_callback |
| `src/orchestration/orchestration_wiring.py` | Exists | Update wiring for new config structure |
| `src/orchestration/types.py` | Exists | No changes needed (validation config passed separately) |
| `src/pipeline/run_coordinator.py` | Exists | No changes needed (run_validation already mode-agnostic) |
| `src/pipeline/gate_runner.py` | Exists | No changes needed (gate checking unchanged) |
| `docs/cli-reference.md` | Exists | Remove "(watch mode only)" from `--validate-every` |
| `tests/unit/pipeline/test_watch_mode.py` | Exists | Update existing tests to reflect config split |
| `tests/unit/pipeline/test_periodic_validation.py` | **New** | Tests for non-watch periodic validation |
| `tests/unit/cli/test_cli.py` | Exists | Update CLI tests for new behavior |
| `tests/unit/orchestration/test_orchestrator.py` | Exists | Update orchestrator tests |
| `tests/unit/orchestration/test_orchestration_wiring.py` | Exists | Update wiring tests |
| `tests/integration/pipeline/test_watch_mode.py` | Exists | Add integration tests for non-watch validation |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes
- **`--validate-every` not specified, no `--watch`**: No periodic validation (callback not passed)
- **`--validate-every` not specified, with `--watch`**: Periodic validation every 10 (preserve existing)
- **`--validate-every N` without `--watch`**: Periodic validation every N issues
- **`--validate-every 1` without `--watch`**: Validates after every single issue (as expected)
- **`--max-issues N` with `--validate-every M` where M > N**: Final validation runs but no periodic (correct)
- **`--validate-every N` with `--max-issues M` where M = N*k**: No double-run (`last_validation_at` tracking)
- **SIGINT during non-watch run**: Final validation in orchestrator's `finally` block ensures it runs
- **Small issue count**: If `validate_every=10` and we run 5 issues, periodic won't hit, but orchestrator's final validation triggers

### Breaking Changes & Compatibility
- **Behavior change**: `--validate-every N` without `--watch` will now trigger periodic validation
- **CLI API change**: `--validate-every` default changes from 10 to None (but effective behavior for existing `--watch` usage is preserved)
- **Config API change**: Code instantiating `WatchConfig` directly will need updates
- **Risk level**: Low — this behavior was likely unintentional and unused externally
- **Mitigation**: Update all internal call sites; no external config files affected (CLI-only wiring)

## Testing & Validation Strategy

### Unit Tests
- `tests/unit/pipeline/test_periodic_validation.py` (**New**):
  - Test `IssueExecutionCoordinator` with `watch_enabled=False` and `validation_callback` present
  - Verify callback is invoked after N issues
  - Test threshold advancement (validates at N, 2N, 3N)
  - Test exact-boundary case: `validate_every=5` with exactly 5 completions — verify periodic validation runs at threshold
  - Test SIGINT path triggers final validation when issues completed

- Update `tests/unit/pipeline/test_watch_mode.py`:
  - Update existing tests to reflect config split
  - Verify watch mode still validates periodically with default 10

- Update `tests/unit/cli/test_cli.py`:
  - Test `--validate-every N` without `--watch` enables periodic validation
  - Test `--watch` without `--validate-every` uses default 10
  - Test plain `mala run` (no flags) does NOT enable periodic validation

### Integration Tests
- `tests/integration/pipeline/test_watch_mode.py`:
  - Run `mala run ... --validate-every 2` (no `--watch`) and verify validation output
  - Test with `--max-issues` combination

### Regression Tests
- Ensure `mala run ... --watch` (no explicit `--validate-every`) still validates every 10
- Ensure `mala run ... --watch --validate-every N` still validates every N
- Ensure watch mode polling/idle behavior unchanged
- **Exact-boundary regression test**: `--validate-every 5 --max-issues 5` — document expected validation count (1 or 2) and ensure consistent behavior

### Manual Verification
1. Run `mala run` (no flags) — verify NO periodic validation
2. Run `mala run --validate-every 3` — verify validation at 3, 6, ... and at end
3. Run `mala run --watch` — verify validation at 10, 20, ... (default)
4. Run `mala run --watch --validate-every 3` — verify same as (2)

### Acceptance Criteria Coverage

| Requirement | Covered By |
|-------------|----------|
| `--validate-every` works without `--watch` | CLI enablement logic + callback presence check |
| Periodic validation at N, 2N, 3N... | Keep existing threshold logic, remove `watch_enabled` guard |
| No periodic validation by default (without flags) | CLI default `None`, callback not passed |
| Watch mode default 10 preserved | CLI derivation logic when `--watch` set |
| Config split: WatchConfig + ValidationConfig | Create new `ValidationConfig` dataclass |
| Documentation updated | Update `docs/cli-reference.md` |

## Open Questions

None — all questions resolved during review.

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
