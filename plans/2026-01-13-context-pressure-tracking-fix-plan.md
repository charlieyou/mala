# Implementation Plan: Context Pressure Tracking Fix

## Context & Goals
- **Spec**: `plans/context-pressure-tracking-fix.md`
- Fix bug where SDK per-turn token deltas are overwritten instead of accumulated, causing context pressure to reflect only the last turn rather than cumulative session usage
- Exclude `cache_read_input_tokens` from pressure calculation (currently inflates pressure dramatically with large cache reads, e.g., 719% reported when actual usage is ~3%)
- **Impact**: Prevents premature context exhaustion triggers and ensures accurate monitoring of token limits during long-running agent sessions
- **Note on reset**: `AgentSessionRunner` already creates fresh `LifecycleContext` (with fresh `ContextUsage`) on each loop iteration after `ContextPressureError`, so explicit `reset()` wiring is not needed there

## Scope & Non-Goals

### In Scope
- Add `add_turn()` method to `ContextUsage` for cumulative token tracking
- Add `reset()` method to `ContextUsage` (for testing and potential future use)
- Add `is_tracking_enabled()` helper method
- Modify `pressure_ratio()` to exclude `cache_read_tokens` from calculation
- Update `MessageStreamProcessor` to use `add_turn()` instead of direct field assignment
- Update logging to show cumulative values and pressure ratio (cumulative-only approach)
- Add comprehensive unit tests for new behavior
- Add integration test for accumulation behavior

### Out of Scope (Non-Goals)
- Changes to telemetry/observability systems (`cache_read_tokens` still tracked for telemetry)
- Changes to SDK client or protocol interfaces
- Changes to context pressure thresholds or configuration values
- UI/CLI changes for displaying these metrics

## Assumptions & Constraints

### Assumptions
- SDK `usage.input_tokens` and `usage.output_tokens` are per-turn deltas, not cumulative totals
  - **Basis**: Spec section 1.1 explicitly states "The SDK's `usage` field on `ResultMessage` contains **per‑turn deltas**" based on debug log analysis
  - **Verification**: Debug logs show small per-turn values (e.g., input=487) that would be much larger if cumulative
- Pressure is defined as `(input_tokens + output_tokens) / context_limit`
  - **cache_read_tokens exclusion rationale**: Per spec section 1.2, including cache_read_tokens caused 719% pressure when actual context usage was ~3%. Cache reads represent tokens served from cache prefix, not new context consumption. The SDK reports these separately from input_tokens for this reason.

### Implementation Constraints
- `ContextUsage` remains a dataclass with public fields for compatibility
- `TRACKING_DISABLED` sentinel pattern (`-1`) must be preserved
- No backward-compatibility shims; update all usages directly (per CLAUDE.md)
- Extend existing infrastructure; no new domain entities or modules

### Testing Constraints
- Unit tests for `ContextUsage` methods (new file)
- Unit tests for `MessageStreamProcessor` accumulation behavior (extend existing)
- Integration test for accumulation behavior across multiple messages (optional, extend existing if needed)

## Integration Analysis

### Existing Mechanisms Considered

| Existing Mechanism | Could Serve Feature? | Decision | Rationale |
|--------------------|---------------------|----------|-----------|
| `ContextUsage` dataclass | Yes | Extend | Already handles all token tracking; add methods for accumulation and reset |
| `LifecycleContext.context_usage` | Yes | Use as-is | Already wired to lifecycle |
| `MessageStreamProcessor._process_result_message()` | Yes | Modify | Already handles usage extraction from SDK |
| `AgentSessionRunner.run_session()` | No change needed | Use as-is | Already creates fresh `LifecycleContext` on each loop iteration via `_initialize_session()` |

### Integration Approach
All changes extend existing infrastructure. No new classes, modules, or parallel systems needed. The `reset()` method is added to `ContextUsage` for testing and future use, but is not wired into `AgentSessionRunner` because the existing architecture already creates a fresh `LifecycleContext` (with fresh `ContextUsage`) after catching `ContextPressureError`.

## Prerequisites
- [x] No external prerequisites (all changes are internal to mala)
- [x] Spec definitions for pressure calculation complete (`plans/context-pressure-tracking-fix.md`)

## High-Level Approach

The fix requires changes at two layers:

1. **Domain Layer** (`ContextUsage`): Add `add_turn()` for accumulation, `reset()` for testing/future use, `is_tracking_enabled()` helper, and fix `pressure_ratio()` to exclude cache reads.

2. **Pipeline Layer** (`MessageStreamProcessor`): Replace direct field assignment with `add_turn()` call. Update logging to show cumulative values and pressure ratio. Pass cumulative totals to `ContextPressureError`.

**Note**: No changes needed in `AgentSessionRunner` — the existing `_initialize_session()` call at each loop iteration already creates a fresh `LifecycleContext` with zeroed `ContextUsage`.

## Technical Design

### Architecture

```
SDK ResultMessage
    |
    v
MessageStreamProcessor._process_result_message()
    |
    v (per-turn deltas)
ContextUsage.add_turn()  <-- NEW: accumulates instead of overwrites
    |
    v (cumulative totals)
ContextUsage.pressure_ratio()  <-- FIXED: excludes cache_read
    |
    v (if threshold exceeded)
ContextPressureError
    |
    v
AgentSessionRunner catches error
    |
    v
ContextPressureHandler.handle_pressure_error()
    |
    v
_initialize_session() creates fresh LifecycleContext  <-- EXISTING: already zeroes counters
    |
    v
New SDK stream starts with fresh counters (input=0, output=0, cache_read=0)
```

### Data Model

**ContextUsage changes:**
- `input_tokens`: Cumulative input tokens for current context (was: last turn only)
- `output_tokens`: Cumulative output tokens for current context (was: last turn only)
- `cache_read_tokens`: Cumulative cache read tokens for telemetry only (unchanged semantically)
- New method: `add_turn(input_tokens, output_tokens, cache_read_tokens)` — accumulates deltas
- New method: `reset()` — clears counters (for testing; not wired in production since fresh context is created)
- New method: `is_tracking_enabled()` — returns `not self.tracking_disabled`
- Modified: `pressure_ratio()` — uses only `input_tokens + output_tokens`, ignores `cache_read_tokens`

### API/Interface Design

```python
# ContextUsage additions (src/domain/lifecycle.py)

def add_turn(
    self,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> None:
    """Accumulate per-turn usage into the current context totals.

    No-op if tracking is disabled. Values are clamped to >= 0.
    """
    if self.tracking_disabled:
        return
    self.input_tokens += max(0, input_tokens)
    self.output_tokens += max(0, output_tokens)
    self.cache_read_tokens += max(0, cache_read_tokens)

def reset(self) -> None:
    """Reset cumulative usage for a new context.

    Sets all counters to 0. If tracking was disabled, it remains disabled.
    """
    if self.tracking_disabled:
        # Keep tracking disabled, just zero out counters
        self.output_tokens = 0
        self.cache_read_tokens = 0
        # input_tokens stays at TRACKING_DISABLED
    else:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0

def is_tracking_enabled(self) -> bool:
    """Return True if tracking is not disabled."""
    return not self.tracking_disabled

# pressure_ratio() modification
def pressure_ratio(self, limit: int) -> float:
    """Return ratio of total tokens used to the limit.

    Uses only input_tokens + output_tokens; cache_read_tokens is excluded.
    """
    if limit <= 0 or self.tracking_disabled:
        return 0.0
    live_tokens = max(0, self.input_tokens) + max(0, self.output_tokens)
    return live_tokens / limit
```

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/domain/lifecycle.py` | Exists | Add `add_turn()`, `reset()`, `is_tracking_enabled()` to `ContextUsage`; fix `pressure_ratio()` |
| `src/pipeline/message_stream_processor.py` | Exists | Replace direct assignment with `add_turn()`; update logging; pass cumulative totals to error |
| `tests/unit/domain/test_context_usage.py` | **New** | Unit tests for `ContextUsage` methods |
| `tests/unit/pipeline/test_message_stream_processor.py` | Exists | Add tests for accumulation behavior |

**Note**: `src/pipeline/agent_session_runner.py` does not need modification — existing `_initialize_session()` already creates fresh `ContextUsage`.

## Risks, Edge Cases & Breaking Changes

### Risks
- **Behavioral change**: `ContextUsage.input_tokens/output_tokens` now represent cumulative totals, not last turn. Any consumer expecting last-turn values would see incorrect data.
  - Mitigation: Code search shows only `MessageStreamProcessor` and telemetry use these fields; both expect cumulative semantics.
- **Double-counting risk**: If any provider returns cumulative totals instead of per-request, this change would cause double-counting.
  - Mitigation: Verify standard provider (Anthropic) behavior returns per-request usage. No known providers return cumulative.

### Edge Cases & Failure Modes
- **`add_turn()` with negative values**: Clamp to 0 per spec requirement
- **`reset()` when tracking disabled**: Keep tracking disabled, set `output_tokens` and `cache_read_tokens` to 0 (user decision)
- **`pressure_ratio()` with `context_limit <= 0`**: Return 0.0 (defensive, already implemented)
- **Very large token totals**: `pressure_ratio` can exceed 1.0 (expected; signals over-limit)
- **Missing `usage` field in ResultMessage**: Existing behavior preserved — `disable_tracking()` is called

### Breaking Changes & Compatibility
- **Potential Breaking Changes**:
  - `pressure_ratio()` no longer includes `cache_read_tokens` — will return lower values for sessions with cache reads
  - `input_tokens`/`output_tokens` now cumulative instead of last-turn
- **Mitigations**:
  - Both changes are bug fixes to match intended behavior per spec
  - No API signature changes
  - `cache_read_tokens` still tracked for telemetry

## Testing & Validation Strategy

### Unit Tests

**`ContextUsage` tests (New: `tests/unit/domain/test_context_usage.py`):**
1. `test_add_turn_accumulates_tokens` — Multiple `add_turn()` calls accumulate correctly
2. `test_pressure_ratio_excludes_cache_read` — Large `cache_read_tokens` doesn't affect pressure
3. `test_add_turn_noop_when_disabled` — `add_turn()` is no-op when tracking disabled
4. `test_disabled_tracking_returns_zero_pressure` — `pressure_ratio()` returns 0.0 when disabled
5. `test_reset_clears_counters` — `reset()` sets all counters to 0
6. `test_reset_preserves_disabled_state` — `reset()` keeps tracking disabled if it was disabled
7. `test_is_tracking_enabled` — Helper returns correct boolean
8. `test_add_turn_clamps_negative_values` — Negative values clamped to 0
9. `test_pressure_ratio_with_zero_limit` — Returns 0.0 when limit <= 0
10. `test_pressure_ratio_can_exceed_one` — Values > 1.0 allowed when over limit

**`MessageStreamProcessor` tests (add to `tests/unit/pipeline/test_message_stream_processor.py`):**
1. `test_accumulates_usage_across_multiple_messages` — Two ResultMessages accumulate tokens
2. `test_pressure_calculation_ignores_cache_read` — Large cache_read doesn't trigger error
3. `test_context_pressure_error_contains_cumulative_totals` — Error has cumulative values
4. `test_missing_usage_disables_tracking` — Existing behavior preserved

### Integration Tests

**Accumulation behavior test (add to existing integration tests if needed):**
1. Configure `context_limit` and threshold to trigger `ContextPressureError` after multiple turns
2. Send multiple ResultMessages with token usage
3. Verify tokens accumulate correctly across messages
4. Verify `ContextPressureError` is raised with cumulative totals when threshold exceeded
5. Verify large `cache_read_tokens` values do not affect pressure calculation

**Note**: Context restart flow does not need integration testing — `AgentSessionRunner` already creates fresh `LifecycleContext` via existing `_initialize_session()` call.

### Manual Verification
- Run agent session with debug logging enabled
- Observe cumulative token values in logs increase across turns
- Verify large cache_read values don't inflate pressure
- Trigger context restart and verify pressure resets

### Acceptance Criteria Coverage

| Spec AC | Covered By |
|---------|------------|
| Track cumulative input/output tokens | `add_turn()` method; Unit tests 1-2, MSP test 1 |
| Exclude cache_read from pressure | `pressure_ratio()` fix; Unit test 2, MSP test 2 |
| Reset on context restart | Existing `_initialize_session()` creates fresh `ContextUsage`; `reset()` method for testing |
| Preserve TRACKING_DISABLED semantics | `add_turn()` no-op check; Unit tests 3-4, 6 |

## Open Questions
*All questions resolved during interview:*

1. **Logging approach**: Cumulative-only logging — show cumulative values and pressure ratio (not per-turn + cumulative)
2. **reset() behavior with disabled tracking**: Keep tracking disabled (set totals to 0 but don't re-enable)

## Next Steps
After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
