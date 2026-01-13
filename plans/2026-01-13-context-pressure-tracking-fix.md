# Fixing Context Pressure Tracking in `mala`

## 1. Problem Statement

### 1.1 Current behavior

- The SDK's `usage` field on `ResultMessage` contains **per‑turn deltas**:
  - `input_tokens`
  - `output_tokens`
  - `cache_read_input_tokens`
- In `MessageStreamProcessor._process_result_message` (see `src/pipeline/message_stream_processor.py:331`), we currently:
  - Read the usage values from the message.
  - **Overwrite** `lifecycle_ctx.context_usage.input_tokens`, `output_tokens`, and `cache_read_tokens` with these per‑turn values.
  - Call `lifecycle_ctx.context_usage.pressure_ratio(context_limit)` to decide whether to raise `ContextPressureError`.

### 1.2 Bug

- Because the SDK's usage is per‑turn, overwriting `ContextUsage` means:
  - We are effectively using **only the last turn's tokens** instead of the total across the current context/session.
  - Context pressure is underestimated over long sessions unless the last turn is large.
- Additionally, `cache_read_input_tokens` (tokens read from cache) is:
  - Currently included in pressure calculations (via `ContextUsage.pressure_ratio` / how we pass numbers into it).
  - This **inflates pressure** dramatically when large cache reads occur.

Example from debug logs:

- Reported by SDK for a turn:  
  - `input_tokens = 487`  
  - `output_tokens = 5880`  
  - `cache_read_input_tokens = 1,432,262`
- With a `context_limit` of 200,000:
  - Current logic effectively uses `(487 + 5880 + 1,432,262) / 200,000 ≈ 7.19` → 719% "pressure"
  - Actual context usage relevant to exhaustion is ≈ 6.3K tokens, i.e., only ~3–4% of limit.
- Result: spurious `ContextPressureError` and unnecessary context restarts.

### 1.3 Desired behavior

- Track **cumulative** input/output tokens across the current context/session using per‑turn deltas.
- **Exclude `cache_read_input_tokens` from pressure calculation**. It should be tracked only for telemetry/observability.
- Provide a way to **reset** context usage when we explicitly restart the context (new SDK session / continuation).
- Maintain a sentinel (`TRACKING_DISABLED`) behavior for cases where the SDK does not return usage (no pressure detection).

---

## 2. Proposed Solution & `ContextUsage` API Changes

All changes in this section apply to `src/domain/lifecycle.py` around `ContextUsage` and `TRACKING_DISABLED`.

### 2.1 `ContextUsage` responsibilities

`ContextUsage` should be the canonical place to:

- **Accumulate** token usage across turns within the *current context*.
- Compute pressure against a context limit by looking **only at input + output**, never cache reads.
- Represent "usage tracking disabled" (e.g., SDK returned no usage) via `TRACKING_DISABLED`.
- Allow **resetting** the counters when a context restart occurs.

### 2.2 `ContextUsage` fields (unchanged / clarified)

Keep the existing dataclass (names may already exist; this clarifies intended fields):

- `input_tokens: int`  
  Cumulative input tokens for current context (sum of per‑turn `input_tokens`).
- `output_tokens: int`  
  Cumulative output tokens for current context.
- `cache_read_tokens: int`  
  Cumulative cache read tokens for telemetry only.
- `tracking_state: int` (or reuse a field / pattern we already have)  
  - Holds `TRACKING_DISABLED` when usage tracking is disabled.
  - Otherwise, indicates tracking is enabled.

If current implementation uses a simpler pattern (e.g., `input_tokens == TRACKING_DISABLED`), the spec should be adapted to reuse that sentinel, but the semantics below must hold.

### 2.3 New `add_turn()` API

Add a method to `ContextUsage` that accumulates per‑turn deltas:

```python
def add_turn(
    self,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> None:
    """Accumulate per-turn usage into the current context totals.

    - No-op if tracking is disabled (TRACKING_DISABLED).
    - Each argument is treated as a delta for this turn.
    - All values are clamped at >= 0 to avoid negative totals.
    """
```

Behavior:

- If tracking is disabled (`self.is_tracking_enabled()` is false), `add_turn` is a **no-op**.
- Otherwise:
  - `self.input_tokens += max(0, input_tokens)`
  - `self.output_tokens += max(0, output_tokens)`
  - `self.cache_read_tokens += max(0, cache_read_tokens)`
- This changes semantics from "store last turn" to "accumulate this turn into totals".

### 2.4 Reset API

Add a method to clear counters for a new context:

```python
def reset(self) -> None:
    """Reset cumulative usage for a new context.

    - Sets input/output/cache_read totals to 0.
    - Keeps tracking state enabled (unless previously disabled).
    """
```

Behavior:

- If tracking is enabled:
  - Set `input_tokens = 0`
  - Set `output_tokens = 0`
  - Set `cache_read_tokens = 0`
- If tracking is disabled (e.g., we previously called `disable_tracking()`), `reset()` can:
  - Either be a no-op, or set totals to 0 but keep tracking disabled.
  - Recommended: totals set to 0, leave tracking disabled until explicit usage data arrives again; this matches existing semantics where pressure is effectively 0 when disabled.

### 2.5 `disable_tracking()` semantics (existing, clarified)

Ensure existing `disable_tracking()`:

```python
def disable_tracking(self) -> None:
    """Disable context pressure tracking for this session.

    - Sets tracking sentinel (TRACKING_DISABLED).
    - Totals may be kept for telemetry but pressure_ratio must return 0.
    """
```

Behavior:

- After calling `disable_tracking()`:
  - `pressure_ratio(limit)` must always return `0.0`.
  - `add_turn()` must be a no-op.
  - Any consumer interpreting `ContextUsage` for pressure should treat it as "pressure tracking off".

### 2.6 `pressure_ratio()` semantics (updated)

`ContextUsage.pressure_ratio()` should:

```python
def pressure_ratio(self, context_limit: int) -> float:
    """Return usage fraction (0.0–∞) relative to limit.

    - Uses cumulative input + output only.
    - Ignores cache_read_tokens entirely.
    - Returns 0.0 if tracking disabled or limit <= 0.
    """
```

Behavior:

- If tracking disabled → `0.0`
- If `context_limit <= 0` → `0.0` (defensive)
- Otherwise:

```python
live_tokens = max(0, self.input_tokens) + max(0, self.output_tokens)
return live_tokens / context_limit
```

- `cache_read_tokens` is **not** included and plays no role in pressure calculation.
- Consumers (e.g., `MessageStreamProcessor`) continue to interpret `pressure_ratio` the same way; they just get a corrected value.

### 2.7 Helper / convenience methods (optional but recommended)

- `is_tracking_enabled() -> bool`  
  Returns `True` if tracking is not disabled, `False` otherwise. This can simplify checks in callers and tests.

---

## 3. Changes in `message_stream_processor.py`

All changes in this section are local to `src/pipeline/message_stream_processor.py`.

### 3.1 Use `ContextUsage.add_turn()` instead of overwriting

In `_process_result_message` (currently lines ~343–381):

Current behavior (simplified):

```python
input_tokens = ...
output_tokens = ...
cache_read = ...
lifecycle_ctx.context_usage.input_tokens = input_tokens
lifecycle_ctx.context_usage.output_tokens = output_tokens
lifecycle_ctx.context_usage.cache_read_tokens = cache_read

pressure = lifecycle_ctx.context_usage.pressure_ratio(self.config.context_limit)
...
raise ContextPressureError(..., input_tokens=input_tokens, output_tokens=output_tokens, cache_read_tokens=cache_read, pressure_ratio=pressure)
```

Proposed behavior:

1. Extract per‑turn deltas exactly as today:
   - Preserve dict vs object handling.
   - Preserve `or 0` semantics.

2. Replace direct assignment with `add_turn()`:

```python
context_usage = lifecycle_ctx.context_usage
context_usage.add_turn(
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    cache_read_tokens=cache_read,
)
```

3. Compute pressure using updated cumulative totals:

```python
pressure = context_usage.pressure_ratio(self.config.context_limit)
```

4. When constructing `ContextPressureError`, pass **cumulative** totals (not just last‑turn values):

```python
raise ContextPressureError(
    session_id=state.session_id or "",
    input_tokens=context_usage.input_tokens,
    output_tokens=context_usage.output_tokens,
    cache_read_tokens=context_usage.cache_read_tokens,
    pressure_ratio=pressure,
)
```

5. Logging: log **both** per‑turn and cumulative values for debuggability, or at least the cumulative totals:

Example updated log:

```python
logger.debug(
    "Context usage: turn=(input=%d output=%d cache_read=%d) "
    "cumulative=(input=%d output=%d cache_read=%d) "
    "limit=%d pressure=%.1f%%",
    input_tokens, output_tokens, cache_read,
    context_usage.input_tokens,
    context_usage.output_tokens,
    context_usage.cache_read_tokens,
    self.config.context_limit,
    pressure * 100,
)
```

This makes it easy to confirm accumulation is behaving as expected.

### 3.2 Handling missing `usage` (unchanged)

Keep the existing branch when `usage` is `None`:

- Log a warning that `ResultMessage` has no `usage`.
- Call `lifecycle_ctx.context_usage.disable_tracking()` to turn off pressure tracking.
- No change to that behavior is required; it still makes sense with the new API.

### 3.3 No use of `cache_read_tokens` in pressure

Ensure that:

- `pressure_ratio()` no longer depends on `cache_read_tokens`.
- `_process_result_message` does not manually fold `cache_read_tokens` into any pressure calculation.
- `cache_read_tokens` is only used:
  - For logging.
  - For telemetry (e.g., metrics emitted elsewhere, not in this file).

---

## 4. Reset Behavior on Context Restart

Context restarts are handled via `ContextPressureHandler` in `src/pipeline/context_pressure_handler.py` and higher‑level orchestration (e.g., AgentSessionRunner / lifecycle driver).

### 4.1 When to reset

Reset should happen whenever a **new SDK context** is started, i.e.:

- After we handle a `ContextPressureError` and:
  - Fetch a checkpoint (if any) via `ContextPressureHandler.fetch_checkpoint`.
  - Build a continuation prompt via `ContextPressureHandler.build_continuation_prompt`.
  - Start a **new session** / **fresh SDK context** for the continuation.

At that point, we want `ContextUsage` to measure **pressure within the new context only**, not accumulate across the prior session.

### 4.2 Where to reset

Implementation‑wise, this should be done in the orchestrator that:

- Catches `ContextPressureError`.
- Calls `ContextPressureHandler.handle_pressure_error(...)`.
- Creates the new SDK query/stream for continuation.

Spec:

- Immediately before starting the new SDK stream for the continuation, call:

```python
lifecycle_ctx.context_usage.reset()
```

This ensures:

- `input_tokens`, `output_tokens`, and `cache_read_tokens` start at 0 for the new context.
- `pressure_ratio()` reflects the new context's usage, not historical totals.
- Sentinel state:
  - If tracking was disabled (no usage in previous context), `reset()` should not force-enable tracking; tracking remains off until the new SDK session provides usage.

### 4.3 Logging on reset

Optional but recommended:

- Log at INFO or DEBUG level when resetting:

```python
logger.debug(
    "Session %s: resetting context usage for restart (continuation #%d)",
    issue_id,
    continuation_count + 1,
)
```

This makes it easier to trace usage across multiple restarts.

---

## 5. Test Cases to Add

### 5.1 `ContextUsage` unit tests (`src/domain/lifecycle.py`)

Create or extend `ContextUsage` tests in the lifecycle test module (e.g., `tests/domain/test_lifecycle_context_usage.py`, or colocated with existing lifecycle tests).

New tests:

1. **Accumulation of per‑turn usage**
   - Given a fresh `ContextUsage` with tracking enabled:
     - Call `add_turn(input_tokens=100, output_tokens=200, cache_read_tokens=1000)`.
     - Call `add_turn(input_tokens=50, output_tokens=75, cache_read_tokens=500)`.
   - Assert:
     - `input_tokens == 150`
     - `output_tokens == 275`
     - `cache_read_tokens == 1500`

2. **`pressure_ratio` ignores cache read**
   - Start with `input_tokens = 1000`, `output_tokens = 2000`, `cache_read_tokens = 1_000_000`.
   - For `context_limit = 10_000`:
     - `pressure_ratio(context_limit)` should return `(1000 + 2000) / 10_000 == 0.3`.
   - Confirm that changing `cache_read_tokens` does **not** change `pressure_ratio`.

3. **Disabled tracking returns 0 pressure**
   - `disable_tracking()`, then:
     - `pressure_ratio(10_000) == 0.0`
   - Call `add_turn(...)` while disabled:
     - Token totals should not change.

4. **Reset behavior**
   - Accumulate some usage via `add_turn`.
   - Call `reset()`.
   - Assert:
     - `input_tokens == 0`
     - `output_tokens == 0`
     - `cache_read_tokens == 0`
   - If tracking was disabled before `reset`, assert that:
     - `pressure_ratio()` is still `0.0` (tracking remains disabled).

5. **Context limit edge cases**
   - `context_limit <= 0` → `pressure_ratio` is `0.0` independent of tokens.
   - Very large totals → `pressure_ratio` can exceed 1.0, and this is allowed (we use it as a signal that we overshot the limit).

### 5.2 `MessageStreamProcessor` unit tests (`src/pipeline/message_stream_processor.py`)

Create or update tests for `_process_result_message`; use fake message objects and fake `LifecycleContext`.

1. **Accumulation across multiple `ResultMessage`s**
   - Prepare a fake `LifecycleContext` with a real `ContextUsage` instance.
   - Call `_process_result_message` twice with messages:
     - Message 1: `usage = {input_tokens: 500, output_tokens: 1000, cache_read_input_tokens: 10000}`
     - Message 2: `usage = {input_tokens: 600, output_tokens: 1200, cache_read_input_tokens: 20000}`
   - With `context_limit = 10_000`, after both calls:
     - `context_usage.input_tokens == 1100`
     - `context_usage.output_tokens == 2200`
     - `context_usage.cache_read_tokens == 30000`
   - Confirm that no `ContextPressureError` is raised if threshold > `(1100 + 2200) / 10_000`.

2. **Correct pressure calculation independent of cache read**
   - Use a message with small input/output but huge cache read:
     - `input_tokens = 487`, `output_tokens = 5880`, `cache_read_input_tokens = 1_432_262`.
   - For `context_limit = 200_000` and threshold e.g. `0.7`:
     - Ensure no `ContextPressureError` is raised:
       - `pressure_ratio == (487 + 5880) / 200_000 ≈ 0.0318`.
   - Verify `ContextPressureError` is not triggered solely due to large cache_read.

3. **`ContextPressureError` uses cumulative totals**
   - Configure small `context_limit` and threshold (e.g. 0.5).
   - Send:
     - Turn 1: `input_tokens = 200`, `output_tokens = 200` (pressure 0.4).
     - Turn 2: `input_tokens = 100`, `output_tokens = 100` (cumulative input=300, output=300 → 600/ context_limit=1000 = 0.6).
   - On second call, expect `ContextPressureError`:
     - Assert that `error.input_tokens == 300`
     - Assert that `error.output_tokens == 300`
     - Assert that `error.cache_read_tokens` equals sum of cache_read deltas.

4. **Missing usage disables tracking**
   - Send a `ResultMessage` with `usage=None`.
   - Assert:
     - `lifecycle_ctx.context_usage` is disabled (per sentinel / helper).
     - Subsequent `_process_result_message` calls with usage data:
       - Either re-enable tracking or remain disabled per current policy (spec: keep existing behavior; tests should match whatever is currently intended).

### 5.3 Integration tests (context restart)

If there is an integration test harness for context restart (e.g., in `tests/pipeline/test_context_pressure_handler.py` or session runner tests):

1. Trigger `ContextPressureError` by:
   - Using a very small `context_limit`.
   - Sending multiple messages to accumulate beyond threshold.

2. Verify that:
   - `ContextPressureHandler.handle_pressure_error` is called.
   - After restart:
     - `lifecycle_ctx.context_usage` has been `reset()`.
     - New usage accumulation starts from 0 in the new SDK session.
     - The next pressure computation is based only on post‑restart tokens.

---

## 6. Migration Notes & Backward Compatibility

### 6.1 Public API surface

- `ContextUsage` remains a dataclass with public fields:
  - `input_tokens`
  - `output_tokens`
  - `cache_read_tokens`
- New methods (`add_turn`, `reset`, `is_tracking_enabled`) are additive.
- `pressure_ratio()` signature is unchanged; only semantics are corrected (now uses cumulative input/output and ignores cache_read).

### 6.2 Behavioral changes

- **Context usage semantics**:
  - Previously: `input_tokens` and `output_tokens` in `ContextUsage` reflected **last turn only**.
  - Now: they reflect **cumulative totals for the current context**.
- Any consumer that was interpreting `ContextUsage.input_tokens` as "last turn tokens" will see a change; however:
  - Current usage in `mala` is focused on context pressure detection and telemetry, where cumulative is the correct interpretation.
  - There are no known consumers that depend on the "last turn only" behavior; code search should confirm this before rollout.

### 6.3 Cache read semantics

- `cache_read_tokens` continues to be tracked and exposed for telemetry:
  - Logging, metrics, and debugging can still use it.
- The only change is that `cache_read_tokens` is **no longer part of pressure calculation.**
  - This is a breaking fix for the bug where huge cache reads inflated pressure:
    - Expected, desired, and safer.

### 6.4 Rollout and verification

- Deploy change behind existing configuration (`context_limit`, `context_restart_threshold`) with no new config flags.
- Operational verification:
  - Observe logs for:
    - Cumulative usage vs. per‑turn usage.
    - Context restarts and their reported `pressure_ratio`.
  - Confirm that:
    - Pressure values are in a realistic range (0–100% for typical workloads).
    - No more ">100%" values caused solely by cache reads.
- If any external consumers had been relying on last‑turn semantics, adapt them to:
  - Either compute their own deltas.
  - Or reference per‑turn values explicitly from the SDK messages rather than `ContextUsage`.

---

This spec provides the required API changes to `ContextUsage`, wiring in `MessageStreamProcessor`, reset behavior on restart, and the tests/migration steps necessary to fix context pressure tracking in `mala` while preserving backward compatibility at the API level.
