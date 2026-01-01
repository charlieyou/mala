# Implementation Plan: Improve Terminal Logging

## Context & Goals
- **Spec**: docs/2026-01-01-improve-terminal-logging-spec.md
- Add `on_validation_started` event to surface validation start logs (only result exists today)
- Make `on_gate_started` and `on_review_started` log in normal mode (currently verbose-only)
- Include issue IDs in major event logs for better traceability during long-running operations

## Scope & Non-Goals
- **In Scope**
  - Add `on_validation_started` event to MalaEventSink protocol, NullEventSink, ConsoleEventSink
  - Change `on_gate_started` and `on_review_started` from `log_verbose()` to `log()`
  - Add `issue_id` parameter to `log()` function for dual ID display
  - Add `issue_id` parameter to major event sink methods
  - Add tests for new logging behavior
- **Out of Scope (Non-Goals)**
  - Changing the underlying event sink architecture
  - Adding JSON/machine-readable output format
  - Adding progress bars or spinners
  - Changing verbose vs quiet mode behavior fundamentally
  - Adding new CLI flags
  - Adding timestamps to tool invocations (`log_tool()`, `log_agent_text()` stay compact)

## Assumptions & Constraints
- Console output is the only affected surface; no external integrations depend on exact log formatting
- `AgentSessionRunner` is the primary emission point for gate/review/validation events
- Direct deployment without feature flags (internal logging changes)
- Rollback plan: Revert code changes to restore previous log format

### Implementation Constraints
- Extend existing modules; no new services or re-export shims
- Update protocol + both `NullEventSink` and `ConsoleEventSink` together
- Do not modify `log_tool()` or `log_agent_text()` output
- Issue ID display format: `[ISSUE-123]`; fallback to `[agent-a1b2]`
- Agent ID still used for color mapping even when issue_id displayed

### Testing Constraints
- Coverage threshold: 85% (enforced at quality gate)
- Use `capsys` for stdout capture in format verification tests
- Use `unittest.mock.patch` for datetime mocking in timestamp tests
- Extend existing test files (`tests/test_logging_console.py`, `tests/test_event_sink.py`)

## Prerequisites
- [ ] None - all dependencies are internal to the codebase

## High-Level Approach

1. **Phase 1**: Extend console `log()` function to accept `issue_id` and update prefix logic
2. **Phase 2**: Update event sink protocol and implementations with new/modified methods
3. **Phase 3**: Wire event emission with issue context at call sites
4. **Phase 4**: Validate behavior with focused unit tests

## Detailed Plan

### Task 1: Add `issue_id` support to console log prefix
- **Goal**: Display issue IDs in major event logs while keeping agent colors and fallback behavior
- **Covers**: AC #3 (issue ID shown), AC #4 (agent ID fallback), AC #5 (no logs removed)
- **Depends on**: None
- **Changes**:
  - `src/infra/io/log_output/console.py:116-142` — Exists
- **Verification**:
  - Add unit tests in `tests/test_logging_console.py`:
    - Assert `[ISSUE-123]` shown when issue_id provided
    - Assert agent_id NOT shown in output when issue_id provided
    - Assert fallback `[agent-a1b2]` appears when issue_id is None
    - Assert agent_id still used for color mapping
- **Rollback**: Revert changes in `src/infra/io/log_output/console.py`

**Implementation details:**
```python
def log(
    icon: str,
    message: str,
    color: str = Colors.RESET,
    dim: bool = False,
    agent_id: str | None = None,
    issue_id: str | None = None,  # NEW parameter
) -> None:
```

Update prefix logic (lines 134-138):
```python
if issue_id:
    # Use issue_id as display, agent_id for color mapping
    agent_color = get_agent_color(agent_id) if agent_id else Colors.CYAN
    prefix = f"{agent_color}[{issue_id}]{Colors.RESET} "
elif agent_id:
    agent_color = get_agent_color(agent_id)
    prefix = f"{agent_color}[{agent_id}]{Colors.RESET} "
else:
    prefix = ""
```

### Task 2: Extend MalaEventSink protocol and NullEventSink
- **Goal**: Add `on_validation_started` and propagate optional `issue_id` on major events
- **Covers**: AC #1 (validation started event), AC #3 (issue ID parameter)
- **Depends on**: None (can run in parallel with Task 1)
- **Changes**:
  - `src/infra/io/event_sink.py:56-599` — Exists (MalaEventSink protocol)
  - `src/infra/io/event_sink.py:601-835` — Exists (NullEventSink)
- **Verification**:
  - Update `tests/test_event_sink.py`:
    - Verify `on_validation_started` method exists
    - Verify NullEventSink implements it as no-op
    - Verify `issue_id=None` accepted without error
  - Run `uv run pytest -m unit tests/test_event_sink.py`
- **Rollback**: Revert protocol and NullEventSink changes in `src/infra/io/event_sink.py`

**Protocol changes** (add after `on_validation_result` at ~line 399):
```python
def on_validation_started(self, agent_id: str, issue_id: str | None = None) -> None:
    """Called when per-issue validation begins.

    Args:
        agent_id: Agent being validated.
        issue_id: Issue being validated (for display).
    """
    ...
```

**Methods to add `issue_id: str | None = None` parameter**:
- `on_validation_result` (~line 388)
- `on_gate_started` (~line 195)
- `on_gate_passed` (~line 210)
- `on_gate_failed` (~line 218)
- `on_gate_retry` (~line 233)
- `on_gate_result` (~line 248)
- `on_review_started` (~line 270)
- `on_review_passed` (~line 285)
- `on_review_retry` (~line 293)

**NullEventSink** (add after `on_validation_result` at ~line 752):
```python
def on_validation_started(self, agent_id: str, issue_id: str | None = None) -> None:
    pass
```

### Task 3: Update ConsoleEventSink with new/modified methods
- **Goal**: Implement console output for new events, pass issue_id to `log()`, switch start events to normal mode
- **Covers**: AC #1 (validation started logs), AC #2 (gate/review start visible), AC #3 (issue ID shown), AC #5 (no logs removed)
- **Depends on**: Tasks 1, 2
- **Changes**:
  - `src/infra/io/event_sink.py:837-1397` — Exists (ConsoleEventSink)
- **Verification**:
  - Add tests in `tests/test_logging_console.py`:
    - `on_validation_started` produces expected output with issue_id
    - Gate/review start logs appear without verbose mode
    - Use patched datetime for deterministic timestamp assertions
- **Rollback**: Revert ConsoleEventSink changes in `src/infra/io/event_sink.py`

**Implementation details:**

Add `on_validation_started` (after `on_validation_result` at ~line 1248):
```python
def on_validation_started(self, agent_id: str, issue_id: str | None = None) -> None:
    """Log validation start."""
    log("◐", "Starting validation...", Colors.MUTED, agent_id=agent_id, issue_id=issue_id)
```

Update `on_gate_started` (~line 1074-1087):
```python
def on_gate_started(
    self,
    agent_id: str | None,
    attempt: int,
    max_attempts: int,
    issue_id: str | None = None,  # NEW
) -> None:
    """Log quality gate check start."""
    scope = "run-level " if agent_id is None else ""
    log(  # Changed from log_verbose
        "→",
        f"Quality gate {scope}check (attempt {attempt}/{max_attempts})",
        Colors.MUTED,
        agent_id=agent_id,
        issue_id=issue_id,
    )
```

Update `on_review_started` (~line 1146-1158):
```python
def on_review_started(
    self,
    agent_id: str,
    attempt: int,
    max_attempts: int,
    issue_id: str | None = None,  # NEW
) -> None:
    """Log review start."""
    log(  # Changed from log_verbose
        "→",
        f"Review (attempt {attempt}/{max_attempts})",
        Colors.MUTED,
        agent_id=agent_id,
        issue_id=issue_id,
    )
```

Update remaining handlers with `issue_id` parameter and pass to `log()`:
- `on_gate_passed`, `on_gate_failed`, `on_gate_retry`, `on_gate_result`
- `on_review_passed`, `on_review_retry`
- `on_validation_result`

### Task 4: Emit events with issue_id at call sites
- **Goal**: Pass issue_id from pipeline to event sink methods, emit `on_validation_started`
- **Covers**: AC #1 (validation started emitted), AC #3 (issue ID propagated)
- **Depends on**: Tasks 2, 3
- **Changes**:
  - `src/pipeline/agent_session_runner.py:856-930` — Exists (gate/review event emissions)
- **Verification**:
  - Add unit test to verify `on_validation_started` is called before validation
  - Manual inspection of execution ordering during `mala run`
- **Rollback**: Revert emission changes in `src/pipeline/agent_session_runner.py`

**Implementation details:**

The `AgentSessionRunner` currently passes `input.issue_id` as the first argument (agent_id parameter) to event sink methods. This provides consistent color-per-issue mapping. We maintain this behavior and add the new `issue_id` kwarg for display.

**Clarification on validation vs gate events:** In mala, "validation" refers to the per-issue quality gate validation that runs after agent messages complete. The `on_validation_started` and `on_validation_result` events bracket the gate check. The spec's timeline (14:30:02-14:30:05 for validation, 14:30:05-14:30:08 for gate) is illustrative—validation encompasses the gate check. Emit `on_validation_started` just before the RUN_GATE effect handling, and `on_validation_result` is already emitted after the gate result is processed.

```python
# In AgentSessionRunner.run_session(), at the RUN_GATE handling (~line 874):
if result.effect == Effect.RUN_GATE:
    # Emit validation started BEFORE the gate check
    if self.event_sink is not None:
        self.event_sink.on_validation_started(
            input.issue_id,  # Keep using issue_id for color mapping (current behavior)
            issue_id=input.issue_id,  # NEW: also pass for display
        )

    if self.event_sink is not None:
        self.event_sink.on_gate_started(
            input.issue_id,  # Keep using issue_id for color mapping (current behavior)
            lifecycle_ctx.retry_state.gate_attempt,
            self.config.max_gate_retries,
            issue_id=input.issue_id,  # NEW: pass for display
        )
    # ... existing gate check code ...
```

Similarly update all other event emissions to add `issue_id=input.issue_id` as a new keyword argument (keep existing first argument unchanged).

### Task 5: Add comprehensive tests for new behavior
- **Goal**: Validate issue ID display, validation-start log, and normal-mode gate/review start logs
- **Covers**: AC #1, AC #2, AC #3, AC #4, AC #6 (all tests pass)
- **Depends on**: Tasks 1-4
- **Changes**:
  - `tests/test_logging_console.py` — Exists
  - `tests/test_event_sink.py` — Exists
- **Verification**: `uv run pytest -m unit`
- **Rollback**: Revert test changes

**Test cases for `tests/test_logging_console.py`:**
```python
def test_log_with_issue_id_shows_issue_only(capsys):
    """Test that issue_id is displayed (not agent_id) when both provided."""
    log("▶", "Agent started", Colors.BLUE, agent_id="agent-abc", issue_id="ISSUE-123")
    captured = capsys.readouterr()
    assert "[ISSUE-123]" in captured.out
    assert "agent-abc" not in captured.out

def test_log_with_issue_id_uses_agent_color(capsys, monkeypatch):
    """Test that agent_id is still used for color mapping."""
    monkeypatch.setattr(console, "_agent_color_map", {})
    monkeypatch.setattr(console, "_agent_color_index", 0)
    log("▶", "Test", agent_id="agent-1", issue_id="ISSUE-123")
    assert "agent-1" in console._agent_color_map

def test_log_with_issue_id_only_uses_cyan():
    """Test that issue_id without agent_id uses default cyan color."""
    # Verify cyan color code appears in output

def test_log_fallback_shows_agent_id_when_no_issue_id(capsys):
    """Test fallback: agent_id displayed when issue_id not provided.

    Note: In the existing codebase, event sinks are called with the issue_id
    (like 'ISSUE-123') as the agent_id parameter. This means the "fallback"
    will display the issue ID in the agent ID format [ISSUE-123]. This test
    verifies the fallback mechanism works correctly for standalone operations
    where a true agent ID (like 'agent-a1b2') is used without issue context.
    """
    log("▶", "Standalone op", Colors.BLUE, agent_id="agent-a1b2")
    captured = capsys.readouterr()
    assert "[agent-a1b2]" in captured.out
```

**Test cases for `tests/test_event_sink.py`:**
```python
def test_validation_started_logs_message(capsys):
    """Test that on_validation_started produces expected output."""
    sink = ConsoleEventSink()
    with patch('src.infra.io.log_output.console.datetime') as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "14:30:02"
        sink.on_validation_started("agent-abc", issue_id="ISSUE-123")
    captured = capsys.readouterr()
    assert "14:30:02" in captured.out
    assert "Starting validation" in captured.out
    assert "[ISSUE-123]" in captured.out

def test_null_sink_on_validation_started():
    """Test NullEventSink.on_validation_started is no-op."""
    sink = NullEventSink()
    assert sink.on_validation_started("agent-1", issue_id="ISSUE-1") is None

@patch("src.infra.io.event_sink.log")
def test_on_gate_started_uses_log_not_verbose(mock_log):
    """Test that on_gate_started uses log() not log_verbose()."""
    sink = ConsoleEventSink()
    sink.on_gate_started("agent-1", 1, 3, issue_id="ISSUE-1")
    mock_log.assert_called_once()

@patch("src.infra.io.event_sink.log")
def test_on_review_started_uses_log_not_verbose(mock_log):
    """Test that on_review_started uses log() not log_verbose()."""
    sink = ConsoleEventSink()
    sink.on_review_started("agent-1", 1, 3, issue_id="ISSUE-1")
    mock_log.assert_called_once()
```

### Task 6: Verify no existing logs removed
- **Goal**: Ensure no existing log messages are removed; only new/augmented output added
- **Covers**: AC #5 (no log messages removed), AC #6 (all tests pass)
- **Depends on**: Tasks 1-5
- **Changes**:
  - Review existing tests in `tests/test_event_sink.py`
  - Add regression assertions if needed
- **Verification**: `uv run pytest -m unit`
- **Rollback**: Revert test changes

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes
- **Standalone operations**: Display agent ID when issue_id is missing (graceful fallback per spec)
- **Fast operations**: Same-second timestamps are acceptable (spec edge case)
- **Concurrent agents**: Logs distinguished by `[ISSUE-ID]` prefix; interleaved but identifiable
- **Non-numeric issue IDs**: Display as provided; no validation needed

### Breaking Changes & Compatibility
- **Potential Breaking Changes**:
  - Console output format changes for gate/review start logs (now visible in normal mode)
  - Issue ID prefix instead of agent ID when issue context available
- **Mitigations**:
  - Limited to human-readable console output; no APIs changed
  - No machine-readable output affected
- **Rollout Strategy**:
  - Direct deployment; no feature flags needed for internal logging changes

## Testing & Validation
- **Unit Tests**
  - `tests/test_logging_console.py`: issue_id prefix, fallback behavior
  - `tests/test_event_sink.py`: protocol additions, ConsoleEventSink output, NullEventSink no-op
- **Integration / End-to-End Tests**
  - Not required by spec; rely on unit tests and manual output inspection
- **Regression Tests**
  - Verify existing log messages still present; no removals
  - Existing tests must continue to pass
- **Manual Verification**
  - Run `mala run` and compare output to spec's primary flow example
- **Monitoring / Observability**
  - Watch for missing or malformed prefixes in console output during use

### Acceptance Criteria Coverage
| Spec AC | Covered By |
|---------|------------|
| AC #1: validation started emits event and logs | Tasks 2, 3, 4, 5 |
| AC #2: gate/review start visible in normal mode | Tasks 3, 5 |
| AC #3: issue ID shown for major event logs | Tasks 1, 2, 3, 4, 5 |
| AC #4: fallback to agent ID without issue | Tasks 1, 5 |
| AC #5: no log messages removed | Tasks 3, 6 |
| AC #6: all tests pass | Tasks 5, 6 |

## Rollback Strategy (Plan-Level)
1. Revert code changes in:
   - `src/infra/io/log_output/console.py`
   - `src/infra/io/event_sink.py`
   - `src/pipeline/agent_session_runner.py`
2. Delete new test functions (keep existing tests)
3. Re-run `uv run pytest -m unit` to confirm previous behavior restored
4. No migration needed (console output only)

## Open Questions
None - all questions resolved in spec interview phase.

## File Existence Verification
| Path | Status |
|------|--------|
| src/infra/io/log_output/console.py | Exists |
| src/infra/io/event_sink.py | Exists |
| src/pipeline/agent_session_runner.py | Exists |
| src/pipeline/gate_runner.py | Exists |
| tests/test_logging_console.py | Exists |
| tests/test_event_sink.py | Exists |
