# Improve Terminal Logging

## Overview

Enhance the terminal logging system to provide better visibility into orchestrator operations by adding missing start events, making existing verbose-only start events visible in normal mode, and including issue ID context throughout agent execution. This helps users understand what's happening during long-running operations and correlate logs with specific issues.

## Goals

- Add `on_validation_started` event (currently only `on_validation_result` exists)
- Make `on_gate_started` and `on_review_started` log in normal mode (currently verbose-only)
- Include issue IDs alongside agent IDs in logs for better traceability
- Improve user's ability to understand what's happening during long-running operations

## Non-Goals (Out of Scope)

- Changing the underlying event sink architecture
- Adding JSON/machine-readable output format
- Adding progress bars or spinners
- Changing verbose vs quiet mode behavior fundamentally
- Adding new CLI flags
- Adding timestamps to tool invocations (the `log()` function already includes timestamps; `log_tool()` and `log_agent_text()` intentionally omit them for compact output)

## Ownership

- Product/feature owner: cyou
- Technical owner: cyou
- Key code areas:
  - `src/infra/io/log_output/console.py` - Core console logging functions
  - `src/infra/io/event_sink.py` - MalaEventSink protocol and ConsoleEventSink implementation
  - `src/orchestration/orchestrator.py` - Event emission points

## User Stories

- As a user running mala, I want to see when validation starts (not just when it ends) so I know what's happening
- As a user running mala, I want to see when gates and reviews start without needing verbose mode
- As a user running mala, I want to see which issue an agent is working on so I can correlate logs with issues

## Acceptance Criteria

- Given a validation operation, when it starts, then `on_validation_started` is emitted and a start log is displayed
- Given a gate or review operation, when it starts, then the start log is visible in normal mode (not just verbose)
- Given an agent working on an issue, when major event logs are emitted via `log()`, then the issue ID is shown as `[ISSUE-123]` prefix (agent_id used internally for color mapping but not displayed when issue_id is available)
- Given an agent without issue context (standalone operations), when logs are emitted, then the agent ID is shown as `[agent-a1b2]` (graceful fallback)
- Given existing log messages, when the feature is deployed, then no log messages are removed (only enhanced or added)
- Given the test suite, when tests are run, then all tests pass including new tests for logging behavior

## Technical Design

### Architecture

The existing event-driven architecture will be preserved:

1. `MalaEventSink` protocol defines semantic events
2. `ConsoleEventSink` implements console output for each event
3. Orchestrator emits events at appropriate points

No architectural changes are needed—this is an enhancement to existing patterns.

### Key Components

| Component | Location | Changes | Owner |
|-----------|----------|---------|-------|
| `log()` function | `src/infra/io/log_output/console.py:116-142` | Add optional `issue_id` parameter for dual ID display | infra/io |
| `MalaEventSink` protocol | `src/infra/io/event_sink.py:388` | Add `on_validation_started(agent_id: str)` method | infra/io |
| `ConsoleEventSink` | `src/infra/io/event_sink.py:837-1397` | Implement `on_validation_started`; change `on_gate_started` and `on_review_started` from `log_verbose()` to `log()` | infra/io |
| Orchestrator/Pipeline | `src/orchestration/` | Emit `on_validation_started` before validation; maintain agent_id→issue_id mapping for event context | orchestration |

### Specific Changes Required

#### 1. Add `on_validation_started` Event

**Protocol addition** (add after `on_validation_result` at ~line 399):
```python
def on_validation_started(self, agent_id: str) -> None:
    """Called when per-issue validation begins.

    Args:
        agent_id: Agent being validated.
    """
    ...
```

**NullEventSink implementation** (add after `on_validation_result` at ~line 752):
```python
def on_validation_started(self, agent_id: str) -> None:
    pass
```

**ConsoleEventSink implementation** (add after `on_validation_result` at ~line 1248):
```python
def on_validation_started(self, agent_id: str) -> None:
    """Log validation start."""
    log("◐", "Starting validation...", Colors.MUTED, agent_id=agent_id)
```

**Emission point**: In the pipeline runner, emit `on_validation_started` before calling the validation logic.

#### 2. Make Start Events Visible in Normal Mode

Change these handlers in `ConsoleEventSink`:

**`on_gate_started`** (line 1074-1087):
```python
# Before:
log_verbose("→", f"Quality gate {scope}check ...", Colors.MUTED, agent_id=agent_id)

# After:
log("→", f"Quality gate {scope}check ...", Colors.MUTED, agent_id=agent_id)
```

**`on_review_started`** (line 1146-1158):
```python
# Before:
log_verbose("→", f"Review (attempt {attempt}/{max_attempts})", Colors.MUTED, agent_id=agent_id)

# After:
log("→", f"Review (attempt {attempt}/{max_attempts})", Colors.MUTED, agent_id=agent_id)
```

#### 3. Issue ID Propagation

**Step 1**: Add `issue_id` parameter to `log()` function in `console.py`:
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

**Step 2**: Update the prefix logic in `log()`:
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

**Step 3**: The orchestrator already maintains the agent_id→issue_id mapping in `AgentSessionRunner`. For this initial implementation, pass `issue_id` only to the `log()` calls within `ConsoleEventSink` handlers for major events.

**Events to update in ConsoleEventSink** (add `issue_id` parameter and pass to `log()`):
- `on_agent_started` - already has both agent_id and issue_id
- `on_agent_completed` - already has both
- `on_validation_started` (new) - add issue_id parameter
- `on_validation_result` - add issue_id parameter
- `on_gate_started` - add issue_id parameter
- `on_gate_passed` - add issue_id parameter
- `on_gate_failed` - add issue_id parameter
- `on_gate_retry` - add issue_id parameter
- `on_gate_result` - add issue_id parameter
- `on_review_started` - add issue_id parameter
- `on_review_passed` - add issue_id parameter
- `on_review_retry` - add issue_id parameter

**Note on tool/text logging**: `log_tool()` and `log_agent_text()` are intentionally NOT updated with issue_id display. They remain compact without timestamps or ID prefixes, as specified in Non-Goals.

**MalaEventSink protocol updates**: Add optional `issue_id: str | None = None` parameter to the above methods. Both `NullEventSink` and `ConsoleEventSink` must be updated. These are the only two implementations in the codebase (tests subclass `NullEventSink` or create minimal fakes).

### Testing Strategy

Tests should:
1. Use `unittest.mock.patch` to mock `datetime.now()` for deterministic timestamp assertions
2. Capture stdout using `capsys` or `io.StringIO` and verify log format patterns
3. Extend existing tests in `tests/test_logging_console.py`

Example test patterns:
```python
from unittest.mock import patch

def test_validation_started_logs_message(capsys):
    """Test that on_validation_started produces expected output."""
    sink = ConsoleEventSink()
    with patch('src.infra.io.log_output.console.datetime') as mock_dt:
        mock_dt.now.return_value.strftime.return_value = "14:30:02"
        sink.on_validation_started("agent-abc")

    captured = capsys.readouterr()
    assert "14:30:02" in captured.out
    assert "Starting validation" in captured.out
    assert "[agent-abc]" in captured.out


def test_log_with_issue_id_shows_issue_only(capsys):
    """Test that issue_id is displayed (not agent_id) when both provided."""
    log("▶", "Agent started", Colors.BLUE, agent_id="agent-abc", issue_id="ISSUE-123")

    captured = capsys.readouterr()
    assert "[ISSUE-123]" in captured.out
    assert "agent-abc" not in captured.out  # agent_id used for color only
```

### Data Model

Not applicable — no new or changed data models.

### API Design

Not applicable — no new or changed external APIs. Internal protocol changes documented above.

### Backwards Compatibility

- **Existing behaviors affected**:
  - Gate and review start messages now visible without `-v` flag
  - Agent logs may include issue ID prefix when available
- **Impact on clients/integrations**: None — this is console output only, not machine-readable API
- **Rollout strategy**: Direct deployment (no feature flags needed for internal logging changes)
- **Rollback plan**: Revert code changes to restore previous log format

## User Experience

### Primary Flow

User runs `mala run` and sees:
```
14:30:01 ● mala orchestrator
14:30:01 ◐ repo: /path/to/repo
14:30:01 ◐ Ready issues: ISSUE-123, ISSUE-456
14:30:02 [ISSUE-123] ▶ Agent started
14:30:02 [ISSUE-123] ◐ Starting validation...
14:30:05 [ISSUE-123] ✓ Validation passed
14:30:05 [ISSUE-123] → Quality gate check (attempt 1/3)
14:30:08 [ISSUE-123] ✓ Gate passed
14:30:08 [ISSUE-123] → Review (attempt 1/3)
14:30:15 [ISSUE-123] ✓ Review passed
14:30:15 [ISSUE-123] ✓ Agent completed (13.0s): Issue resolved
```

Note: Tool invocations (`⚙ Read`, `⚙ Bash`, etc.) intentionally remain without timestamps to keep output compact.

### Error States

- **Missing issue_id**: Fall back to agent_id only (e.g., `[agent-a1b2]`)
- **Timestamp generation failure**: Should never fail (uses system time)

### Edge Cases

- **Standalone operations**: Agents running without issue context display agent_id only
- **Fast operations**: Operations starting and ending within the same second display identical timestamps (acceptable)
- **Concurrent agents**: Interleaved logs distinguished by their `[ISSUE-ID]` prefix
- **Non-numeric issue IDs**: Use whatever format the issue tracker provides (no special handling needed)

## Open Questions

None — all questions resolved during interview.

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Timestamp scope | Major events only (via `log()`) | `log()` already includes timestamps; `log_tool()` and `log_agent_text()` intentionally omit them for compact output |
| ID display format | `[ISSUE-123]` when issue_id available, `[agent-a1b2]` as fallback | Issue ID is the primary user-facing identifier; agent_id is internal and used only for color mapping |
| Backwards compatibility | Update formats directly | Simpler implementation; no users depend on exact log format |
| Start log visibility | Normal mode (not verbose-only) | Users need to see what's happening without requiring `-v` flag |
| Tool/text logging | No issue_id prefix | Keep compact; users can correlate via surrounding major event logs |
