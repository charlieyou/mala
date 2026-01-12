# Code Review System Refinement

**Tier:** M (medium complexity - touches multiple flows, requires coordination)
**Owner:** cyou
**Target ship:** 2026-01-12
**Links:** —

## 1. Outcome & Scope

**Problem / context**

The code review system has configuration and execution mechanisms in place, but visibility into review operations is limited:

1. **Trigger code reviews run silently** - No terminal output for when reviews start, what they're reviewing, or their results (only failures are logged)
2. **Inconsistent fixer behavior** - Trigger reviews spawn a dedicated fixer agent for findings above threshold; per-issue reviews ask the same agent to fix
3. **Logging only visible when reviews fail** - Users have no insight into successful reviews or review progress

Current state (verified via codebase research):
- **Configuration**: `per_issue_review` and `validation_triggers.*.code_review` both work via mala.yaml
- **mala init**: Already prompts for per-issue review settings
- **Trigger fixer loop**: `_run_code_review_remediation()` exists in run_coordinator.py (lines 1443-1549)
- **Per-issue review logging**: Events exist but per-issue review is disabled by default

**Goal**

Enable users to monitor code review execution with comprehensive terminal logging, and provide consistent fixer behavior for addressing findings above the configured threshold.

**Success criteria**

- When `code_review.enabled: true`, terminal shows start event followed by exactly one end event (passed/failed/skipped/error)
- When `code_review` block is absent or `enabled: false`, terminal stays silent (no events)
- When fixer loop runs for code review findings, terminal shows fixer attempt events
- All events appear in normal mode (not verbose-only) via ConsoleEventSink

**Non-goals**

- Changing the underlying review architecture (Cerberus vs agent_sdk)
- Adding new configuration options beyond what's needed for logging and fixer control
- Modifying the review finding format or priorities
- Building a new config system (must extend existing mala.yaml patterns)
- Changing per-issue review fixer behavior (keep same-agent model)
- Adding detailed per-file progress logging (keep it minimal: start/end only)
- Per-issue review logging verification (focus on trigger reviews only)

## 2. User Experience & Flows

**Primary flow** (Trigger code review with logging)

1. User runs `mala run` with `validation_triggers.run_end.code_review.enabled: true`
2. Terminal shows "→ CODE REVIEW run_end" (start event)
3. CumulativeReviewRunner checks for changes since baseline:
   - `since_run_start`: uses `run_metadata.run_start_commit` (captured at mala run start)
   - `since_last_review`: uses `run_metadata.last_cumulative_review_commits[trigger_key]` (persisted after each review); falls back to `since_run_start` if missing
4. If empty diff (no git changes between baseline commit and HEAD): emit skipped event, done
5. If changes exist: review runs (internal processing, no per-file logging)
6. If findings exceed threshold and `failure_mode: remediate`:
   - Terminal shows fixer attempt events (per attempt)
   - Fixer runs, review re-runs internally (no intermediate pass/fail events)
   - Process repeats up to max_retries
7. Terminal shows exactly one final result (end event):
   - "✓ CODE REVIEW run_end passed" OR
   - "✗ CODE REVIEW run_end failed: N blocking findings" OR
   - "✗ CODE REVIEW run_end error: {error message}"

**Key states**

**Configuration truth table** (review execution AND events gated identically):

| Config state | Review runs? | Events emitted? |
|--------------|--------------|-----------------|
| `code_review` block absent | No | No |
| `code_review: {}` (block present, `enabled` omitted) | No | No |
| `code_review.enabled: false` | No | No |
| `code_review.enabled: true` | Yes | Yes |

**Rule:** `enabled` must be explicitly `true` for review to run and events to emit. Any other case is disabled.

Operational states (events emitted when `code_review.enabled: true`):
- Skipped (empty diff): emit start event, then "→ CODE REVIEW {trigger_type} skipped: no changes" as end event
- Passed: emit start event "→ CODE REVIEW {trigger_type}", then "✓ CODE REVIEW {trigger_type} passed" as end event
- Failed (findings): emit start event, then "✗ CODE REVIEW {trigger_type} failed: N blocking findings" as end event
- Error (execution): emit start event, then "✗ CODE REVIEW {trigger_type} error: {error}" as end event

**Event invariant:** When `code_review.enabled: true`, always emit exactly one start event followed by exactly one end event (skipped/passed/failed/error). No events for disabled/unconfigured.

**Remediation loop exit decision table:**

| Last review result | Exception? | Final end event |
|-------------------|------------|-----------------|
| No blocking findings | No | `passed` |
| Blocking findings remain after max_retries | No | `failed` |
| Fixer fails all attempts | No | `failed` |
| Review throws exception | Yes | `error` |
| Fixer throws exception | Yes | `error` |

**Implementation pattern:** Use single end-event emission point after all retry logic completes. Set a result status variable during execution; emit end event in `finally` block based on status. Never emit end events inside the remediation loop.

**Blocking findings definition:** A finding is "blocking" if its priority is at or above the configured `finding_threshold` (e.g., with `finding_threshold: P1`, P0 and P1 findings are blocking; P2/P3 are informational). `blocking_count` is the count of such findings in the final review result after all remediation attempts.

## 3. Requirements + Verification

**R1 — Trigger code review start event**
- **Requirement:** The system MUST emit a start event to terminal when trigger code review begins
- **Verification:**
  - Given a trigger with `code_review.enabled: true`, When the review starts, Then terminal shows "→ CODE REVIEW {trigger_type}"

**R2 — Trigger code review end event**
- **Requirement:** The system MUST emit exactly one end event to terminal when trigger code review completes or is skipped, indicating passed, failed, or skipped status
- **Verification:**
  - **Skipped case:** Given an empty diff (no changes since baseline), When trigger-based review is invoked, Then terminal shows "→ CODE REVIEW {trigger_type} skipped: no changes" and no passed/failed message
  - **Passed case:** Given a successful review with no blocking findings, When review completes, Then terminal shows "✓ CODE REVIEW {trigger_type} passed" and not failed/skipped
  - **Failed case:** Given findings above threshold after fixer exhaustion, When review completes, Then terminal shows "✗ CODE REVIEW {trigger_type} failed: N blocking findings" and not passed/skipped

**R3 — Code review fixer visibility**
- **Requirement:** The system MUST emit fixer lifecycle events during the code review remediation loop using existing ConsoleEventSink handlers
- **Verification:**
  - Given `failure_mode: remediate` and findings above threshold, When fixer runs, Then terminal shows fixer "started" message followed by either "completed" or "failed" for each remediation attempt
  - **Example:** "→ FIXER Attempt 1/N: addressing M findings" ... then "✓ FIXER completed" or "✗ FIXER failed"
  - (Note: Fixer events `on_fixer_started/completed/failed` already exist in ConsoleEventSink; need to ensure they're emitted during code review remediation)

**R4 — Execution error handling**
- **Requirement:** The system MUST emit an error event when trigger code review execution fails unexpectedly (exception, timeout, reviewer crash)
- **Verification:**
  - Given a code review that throws an exception, When the exception is caught, Then terminal shows "✗ CODE REVIEW {trigger_type} error: {error message}" as the single end event
  - The error event MUST be mutually exclusive with passed/failed/skipped events (exactly one end event per invocation)
- **Error message format:** Terminal error string MUST be stable and user-friendly: `{ExceptionClass}: {short message}` (e.g., "TimeoutError: review timed out after 300s"). Full stack traces remain in debug/verbose logs only, not in normal terminal output.

## 4. Instrumentation & Release Checks

**Instrumentation**

New events needed in MalaEventSink protocol:
- `on_trigger_code_review_started(trigger_type: str)` - emitted before review runs
- `on_trigger_code_review_skipped(trigger_type: str, reason: str)` - emitted when skipped (empty diff, etc.)
- `on_trigger_code_review_passed(trigger_type: str)` - emitted when review passes
- `on_trigger_code_review_failed(trigger_type: str, blocking_count: int)` - emitted when findings exceed threshold
- `on_trigger_code_review_error(trigger_type: str, error: str)` - emitted on execution failure (exception, timeout)

**Event routing:** All new events MUST be implemented in `ConsoleEventSink` using `log()` (not `log_verbose()`) to ensure visibility in normal terminal output. Events should also propagate to any structured logging sinks that receive existing trigger events.

**Message format stability:** Event method signatures (names and parameters) are stable contracts. Console message formatting (symbols, exact text) may evolve but should follow existing patterns:
- Start: `"→ CODE REVIEW {trigger_type}"`
- Skipped: `"→ CODE REVIEW {trigger_type} skipped: {reason}"`
- Passed: `"✓ CODE REVIEW {trigger_type} passed"`
- Failed: `"✗ CODE REVIEW {trigger_type} failed: {N} blocking findings"`
- Error: `"✗ CODE REVIEW {trigger_type} error: {message}"`

Existing events to leverage:
- `on_fixer_started/completed/failed` - already in ConsoleEventSink, need to ensure code review remediation emits them

**Testing approach**

Tests should be added to `tests/unit/pipeline/test_run_coordinator.py` and `tests/integration/` covering:
- **Enabled trigger review:** Event sink receives exactly `started` + one of {`skipped`, `passed`, `failed`, `error`}
- **Disabled/unconfigured:** Event sink receives no code review events
- **Remediation loop:** Fixer events emitted per attempt; final end event reflects post-remediation state
- **Event exclusivity:** Assert no double-logging of end events (e.g., mock sink counts event calls)

Use `FakeEventSink` from `tests/fakes/event_sink.py` to capture and assert event sequences.

**Decisions made**

- Trigger code review fixer loop already exists and works (visibility issue only)
- Configuration via mala.yaml is complete
- mala init prompts are complete
- Per-issue fixer behavior stays as-is (same agent fixes own code)
- Logging is minimal: start/end events only, no per-file progress
- Empty diff shows skipped message (not silent)
- Scope limited to trigger code reviews (per-issue review logging is out of scope)
- Execution errors emit a separate `on_trigger_code_review_error` event (not overloaded on `_failed`)
- Events use `log()` not `log_verbose()` for normal-mode visibility
- During remediation loop, only fixer events are emitted per attempt; pass/fail/error is emitted once at the end

**Open questions**

None - all questions resolved.
