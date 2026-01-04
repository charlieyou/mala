# Mala Logging Improvement Plan

## Context & Goals

During a real deadlock between agents mala-skxg.2 and mala-skxg.13, the deadlock detection system failed to trigger. Diagnosis was impossible because there was **insufficient logging** across the entire codebase.

### Specific Gaps Identified

1. Whether PreToolUse hooks were being called
2. Whether lock events were being emitted to DeadlockMonitor
3. Whether WaitForGraph state was being updated correctly
4. Whether cycle detection was running but not finding cycles
5. Agent lifecycle state transitions
6. Session execution flow
7. Gate/review coordination

### Goals

1. Add sufficient logging to diagnose deadlock detection failures
2. Provide visibility into configuration, event flow, graph state, and resolution
3. Follow structured logging patterns for correlation across agents
4. Maintain performance by keeping high-frequency events at DEBUG level

---

## Scope & Non-Goals

### In Scope

- Adding ~125 LOC of structured logging across 13 files
- Implementing `AgentLoggerAdapter` for agent_id propagation
- Documenting expected debugging workflow

### Non-Goals

- **No behavior changes to deadlock detection logic** - this is logging-only
- No changes to lock semantics or graph algorithms
- No new dependencies or external logging infrastructure
- No metrics or alerting (future work)
- No changes to existing log levels of current statements

---

## Assumptions & Constraints

### Assumptions

1. Python's built-in `logging` module is sufficient (no need for structlog)
2. DEBUG-level logs are acceptable for high-frequency events
3. Existing log configuration (handlers, formatters) remains unchanged
4. Agent IDs are available in all contexts where logging is added

### Constraints

1. Must not introduce measurable performance regression in hot paths
2. Must not log secrets, full commands, or unbounded data structures
3. Must follow existing codebase patterns for logger initialization
4. All new loggers use `logging.getLogger(__name__)` pattern

---

## Prerequisites

1. Read access to all 13 target files (confirmed via audit)
2. Understanding of existing logging patterns in `agent_session_runner.py`
3. Familiarity with `LoggerAdapter` pattern from Python stdlib

---

## High-Level Approach

### Logging Infrastructure Decision: LoggerAdapter

**Selected approach**: `LoggerAdapter` (not `contextvars`)

**Rationale**:
- Simpler to implement - no filter configuration needed
- Explicit propagation - clear where agent_id comes from
- Already familiar pattern - used in Python stdlib documentation
- Adequate for this use case - we pass agent_id explicitly through call chains

**Implementation location**: Each module creates its own adapter instance when agent_id is available:

```python
# In modules with agent context
import logging

logger = logging.getLogger(__name__)

class AgentLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        agent_id = self.extra.get('agent_id', '?')
        return f"[{agent_id}] {msg}", kwargs

# Usage in methods with agent_id
def some_method(self, agent_id: str, ...):
    log = AgentLoggerAdapter(logger, {"agent_id": agent_id})
    log.info("Event occurred: detail=%s", detail)
```

For modules without agent context, use the standard logger directly.

### Root Causes Addressed

1. **`WaitForGraph._waits` overwrites**: Add WARNING log when overwrite occurs
2. **Zero logging in domain layer**: Add logger to `deadlock.py` and `lifecycle.py`
3. **Minimal logging in hooks**: Add DEBUG logs for event emission
4. **No visibility into lock operations**: Add contention and timeout logs
5. **No hook wiring visibility**: Log when hooks are attached in `agent_runtime.py`
6. **No logging in most pipeline modules**: Add loggers to 4 pipeline files
7. **Orchestration wiring is opaque**: Add config and factory logs
8. **Lifecycle state machine is silent**: Log state transitions

---

## Detailed Tasks

### Phase 1: Configuration & Wiring Visibility (Highest Priority)

**Goal**: Answer "Is the system even active?" immediately during incident triage.

#### 1.1 src/orchestration/orchestrator.py - Monitor Config

Add logs in `__init__` method around deadlock monitor initialization (lines 218-222):

| Location | Log Statement | Level |
|----------|---------------|-------|
| After `if self._mala_config.deadlock_detection_enabled:` (L219), inside the if block | `"Deadlock detection enabled"` | **INFO** |
| In the `else` block (L221) | `"Deadlock detection disabled by config"` | **INFO** |

**Verification**: Run orchestrator with detection enabled/disabled, confirm appropriate log appears.
**Rollback**: Remove the 2 log statements.

#### 1.2 src/infra/agent_runtime.py - Hook Wiring

Add logs in the `build()` method where deadlock hooks are wired (lines 261-280):

| Location | Log Statement | Level |
|----------|---------------|-------|
| At start of `if self._deadlock_monitor is not None:` block (L262) | `"Wiring deadlock monitor hooks: agent_id=%s"` | **INFO** |
| After the `if self._deadlock_monitor` block ends (after L279) | `"No deadlock monitor configured; skipping lock event hooks"` | **INFO** |
| After hooks_dict is built (L294) | `"Built hooks: PreToolUse=%d PostToolUse=%d Stop=%d"` | DEBUG |

**Verification**: Start an agent session, confirm hook wiring log appears.
**Rollback**: Remove the 3 log statements.

---

### Phase 2: Event Ingestion & Flow (High Priority)

**Goal**: Answer "Are events flowing through the system?"

#### 2.1 src/infra/hooks/deadlock.py - Event Emission

| Location | Log Statement | Level |
|----------|---------------|-------|
| In `_make_lock_event_hook` after calling `emit_event()` | `"Lock event emitted: type=%s agent_id=%s lock_path=%s"` | DEBUG |
| In `_make_lock_wait_hook` after calling `emit_event()` | `"Lock event emitted: type=WAITING agent_id=%s lock_path=%s (pre-tool)"` | DEBUG |
| When multiple lock-wait commands detected | `"Multiple lock-wait commands found; emitting %d waits (may overwrite graph)"` | **WARNING** |
| In batch safety check | `"Batch safety: safe=%s, processing %d/%d commands"` | DEBUG |
| When skipping event (no event type) | `"Skipping event: cmd_type=%s exit_code=%s (no event type)"` | DEBUG |

**Verification**: Execute a lock-wait command, confirm DEBUG log appears.
**Rollback**: Remove the 5 log statements.

#### 2.2 src/domain/deadlock.py - Event Ingestion (DeadlockMonitor)

| Location | Log Statement | Level |
|----------|---------------|-------|
| `handle_event()` entry | `"Event received: type=%s agent_id=%s lock_path=%s"` | DEBUG |
| After graph update | `"Graph updated: holds=%d waits=%d"` | DEBUG |
| After cycle check | `"Cycle check: found=%s"` | DEBUG |

**Verification**: Emit a lock event, confirm DEBUG logs appear in monitor.
**Rollback**: Remove the 3 log statements.

---

### Phase 3: Graph State & Invariant Violations (High Priority)

**Goal**: Answer "What changed in the graph? Were there anomalies?"

#### 3.1 src/domain/deadlock.py - Graph State Changes

| Location | Log Statement | Level |
|----------|---------------|-------|
| `add_wait` when overwriting existing wait | `"Wait edge overwritten: agent_id=%s old_lock=%s new_lock=%s"` | **WARNING** |
| `add_hold` | `"Lock acquired: agent_id=%s lock_path=%s"` | DEBUG |
| `remove_hold` | `"Lock released: agent_id=%s lock_path=%s"` | DEBUG |
| `add_wait` | `"Wait added: agent_id=%s lock_path=%s"` | DEBUG |
| Cycle detected in `_check_for_deadlock` | `"Cycle detected: agents=%s"` | **WARNING** |

#### 3.2 src/domain/deadlock.py - Invariant Violations

| Location | Log Statement | Level |
|----------|---------------|-------|
| `add_hold` if lock held by other agent | `"Invariant: ACQUIRED for lock held by other agent: lock=%s holder=%s new_agent=%s"` | **WARNING** |
| `remove_hold` if not held by agent | `"Invariant: RELEASED for lock not held by agent: lock=%s holder=%s agent=%s"` | **WARNING** |
| `add_wait` if waiting on own lock | `"Invariant: WAITING on lock already held by same agent: agent=%s lock=%s"` | **WARNING** |
| Event for unregistered agent | `"Event for unregistered agent: agent_id=%s"` | **WARNING** |

#### 3.3 src/domain/deadlock.py - Agent Registration

| Location | Log Statement | Level |
|----------|---------------|-------|
| `register_agent` | `"Agent registered: agent_id=%s issue_id=%s"` | INFO |
| `unregister_agent` | `"Agent unregistered: agent_id=%s"` | INFO |
| `_select_victim` | `"Victim selected: agent_id=%s start_time=%f (youngest in cycle)"` | INFO |

**Verification**: Trigger a lock operation and confirm "Lock acquired" appears in DEBUG logs.
**Rollback**: Remove logger and all 13 log statements from this file.

---

### Phase 4: Resolution Path (High Priority)

**Goal**: Answer "Did resolution attempt succeed or fail?"

#### 4.1 src/orchestration/orchestrator.py - Deadlock Resolution

| Location | Log Statement | Level |
|----------|---------------|-------|
| `_handle_deadlock` entry | `"Deadlock resolution started: victim_id=%s issue_id=%s blocked_on=%s"` | **INFO** |
| After victim kill | `"Victim killed: agent_id=%s"` | INFO |
| After lock cleanup | `"Victim locks cleaned: agent_id=%s count=%d"` | INFO |
| Resolution failed (exception handler) | `"Deadlock resolution failed: agent_id=%s error=%s"` | **ERROR** |
| Before acquiring resolution lock | `"Acquiring deadlock resolution lock for victim %s"` | DEBUG |
| When skipping cleanup (already handled) | `"Skipped cleanup for %s (handled during deadlock)"` | DEBUG |

**Verification**: Trigger synthetic deadlock, confirm resolution logs appear.
**Rollback**: Remove the 6 log statements.

---

### Phase 5: Lock Operations (Medium Priority)

#### 5.1 src/infra/tools/locking.py

| Location | Log Statement | Level |
|----------|---------------|-------|
| `try_lock` success | `"Lock acquired: path=%s agent_id=%s"` | DEBUG |
| `try_lock` contention | `"Lock contention: path=%s holder=%s requester=%s"` | DEBUG |
| `wait_for_lock` timeout | `"Lock timeout: path=%s agent_id=%s after=%.1fs"` | **WARNING** |
| `release_lock` | `"Lock released: path=%s agent_id=%s"` | DEBUG |
| `cleanup_agent_locks` | `"Agent locks cleaned: agent_id=%s count=%d"` | INFO |

**Verification**: Acquire and release a lock, confirm DEBUG logs appear.
**Rollback**: Remove the 5 log statements.

---

### Phase 6: Lifecycle and State Machine

#### 6.1 src/domain/lifecycle.py

| Location | Log Statement | Level |
|----------|---------------|-------|
| `start()` | `"Lifecycle started: issue_id=%s state=%s"` | INFO |
| `on_gate_result()` | `"Gate result: issue_id=%s outcome=%s attempt=%d → state=%s"` | INFO |
| `on_review_result()` | `"Review result: issue_id=%s outcome=%s attempt=%d → state=%s"` | INFO |
| `on_messages_complete()` | `"Messages complete: issue_id=%s effect=%s"` | DEBUG |
| Retry decisions | `"Retry triggered: issue_id=%s reason=%s attempt=%d/%d"` | DEBUG |
| Terminal states | `"Lifecycle terminal: issue_id=%s state=%s message=%s"` | INFO |

**Verification**: Run an issue through gate/review, confirm lifecycle logs appear.
**Rollback**: Remove logger and 6 log statements.

---

### Phase 7: Pipeline Module Logging

#### 7.1 src/pipeline/issue_execution_coordinator.py

| Location | Log Statement | Level |
|----------|---------------|-------|
| `register_task()` | `"Task registered: issue_id=%s"` | DEBUG |
| `mark_failed()` | `"Issue marked failed: issue_id=%s"` | INFO |
| `mark_completed()` | `"Issue marked completed: issue_id=%s"` | DEBUG |
| `request_abort()` | `"Abort requested: reason=%s"` | WARNING |
| `run_loop()` | `"Loop iteration: active=%d pending=%d"` | DEBUG |

**Verification**: Run coordinator loop, confirm iteration logs appear at DEBUG.
**Rollback**: Remove logger and 5 log statements.

#### 7.2 src/pipeline/run_coordinator.py

| Location | Log Statement | Level |
|----------|---------------|-------|
| Lock acquisition | `"Acquiring lock: issue_id=%s path=%s"` | DEBUG |
| Lock wait | `"Waiting for lock: issue_id=%s path=%s"` | DEBUG |
| Step completion | `"Step completed: issue_id=%s step=%s"` | DEBUG |

**Verification**: Run a coordinated step, confirm lock logs appear.
**Rollback**: Remove logger and 3 log statements.

#### 7.3 src/pipeline/gate_runner.py

| Location | Log Statement | Level |
|----------|---------------|-------|
| `run_per_issue_gate()` | `"Gate check: issue_id=%s attempt=%d"` | DEBUG |
| No-progress detection | `"No progress detected: issue_id=%s"` | WARNING |
| Spec caching | `"Validation spec cached: issue_id=%s"` | DEBUG |

**Verification**: Run gate check, confirm DEBUG log appears.
**Rollback**: Remove logger and 3 log statements.

#### 7.4 src/pipeline/review_runner.py

| Location | Log Statement | Level |
|----------|---------------|-------|
| `run_review()` | `"Review started: issue_id=%s diff_range=%s"` | INFO |
| Review result | `"Review result: issue_id=%s passed=%s issues=%d"` | INFO |
| No-progress check | `"Review no-progress: issue_id=%s"` | WARNING |

**Verification**: Run a review, confirm INFO logs appear.
**Rollback**: Remove logger and 3 log statements.

---

### Phase 8: Orchestration & Infrastructure

#### 8.1 src/orchestration/factory.py

| Location | Log Statement | Level |
|----------|---------------|-------|
| `create_orchestrator()` | `"Orchestrator created: max_agents=%d timeout=%ds"` | INFO |
| `_check_review_availability()` | `"Review disabled: reason=%s"` | INFO |
| `_derive_config()` | `"Derived config: braintrust=%s timeout=%ds"` | DEBUG |

**Verification**: Create orchestrator, confirm INFO log appears.
**Rollback**: Remove logger and 3 log statements.

#### 8.2 src/infra/git_utils.py

| Location | Log Statement | Level |
|----------|---------------|-------|
| Git command failures | `"Git command failed: cmd=%s stderr=%s"` | WARNING |
| `get_baseline_for_issue()` | `"Baseline resolved: issue_id=%s commit=%s"` | DEBUG |

**Verification**: Resolve a baseline, confirm DEBUG log appears.
**Rollback**: Remove logger and 2 log statements.

#### 8.3 src/infra/clients/cerberus_review.py

| Location | Log Statement | Level |
|----------|---------------|-------|
| Review result | `"Review completed: issue_id=%s passed=%s issues=%d"` | INFO |
| Stale gate recovery | `"Stale gate resolved: issue_id=%s"` | INFO |
| Timeout handling | `"Review timeout: issue_id=%s after=%ds"` | WARNING |

**Verification**: Complete a review, confirm INFO log appears.
**Rollback**: Remove logger and 3 log statements.

---

## Risks, Edge Cases & Breaking Changes

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Log flooding in production | Low | Medium | All high-frequency events at DEBUG; guard expensive operations |
| Performance regression in hot paths | Low | Medium | Keep high-frequency events at DEBUG level |
| Incorrect log placement | Low | Low | Per-task verification catches misplacement |

### Edge Cases

1. **Agent ID unavailable**: Use `"?"` as fallback in LoggerAdapter
2. **Exceptions during logging**: Python's logging module handles this gracefully
3. **Concurrent access to graph during logging**: Only log counts, not data structures

### Breaking Changes

**None** - this is a logging-only change with no API or behavior modifications.

---

## Testing & Validation

### Automated Testing

**Unit test for `AgentLoggerAdapter`** (if centralized):
```python
def test_agent_logger_adapter_prefix():
    adapter = AgentLoggerAdapter(logger, {"agent_id": "test-1"})
    msg, _ = adapter.process("Event occurred", {})
    assert msg == "[test-1] Event occurred"
```

---

## Plan-Level Rollback Strategy

This is a logging-only change with minimal risk. Rollback options:

1. **Disable via log level**: Set logger level to WARNING in production config to suppress DEBUG/INFO logs without code changes
2. **Revert commits**: Standard `git revert` of logging commits removes all changes
3. **Per-module rollback**: Each phase can be reverted independently as commits are grouped by phase

**Recommended approach**: If performance issues are observed, first try option 1 (log level) before reverting code.

---

## Logging Anti-Patterns to Avoid

### 1. Don't Double-Log the Same Event
- Wrong: INFO in hook + INFO in monitor + INFO in orchestrator
- Right: Pick one canonical INFO location, use DEBUG elsewhere

### 2. Don't Dump Unbounded Data
- Wrong: `logger.debug("Graph state: %s", graph._holds)`
- Right: `logger.debug("Graph state: holds=%d waits=%d", len(graph._holds), len(graph._waits))`

### 3. Guard Expensive Computations
```python
if logger.isEnabledFor(logging.DEBUG):
    logger.debug("Detailed state: %s", expensive_computation())
```

### 4. Don't Log Secrets or Full Commands
- Wrong: `logger.info("Command: %s", full_command)`
- Right: `logger.info("Command: lock-wait.sh path=%s", truncated_path[:100])`

### 5. Use Lazy Formatting
- Wrong: `logger.info(f"Event: {event_type}")`
- Right: `logger.info("Event: %s", event_type)`

---

## Log Level Guidelines

| Level | Use For | Example |
|-------|---------|---------|
| **ERROR** | Resolution failures, unrecoverable errors | Deadlock resolution failed with exception |
| **WARNING** | Anomalies, invariant violations, timeouts | Wait edge overwritten, cycle detected, lock timeout |
| **INFO** | Configuration, lifecycle boundaries, outcomes | Detection enabled, agent registered, victim selected |
| **DEBUG** | High-frequency events, state changes, flow tracing | Lock acquired, event received, graph updated |

---

## Performance Considerations

1. **Keep high-frequency events at DEBUG**: Lock ACQUIRED/RELEASED, graph mutations, hook invocations
2. **De-duplicate "still waiting" messages**: Use time threshold per `(agent_id, lock_path)` pair
3. **Guard state dumps**: `if logger.isEnabledFor(logging.DEBUG)`
4. **Reuse computed values**: Don't canonicalize paths just for logging
5. **Lazy formatting**: Use `%s` instead of f-strings in log statements

---

## Implementation Order (Reordered for Incident Value)

1. **Phase 1**: Config/wiring logs → Immediately answers "is the system active?"
2. **Phase 2**: Hook emission + monitor ingestion → Answers "are events flowing?"
3. **Phase 3**: Graph mutations + invariant violations → Answers "what changed? were there anomalies?"
4. **Phase 4**: Resolution path → Answers "did we try to fix it? did it work?"
5. **Phase 5**: Lock operations → Deeper debugging of contention
6. **Phase 6-8**: Lifecycle, pipeline, infrastructure → Broader context

---

## Expected Debugging Workflow After Implementation

When deadlock detection fails, logs should answer these questions in order:

1. **Was detection enabled?** → `grep "Deadlock detection"` → expect "enabled" at INFO
2. **Were hooks wired?** → `grep "Wiring deadlock monitor"` → expect per-agent INFO
3. **Were events emitted?** → `grep "Lock event emitted"` → expect DEBUG per lock operation
4. **Were events received?** → `grep "Event received"` → expect DEBUG in monitor
5. **Were there overwrites?** → `grep "Wait edge overwritten"` → expect WARNING if bug triggered
6. **Were there invariant violations?** → `grep "Invariant:"` → expect WARNING if state inconsistent
7. **Was cycle detected?** → `grep "Cycle detected"` → expect WARNING if deadlock found
8. **Was resolution attempted?** → `grep "Deadlock resolution"` → expect INFO for attempt
9. **Did resolution succeed?** → `grep "Victim killed"` or `grep "resolution failed"` → expect INFO or ERROR

---

## Files Modified Summary

| File | Changes | LOC Delta |
|------|---------|-----------|
| **Phase 1-4: Critical Path** |||
| src/orchestration/orchestrator.py | Config + resolution logging | +12 |
| src/infra/agent_runtime.py | Add logger, hook wiring logs | +8 |
| src/infra/hooks/deadlock.py | Event emission + multi-wait warning | +10 |
| src/domain/deadlock.py | Add logger, ingestion + invariants + graph state | +25 |
| **Phase 5: Lock Operations** |||
| src/infra/tools/locking.py | Contention + timeout logging | +10 |
| **Phase 6: Lifecycle** |||
| src/domain/lifecycle.py | Add logger, state transition logs | +12 |
| **Phase 7: Pipeline** |||
| src/pipeline/issue_execution_coordinator.py | Add logger, task tracking | +8 |
| src/pipeline/run_coordinator.py | Add logger, lock acquisition logs | +8 |
| src/pipeline/gate_runner.py | Add logger, gate check logs | +6 |
| src/pipeline/review_runner.py | Add logger, review lifecycle logs | +6 |
| **Phase 8: Orchestration & Infra** |||
| src/orchestration/factory.py | Add logger, config logs | +6 |
| src/infra/git_utils.py | Add logger, baseline resolution logs | +6 |
| src/infra/clients/cerberus_review.py | Add logger, review outcome logs | +8 |
| **Total** | **13 files** | **~125 LOC** |

---

## Open Questions

None at this time. All design decisions have been made:
- LoggerAdapter selected over contextvars
- Per-task verification and rollback defined
- Testing strategy includes both unit and integration tests
