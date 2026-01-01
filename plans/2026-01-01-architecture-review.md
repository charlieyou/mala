# Architecture Review: Breaking Up God Classes/Modules

**Date**: 2026-01-01
**Focus**: Breaking up god classes/modules/files
**Review Status**: Passed (claude, codex, gemini)

## Method

> **Note**: File paths updated to reflect post-iy6l structure (mala-iy6l must complete first).

- **Tools run**: lizard (complexity analysis), grimp (import graph analysis)
- **Entry points reviewed**: `src/cli/main.py`, `src/cli/cli.py`, `src/orchestration/orchestrator.py`
- **Key files scanned**: `src/orchestration/orchestrator.py` (1564 LOC), `src/infra/io/event_sink.py` (1396 LOC), `src/infra/epic_verifier.py` (1247 LOC), `src/pipeline/agent_session_runner.py` (1055 LOC), `src/cli/cli.py` (922 LOC)
- **Hotspot inventory**:
  - `run_session` (CCN 84, 501 NLOC) - extreme complexity
  - `_run_refresh` in coverage.py (CCN 54, 173 NLOC)
  - `on_run_started` in event_sink.py (CCN 37, 106 NLOC)
  - `verify_epic_with_options` (CCN 27, 164 NLOC)
  - `_run_main_loop` (CCN 25, 86 NLOC)
- **Assumptions**: Focus is on breaking up god modules; correctness issues deferred unless they indicate structural problems
- **Generator models used**: codex

---

## Summary

| # | Priority | Issue | Primary File | Key Metric |
|---|----------|-------|--------------|------------|
| 1 | High | `run_session` monolithic method | `pipeline/agent_session_runner.py:380-1055` | CCN 84, 675 lines |
| 2 | High | `MalaOrchestrator` mixed concerns | `orchestration/orchestrator.py` | 1564 LOC, 31 imports |
| 3 | High | EventSink duplication | `infra/io/event_sink.py` | ~45 methods duplicated |
| 4 | Medium | EpicVerifier duplicated flow | `infra/epic_verifier.py` | 2 methods with near-identical logic |
| 5 | Medium | CLI run command complexity | `cli/cli.py:241-572` | CCN 28, 281 NLOC |
| 6 | Medium | `_run_refresh` complexity | `domain/validation/coverage.py` | CCN 54, 173 NLOC |
| 7 | Low | Protocol definitions scattered | Multiple files | 5+ locations |
| 8 | Low | Prompt loading inconsistency | Multiple files | Mixed caching patterns |

---

## Issues

### [High] AgentSessionRunner.run_session is a 675-line monolithic method

**Primary files**: `src/pipeline/agent_session_runner.py:380-1055`
**Category**: Complexity | Cohesion
**Type**: task
**Confidence**: High
**Source**: codex
**Context**:
`run_session` has CCN 84 with 501 NLOC, far exceeding the 15 CCN / 80 line thresholds. It handles SDK client creation, message streaming, idle timeout retry logic, lifecycle state transitions, gate checks, review checks, and error handling all in a single method. This makes the method nearly impossible to unit test in isolation and creates high cognitive load for maintainers.

**Fix**: Extract cohesive phases into separate private methods or helper classes:
1. Extract SDK client setup and options building into `_build_sdk_client_options()`
2. Extract the message iteration loop (lines 507-733) into `_run_message_iteration()` returning success/retry state
3. Extract lifecycle effect handlers (gate check, review check, log waiting) into `_handle_gate_effect()`, `_handle_review_effect()`, `_handle_log_waiting()`
4. Consider a small state machine class for the idle retry logic

**Non-goals**:
- Changing the lifecycle state machine design
- Modifying the SDK client protocol

**Acceptance Criteria**:
- `run_session` reduced to <100 lines orchestrating extracted helpers
- Each extracted method has focused responsibility and is independently testable
- No behavioral changes to existing tests

**Test Plan**:
- Existing agent session tests pass unchanged
- Add unit tests for extracted helper methods

**Agent Notes**: The idle retry logic (lines 509-727) is tightly coupled to client lifecycle - consider a context manager or dedicated retry handler class.

---

### [High] MalaOrchestrator mixes configuration, wiring, and runtime orchestration

**Primary files**: `src/orchestration/orchestrator.py:301-755` (initialization), `src/orchestration/orchestrator.py:1351-1449` (main loop)
**Category**: Cohesion | Boundaries
**Type**: task
**Confidence**: High
**Source**: codex
**Context**:
`MalaOrchestrator` is 1564 lines with 31 import statements. It has dual initialization paths (`_init_legacy` 145 NLOC, `_init_from_factory`), contains both configuration defaults and runtime coordination, and mixes issue lifecycle management with validation coordination. This makes testing require heavy mocking and makes the class difficult to understand.

**Fix**:
1. Move the legacy initialization path to a separate `legacy_orchestrator_factory.py` module - it's backward compatibility code that shouldn't live in the core class
2. Extract `_run_main_loop` and related methods (`spawn_agent`, `_finalize_issue_result`, `_abort_active_tasks`) into a new `IssueExecutionCoordinator` class that receives pre-built dependencies
3. Keep `MalaOrchestrator` as a thin facade that composes the coordinator with configuration

**Non-goals**:
- Changing the factory pattern in `orchestrator_factory.py`
- Breaking the public `MalaOrchestrator` API

**Acceptance Criteria**:
- `MalaOrchestrator` reduced to <400 lines
- Legacy init path moved to separate module
- New `IssueExecutionCoordinator` is testable without SDK dependencies

**Test Plan**:
- Existing orchestrator tests pass
- Add focused unit tests for extracted coordinator

**Agent Notes**: The `_init_legacy` method has 24 parameters - this is a code smell indicating too many responsibilities. The factory path is much cleaner.

---

### [High] ConsoleEventSink duplicates NullEventSink method signatures

**Primary files**: `src/infra/io/event_sink.py:601-835` (NullEventSink), `src/infra/io/event_sink.py:837-1396` (ConsoleEventSink)
**Category**: Duplication | Abstraction
**Type**: task
**Confidence**: High
**Source**: codex
**Context**:
`src/infra/io/event_sink.py` is 1396 lines, primarily because `NullEventSink` (234 lines) and `ConsoleEventSink` (559 lines) both implement ~45 methods from `MalaEventSink` protocol. Every time a new event is added, 3 places must be updated (protocol, NullEventSink, ConsoleEventSink). The NullEventSink is pure boilerplate with `pass` implementations.

**Fix**:
1. Create a `BaseEventSink` class with default no-op implementations for all methods
2. Have `NullEventSink` simply be an alias or thin subclass of `BaseEventSink`
3. Have `ConsoleEventSink` inherit from `BaseEventSink` and only override methods that need console output
4. Alternatively, consider generating `NullEventSink` or using `__getattr__` to return no-op callables

**Non-goals**:
- Changing the event sink protocol itself
- Reducing the number of events

**Acceptance Criteria**:
- Single source of truth for event method signatures
- Adding a new event requires updating only 2 places (protocol + implementation in ConsoleEventSink)
- `NullEventSink` reduced to <20 lines

**Test Plan**:
- Event sink protocol check still passes
- Console output behavior unchanged

---

### [Medium] EpicVerifier has duplicated verification flow logic

**Primary files**: `src/infra/epic_verifier.py:353-525` (verify_and_close_eligible), `src/infra/epic_verifier.py:712-897` (verify_epic_with_options)
**Category**: Duplication | Complexity
**Type**: task
**Confidence**: High
**Source**: codex
**Context**:
`verify_and_close_eligible` (CCN 24, 134 NLOC) and `verify_epic_with_options` (CCN 27, 164 NLOC) contain nearly identical verification flow logic: acquire lock, emit events, call `_verify_epic_with_context`, handle pass/fail/human-review outcomes, release lock. The main difference is that one iterates over eligible epics while the other handles a single epic.

**Fix**:
1. Extract the core verification+outcome-handling logic into `_execute_verification_for_epic(epic_id, human_override, close_epic)` returning a single-epic result
2. Have `verify_and_close_eligible` iterate and call this helper
3. Have `verify_epic_with_options` call this helper for the single epic
4. Lock acquisition/release should be part of the helper

**Non-goals**:
- Changing verification model interface
- Modifying beads client interaction patterns

**Acceptance Criteria**:
- Single implementation of verification outcome handling
- Both public methods reduced to <40 lines each
- Reduced total LOC in src/infra/epic_verifier.py by ~100 lines

**Test Plan**:
- Epic verification tests pass unchanged
- Add test for extracted helper method

---

### [Medium] CLI run command embeds configuration parsing logic

**Primary files**: `src/cli/cli.py:241-572`
**Category**: Cohesion | Testability
**Type**: task
**Confidence**: Medium
**Source**: codex
**Context**:
The `run` command function (281 NLOC, CCN 28) mixes Typer parameter handling with configuration parsing (cerberus args, env parsing), validation, and orchestrator construction. This makes the CLI hard to test without running the actual command. The configuration parsing logic should be reusable for programmatic usage.

**Fix**:
1. Extract configuration validation and parsing into a `RunConfig` dataclass with a `from_cli_args()` factory method in `cli_support.py`
2. The `run` command becomes: parse args -> build RunConfig -> create orchestrator -> run
3. Tests can construct `RunConfig` directly without going through Typer

**Non-goals**:
- Changing Typer option definitions
- Modifying the orchestrator factory interface

**Acceptance Criteria**:
- `run` command reduced to <80 lines
- Configuration parsing logic is independently testable
- No changes to CLI user experience

**Test Plan**:
- CLI integration tests pass
- Add unit tests for `RunConfig.from_cli_args()`

---

### [Medium] validation/coverage.py _run_refresh is a 233-line function

**Primary files**: `src/domain/validation/coverage.py:461-694`
**Category**: Complexity
**Type**: task
**Confidence**: High
**Source**: codex
**Context**:
`_run_refresh` has CCN 54 (extreme cyclomatic complexity) with 173 NLOC. It handles coverage XML parsing, baseline comparison, threshold checking, and result formatting all in one function. This is the highest-complexity function in the validation subsystem.

**Fix**:
1. Extract coverage XML parsing into `_parse_and_validate_coverage(xml_path) -> CoverageData`
2. Extract baseline comparison into `_compare_with_baseline(current, baseline) -> ComparisonResult`
3. Extract threshold checking into `_check_threshold(data, threshold) -> ThresholdResult`
4. `_run_refresh` becomes an orchestrator calling these pure functions

**Non-goals**:
- Changing coverage report format
- Modifying how coverage baseline is stored

**Acceptance Criteria**:
- `_run_refresh` reduced to <50 lines
- Extracted functions are pure and independently testable
- Each extracted function has single responsibility

**Test Plan**:
- Coverage validation tests pass unchanged
- Add unit tests for extracted pure functions

---

### [Low] Protocol definitions spread across multiple modules

**Primary files**: `src/pipeline/agent_session_runner.py:95-155`, `src/lifecycle.py:47-125`, `src/infra/telemetry.py:33-66`, `src/infra/io/event_sink.py:56`, `src/core/protocols.py`
**Category**: Boundaries | Abstraction
**Type**: chore
**Confidence**: Medium
**Source**: synthesis
**Context**:
Protocol definitions are spread across multiple modules: `src/core/protocols.py` contains the main data and service protocols (13 protocols), but other protocols are defined inline near their implementations: `SDKClientProtocol`/`SDKClientFactory` in agent_session_runner.py, `GateOutcome`/`ReviewIssue`/`ReviewOutcome` in lifecycle.py, `TelemetrySpan`/`TelemetryProvider` in telemetry.py, and `MalaEventSink` in event_sink.py. This distributed pattern makes protocol discovery harder and creates ambiguity about where new protocols should be defined.

**Fix**:
1. Establish a clear convention: domain/data protocols in `src/core/protocols.py`, infrastructure protocols co-located with their implementations
2. Document this convention in CLAUDE.md
3. Consider moving `SDKClientProtocol` to core protocols since it's used across pipeline modules

**Non-goals**:
- Moving all protocols to a single file (some co-location is appropriate)
- Changing protocol method signatures

**Acceptance Criteria**:
- Clear documented convention for protocol placement
- No ambiguity about where new protocols should be defined

**Test Plan**:
- All imports continue to work
- Type checking passes

---

### [Low] Prompt template loading patterns differ across modules

**Primary files**: `src/orchestration/orchestrator.py:121-136`, `src/pipeline/agent_session_runner.py:73-82`, `src/infra/epic_verifier.py:86-93`
**Category**: Duplication
**Type**: chore
**Confidence**: Medium
**Source**: synthesis
**Context**:
Multiple modules have prompt loading functions that follow similar patterns. In `orchestration/orchestrator.py` and `pipeline/agent_session_runner.py`, these use `@functools.cache` decorators (`_get_implementer_prompt()`, `_get_review_followup_prompt()`, `_get_idle_resume_prompt()`). However, `_load_prompt_template()` in `src/infra/epic_verifier.py:86` does NOT use caching - it reads the file on every call and also does template escaping inline.

**Fix**:
1. Create a `src/prompts/__init__.py` with a generic `load_prompt(name: str) -> str` function using `@cache`
2. For template escaping needs (like epic_verifier), add optional escaping parameter or separate function
3. Replace individual loading functions with calls to the centralized loader

**Non-goals**:
- Changing prompt file locations
- Adding template variable support beyond what exists

**Acceptance Criteria**:
- Consistent prompt loading implementation across modules
- Epic verifier prompt loading benefits from caching
- Reduced code duplication (~30 lines)

**Test Plan**:
- Prompts load correctly in all contexts
- Cache behavior preserved where it existed
