# Implementation Plan: Architecture Review High-Priority Refactors

## Context & Goals
- **Spec**: `docs/2026-01-01-architecture-review.md`
- **Review Status**: Passed (gemini, claude) / Needs Work addressed (codex)
- Implement the 3 high-priority refactors identified in the architecture review:
  1. EventSink base class to eliminate duplication (NullEventSink <20 lines)
  2. AgentSessionRunner.run_session extraction to reduce complexity (675 → <100 lines)
  3. MalaOrchestrator split with IssueExecutionCoordinator + legacy init removal (1565 → <1000 lines)
- Reduce complexity and duplication without behavioral changes; deliver as incremental PRs
- Target maintainability for contributors modifying events, session flow, and orchestration logic

## Scope & Non-Goals
- **In Scope**
  - EventSink refactor with BaseEventSink and NullEventSink simplification
  - AgentSessionRunner.run_session extraction into focused helper methods + unit tests
  - MalaOrchestrator legacy init removal and coordination extraction into IssueExecutionCoordinator + unit tests
- **Out of Scope (Non-Goals)**
  - Any behavioral or functional changes
  - Feature flags or rollout mechanisms
  - Medium/Low priority issues from the architecture review
  - Changes to the event types themselves or ConsoleEventSink logic

## Assumptions & Constraints
- Pure refactoring only; observable behavior stays the same
- PRs are incremental and independent; sequence is EventSink → AgentSessionRunner → MalaOrchestrator
- Existing tests cover behavior sufficiently to catch regressions
- The `MalaEventSink` protocol is stable; no new events need adding during this refactor
- **Verified**: `MalaEventSink` is decorated with `@runtime_checkable` (line 55 of event_sink.py), so `isinstance()` checks will work

### Implementation Constraints
- Extend existing modules; do not add new services
- Remove legacy init path in `src/orchestration/orchestrator.py` without adding backward-compatibility shims
- Follow existing patterns: `SessionCallbacks`, protocol-based DI, dataclass I/O
- Helper extraction pattern: follow existing `_build_hooks()`, `_build_agent_env()` style

### Testing Constraints
- Existing test suite must pass (85% coverage threshold enforced)
- Add unit tests for new helpers and IssueExecutionCoordinator
- Use existing test file locations (`tests/test_*.py` flat structure)

## Prerequisites
- [x] Read and align on `docs/2026-01-01-architecture-review.md`
- [x] Confirm PR sequencing and review expectations (EventSink → AgentSessionRunner → MalaOrchestrator)
- [x] Ensure test tooling (`uv`, `ruff`, `ty`, `pytest`) is available

## High-Level Approach
1. **PR 1 (EventSink)**: Introduce a no-op `BaseEventSink` class, reducing duplication and simplifying `NullEventSink` to a simple alias or thin subclass.
2. **PR 2 (AgentSessionRunner)**: Extract `run_session` into focused helper methods (`_build_sdk_options()`, `_run_message_iteration()`, `_handle_log_waiting()`, `_handle_gate_effect()`, `_handle_review_effect()`) and add unit tests.
3. **PR 3 (MalaOrchestrator)**: Create `IssueExecutionCoordinator` to hold execution loop logic, remove legacy init path entirely, and update ~55 test locations to use factory pattern.

## Prerequisites: Import Restructure (mala-iy6l)

**This plan assumes mala-iy6l is complete.** The iy6l epic restructures the codebase into layer-based packages. Key paths this plan depends on:
- `src/orchestrator.py` → `src/orchestration/orchestrator.py` (iy6l.6)
- `src/cli.py` → `src/cli/cli.py` (iy6l.7)

## File Existence Verification

| Path | Status | Notes |
|------|--------|-------|
| `src/pipeline/agent_session_runner.py` | Exists | 1064 lines, run_session at 389-1064 |
| `src/orchestration/orchestrator.py` | Exists (post-iy6l) | 1565 lines, dual init paths |
| `src/infra/io/event_sink.py` | Exists | 1396 lines, 38 event methods, `@runtime_checkable` on line 55 |
| `src/core/protocols.py` | Exists | Protocol definitions including `IssueProvider` |
| `src/pipeline/__init__.py` | Exists | Pipeline module exports |
| `src/pipeline/issue_execution_coordinator.py` | **New** | To be created in PR 3 |
| `tests/test_issue_execution_coordinator.py` | **New** | To be created in PR 3 |
| `tests/test_event_sink.py` | Exists | Extend for BaseEventSink tests |
| `tests/test_agent_session_runner.py` | Exists | Extend for helper method tests |
| `tests/conftest.py` | Exists | Add `make_orchestrator()` fixture in PR 3 |

## Detailed Plan

### Task 1: Introduce BaseEventSink and simplify NullEventSink (PR 1)
- **Goal**: Create a single source of truth for event method signatures with no-op defaults and reduce NullEventSink to <20 lines
- **Covers**: AC #1 (EventSink single source; new event updates only 2 places; NullEventSink <20 lines)
- **Depends on**: Prerequisites
- **Changes**:
  - `src/infra/io/event_sink.py`:
    - Add `BaseEventSink` class (after `MalaEventSink` protocol, around line 600) with explicit `pass` implementations for all 38 protocol methods
    - Replace `NullEventSink` class (lines 601-835) with: `NullEventSink = BaseEventSink` or a thin 3-line subclass with docstring
    - Update `ConsoleEventSink` (lines 837-1396) to inherit from `BaseEventSink` instead of defining all methods
    - Remove overridden methods in `ConsoleEventSink` that were just `pass` statements
  - `tests/test_event_sink.py`: Add tests for:
    - `BaseEventSink` implements `MalaEventSink` protocol (isinstance check)
    - `NullEventSink` is usable as `MalaEventSink`
    - `ConsoleEventSink` still works correctly (existing tests should cover this)
- **Verification**:
  - Run `uvx ty check` - type checking passes
  - Run `uv run pytest tests/test_event_sink.py -v` - all tests pass
  - Run `uv run pytest -m "unit or integration"` - full test suite passes
  - Verify `NullEventSink` is <20 lines in the file
  - Verify adding a new event method only requires: 1) protocol, 2) BaseEventSink, 3) ConsoleEventSink override if needed
- **Rollback**:
  - Revert commit; no data changes

### Task 2: Extract _build_sdk_options helper (PR 2, subtask 2a)
- **Goal**: Extract SDK options construction from run_session into a focused helper method
- **Covers**: AC #2 (run_session <100 lines; helpers testable)
- **Depends on**: Task 1 merged
- **Changes**:
  - `src/pipeline/agent_session_runner.py`:
    - Create `_build_sdk_options(self, agent_id: str) -> ClaudeAgentOptions` method
    - Move lines 446-468 (SDK options construction) into this helper
    - Update `run_session` to call `options = self._build_sdk_options(agent_id)`
  - `tests/test_agent_session_runner.py`: Add unit test for `_build_sdk_options()`
- **Verification**:
  - Run `uv run pytest tests/test_agent_session_runner.py -v`
  - Existing agent session tests pass unchanged
- **Rollback**:
  - Revert commit

### Task 3: Extract _run_message_iteration helper (PR 2, subtask 2b)
- **Goal**: Extract message streaming loop with idle timeout handling
- **Covers**: AC #2
- **Depends on**: Task 2
- **Changes**:
  - `src/pipeline/agent_session_runner.py`:
    - Create `MessageIterationContext` dataclass to hold mutable state:
      - `session_id: str | None`
      - `tool_calls_count: int`
      - `pending_lint_commands: dict[str, tuple[str, str]]`
    - Create `async def _run_message_iteration(self, client, input, lifecycle_ctx, iter_ctx: MessageIterationContext) -> bool`
    - Move lines 516-748 (message iteration with idle retry) into this helper
    - Returns `success: bool`; mutable state updated via `iter_ctx`
    - Handle `IdleTimeoutError` internally with retry logic
  - `tests/test_agent_session_runner.py`: Add unit tests for `_run_message_iteration()`:
    - Test normal message flow
    - Test idle timeout with retry
    - Test max retries exceeded
- **Verification**:
  - Run `uv run pytest tests/test_agent_session_runner.py -v`
  - Existing agent session tests pass unchanged
- **Rollback**:
  - Revert commit

### Task 4: Extract _handle_log_waiting helper (PR 2, subtask 2c)
- **Goal**: Extract log file waiting logic into focused helper
- **Covers**: AC #2
- **Depends on**: Task 3
- **Changes**:
  - `src/pipeline/agent_session_runner.py`:
    - Create `async def _handle_log_waiting(self, session_id, lifecycle, lifecycle_ctx) -> tuple[Path | None, TransitionResult]`
    - Move lines 772-802 (WAIT_FOR_LOG handling) into this helper
  - `tests/test_agent_session_runner.py`: Add unit test for `_handle_log_waiting()`
- **Verification**:
  - Run `uv run pytest tests/test_agent_session_runner.py -v`
- **Rollback**:
  - Revert commit

### Task 5: Extract _handle_gate_effect helper (PR 2, subtask 2d)
- **Goal**: Extract gate check and retry logic into focused helper
- **Covers**: AC #2
- **Depends on**: Task 4
- **Changes**:
  - `src/pipeline/agent_session_runner.py`:
    - Create `async def _handle_gate_effect(self, input, log_path, lifecycle, lifecycle_ctx) -> tuple[str | None, bool]`
    - Move lines 804-879 (RUN_GATE handling) into this helper
    - Returns `(pending_query: str | None, should_break: bool)`
  - `tests/test_agent_session_runner.py`: Add unit tests for `_handle_gate_effect()`:
    - Test gate pass
    - Test gate fail
    - Test gate retry
- **Verification**:
  - Run `uv run pytest tests/test_agent_session_runner.py -v`
- **Rollback**:
  - Revert commit

### Task 6: Extract _handle_review_effect helper (PR 2, subtask 2e)
- **Goal**: Extract review check and retry logic into focused helper
- **Covers**: AC #2
- **Depends on**: Task 5
- **Changes**:
  - `src/pipeline/agent_session_runner.py`:
    - Create `async def _handle_review_effect(self, input, log_path, lifecycle, lifecycle_ctx) -> tuple[str | None, bool, str | None]`
    - Move lines 881-1033 (RUN_REVIEW handling) into this helper
    - Returns `(pending_query, should_break, cerberus_review_log_path)`
  - `tests/test_agent_session_runner.py`: Add unit tests for `_handle_review_effect()`
- **Verification**:
  - Run `uv run pytest tests/test_agent_session_runner.py -v`
  - Verify `run_session` is now <100 lines
- **Rollback**:
  - Revert commit

### Task 7: Create IssueExecutionCoordinator (PR 3, subtask 3a)
- **Goal**: Create new coordinator class to hold execution loop logic
- **Covers**: AC #3 (coordinator testable without SDK dependencies)
- **Depends on**: Task 6 merged (PR 2 complete)
- **Changes**:
  - **New**: `src/pipeline/issue_execution_coordinator.py`:
    - Create `IssueExecutionCoordinator` class with protocol-based dependencies
    - Constructor takes: `beads: IssueProvider`, `event_sink: MalaEventSink`, `config: CoordinatorConfig`
    - Note: `IssueProvider` is imported from `src.core.protocols`
    - Create `CoordinatorConfig` dataclass with: `max_agents`, `max_issues`, `epic_id`, `only_ids`, `prioritize_wip`, `focus`
    - Implement `async def run_loop(self, spawn_callback, finalize_callback) -> int`
    - Implement `async def abort_active_tasks(self, reason: str) -> None`
  - `src/pipeline/__init__.py`:
    - Add import: `from src.pipeline.issue_execution_coordinator import IssueExecutionCoordinator, CoordinatorConfig`
    - Add to `__all__`: `"IssueExecutionCoordinator"`, `"CoordinatorConfig"`
  - **New**: `tests/test_issue_execution_coordinator.py`:
    - Test coordinator with mock `IssueProvider` and `NullEventSink`
    - Test spawn/completion flow without SDK
    - Test abort behavior
- **Verification**:
  - Run `uv run pytest tests/test_issue_execution_coordinator.py -v`
  - Run `uvx ty check`
- **Rollback**:
  - Delete new files, revert `__init__.py` changes

### Task 8: Migrate MalaOrchestrator to use IssueExecutionCoordinator (PR 3, subtask 3b)
- **Goal**: Update orchestrator to delegate to coordinator
- **Covers**: AC #3
- **Depends on**: Task 7
- **Changes**:
  - `src/orchestration/orchestrator.py`:
    - Import `IssueExecutionCoordinator` from `src.pipeline.issue_execution_coordinator`
    - Create coordinator instance in factory init path
    - Update `_run_main_loop` to delegate to `coordinator.run_loop()`
    - Keep `spawn_agent` and `_finalize_issue_result` as callbacks passed to coordinator
- **Verification**:
  - Run `uv run pytest tests/test_orchestrator.py -v`
  - Existing orchestrator tests pass
- **Rollback**:
  - Revert commit

### Task 9: Remove legacy init from MalaOrchestrator (PR 3, subtask 3c)
- **Goal**: Remove legacy initialization path entirely
- **Covers**: AC #3 (legacy init removed)
- **Depends on**: Task 8
- **Changes**:
  - `src/orchestration/orchestrator.py`:
    - Remove legacy `__init__` overload (lines 312-340)
    - Remove `_init_legacy` method entirely
    - Keep only factory-based initialization
    - Update `__init__` to require factory parameters (remove defaults)
  - **Test files requiring updates** (55+ locations use legacy `MalaOrchestrator()` constructor):
    - `tests/test_orchestrator.py` (~45 locations) - primary test file
    - `tests/test_epic_verifier.py` (2 locations)
    - `tests/test_run_level_validation.py` (4 locations)
    - `tests/test_wip_prioritization.py` (4 locations)
    - `tests/test_morph_integration.py` (1 location)
  - Create helper fixture `make_orchestrator()` in `tests/conftest.py` that uses factory pattern
  - Update all test files to use the new fixture or factory pattern
  - `src/cli/cli.py`: Verify uses factory pattern (should already via `create_orchestrator()`)
- **Verification**:
  - Run `uv run pytest tests/test_orchestrator.py -v`
  - Run `uv run pytest -m "unit or integration"` - full test suite passes
  - Verify no legacy init usage remains: `grep -r "MalaOrchestrator(" tests/ src/`
  - Verify `src/orchestration/orchestrator.py` line count significantly reduced (target: <1000 lines after coordinator extraction)
- **Rollback**:
  - Revert commit

**Note on line count target**: The original spec target of <400 lines is not achievable with IssueExecutionCoordinator extraction alone (~275 lines moved). The realistic target is <1000 lines. Achieving <400 lines would require additional extraction of `spawn_agent`, `run_implementer`, and validation logic into separate modules, which is out of scope for this plan.

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes
- **Event method additions**: After refactor, `BaseEventSink` must be updated when new events are added to protocol. Document this in code comments.
- **Session idle timeout behavior**: Must be preserved exactly in `_run_message_iteration()`. Use existing tests as behavioral specification.
- **WAIT_FOR_LOG / RUN_GATE / RUN_REVIEW flows**: Each effect handler must maintain exact lifecycle state transitions.
- **Coordinator lifecycle**: Agent spawning/cancellation/finalization callbacks must preserve all side effects.

### Breaking Changes & Compatibility
- **Potential Breaking Changes**:
  - Legacy init removal in Task 9 affects any external callers using direct `MalaOrchestrator()` constructor
- **Mitigations**:
  - Grep codebase for `MalaOrchestrator(` to find all callers before Task 9
  - All known internal callers use factory pattern already
- **Rollout Strategy**:
  - No feature flags; merge via sequential PRs with full test verification
  - Each PR is independently revertable

## Testing & Validation

- **Unit Tests**
  - `tests/test_event_sink.py`: BaseEventSink protocol compliance
  - `tests/test_agent_session_runner.py`: Helper method unit tests for each extracted method
  - `tests/test_issue_execution_coordinator.py`: Coordinator behavior without SDK
- **Integration / End-to-End Tests**
  - Existing `tests/test_orchestrator.py` covers orchestrator integration
  - Existing `tests/test_agent_session_runner.py` covers session flow
  - E2E tests (`uv run pytest -m e2e`) validate end-to-end behavior
- **Regression Tests**
  - All existing tests serve as regression coverage
  - Run full suite: `uv run pytest -m "unit or integration" -n auto`
- **Manual Verification**
  - Review diff to confirm no behavioral changes
  - Line count checks: NullEventSink <20, run_session <100, orchestrator.py <1000

### Acceptance Criteria Coverage
| Spec AC | Covered By |
|---------|------------|
| AC #1: EventSink single source; new event updates 2 places; NullEventSink <20 lines | Task 1 |
| AC #2: run_session <100 lines; helpers testable; no behavior change | Tasks 2-6 |
| AC #3: legacy init removed; coordinator testable; orchestrator significantly reduced | Tasks 7-9 |

**Note**: AC #3 orchestrator line count target revised from <400 to <1000 (see Task 9 note).

## Rollback Strategy (Plan-Level)
- Revert PRs in reverse order: PR 3 → PR 2 → PR 1
- Each PR is atomic and independently revertable
- Verify rollback success by rerunning `uv run pytest -m "unit or integration" -n auto`
- No data migrations; no cleanup beyond file reverts

## Open Questions
- None remaining. All review findings addressed:
  - ✅ Protocol `isinstance` check: Confirmed `MalaEventSink` is `@runtime_checkable`
  - ✅ Line-count targets: Revised orchestrator target from <400 to <1000 (realistic)
  - ✅ Legacy init test locations: Enumerated 55+ locations across 5 test files
  - ✅ `_run_message_iteration` return type: Added `MessageIterationContext` dataclass
  - ✅ `__init__.py` update: Added to Task 7
  - ✅ `IssueProvider` import path: Clarified in Task 7
