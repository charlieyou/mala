<!-- review-type: architecture-review -->

# Architecture Review (Synthesized)

## Method

- **Generator models**: Codex (GPT-5.2), Gemini, Claude - read-only analysis with independent traversal
- **Entry points reviewed**: `src/cli/cli.py`, `src/orchestration/orchestrator.py`, `src/pipeline/agent_session_runner.py`
- **Key files scanned**: `src/core/protocols.py` (~2348 LOC), `src/pipeline/run_coordinator.py` (~1582 LOC), `src/domain/validation/config.py` (~1231 LOC), `src/domain/validation/config_loader.py` (~1302 LOC), `src/infra/io/log_output/console.py`
- **Hotspot inventory**: Top files by LOC - `protocols.py` (2348), `orchestrator.py` (1590), `run_coordinator.py` (1582), `cli.py` (1560), `config_loader.py` (1301), `config.py` (1231), `agent_session_runner.py` (1162)
- **Assumptions**: Import-linter contracts enforced in CI; codebase has test coverage per guidelines

---

## Issues

### [High] MalaOrchestrator is a God Object

**Primary files**: `src/orchestration/orchestrator.py:200-1500`

**Category**: Cohesion | Boundaries | Testability

**Type**: Structural

**Confidence**: Very High (flagged by 2/3 models)

**Source**: Codex [P1], Gemini [P1]

**Context**: `MalaOrchestrator` owns too many responsibilities: CLI-derived config, runtime state (`OrchestratorState`), SIGINT handling (`_handle_sigint`), deadlock wiring, session callback wiring, per-issue execution (`run_implementer`), and main loop plumbing (`_run_main_loop`). The class already follows a factory pattern (docstring says "Use create_orchestrator() factory function" and constructor takes many deps as parameters), but it still constructs some runtime deps internally via `_build_runtime_deps` (`CommandRunner`, `EnvConfig`, `LockManager`). Even with extracted components like `RunCoordinator`, the orchestrator remains the nexus that knows every subsystem. This increases the blast radius for changes and makes the class harder to test in isolation.

**Fix**: Complete the existing factory pattern by moving remaining internal construction out:
1. Move `_build_runtime_deps` construction (`CommandRunner`, `EnvConfig`, `LockManager`) into the `create_orchestrator()` factory, returning via an `OrchestratorDeps` dataclass
2. Extract signal/lifecycle handling to a dedicated `LifecycleController`
3. Make `MalaOrchestrator` accept all deps and focus solely on sequencing

**Acceptance Criteria**:
- `MalaOrchestrator._build_runtime_deps` eliminated; `CommandRunner`, `EnvConfig`, `LockManager` provided via `create_orchestrator()` factory
- Orchestrator exposes a small surface: `run()`, `run_sync()`, and minimal lifecycle hooks
- A unit-testable "plan builder" exists that turns config/state to steps without touching IO
- Class LOC reduced by >25%

**Test Plan**:
- Unit test the extracted factory/builder without IO mocking
- Integration test that orchestrator works with injected deps
- Verify SIGINT handling works with the extracted controller

---

### [High] Callback-Factory Wiring Captures Mutable Orchestrator Internals

**Primary files**: `src/orchestration/orchestrator.py:360-395` (`_init_pipeline_runners`), `src/orchestration/orchestration_wiring.py:249-305` (`build_session_callback_factory`), `src/pipeline/agent_session_runner.py:339+` (`SessionCallbacks` dataclass)

**Category**: Boundaries | Testability | Abstraction

**Type**: Structural

**Confidence**: High (flagged by 2/3 models with slight variation)

**Source**: Codex [P1] "lambda getter soup", Gemini [P2] "Callback Bag Abstractions"

**Context**: `_init_pipeline_runners` builds the session callback factory with many lambdas that close over orchestrator state (`self._state.issue_base_shas`, `self._interrupt_event`, `self.run_coordinator.run_metadata`, `self.issue_coordinator.abort_event`). The heavy lambda/getter wiring occurs in `orchestrator.py` and is threaded through `build_session_callback_factory` in `orchestration_wiring.py`. The `SessionCallbacks` dataclass (in `agent_session_runner.py`) has 9 callable fields and intentionally supports late-bound getters. This makes the callback surface implicitly depend on orchestrator object identity and mutation timing ("set later, read later"), which is difficult to reason about, weakens type safety, and complicates control flow analysis compared to explicit Protocol-based injection.

**Fix**: Replace callback collections with defined Protocols:
1. Create a `SessionCallbackContext` dataclass containing only needed stable references
2. Define strict Protocols for major dependencies (e.g., `IGateRunner`, `IReviewRunner`, `GateKeeper`, `Reviewer`)
3. Thread `RunContext` (run_metadata, interrupt_event, abort_event) explicitly into `build()` calls instead of capturing `self`
4. Pass objects implementing protocols to runners, not individual closures

**Acceptance Criteria**:
- `build_session_callback_factory(...)` takes one context object instead of >8 lambdas
- Callbacks do not read orchestrator fields directly (only via provided context)
- Session callback construction becomes deterministic given `(issue_id, run_context)`
- `SessionCallbacks` replaced by explicit Protocols

**Test Plan**:
- Unit test callback factory with mock context object
- Verify type checker can validate protocol implementations
- Integration test that session runner works with injected protocol objects

---

### [Medium] RunCoordinator Mixed Concerns

**Primary files**: `src/pipeline/run_coordinator.py:344-834`

**Category**: Cohesion | Boundaries

**Type**: Structural

**Confidence**: High (flagged by 2/3 models)

**Source**: Codex [P2], Gemini [P2]

**Context**: `RunCoordinator` manages the global run loop but also contains extensive logic for "Trigger Validation" (queuing via `queue_trigger_validation`, `run_trigger_validation`, `_run_trigger_validation_loop`) and "Fixer Agent" spawning (`_run_fixer_agent`). The orchestrator contains trigger policy checks (e.g., `_check_and_queue_periodic_trigger`) while `RunCoordinator` owns the trigger queue and runs validations. This policy split across two places creates indirection when tracing trigger flow.

**Fix**: Centralize trigger policy in one module:
- Option A (functional): Create `trigger_policy.py` that takes `(event, state, config)` and returns `TriggerActions` (queue/run/skip reasons)
- Option B (OO-lite): Create a `TriggerEngine` owned by `RunCoordinator` with explicit methods `on_issue_completed`, `on_run_end`, `on_epic_closed`
- Extract fixer spawning into `FixerAgentService`

**Acceptance Criteria**:
- Orchestrator no longer computes fire-on conditions directly; it emits events to trigger engine
- Only one place in code decides "queue vs run now vs skip" for triggers
- Trigger logic and fixer logic in separate services
- `RunCoordinator` delegates to these services via interfaces

**Test Plan**:
- Unit test trigger policy with various event/state combinations
- Unit test fixer service independently
- Integration test full trigger flow

---

### [Medium] Global Mutable Console State

**Primary files**: `src/infra/io/log_output/console.py` (~455 LOC, globals throughout)

**Category**: Testability | Boundaries

**Type**: Structural

**Confidence**: Medium (flagged by 1/3 models)

**Source**: Codex [P2]

**Context**: Logging behavior is controlled by module globals (`_verbose_enabled`, `_agent_color_map`, `_agent_color_index`) accessed via `set_verbose()`, `is_verbose_enabled()`, and color assignment functions. Currently `src/cli/cli.py` calls `set_verbose()` at startup. This makes behavior dependent on import-time singleton state and can produce cross-test coupling and surprising behavior if multiple orchestrations occur in one process (e.g., watch mode).

**Fix**: Make logging a dependency:
1. Introduce a `ConsoleLogger` class (or pure functions taking a `ConsoleConfig` struct)
2. Pass it through `RuntimeDeps` / event sinks
3. Keep module-level functions as thin wrappers only if absolutely needed for CLI entry point

**Acceptance Criteria**:
- CLI calls `set_verbose()` only at startup; pipeline/orchestration receives verbosity via injected config
- Unit tests can construct loggers with deterministic color mapping and verbosity
- No cross-test state pollution via console globals

**Test Plan**:
- Unit test logger with explicit config
- Verify tests with different verbosity don't affect each other
- Integration test console output with injected logger

---

### [Medium] Pipeline Layer Depends on Infra Layer (Inverted Dependency)

**Primary files**: `src/pipeline/agent_session_runner.py:34`, `src/pipeline/run_coordinator.py`, `src/infra/agent_runtime.py`

**Category**: Boundaries

**Type**: Structural

**Confidence**: Medium (flagged by 1/3 models)

**Source**: Gemini [P2]

**Context**: The `pipeline` layer (business logic) directly imports and instantiates `AgentRuntimeBuilder` from the `infra` layer (`from src.infra.agent_runtime import AgentRuntimeBuilder`). This creates a coupling where the core runner depends on a specific runtime implementation, hindering testing and modularity. Note: import-linter does not currently enforce this specific boundary.

**Fix**: Define a `RuntimeFactory` protocol in `core` or `pipeline` and inject the implementation:
1. Create `RuntimeFactoryProtocol` in `src/core/protocols.py`
2. Have `AgentRuntimeBuilder` implement this protocol
3. Inject the factory into `AgentSessionRunner` instead of direct instantiation

**Acceptance Criteria**:
- `AgentSessionRunner` receives a factory protocol instead of importing `AgentRuntimeBuilder` directly
- Runtime creation uses an injected factory protocol
- Session runner testable with mock factory

**Test Plan**:
- Unit test session runner with mock runtime factory
- Consider adding import-linter contract for this boundary
- Integration test with real factory

---

### [Medium] Oversized protocols.py Mixes Unrelated Contracts

**Primary files**: `src/core/protocols.py:1-2348`

**Category**: Cohesion

**Type**: Structural

**Confidence**: Medium (flagged by 1/3 models, but 2348 LOC is significant evidence)

**Source**: Claude [P2]

**Context**: The protocols module defines 30+ protocols (IssueProvider, CodeReviewer, GateChecker, LogProvider, SDKClientProtocol, CommandRunnerPort, LockManagerPort, etc.) spanning completely different domains. At ~2350 lines, it acts as a "god module" for all interface definitions. Finding the relevant protocol for a change requires scanning the entire file. The module also includes concrete dataclasses (TriggerSummary, ValidationTriggersSummary, EventRunConfig) mixed with protocols.

**Fix**: Split into domain-focused protocol modules:
- `src/core/protocols/issue.py` - IssueProvider, IssueResolutionProtocol
- `src/core/protocols/review.py` - CodeReviewer, ReviewResultProtocol
- `src/core/protocols/validation.py` - GateChecker, ValidationSpecProtocol
- `src/core/protocols/infra.py` - CommandRunnerPort, LockManagerPort, EnvConfigPort
- `src/core/protocols/sdk.py` - SDKClientProtocol, SDKClientFactoryProtocol
- `src/core/protocols/events.py` - MalaEventSink, EventRunConfig, TriggerSummary

Move concrete dataclasses to `src/core/models.py` or dedicated modules.

**Acceptance Criteria**:
- Each protocol file is <300 LOC
- Concrete dataclasses separated from protocol definitions
- All imports updated directly (per repo "no re-exports" policy)

**Test Plan**:
- Update all imports across codebase when splitting
- Check import-linter contracts still pass
- Run full test suite to confirm no regressions

---

### [Medium] Config Parsing Duplication and Complexity

**Primary files**: `src/domain/validation/config.py:846-1141`, `src/domain/validation/config_loader.py:409-745`, `src/domain/validation/config_loader.py:817-1096`

**Category**: Duplication | Complexity

**Type**: Structural

**Confidence**: Medium (flagged by 1/3 models)

**Source**: Claude [P2], [P3]

**Context**: Parsing logic is spread across two files with overlapping concerns. `ValidationConfig.from_dict()` is a large method doing field extraction and validation, then `config_loader.py` has separate parser functions for triggers, code review, cerberus config, etc. The `config.py` file uses lazy imports back to `config_loader.py` (around lines 960, 1091, 1111), creating a circular conceptual dependency. Similar validation patterns (checking bool subclass of int, string emptiness, unknown fields) are repeated multiple times across both files (estimated ~10-15 occurrences). `_parse_code_review_config()` is also substantial in size.

**Fix**: Extract shared parsing utilities and restructure:
1. Create `src/domain/validation/config_parsing.py` with helpers: `parse_string_field()`, `parse_int_field()`, `parse_string_list()`, `validate_unknown_fields()`
2. Move all `_parse_*` functions from config_loader.py into config_parsing.py
3. Use declarative field specification pattern to reduce per-field code from ~15 lines to 1 line
4. Eliminate lazy imports by having config.py only define dataclasses

**Acceptance Criteria**:
- No lazy imports in config.py
- Validation helpers reused (reduce repeated bool/int check patterns)
- Parser functions simplified via shared helpers
- Clear unidirectional dependency: config_parsing.py -> config.py

**Test Plan**:
- Unit test parsing helpers with edge cases
- Verify all existing config parsing tests pass
- Test config loading with malformed YAML

---

### [Low] Domain Logic Leaking into CLI

**Primary files**: `src/cli/cli.py:1-220`, `src/cli/cli.py:1507+`

**Category**: Boundaries | Abstraction

**Type**: Structural

**Confidence**: Medium (flagged by 2/3 models with different angles)

**Source**: Codex [P2] "lazy import indirection", Gemini [P3] "parse_scope in CLI"

**Context**: Two related issues in CLI:
1. `src/cli/cli.py` implements a custom lazy loader (`__getattr__` + `_lazy_modules`) to avoid importing SDK-dependent modules before `bootstrap()`. This introduces dynamic behavior making static analysis and refactoring harder.
2. The CLI module contains domain logic like `parse_scope` which defines how scope strings (`ids:`, `epic:`) are interpreted. This logic should reside in the domain layer.

**Fix**:
1. Make bootstrapping explicit: Move SDK-dependent imports inside command handlers after `bootstrap()`, or create a dedicated `cli_runtime.py` imported only after bootstrap
2. Move `parse_scope` and `ScopeConfig` to a domain module (e.g., `src/domain/scope.py`)

**Acceptance Criteria**:
- No `__getattr__` indirection in `src/cli/cli.py`
- Import graph becomes statically understandable (tools like `grimp`/`ruff` can reason about boundaries)
- `parse_scope` logic moved to domain layer
- CLI delegates parsing to domain module

**Test Plan**:
- Unit test parse_scope in domain module
- Verify CLI startup time not affected by import changes
- Static analysis tools can trace imports

---

### [Low] MalaEventSink Protocol Has ~68 Methods

**Primary files**: `src/core/protocols.py:1514-2348`

**Category**: Abstraction

**Type**: Structural

**Confidence**: Low (flagged by 1/3 models)

**Source**: Claude [P3]

**Context**: `MalaEventSink` defines ~68 event methods spanning run lifecycle, agent lifecycle, gate events, review events, fixer events, validation events, trigger events, and session events. Implementing this interface requires stubbing all methods even if only interested in a subset.

**Fix**: Split into focused event sink protocols:
- `RunEventSink` - on_run_started, on_run_completed, on_ready_issues
- `AgentEventSink` - on_agent_started, on_agent_completed, on_tool_use
- `GateEventSink` - on_gate_started/passed/failed/retry
- `ReviewEventSink` - on_review_started/passed/retry
- `ValidationEventSink` - on_validation_started/result, on_trigger_* methods

Provide `CompositeEventSink` that accepts multiple focused sinks.

**Acceptance Criteria**:
- Each focused sink protocol has <15 methods
- Tests can implement minimal sink for their scope
- Existing console sink still works via composition

**Test Plan**:
- Verify existing event sink implementations still work
- Unit test with minimal focused sinks
- Integration test composite sink

---

## Summary

**High-confidence issues (multi-model agreement):**
1. MalaOrchestrator God Object - extract factory and lifecycle handling
2. Callback-factory wiring - replace lambda soup with explicit protocols
3. RunCoordinator mixed concerns - extract trigger and fixer services

**Medium-confidence issues (single model, strong evidence):**
4. Global mutable console state - inject logger dependency
5. Pipeline->Infra inverted dependency - use runtime factory protocol
6. Oversized protocols.py - split into domain-focused modules
7. Config parsing duplication - extract shared utilities

**Lower priority improvements:**
8. Domain logic in CLI - move parse_scope to domain
9. MalaEventSink size - split into focused protocols

No Critical/P0 issues identified. The architecture has strong foundations with enforced layer boundaries via import-linter. The identified issues are maintainability improvements that will reduce blast radius for changes and improve testability.
