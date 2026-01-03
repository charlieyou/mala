<!-- review-type: architecture-review -->

# Synthesized Architecture Review

## Method

- **Tools run**: lizard (complexity metrics), grimp (import graph analysis), rg/grep (code search), manual file inspection
- **Entry points**: `src/cli/main.py` -> `src/cli/cli.py` (Typer CLI), `src/orchestration/factory.py` (programmatic API)
- **Key files**: orchestrator.py (1110 LOC), agent_session_runner.py (1531 LOC), event_sink.py (1587 LOC), epic_verifier.py (1254 LOC), coverage.py (773 LOC), cli.py (956 LOC), cerberus_review.py (692 LOC)
- **Hotspot inventory**:
  - By complexity: `_run_refresh` (CCN 54), `on_run_started` (CCN 38), `_handle_review_effect` (CCN 34), `_run_message_iteration` (CCN 32), `cli.run` (CCN 31), `verify_and_close_eligible` (CCN 26)
  - By LOC: event_sink.py (1587), agent_session_runner.py (1531), epic_verifier.py (1254), orchestrator.py (1110)
- **Assumptions/unknowns**: Test coverage at 85% threshold per CLAUDE.md; protocol-based design patterns are intentional; no access to runtime profiling data
- **Generator models used**: codex (gpt-5.2-codex), gemini, claude

---

## Issues

### [High] Coverage baseline refresh is monolithic and side-effect heavy

**Primary files**: `src/domain/validation/coverage.py:526-774`
**Category**: Complexity
**Type**: task
**Confidence**: High
**Source**: multiple (codex, gemini, claude)
**Context**:
- `_run_refresh` has CCN 54 (threshold is 15), 179 NLOC, and 248 lines. It mixes lock handling, worktree creation, dependency installation (`uv sync`), command argument rewriting, coverage execution with fallback logic, and result parsing in a single method.
- This complexity makes the method hard to test in isolation, prone to subtle bugs, and creates significant risk when modifying coverage behavior.
**Fix**: Decompose into distinct phases: (a) pure command-rewrite helpers for argument manipulation, (b) a `BaselineRefresher` service with injected `CommandRunner`, `WorktreeManager`, and `LockManager`, (c) a separate `_run_coverage_with_fallback()` for combine/xml fallback logic, and (d) a pure parser for baseline extraction.
**Non-goals**:
- Changing the baseline refresh semantics or worktree behavior
- Adding new coverage backends
**Acceptance Criteria**:
- `_run_refresh` CCN reduced below 20
- Each extracted helper has focused responsibility
- Core command-rewrite logic covered by unit tests without filesystem or subprocesses
**Test Plan**:
- Unit tests for pure command manipulation helpers with various pytest marker combinations
- Integration test verifying full refresh flow in isolated temp directory
**Agent Notes**: The combine/xml fallback path has three retry strategies; preserve all edge cases when extracting.

---

### [High] `AgentSessionRunner._run_message_iteration` handles too many concerns

**Primary files**: `src/pipeline/agent_session_runner.py:1232-1491`
**Category**: Cohesion
**Type**: task
**Confidence**: High
**Source**: multiple (gemini, claude)
**Context**:
- This 260-line method (CCN 32) handles SDK client creation, query sending, response streaming, idle timeout detection, retry logic with backoff, tool result tracking, lint cache updates, and disconnect handling.
- The nested async iterator `_iter_messages()` and multiple try/except blocks indicate tangled responsibilities. Changes to idle timeout logic require understanding the entire method.
**Fix**: Extract retry orchestration into a separate `_retry_with_backoff()` coroutine. Extract message handling into `_process_sdk_stream()`. The timeout wrapper can be a standalone async context manager. This allows testing retry policy without mocking the full SDK.
**Non-goals**:
- Changing SDK message protocol handling
- Modifying existing timeout durations
**Acceptance Criteria**:
- `_run_message_iteration` CCN reduced below 20
- Retry logic testable without SDK mocks
- Idle timeout behavior unchanged
**Test Plan**:
- Unit tests for retry policy with mocked sleep/delay
- Integration test verifying full message iteration with test SDK client
**Agent Notes**: The idle timeout uses a sliding window; extracted helper must preserve the renewal logic on each received message.

---

### [High] Domain layer depends on infra/IO, breaking dependency direction

**Primary files**: `src/domain/quality_gate.py:21-22`, `src/domain/validation/spec_executor.py:21-23`, `src/domain/validation/coverage.py:25,461-546`, `src/domain/validation/e2e.py:24-25`, `src/domain/validation/helpers.py:12`
**Category**: Boundaries
**Type**: task
**Confidence**: High
**Source**: codex
**Context**:
- Domain modules import infra command execution (`run_command`), logging (`console.log`), locking (`lock_path`, `wait_for_lock`), and config helpers. This blurs the "functional core, imperative shell" split.
- Makes domain behavior hard to test without real IO and forces infra changes to ripple into domain layer.
**Fix**: Introduce explicit ports (protocols) for command execution, logging, and lock management in `src/core/protocols.py` (or a `domain/ports` module). Move IO implementations into infra and inject them into domain services via constructor/factory.
**Non-goals**:
- Creating adapter layers for every utility function
- Moving all domain code to pure functions (some orchestration is acceptable)
**Acceptance Criteria**:
- Domain modules no longer import `src.infra` directly; they depend on ports/interfaces
- Orchestrator/pipeline wires concrete infra implementations into domain services
- No behavioral changes
**Test Plan**:
- Verify no `from src.infra` imports in `src/domain/` after refactor
- Domain unit tests use fake implementations of ports
**Agent Notes**: The re-export shim at `src/domain/validation/command_runner.py` should be removed as part of this work.

---

### [High] MalaOrchestrator spans coordination, policy, and IO (god object)

**Primary files**: `src/orchestration/orchestrator.py:348-1040`
**Category**: Cohesion
**Type**: task
**Confidence**: High
**Source**: codex
**Context**:
- `MalaOrchestrator` owns gate/review execution, epic verification loops, run metadata, prompt assembly, issue finalization, and task scheduling. The class is 1110 LOC with fan-out to 27 modules.
- This centralizes too many responsibilities and makes high-risk edits (every change touches a large class).
**Fix**: Extract discrete coordinators (e.g., `IssueFinalizer`, `EpicVerificationCoordinator`, `GateReviewCoordinator`) with explicit inputs/outputs and inject them into the orchestrator. Keep orchestrator as thin wiring + flow control.
**Non-goals**:
- Creating abstract factory patterns for every responsibility
- Breaking the factory.py dependency injection pattern
**Acceptance Criteria**:
- Orchestrator delegates at least epic verification, gate/review execution, and result finalization to separate components
- Each new component has focused tests without requiring full orchestrator setup
- Orchestrator imports reduced below 20 modules
**Test Plan**:
- Unit tests for each extracted coordinator with mock dependencies
- Integration test verifying orchestrator correctly wires coordinators
**Agent Notes**: Start with `_finalize_issue_result` as it has clearest boundaries; epic verification has more git/beads entanglement.

---

### [Medium] Duplicated lock acquisition/release pattern in epic_verifier

**Primary files**: `src/infra/epic_verifier.py:428-449`, `src/infra/epic_verifier.py:807-831`, `src/infra/epic_verifier.py:898-911`
**Category**: Duplication
**Type**: task
**Confidence**: High
**Source**: claude
**Context**:
- Lock acquisition with `wait_for_lock` and release with `lp.unlink()` appears in three locations with nearly identical try/finally patterns. The `try: from src.infra.tools.locking import ...` pattern is repeated each time.
- Creates maintenance burden and risk of divergent lock handling.
**Fix**: Create an `@asynccontextmanager` helper `epic_verify_lock(epic_id, repo_path)` that encapsulates acquisition, timeout handling, and cleanup. Use it in `verify_and_close_eligible` and `verify_epic_with_options`.
**Non-goals**:
- Changing lock timeout values
- Introducing distributed locking
**Acceptance Criteria**:
- Lock handling consolidated into one async context manager
- All three lock sites use the same helper
- Lock behavior unchanged
**Test Plan**:
- Unit test for context manager with mock lock file operations
- Integration test verifying concurrent epic verification respects locks
**Agent Notes**: The lock timeout is configured via environment variable; ensure the helper accepts timeout as parameter.

---

### [Medium] CLI `run` command is a 366-line monolith

**Primary files**: `src/cli/cli.py:241-606`
**Category**: Cohesion
**Type**: task
**Confidence**: High
**Source**: multiple (gemini, claude)
**Context**:
- The `run()` function handles CLI argument parsing, validation, dry-run logic, configuration building, config overrides, OrchestratorConfig construction, and orchestrator invocation all in one function.
- This mixing of validation, config mutation, and orchestration makes testing difficult without invoking the full CLI.
**Fix**: Extract validation into `_validate_run_args()`, config overrides into `_apply_config_overrides(config, cli_overrides)`, and dry-run into `_handle_dry_run()`. The `run()` function becomes a thin orchestrator of these phases. Consider extracting a `ConfigurationFactory` in `src/orchestration/bootstrap.py` for reuse by programmatic callers.
**Non-goals**:
- Changing CLI argument names or behavior
- Splitting into multiple subcommands
**Acceptance Criteria**:
- `run()` function under 100 lines
- Config override logic unit-testable without typer
- Validation errors have clear test coverage
**Test Plan**:
- Unit tests for `_validate_run_args` with various invalid combinations
- Unit tests for `_apply_config_overrides` with partial overrides
**Agent Notes**: CLI pulls in underscore helpers (`_parse_cerberus_args/_parse_cerberus_env`) from infra; consider exposing public config resolution API instead.

---

### [Medium] `_handle_review_effect` mixes event emission with control flow

**Primary files**: `src/pipeline/agent_session_runner.py:1008-1230`
**Category**: Cohesion
**Type**: task
**Confidence**: High
**Source**: claude
**Context**:
- This 223-line method (CCN 34) interleaves event_sink calls, no-progress checks, retry policy decisions, and follow-up prompt building. The method returns a 4-tuple which indicates it's doing too much.
- Modifying event emission requires understanding the full review flow.
**Fix**: Separate concerns: `_check_review_no_progress()` for early exit, `_emit_review_events()` for event sink calls, `_build_review_retry_prompt()` for prompt formatting. Return a dataclass instead of a 4-tuple.
**Non-goals**:
- Changing review retry semantics
- Modifying event sink protocol
**Acceptance Criteria**:
- Return type simplified to dataclass
- Event emission testable without running full review
- CCN reduced below 15
**Test Plan**:
- Unit tests for each extracted helper
- Integration test verifying review effect handling end-to-end

---

### [Medium] `EpicVerifier` mixes domain, infra, and orchestration concerns

**Primary files**: `src/infra/epic_verifier.py:362-539`, `src/infra/epic_verifier.py:724-920`
**Category**: Boundaries
**Type**: task
**Confidence**: Medium
**Source**: gemini
**Context**:
- `EpicVerifier` performs raw Git operations (`_compute_scoped_commits`), interacts with the LLM (`verify`), and manages issue lifecycle (`create_remediation_issues`, `verify_and_close_eligible`). The git logic is complex and buried within the infra class.
- Hard to test the verification logic (pure domain) separately from git/beads side effects. Also contains duplicated processing logic between `verify_and_close_eligible` and `verify_epic_with_options`.
**Fix**: Extract `EpicScopeAnalyzer` (Git operations) to `src/domain/epic/scope.py`. Have `verify_and_close_eligible` delegate per-epic work to `verify_epic_with_options` in a loop, removing duplicated processing logic.
**Non-goals**:
- Changing LLM verification prompts
- Modifying beads client interface
**Acceptance Criteria**:
- Git logic (`_compute_scoped_commits`, `_summarize_commit_range`) moved to domain layer
- Shared verification logic exists in one location
- Both public methods behave identically for overlapping cases
**Test Plan**:
- Unit tests for scope analysis on complex git histories
- Integration test verifying epic verification flow
**Agent Notes**: The `verify_epic_with_options` was added later with additional parameters; ensure backward compatibility when consolidating.

---

### [Medium] Multiple logging paths bypass the event sink

**Primary files**: `src/domain/validation/spec_executor.py:146,221,363,380`, `src/infra/clients/cerberus_review.py:195,590`
**Category**: Testability
**Type**: task
**Confidence**: High
**Source**: codex
**Context**:
- Direct console logging via `log()` in domain/infra bypasses the `MalaEventSink` abstraction. This makes output harder to suppress, test, or redirect.
- Splits formatting responsibilities across modules.
**Fix**: Route logging through an injected sink/logger (or return structured events) from spec executor and reviewer; keep console output centralized in `ConsoleEventSink`.
**Non-goals**:
- Adding structured logging infrastructure
- Changing log message formats
**Acceptance Criteria**:
- No direct `log()` calls from domain/infra execution paths; output flows through injected interface
- Tests can silence output via `NullEventSink` without patching `console.log`
**Test Plan**:
- Verify no direct `log()` calls in domain/ after refactor
- Unit tests with NullEventSink verify no console output
**Agent Notes**: Some `log()` calls are in error paths; ensure error events are still emitted through sink.

---

### [Medium] `ConsoleEventSink.on_run_started` has excessive complexity

**Primary files**: `src/infra/io/event_sink.py:978-1103`
**Category**: Complexity
**Type**: task
**Confidence**: High
**Source**: claude
**Context**:
- This 125-line method (CCN 38) handles all startup logging with deeply nested conditionals for every config option.
- While the event sink class is intentionally cohesive, this specific method could be decomposed without fragmenting the class.
**Fix**: Extract helper methods: `_log_limits()`, `_log_review_config()`, `_log_morph_config()`, `_log_cli_args()`. Each handles one concern with simple conditionals.
**Non-goals**:
- Splitting ConsoleEventSink into multiple classes
- Changing startup log format
**Acceptance Criteria**:
- `on_run_started` CCN reduced below 15
- Startup log format unchanged
- Helpers remain private to the class
**Test Plan**:
- Snapshot test for startup log output format
- Unit tests for individual helper methods

---

### [Medium] Config parsing/normalization scattered across CLI and config

**Primary files**: `src/cli/cli.py:527-555`, `src/infra/io/config.py:32-67,340-395`, `src/orchestration/run_config.py:1-80`
**Category**: Abstraction
**Type**: task
**Confidence**: Medium
**Source**: codex
**Context**:
- CLI pulls in underscore helpers (`_parse_cerberus_args/_parse_cerberus_env`) directly, while derived fields are computed in run_config.py. This spreads configuration logic across layers.
- Makes it easy for CLI vs env config to diverge.
**Fix**: Introduce a `ResolvedConfig` builder in `infra/io/config.py` (or a small `config_resolver.py`) that consumes raw CLI overrides + env and returns a single validated object for orchestrator + event sink. CLI should not import private parse helpers.
**Non-goals**:
- Adding configuration file format
- Changing environment variable names
**Acceptance Criteria**:
- CLI no longer imports private `_parse_*` helpers
- Run config/event sink uses a single resolved config object for derived fields
**Test Plan**:
- Unit tests for ResolvedConfig builder with various env/CLI combinations
- Integration test verifying CLI and programmatic API produce equivalent configs
**Agent Notes**: Consider making `_parse_cerberus_args` public as `parse_cerberus_args` if external callers need it.

---

### [Low] Prompt loading scattered across layers with file IO in pipeline

**Primary files**: `src/orchestration/prompts.py:19`, `src/domain/prompts.py:24-29`, `src/pipeline/agent_session_runner.py:80-94,1222,1460`, `src/pipeline/run_coordinator.py:63-69,393`
**Category**: Abstraction
**Type**: chore
**Confidence**: Medium
**Source**: codex
**Context**:
- Prompts live in multiple modules and are read from disk inside pipeline stages. This fragments prompt ownership and complicates testing or swapping prompt sets.
**Fix**: Centralize prompt loading in a `PromptProvider` (pure data object or interface) and pass prompts into pipeline components; keep file IO at the boundary.
**Non-goals**:
- Template engine or dynamic prompt generation
- Changing prompt content
**Acceptance Criteria**:
- Pipeline components accept prompt strings/provider without reading files directly
- Prompt file IO is centralized in one module
**Test Plan**:
- Unit tests for pipeline components with injected prompts
- Verify prompt files loaded once at startup

---

### [Low] Event sink module mixes protocol, base class, and console formatting

**Primary files**: `src/infra/io/event_sink.py:1-1587`
**Category**: Cohesion
**Type**: chore
**Confidence**: Medium
**Source**: codex
**Context**:
- `event_sink.py` defines the protocol (46 methods), base sink, and full console rendering in one 1,587-line file. This makes changes risky and discourages adding new sinks.
**Fix**: Split into `event_protocol.py` (protocol only), `base_sink.py` (BaseEventSink), and `console_sink.py` (ConsoleEventSink + helpers). Dependencies minimal per module.
**Non-goals**:
- Reducing the number of event sink methods
- Breaking existing imports
**Acceptance Criteria**:
- Protocol, base class, and console sink live in separate modules
- New sinks can be added without importing console formatting
- Existing `from src.infra.io.event_sink import ...` continues to work via re-exports
**Test Plan**:
- Verify existing tests pass after split
- Add new test sink using only protocol module

---

### [Low] Back-compat re-export in domain/validation/command_runner blurs boundaries

**Primary files**: `src/domain/validation/command_runner.py:1-23`
**Category**: Boundaries
**Type**: chore
**Confidence**: High
**Source**: codex
**Context**:
- The domain package re-exports infra `CommandRunner`, which encourages domain consumers to depend on infra IO and obscures the true dependency direction.
**Fix**: Remove the re-export and update imports to infra directly (or define a domain protocol per the domain-infra boundary fix above).
**Non-goals**:
- Supporting external callers using this re-export (CLAUDE.md states no backward-compatibility shims)
**Acceptance Criteria**:
- No re-export module in `src/domain/validation/command_runner.py`
- Call sites use explicit infra imports or a domain-level interface
**Test Plan**:
- Grep for imports from `src/domain/validation/command_runner` and update
- Verify no regressions in test suite
**Agent Notes**: Per CLAUDE.md, no backward-compatibility shims are allowed; remove directly rather than deprecating.
**Dependencies**: Should be completed alongside or after "Domain layer depends on infra/IO" fix.

---

### [Low] Event sink protocol has 46 methods

**Primary files**: `src/infra/io/event_sink.py:57-637`
**Category**: Abstraction
**Type**: chore
**Confidence**: Low
**Source**: claude
**Context**:
- The `MalaEventSink` protocol defines 46 methods. While `BaseEventSink` provides no-op defaults, implementing custom event sinks requires awareness of all methods.
**Fix**: Consider grouping related events into sub-protocols (`RunLifecycleEvents`, `AgentEvents`, `GateEvents`) that the main protocol extends. This allows partial implementations for specialized sinks.
**Non-goals**:
- Reducing the number of events
- Breaking existing event sink implementations
**Acceptance Criteria**:
- Sub-protocols defined for logical groupings (optional - may not provide enough value)
- Existing implementations unchanged
- New event sinks can implement only relevant sub-protocols
**Test Plan**:
- Define one sub-protocol and verify partial implementation works
**Agent Notes**: This is a nice-to-have; the current `BaseEventSink` with no-op defaults is sufficient for most use cases.
