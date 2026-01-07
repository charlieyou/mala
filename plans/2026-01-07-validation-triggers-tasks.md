# Validation Triggers - Task Breakdown

**Plan**: `/home/cyou/mala/plans/2026-01-07-validation-triggers-plan.md`
**Spec**: `/home/cyou/mala/plans/2026-01-07-epic-validation-trigger-spec.md`
**Generated**: 2026-01-07

## Task Summary

| Phase | Tasks | Parallel | Dependencies |
|-------|-------|----------|--------------|
| Foundation | 1 | 0 | None |
| Config & Migration | 4 | 0 | T001 (sequential: file overlap on test file) |
| Trigger Execution | 3 | 0 | T002-T005 |
| Integration Points | 4 | 0 | T006-T008 (sequential: file overlap on orchestrator.py and test file) |
| Instrumentation | 1 | 0 | T008 |
| Documentation | 1 | 0 | T009-T012 |

**Total**: 14 tasks

---

## Phase 1: Foundation

### [T001] Add trigger config enums and dataclasses to config.py

**Type**: task
**Priority**: P1
**Story**: Foundation
**Parallel**:
**Primary Files**: src/domain/validation/config.py
**Subsystems**: domain
**Dependencies**: None

**Goal**:
Add all trigger-related enums (TriggerType, FailureMode, EpicDepth, FireOn) and dataclasses (TriggerCommandRef, BaseTriggerConfig, EpicCompletionTriggerConfig, SessionEndTriggerConfig, PeriodicTriggerConfig, ValidationTriggersConfig) to the config module. Also update the existing ValidationConfig dataclass to include the new validation_triggers field.

**Context**:
- Foundation for all trigger configuration
- Follows existing frozen dataclass pattern in config.py
- Enums parse from YAML strings (e.g., "abort" → FailureMode.ABORT)

**Scope**:
- In: Enums, dataclasses, type definitions, ValidationConfig field update
- Out: Parsing logic (T003), validation logic (T004)

**Changes**:
- `src/domain/validation/config.py` — Exists — Add:
  - `TriggerType` enum: EPIC_COMPLETION, SESSION_END, PERIODIC
  - `FailureMode` enum: ABORT, CONTINUE, REMEDIATE
  - `EpicDepth` enum: TOP_LEVEL, ALL
  - `FireOn` enum: SUCCESS, FAILURE, BOTH
  - `TriggerCommandRef` frozen dataclass: ref, command?, timeout?
  - `BaseTriggerConfig` frozen dataclass: failure_mode, commands, max_retries?
  - `EpicCompletionTriggerConfig(BaseTriggerConfig)`: epic_depth, fire_on
  - `SessionEndTriggerConfig(BaseTriggerConfig)`: (no additional fields)
  - `PeriodicTriggerConfig(BaseTriggerConfig)`: interval
  - `ValidationTriggersConfig` frozen dataclass: epic_completion?, session_end?, periodic?, is_empty()
  - **UPDATE ValidationConfig**: Add `validation_triggers: ValidationTriggersConfig | None = None` field

**Acceptance Criteria**:
- All 4 enums defined with correct values
- All 6 new dataclasses are frozen
- TriggerCommandRef supports optional command/timeout overrides
- BaseTriggerConfig has failure_mode (FailureMode), commands (tuple), max_retries (int|None)
- ValidationTriggersConfig.is_empty() returns True when all triggers are None
- **ValidationConfig has new validation_triggers field with None default**

**Verification**:
- Type check: `uvx ty check src/domain/validation/config.py`
- Import test: `python -c "from src.domain.validation.config import TriggerType, FailureMode, ValidationTriggersConfig, ValidationConfig"`
- Verify: `python -c "from src.domain.validation.config import ValidationConfig; print(ValidationConfig.__dataclass_fields__['validation_triggers'])"`

**Notes for Agent**:
- Use `tuple[TriggerCommandRef, ...]` not `list` for frozen dataclass fields
- Follow existing CommandConfig pattern for TriggerCommandRef
- All dataclasses must be frozen=True
- The validation_triggers field on ValidationConfig must default to None for backward compatibility

---

## Phase 2: Config & Migration

### [T002] [integration-path-test] Config parsing skeleton + failing integration test

**Type**: task
**Priority**: P1
**Story**: Config & Migration
**Parallel**:
**Primary Files**: src/domain/validation/config_loader.py, tests/integration/test_validation_triggers.py
**Subsystems**: domain, tests
**Dependencies**: T001

**Goal**:
Add skeleton parsing methods in config_loader.py (stubs that raise NotImplementedError) and create a failing integration test that exercises the config loading path with validation_triggers.

**Context**:
- TDD: Create the test first, then implement
- Integration test should load config via the normal path (config_loader.load_config)
- Skeleton ensures imports/wiring work before implementation

**Scope**:
- In: Skeleton parsing, failing integration test
- Out: Actual parsing implementation (T003)

**Changes**:
- `src/domain/validation/config_loader.py` — Exists — Add:
  - Update `_ALLOWED_TOP_LEVEL_FIELDS` to include `validation_triggers`
  - Stub `_parse_validation_triggers(data: dict) -> ValidationTriggersConfig | None` that raises NotImplementedError
  - Call stub from main loading path
- `tests/integration/test_validation_triggers.py` — New — Add:
  - `test_config_loads_validation_triggers_via_normal_path()` - loads mala.yaml with validation_triggers via load_config(), asserts ValidationTriggersConfig populated

**Acceptance Criteria**:
- Integration test exists and FAILS for the right reason (NotImplementedError)
- Skeleton method is called from main config loading path
- Test uses real config loading path (not direct dataclass construction)

**Verification**:
- Run: `uv run pytest tests/integration/test_validation_triggers.py::test_config_loads_validation_triggers_via_normal_path -v`
- Expected: FAILS with NotImplementedError (skeleton)
- After T003: Test passes

**Integration Path Test**:
This test exercises the config loading path through load_config() → ValidationConfig construction.

**Notes for Agent**:
- Create tests/integration/test_validation_triggers.py as new file
- Test should create a temp mala.yaml with validation_triggers section
- Skeleton should be minimal (just enough to wire up and fail)

---

### [T003] Implement config parsing for validation_triggers

**Type**: task
**Priority**: P1
**Story**: Config & Migration
**Parallel**:
**Primary Files**: src/domain/validation/config_loader.py, tests/unit/domain/test_validation_config.py
**Subsystems**: domain
**Dependencies**: T002

**Goal**:
Implement full parsing of validation_triggers from mala.yaml, including nested trigger configs and command refs.

**Context**:
- Replace NotImplementedError stub from T002
- Parse YAML dict into ValidationTriggersConfig and nested dataclasses
- Validate required fields, enum values, ref existence deferred to T004

**Scope**:
- In: YAML → dataclass parsing, unit tests
- Out: Migration validation (T004), merger logic (T005)

**Changes**:
- `src/domain/validation/config_loader.py` — Exists — Update:
  - Implement `_parse_validation_triggers()` with full parsing
  - Implement `_parse_trigger_command_ref(data: dict) -> TriggerCommandRef`
  - Implement `_parse_epic_completion_trigger(data: dict) -> EpicCompletionTriggerConfig`
  - Implement `_parse_session_end_trigger(data: dict) -> SessionEndTriggerConfig`
  - Implement `_parse_periodic_trigger(data: dict) -> PeriodicTriggerConfig`
  - Validate required fields: failure_mode for all triggers, max_retries when failure_mode=remediate, interval for periodic
- `tests/unit/domain/test_validation_config.py` — Exists — Add:
  - Test valid parsing with all triggers and fields
  - Test enum parsing (failure_mode, epic_depth, fire_on)
  - Test TriggerCommandRef with/without overrides
  - Test required field validation (failure_mode missing → error)
  - Test max_retries required when failure_mode=remediate
  - Test empty commands list is valid

**Wiring Map**:
- `mala.yaml validation_triggers → _parse_validation_triggers() → ValidationConfig.validation_triggers → ValidationTriggersConfig`

**Acceptance Criteria**:
- All trigger types parse correctly from YAML
- Enums parse from string values ("abort" → FailureMode.ABORT)
- Required field validation raises ConfigError with clear message
- TriggerCommandRef supports ref-only and ref+override forms
- Empty `commands: []` is valid (parsed as empty tuple)

**Verification**:
- Unit tests: `uv run pytest tests/unit/domain/test_validation_config.py -k trigger -v`
- Integration test from T002 now passes
- Type check: `uvx ty check src/domain/validation/config_loader.py`

**Notes for Agent**:
- Error message format: "failure_mode required for trigger epic_completion"
- Error message format: "max_retries required when failure_mode=remediate for trigger X"
- Use ConfigError for all validation failures

---

### [T004] Implement migration validation for deprecated configs

**Type**: task
**Priority**: P1
**Story**: Config & Migration
**Parallel**: (sequential - shares test file with T005)
**Primary Files**: src/domain/validation/config_loader.py, tests/unit/domain/test_validation_config.py
**Subsystems**: domain
**Dependencies**: T003, T005

**Goal**:
Add `_validate_migration()` function that fails fast on deprecated config patterns (validate_every, global_validation_commands without validation_triggers).

**Context**:
- BREAKING CHANGE: Existing configs with only global_validation_commands now fail
- Migration validation runs on effective merged config (after preset merge)
- Error messages must include migration guide URL and example snippet

**Scope**:
- In: Migration validation logic, unit tests
- Out: Actually merging configs (T005)

**Changes**:
- `src/domain/validation/config_loader.py` — Exists — Add:
  - `_validate_migration(config: ValidationConfig) -> None` function
  - Check 1: If `validate_every` present → ConfigError with deprecation message
  - Check 2: If `global_validation_commands` non-empty AND `validation_triggers` is None → ConfigError
  - Call `_validate_migration()` after config loading but before returning
- `tests/unit/domain/test_validation_config.py` — Exists — Add:
  - Test validate_every present → error with correct message
  - Test global_validation_commands without validation_triggers → error
  - Test empty global_validation_commands without validation_triggers → OK (empty pool is valid)
  - Test global_validation_commands WITH validation_triggers → OK
  - Test empty validation_triggers {} → OK (explicit opt-out)
  - Error message assertions (contains URL, example snippet)

**Acceptance Criteria**:
- validate_every triggers hard error at startup
- Non-empty global_validation_commands without validation_triggers triggers hard error
- Error messages include migration guide URL: https://docs.mala.ai/migration/validation-triggers
- Error messages include concrete example snippet

**Verification**:
- Unit tests: `uv run pytest tests/unit/domain/test_validation_config.py -k migration -v`
- Manual: Create mala.yaml with only global_validation_commands → startup fails with migration error

**Notes for Agent**:
- Error format for validate_every: "validate_every is deprecated. Use validation_triggers.periodic with interval field. See migration guide at https://docs.mala.ai/migration/validation-triggers"
- Error format for global_validation_commands: "validation_triggers required when global_validation_commands is defined. See migration guide at https://docs.mala.ai/migration/validation-triggers\n\nExample:\nvalidation_triggers:\n  session_end:\n    failure_mode: remediate\n    max_retries: 3\n    commands:\n      - ref: test"

---

### [T005] Implement field-level deep merge for global_validation_commands

**Type**: task
**Priority**: P1
**Story**: Config & Migration
**Parallel**: (sequential - shares test file with T004)
**Primary Files**: src/domain/validation/config_merger.py, tests/unit/domain/test_validation_config.py
**Subsystems**: domain
**Dependencies**: T003

**Goal**:
Implement field-level deep merge for global_validation_commands so project commands override preset commands by name, with unspecified fields inheriting from preset.

**Context**:
- Preset: `{test: {command: 'pytest', timeout: 300}}`
- Project: `{test: {timeout: 120}}`
- Effective: `{test: {command: 'pytest', timeout: 120}}`
- Per spec: project-specified fields override, unspecified fields inherit

**Scope**:
- In: Field-level merge logic, unit tests
- Out: validation_triggers merge (per spec, no inheritance - project-only)

**Changes**:
- `src/domain/validation/config_merger.py` — Exists — Update:
  - Modify merge logic for `global_validation_commands` field
  - For each command name in project: deep merge with preset entry (if exists)
  - Field-level precedence: project field > preset field > system default (120s for timeout)
- `tests/unit/domain/test_validation_config.py` — Exists — Add:
  - Test preset-only command inherited
  - Test project-only command added
  - Test field-level merge: project timeout overrides preset timeout, preset command inherited
  - Test project command field overrides preset command field
  - Test empty project global_validation_commands: {} → uses preset pool
  - Test project adds new command to preset pool

**Acceptance Criteria**:
- Project commands merge field-by-field with preset commands
- New project commands added to pool
- Empty project global_validation_commands uses preset pool
- Triggers can reference commands from either preset or project

**Verification**:
- Unit tests: `uv run pytest tests/unit/domain/test_validation_config.py -k merger -v`
- Type check: `uvx ty check src/domain/validation/config_merger.py`

**Notes for Agent**:
- validation_triggers does NOT merge (per spec: "presets only provide base pool")
- Only global_validation_commands gets field-level deep merge
- Keep existing merge logic for other fields unchanged

---

## Phase 3: Trigger Execution Engine

### [T006] [integration-path-test] RunCoordinator trigger queue skeleton + failing integration test

**Type**: task
**Priority**: P1
**Story**: Trigger Execution
**Parallel**:
**Primary Files**: src/pipeline/run_coordinator.py, tests/integration/test_validation_triggers.py
**Subsystems**: pipeline, tests
**Dependencies**: T002, T003, T004, T005

**Goal**:
Add skeleton methods for trigger queue management in RunCoordinator (queue_trigger_validation, run_trigger_validation, clear_trigger_queue) and create failing integration test that exercises the queue path.

**Context**:
- TDD: Create test first, implement later
- RunCoordinator already handles global validation and fixer agents
- Trigger queue is simple FIFO list

**Scope**:
- In: Skeleton methods, trigger queue data structure, failing integration test
- Out: Command execution implementation (T007), failure mode handling (T008)

**Changes**:
- `src/pipeline/run_coordinator.py` — Exists — Add:
  - `self._trigger_queue: list[tuple[TriggerType, dict[str, Any]]] = []` in __init__
  - `queue_trigger_validation(trigger_type: TriggerType, context: dict[str, Any]) -> None` - appends to queue
  - `run_trigger_validation(dry_run: bool = False) -> TriggerValidationResult` - stub raises NotImplementedError
  - `clear_trigger_queue(reason: str) -> None` - clears queue, emits skipped events (stub)
  - `TriggerValidationResult` dataclass: status (passed/failed/aborted), details
- `tests/integration/test_validation_triggers.py` — Exists — Add:
  - `test_trigger_queues_and_executes_via_run_coordinator()` - queues trigger, calls run_trigger_validation(), asserts result

**Acceptance Criteria**:
- Integration test exists and FAILS for the right reason (NotImplementedError)
- Trigger queue is initialized as empty list
- queue_trigger_validation appends to queue correctly
- Test exercises the path through RunCoordinator (not direct method calls)

**Verification**:
- Run: `uv run pytest tests/integration/test_validation_triggers.py::test_trigger_queues_and_executes_via_run_coordinator -v`
- Expected: FAILS with NotImplementedError (skeleton)
- After T007: Test passes

**Integration Path Test**:
This test exercises RunCoordinator.queue_trigger_validation() → run_trigger_validation() path.

**Notes for Agent**:
- TriggerValidationResult should be a frozen dataclass
- Status can be Literal["passed", "failed", "aborted"] or an enum

---

### [T007] Implement trigger command resolution and execution

**Type**: task
**Priority**: P1
**Story**: Trigger Execution
**Parallel**:
**Primary Files**: src/pipeline/run_coordinator.py, tests/unit/pipeline/test_trigger_execution.py
**Subsystems**: pipeline
**Dependencies**: T006

**Goal**:
Implement command resolution from base pool and sequential command execution with fail-fast behavior.

**Context**:
- Commands reference base pool via `ref` field
- Overrides (command, timeout) applied on top of base pool entry
- First failure stops remaining commands (fail-fast)
- Timeout treated as failure

**Scope**:
- In: Command resolution, sequential execution, fail-fast, dry-run mode
- Out: Failure mode handling (T008)

**Changes**:
- `src/pipeline/run_coordinator.py` — Exists — Update:
  - Implement `_resolve_trigger_commands(trigger_config, base_pool) -> list[ResolvedCommand]`
  - ResolvedCommand: ref, effective_command, effective_timeout
  - Implement `_execute_trigger_commands(commands, dry_run) -> list[CommandResult]`
  - Implement `run_trigger_validation()` body: resolve commands, execute, return result
  - Fail-fast: stop on first failure
  - Dry-run: skip subprocess execution, all commands treated as passed
- `tests/unit/pipeline/test_trigger_execution.py` — New — Add:
  - Test command resolution from base pool
  - Test command override (command string override)
  - Test timeout override
  - Test missing ref → ConfigError
  - Test fail-fast: second command fails → third not executed
  - Test timeout treated as failure
  - Test dry-run mode: all commands pass, no subprocess
  - Test empty command list → passed result

**Wiring Map**:
- `ValidationTriggersConfig → RunCoordinator.run_trigger_validation() → _resolve_trigger_commands() → _execute_trigger_commands() → CommandResult`

**Acceptance Criteria**:
- Commands resolve correctly from base pool
- Overrides applied field-by-field (command, timeout)
- Missing ref raises ConfigError with message listing available commands
- Fail-fast stops execution on first failure
- Dry-run mode skips subprocess, returns passed for all
- Empty command list returns passed status

**Verification**:
- Unit tests: `uv run pytest tests/unit/pipeline/test_trigger_execution.py -v`
- Integration test from T006 now passes

**Notes for Agent**:
- Error format for missing ref: "trigger epic_completion references unknown command 'typo'. Available: test, lint, typecheck"
- ResolvedCommand can be a dataclass or NamedTuple
- CommandResult should include: ref, passed, duration_seconds, error_message?

---

### [T008] Implement failure mode handling (abort/continue/remediate) + SIGINT

**Type**: task
**Priority**: P1
**Story**: Trigger Execution
**Parallel**:
**Primary Files**: src/pipeline/run_coordinator.py, tests/unit/pipeline/test_trigger_execution.py
**Subsystems**: pipeline
**Dependencies**: T007

**Goal**:
Implement failure_mode handling: abort sets result to aborted, continue logs and proceeds, remediate spawns fixer with retry loop. Also implement SIGINT handling for graceful interrupt during validation.

**Context**:
- Uses existing `_run_fixer_agent()` mechanism for remediation
- Remediation exhaustion → abort (terminal behavior per spec)
- Abort clears queue and emits skipped events
- SIGINT during validation must be responsive (no indefinite blocking)

**Scope**:
- In: Failure mode switch, remediation retry loop, abort handling, SIGINT handling
- Out: Integration points (T009-T012)

**Changes**:
- `src/pipeline/run_coordinator.py` — Exists — Update:
  - Add failure_mode handling in `run_trigger_validation()`:
    - ABORT: set result.status = "aborted", call clear_trigger_queue("run_aborted")
    - CONTINUE: log failure, set result.status = "failed", return (don't abort)
    - REMEDIATE: for attempt in range(max_retries): call _run_fixer_agent(), re-run failed command
  - Implement `clear_trigger_queue(reason)`: emit on_trigger_validation_skipped for each queued item, clear list
  - Remediation exhaustion → abort
  - **SIGINT handling**:
    - Register signal handler when validation starts
    - On SIGINT: send SIGTERM to subprocess, set internal `_interrupted` flag
    - Check `_interrupted` after each command; if set, abort remaining commands and clear queue
    - Restore original signal handler when validation completes
- `tests/unit/pipeline/test_trigger_execution.py` — Exists — Add:
  - Test abort mode: failure sets aborted status, clears queue
  - Test continue mode: failure logs, returns failed status, doesn't abort
  - Test remediate mode: spawns fixer, re-runs command
  - Test remediate exhaustion: aborts after max_retries
  - Test remediate success: fixer fixes, command passes, continues
  - Test max_retries=0: no fixer spawned, immediate abort
  - Test SIGINT: validation interrupted, queue cleared, result is aborted

**Acceptance Criteria**:
- ABORT mode sets aborted status and clears queue
- CONTINUE mode logs failure but doesn't abort
- REMEDIATE mode spawns fixer up to max_retries times
- Remediation exhaustion aborts the run
- max_retries=0 means no fixer spawned
- SIGINT immediately releases blocking wait and aborts

**Verification**:
- Unit tests: `uv run pytest tests/unit/pipeline/test_trigger_execution.py -k failure_mode -v`
- Type check: `uvx ty check src/pipeline/run_coordinator.py`

**Notes for Agent**:
- Reuse existing _run_fixer_agent() - don't reimplement
- Event emission for remediation handled in T013
- Keep failure_mode switch clean and readable
- Per spec: SIGINT must be responsive (user can always cancel)
- Signal handling: register handler at start of validation, restore original after

---

## Phase 4: Integration Points

### [T009] [integration-path-test] Orchestrator trigger hooks skeleton + failing integration test

**Type**: task
**Priority**: P1
**Story**: Integration Points
**Parallel**:
**Primary Files**: src/orchestration/orchestrator.py, tests/integration/test_validation_triggers.py
**Subsystems**: orchestration, tests
**Dependencies**: T006, T007, T008

**Goal**:
Add skeleton hooks in orchestrator for trigger firing (periodic counter, session_end location, blocking wait) and create failing integration test.

**Context**:
- Orchestrator main loop needs hooks for periodic and session_end triggers
- Epic completion hooked in EpicVerificationCoordinator (T010)
- Blocking: main loop waits for run_trigger_validation() before next issue

**Scope**:
- In: Orchestrator skeleton hooks, non_epic_completed_count, failing integration test
- Out: Actual trigger integration (T010-T012)

**Changes**:
- `src/orchestration/orchestrator.py` — Exists — Add:
  - `self._non_epic_completed_count: int = 0` state variable
  - Stub `_check_and_queue_periodic_trigger()` - increments counter, checks interval, queues if match
  - Stub `_fire_session_end_trigger()` - checks skip conditions, queues trigger
  - Identify blocking wait location (after queueing, before next issue assignment)
- `tests/integration/test_validation_triggers.py` — Exists — Add:
  - `test_orchestrator_fires_periodic_trigger_at_interval()` - configures interval=2, runs 2 issues, asserts trigger queued

**Acceptance Criteria**:
- Integration test exists and FAILS (skeleton raises NotImplementedError or doesn't queue)
- non_epic_completed_count initialized to 0
- Stub methods exist at correct locations in orchestrator

**Verification**:
- Run: `uv run pytest tests/integration/test_validation_triggers.py::test_orchestrator_fires_periodic_trigger_at_interval -v`
- Expected: FAILS (skeleton doesn't implement logic yet)
- After T011: Test passes

**Integration Path Test**:
This test exercises orchestrator main loop → periodic trigger queueing path.

**Notes for Agent**:
- Identify the exact line (~1135 per plan) for session_end trigger
- Don't implement full logic yet - just stubs and test

---

### [T010] Implement epic_completion trigger integration

**Type**: task
**Priority**: P1
**Story**: Integration Points
**Parallel**: (sequential - shares test file with T011, T012)
**Primary Files**: src/pipeline/epic_verification_coordinator.py, tests/unit/pipeline/test_trigger_execution.py
**Subsystems**: pipeline
**Dependencies**: T009

**Goal**:
Hook epic_completion trigger into EpicVerificationCoordinator.check_epic_closure() to queue validation after epic verification.

**Context**:
- Fires after epic verification completes (success or failure)
- Respects epic_depth filter (top_level vs all)
- Respects fire_on filter (success, failure, both)

**Scope**:
- In: Epic completion trigger hook, filter logic
- Out: Other triggers (T011, T012)

**Changes**:
- `src/pipeline/epic_verification_coordinator.py` — Exists — Update:
  - After verification result determined, check if epic_completion trigger configured
  - Apply epic_depth filter: if top_level, check epic has no epic parent
  - Apply fire_on filter: success, failure, or both
  - If filters pass, call run_coordinator.queue_trigger_validation(TriggerType.EPIC_COMPLETION, context)
  - Context: epic_id, depth, verification_result
- `tests/unit/pipeline/test_trigger_execution.py` — Exists — Add:
  - Test epic_depth=top_level: nested epic doesn't fire
  - Test epic_depth=all: nested epic fires
  - Test fire_on=success: only fires on pass
  - Test fire_on=failure: only fires on fail
  - Test fire_on=both: always fires
  - Test leaf epic with no children fires

**Acceptance Criteria**:
- epic_completion trigger fires after check_epic_closure()
- epic_depth=top_level only fires for epics with no epic parent
- fire_on filter respected correctly
- Context includes epic_id, depth, verification_result

**Verification**:
- Unit tests: `uv run pytest tests/unit/pipeline/test_trigger_execution.py -k epic_completion -v`
- Type check: `uvx ty check src/pipeline/epic_verification_coordinator.py`

**Notes for Agent**:
- EpicVerificationCoordinator needs access to run_coordinator (may need to inject)
- Check existing epic parent detection logic in codebase

---

### [T011] Implement periodic trigger integration

**Type**: task
**Priority**: P1
**Story**: Integration Points
**Parallel**: (sequential - shares orchestrator.py with T012, shares test file with T010, T012)
**Primary Files**: src/orchestration/orchestrator.py, tests/unit/pipeline/test_trigger_execution.py
**Subsystems**: orchestration
**Dependencies**: T009, T010

**Goal**:
Implement periodic trigger logic: increment non_epic_completed_count after non-epic issue completion, fire trigger at interval multiples.

**Context**:
- Counter increments AFTER per-session validation, BEFORE checking periodic condition
- Only non-epic issues increment counter (epic completions don't)
- Fires at interval, 2*interval, 3*interval... (continuous counter)

**Scope**:
- In: Periodic trigger counter and firing logic
- Out: Other triggers (T010, T012)

**Changes**:
- `src/orchestration/orchestrator.py` — Exists — Update:
  - Implement `_check_and_queue_periodic_trigger()`:
    - Only call for non-epic issues
    - Increment _non_epic_completed_count
    - If count % interval == 0 AND periodic trigger configured: queue trigger
  - Call from main loop after per-session validation completes
- `tests/unit/pipeline/test_trigger_execution.py` — Exists — Add:
  - Test counter increments only for non-epic issues
  - Test trigger fires at exact interval (5, 10, 15...)
  - Test epic issue doesn't increment counter
  - Test interval=1 fires every issue
  - Test fewer issues than interval → never fires

**Acceptance Criteria**:
- Counter only increments for non-epic issues
- Trigger fires at interval multiples (continuous counter, no reset)
- Counter state is in-memory only (resets on restart)

**Verification**:
- Unit tests: `uv run pytest tests/unit/pipeline/test_trigger_execution.py -k periodic -v`
- Integration test from T009 now passes

**Notes for Agent**:
- Find the exact location in main loop after per-session validation
- Counter is simple int, no persistence needed

---

### [T012] Implement session_end trigger integration with blocking

**Type**: task
**Priority**: P1
**Story**: Integration Points
**Parallel**: (sequential - shares orchestrator.py with T011, shares test file with T010, T011)
**Primary Files**: src/orchestration/orchestrator.py, tests/unit/pipeline/test_trigger_execution.py
**Subsystems**: orchestration
**Dependencies**: T009, T010, T011

**Goal**:
Implement session_end trigger with skip conditions and blocking execution that prevents next issue assignment until validation completes.

**Context**:
- Skip conditions: abort_run=True OR success_count==0
- Blocking: main loop waits for run_trigger_validation() before assigning next issue
- session_end fires once at end of run

**Scope**:
- In: Session end trigger, skip conditions, blocking wait
- Out: Other triggers (T010, T011)

**Changes**:
- `src/orchestration/orchestrator.py` — Exists — Update:
  - Implement `_fire_session_end_trigger()`:
    - Check skip: if abort_run=True, skip (abort takes precedence)
    - Check skip: if success_count==0, skip (nothing to validate)
    - If session_end configured: queue trigger, call run_trigger_validation()
  - Add blocking wait after queueing ANY trigger (not just session_end):
    - Call run_coordinator.run_trigger_validation() synchronously
    - Check result: if aborted, set abort_run=True
  - Call _fire_session_end_trigger() at existing session end location (~line 1135)
- `tests/unit/pipeline/test_trigger_execution.py` — Exists — Add:
  - Test session_end fires when success_count > 0 and !abort_run
  - Test session_end skipped when abort_run=True
  - Test session_end skipped when success_count==0
  - Test blocking: no new issue assigned during validation
  - Test abort from trigger sets abort_run=True

**Acceptance Criteria**:
- session_end fires at correct location (existing session end)
- Skip conditions respected (abort_run, success_count)
- Blocking wait prevents next issue assignment
- Abort result sets orchestrator abort_run flag

**Verification**:
- Unit tests: `uv run pytest tests/unit/pipeline/test_trigger_execution.py -k session_end -v`
- Type check: `uvx ty check src/orchestration/orchestrator.py`

**Notes for Agent**:
- success_count includes both epics and non-epic issues
- Blocking is synchronous call to run_trigger_validation()
- Find exact line for session_end trigger (~1135 per plan)

---

## Phase 5: Instrumentation

### [T013] Add MalaEventSink trigger events and console handlers

**Type**: task
**Priority**: P2
**Story**: Instrumentation
**Parallel**: (sequential - shares run_coordinator.py with T007, T008)
**Primary Files**: src/core/protocols.py, src/infra/io/log_output/console.py, src/pipeline/run_coordinator.py
**Subsystems**: core, infra, pipeline
**Dependencies**: T006, T007, T008

**Goal**:
Add 10 new MalaEventSink methods for trigger lifecycle and implement console handlers to log trigger events.

**Context**:
- Events defined in spec: queued, started, command_started, command_completed, passed, failed, skipped, remediation_started, remediation_succeeded, remediation_exhausted
- Console handlers provide user visibility into trigger execution

**Scope**:
- In: Protocol methods, console handlers, event emission in RunCoordinator
- Out: Metrics/telemetry (future work)

**Changes**:
- `src/core/protocols.py` — Exists — Add to MalaEventSink:
  - `on_trigger_validation_queued(trigger_type: str, trigger_context: str) -> None`
  - `on_trigger_validation_started(trigger_type: str, commands: list[str]) -> None`
  - `on_trigger_command_started(trigger_type: str, command_ref: str, index: int) -> None`
  - `on_trigger_command_completed(trigger_type: str, command_ref: str, index: int, passed: bool, duration_seconds: float) -> None`
  - `on_trigger_validation_passed(trigger_type: str, duration_seconds: float) -> None`
  - `on_trigger_validation_failed(trigger_type: str, failed_command: str, failure_mode: str) -> None`
  - `on_trigger_validation_skipped(trigger_type: str, reason: str) -> None`
  - `on_trigger_remediation_started(trigger_type: str, attempt: int, max_retries: int) -> None`
  - `on_trigger_remediation_succeeded(trigger_type: str, attempt: int) -> None`
  - `on_trigger_remediation_exhausted(trigger_type: str, attempts: int) -> None`
- `src/infra/io/log_output/console.py` — Exists — Add handlers:
  - Implement all 10 methods with console logging
  - Format: "[trigger_type] event_name: details"
- `src/pipeline/run_coordinator.py` — Exists — Add event emission:
  - Emit events at appropriate points in run_trigger_validation()
  - Emit events in queue/clear methods

**Acceptance Criteria**:
- All 10 event methods added to MalaEventSink protocol
- Console handlers log all events with trigger_type and details
- Events emitted at correct lifecycle points in RunCoordinator
- Existing MalaEventSink implementations updated (if any besides console)

**Verification**:
- Type check: `uvx ty check src/core/protocols.py`
- Manual: Run with triggers configured, verify console output shows trigger events
- All MalaEventSink implementers compile without errors

**Notes for Agent**:
- Check for other MalaEventSink implementers that need updating
- Events should be non-blocking (synchronous but fast)
- Follow existing event method patterns in protocols.py

---

## Phase 6: Documentation

### [T014] Create validation-triggers.md user documentation

**Type**: chore
**Priority**: P3
**Story**: Documentation
**Parallel**:
**Primary Files**: docs/validation-triggers.md
**Subsystems**: docs
**Dependencies**: T009, T010, T011, T012

**Goal**:
Create comprehensive user documentation for validation trigger configuration.

**Context**:
- Users need to understand the new trigger system
- Must cover migration from old config format
- Include practical examples for common use cases

**Scope**:
- In: User documentation, examples, migration guide
- Out: API docs (auto-generated), CLI help (separate task if needed)

**Changes**:
- `docs/validation-triggers.md` — New — Create:
  - Overview of trigger system
  - Configuration reference (all fields, defaults, valid values)
  - Trigger types: epic_completion, session_end, periodic
  - Failure modes: abort, continue, remediate
  - Command list structure with base pool references
  - Migration guide from old config format
  - Example configurations for common use cases:
    - Fast feedback at epic completion
    - Comprehensive validation at session end
    - Periodic lint checks during long sessions
  - Troubleshooting common errors

**Acceptance Criteria**:
- All trigger types documented with examples
- All failure modes explained with behavior
- Migration guide covers both breaking changes
- Example configs are copy-pasteable and valid

**Verification**:
- Manual: Review docs for completeness
- Example configs can be parsed without errors

**Notes for Agent**:
- Reference the spec for exact error messages
- Include the migration URL in appropriate places
- Link to this doc from project-config.md

---

## Dependencies Graph

```
T001 (Foundation)
  │
  └──► T002 (Config skeleton + integration test)
         │
         └──► T003 (Config parsing)
                │
                └──► T005 (Merger)
                       │
                       └──► T004 (Migration)
                              │
                              └──► T006 (RunCoordinator skeleton + integration test)
                                     │
                                     └──► T007 (Command execution)
                                            │
                                            └──► T008 (Failure modes)
                                                   │
                                                   ├──► T009 (Orchestrator skeleton + integration test)
                                                   │      │
                                                   │      └──► T010 (epic_completion)
                                                   │             │
                                                   │             └──► T011 (periodic)
                                                   │                    │
                                                   │                    └──► T012 (session_end)
                                                   │                           │
                                                   │                           └──► T014 (Docs)
                                                   │
                                                   └──► T013 (Events)
```

**Dependency notation**: Arrow points FROM blocker TO dependent (T001 → T002 means T001 blocks T002)

**Note**: All parallelism removed due to file overlaps:
- T004/T005: both write to tests/unit/domain/test_validation_config.py
- T011/T012: both write to src/orchestration/orchestrator.py
- T010/T011/T012: all write to tests/unit/pipeline/test_trigger_execution.py
- T007/T008/T013: all write to src/pipeline/run_coordinator.py

## AC Coverage

| Spec AC | Task | Coverage |
|---------|------|----------|
| R1: Hard migration error | T004 | Primary |
| R2: Supported triggers | T009-T012 | T010 (epic), T011 (periodic), T012 (session_end) |
| R3: Command list selection | T003, T007 | T003 (parsing), T007 (resolution) |
| R4: Fail-fast execution | T007 | Primary |
| R5: Failure modes | T008 | Primary |
| R6: Migration/presets | T004, T005 | T004 (migration), T005 (merge) |

## Sizing Summary

| Task | Files | Subsystems | ACs | Status |
|------|-------|------------|-----|--------|
| T001 | 1 | 1 | 2 | OK |
| T002 | 2 | 2 | 2 | OK |
| T003 | 2 | 1 | 3 | OK |
| T004 | 2 | 1 | 3 | OK |
| T005 | 2 | 1 | 2 | OK |
| T006 | 2 | 2 | 2 | OK |
| T007 | 2 | 1 | 3 | OK |
| T008 | 2 | 1 | 3 | OK |
| T009 | 2 | 2 | 2 | OK |
| T010 | 2 | 1 | 3 | OK |
| T011 | 2 | 1 | 3 | OK |
| T012 | 2 | 1 | 3 | OK |
| T013 | 3 | 3 | 2 | OK (at limit) |
| T014 | 1 | 1 | 1 | OK |
