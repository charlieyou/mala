# Implementation Plan: Per-Trigger Validation Configuration

**Date**: 2026-01-07
**Spec**: `/home/cyou/mala/plans/2026-01-07-epic-validation-trigger-spec.md`
**Status**: Ready (synthesized from 3 generator drafts)

## Context & Goals

Implement per-trigger validation configuration system that allows users to specify exactly which validation commands run at specific trigger points (epic_completion, session_end, periodic). This enables faster feedback loops and cost-effective validation strategies.

**Key Features**:
- Configure different validation commands for different triggers (epic_completion, session_end, periodic)
- Per-trigger failure modes: abort, remediate, continue
- Command list structure with base pool + per-trigger overrides
- BREAKING CHANGES: Repurpose `global_validation_commands`, deprecate `validate_every`

## Scope & Non-Goals

**In Scope**:
- Three trigger types: epic_completion, session_end, periodic
- Per-trigger command configuration via command list structure
- Failure modes: abort, remediate (with retry loop), continue
- Migration validation with hard startup errors for old configs
- Blocking validation execution to prevent workspace conflicts
- Instrumentation events for all trigger lifecycle stages

**Out of Scope (Non-Goals)**:
- issue_completion trigger (redundant with per-session validation)
- Parallel validation execution (always sequential)
- Manual user intervention during validation failures
- Per-session commands in trigger configuration
- Trigger definitions in presets (presets only provide base command pool)

## Assumptions & Constraints

- Config represented as frozen dataclasses, loaded/validated in `config_loader.py`
- Presets merged with project config via `config_merger.py`; migration validation runs on merged effective config
- Validation execution and fixer remediation centralized in `RunCoordinator` with existing retry loop
- Orchestrator blocks new issue assignments during validation, but **active agents continue running** (intentional tradeoff - see Workspace Safety section below)
- `non_epic_completed_count` and trigger queue are in-memory only (lost on crash/restart)

### Implementation Constraints

- Validation execution must block next issue assignment (workspace safety - see below)
- Remediation uses existing `RunCoordinator._run_fixer_agent()` mechanism
- Event emission via MalaEventSink protocol
- Extend RunCoordinator for trigger queue and execution (no new validation service)
- No backward-compatibility shims or re-export modules; update imports directly
- Keep validation executions strictly sequential (single FIFO queue + single runner)
- Do not change signatures of existing validation events; only add new trigger-specific events

### Workspace Safety Semantics

**Blocking behavior**: When a trigger validation starts, the orchestrator **blocks new issue assignments** but **does NOT pause/drain active agents**. This is an intentional design tradeoff:

- **What is blocked**: No new issues are assigned to agents during validation. The orchestrator waits at the issue assignment point.
- **What continues**: Agents that already have issues in progress continue working. They may commit code, modify files, etc.

**Why this tradeoff?**
1. **Avoid killing in-progress work**: Pausing/draining active agents would interrupt possibly hours of work, potentially losing progress.
2. **Simplicity**: Adding workspace locks or agent coordination is complex and error-prone.
3. **Acceptable inconsistency**: Validation results may include changes from in-flight agents. This is similar to running `git status` while you're editing files - you accept point-in-time semantics.

**Implications for validation results**:
- Validation commands operate on the workspace as-is at execution time
- Results reflect a snapshot that may include partial work from active agents
- For `failure_mode=remediate`, the fixer agent runs concurrently with active agents (same as existing fixer behavior)
- False positives/negatives are possible if active agents modify files between trigger and validation completion

**Multi-agent session guidance**:
- For deterministic validation results, run with `max_agents: 1`
- For multi-agent sessions, treat trigger validations as "best effort" checkpoints
- Critical validations (e.g., before merge) should use `session_end` trigger which fires after all agents complete

**Test implications**:
- Tests should not assume quiescent workspace state during validation
- Integration tests with `max_agents > 1` should verify validation completes (correctness), not specific validation outcomes (determinism)

### Testing Constraints

- Must verify migration hard-error behavior
- Must test all three triggers with all three failure modes
- Include meaningful unit tests for config parsing/validation/merging
- Include integration tests for orchestrator/run coordinator wiring without executing real commands
- Include E2E coverage with `--dry-run` mode (see Dry-Run Semantics below)
- Ensure regression coverage for existing non-trigger validation flows

### Dry-Run Semantics

The `--dry-run` flag enables fast E2E testing without executing shell commands. To prevent semantic drift between dry-run and real execution:

**What dry-run DOES**:
- Queues triggers normally (same as real execution)
- Validates all command refs exist in base pool (catches config errors)
- Resolves command overrides (command string, timeout)
- Emits all lifecycle events: `on_trigger_validation_queued`, `on_trigger_validation_started`, etc.
- Emits per-command events: `on_trigger_command_started`, `on_trigger_command_completed`
- Respects `failure_mode` logic (abort/continue/remediate paths)

**What dry-run SKIPS**:
- Actual shell command execution (no subprocess spawn)
- Real validation output/logs

**Synthetic results in dry-run**:
- All commands treated as **passed** by default (no failures to simulate)
- `duration_seconds` set to 0.0 for all commands
- Remediation path: if `failure_mode=remediate`, dry-run skips fixer agent spawn but still emits `on_trigger_remediation_started` event (then marks as passed)

**Config validation in dry-run**:
- Missing `ref` in base pool → **FAIL** (same as real execution)
- Invalid enum values → **FAIL** (same as real execution)
- Required field missing → **FAIL** (same as real execution)

This ensures E2E tests catch config errors while allowing fast verification of trigger lifecycle and event emission.

### Breaking Changes

- **BREAKING #1**: `global_validation_commands` repurposed to base command pool (migration check on effective merged config)
- **BREAKING #2**: `validate_every` deprecated (hard startup error with migration guide)

## Prerequisites

- [x] Understanding of existing validation config system (`config.py`, `config_merger.py`)
- [x] Understanding of existing fixer agent mechanism (`RunCoordinator._run_fixer_agent()`)
- [x] Understanding of MalaEventSink protocol extension patterns
- [ ] Confirm precise YAML schema for triggers (field names/enums for `epic_depth`, `fire_on`, `interval`, command `ref/override`) from spec
- [ ] Identify CLI entrypoint/arg parsing location to add `--dry-run` plumbing
- [ ] Inventory all `MalaEventSink` implementers so new protocol methods can be added consistently
- [ ] Documentation URL for migration guide (https://docs.mala.ai/migration/validation-triggers)

## High-Level Approach

The implementation centers around extending `RunCoordinator` to orchestrate trigger-based validations through a simple FIFO queue. Configuration changes introduce a new `validation_triggers` structure that defines which commands run at specific trigger points (epic_completion, session_end, periodic), with each trigger specifying its failure mode (abort/continue/remediate). The existing `global_validation_commands` field is repurposed as a base command pool that triggers reference via `commands: [{ref: name, ...}]` syntax, enabling command reuse across triggers.

Migration validation runs in `config_loader.py` on the effective merged config (preset + project) and fails fast with actionable errors for deprecated patterns: (1) non-empty `global_validation_commands` without `validation_triggers` defined, (2) presence of `validate_every` field. This ensures users cannot inadvertently use the old behavior and provides clear migration guidance.

Trigger execution follows a strictly sequential model: triggers fire from orchestration points (epic verification, main loop counter, session end), queue validation jobs in RunCoordinator, and block the orchestrator before assigning the next issue. This blocking behavior prevents workspace conflicts during validation. The remediation failure mode reuses the existing `_run_fixer_agent()` retry loop mechanism. Comprehensive instrumentation via new MalaEventSink events provides observability for trigger lifecycle, command execution, and remediation attempts.

**Implementation Phases**:
1. **Config Schema & Parsing**: Add `ValidationTriggersConfig` dataclasses with nested trigger configs, enums for trigger types/failure modes, and `TriggerCommandRef` for pool references
2. **Migration Validation**: Implement hard startup errors in `config_loader.py` for deprecated patterns with migration guide URLs
3. **Trigger Execution Engine**: Extend `RunCoordinator` with trigger queue, command resolution (base pool + overrides), sequential execution with fail-fast, and failure mode handling
4. **Integration Points**: Hook epic_completion in `EpicVerificationCoordinator`, periodic counter and session_end in `orchestrator.py` main loop
5. **Instrumentation**: Extend `MalaEventSink` protocol with 10 new trigger-specific events for lifecycle tracking

## Technical Design

### Architecture

```
┌─────────────────────┐
│   mala.yaml         │
│ validation_triggers │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  config_loader.py   │ ◄─── Migration validation (fail-fast)
│  + config_merger.py │      Field-level deep merge for base pool
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  RunCoordinator     │ ◄─── Trigger queue + execution engine
│  (extended)         │      Reuses _run_fixer_agent for remediate mode
└──────────┬──────────┘
           │
    ┌──────┴──────┬─────────────────┐
    │             │                 │
    ▼             ▼                 ▼
┌────────────┐  ┌───────────┐  ┌─────────────────┐
│ epic       │  │ periodic  │  │ session_end     │
│ verify     │  │ counter   │  │ hook            │
│ hook       │  │ (main     │  │ (orchestrator   │
│            │  │  loop)    │  │  line ~1135)    │
└────────────┘  └───────────┘  └─────────────────┘
```

**Key components**:
- **Config layer** (`src/domain/validation/config.py`): New dataclasses for `validation_triggers` with nested trigger configs
- **Config merger** (`src/domain/validation/config_merger.py`): Field-level deep merge for `global_validation_commands` base pool
- **Migration validator** (`src/domain/validation/config_loader.py`): Check old configs and fail with migration errors on effective merged config
- **Trigger engine** (`src/pipeline/run_coordinator.py`): FIFO queue and sequential execution of validation triggers
- **Event emission** (`src/core/protocols.py`): Extend MalaEventSink with new trigger events

**Data flow**:
1. **Trigger fires**: Epic verified, issue count reaches interval, or session ends
2. **Queue validation**: Trigger context created with type and metadata, queued in `RunCoordinator.trigger_queue`
3. **Block orchestrator**: Main loop waits before assigning next issue (prevents workspace conflicts)
4. **Execute sequentially**: Commands resolve from base pool + overrides, execute with fail-fast (first failure stops remaining commands)
5. **Handle failure**: Apply failure_mode → abort (stop run), continue (log only), remediate (spawn fixer, retry up to max_retries)
6. **Emit events**: Lifecycle events at each stage (queued, started, command_started, command_completed, passed/failed)

**Fail-fast + failure_mode interaction**:
- **Fail-fast** determines *which commands execute within a trigger*: on first command failure, remaining commands in that trigger are skipped immediately.
- **failure_mode** determines *what happens to the run after fail-fast*: whether to abort the entire run, continue to next issue, or attempt remediation.
- In other words: fail-fast controls intra-trigger behavior; failure_mode controls post-trigger behavior.
- Example: Commands [A, B, C] with `failure_mode=continue` → if B fails, C is skipped (fail-fast), failure logged, run continues to next issue (failure_mode=continue).

### Data Model

**New Config Structures** (`src/domain/validation/config.py`):
```python
from enum import Enum
from dataclasses import dataclass, field

class TriggerType(Enum):
    """Supported validation trigger types."""
    EPIC_COMPLETION = "epic_completion"
    SESSION_END = "session_end"
    PERIODIC = "periodic"

class FailureMode(Enum):
    """How to handle validation failures."""
    ABORT = "abort"         # Stop run immediately
    CONTINUE = "continue"   # Log failure, continue run
    REMEDIATE = "remediate" # Spawn fixer, retry up to max_retries

class EpicDepth(Enum):
    """Which epics trigger epic_completion validation."""
    TOP_LEVEL = "top_level"  # Only epics with no epic parent
    ALL = "all"              # Any epic closure

class FireOn(Enum):
    """When epic_completion trigger fires."""
    SUCCESS = "success"  # Only on successful verification
    FAILURE = "failure"  # Only on failed verification
    BOTH = "both"        # Always fire

@dataclass(frozen=True)
class TriggerCommandRef:
    """Reference to command from base pool with optional overrides.

    Attributes:
        ref: Name of command in base pool (global_validation_commands)
        command: Optional override for the command string
        timeout: Optional override for timeout (seconds)
    """
    ref: str
    command: str | None = None
    timeout: int | None = None

@dataclass(frozen=True)
class BaseTriggerConfig:
    """Base configuration common to all triggers.

    Attributes:
        failure_mode: How to handle validation failures (REQUIRED)
        max_retries: Fixer attempts when failure_mode=remediate (REQUIRED for remediate)
        commands: Ordered list of command references to execute
    """
    failure_mode: FailureMode
    commands: tuple[TriggerCommandRef, ...] = field(default_factory=tuple)
    max_retries: int | None = None

@dataclass(frozen=True)
class EpicCompletionTriggerConfig(BaseTriggerConfig):
    """Configuration for epic_completion trigger.

    Attributes:
        epic_depth: Which epics trigger validation (top_level or all)
        fire_on: When to fire (success, failure, both)
    """
    epic_depth: EpicDepth = EpicDepth.TOP_LEVEL
    fire_on: FireOn = FireOn.SUCCESS

@dataclass(frozen=True)
class SessionEndTriggerConfig(BaseTriggerConfig):
    """Configuration for session_end trigger.

    Only fires when success_count > 0 and abort_run=False.
    """
    pass

@dataclass(frozen=True)
class PeriodicTriggerConfig(BaseTriggerConfig):
    """Configuration for periodic trigger.

    Attributes:
        interval: Fire every N non-epic issue completions (REQUIRED)
    """
    interval: int = 5  # Default, validated at parse time

@dataclass(frozen=True)
class ValidationTriggersConfig:
    """Per-trigger validation configuration.

    Attributes:
        epic_completion: Optional epic completion trigger config
        session_end: Optional session end trigger config
        periodic: Optional periodic trigger config
    """
    epic_completion: EpicCompletionTriggerConfig | None = None
    session_end: SessionEndTriggerConfig | None = None
    periodic: PeriodicTriggerConfig | None = None

    def is_empty(self) -> bool:
        """Check if all triggers are unconfigured."""
        return all([
            self.epic_completion is None,
            self.session_end is None,
            self.periodic is None,
        ])
```

**Trigger Queue** (in `RunCoordinator`):
- Simple FIFO queue using Python `list` (append + sequential processing)
- Structure: `list[tuple[TriggerType, dict[str, Any]]]` where dict contains trigger context (epic_id, depth, verification_result, etc.)

**State Tracking** (in `orchestrator.py`):
- `non_epic_completed_count: int` — Counter for periodic trigger (in-memory only, resets on restart)

### Abort Propagation & Queue Clearing

**Abort flag ownership**: The `abort_run` flag is owned by orchestrator's `IssueCoordinator` (existing pattern). RunCoordinator reads this flag via callback or shared reference.

**Abort propagation path**:
1. **Trigger**: `failure_mode=abort` result OR SIGINT signal
2. **RunCoordinator**: Sets result status to `aborted`, calls `clear_trigger_queue(reason)` to emit skipped events for queued validations
3. **Orchestrator**: Checks `run_trigger_validation()` result; if `aborted`, sets `IssueCoordinator.abort_run = True` with reason
4. **Main loop**: Existing logic respects `abort_run` flag, stops assigning issues, skips `session_end` trigger

**Queue clearing semantics**:
- On abort: Clear queue immediately, emit `on_trigger_validation_skipped(trigger_type, reason='run_aborted')` for each queued item
- On SIGINT: Same as abort, but reason='sigint'; also SIGTERM in-progress subprocess
- On normal completion: Queue is empty after all validations execute

**SIGINT handling** (in RunCoordinator):
1. Register signal handler when validation starts
2. On SIGINT: send SIGTERM to subprocess, set internal `_interrupted` flag
3. Check `_interrupted` after each command; if set, abort remaining commands and clear queue
4. Restore original signal handler when validation completes

### API/Interface Design

**RunCoordinator Interface Extensions**:
```python
# Add to src/pipeline/run_coordinator.py

def queue_trigger_validation(
    self, trigger_type: TriggerType, context: dict[str, Any]
) -> None:
    """Queue a trigger validation for later execution.

    Args:
        trigger_type: Type of trigger firing
        context: Trigger metadata (epic_id, depth, verification_result, etc.)
    """
    ...

def run_trigger_validation(self, dry_run: bool = False) -> TriggerValidationResult:
    """Execute all queued trigger validations sequentially (blocking).

    Args:
        dry_run: If True, skip command execution but still validate refs and emit events

    Returns:
        TriggerValidationResult with status (passed/failed/aborted) and details
    """
    ...

def clear_trigger_queue(self, reason: str) -> None:
    """Clear all queued validations and emit skipped events.

    Args:
        reason: Why validations were skipped (e.g., 'run_aborted', 'sigint')
    """
    ...
```

**MalaEventSink Extensions** (`src/core/protocols.py`):
```python
# New event methods to add to MalaEventSink protocol

def on_trigger_validation_queued(
    self, trigger_type: str, trigger_context: str
) -> None:
    """Called when a trigger fires and queues validation."""
    ...

def on_trigger_validation_started(
    self, trigger_type: str, commands: list[str]
) -> None:
    """Called when queued validation begins executing."""
    ...

def on_trigger_command_started(
    self, trigger_type: str, command_ref: str, index: int
) -> None:
    """Called when individual command within trigger starts."""
    ...

def on_trigger_command_completed(
    self, trigger_type: str, command_ref: str, index: int,
    passed: bool, duration_seconds: float
) -> None:
    """Called when individual command completes."""
    ...

def on_trigger_validation_passed(
    self, trigger_type: str, duration_seconds: float
) -> None:
    """Called when all commands in trigger pass."""
    ...

def on_trigger_validation_failed(
    self, trigger_type: str, failed_command: str, failure_mode: str
) -> None:
    """Called when any command fails."""
    ...

def on_trigger_validation_skipped(
    self, trigger_type: str, reason: str
) -> None:
    """Called when trigger validation is skipped (no_commands, run_aborted)."""
    ...

def on_trigger_remediation_started(
    self, trigger_type: str, attempt: int, max_retries: int
) -> None:
    """Called when fixer agent spawns for remediation."""
    ...

def on_trigger_remediation_succeeded(
    self, trigger_type: str, attempt: int
) -> None:
    """Called when remediation fixes the issue."""
    ...

def on_trigger_remediation_exhausted(
    self, trigger_type: str, attempts: int
) -> None:
    """Called when all retry attempts fail."""
    ...
```

**Config Schema** (mala.yaml):
```yaml
# Base command pool (repurposed from legacy global_validation_commands)
global_validation_commands:
  pytest:
    command: "uv run pytest tests/"
    timeout: 300
  ruff:
    command: "uvx ruff check ."
    timeout: 60

# New trigger configuration
validation_triggers:
  # Epic completion: fast feedback on epic-level changes
  epic_completion:
    epic_depth: top_level  # top_level | all
    fire_on: success       # success | failure | both
    failure_mode: remediate
    max_retries: 3
    commands:
      - ref: ruff  # Use base pool command as-is
      - ref: pytest
        timeout: 120  # Override timeout for this trigger

  # Session end: comprehensive validation before marking session complete
  session_end:
    failure_mode: abort
    commands:
      - ref: ruff
      - ref: pytest

  # Periodic: regular validation during long sessions
  periodic:
    interval: 5  # Every 5 non-epic issue completions
    failure_mode: continue
    commands:
      - ref: ruff
```

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/domain/validation/config.py` | Exists | Add TriggerType, FailureMode, EpicDepth, FireOn enums; TriggerCommandRef, BaseTriggerConfig, EpicCompletionTriggerConfig, SessionEndTriggerConfig, PeriodicTriggerConfig, ValidationTriggersConfig dataclasses; update ValidationConfig with validation_triggers field |
| `src/domain/validation/config_loader.py` | Exists | Add `_validate_migration()` function for hard errors on deprecated patterns; update `_ALLOWED_TOP_LEVEL_FIELDS` to include `validation_triggers` |
| `src/domain/validation/config_merger.py` | Exists | Add field-level deep merge for `global_validation_commands` base pool; add merge logic for `validation_triggers` (project-only, no preset inheritance per spec) |
| `src/pipeline/run_coordinator.py` | Exists | Add trigger queue (`list[tuple[TriggerType, dict]]`), `queue_trigger_validation()`, `run_trigger_validation()`, `_resolve_trigger_commands()`, `_execute_trigger_commands()`, remediation retry loop integration |
| `src/pipeline/epic_verification_coordinator.py` | Exists | Add callback for queueing epic_completion trigger after verification completes in `check_epic_closure()` |
| `src/orchestration/orchestrator.py` | Exists | Add `non_epic_completed_count` state; hook periodic trigger in main loop after per-session validation; modify session_end (line ~1135) to use trigger system; add blocking wait for trigger validation |
| `src/core/protocols.py` | Exists | Add 10 new event methods to MalaEventSink protocol (on_trigger_validation_queued, on_trigger_validation_started, etc.) |
| `src/infra/io/log_output/console.py` | Exists | Add handlers for new trigger events (console output logging) |
| **New:** `docs/validation-triggers.md` | **New** | User documentation for trigger configuration setup and examples |
| **New:** `tests/unit/domain/validation/test_trigger_config.py` | **New** | Unit tests for config parsing, validation, enum handling, dataclass constraints |
| **New:** `tests/unit/domain/validation/test_trigger_merger.py` | **New** | Unit tests for field-level deep merge of command pool and trigger configs |
| **New:** `tests/unit/domain/validation/test_migration_validation.py` | **New** | Unit tests for migration errors (validate_every, global_validation_commands without triggers) |
| **New:** `tests/unit/pipeline/test_trigger_execution.py` | **New** | Unit tests for trigger queue, command resolution, override application, fail-fast behavior |
| **New:** `tests/integration/test_validation_triggers.py` | **New** | Integration tests for trigger execution with mocked commands, all failure modes, orchestrator wiring |
| **New:** `tests/e2e/test_trigger_execution.py` | **New** | E2E tests with dry-run mode for full trigger lifecycle, event emission verification |

### Integration Points

**Epic Verification Integration** (`src/pipeline/epic_verification_coordinator.py`):
- **Location**: `check_epic_closure()` method after verification completes
- **Logic**: After epic verification result determined (success/failure), check trigger config:
  1. If `validation_triggers.epic_completion` is None, skip
  2. Check `epic_depth` filter: if `top_level`, verify epic has no epic parent
  3. Check `fire_on` filter: if `success`, only fire on verification pass; if `failure`, only on fail; if `both`, always fire
  4. If filters pass, call `run_coordinator.queue_trigger_validation(TriggerType.EPIC_COMPLETION, context={'epic_id': epic.id, 'depth': depth, 'verification_result': result})`

**Orchestrator Integration** (`src/orchestration/orchestrator.py`):
- **session_end trigger**: Line ~1135 (existing validation check location)
  - Replace existing validation logic with trigger-based approach
  - Fire when `success_count > 0` and `abort_run=False`
  - Call `run_coordinator.queue_trigger_validation(TriggerType.SESSION_END, context={'success_count': success_count})`

- **periodic trigger**: Main orchestrator loop after per-session validation
  - Add `non_epic_completed_count` state variable (initialized to 0)
  - After each non-epic issue completion (any issue where `is_epic=False`), increment counter
  - If `non_epic_completed_count % interval == 0`, call `run_coordinator.queue_trigger_validation(TriggerType.PERIODIC, context={'count': non_epic_completed_count})`

- **Blocking execution**: After queueing any trigger, call `run_coordinator.run_trigger_validation()` before assigning next issue

**Remediation Integration** (`src/pipeline/run_coordinator.py`):
- **Reuse existing mechanism**: `_run_fixer_agent()` with retry loop
- **Integration approach**:
  1. When command fails and `failure_mode == FailureMode.REMEDIATE`:
  2. For attempt in range(max_retries):
     - Emit `on_trigger_remediation_started(trigger_type, attempt, max_retries)`
     - Call `_run_fixer_agent()` with failure context
     - If fixer succeeds, re-run failed command
     - If command passes, emit `on_trigger_remediation_succeeded()`, continue to next command
     - If command still fails, continue retry loop
  3. If all retries exhausted, emit `on_trigger_remediation_exhausted()`, **abort run** (terminal behavior per spec)

## Risks, Edge Cases & Breaking Changes

### Breaking Changes & Compatibility

**Potential Breaking Changes**:
1. **`global_validation_commands` repurpose (BREAKING #1)**: Now defines base command pool, not auto-run commands. Migration check applies to **effective merged config** (preset + project). Users with preset-only configs (e.g., `preset: python-uv` with non-empty `global_validation_commands`) without project `validation_triggers` get hard startup error.
   - **Error message**: "Config error: validation_triggers required when global_validation_commands is defined. The global_validation_commands field now serves as a base command pool referenced by triggers. See migration guide at https://docs.mala.ai/migration/validation-triggers"
   - **Mitigation**: Include example snippet in error message showing minimal `validation_triggers.session_end` config

2. **`validate_every` deprecation (BREAKING #2)**: Hard startup error with migration guide.
   - **Error message**: "Config error: validate_every is deprecated. Use validation_triggers.periodic with interval field instead. See migration guide at https://docs.mala.ai/migration/validation-triggers"
   - **Mitigation**: Provide 1-line conversion example in error (e.g., "validate_every: 5 → validation_triggers.periodic.interval: 5")

**Valid non-breaking configs**:
- Empty `validation_triggers: {}` (explicit opt-out of all triggers)
- Empty `global_validation_commands: {}` without `validation_triggers` (empty pool, no error)
- Project with `validation_triggers` defined (opt-in to new system)

### Edge Cases & Failure Modes

From spec requirements:
- **Back-to-back epic completions while periodic validations queued** → Both queue sequentially, no deduplication, FIFO order preserved
- **Abort signal during validation** → In-progress subprocess gets SIGTERM, queued validations discarded with `on_trigger_validation_skipped(reason='run_aborted')` events
- **Periodic interval never reached** (fewer issues than interval) → Periodic validation never fires (valid scenario, no error)
- **Empty command list** (`commands: []`) → Trigger fires but no commands execute, emit `on_trigger_validation_skipped(reason='no_commands')`, result=passed
- **Leaf epic completes with `epic_depth: top_level`** → Fires if leaf epic has no epic parent (even if it's the only epic in the tree)
- **Session ends with `success_count > 0` but `abort_run=True`** → session_end trigger skipped (abort takes precedence)
- **SIGINT during remediation** → Current fixer subprocess interrupted (SIGTERM), blocking wait released, run aborts, session_end trigger skipped
- **Fixer succeeds but trigger re-run fails on different command** → Counts as failed attempt, continues retry loop
- **Command times out** → Treated as failure, triggers fail-fast behavior (remaining commands in trigger skipped)
- **Invalid trigger ref** (command not in base pool) → Config loading fails with clear error pointing to trigger and missing ref

### Risks

**Implementation risks**:
- **Workspace conflicts during remediation**: Fixer agent modifies workspace while trigger validation owns it. Mitigated by existing `_run_fixer_agent()` mechanism which already handles workspace safety.
- **Queue growth**: FIFO queue can grow unbounded if triggers fire faster than execution. Low risk given sequential blocking execution, but could add optional queue depth logging.
- **Migration validation false positives**: Preset-only users might hit migration errors unexpectedly. Mitigated by checking effective merged config and clear error messages.
- **Event implementer burden**: All `MalaEventSink` implementers must add 10 new methods. Mitigated by providing default no-op implementations in base protocol.

## Testing & Validation Strategy

Testing follows the standard pyramid with focused coverage at each level. All three drafts agree on dry-run mode for E2E tests to enable fast verification without executing real commands.

### Unit Tests

**Config Parsing & Validation** (`tests/unit/domain/validation/test_trigger_config.py`):
- `ValidationTriggersConfig` parsing with all trigger types and fields
- Validation of required fields: `failure_mode` on all triggers, `max_retries` when `failure_mode=remediate`, `interval` for periodic trigger
- Enum parsing: `TriggerType`, `FailureMode`, `EpicDepth`, `FireOn` with rejection of invalid values
- `TriggerCommandRef` parsing with and without overrides (command, timeout)
- Nested dataclass immutability (frozen validation)

**Migration Validation** (`tests/unit/domain/validation/test_migration_validation.py`):
- Hard error when `global_validation_commands` is non-empty without `validation_triggers` defined
- Hard error when `validate_every` field is present
- Valid cases: empty `validation_triggers: {}`, empty `global_validation_commands: {}` without triggers
- Error message content: includes migration guide URL and brief example snippet
- Validation runs on effective merged config (preset + project combination)

**Config Merging** (`tests/unit/domain/validation/test_trigger_merger.py`):
- Field-level deep merge for `global_validation_commands` base pool
- Project commands override preset commands by matching name/key
- Deterministic precedence: project config overrides preset entries
- Merge handles both empty and populated configs correctly

**Trigger Execution** (`tests/unit/pipeline/test_trigger_execution.py`):
- Command resolution from base pool via `ref` field
- Override application (command string override, timeout override)
- Fail-fast execution: first command failure stops remaining commands in trigger
- Timeout handling: command timeout treated as failure, triggers fail-fast
- Empty command list: trigger fires with `result=passed`, `reason='no_commands'`
- Invalid ref detection: missing command in base pool raises config error

### Integration Tests

**Trigger Execution Flow** (`tests/integration/test_validation_triggers.py`):
- Epic completion trigger fires after `check_epic_closure()` completes
  - Respects `epic_depth` filter (top_level vs all)
  - Respects `fire_on` filter (success, failure, both)
- Periodic trigger fires at correct intervals (based on `non_epic_completed_count`)
- Session end trigger fires with correct skip conditions (`success_count > 0`, `abort_run=False`)
- Orchestrator blocking: main loop waits for `run_trigger_validation()` before assigning next issue
- FIFO queue ordering preserved when multiple triggers fire in sequence
- Abort behavior: SIGINT cancels queued validations with `on_trigger_validation_skipped(reason='run_aborted')`

**Remediation Integration** (mocked fixer):
- Remediation retry loop invokes `_run_fixer_agent()` up to `max_retries` times
- Event emission: `on_trigger_remediation_started`, `on_trigger_remediation_succeeded`, `on_trigger_remediation_exhausted`
- Fixer success followed by command re-run (verify fix worked)
- Retry exhaustion behavior: falls back to abort mode after `max_retries` failures

### E2E Tests

**Full Trigger Lifecycle** (`tests/e2e/test_trigger_execution.py`):
- Full session with all three triggers configured using `--dry-run` mode
- Verify trigger queueing and event emission without executing shell commands
- Event sequence verification: `queued` → `started` → `command_started` → `command_completed` → `passed/failed`
- All failure modes exercised: abort (stops run), continue (logs, proceeds), remediate (spawns fixer path)

**Migration Error Scenarios**:
- Real config loading with deprecated `validate_every` field
- Real config loading with `global_validation_commands` but no `validation_triggers`
- Startup failure with actionable error message verification

### Regression Coverage

- Existing global validation + fixer behavior remains correct (if any non-trigger validation paths remain)
- Orchestrator `success_count` and `abort_run` logic consistent with new trigger behavior
- `MalaEventSink` existing events unchanged; only new trigger-specific events added

### Manual Verification

- Run sample config with each trigger enabled in local environment
- Verify validations block assignment of next issue
- Verify `remediate` invokes fixer attempts and retries commands
- Verify logs/events show trigger queueing and command outcomes

### Monitoring & Observability

- New events include `trigger_type` for correlation
- Command duration tracked in `on_trigger_command_completed(duration_seconds)`
- Optional: queue depth logging (nice-to-have, not blocking)

### Acceptance Criteria Coverage

| Spec AC | Implementation Approach | Test Coverage |
|---------|------------------------|---------------|
| Fast feedback queueing (epic_completion queued within 10s of epic closure) | Hook in `EpicVerificationCoordinator.check_epic_closure()` immediately after verification result; call `queue_trigger_validation()` synchronously | Integration test timing assertion on `on_trigger_validation_queued` event timestamp vs epic completion time |
| Correct trigger execution (periodic interval=5 fires every 5 non-epic completions) | `non_epic_completed_count` counter in orchestrator; modulo check `count % interval == 0` | Unit tests for counter logic; integration test with 5+ issues verifying exactly N/5 trigger fires |
| Migration clarity (hard error with guide URL for deprecated config) | `_validate_migration()` in `config_loader.py` checks merged config for `validate_every` presence and `global_validation_commands` without `validation_triggers` | Unit tests verify error message content includes URL and example snippet; E2E tests verify startup failure |
| Remediation loop correctness (max_retries behavior) | For-loop in `run_trigger_validation()` from 0 to `max_retries-1`; emit remediation events; call `_run_fixer_agent()` then re-run failed command | Unit tests for retry loop bounds; integration tests with mock fixer verify attempt count and event sequence |
| Blocking execution (validation completes before next issue assignment) | `orchestrator.py` calls `run_coordinator.run_trigger_validation()` synchronously before `_assign_next_issue()` | Integration test verifies no issue assignment events during validation execution |
| All failure modes work correctly (abort/continue/remediate) | Switch on `FailureMode` enum in `run_trigger_validation()`: abort sets `abort_run=True`, continue logs and returns, remediate enters retry loop | Unit tests for each mode branch; E2E dry-run tests verify correct state transitions per mode |

## Open Questions

**Resolved during synthesis** (all three drafts agree):

1. ~~Where should the trigger queue/engine live?~~ → **Extend `RunCoordinator`** (centralizes all validation orchestration logic, reuses existing fixer agent mechanism)

2. ~~Where should migration validator live?~~ → **In `config_loader.py`** (fails fast at startup, runs on effective merged config)

3. ~~How to persist `non_epic_completed_count` across crashes?~~ → **In-memory only** (spec explicitly allows loss on restart; no persistence needed)

4. ~~Event emission strategy - batch or real-time?~~ → **Real-time emission** (emit events synchronously at each lifecycle stage for immediate observability)

5. ~~Test coverage requirements?~~ → **Comprehensive coverage** across all three tiers (unit/integration/E2E) with dry-run mode for fast E2E tests

6. ~~Migration guide documentation?~~ → **URL in error message** (https://docs.mala.ai/migration/validation-triggers) plus inline example snippet in error output

**Remaining open questions** (low priority, can decide during implementation):

1. **Queue depth logging** — Should we add logging/metrics for trigger queue depth? All drafts mention this as optional/nice-to-have. Can add as follow-up if observability needs arise.

2. **Trigger execution latency metrics** — Should we emit separate latency metrics beyond duration in events? Nice-to-have, not blocking MVP. Events already include `duration_seconds`.

3. **Console output handler location** — Gemini draft mentions `src/infra/telemetry.py`, Claude draft mentions `src/infra/io/log_output/console.py`. Need to verify which file handles console output for MalaEventSink. (Implementation detail, will resolve during implementation.)

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
