# Implementation Plan: Architecture Contracts Fix (Phases 1-2)

## Context & Goals
- **Spec**: `doc/architecture-review.md`
- Fix all 4 broken import-linter contracts identified in the architecture review
- Phases 1-2 focus: Quick Wins + Domain Purity (no CCN optimization or package restructure)
- Clean break approach: no backward-compatibility re-exports

## Scope & Non-Goals

### In Scope
- Fix orchestrator ↔ orchestrator_factory circular dependency
- Fix protocols.py TYPE_CHECKING imports that import-linter flags as violations
- Fix domain layer independence contract (lifecycle ↔ models)
- Fix CLI reaching into infra modules directly
- Fix infra cross-imports (event_sink_console, telemetry)
- Update contract definitions where intent doesn't match implementation
- Update all import paths (clean break, no re-exports)

### Out of Scope (Non-Goals)
- Cyclomatic complexity optimization (run_session CCN=67, _run_refresh CCN=54, etc.)
- Full package restructure (src/domain/, src/infra/, src/orchestration/)
- Quality gate extraction to subdirectory
- Any new features or behavioral changes
- Introducing alternative orchestrators

## Assumptions & Constraints

- The project uses `import-linter` for architectural enforcement
- `src/models.py` is in the domain layer (pyproject.toml line 147)
- Types like `IssueResolution`, `ResolutionOutcome`, `ValidationArtifacts` are defined in `src/models.py`
- Protocol-based DI pattern is already established
- `OrchestratorConfig` and `OrchestratorDependencies` are dataclasses
- import-linter flags TYPE_CHECKING imports as violations (not just runtime imports)

### Implementation Constraints
- Must pass all import-linter contracts after changes
- No new external dependencies
- Extend existing patterns (TYPE_CHECKING, protocols, dataclasses)
- Maintain existing test coverage
- All changes must be incremental and reviewable

### Testing Constraints
- 85% coverage threshold enforced
- Run `uvx --from import-linter lint-imports` to verify contracts
- Run `uvx ruff check .` and `uvx ty check` for linting/type checking
- Run `uv run pytest -m "unit or integration"` for tests

## Prerequisites

- [ ] Verify `import-linter` configuration is present and baseline contracts status
- [ ] Confirm baseline test pass (`uv run pytest`)
- [ ] Identify all test files that import from modules being refactored

## High-Level Approach

1. **Extract orchestrator shared types** to break the circular dependency
2. **Fix protocols by using local Protocol types** - avoid importing concrete types from domain/infra by defining protocol interfaces locally
3. **Fix Contract 2** by changing contract type to match stated intent (domain not depending on orchestration/CLI)
4. **Create CLI support module** for CLI-specific utilities to route through orchestrator layer
5. **Merge event sink files** to eliminate infra same-layer import
6. **Fix telemetry→braintrust imports** by moving BraintrustProvider/BraintrustSpan
7. **Update test imports** and run full verification

## Detailed Plan

### Task 1: Create `src/orchestrator_types.py` for shared types

- **Goal**: Break circular dependency between orchestrator ↔ orchestrator_factory
- **Covers**: Contract "Layered Architecture" - cycle violation
- **Depends on**: None
- **Changes**:
  - **New**: `src/orchestrator_types.py` containing:
    - `OrchestratorConfig` dataclass (move from `orchestrator_factory.py` lines 48-90)
    - `OrchestratorDependencies` dataclass (move from `orchestrator_factory.py` lines 92-100+)
    - `_DerivedConfig` dataclass (if it exists in factory, move it)
    - `DEFAULT_AGENT_TIMEOUT_MINUTES` constant (move from `orchestrator_factory.py` line 45)
  - Update `src/orchestrator_factory.py`:
    - Import from `orchestrator_types` instead of defining locally
    - Remove moved definitions
  - Update `src/orchestrator.py` line 76:
    - Change `from .orchestrator_factory import DEFAULT_AGENT_TIMEOUT_MINUTES` to `from .orchestrator_types import DEFAULT_AGENT_TIMEOUT_MINUTES`
    - Update TYPE_CHECKING imports (line 83) for `OrchestratorConfig`, `_DerivedConfig`
  - Update `src/cli.py` lazy imports:
    - Update to get `OrchestratorConfig` from `orchestrator_types`
  - Update `pyproject.toml` line 143:
    - Change: `"src.orchestrator | src.orchestrator_factory"`
    - To: `"src.orchestrator | src.orchestrator_factory | src.orchestrator_types"`
- **Verification**:
  - `uvx --from import-linter lint-imports --contract "Layered Architecture"` shows improvement
  - `uvx ruff check src/orchestrator_types.py src/orchestrator.py src/orchestrator_factory.py`
  - `uvx ty check`
  - `uv run pytest tests/unit/test_orchestrator*.py -v`
- **Rollback**: Delete `src/orchestrator_types.py` and revert imports

### Task 2: Fix protocols by defining local Protocol types

- **Goal**: Resolve import-linter flagging protocols → infra/domain module imports
- **Covers**: Contract "Layered Architecture" - protocols importing from cerberus_review, quality_gate, models, etc.
- **Depends on**: Task 1
- **Design Decision**: Instead of moving protocols to a different layer (which creates new issues with protocols importing from domain types), define local Protocol types in protocols.py that match the shapes needed. This avoids cross-layer imports entirely.
- **Changes**:
  - Update `src/protocols.py`:
    - Remove TYPE_CHECKING imports from domain/infra modules (lines 28-32):
      ```python
      # REMOVE these TYPE_CHECKING imports:
      # from .cerberus_review import ReviewResult
      # from .models import EpicVerdict
      # from .quality_gate import CommitResult, GateResult, ValidationEvidence
      # from .session_log_parser import JsonlEntry
      # from .validation.spec import ValidationSpec
      ```
    - Define local Protocol types for each needed type:
      - `ReviewResultProtocol` - matches `cerberus_review.ReviewResult` shape
      - `EpicVerdictProtocol` - matches `models.EpicVerdict` shape
      - `GateResultProtocol` - matches `quality_gate.GateResult` shape
      - `CommitResultProtocol` - matches `quality_gate.CommitResult` shape
      - `ValidationEvidenceProtocol` - matches `quality_gate.ValidationEvidence` shape
      - `ValidationSpecProtocol` - matches `validation.spec.ValidationSpec` shape
      - `JsonlEntryProtocol` - matches `session_log_parser.JsonlEntry` shape
    - Update method signatures to use new Protocol types instead of concrete types
  - Update consumers to use structural typing (no import changes needed - protocols match shapes)
  - Note: Keep `src.protocols` in infra layer (line 149) - no layer changes needed
- **Verification**:
  - `uvx --from import-linter lint-imports --contract "Layered Architecture"` passes for protocols
  - `uvx ty check` (ensure structural typing works)
  - `uv run pytest tests/ -v -k protocol`
- **Rollback**: Restore original TYPE_CHECKING imports, remove Protocol definitions

### Task 3: Fix Contract 2 by updating contract definition

- **Goal**: Fix "Domain layer independence" contract to match its stated intent
- **Covers**: Contract 2 "Domain layer independence"
- **Depends on**: Task 2
- **Analysis**:
  - Contract 2 comment says: "Core domain modules must not depend on orchestration or CLI"
  - But the actual contract is `type = "independence"` between `lifecycle` and `models`
  - This prevents lifecycle↔models imports, which is NOT the stated intent
  - The stated intent is domain not depending on upper layers (orchestration/CLI)
- **Changes**:
  - Update `pyproject.toml` Contract 2 (lines 157-163):
    - Change contract type from `independence` to `forbidden`
    - Change to match stated intent: domain modules must not import orchestration/CLI
    ```toml
    [[tool.importlinter.contracts]]
    name = "Domain layer independence"
    type = "forbidden"
    source_modules = [
        "src.lifecycle",
        "src.models",
        "src.quality_gate",
        "src.validation",
        "src.prompts",
    ]
    forbidden_modules = [
        "src.cli",
        "src.main",
        "src.orchestrator",
        "src.orchestrator_factory",
        "src.orchestrator_types",
    ]
    ```
  - **No code changes needed** - lifecycle can continue importing from validation.spec
  - Note: `validation.spec` re-exports from `models` which is allowed (same layer)
- **Verification**:
  - `uvx --from import-linter lint-imports --contract "Domain layer independence"` passes
  - Verify lifecycle.py doesn't need changes (current import from validation.spec is fine)
- **Rollback**: Revert pyproject.toml contract definition

### Task 4: Create CLI support module for CLI-specific utilities

- **Goal**: CLI should not directly import from `src.tools.*`, `src.log_output.*`
- **Covers**: Contract "CLI only depends on orchestrator"
- **Depends on**: Task 1
- **Design Decision**: Create `src/cli_support.py` (NOT orchestrator_types.py) to keep orchestration layer clean and avoid polluting it with infrastructure dependencies
- **Changes**:
  - **New**: `src/cli_support.py` containing re-exports:
    - From `src.tools.env`: `USER_CONFIG_DIR`, `SCRIPTS_DIR`, `get_runs_dir`, `load_user_env`
    - From `src.tools.locking`: `get_lock_dir`
    - From `src.log_output.run_metadata`: `get_running_instances`, `get_running_instances_for_dir`
    - From `src.log_output.console`: `Colors`, `log`, `set_verbose`
  - Update `src/cli.py`:
    - Line 18: Change `from .tools.env import USER_CONFIG_DIR, get_runs_dir, load_user_env` to `from .cli_support import USER_CONFIG_DIR, get_runs_dir, load_user_env`
    - Line 106 area: Update `Colors`, `log`, `set_verbose` imports to come from `cli_support`
    - Lines 804-815: Update lazy imports to use `cli_support`:
      - `get_lock_dir` (currently from `src.tools.locking` at line 805)
      - `get_running_instances` (currently from `src.log_output.run_metadata` at line 809)
      - `get_running_instances_for_dir` (currently from `src.log_output.run_metadata` at line 813)
  - Update `pyproject.toml`:
    - Add `src.cli_support` to orchestration layer (line 143):
      `"src.orchestrator | src.orchestrator_factory | src.orchestrator_types | src.cli_support"`
- **Verification**:
  - `uvx --from import-linter lint-imports --contract "CLI only depends on orchestrator"`
  - `uvx ruff check src/cli.py src/cli_support.py`
  - `uvx ty check`
  - `uv run pytest tests/unit/test_cli*.py -v`
  - Manual test: `uv run mala status`
- **Rollback**: Delete `src/cli_support.py`, revert imports in `cli.py`

### Task 5: Merge event_sink_console into event_sink

- **Goal**: Fix `event_sink_console` importing from `event_sink` and `log_output.console`
- **Covers**: Contract "Layered Architecture" (infra same-layer import flagged)
- **Depends on**: Task 4
- **Changes**:
  - Merge `ConsoleEventSink` class from `src/event_sink_console.py` into `src/event_sink.py`:
    - Add imports from `log_output.console` (same layer)
    - Move entire `ConsoleEventSink` class implementation (~300 lines)
  - Delete `src/event_sink_console.py`
  - Update all imports of `ConsoleEventSink`:
    - `src/orchestrator.py` line 13: Change `from .event_sink_console import ConsoleEventSink` to `from .event_sink import ConsoleEventSink`
    - `src/orchestrator_factory.py`: Update import similarly
  - Update `pyproject.toml`:
    - Remove `src.event_sink_console` from layer definition (line 149)
    - Remove `src.event_sink_console` from Contract 4 forbidden_modules (line 202)
    - Remove `src.event_sink_console` from Contract 8 source_modules (line 310)
- **Verification**:
  - `uvx --from import-linter lint-imports`
  - `uvx ruff check src/event_sink.py`
  - `uvx ty check`
  - `uv run pytest tests/ -v -k event_sink`
- **Rollback**: Restore `src/event_sink_console.py`, revert `src/event_sink.py`

### Task 6: Fix telemetry → braintrust_integration import

- **Goal**: `src/telemetry.py` should not import from `src/braintrust_integration.py`
- **Covers**: Contract "Layered Architecture" (infra cross-imports)
- **Depends on**: Task 5
- **Current State**: `src/telemetry.py` contains `BraintrustProvider` (lines 199-250) and `BraintrustSpan` (lines 154-197) classes which import from `braintrust_integration` at runtime (lines 166, 220, 248)
- **Changes**:
  - Move from `src/telemetry.py` to `src/braintrust_integration.py`:
    - `BraintrustSpan` class (lines 154-197)
    - `BraintrustProvider` class (lines 199-250)
  - `src/telemetry.py` after changes:
    - Keep `TelemetryProvider` protocol
    - Keep `TelemetrySpan` protocol
    - Keep `NullTelemetryProvider` class
    - Keep `NullSpan` class
    - Remove `BraintrustProvider` and `BraintrustSpan` classes
  - `src/braintrust_integration.py` additions:
    - Add `BraintrustSpan` class (move, keep runtime imports to same file)
    - Add `BraintrustProvider` class (move, keep runtime imports to same file)
  - Update imports in consumers:
    - `src/orchestrator.py` line 15: Change `from .telemetry import BraintrustProvider` to `from .braintrust_integration import BraintrustProvider`
    - `src/orchestrator_factory.py`: Same change
- **Verification**:
  - `uvx --from import-linter lint-imports`
  - `uvx ruff check src/telemetry.py src/braintrust_integration.py`
  - `uvx ty check`
  - `uv run pytest tests/ -v -k "telemetry or braintrust"`
- **Rollback**: Revert changes to both files

### Task 7: Update test imports and final verification

- **Goal**: Ensure all tests pass with updated import paths
- **Covers**: All contracts - final verification
- **Depends on**: Tasks 1-6
- **Changes**:
  - Search for test files importing from moved modules:
    ```bash
    # Orchestrator types
    grep -r "from src.orchestrator_factory import OrchestratorConfig" tests/
    grep -r "from src.orchestrator_factory import OrchestratorDependencies" tests/
    grep -r "from src.orchestrator_factory import DEFAULT_AGENT_TIMEOUT" tests/

    # Event sink
    grep -r "from src.event_sink_console import" tests/

    # Telemetry
    grep -r "from src.telemetry import BraintrustProvider" tests/
    grep -r "from src.telemetry import BraintrustSpan" tests/

    # CLI utilities (if any tests use them directly)
    grep -r "from src.tools.env import" tests/
    grep -r "from src.tools.locking import get_lock_dir" tests/
    grep -r "from src.log_output.run_metadata import get_running" tests/
    ```
  - Update test imports to use new paths
  - Update any mock paths that reference old module locations (e.g., `@patch("src.telemetry.BraintrustProvider")`)
- **Verification**:
  - `uvx --from import-linter lint-imports` (ALL 9 contracts pass)
  - `uvx ruff check .`
  - `uvx ty check`
  - `uv run pytest -m "unit or integration" -n auto`
  - Coverage threshold: 85%
- **Rollback**: Revert test file changes

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes
- **Circular import at runtime**: Moving classes between modules may create new cycles
  - Mitigation: Test each change incrementally with `python -c "from src.X import Y"`
- **Protocol structural typing failures**: New Protocol types may not match implementation shapes
  - Mitigation: Run type checker after each protocol change, verify with actual usage
- **Contract definition changes**: Changing Contract 2 type may have unintended effects
  - Mitigation: Run full import-linter after contract changes, verify intent matches behavior
- **MyPy/ty errors**: Moving types might break type narrowing or aliases
  - Mitigation: Check with `uvx ty check` after each task

### Breaking Changes & Compatibility
- **Changed Import Paths**:
  - `from src.orchestrator_factory import OrchestratorConfig` → `from src.orchestrator_types import OrchestratorConfig`
  - `from src.orchestrator_factory import DEFAULT_AGENT_TIMEOUT_MINUTES` → `from src.orchestrator_types import DEFAULT_AGENT_TIMEOUT_MINUTES`
  - `from src.event_sink_console import ConsoleEventSink` → `from src.event_sink import ConsoleEventSink`
  - `from src.telemetry import BraintrustProvider` → `from src.braintrust_integration import BraintrustProvider`
  - CLI utilities: `from src.tools.env import X` → `from src.cli_support import X`
- **Mitigations**:
  - Clean break approach per user decision - no re-exports
  - All internal references updated in same PR
- **Rollout Strategy**:
  - Single PR with all changes
  - All tests must pass before merge

## Testing & Validation

- **Unit Tests**
  - Test each new module imports correctly
  - Test protocol conformance for implementations (structural typing)
- **Integration / End-to-End Tests**
  - Run full test suite: `uv run pytest -m "unit or integration" -n auto`
  - Verify import-linter contracts: `uvx --from import-linter lint-imports`
- **Regression Tests**
  - All existing orchestrator tests must pass
  - All CLI tests must pass
  - No behavior changes expected
- **Manual Verification**
  - Run `mala status` to verify CLI works
  - Run `mala run --dry-run` in a test repo

### Acceptance Criteria Coverage

| Spec AC | Covered By |
|---------|------------|
| "Layered Architecture" passes | Task 1, Task 2, Task 5, Task 6 |
| "Domain layer independence" passes | Task 3 (contract fix) |
| "Domain must not depend on infra" passes | Task 2 |
| "CLI only depends on orchestrator" passes | Task 4 |
| "Infra modules independent" passes | Task 5, Task 6 |

## Rollback Strategy (Plan-Level)

- All changes are code reorganization only - no database/config changes
- Rollback: `git revert <commit-range>` or `git reset --hard <pre-refactor-commit>`
- Verification: `uvx --from import-linter lint-imports` shows original (broken) state
- No data repair needed - this is pure refactoring

## Open Questions

- None - scope is clear from architecture review and user decisions
