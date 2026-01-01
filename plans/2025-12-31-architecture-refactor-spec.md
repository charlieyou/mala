# Architecture Refactor: Orchestrator Init, Hooks Split, Evidence Extraction

## Overview

Refactor the Mala orchestrator codebase to address three high-severity architectural issues identified in the architecture review:

1. **Oversized `MalaOrchestrator.__init__`** (250+ lines): Extract initialization logic into a factory function
2. **`hooks.py` mixes unrelated concerns** (800+ lines): Split into a focused subpackage with separate modules
3. **`_finalize_issue_result` duplication** (~180 lines): Extract evidence extraction into a pure function

This is a maintainability refactor with no behavioral changes. All existing tests must pass after each phase.

## Goals

- Reduce `MalaOrchestrator.__init__` from 250+ lines to under 60 lines
- Split `hooks.py` (806 lines) into focused modules under 250 lines each
- Eliminate duplicate evidence extraction code in `_finalize_issue_result`
- Improve testability by enabling isolated unit testing of extracted components
- Maintain full backward compatibility for all public APIs

## Non-Goals (Out of Scope)

- Changing runtime behavior or adding new features
- Modifying the protocol interfaces in `src/protocols.py`
- Refactoring `IssueRunner` or other Medium-severity issues from the review
- Adding new functionality to hooks or orchestrator
- Changing the public API surface

## User Stories

- As a developer, I want the orchestrator constructor to be under 60 lines so I can understand its responsibilities at a glance
- As a developer, I want hooks organized by concern so I can find and modify caching logic without reading security patterns
- As a developer, I want evidence extraction as a pure function so I can unit test it without mocking the entire orchestrator

## Technical Design

### Architecture

This refactor follows existing patterns in the codebase:

- **Factory pattern**: Matches `build_validation_spec()` in `src/validation/spec.py`
- **Subpackage with re-exports**: Matches `src/pipeline/` and `src/validation/` organization
- **Dataclass configs**: Matches `GateRunnerConfig`, `ReviewRunnerConfig` patterns

### Phase 1: Split hooks.py into Subpackage

**New structure:**
```
src/hooks/
├── __init__.py           # Re-exports for backward compatibility
├── dangerous_commands.py # DANGEROUS_PATTERNS, DESTRUCTIVE_GIT_PATTERNS, block_dangerous_commands
├── file_cache.py         # FileReadCache, CachedFileInfo, make_file_read_cache_hook
├── lint_cache.py         # LintCache, LintCacheEntry, _detect_lint_command, make_lint_cache_hook
└── locking.py            # make_lock_enforcement_hook, make_stop_hook
```

**Key components:**
- `dangerous_commands.py`: Security patterns and blocking hooks (~120 lines)
- `file_cache.py`: File read caching to prevent redundant reads (~200 lines)
- `lint_cache.py`: Lint command caching to skip unchanged checks (~200 lines)
- `locking.py`: Lock enforcement and stop hooks (~80 lines)
- `__init__.py`: Re-exports all public symbols for backward compatibility

### Phase 2: Extract Orchestrator Factory

**New file: `src/orchestrator_factory.py`**

```python
@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator construction.

    Groups all configuration that was previously scattered across
    __init__ parameters and environment loading.
    """
    repo_path: Path
    max_agents: int | None = None
    timeout_minutes: int | None = None
    max_issues: int | None = None
    epic_id: str | None = None
    only_ids: set[str] | None = None
    braintrust_enabled: bool | None = None
    max_gate_retries: int = 3
    max_review_retries: int = 3
    disable_validations: set[str] | None = None
    coverage_threshold: float | None = None
    morph_enabled: bool | None = None
    prioritize_wip: bool = False
    focus: bool = True
    cli_args: dict[str, object] | None = None
    epic_override_ids: set[str] | None = None


@dataclass
class OrchestratorDependencies:
    """Pre-constructed dependencies for orchestrator.

    Holds all protocol implementations that the orchestrator needs,
    enabling test injection and explicit dependency management.

    Note: AgentSessionRunner is intentionally NOT included here because
    it is constructed per-issue in run_implementer(), not at orchestrator
    initialization time. Similarly, per_issue_spec is built at run() time,
    not construction time, so it remains an instance variable.
    """
    issue_provider: IssueProvider
    code_reviewer: CodeReviewer
    gate_checker: GateChecker
    log_provider: LogProvider
    telemetry_provider: TelemetryProvider
    event_sink: MalaEventSink
    epic_verifier: EpicVerifier | None
    gate_runner: GateRunner
    review_runner: ReviewRunner
    run_coordinator: RunCoordinator


def create_orchestrator(
    config: OrchestratorConfig,
    *,
    mala_config: MalaConfig | None = None,
    # Protocol overrides for testing (individual components)
    issue_provider: IssueProvider | None = None,
    code_reviewer: CodeReviewer | None = None,
    gate_checker: GateChecker | None = None,
    log_provider: LogProvider | None = None,
    telemetry_provider: TelemetryProvider | None = None,
    event_sink: MalaEventSink | None = None,
    # Full dependency override (for complete control, e.g., in tests)
    deps: OrchestratorDependencies | None = None,
) -> MalaOrchestrator:
    """Factory function to construct a fully-configured MalaOrchestrator.

    This function encapsulates all the initialization logic that was
    previously in MalaOrchestrator.__init__, including:
    - Loading MalaConfig from environment if not provided
    - Deriving feature flags from config
    - Constructing default protocol implementations
    - Building pipeline runners (GateRunner, ReviewRunner, RunCoordinator)
    - Setting up EpicVerifier when appropriate

    Args:
        config: OrchestratorConfig with all orchestration parameters
        mala_config: Optional pre-loaded MalaConfig (loads from env if None)
        issue_provider: Override for IssueProvider (BeadsClient by default)
        code_reviewer: Override for CodeReviewer (DefaultReviewer by default)
        gate_checker: Override for GateChecker (QualityGate by default)
        log_provider: Override for LogProvider (FileSystemLogProvider by default)
        telemetry_provider: Override for TelemetryProvider
        event_sink: Override for MalaEventSink (ConsoleEventSink by default)
        deps: Complete OrchestratorDependencies override (takes precedence over
              individual overrides; use for full test control)

    Returns:
        Fully-constructed MalaOrchestrator ready to run.

    Usage patterns:
        # Production: minimal args, factory builds everything
        orch = create_orchestrator(OrchestratorConfig(repo_path=Path(".")))

        # Testing with specific overrides
        orch = create_orchestrator(config, issue_provider=mock_provider)

        # Testing with full control
        orch = create_orchestrator(config, deps=fully_mocked_deps)
    """
    ...
```

**Backward Compatibility Strategy:**

The current `MalaOrchestrator.__init__` signature will be preserved for backward compatibility. The refactored constructor will support both patterns:

1. **Legacy signature** (backward compatible): All current parameters continue to work
2. **New signature** (preferred): `OrchestratorConfig` + `OrchestratorDependencies`

```python
def __init__(
    self,
    # Legacy signature parameters (all optional when using new pattern)
    repo_path: Path | None = None,
    max_agents: int | None = None,
    timeout_minutes: int | None = None,
    # ... all existing parameters with defaults ...

    # New pattern parameters
    _config: OrchestratorConfig | None = None,
    _deps: OrchestratorDependencies | None = None,
    _mala_config: MalaConfig | None = None,
):
    """Initialize orchestrator.

    Supports two construction patterns:
    1. Legacy: Pass individual parameters (existing API, backward compatible)
    2. Factory: Use create_orchestrator() which passes _config, _deps, _mala_config

    The legacy pattern will be deprecated in a future version but remains
    fully supported. All existing call sites continue to work unchanged.
    """
    if _config is not None and _deps is not None:
        # New pattern: use pre-constructed config and dependencies
        self._init_from_config(_config, _deps, _mala_config)
    else:
        # Legacy pattern: construct from individual parameters
        self._init_legacy(repo_path, max_agents, timeout_minutes, ...)
```

**Config Precedence Rules:**

When using the factory function, precedence is (highest to lowest):
1. Explicit protocol overrides passed to `create_orchestrator()`
2. Values in `OrchestratorConfig`
3. Values in `MalaConfig` (from environment)
4. Built-in defaults

This matches the existing precedence in `MalaOrchestrator.__init__`.

### Phase 3: Extract Evidence Extraction

**Location**: Keep types and functions in `src/orchestrator.py` to minimize file count.
The functions are private (`_build_gate_metadata`) and tightly coupled to orchestrator internals.
If the orchestrator file remains too large after refactoring, extraction to `src/evidence.py`
can be done as a follow-up without affecting the public API.

**New types in `src/orchestrator.py`**:

```python
@dataclass
class GateMetadata:
    """Extracted metadata from a gate result for run recording."""
    quality_gate_result: QualityGateResult | None
    validation_result: MetaValidationResult | None


def _build_gate_metadata(
    gate_result: GateResult | None,
    log_path: Path | None,
    quality_gate: GateChecker,
    per_issue_spec: ValidationSpec | None,
) -> GateMetadata:
    """Extract gate metadata from a gate result.

    This pure function encapsulates the evidence extraction logic that
    was previously duplicated across three branches in _finalize_issue_result.

    Args:
        gate_result: The stored gate result (may be None for fallback case)
        log_path: Path to session log (for fallback parsing)
        quality_gate: GateChecker for fallback evidence parsing
        per_issue_spec: ValidationSpec for spec-driven parsing

    Returns:
        GateMetadata with quality_gate_result and validation_result,
        either or both may be None if insufficient data.
    """
    if gate_result is None:
        return GateMetadata(None, None)

    evidence = gate_result.validation_evidence
    commit_hash = gate_result.commit_hash

    # Build evidence dict from stored evidence
    evidence_dict: dict[str, bool] = {}
    if evidence is not None:
        evidence_dict = evidence.to_evidence_dict()
    evidence_dict["commit_found"] = commit_hash is not None

    quality_gate_result = QualityGateResult(
        passed=gate_result.passed,
        evidence=evidence_dict,
        failure_reasons=list(gate_result.failure_reasons),
    )

    validation_result = None
    if evidence is not None:
        commands_run = [
            QualityGate.KIND_TO_NAME.get(kind, kind.value)
            for kind, ran in evidence.commands_ran.items()
            if ran
        ]
        gate_failed_commands = [
            cmd for cmd in evidence.failed_commands
            if cmd not in QUALITY_GATE_IGNORED_COMMANDS
        ]
        validation_result = MetaValidationResult(
            passed=gate_result.passed,
            commands_run=commands_run,
            commands_failed=gate_failed_commands,
        )

    return GateMetadata(quality_gate_result, validation_result)


def _build_gate_metadata_from_logs(
    log_path: Path,
    result: IssueResult,
    quality_gate: GateChecker,
    per_issue_spec: ValidationSpec | None,
) -> GateMetadata:
    """Fallback: Extract gate metadata by parsing logs directly.

    This handles the edge case where no stored gate result exists
    (e.g., in tests that mock run_implementer or early failures).

    Args:
        log_path: Path to session log file.
        result: The IssueResult containing failure summary.
        quality_gate: GateChecker for evidence parsing.
        per_issue_spec: ValidationSpec for spec-driven parsing (may be None if disabled).

    Returns:
        GateMetadata with quality_gate_result and validation_result.
        Uses result.success to determine passed status.
    """
    # Handle case where validations are disabled
    if per_issue_spec is None:
        return GateMetadata(None, None)

    evidence = quality_gate.parse_validation_evidence_with_spec(log_path, per_issue_spec)
    commit_result = quality_gate.check_commit_exists(result.issue_id)

    # Extract failure reasons from result summary (only if failed)
    failure_reasons = []
    if not result.success and "Quality gate failed:" in result.summary:
        reasons_part = result.summary.replace("Quality gate failed: ", "")
        failure_reasons = [r.strip() for r in reasons_part.split(";")]

    # Build evidence dict
    evidence_dict = evidence.to_evidence_dict()
    evidence_dict["commit_found"] = commit_result.exists

    # Use result.success to determine passed status (not hardcoded False)
    quality_gate_result = QualityGateResult(
        passed=result.success,
        evidence=evidence_dict,
        failure_reasons=failure_reasons,
    )

    # Build validation result from evidence
    commands_run = [
        QualityGate.KIND_TO_NAME.get(kind, kind.value)
        for kind, ran in evidence.commands_ran.items()
        if ran
    ]
    gate_failed_commands = [
        cmd for cmd in evidence.failed_commands
        if cmd not in QUALITY_GATE_IGNORED_COMMANDS
    ]
    validation_result = MetaValidationResult(
        passed=result.success,
        commands_run=commands_run,
        commands_failed=gate_failed_commands,
    )

    return GateMetadata(quality_gate_result, validation_result)
```

**Refactored `_finalize_issue_result`** (target: ~80 lines):

```python
async def _finalize_issue_result(
    self,
    issue_id: str,
    result: IssueResult,
    run_metadata: RunMetadata,
) -> None:
    """Record an issue result, update metadata, and emit logs."""
    log_path = self.session_log_paths.get(issue_id)
    stored_gate_result = self.last_gate_results.get(issue_id)

    # Extract metadata once using the pure function
    if result.success and stored_gate_result is not None:
        metadata = _build_gate_metadata(stored_gate_result, log_path, self.quality_gate, self.per_issue_spec)
        # Force passed=True for success case
        if metadata.quality_gate_result:
            metadata.quality_gate_result = QualityGateResult(
                passed=True,
                evidence=metadata.quality_gate_result.evidence,
                failure_reasons=[],
            )
    elif stored_gate_result is not None:
        metadata = _build_gate_metadata(stored_gate_result, log_path, self.quality_gate, self.per_issue_spec)
    elif log_path and log_path.exists():
        # Fallback: parse logs directly
        metadata = _build_gate_metadata_from_logs(log_path, result, self.quality_gate, self.per_issue_spec)
    else:
        metadata = GateMetadata(None, None)

    # Success path: close issue and check epic
    if result.success:
        if await self.beads.close_async(issue_id):
            self.event_sink.on_issue_closed(issue_id, issue_id)
            await self._check_epic_closure(issue_id)

    # Record to run metadata
    self._record_issue_run(issue_id, result, run_metadata, log_path, metadata)

    # Cleanup and emit
    self._cleanup_session_paths(issue_id)
    self._emit_completion(issue_id, result)
```

## Implementation Plan

### Phase 1: Split hooks.py into Subpackage

1. [ ] Create `src/hooks/` directory
2. [ ] Create `src/hooks/dangerous_commands.py` with:
   - `DANGEROUS_PATTERNS`
   - `DESTRUCTIVE_GIT_PATTERNS`
   - `SAFE_GIT_ALTERNATIVES`
   - `BASH_TOOL_NAMES`
   - `block_dangerous_commands()`
   - `block_morph_replaced_tools()` (MCP tool blocking, grouped with dangerous patterns)
3. [ ] Create `src/hooks/file_cache.py` with:
   - `CachedFileInfo`
   - `FileReadCache`
   - `make_file_read_cache_hook()`
   - `FILE_WRITE_TOOLS`, `FILE_PATH_KEYS` constants
4. [ ] Create `src/hooks/lint_cache.py` with:
   - `LINT_COMMAND_PATTERNS`
   - `LintCacheEntry`
   - `LintCache`
   - `_get_git_state()`
   - `_detect_lint_command()`
   - `make_lint_cache_hook()`
5. [ ] Create `src/hooks/locking.py` with:
   - `make_lock_enforcement_hook()`
   - `make_stop_hook()`
6. [ ] Create `src/hooks/__init__.py` with re-exports of all public symbols
7. [ ] Update internal imports between hooks modules (cross-module imports within `src/hooks/`)
8. [ ] Verify external consumers work via re-exports (no changes needed to `src/pipeline/agent_session_runner.py` etc. due to `__init__.py` re-exports)
9. [ ] Delete `src/hooks.py`
10. [ ] Run tests: `uv run pytest -m "unit or integration"`
11. [ ] Run linting: `uvx ruff check . && uvx ty check`

### Phase 2: Extract Orchestrator Factory

1. [ ] Create `src/orchestrator_factory.py` with `OrchestratorConfig` dataclass
2. [ ] Add `OrchestratorDependencies` dataclass for pre-constructed dependencies
3. [ ] Implement `_build_default_dependencies()` helper (extracted from current `__init__`)
4. [ ] Implement `create_orchestrator()` factory function
5. [ ] Refactor `MalaOrchestrator.__init__` to accept config and deps (simple assignments only)
6. [ ] Update `src/cli.py` to use `create_orchestrator()`
7. [ ] Update test fixtures to use factory or direct `__init__` with mocks
8. [ ] Run tests: `uv run pytest -m "unit or integration"`
9. [ ] Run linting: `uvx ruff check . && uvx ty check`

### Phase 3: Extract Evidence Extraction

1. [ ] Add `GateMetadata` dataclass to `src/orchestrator.py`
2. [ ] Implement `_build_gate_metadata()` pure function
3. [ ] Implement `_build_gate_metadata_from_logs()` for fallback case
4. [ ] Refactor `_finalize_issue_result()` to use new functions
5. [ ] Extract helper methods: `_check_epic_closure()`, `_record_issue_run()`, `_cleanup_session_paths()`, `_emit_completion()`
6. [ ] Add unit tests for `_build_gate_metadata()` with various inputs
7. [ ] Run tests: `uv run pytest -m "unit or integration"`
8. [ ] Run linting: `uvx ruff check . && uvx ty check`

### Final Validation

1. [ ] Run full test suite: `uv run pytest -m "unit or integration" -n auto`
2. [ ] Verify coverage: `uv run pytest --cov --cov-fail-under=85`
3. [ ] Final lint: `uvx ruff check . && uvx ruff format . && uvx ty check`

## Testing Strategy

### Unit Tests

- **Phase 1**: Existing hook tests pass via re-exports; add focused tests for each new module
- **Phase 2**: Test `create_orchestrator()` with various config combinations; test that direct `__init__` works with mock dependencies
- **Phase 3**: Test `_build_gate_metadata()` with:
  - Successful gate result with full evidence
  - Failed gate result with partial evidence
  - None gate result (fallback path)
  - Edge cases: empty failure reasons, missing commit hash

### Integration Tests

- All existing integration tests must pass unchanged
- Verify orchestrator behavior is identical before/after refactor

### Manual Testing

- Run `mala` CLI with refactored code
- Verify identical output and behavior

## Open Questions

None - all design decisions have been made.

## Decisions Made

1. **Factory function over builder class**: Simpler, more Pythonic, matches existing `build_validation_spec()` pattern
2. **Subpackage over flat modules**: Better organization, matches `src/pipeline/` and `src/validation/` patterns
3. **Phase order (hooks → orchestrator → evidence)**: Start with lowest risk (hooks is self-contained), then orchestrator (larger but isolated), then evidence (depends on orchestrator understanding)
4. **Evidence as pure function returning dataclass**: Enables isolated testing, matches `GateResult` pattern, clearly separates data extraction from side effects
5. **Keep MalaOrchestrator class**: Factory creates it, but class remains for backward compatibility and test injection
6. **Re-export all public symbols**: Ensures `from src.hooks import X` continues to work after split
7. **Preserve legacy `__init__` signature**: The constructor will support both legacy (individual params) and new (config+deps) patterns. Legacy pattern is deprecated but fully functional, ensuring no breaking changes
8. **Config precedence matches existing behavior**: Factory uses same precedence as current `__init__`: explicit overrides > OrchestratorConfig > MalaConfig > defaults
9. **Use dataclass for configs**: `OrchestratorConfig` uses `@dataclass` to match `GateRunnerConfig`, `ReviewRunnerConfig` patterns already in `src/pipeline/`. No Pydantic needed for simple value objects
