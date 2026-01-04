# Implementation Plan: Test Directory Reorganization

## Context & Goals
- **Spec**: N/A — derived from user description
- Reorganize flat test directory into category-based structure (unit/integration/e2e)
- Enable path-based auto-marking via enhanced `pytest_collection_modifyitems` hook
- Scope conftest.py to appropriate test categories
- Improve test isolation and discoverability

## Scope & Non-Goals
- **In Scope**
  - Move ~45 unit tests to tests/unit/ with source-mirroring subdirectories
  - Move ~7 integration tests to tests/integration/
  - Move ~4 e2e tests to tests/e2e/
  - Create category-specific conftest.py files (tests/e2e/conftest.py)
  - Configure pytest for path-based marker auto-assignment
  - Add `--strict-markers` to pyproject.toml addopts
  - Remove `_e2e` and `_integration` suffixes from file names after moving
  - Delete empty tests/domain/ and tests/infra/ directories after consolidation
  - Move tests/claude_auth.py to tests/e2e/claude_auth.py
- **Out of Scope (Non-Goals)**
  - Refactoring test code itself (beyond import path fixes)
  - Adding new tests
  - Changing test behavior
  - CI/CD pipeline modifications (assumes `pytest tests/` invocation pattern)
  - Documentation updates (README, contributing guides)

## Assumptions & Constraints
- **Migration Strategy**: Big-bang (single PR) — all changes in one commit to avoid inconsistent states
- **Git History**: Use `git mv` to preserve history where possible
- **No Compatibility Shims**: Per CLAUDE.md, update all imports directly without creating re-export shims
- **Package Structure**: New directories require `__init__.py` files for Python imports to work (e.g., `from tests.e2e.claude_auth import ...`). While pytest discovers tests without them, helper module imports require proper package structure.

### Implementation Constraints
- Python imports within tests referencing `tests.claude_auth` must update to `tests.e2e.claude_auth`
- Avoid circular imports when moving helper files
- Ensure `pytest -n auto` (xdist) continues to work correctly
- Preserve existing fixture scopes — `make_orchestrator` stays in root conftest.py (used by all categories)

### Testing Constraints
- All existing tests must pass after migration
- Test discovery must work with new structure
- Coverage reporting must remain accurate
- Parallel execution (`-n auto`) must verify no new race conditions
- `--strict-markers` will fail build if any test lacks a valid marker

## Prerequisites
- [x] Context provided for target structure
- [ ] Verify no in-flight test changes that would cause merge conflicts
- [ ] Dynamic categorization of existing tests (based on file content analysis or path conventions)

## High-Level Approach

1. **Create Directory Structure**: Create tests/unit/{cli,core,domain,infra/hooks,orchestration,pipeline}/, tests/integration/{domain,infra,orchestration,pipeline}/, and tests/e2e/ with `__init__.py` files in each
2. **Update Configuration**: Modify `pyproject.toml` to add `--strict-markers` to addopts; update `tests/conftest.py` with enhanced path-based marking hook (using `item.path` for pytest 7+ compatibility)
3. **Move Files**: Execute `git mv` for all test files to their new locations, renaming to remove `_e2e`/`_integration` suffixes
4. **Create e2e conftest**: Create `tests/e2e/conftest.py` with Claude auth fixtures (credential copying logic currently in root conftest.py)
5. **Fix Imports**: Search and replace `tests.claude_auth` → `tests.e2e.claude_auth`
6. **Cleanup**: Delete empty tests/domain/, tests/infra/ directories (including their `__init__.py` files)
7. **Verify**: Run `pytest -n auto` to verify all tests pass with parallel safety

## Technical Design

### Architecture
The new structure mirrors source code organization for unit tests and groups by test type at the top level:

```
tests/
├── conftest.py              # Global: env setup, make_orchestrator, log_provider, path-based marking hook
├── fixtures/                # Sample projects/configs (unchanged)
├── unit/
│   ├── cli/
│   ├── core/
│   ├── domain/
│   ├── infra/
│   │   └── hooks/
│   ├── orchestration/
│   └── pipeline/
├── integration/
│   ├── conftest.py          # Integration-specific setup (if needed)
│   ├── domain/
│   ├── infra/
│   ├── orchestration/
│   └── pipeline/
└── e2e/
    ├── conftest.py          # Claude auth fixtures, credential copying
    ├── claude_auth.py       # Auth helpers (moved from tests/)
    └── test_*.py
```

### Data Model
N/A — purely structural reorganization.

### API/Interface Design
N/A — no API changes.

### Pytest Configuration Changes

**`tests/conftest.py` (Root) — Modify:**
- **Update** `pytest_collection_modifyitems` to implement path-based marking:
  ```python
  def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
      """Apply markers based on test file path."""
      for item in items:
          # Skip if already marked
          if any(marker in item.keywords for marker in ("unit", "integration", "e2e")):
              continue
          # Path-based auto-marking (use item.path for pytest 7+ compatibility)
          path = str(item.path)
          if "/e2e/" in path:
              item.add_marker(pytest.mark.e2e)
          elif "/integration/" in path:
              item.add_marker(pytest.mark.integration)
          elif "/unit/" in path:
              item.add_marker(pytest.mark.unit)
          # No default — --strict-markers will fail if unmarked
  ```
- **Keep**: `pytest_configure` (env setup), `make_orchestrator` fixture, `log_provider` fixture
- **Remove**: Credential copying logic (moves to e2e/conftest.py)

**`tests/e2e/conftest.py` — New:**
- Claude credential copying logic (from root conftest.py)
- Any e2e-specific fixtures

**`pyproject.toml` — Modify:**
- Add `--strict-markers` to `[tool.pytest.ini_options].addopts`
- Current: `addopts = "-m 'unit or integration' -q --tb=short --no-header --disable-warnings"`
- Updated: `addopts = "--strict-markers -m 'unit or integration' -q --tb=short --no-header --disable-warnings"`
- **Note**: Markers `unit`, `integration`, `e2e` are already registered in `[tool.pytest.ini_options].markers` (lines 103-107 of pyproject.toml)

### File Impact Summary

**Directories to Create:**
| Path | Status | Description |
|------|--------|-------------|
| tests/unit/cli/ | **New** | Unit tests for CLI layer |
| tests/unit/core/ | **New** | Unit tests for core layer |
| tests/unit/domain/ | **New** | Unit tests for domain layer |
| tests/unit/infra/ | **New** | Unit tests for infra layer |
| tests/unit/infra/hooks/ | **New** | Unit tests for hooks |
| tests/unit/orchestration/ | **New** | Unit tests for orchestration layer |
| tests/unit/pipeline/ | **New** | Unit tests for pipeline layer |
| tests/integration/domain/ | **New** | Integration tests for domain |
| tests/integration/infra/ | **New** | Integration tests for infra |
| tests/integration/orchestration/ | **New** | Integration tests for orchestration |
| tests/integration/pipeline/ | **New** | Integration tests for pipeline |
| tests/e2e/ | **New** | End-to-end tests |

**Files to Move (Unit -> tests/unit/):**

| Current Location | Target Location |
|------------------|-----------------|
| tests/test_cli.py | tests/unit/cli/test_cli.py |
| tests/test_main.py | tests/unit/cli/test_main.py |
| tests/test_config.py | tests/unit/infra/test_config.py |
| tests/test_config_loader.py | tests/unit/infra/test_config_loader.py |
| tests/test_config_merger.py | tests/unit/infra/test_config_merger.py |
| tests/test_beads_client.py | tests/unit/infra/test_beads_client.py |
| tests/test_hooks.py | tests/unit/infra/hooks/test_hooks.py |
| tests/test_event_sink.py | tests/unit/infra/test_event_sink.py |
| tests/test_telemetry.py | tests/unit/infra/test_telemetry.py |
| tests/test_git_utils.py | tests/unit/infra/test_git_utils.py |
| tests/test_session_log_parser.py | tests/unit/infra/test_session_log_parser.py |
| tests/test_cerberus_review.py | tests/unit/infra/test_cerberus_review.py |
| tests/test_cerberus_gate_cli.py | tests/unit/infra/test_cerberus_gate_cli.py |
| tests/test_braintrust_integration.py | tests/unit/infra/test_braintrust_integration.py |
| tests/test_agent_runtime.py | tests/unit/infra/test_agent_runtime.py |
| tests/test_context_pressure_handler.py | tests/unit/pipeline/test_context_pressure_handler.py |
| tests/test_message_stream_processor.py | tests/unit/pipeline/test_message_stream_processor.py |
| tests/test_gate_runner.py | tests/unit/pipeline/test_gate_runner.py |
| tests/test_review_runner.py | tests/unit/pipeline/test_review_runner.py |
| tests/test_issue_execution_coordinator.py | tests/unit/pipeline/test_issue_execution_coordinator.py |
| tests/test_run_metadata.py | tests/unit/pipeline/test_run_metadata.py |
| tests/test_orchestrator.py | tests/unit/orchestration/test_orchestrator.py |
| tests/test_orchestration_helpers.py | tests/unit/orchestration/test_orchestration_helpers.py |
| tests/test_issue_manager.py | tests/unit/orchestration/test_issue_manager.py |
| tests/test_wip_prioritization.py | tests/unit/orchestration/test_wip_prioritization.py |
| tests/test_domain_prompts.py | tests/unit/domain/test_domain_prompts.py |
| tests/test_spec.py | tests/unit/domain/test_spec.py |
| tests/test_spec_workspace.py | tests/unit/domain/test_spec_workspace.py |
| tests/test_epic_scope.py | tests/unit/domain/test_epic_scope.py |
| tests/test_epic_verification_retry.py | tests/unit/domain/test_epic_verification_retry.py |
| tests/test_validation.py | tests/unit/domain/test_validation.py |
| tests/test_validation_config.py | tests/unit/domain/test_validation_config.py |
| tests/test_validation_gating.py | tests/unit/domain/test_validation_gating.py |
| tests/test_run_level_validation.py | tests/unit/domain/test_run_level_validation.py |
| tests/test_quality_gate.py | tests/unit/domain/test_quality_gate.py |
| tests/test_coverage.py | tests/unit/domain/test_coverage.py |
| tests/test_coverage_args.py | tests/unit/domain/test_coverage_args.py |
| tests/test_code_pattern_matcher.py | tests/unit/domain/test_code_pattern_matcher.py |
| tests/test_dangerous_commands.py | tests/unit/domain/test_dangerous_commands.py |
| tests/test_locking_key.py | tests/unit/infra/test_locking_key.py |
| tests/test_lint_cache.py | tests/unit/infra/test_lint_cache.py |
| tests/test_log_events.py | tests/unit/core/test_log_events.py |
| tests/test_logging_console.py | tests/unit/infra/test_logging_console.py |
| tests/test_lifecycle.py | tests/unit/infra/test_lifecycle.py |
| tests/test_lazy_imports.py | tests/unit/test_lazy_imports.py |
| tests/test_review_output_parser.py | tests/unit/infra/test_review_output_parser.py |
| tests/test_tool_name_extractor.py | tests/unit/infra/test_tool_name_extractor.py |
| tests/test_test_mutex.py | tests/unit/infra/test_test_mutex.py |
| tests/test_worktree.py | tests/unit/infra/test_worktree.py |

**Existing Nested Files to Consolidate:**

| Current Location | Target Location |
|------------------|-----------------|
| tests/domain/test_deadlock.py | tests/unit/domain/test_deadlock.py |
| tests/infra/hooks/test_deadlock_hook.py | tests/unit/infra/hooks/test_deadlock_hook.py |

**Files to Move (Integration -> tests/integration/):**

| Current Location | Target Location | Notes |
|------------------|-----------------|-------|
| tests/test_agent_session_runner.py | tests/integration/pipeline/test_agent_session_runner.py | |
| tests/test_command_runner.py | tests/integration/infra/test_command_runner.py | |
| tests/test_epic_verifier.py | tests/integration/domain/test_epic_verifier.py | |
| tests/test_lock_integration.py | tests/integration/infra/test_lock.py | Remove `_integration` suffix |
| tests/test_lock_scripts.py | tests/integration/infra/test_lock_scripts.py | |
| tests/test_preset_registry.py | tests/integration/domain/test_preset_registry.py | |
| tests/test_validation_config_integration.py | tests/integration/domain/test_validation_config.py | Remove `_integration` suffix |
| tests/integration/test_deadlock_integration.py | tests/integration/domain/test_deadlock.py | Remove `_integration` suffix |

**Files to Move (E2E -> tests/e2e/):**

| Current Location | Target Location | Notes |
|------------------|-----------------|-------|
| tests/test_e2e.py | tests/e2e/test_e2e.py | |
| tests/test_epic_verifier_sdk_e2e.py | tests/e2e/test_epic_verifier_sdk.py | Remove `_e2e` suffix |
| tests/test_review_gate_e2e.py | tests/e2e/test_review_gate.py | Remove `_e2e` suffix |
| tests/test_lock_sdk_integration.py | tests/e2e/test_lock_sdk.py | SDK tests are e2e (require real auth) |
| tests/claude_auth.py | tests/e2e/claude_auth.py | Helper module, not a test file |

**Files to Modify:**

| Path | Status | Description |
|------|--------|-------------|
| tests/conftest.py | Exists | Update path-based marking hook, keep global fixtures |
| pyproject.toml | Exists | Add --strict-markers to addopts |

**Files to Create:**

| Path | Status | Description |
|------|--------|-------------|
| tests/e2e/conftest.py | **New** | e2e-specific fixtures, Claude credential copying |
| tests/unit/__init__.py | **New** | Package marker for unit test imports |
| tests/unit/cli/__init__.py | **New** | Package marker |
| tests/unit/core/__init__.py | **New** | Package marker |
| tests/unit/domain/__init__.py | **New** | Package marker |
| tests/unit/infra/__init__.py | **New** | Package marker |
| tests/unit/infra/hooks/__init__.py | **New** | Package marker |
| tests/unit/orchestration/__init__.py | **New** | Package marker |
| tests/unit/pipeline/__init__.py | **New** | Package marker |
| tests/integration/domain/__init__.py | **New** | Package marker |
| tests/integration/infra/__init__.py | **New** | Package marker |
| tests/integration/orchestration/__init__.py | **New** | Package marker |
| tests/integration/pipeline/__init__.py | **New** | Package marker |
| tests/e2e/__init__.py | **New** | Package marker (required for `from tests.e2e.claude_auth import ...`) |

**Directories to Delete (after moves):**

| Path | Contents Deleted | Reason |
|------|------------------|--------|
| tests/domain/ | `__init__.py`, `test_deadlock.py` | Empty after test moves to unit/domain/ |
| tests/infra/ | `__init__.py`, `hooks/__init__.py`, `hooks/test_deadlock_hook.py` | Empty after test moves to unit/infra/hooks/ |
| tests/unit/ (if empty placeholder) | Nothing | Recreated with content |

**Note**: The existing `tests/integration/__init__.py` will be preserved as the integration directory is restructured in place.

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes
- **Stranded Tests**: A test file left in root or moved to wrong folder will fail due to lack of markers (--strict-markers enforces this)
  - *Handling*: Build fails, preventing merge
- **Import Path Breakage**: Tests importing `tests.claude_auth` will break
  - *Handling*: Grep and replace all occurrences with `tests.e2e.claude_auth`
- **conftest Scope Change**: Moving auth fixtures to e2e/conftest.py makes them unavailable to unit/integration
  - *Handling*: Verify no unit/integration tests rely on Claude auth fixtures (they shouldn't by design)
- **Parallel Safety**: New directory structure could surface latent race conditions
  - *Handling*: Run `pytest -n auto` as validation step

### Breaking Changes & Compatibility
- **Potential Breaking Changes**:
  - Developers used to running `pytest tests/test_foo.py` will need to update paths
  - Any scripts/tooling referencing old test paths will break
- **Mitigations**:
  - Big-bang migration ensures no intermediate inconsistent states
  - `git mv` preserves history for easy path lookup
- **Rollout Strategy**:
  - Single PR with all changes
  - No feature flags needed (internal tooling only)

## Testing & Validation Strategy

- **Pre-Migration Verification**
  - Run `pytest --collect-only -q` to capture baseline test count per category

- **Post-Migration Verification**
  - Run `pytest --collect-only -m unit` and verify count matches expectation (~45)
  - Run `pytest --collect-only -m integration` and verify count matches expectation (~8)
  - Run `pytest --collect-only -m e2e` and verify count matches expectation (~4)
  - Run `pytest --collect-only` to check for warnings/errors about unmarked tests

- **Automated Verification**
  - Run full suite: `pytest -n auto` to verify parallel safety
  - Run category-specific: `pytest -m e2e` (uses marker filter, not path — addopts default excludes e2e)
  - Run e2e explicitly: `pytest -o "addopts=" -m e2e` (override addopts to include e2e)
  - Run coverage: `pytest --cov=src` to verify coverage reporting accuracy

- **Manual Verification**
  - Verify `--strict-markers` fails for unmarked test (can add temporary unmarked test to verify)
  - Verify path-based auto-marking works via `pytest --collect-only -v`

### Acceptance Criteria Coverage
| Requirement | Covered By |
|-------------|------------|
| Flat structure replaced with category-based hierarchy | Directory structure: unit/, integration/, e2e/ |
| Source-mirroring for unit tests | Subdirs match src/ structure: cli/, domain/, infra/, etc. |
| Path-based auto-marking | Enhanced `pytest_collection_modifyitems` hook in conftest.py |
| Strict marker enforcement | `--strict-markers` in pyproject.toml addopts |
| Scoped conftest.py | tests/e2e/conftest.py with auth fixtures |
| File suffix cleanup | Remove `_e2e`/`_integration` suffixes from moved files |
| Empty directory cleanup | Delete tests/domain/, tests/infra/ after consolidation |
| Parallel safety verified | Run `pytest -n auto` as validation |
| All tests pass | Full pytest run after migration |

## Rollback Strategy

- **Git Revert**: Since this is a structural file-move change, `git revert <commit_hash>` cleanly restores previous state
- **Verification**: Run `pytest` after revert to ensure all tests pass with old structure
- **No Data Cleanup Required**: No databases, feature flags, or runtime state to revert

## Open Questions

None — all key decisions have been made:
1. Migration: Big-bang (single PR) - **Decided**
2. Auto-marking: Path-based pytest hook - **Decided**
3. Naming: Remove suffixes after moving - **Decided**
4. Strict markers: Enable `--strict-markers` - **Decided**
5. Cleanup: Delete empty directories - **Decided**
6. Validation: Run with `-n auto` for parallel safety - **Decided**

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` -> Beads issues with dependencies for multi-agent execution
- (default) -> TODO.md checklist for simpler tracking
