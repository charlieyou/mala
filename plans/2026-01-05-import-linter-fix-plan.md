# Implementation Plan: Enable Aspirational Import-Linter Contracts

## Context & Goals
- **Spec**: Aspirational contracts defined in `pyproject.toml` (currently commented out)
- Enable 4 commented-out import-linter contracts that define stricter architectural boundaries:
  1. Infra hooks do not import clients, IO, or telemetry
  2. Only orchestration.factory imports infra.clients
  3. Hooks do not import agent runtime
  4. No claude_agent_sdk outside infra SDK boundary
- Improves architectural clarity by enforcing layered boundaries more strictly
- Benefits maintainability by making dependency violations fail fast at lint time

## Scope & Non-Goals
- **In Scope**
  - Fix code violations so aspirational contracts can pass
  - Uncomment the 4 contracts in pyproject.toml once code is fixed
  - Update all affected imports and callers
  - Update tests to reflect new patterns (mock factories instead of direct classes)
  - **Modify Contract 4 definition** to use `allow_indirect_imports = true` with `ignore_imports` (matching existing braintrust/anthropic contract patterns)
- **Out of Scope (Non-Goals)**
  - Adding new functionality beyond enabling the contracts
  - Refactoring unrelated code patterns
  - Backwards-compatibility shims (per CLAUDE.md: "No backward-compatibility shims")

## Assumptions & Constraints
- All changes should maintain existing behavior (no observable changes to CLI or orchestrator)
- No backwards-compatibility shims (per CLAUDE.md)
- Test suite must pass after changes (unit + integration)
- Coverage threshold of 72% must be maintained

### Implementation Constraints
- Cannot introduce new import violations while fixing existing ones
- Must use dependency injection patterns where needed
- Factory pattern already exists in `src/orchestration/factory.py`
- Import-linter tracks ALL imports including `TYPE_CHECKING` and deferred/dynamic imports
- Per CLAUDE.md: "No re-exports" - don't create modules that just import and re-export

### Testing Constraints
- Run `uv run lint-imports` after each fix to verify progress
- Run `uv run pytest -m unit` to ensure no regressions
- Coverage threshold of 72% must be maintained
- Tests should mock at factory/protocol boundaries, not internal classes

## Prerequisites
- [x] Understand aspirational contracts from pyproject.toml
- [x] Identify all import chains that violate each contract
- [x] Confirm existing test patterns for factory injection
- [x] **Verified**: Contract 3 already passes (tested by temporarily uncommenting)

## High-Level Approach

**Recommended sequencing** (to minimize rework):

1. **Contract 1 (Hooks → IO)**: Create `src/infra/tools/cerberus.py` — independent, can start immediately
2. **Contract 3 (Hooks → Agent Runtime)**: Already passes — just uncomment
3. **Contract 2 (Orchestration → Clients)**: Remove direct client imports; use protocol methods and factory helpers
4. **Contract 4 (SDK boundary)**: Modify contract to use `allow_indirect_imports = true` with `ignore_imports`; ensure only allowed modules import SDK

Contracts 1 and 3 are independent. Contract 4 should be done after Contract 2 since both affect factory.py wiring.

## Technical Design

### Architecture

The changes enforce a cleaner layered architecture:

```
CLI (src/cli)
    |
    v
Orchestration (src/orchestration)
    |
    +-- factory.py (allowed to import infra.clients)
    +-- orchestrator.py (uses protocols, receives dependencies via DI)
    +-- orchestration_wiring.py (imports from infra.mcp)
    |
    v
Core (src/core)
    |
    +-- protocols.py (defines CodeReviewer, IssueProvider interfaces)
    +-- models.py (shared data classes like EpicVerificationResult)
    |
    v
Infra (src/infra)
    +-- clients/ (BeadsClient, CerberusReviewer, braintrust_integration)
    +-- tools/ (env.py, locking.py, locking_mcp.py, cerberus.py)
    +-- sdk_adapter.py
    +-- epic_verifier.py
    +-- agent_runtime/
    +-- hooks/ (cannot import clients, IO, telemetry, or agent runtime)
```

**SDK Boundary** (for Contract 4): The modules allowed to import `claude_agent_sdk` are:
- `src.infra.clients.**` (includes braintrust_integration)
- `src.infra.sdk_adapter`
- `src.infra.epic_verifier`
- `src.infra.tools.locking_mcp`
- `src.infra.agent_runtime.**`

### Data Model
N/A - No new data models. `EpicVerificationResult` already exists in `src/core/models.py`.

### API/Interface Design

**New protocol method on `CodeReviewer`:**
```python
class CodeReviewer(Protocol):
    # ... existing methods ...

    def overrides_disabled_setting(self) -> bool:
        """Return True if this reviewer should run even when review is disabled.

        DefaultReviewer returns False (respects disabled settings).
        Custom/injected reviewers return True (user explicitly provided them).
        """
        ...
```

**New factory function in `factory.py`:**
```python
def create_issue_provider(
    repo_path: Path,
    log_warning: Callable[[str], None] | None = None
) -> IssueProvider:
    """Create an IssueProvider for the given repo.

    Used by CLI for dry-run mode instead of importing BeadsClient directly.
    """
```

---

### Contract 1: Infra hooks do not import clients, IO, or telemetry

**Current violation chain:**
```
src.infra.hooks.locking -> src.infra.tools.locking -> src.infra.tools.env -> src.infra.io.config
```

**Root cause:** `EnvConfig.find_cerberus_bin_path()` in `env.py` imports `_find_cerberus_bin_path` from `config.py` (IO layer).

**Solution:**
Create a new `src/infra/tools/cerberus.py` module and move `_find_cerberus_bin_path` there. This provides better separation of concerns and makes the cerberus-specific logic easier to find.

**Dependency direction note:** This creates an IO → tools dependency (`config.py` imports from `tools/cerberus.py`). This is acceptable because:
- `tools/` is a leaf layer containing pure utility functions with no internal dependencies
- `io/` is a higher layer that handles file/config operations
- The direction IO → tools is consistent with tools being lower-level utilities
- This does NOT create a cycle: hooks → tools → (nothing), io → tools → (nothing)

**Implementation steps:**
1. Create `src/infra/tools/cerberus.py` with `find_cerberus_bin_path()` function (extracted from config.py)
2. Update `src/infra/tools/env.py` `EnvConfig.find_cerberus_bin_path()` to import from tools.cerberus directly
3. Update `src/infra/io/config.py` to import from `src.infra.tools.cerberus` (for any remaining usages)
4. Remove `_find_cerberus_bin_path` from `src/infra/io/config.py`
5. Verify no circular imports: run `python -c "from src.infra.io.config import MalaConfig"`

**Verification:** Uncomment contract, run `uv run lint-imports`, expect KEPT.

---

### Contract 2: Only orchestration.factory imports infra.clients

**Violation chains (when contract is enabled):**
1. `src.orchestration.cli_support -> src.infra.clients.beads_client`
2. `src.orchestration.orchestration_wiring -> src.infra.epic_verifier -> src.infra.clients.beads_client`
3. `src.orchestration.orchestrator -> src.infra.clients.cerberus_review`
4. `src.orchestration.orchestrator -> src.infra.epic_verifier -> src.infra.clients.beads_client`

**Solutions:**

**2a. cli_support.py BeadsClient export:**
Remove `BeadsClient` re-export from `cli_support.py`. Add a factory helper `create_issue_provider()` to `orchestration.factory.py` that CLI can call for dry-run.

**Implementation steps:**
1. Add `create_issue_provider(repo_path, log_warning=None) -> IssueProvider` to `src/orchestration/factory.py`
2. Update `src/cli/cli.py` to use `create_issue_provider()` instead of lazy-loading `BeadsClient`
3. Remove `BeadsClient` import and export from `src/orchestration/cli_support.py`
4. Update test mocks in `tests/unit/cli/test_cli.py` to patch the factory function instead of `cli_support.BeadsClient`

**2b. orchestrator.py DefaultReviewer import:**
Line 27: `from src.infra.clients.cerberus_review import DefaultReviewer`
Used in `_is_review_enabled()` to check if reviewer is the default.

Add a protocol method `overrides_disabled_setting()` to `CodeReviewer` protocol. This replaces the `isinstance(reviewer, DefaultReviewer)` check.

**Semantic meaning:**
- `DefaultReviewer` returns `False` — it's the production reviewer that respects the `--disable review` flag
- Test fakes and custom injected reviewers return `True` — if a user explicitly provides a reviewer, they want it to run even if review is "disabled"

**Note:** `DefaultReviewer` IS the actual Cerberus CLI reviewer (not a no-op). The name reflects that it's the default/production implementation.

**Implementation steps:**
1. Add `overrides_disabled_setting() -> bool` method to `CodeReviewer` protocol in `src/core/protocols.py`
2. Implement in `DefaultReviewer` (returns `False`) — respects disabled settings
3. **Search for ALL CodeReviewer implementations** using `grep -r "CodeReviewer" tests/` and update each:
   - `tests/unit/pipeline/test_review_runner.py` — `FakeCodeReviewer` should return `True`
   - `tests/conftest.py` — any fixtures implementing CodeReviewer should return `True`
   - `tests/unit/orchestration/test_orchestrator.py` — any test reviewers should return `True`
4. Update `_is_review_enabled()` in orchestrator:
   ```python
   # Before: not isinstance(self.review_runner.code_reviewer, DefaultReviewer)
   # After:  self.review_runner.code_reviewer.overrides_disabled_setting()
   ```
5. Remove `DefaultReviewer` import from orchestrator

**2c. epic_verifier imports:**
Both `orchestration_wiring.py` and `orchestrator.py` import from `src.infra.epic_verifier`, creating transitive chains to `infra.clients.beads_client`. Import-linter tracks TYPE_CHECKING imports.

- `orchestration_wiring.py` imports `EpicVerificationResult` — already exists in `src.core.models`
- `orchestrator.py` imports `EpicVerifier` (the class) — needs a new protocol

**Implementation steps:**
1. Create `EpicVerifierProtocol` in `src/core/protocols.py` with `verify_and_close_epic` method signature
2. Update `src/orchestration/orchestrator.py` to import `EpicVerifierProtocol` from `core.protocols` instead of `EpicVerifier` from `infra.epic_verifier`
3. Update `src/orchestration/orchestration_wiring.py` to import `EpicVerificationResult` from `src.core.models` instead of `src.infra.epic_verifier`

**Verification:** Uncomment contract, run `uv run lint-imports`, expect KEPT.

---

### Contract 3: Hooks do not import agent runtime

**Status: VERIFIED PASSING** ✓

Tested by temporarily uncommenting the contract and running `uv run lint-imports`. Result: KEPT.

**Implementation:** Simply uncomment the contract in pyproject.toml.

---

### Contract 4: No claude_agent_sdk outside infra SDK boundary

**Critical insight:** With `allow_indirect_imports = false` (the default), ANY transitive path from source_modules to `claude_agent_sdk` is a violation. This means `orchestration.factory -> infra.clients.braintrust_integration -> claude_agent_sdk` would fail even though the SDK import is in infra.

**Solution:** Follow the pattern used by existing `braintrust` and `anthropic` contracts (pyproject.toml lines 512-531):
- Set `allow_indirect_imports = true` to only catch **direct** imports
- Use `ignore_imports` to allow specific modules within the SDK boundary

**Revised contract definition:**
```toml
[[tool.importlinter.contracts]]
name = "No claude_agent_sdk outside infra SDK boundary"
type = "forbidden"
source_modules = ["src"]
forbidden_modules = ["claude_agent_sdk"]
allow_indirect_imports = true
ignore_imports = [
    "src.infra.clients.** -> claude_agent_sdk",
    "src.infra.sdk_adapter -> claude_agent_sdk",
    "src.infra.epic_verifier -> claude_agent_sdk",
    "src.infra.tools.locking_mcp -> claude_agent_sdk",
    "src.infra.agent_runtime.** -> claude_agent_sdk",
]
```

**Why this works:**
- `allow_indirect_imports = true` means orchestration can import infra modules that import SDK (transitive allowed)
- `ignore_imports` explicitly permits the infra modules within the SDK boundary to import SDK directly
- Any **direct** import of `claude_agent_sdk` from outside the boundary (e.g., `cli.cli -> claude_agent_sdk`) would fail
- This matches the existing patterns for `braintrust` and `anthropic` SDKs

**Tradeoff:** This contract prevents **direct** SDK imports outside the boundary but intentionally allows **transitive** dependencies through infra. For example, `cli -> orchestration -> infra.sdk_adapter -> claude_agent_sdk` is allowed because the SDK import is within the boundary. This is the same tradeoff made for braintrust/anthropic SDKs and provides practical enforcement while keeping the codebase buildable.

**Implementation steps:**
1. Modify the commented contract definition in pyproject.toml to use the revised pattern above
2. Uncomment the modified contract
3. Verify with `uv run lint-imports`

**No code changes required** — the contract definition change is sufficient.

**Verification:** Uncomment modified contract, run `uv run lint-imports`, expect KEPT.

---

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/infra/tools/cerberus.py` | **New** | Create with `find_cerberus_bin_path` function (extracted from config.py) |
| `src/infra/io/config.py` | Exists | Remove `_find_cerberus_bin_path`, import from tools.cerberus |
| `src/infra/tools/env.py` | Exists | Update import to use tools.cerberus directly |
| `src/orchestration/cli_support.py` | Exists | Remove BeadsClient import/export |
| `src/orchestration/orchestrator.py` | Exists | Remove DefaultReviewer import, use protocol method; replace EpicVerifier with EpicVerifierProtocol |
| `src/orchestration/orchestration_wiring.py` | Exists | Import EpicVerificationResult from core.models |
| `src/orchestration/factory.py` | Exists | Add `create_issue_provider()` helper |
| `src/core/protocols.py` | Exists | Add `overrides_disabled_setting()` to CodeReviewer protocol; add `EpicVerifierProtocol` |
| `src/infra/clients/cerberus_review.py` | Exists | Implement `overrides_disabled_setting()` returning `False` for DefaultReviewer |
| `src/cli/cli.py` | Exists | Use factory instead of BeadsClient directly |
| `tests/unit/cli/test_cli.py` | Exists | Update mocks to patch factory function |
| `tests/**/*` | Exists | Update all CodeReviewer test fakes to include `overrides_disabled_setting()` returning `True` |
| `pyproject.toml` | Exists | Uncomment 3 contracts as-is; modify and uncomment Contract 4 |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes
- **Circular imports**: Moving code between modules may introduce new circular import issues. Each change should be tested individually with `uv run lint-imports`.
- **TYPE_CHECKING imports**: These are tracked by import-linter; must be removed or redirected to allowed sources.
- **Test fake completeness**: All test fakes implementing `CodeReviewer` must be updated with `is_review_available()`.

### Breaking Changes & Compatibility
- **Potential Breaking Changes**:
  - Removing `BeadsClient` from `cli_support.py` may break external callers that import from there
- **Mitigations**:
  - This is an internal module; external users should use the CLI or factory directly
  - Per CLAUDE.md: no backwards-compatibility shims allowed - all callers must be updated

## Testing & Validation Strategy

- **Unit Tests**
  - Update `tests/unit/cli/test_cli.py` to mock `create_issue_provider` instead of `BeadsClient`
  - Verify existing tests pass with new protocol methods
  - Add test for `overrides_disabled_setting()` protocol method
  - Update all CodeReviewer test fakes to return `True` from `overrides_disabled_setting()`

- **Integration / End-to-End Tests**
  - Run `uv run pytest -m integration` for cross-layer validation
  - Verify orchestrator still works with DI changes

- **Regression Tests**
  - Full test suite must pass: `uv run pytest`
  - Coverage threshold of 72% must be maintained

- **Manual Verification**
  - Ensure `mala` CLI still runs correctly end-to-end
  - Test dry-run mode uses factory correctly

- **Monitoring / Observability**
  - N/A - no runtime changes, only import structure changes

### Acceptance Criteria Coverage
| Contract | Verification Method |
|----------|---------------------|
| Hooks do not import IO | Uncomment contract, run `uv run lint-imports`, shows KEPT |
| Only factory imports clients | Uncomment contract, run `uv run lint-imports`, shows KEPT |
| Hooks do not import agent runtime | Uncomment contract (already verified), shows KEPT |
| No SDK outside boundary | Modify contract definition, uncomment, run `uv run lint-imports`, shows KEPT |

### Incremental Validation Approach
1. **Contract 1**: Create cerberus.py, update imports, uncomment contract, verify KEPT
2. **Contract 3**: Uncomment contract (already verified passing)
3. **Contract 2**: Add protocol method, factory helper, update tests, uncomment contract, verify KEPT
4. **Contract 4**: Modify contract definition to use `allow_indirect_imports = true` pattern, uncomment, verify KEPT
5. Run full lint-imports with all 4 contracts uncommented
6. Run full test suite

## Open Questions

None — all key design decisions resolved:
- Contract 4 uses `allow_indirect_imports = true` pattern (matching braintrust/anthropic contracts)
- Contract 3 verified as already passing
- Test fake updates explicitly called out

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
