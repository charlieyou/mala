# Import Linter Restructure Proposal

**Date**: 2024-12-31
**Status**: Proposal (Revision 4 - No Shims)

> **Revision 4 Update**: This revision removes all shim/re-export references. All migrations use direct import updates per CLAUDE.md rules: "No backward-compatibility shims" and "No re-exports".

---

## Context & Goals

### Problem Statement

The current import-linter configuration requires manually listing modules in 15 contracts. When new modules are added, developers must remember to update multiple `forbidden_modules` and `source_modules` lists. This is error-prone and doesn't scale.

**Current maintenance burden:**
- Adding a new domain module requires updating ~5 contracts
- Adding a new infra module requires updating ~8 contracts
- Forgetting to update a list silently weakens architectural enforcement

### Goals

1. **Reduce maintenance burden**: Adding a module should require updating one location, not multiple contracts
2. **Preserve architectural constraints**: All existing rules must be maintained during transition
3. **Enable future simplification**: Position codebase for simplified layer-based contracts
4. **Zero-downtime migration**: No broken imports at any point during transition

### Current Codebase Structure

```
src/
├── 24 top-level .py modules (flat structure)
├── hooks/
├── log_output/
├── pipeline/
├── prompts/
├── tools/
└── validation/
```

15 import-linter contracts with hand-maintained module lists (see pyproject.toml lines 128-584).

---

## Scope & Non-Goals

### In Scope

- ~~Phase 1: Generator script~~ (Cancelled - manual updates sufficient)
- Phase 2: Gradual package migration with direct import updates (no shims)
- Phase 3: Simplified layer-based contracts

### Non-Goals

- Changing the architectural boundaries themselves (only the enforcement mechanism)
- Adding new modules or features during this restructure
- Modifying external API surfaces

---

## Assumptions & Constraints

### Assumptions

1. All existing imports are valid and pass current contracts
2. External tools (IDEs, CI) handle import path changes correctly
3. The team prefers incremental migration over big-bang rewrites

### Constraints

1. Each phase must be independently deployable
2. **No shims or re-exports**: All imports must be updated directly when modules move
3. No changes to public module interfaces

---

## Prerequisites

Before starting Phase 1:

1. **Tooling setup**:
   - Ensure `import-linter` is in dev dependencies (already present)
   - Verify pre-commit is configured (already present)

2. **Baseline verification**:
   - Run `uv run import-linter` to confirm all 15 contracts pass
   - Save current contract output as baseline for comparison

3. **Generator script location**:
   - Create `scripts/` directory at repo root (distinct from `src/tools/` Python package)
   - This avoids confusion with the existing `src/tools/` package

---

## High-Level Approach

**Option C: Hybrid Approach** - A three-phase strategy that solves the maintenance problem incrementally with minimal risk.

| Approach | Maintenance | Risk | Effort |
|----------|-------------|------|--------|
| **A: Full restructure now** | Best long-term | High (all imports break) | Large |
| **B: Only src/core/** | Marginal improvement | Low | Small |
| **C: Hybrid (proposed)** | Good, improves over time | Low (incremental) | Medium |

---

## Detailed Tasks

### Phase 1: Contract Generator Script

**Goal**: Automate contract generation to eliminate manual list maintenance.

**Dependencies**: None (can start immediately)

#### Task 1.1: Create Generator Script

Create `scripts/generate_import_contracts.py` that:

1. **Defines module→layer mapping** in a single data structure:

   ```python
   LAYER_MAPPING = {
       "cli": ["cli", "main"],
       "orchestration": ["orchestrator", "orchestrator_factory", "orchestrator_types", "cli_support"],
       "pipeline": ["pipeline"],
       "domain": ["lifecycle", "quality_gate", "validation", "prompts"],
       "infra": [
           "anthropic_client", "beads_client", "braintrust_integration", "cerberus_review",
           "config", "event_sink", "git_utils", "log_output", "mcp",
           "session_log_parser", "telemetry", "tools", "epic_verifier", "issue_manager",
       ],
       "core": ["models", "protocols", "log_events"],
   }

   # Special-case modules with unique constraints
   SPECIAL_CONSTRAINTS = {
       "hooks": {
           # Hooks are in infra but have additional restrictions:
           # Cannot import from pipeline, domain, or orchestration
           "layer": "infra",
           "additional_forbidden": ["pipeline", "domain", "orchestration"],
       },
       "telemetry": {
           # Telemetry must not import braintrust (abstraction purity)
           "layer": "infra",
           "additional_forbidden": ["braintrust_integration", "braintrust"],
       },
       "validation.result": {
           # Validation result has specific forbidden modules
           "layer": "domain",
           "additional_forbidden": ["tools", "config", "log_output"],
       },
   }

   # Leaf modules that cannot import anything except stdlib
   LEAF_MODULES = ["models", "protocols", "log_events"]

   # Independence contracts (modules that must be acyclic)
   INDEPENDENCE_CONTRACTS = {
       "Pipeline modules acyclic": [
           "src.pipeline.agent_session_runner",
           "src.pipeline.gate_runner",
           "src.pipeline.review_runner",
           "src.pipeline.run_coordinator",
       ],
   }
   ```

2. **Generates all contract types**:
   - Standard layer-based forbidden imports
   - Special constraint contracts (hooks, telemetry, validation.result)
   - Leaf module contracts
   - Independence contracts

3. **Outputs TOML** to stdout or updates pyproject.toml in-place

**Concrete Changes**:
- Create `scripts/generate_import_contracts.py` (~200-300 lines)
- Add `generate-contracts` script to pyproject.toml scripts section

#### Task 1.2: Integrate with CI/Pre-commit

**Option A (Recommended): CI validation**
- Add CI job that runs generator and diffs against committed contracts
- Fails if generated output differs from committed version
- Developers must run generator locally and commit changes

**Option B: Pre-commit hook**
- Add pre-commit hook that regenerates contracts on commit
- Auto-stages changes if contracts are modified

**Concrete Changes**:
- Add `.github/workflows/validate-contracts.yml` or update existing workflow
- OR add entry to `.pre-commit-config.yaml`

#### Task 1.3: Verification

**Pre-task verification**:
```bash
uv run import-linter  # Save output as baseline
```

**Post-task verification**:
```bash
# Generate new contracts
uv run python scripts/generate_import_contracts.py > /tmp/generated.toml

# Compare with current (should be semantically equivalent)
diff <(uv run import-linter 2>&1) <(uv run import-linter 2>&1)  # Both should pass

# Run full test suite
uv run pytest -m "unit or integration"

# Verify no regressions in contract enforcement
uv run import-linter
```

#### Task 1.4: Generator Script Lifecycle

The generator script serves different purposes across phases:

- **Phase 1-2**: Generates current complex configuration (15 contracts)
- **Phase 2 (during migration)**: Updated to handle both old and new import paths
- **Phase 3 (post-restructure)**: Either:
  - Simplified to generate only 3-5 layer contracts, OR
  - Retired if manual maintenance of 3-5 contracts is acceptable

This decision will be made at the start of Phase 3 based on team preference.

#### Phase 1 Rollback

If Phase 1 causes issues:
1. Delete `scripts/generate_import_contracts.py`
2. Remove CI job / pre-commit hook entry
3. Contracts in pyproject.toml remain unchanged (generator only validates, doesn't modify)

---

### Phase 2: Gradual Package Migration

**Goal**: Reorganize flat modules into layer-based packages with direct import updates.

**Dependencies**: None (Phase 1 cancelled)

#### Important: Direct Import Update Strategy

**No shims or re-exports.** When moving modules:
1. Move the file to the new location
2. Update ALL imports across the codebase to use the new path
3. Delete the old file/directory entirely
4. Update import-linter contracts

This is a pure refactor - each migration PR updates all imports atomically.

#### Task 2a: Create `src/core/` Package

**Concrete Changes**:
1. Create `src/core/__init__.py`
2. Move `src/models.py` → `src/core/models.py`
3. Move `src/protocols.py` → `src/core/protocols.py`
4. Move `src/log_events.py` → `src/core/log_events.py`
5. Update all imports across codebase (use grep/sed or IDE refactor)
6. Delete old files at root level
7. Update contracts to use new paths

**Verification**:
```bash
uv run pytest -m "unit or integration"
uv run import-linter
uvx ruff check .
uvx ty check
# Smoke test CLI
uv run mala --help
```

**Rollback**:
1. Revert commit (git revert)

#### Task 2b: Create `src/infra/` Package

**Concrete Changes**:
1. Create directory structure:
   ```
   src/infra/
   ├── __init__.py
   ├── clients/
   │   ├── __init__.py
   │   ├── anthropic_client.py
   │   ├── beads_client.py
   │   ├── braintrust_integration.py
   │   └── cerberus_review.py
   ├── io/
   │   ├── __init__.py
   │   ├── config.py
   │   ├── event_sink.py
   │   └── session_log_parser.py
   ├── git_utils.py
   ├── mcp.py
   ├── telemetry.py
   ├── epic_verifier.py
   └── issue_manager.py
   ```
2. Move existing packages (update all imports, delete old directories):
   - `src/tools/` → `src/infra/tools/`
   - `src/log_output/` → `src/infra/io/log_output/`
   - `src/hooks/` → `src/infra/hooks/`
3. Move standalone modules (update all imports, delete old files)
4. Update contracts

**Verification**: Same as 2a

**Rollback**: Same pattern as 2a

#### Task 2c: Create `src/domain/` Package

**Concrete Changes**:
1. Create directory structure:
   ```
   src/domain/
   ├── __init__.py
   ├── lifecycle.py
   ├── quality_gate.py
   └── prompts.py
   ```
2. Move `src/validation/` → `src/domain/validation/` (update all imports, delete old directory)
3. Move `src/prompts.py` → `src/domain/prompts.py` (update all imports, delete old file)
4. Move standalone modules (update all imports, delete old files)
5. Update contracts

**Verification**: Same as 2a

**Rollback**: Same pattern as 2a

#### Task 2d: Create `src/orchestration/` Package

**Concrete Changes**:
1. Create directory structure:
   ```
   src/orchestration/
   ├── __init__.py
   ├── orchestrator.py
   ├── factory.py      # from orchestrator_factory.py
   ├── types.py        # from orchestrator_types.py
   └── cli_support.py
   ```
2. Update all imports across codebase
3. Delete old files at root level
4. Update contracts

**Verification**: Same as 2a

**Rollback**: Same pattern as 2a

#### Task 2e: Create `src/cli/` Package

**Concrete Changes**:
1. Create directory structure:
   ```
   src/cli/
   ├── __init__.py
   ├── main.py
   └── cli.py
   ```
2. Update all imports across codebase
3. Delete old files at root level
4. Update contracts

**Verification**: Same as 2a

**Rollback**: Same pattern as 2a

---

### Phase 3: Simplified Contracts

**Goal**: Replace 15 complex contracts with 10 cleaner contracts (1 layers + 9 forbidden/independence).

**Dependencies**: Phase 2 fully complete, all tests passing

#### Preserved Constraints

The simplified contracts MUST preserve these nuanced constraints from current contracts:

| Current Contract | Constraint | How Preserved |
|-----------------|------------|---------------|
| Contract 4 | Pipeline modules acyclic (independence) | Keep as separate contract |
| Contract 7 | Hooks isolated from domain/pipeline/orchestration | Keep as separate forbidden contract |
| Contract 12 | Telemetry cannot import braintrust | Keep as separate forbidden contract |
| Contract 14 | validation.result cannot import tools/config/log_output | Keep as separate forbidden contract (layers would allow domain→infra) |
| Contracts 9-11 | Leaf modules (models, protocols, log_events) cannot import other src modules | Keep as separate forbidden contracts (layers only enforce direction, not isolation) |

#### Dropped/Simplified Constraints

These constraints become redundant with package-based layers:

| Current Contract | Why Redundant |
|-----------------|---------------|
| Contracts 13, 15 | Foundational modules (tools, config) - enforced by being in `src.infra` below domain |
| Contract 1 | CLI only depends on orchestrator - enforced by layer order |
| Contract 2 | Domain layer independence - enforced by layer order |
| Contract 3 | Pipeline independence - enforced by layer order |
| Contract 8 | Infra independence - enforced by layer order |

#### Task 3.1: Implement Simplified Contracts

**Concrete Changes** to pyproject.toml:

```toml
[tool.importlinter]
root_packages = ["src"]
include_external_packages = true

# Contract 1: Layered Architecture (replaces most current contracts)
[[tool.importlinter.contracts]]
name = "Layered Architecture"
type = "layers"
layers = [
    "src.cli",
    "src.orchestration",
    "src.pipeline",
    "src.domain",
    "src.infra",
    "src.core",
]

# Contract 2: SDK confined to infra (external package restriction)
[[tool.importlinter.contracts]]
name = "SDK confined to infra"
type = "forbidden"
source_modules = ["src.cli", "src.orchestration", "src.pipeline", "src.domain", "src.core"]
forbidden_modules = ["anthropic", "braintrust"]

# Contract 3: Only CLI imports typer
[[tool.importlinter.contracts]]
name = "Only CLI imports typer"
type = "forbidden"
source_modules = ["src.orchestration", "src.pipeline", "src.domain", "src.infra", "src.core"]
forbidden_modules = ["typer"]

# Contract 4: Pipeline modules remain acyclic (independence preserved)
[[tool.importlinter.contracts]]
name = "Pipeline modules acyclic"
type = "independence"
modules = [
    "src.pipeline.agent_session_runner",
    "src.pipeline.gate_runner",
    "src.pipeline.review_runner",
    "src.pipeline.run_coordinator",
]

# Contract 5: Hooks isolation (special constraint preserved)
[[tool.importlinter.contracts]]
name = "Hooks isolated"
type = "forbidden"
source_modules = ["src.infra.hooks"]
forbidden_modules = [
    "src.pipeline",
    "src.domain",
    "src.orchestration",
]

# Contract 6: Telemetry abstraction purity (special constraint preserved)
[[tool.importlinter.contracts]]
name = "Telemetry abstraction pure"
type = "forbidden"
source_modules = ["src.infra.telemetry"]
forbidden_modules = [
    "src.infra.clients.braintrust_integration",
    "braintrust",
]

# Contract 7: Validation result isolation (preserved from Contract 14)
# Layers would allow domain→infra, but validation.result needs stricter isolation
[[tool.importlinter.contracts]]
name = "Validation result minimal"
type = "forbidden"
source_modules = ["src.domain.validation.result"]
forbidden_modules = [
    "src.infra.tools",
    "src.infra.io.config",
    "src.infra.io.log_output",
]
allow_indirect_imports = true

# Contract 8: Core models is leaf (preserved from Contract 9)
# Models should only contain data classes with no internal dependencies
[[tool.importlinter.contracts]]
name = "Models is leaf"
type = "forbidden"
source_modules = ["src.core.models"]
forbidden_modules = [
    "src.cli",
    "src.orchestration",
    "src.pipeline",
    "src.domain",
    "src.infra",
    "src.core.protocols",
    "src.core.log_events",
]

# Contract 9: Core protocols is leaf (preserved from Contract 10)
# Protocols define interfaces only - no implementations
[[tool.importlinter.contracts]]
name = "Protocols is leaf"
type = "forbidden"
source_modules = ["src.core.protocols"]
forbidden_modules = [
    "src.cli",
    "src.orchestration",
    "src.pipeline",
    "src.domain",
    "src.infra",
    "src.core.models",
    "src.core.log_events",
]

# Contract 10: Core log_events is leaf (preserved from Contract 11)
# Log events are pure data structures
[[tool.importlinter.contracts]]
name = "Log events is leaf"
type = "forbidden"
source_modules = ["src.core.log_events"]
forbidden_modules = [
    "src.cli",
    "src.orchestration",
    "src.pipeline",
    "src.domain",
    "src.infra",
    "src.core.models",
    "src.core.protocols",
]
```

**Verification**:
```bash
uv run import-linter  # All 10 contracts should pass
uv run pytest -m "unit or integration"
```

#### Task 3.2: Update/Retire Generator Script

Based on team decision:
- **If keeping generator**: Update to generate simplified contracts
- **If retiring**: Remove from CI/pre-commit, keep in scripts/ as documentation

#### Phase 3 Rollback

1. Restore original 15 contracts from git history or backup
2. Regenerate using Phase 1 generator if needed
3. Run `uv run import-linter` to verify

---

## Risks, Edge Cases & Breaking Changes

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Missed import updates | Medium | High | Use grep/IDE refactor to find all occurrences before deleting old paths |
| Circular imports introduced | Low | High | import-linter catches these |
| Missed special constraints in Phase 3 | Medium | Medium | Explicit constraint mapping table above |

### Edge Cases

1. **Dynamic imports**: `importlib.import_module("src.models")` must be updated to new paths
2. **Type-checking imports**: `if TYPE_CHECKING:` blocks must be updated for new paths
3. **String-based imports**: Search for quoted module paths in config files, tests

### Breaking Changes

- **Each phase is a breaking change** for any code using old import paths
- Mitigated by updating all imports atomically in the same PR

---

## Testing & Validation

### Per-Phase Testing

Each phase must pass before proceeding:

```bash
# Standard verification (all phases)
uv run import-linter                    # All contracts pass
uv run pytest -m "unit or integration"  # All tests pass
uvx ruff check .                        # No lint errors
uvx ty check                            # No type errors
uv run mala --help                      # CLI smoke test
```

### Risky Area Coverage

| Area | Risk | Test Coverage |
|------|------|---------------|
| Moved packages (src.infra.tools, src.infra.hooks, src.domain.validation) | Medium | Existing tests use new import paths |
| Circular import detection | Medium | import-linter catches these |
| Dynamic/string imports | Medium | Grep for old paths before each migration |

### End-to-End Validation

After Phase 2 completion:
```bash
uv run pytest -m e2e  # Full end-to-end tests
```

---

## Plan-Level Rollback Strategy

### Emergency Rollback (Any Phase)

If critical issues occur mid-phase:
```bash
git revert <commit>  # Revert the specific phase commit
uv run import-linter  # Verify contracts still pass
uv run pytest -m "unit or integration"  # Verify tests pass
```

### Staged Rollback

If issues discovered post-merge:

1. **Phase 1**: ~~Delete generator script and CI job~~ (cancelled)
2. **Phase 2**: Revert the migration commit(s) - all imports restored atomically
3. **Phase 3**: Restore original 15 contracts from git history

### Rollback Timeline

- **Phase 1**: Instant rollback (no code changes)
- **Phase 2**: ~1 hour per sub-phase to revert
- **Phase 3**: ~30 minutes to restore old contracts

---

## Open Questions

1. ~~**CI vs pre-commit for generator**~~: Phase 1 cancelled
2. ~~**Shim duration**~~: No shims - direct import updates only
3. **Phase 3 timing**: Proceed immediately after Phase 2, or wait for stabilization period?

---

## Implementation Timeline

| Phase | Scope | Dependencies |
|-------|-------|--------------|
| ~~1~~ | ~~Generator script + CI integration~~ | Cancelled |
| 2a | Create `src/core/` | None |
| 2b | Create `src/infra/` | Phase 2a |
| 2c | Create `src/domain/` | Phase 2b |
| 2d | Create `src/orchestration/` | Phase 2c |
| 2e | Create `src/cli/` | Phase 2d |
| 3 | Simplify contracts | Phase 2e + stabilization |

Phases can be done as independent PRs. Each PR must pass all verification steps before merge.

---

## Decision

- [x] ~~Proceed with Phase 1 (generator script)~~ Cancelled
- [x] Proceed with Phase 2a-2e (package migrations with direct import updates)
- [ ] Proceed with Phase 3 (simplified contracts)
- [ ] Defer restructuring

---

## References

- Current contracts: [pyproject.toml](../pyproject.toml) (lines 128-584)
- Architecture doc: [architecture-review.md](./architecture-review.md)
