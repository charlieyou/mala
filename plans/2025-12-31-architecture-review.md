# Architecture Review

**Date:** 2024-12-31  
**Tools:** lizard (complexity), import-linter (dependencies)

## Executive Summary

The mala codebase has a functional but flat architecture with 32 modules at the top level of `src/`. Static analysis reveals several high-complexity functions and unclear layer boundaries. This document captures the findings and provides a roadmap for incremental refactoring.

## Complexity Hotspots

Functions with cyclomatic complexity (CCN) > 15, sorted by severity:

| Function | CCN | Lines | File | Issue |
|----------|-----|-------|------|-------|
| `run_session` | 67 | 452 | pipeline/agent_session_runner.py | Monolithic event loop with many effect branches |
| `_run_refresh` | 54 | 233 | validation/coverage.py | Mixes worktree mgmt, command rewriting, coverage parsing |
| `run` (CLI) | 28 | 332 | cli.py | Too much business logic in CLI layer |
| `_cli_main` | 27 | 122 | tools/locking.py | CLI mixed with lock management logic |
| `_init_legacy` | 25 | 174 | orchestrator.py | Dual init paths, too many responsibilities |
| `on_run_started` | 37 | 119 | event_sink_console.py | Complex formatting logic |
| `check_with_resolution` | 15 | 162 | quality_gate.py | Many resolution branches |
| `build_validation_spec` | 14 | 168 | validation/spec.py | Complex spec construction |

### Largest Files (LOC)

| File | Lines | Concern |
|------|-------|---------|
| orchestrator.py | 1352 | Mixed responsibilities, dual init |
| epic_verifier.py | 1293 | Model + impl + diffing combined |
| event_sink.py | 824 | Multiple sink types in one file |
| quality_gate.py | 817 | Evidence parsing + policy mixed |
| cli.py | 817 | Business logic leaking into CLI |
| agent_session_runner.py | 801 | Monolithic run_session |

## Current Architecture

```
src/
├── cli.py, main.py                    # CLI Layer
├── orchestrator.py, orchestrator_factory.py  # Orchestration Layer
├── pipeline/                          # Pipeline Layer
│   ├── agent_session_runner.py
│   ├── gate_runner.py
│   ├── review_runner.py
│   └── run_coordinator.py
├── lifecycle.py, quality_gate.py      # Domain Layer
├── validation/                        # Domain Layer
│   ├── spec.py, coverage.py
│   ├── worktree.py, e2e.py
│   └── ...
├── models.py, prompts.py              # Domain Layer
└── (32 other modules)                 # Infrastructure Layer
    ├── beads_client.py, cerberus_review.py
    ├── tools/, hooks/, log_output/
    ├── event_sink*.py, telemetry.py
    └── ...
```

### Intended Layering (top to bottom)

```
CLI → Orchestration → Pipeline → Domain → Infrastructure
```

Lower layers must not import from higher layers.

## Import Linter Contracts

Configuration in `pyproject.toml`. Run with:

```bash
uvx --from import-linter lint-imports
```

### Contract Status

| Contract | Status | Description |
|----------|--------|-------------|
| Layered Architecture | ❌ BROKEN | 5-layer hierarchy enforcement |
| Domain layer independence | ❌ BROKEN | lifecycle, models must be independent |
| Pipeline modules acyclic | ✅ KEPT | No circular deps in pipeline/* |
| Domain must not depend on infra | ❌ BROKEN | Keep domain pure |
| CLI only depends on orchestrator | ❌ BROKEN | No reaching into lower layers |
| Infra modules independent | ✅ KEPT | Integrations don't import each other |
| SDK confined to infra | ✅ KEPT | claude_agent_sdk, braintrust in infra only |
| Only CLI imports typer | ✅ KEPT | Framework confined to entry points |
| Hooks isolated | ✅ KEPT | Hooks don't reach into other infra |

### Key Violations to Fix

#### 1. Circular: orchestrator ↔ orchestrator_factory
```
src.orchestrator -> src.orchestrator_factory (l.76, l.83)
src.orchestrator_factory -> src.orchestrator (l.40, l.377, l.424)
```
**Fix:** Merge or extract shared types to break cycle.

#### 2. Domain importing infra
```
src.lifecycle -> src.validation.spec -> src.models
src.quality_gate -> src.validation.spec
```
**Fix:** Move `ValidationSpec` types to domain, keep builders in validation.

#### 3. Protocols importing implementations
```
src.protocols -> src.cerberus_review
src.protocols -> src.quality_gate
src.protocols -> src.validation.spec
src.protocols -> src.session_log_parser
```
**Fix:** Split protocols by domain, use TYPE_CHECKING imports.

#### 4. CLI reaching into infra
```
src.cli -> src.tools.env
src.cli -> src.tools.locking
src.cli -> src.beads_client
src.cli -> src.log_output.run_metadata
```
**Fix:** Route through orchestrator or create facade.

#### 5. Infra cross-imports
```
src.event_sink_console -> src.event_sink
src.event_sink_console -> src.log_output.console
src.telemetry -> src.braintrust_integration
src.cerberus_review -> src.log_output.console
```
**Fix:** Consolidate event sinks, inject dependencies.

## Recommended Refactoring Plan

### Phase 1: Quick Wins (Low Risk)

1. **Create `src/tools/__init__.py`** ✅ Done
2. **Split `protocols.py`** by domain:
   - `protocols/issue.py` → IssueProvider
   - `protocols/gate.py` → GateChecker
   - `protocols/review.py` → CodeReviewer
   - `protocols/log.py` → LogProvider
   - `protocols/epic.py` → EpicVerificationModel

3. **Extract shared types** from orchestrator:
   - `IssueResult` → `models.py`
   - `RetryConfig` already in models ✅

### Phase 2: Domain Purity

4. **Move `ValidationSpec` types to domain:**
   - Keep spec *types* in `validation/spec_types.py`
   - Keep spec *builders* in `validation/spec.py`
   - Domain imports only types

5. **Break lifecycle → validation dependency:**
   - `lifecycle.py` should only import `ValidationSpec` type, not builder

6. **Isolate quality_gate:**
   - Extract evidence parsing to `quality_gate/evidence.py`
   - Keep policy in `quality_gate/gate.py`

### Phase 3: Orchestrator Decomposition

7. **Resolve orchestrator ↔ factory cycle:**
   - Option A: Merge into single module
   - Option B: Extract `OrchestratorConfig` to shared types module

8. **Extract collaborators from MalaOrchestrator:**
   - `RunPlanner` – issue selection, dry-run ordering
   - `IssueRunExecutor` – drives session/gate/review per issue
   - `RunMetadataRecorder` – writes IssueRun, markers

### Phase 4: High-Complexity Refactors

9. **Refactor `run_session` (CCN=67):**
   - Introduce effect handlers per `Effect` type
   - Each handler ~40 lines, single responsibility
   - Main loop becomes: `while not terminal: effect = await handlers[effect](...)`

10. **Extract `BaselineRefresher` from coverage.py:**
    - `_run_refresh` → `BaselineRefresher.refresh()`
    - Split into: `_build_env`, `_run_uv_sync`, `_normalize_pytest_cmd`, `_ensure_coverage_xml`

11. **Thin CLI layer:**
    - Move `run()` business logic to `cli/run_command.py`
    - Keep Typer function as thin wrapper

### Phase 5: Package Restructure (Optional)

```
src/
├── cli/           # CLI entry points only
├── orchestration/ # Orchestrator + factory + run planner
├── pipeline/      # Session/gate/review runners
├── domain/        # lifecycle, quality_gate, validation, models
└── infra/         # All integrations, tools, event sinks
```

## Strengths to Preserve

- **Strong DI via protocols:** `IssueProvider`, `GateChecker`, `CodeReviewer`, `SDKClientProtocol`
- **Pure lifecycle state machine:** `ImplementerLifecycle` in `lifecycle.py`
- **Pipeline extraction:** `pipeline/*` already partially separated
- **Strict typing:** ty + ruff with all rules at error level
- **85% coverage threshold:** Enforced at quality gate

## Running the Linter

```bash
# Check all contracts
uvx --from import-linter lint-imports

# Verbose output with import chains
uvx --from import-linter lint-imports --verbose

# Check specific contract
uvx --from import-linter lint-imports --contract "Layered Architecture"
```

## References

- [import-linter docs](https://import-linter.readthedocs.io/)
- [lizard complexity analyzer](https://github.com/terryyin/lizard)
- [pyproject.toml](../pyproject.toml) – Contract definitions
