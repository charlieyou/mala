# CLI Redesign: `mala run` Command

## Context & Goals

This plan redesigns the `mala run` CLI to improve discoverability, reduce flag confusion, and fix ordering bugs.

**Goals:**
- Replace 5 interacting flags (`--wip`, `--focus`, `--epic`, `--only`, `--orphans-only`) with 3 orthogonal knobs
- Fix bug where `--only` loses input order by converting to a `set`
- Group options logically using `rich_help_panel`
- Support repeatable flags instead of comma-separated strings
- Rename internal "cerberus" naming to user-facing "review"

## Problem Statement

The current `mala run` command has 20 flat options that are hard to scan, with complex interactions that aren't obvious to users:

1. **Flat structure** - all options dumped together in `--help`
2. **Comma-separated strings** - `--only`, `--disable-validations`, `--epic-override` should be repeatable
3. **Missing short flags** - common options like `--dry-run`, `--wip` lack shortcuts
4. **Implementation leakage** - "cerberus" naming exposes internal details
5. **Interacting flags** - `--wip`, `--focus`, `--epic`, `--only`, `--orphans-only` have non-obvious interactions
6. **Bug**: `--only` loses input order by converting to a `set`

## Current Behavior Analysis

### Filtering Pipeline
```
bd ready -t task
    → (if --wip) merge in_progress issues, drop blocked WIP
    → exclude previously failed IDs
    → --epic: filter to descendants (empty = no issues)
    → --only: filter to specific IDs (intersection with above)
    → enrich with parent_epic
    → drop issues under blocked epics
    → --orphans-only: keep only issues without parent_epic
```

### Ordering Logic
- `--focus` (default): group by parent_epic, order groups by (min priority, max updated DESC)
- `--no-focus`: global priority-only ordering
- `--wip`: stable-sorts to put `status=in_progress` first (can pull WIP from multiple epics to front, undermining "finish one epic" intent)

### Hidden Behaviors
- Blocked WIP issues are silently dropped
- Tasks under blocked epics are silently dropped
- `--epic` + `--only` silently ANDs together (intersection)
- Exit 0 when no work found (no `--fail-on-empty` option)

---

## Proposed Design

### 1. Orthogonal Knobs (Replace 5 Interacting Flags)

Instead of `--wip`, `--focus`, `--epic`, `--only`, `--orphans-only`, use two clear dimensions:

| Knob | Values | Current Equivalent |
|------|--------|-------------------|
| `--scope` | `all` (default), `epic:<id>`, `orphans`, `ids:<id,...>` | `--epic`, `--orphans-only`, `--only` |
| `--order` | `focus` (default), `priority`, `input` | `--focus/--no-focus` |
| `--resume` | flag | `--wip` |

**Examples:**
```bash
# Default: all ready issues, grouped by epic
mala run

# Work on specific epic subtree
mala run --scope epic:E-123

# Orphan issues only, priority order
mala run --scope orphans --order priority

# Specific issues in user-specified order (NEW: preserves order)
mala run --scope ids:T-5,T-3,T-1 --order input

# Resume in-progress work first
mala run --resume
```

**Alternative**: single `--mode` shorthand that sets multiple knobs:
```bash
mala run --mode focus      # default
mala run --mode priority   # --order priority
mala run --mode resume     # --resume
mala run --mode epic E-123 # --scope epic:E-123
```

### 2. Option Groups (via `rich_help_panel`)

```
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│   repo_path      [REPO_PATH]  Path to repository [default: .]               │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Scope & Ordering ───────────────────────────────────────────────────────────╮
│ --scope       -s  TEXT   Issue scope: all|epic:<id>|orphans|ids:<id,...>    │
│ --order           TEXT   Ordering: focus|priority|input [default: focus]    │
│ --resume      -r         Include in-progress issues, prioritize them        │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Execution Limits ───────────────────────────────────────────────────────────╮
│ --max-agents  -n  INT    Maximum concurrent agents                          │
│ --timeout     -t  INT    Timeout per agent in minutes [default: 60]         │
│ --max-issues  -i  INT    Maximum issues to process                          │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Quality Gates ──────────────────────────────────────────────────────────────╮
│ --disable         TEXT   Disable validation (repeatable): post-validate,    │
│                          integration-tests, coverage, e2e, review           │
│ --coverage-threshold     FLOAT  Minimum coverage % (0-100)                  │
│ --max-gate-retries       INT    Gate retry attempts [default: 3]            │
│ --max-review-retries     INT    Review retry attempts [default: 3]          │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Epic Verification ──────────────────────────────────────────────────────────╮
│ --epic-override          TEXT   Skip verification for epic (repeatable)     │
│ --max-epic-retries       INT    Epic verification retries [default: 3]      │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Review Backend ─────────────────────────────────────────────────────────────╮
│ --review-timeout         INT    Review timeout in seconds [default: 1200]   │
│ --review-spawn-args      TEXT   Extra args for review spawn                 │
│ --review-wait-args       TEXT   Extra args for review wait                  │
│ --review-env             TEXT   Extra env for review (JSON or KEY=VALUE)    │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Debugging ──────────────────────────────────────────────────────────────────╮
│ --dry-run     -d               Preview task order without processing        │
│ --verbose     -v               Show full tool arguments                     │
│ --deadlock-detection/--no-deadlock-detection  [default: on]                 │
│ --fail-on-empty                Exit non-zero if no issues to process        │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### 3. Repeatable Options (Replace Comma-Separated)

| Before | After |
|--------|-------|
| `--only "T-1,T-2,T-3"` | `--scope ids:T-1,T-2,T-3` or repeated `--only T-1 --only T-2` |
| `--disable-validations "coverage,review"` | `--disable coverage --disable review` |
| `--epic-override "E-1,E-2"` | `--epic-override E-1 --epic-override E-2` |

### 4. Short Flags

| Option | Short | Rationale |
|--------|-------|-----------|
| `--scope` | `-s` | Primary selection mechanism |
| `--resume` | `-r` | Common operation |
| `--dry-run` | `-d` | Very common |
| `--max-agents` | `-n` | Already exists |
| `--timeout` | `-t` | Already exists |
| `--max-issues` | `-i` | Already exists |
| `--verbose` | `-v` | Already exists |
| `--epic` | `-e` | Already exists (if kept) |

### 5. Naming Changes

| Before | After | Rationale |
|--------|-------|-----------|
| `--cerberus-spawn-args` | `--review-spawn-args` | Hide implementation |
| `--cerberus-wait-args` | `--review-wait-args` | Hide implementation |
| `--cerberus-env` | `--review-env` | Hide implementation |
| `--disable-validations` | `--disable` | Shorter, repeatable |
| `--max-epic-verification-retries` | `--max-epic-retries` | Shorter |

### 6. New Features

| Option | Description |
|--------|-------------|
| `--fail-on-empty` | Exit non-zero if scope filters produce no work |
| `--order input` | For `--scope ids:...`, preserve user-specified order |
| Enhanced `--dry-run` | Show *why* issues were filtered (blocked epic, blocked WIP, etc.) |

---

## Migration Path

### Phase 1: Non-Breaking Additions
1. Add `rich_help_panel` grouping
2. Add new short flags (`-d`, `-r`, `-s`)
3. Add `--fail-on-empty`
4. Fix `--only` to preserve input order
5. Rename cerberus → review (keep old names as hidden aliases)

### Phase 2: Deprecation Warnings
1. Warn on `--wip` → suggest `--resume`
2. Warn on `--focus/--no-focus` → suggest `--order`
3. Warn on `--epic` → suggest `--scope epic:<id>`
4. Warn on `--only` → suggest `--scope ids:<ids>`
5. Warn on `--orphans-only` → suggest `--scope orphans`
6. Warn on comma-separated `--disable-validations` → suggest repeatable `--disable`

### Phase 3: Remove Deprecated Options
After 2-3 releases, remove old option names.

---

## Decisions

1. **Orthogonal knobs** (not single `--mode`) - more flexible, clearer mental model
2. **Keep `--epic`/`--only` as hidden aliases** - for backward compatibility during migration, but not documented
3. **`--order input` only valid for `--scope ids:`** - other scopes use focus/priority ordering only
4. **Conflict resolution: new flags take precedence** - mixing deprecated and new flags raises a warning, new flag wins
5. **Duplicate IDs in `--scope ids:` are deduplicated with warning** - `--scope ids:T-1,T-1,T-2` becomes `["T-1", "T-2"]` with a warning about duplicates being removed (processing an issue twice makes no sense)
6. **`--scope ids:` defaults to `--order focus`** - users must explicitly specify `--order input` to preserve input order; this maintains backward compatibility with the mental model that ordering is a separate concern
7. **Redundant deprecated+new flags emit warning** - `--wip --resume` warns "both --wip and --resume specified; --resume is preferred" (not an error, just informational)

## Scope & Non-Goals

**In Scope:**
- Implement orthogonal `--scope`, `--order`, `--resume` options
- Add `rich_help_panel` groupings to `--help`
- Fix `--only` → `--scope ids:` to preserve input order (use `list` not `set`)
- Add deprecation warnings for old flags
- Rename cerberus → review options

**Non-Goals:**
- Changing the underlying orchestration logic (filtering pipeline, ordering algorithms)
- Adding new filtering capabilities beyond what exists
- Modifying the issue provider or beads integration

## Assumptions & Constraints

- **typer >= 0.9.0** - project already requires this (confirmed in pyproject.toml), supports `rich_help_panel`
- **Backward compatibility required during migration** - existing scripts using old flags must continue working with warnings
- **No config file changes** - all changes are CLI-only

## Prerequisites

- None - this is a pure CLI refactor with no external dependencies

---

## File Impact Summary

### CLI Layer (Primary Changes)

| File | Changes |
|------|---------|
| `src/cli/cli.py` | Main implementation: add new options, `rich_help_panel` groups, deprecation warnings, scope parsing, duplicate ID handling |
| `tests/unit/cli/test_cli.py` | Add unit tests for new scope parsing, deprecation warnings, error handling |

### Type Definitions (only_ids: set → list)

| File | Changes |
|------|---------|
| `src/orchestration/types.py` | Update `OrchestratorConfig.only_ids` and `IssueFilterConfig.only_ids` from `set[str]` to `list[str]` |
| `src/pipeline/issue_execution_coordinator.py` | Update `IssueExecutionCoordinatorConfig.only_ids` from `set[str]` to `list[str]` |
| `src/core/protocols.py` | Update any protocol definitions using `only_ids` |

### Usage Sites (convert list→set internally where needed)

| File | Changes |
|------|---------|
| `src/orchestration/run_config.py` | Update `build_event_run_config` and `build_run_metadata` signatures from `set[str]` to `list[str]` |
| `src/infra/issue_manager.py` | Update `find_missing_ids` to convert `list` to `set` internally for set subtraction |
| `src/orchestration/orchestrator.py` | Update usages of `only_ids` (may need `set()` conversion for membership checks) |
| `src/infra/clients/beads_client.py` | Update any `only_ids` usages |
| `src/orchestration/orchestration_wiring.py` | Update wiring that passes `only_ids` between layers |
| `src/infra/io/console_sink.py` | Update any `only_ids` display logic |
| `src/infra/io/log_output/run_metadata.py` | Update run metadata serialization for `only_ids` |

### No Changes Expected

| File | Reason |
|------|--------|
| `src/cli/__init__.py` | Just exports |
| `src/cli/main.py` | Just entry point |

**Internal API Changes:**
- `ValidatedRunArgs.only_ids` changes from `set[str] | None` to `list[str] | None` to preserve order
- All downstream type definitions updated to `list[str]` for consistency
- Functions that need set operations (like `find_missing_ids`) convert to `set` internally

---

## Testing & Verification Strategy

### Unit Tests (`tests/unit/cli/`)

| Test | Description |
|------|-------------|
| `test_scope_parsing_epic` | `--scope epic:E-123` parses to epic filter |
| `test_scope_parsing_ids` | `--scope ids:T-1,T-2,T-3` parses to ordered list `["T-1", "T-2", "T-3"]` |
| `test_scope_parsing_orphans` | `--scope orphans` enables orphan-only mode |
| `test_scope_parsing_all` | `--scope all` (default) processes all issues |
| `test_scope_invalid_format` | `--scope invalid:xyz` raises error with helpful message |
| `test_scope_malformed_epic` | `--scope epic:` (missing ID) raises error |
| `test_scope_empty_ids` | `--scope ids:` (empty) raises error |
| `test_order_input_requires_ids_scope` | `--order input` without `--scope ids:` raises error |
| `test_deprecation_warning_wip` | `--wip` emits deprecation warning suggesting `--resume` |
| `test_deprecation_warning_epic` | `--epic E-123` emits warning suggesting `--scope epic:E-123` |
| `test_deprecation_warning_only` | `--only T-1,T-2` emits warning suggesting `--scope ids:T-1,T-2` |
| `test_mixed_flags_new_wins` | `--epic E-1 --scope epic:E-2` uses E-2, emits conflict warning |
| `test_ids_order_preserved` | `--scope ids:T-5,T-3,T-1 --order input` → ordered list `["T-5", "T-3", "T-1"]` (bug fix verification) |
| `test_duplicate_ids_deduplicated` | `--scope ids:T-1,T-1,T-2` → `["T-1", "T-2"]` with warning about duplicates removed |
| `test_redundant_wip_resume_warns` | `--wip --resume` emits warning about redundant specification |
| `test_ids_scope_default_order_focus` | `--scope ids:T-1,T-2` without `--order` uses focus ordering (not input) |

### Integration Tests (`tests/integration/`)

| Test | Description |
|------|-------------|
| `test_help_shows_panels` | `mala run --help` output contains `Scope & Ordering`, `Execution Limits` panels |
| `test_backward_compat_epic` | `--epic E-123` still works (with warning) during migration |
| `test_backward_compat_only` | `--only T-1,T-2` still works (with warning) during migration |

### Verification Approach

1. **Before implementation**: Confirm existing tests pass
2. **During implementation**: Add tests for each new scope/order value as implemented
3. **After implementation**: Run full test suite including new tests

---

## Acceptance Criteria

| Criterion | Verification |
|-----------|--------------|
| `mala run --scope ids:T-1,T-2,T-3 --order input` processes issues in exact order T-1, T-2, T-3 | Dry-run output shows order preserved |
| `mala run --scope ids:T-1,T-2,T-3` (no `--order`) uses focus ordering | Dry-run shows focus-ordered output |
| `mala run --scope ids:T-1,T-1,T-2` deduplicates with warning | Warning mentions "duplicate IDs removed", processes T-1, T-2 |
| `mala run --scope epic:E-123` filters to only descendants of E-123 | Dry-run shows only E-123 children |
| `mala run --scope orphans` processes only issues without parent epic | Dry-run shows only orphan issues |
| `mala run --help` shows grouped options under "Scope & Ordering", "Execution Limits", etc. | Visual inspection of help output |
| `mala run --wip` emits deprecation warning mentioning `--resume` | Warning appears on stderr |
| `mala run --wip --resume` emits redundancy warning | Warning mentions both flags specified |
| `mala run --epic E-1 --scope epic:E-2` uses E-2, warns about conflict | Warning on stderr, E-2 used |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing scripts using old flags | Keep old flags as hidden aliases during migration phases |
| Type change in `only_ids` (set→list) breaks downstream code | All affected files enumerated in File Impact Summary; functions needing set operations convert internally |
| Users confused by deprecation warnings | Warnings include clear migration instructions |
| Duplicate IDs cause unexpected behavior | Deduplicate with warning at parse time |

---

## Appendix: Current Options (for reference)

```
--max-agents, -n          INT    Max concurrent agents
--timeout, -t             INT    Timeout per agent (minutes)
--max-issues, -i          INT    Max issues to process
--epic, -e                TEXT   Only children of this epic
--only, -o                TEXT   Comma-separated issue IDs
--max-gate-retries        INT    Gate retry attempts [default: 3]
--max-review-retries      INT    Review retry attempts [default: 3]
--disable-validations     TEXT   Comma-separated validations to skip
--coverage-threshold      FLOAT  Min coverage %
--wip                            Prioritize in_progress issues
--focus/--no-focus               Group by epic [default: focus]
--dry-run                        Preview without processing
--verbose, -v                    Verbose output
--review-timeout          INT    Review timeout (seconds)
--cerberus-spawn-args     TEXT   Extra spawn args
--cerberus-wait-args      TEXT   Extra wait args
--cerberus-env            TEXT   Extra env vars
--epic-override           TEXT   Skip verification for epics
--orphans-only                   Only orphan issues
--max-epic-verification-retries  INT  Epic verify retries
--deadlock-detection/--no-deadlock-detection
```
