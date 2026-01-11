# Implementation Plan: Separate Evidence Check Configuration

## Context & Goals
- **Spec**: `docs/2026-01-10-evidence-check-config-spec.md`
- Decouple evidence gate requirements from run command pool via a new `evidence_check:` config section
- Users explicitly declare which commands require evidence; no `evidence_check:` means no evidence required (breaking change)
- Enables independent control of trigger execution vs. evidence requirements

## Scope & Non-Goals
- **In Scope**
  - Add `evidence_check:` config section parsing with `required:` list
  - Validate `required` references against resolved command map (preset + project merge)
  - Filter evidence check by `evidence_check.required` instead of inferring from commands
  - Maintain scope filtering (E2E excluded from per-session gates)
  - Handle `allow_fail` semantics correctly (affects failure handling, not evidence requirement)

- **Out of Scope (Non-Goals)**
  - Changing how triggers execute commands or how the run command pool is built
  - Changing evidence detection mechanism (patterns, markers)
  - Preset-defined `evidence_check:` that projects can override (deferred)
  - Inline pattern override support in `evidence_check:`
  - Migration/deprecation warning phase (clean break per spec)

## Assumptions & Constraints
- Existing preset loading and config merging infrastructure works correctly
- Resolved command map is available during `build_validation_spec()` after preset merge
- Evidence detection patterns are derived from command strings (existing behavior, unchanged)

### Implementation Constraints
- Follow existing config parsing patterns (model after `code_review:` section parsing)
- Use existing reference validation pattern from `_validate_trigger_command_refs()`
- Validation MUST happen in `spec.py` during `build_validation_spec()` (after preset merge, not during initial YAML parsing)
- No `_fields_set` tracking needed for `EvidenceCheckConfig`; `None` vs present is sufficient
- Modify existing `check_evidence_against_spec()` rather than creating new function

### Testing Constraints
- All 13 test cases from spec MUST be automated
- Use inline fixture presets for validation tests (no external preset file dependencies)
- Unit tests for parsing, validation, and evidence filtering
- Integration tests for end-to-end evidence flow with preset merge

## Integration Analysis

### Existing Mechanisms Considered

| Existing Mechanism | Could Serve Feature? | Decision | Rationale |
|--------------------|---------------------|----------|-----------|
| `config_loader.py` parsing + `_ALLOWED_TOP_LEVEL_FIELDS` | Yes | Extend | Handles all top-level config parsing; add `evidence_check` |
| `_parse_code_review_config()` (config_loader.py:499-722) | Pattern only | Follow pattern | Same structured parsing approach for new section |
| `_validate_trigger_command_refs()` (config_loader.py:1140-1194) | Pattern only | Follow pattern | Same reference validation approach against resolved map |
| `ValidationConfig` dataclass (config.py) | Yes | Extend | Add `evidence_check` field to carry config through |
| `ValidationSpec` dataclass (spec.py) | Yes | Extend | Store resolved evidence requirements for runtime |
| `build_validation_spec()` (spec.py:356-497) | Yes | Extend | Add validation call after preset merge |
| `check_evidence_against_spec()` (evidence_check.py:207-271) | Yes | Extend | Add parameter and filtering logic |

### Integration Approach
All changes extend existing infrastructure. The new config section follows established parsing patterns, validation uses the same approach as trigger command validation, and evidence checking gains filtering logic via a new parameter. No new systems or modules are created.

## Prerequisites
- [ ] Familiarity with `_parse_code_review_config()` pattern in config_loader.py
- [ ] Familiarity with `_validate_trigger_command_refs()` pattern for reference validation
- [ ] Understanding of `build_validation_spec()` flow and when preset merge occurs
- [ ] Understanding of `check_evidence_against_spec()` and scope filtering

## High-Level Approach

1. **Config Schema & Parsing**: Add `EvidenceCheckConfig` dataclass and `_parse_evidence_check_config()` function following the `code_review:` parsing pattern
2. **Config Integration**: Add `evidence_check` field to `ValidationConfig` and wire through `from_dict()`
3. **Reference Validation**: Add `_validate_evidence_check_refs()` in `spec.py` called during `build_validation_spec()` after preset merge completes
4. **Evidence Check Filtering**: Modify `check_evidence_against_spec()` to accept evidence config and filter required commands accordingly
5. **Testing**: Add unit tests for parsing/validation and integration tests for end-to-end flow

## Technical Design

### Architecture

**Data flow:**
```
mala.yaml (with evidence_check:)
    |
config_loader.py: _parse_evidence_check_config()
    | returns EvidenceCheckConfig
ValidationConfig.from_dict() -> ValidationConfig with evidence_check field
    |
spec.py: build_validation_spec()
    +-- Load preset (if specified)
    +-- Merge preset + user config -> resolved command map
    +-- _validate_evidence_check_refs(evidence_check_config, resolved_map)  <- NEW
    +-- Build ValidationSpec (with evidence_required field)
    |
evidence_check.py: check_evidence_against_spec()
    +-- Accept evidence_check_config parameter  <- MODIFIED
    +-- If config is None or required is empty -> no evidence needed, pass
    +-- Filter required by scope (E2E excluded from per-session)
    +-- Check evidence for filtered required commands only
```

### Data Model

**New dataclass: `EvidenceCheckConfig`** (in `config.py`)

```python
@dataclass(frozen=True, slots=True)
class EvidenceCheckConfig:
    """Configuration for evidence check requirements."""
    required: tuple[str, ...] = field(default_factory=tuple)
```

Design notes:
- `required` uses `tuple[str, ...]` matching existing pattern for immutable collections
- Empty tuple means no evidence required (same as absent section)
- No `_fields_set` tracking needed; `None` config vs present config with empty `required` both mean no evidence required

**Modified: `ValidationConfig`** (in `config.py`)

Add field:
```python
evidence_check: EvidenceCheckConfig | None = None
```

**Modified: `ValidationSpec`** (in `spec.py`)

Add field to store resolved evidence requirements:
```python
evidence_required: tuple[str, ...] = field(default_factory=tuple)
```

### API/Interface Design

**Config parsing additions (config_loader.py):**

1. Add `"evidence_check"` to `_ALLOWED_TOP_LEVEL_FIELDS` set (~line 65-82)

2. Add `_parse_evidence_check_config(data: dict, errors: list) -> EvidenceCheckConfig | None`:
   - Validate `evidence_check` is object if present (reject `null`, string, list, etc.)
   - Validate `required` is list of strings if present (reject non-list, non-string entries)
   - Return `None` if section absent; return `EvidenceCheckConfig` otherwise
   - Reject unknown fields within `evidence_check:`

3. Call from `ValidationConfig.from_dict()` and assign to new field

**Validation additions (spec.py):**

Add `_validate_evidence_check_refs(evidence_check: EvidenceCheckConfig | None, resolved_commands: dict[str, ...], errors: list) -> None`:
- If `evidence_check` is `None` or `required` is empty, return (nothing to validate)
- For each key in `required`, check existence in `resolved_commands`
- On invalid key: append error with invalid key name and sorted list of available keys
- Follow `_validate_trigger_command_refs()` error message format

Call location: In `build_validation_spec()` after preset merge creates the resolved command map.

**Evidence check modifications (evidence_check.py):**

Modify `check_evidence_against_spec()` signature:
```python
def check_evidence_against_spec(
    spec: ValidationSpec,
    evidence: list[Evidence],
    scope: EvidenceScope,
    evidence_check_config: EvidenceCheckConfig | None = None,  # NEW
) -> EvidenceCheckResult:
```

Logic changes:
- If `evidence_check_config` is `None` or `required` is empty: return success (no evidence required)
- Build `required_keys = set(evidence_check_config.required)` (set semantics for duplicates)
- Apply scope filtering: `effective_required = scope_filter(required_keys, scope)`
- Check evidence only for commands in `effective_required`
- Maintain existing `allow_fail` logic: missing evidence fails gate; evidence-of-failure with `allow_fail=true` is advisory

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/domain/validation/config.py` | Exists | Add `EvidenceCheckConfig` dataclass; add field to `ValidationConfig` |
| `src/domain/validation/config_loader.py` | Exists | Add `evidence_check` to allowed fields; add `_parse_evidence_check_config()`; wire into `from_dict()` |
| `src/domain/validation/spec.py` | Exists | Add `evidence_required` field to `ValidationSpec`; add `_validate_evidence_check_refs()`; call during `build_validation_spec()` after merge |
| `src/domain/evidence_check.py` | Exists | Modify `check_evidence_against_spec()` to accept and filter by `evidence_check_config` |
| `src/pipeline/gate_runner.py` | Exists | Update call to `check_evidence_against_spec()` to pass `evidence_check_config` |
| `tests/unit/domain/validation/test_config_loader.py` | Exists | Add parsing tests |
| `tests/unit/domain/validation/test_spec.py` | Exists | Add validation tests for `_validate_evidence_check_refs()` |
| `tests/unit/domain/test_evidence_check.py` | Exists | Add filtering tests |
| `tests/integration/domain/test_evidence_detection.py` | Exists | Add end-to-end tests |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes

| Edge Case | Expected Behavior | Test Case # |
|-----------|------------------|-------------|
| `evidence_check:` section absent | No evidence required; gate passes | 1 |
| `evidence_check: {}` (empty object) | No evidence required | 2 |
| `evidence_check.required: null` | Treated as empty list; no evidence required | 2 |
| `evidence_check.required: []` | No evidence required | 2 |
| `evidence_check: null` | Config error (must be object) | N/A (type validation) |
| `evidence_check: "string"` | Config error (must be object) | N/A (type validation) |
| `required: [test, 1]` | Config error (non-string in list) | N/A (type validation) |
| Invalid key in `required` | Config error listing invalid key + available keys | 4 |
| Duplicate keys in `required` | Ignored; treated as set | 5 |
| Preset-only key (e.g., `[test]` with no project override) | Valid; uses preset command | 6 |
| Project custom key | Valid if defined in `commands:` | 7 |
| Project override of preset key | Uses project command string for pattern | 8 |
| Global-only command in per-session gate | Filtered out; not reported as missing | 9 |
| `allow_fail` command not run | Gate fails for missing evidence | 10 |
| `allow_fail` command ran and failed | Gate passes; listed as advisory failure | 11 |
| Overlapping patterns | Single log satisfies multiple keys | 12 |
| Global-only command in `required` list | Validates successfully (R3 is pre-scope) | 13 |

### Breaking Changes & Compatibility

- **Breaking Change:** Projects relying on implicit evidence requirements (inferred from `commands:`) will see the evidence gate pass without checking
- **Migration Path:** Users must add `evidence_check.required: [test, lint, ...]` to mala.yaml to restore evidence checking
- **No Deprecation Phase:** Per spec, this is a clean break with no transition warning
- **Detection:** Users will notice gate always passing when they expected failures
- **Documentation:** Release notes must clearly state behavior change with migration example

## Testing & Validation Strategy

### Unit Tests (config_loader.py) - Parsing

1. Parse valid `evidence_check.required: [test, lint]` -> returns EvidenceCheckConfig with tuple
2. Parse `evidence_check: {}` -> returns EvidenceCheckConfig with empty required
3. Parse `evidence_check.required: null` -> returns EvidenceCheckConfig with empty required
4. Reject `evidence_check: null` -> config error
5. Reject `evidence_check: "string"` -> config error
6. Reject `evidence_check.required: "not-a-list"` -> config error
7. Reject `evidence_check.required: [test, 1]` -> config error (non-string in list)
8. Reject unknown field in `evidence_check:` -> config error

### Unit Tests (spec.py) - Reference Validation

9. Valid key from preset only -> passes (test case 6)
10. Valid key from project `commands:` -> passes (test case 7)
11. Invalid key -> config error with invalid key name and available keys (test case 4)
12. Global-only command validates successfully pre-scope (test case 13)
13. Project override -> uses project pattern (test case 8)

### Unit Tests (evidence_check.py) - Filtering

14. `evidence_check_config=None` -> no evidence required (test case 1)
15. Empty `required` -> no evidence required (test case 2)
16. Non-empty `required` -> checks only those commands (test case 3)
17. Duplicates in `required` -> ignored, set semantics (test case 5)
18. Scope filtering -> E2E excluded from per-session (test case 9)
19. `allow_fail` + not run -> fails for missing evidence (test case 10)
20. `allow_fail` + ran + failed -> passes, advisory failure (test case 11)
21. Overlapping patterns -> single log satisfies multiple keys (test case 12)

### Integration Tests

22. Full flow: config parsing -> preset merge -> validation -> evidence check with filtering
23. Preset + project merge -> uses resolved patterns for detection

### Acceptance Criteria Coverage

| Spec Requirement | Covered By |
|-----------------|------------|
| R1: Separate config section | EvidenceCheckConfig dataclass, `_parse_evidence_check_config()`, tests 1-8 |
| R2: No evidence when absent | `check_evidence_against_spec()` filtering, tests 14-15 |
| R3: Validate against resolved map | `_validate_evidence_check_refs()` in spec.py, tests 9-13 |
| R4: Pattern from resolved command | Existing flow uses resolved; tests 13, 23 |
| R5: Built-in + custom support | Validation checks full resolved map, tests 9-10 |
| R6: Set semantics | `set()` usage in filtering, test 17 |
| R7: Scope filtering | Scope filter applied to required, test 18 |
| R8: allow_fail semantics | Existing logic handles, tests 19-20 |
| R9: Per-key attribution | Existing pattern matching, test 21 |

## Open Questions

*All implementation questions were resolved during interview:*
- Validation location: `spec.py` during `build_validation_spec()` (after preset merge)
- Migration warning: None (clean break)
- Evidence check integration: Modify existing `check_evidence_against_spec()` with parameter
- Config tracking: No `_fields_set` needed for `EvidenceCheckConfig`
- Test isolation: Inline fixture presets for validation tests

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` -> Beads issues with dependencies for multi-agent execution
- (default) -> TODO.md checklist for simpler tracking
