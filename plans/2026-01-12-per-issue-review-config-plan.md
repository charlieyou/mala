# Implementation Plan: Per-Issue Review Configuration

## Context & Goals
- **Spec**: N/A — derived from user description and interview
- Add a configurable `per_issue_review` section to mala.yaml that controls whether post-session code review runs for each issue
- Make the currently hard-coded per-issue review step optional and configurable
- Allow users to enable/disable per-issue reviews independently of trigger-based reviews
- Users gain control over whether automated code review runs after each issue session

## Scope & Non-Goals

### In Scope
- Add `per_issue_review` as a root-level key in `mala.yaml` (same level as `timeout_minutes`, `validation_triggers`, etc.)
- Reuse existing `CodeReviewConfig` dataclass for the new section
- Wire config through `ValidationConfig` → `_DerivedConfig` → orchestrator → lifecycle
- Update `_extract_reviewer_config()` in factory.py to include `per_issue_review` as a source
- Add `mala init` prompts for per-issue review (before triggers section)
- Update documentation (`docs/project-config.md`)

### Out of Scope (Non-Goals)
- CLI flags to override per-issue review settings per-run (config-only)
- Changing trigger-based (`session_end`, `run_end`, `epic_completion`) code review behavior — these remain independent
- Per-issue overrides (all issues use same settings)
- New reviewer types beyond `cerberus`/`agent_sdk`
- Automatic migration of existing `mala.yaml` files (breaking change accepted per user decision)
- Changes to skip-review outcomes (NO_CHANGE, OBSOLETE, ALREADY_COMPLETE, DOCS_ONLY)
- Separate reviewer instances for per-issue vs trigger reviews (shared reviewer is existing behavior)

## Assumptions & Constraints

### Clarification: "Top-Level" Ownership

**`ValidationConfig` IS the container for root-level `mala.yaml` keys.** Despite its name, `ValidationConfig` is documented as "Top-level configuration from mala.yaml" (config.py:782-783) and already contains these root-level YAML keys:

| Root-Level YAML Key | ValidationConfig Field |
|---------------------|------------------------|
| `timeout_minutes:` | `timeout_minutes: int \| None` |
| `context_limit:` | `context_limit: int \| None` |
| `evidence_check:` | `evidence_check: EvidenceCheckConfig \| None` |
| `validation_triggers:` | `validation_triggers: ValidationTriggersConfig \| None` |
| `preset:` | `preset: str \| None` |

Adding `per_issue_review:` to `ValidationConfig` follows this established pattern.

### Breaking Change: Disabled by Default (User Decision)

**Per user decision during interview**, the default behavior changes from "review always enabled" to "review disabled by default". The user explicitly chose:
- **No migration path** — breaking change is intentional
- **No deprecation warnings** — immediate behavior change
- **No preset-based enablement** — presets do not implicitly enable per_issue_review

This is documented in the "Breaking Changes" section with the concrete date (2026-01-12).

### Reviewer Configuration is Global (Existing Behavior)

**The orchestrator creates a single shared `CodeReviewer` instance** (factory.py:632) used by both per-issue reviews and trigger-based reviews. This is existing behavior, not new to this plan.

**Reviewer selection priority** (existing pattern, extended):
1. `per_issue_review` (if `enabled=True`) → use its `reviewer_type`, `cerberus`, `agent_sdk_*` settings
2. First enabled trigger `code_review` (existing behavior)
3. Defaults (`agent_sdk` with timeout=600, model=sonnet)

**When `per_issue_review.enabled=False`**: The `per_issue_review` section is **completely ignored** for all purposes — reviewer selection falls through to triggers or defaults. No fields from a disabled `per_issue_review` are used.

**User-Visible Rule (for docs and init prompts)**:
> **Reviewer Selection**: Enabling `per_issue_review` also determines the reviewer type (Cerberus or Agent SDK) used for all reviews, including trigger-based reviews. To use different reviewer settings for triggers, leave `per_issue_review.enabled: false` so your trigger configs take priority.

### Assumptions
- The `CodeReviewConfig` dataclass (config.py:251) contains all fields needed for per-issue review
- The `_parse_code_review_config()` helper in config_loader.py can be reused directly
- Per-issue review and trigger code_review are semantically distinct: `enabled` flags control whether each type runs, but reviewer settings are shared

### Implementation Constraints
- Extend existing `CodeReviewConfig` parsing — do not create a new dataclass
- Follow existing config parsing patterns in `config.py`/`config_loader.py`
- Wire through existing `_DerivedConfig` path
- Do not introduce a new config file; keep everything in `mala.yaml`

### Audit: `max_review_retries` Scope

**Confirmed per-issue review only.** Audited all usages of `max_review_retries`:

| Component | File | Used For |
|-----------|------|----------|
| `LifecycleConfig.max_review_retries` | lifecycle.py:223 | Per-issue review retry limit |
| `ReviewRunnerConfig.max_review_retries` | review_runner.py:83 | Per-issue review runner |
| `PipelineConfig.max_review_retries` | types.py:212 | Passed to lifecycle/runner for per-issue |
| `OrchestratorConfig.max_review_retries` | types.py:90 | CLI override for per-issue review |

**Trigger-based reviews use different config**: Triggers use their own `code_review.max_retries` field within each trigger's `CodeReviewConfig`, NOT the shared `max_review_retries`.

### `max_review_retries` Precedence Order

**Clear precedence: CLI > YAML > default**

```
1. CLI --max-review-retries flag (OrchestratorConfig.max_review_retries)
   ↓ if not set
2. per_issue_review.max_retries from mala.yaml
   ↓ if not set or per_issue_review.enabled=False
3. Default: 3 (from CodeReviewConfig.max_retries default)
```

**Single source of truth**: `PipelineConfig.max_review_retries` becomes a derived value computed during factory wiring.

**Scope boundary**: `max_review_retries` applies ONLY to per-issue reviews (via `LifecycleConfig.max_review_retries`). Trigger-based reviews use their own `trigger_config.code_review.max_retries` field, which is completely separate. The CLI `--max-review-retries` flag is moot when per-issue review is disabled (the flag affects per-issue retry count, not trigger retries).

### Testing Constraints
- Unit tests for config parsing (enabled/disabled, all fields, defaults)
- Integration test for lifecycle skip when disabled
- Test `mala init` prompts flow
- Must-have regression coverage: existing trigger-based reviews unaffected

## Integration Analysis

### Existing Mechanisms Considered

| Existing Mechanism | Could Serve Feature? | Decision | Rationale |
|--------------------|---------------------|----------|-----------|
| `CodeReviewConfig` dataclass (config.py:251) | Yes | Reuse | Already has all needed fields |
| `_parse_code_review_config()` (config_loader.py) | Yes | Reuse | Existing helper parses the exact structure we need |
| `ValidationConfig` | Yes | Extend | Already the container for root-level mala.yaml keys |
| `_extract_reviewer_config()` (factory.py) | Partial | Extend | Must include `per_issue_review` as a source |
| `orchestrator._is_review_enabled()` (orchestrator.py:693) | Yes | Modify | Canonical decision point for **per-issue** review enablement (does NOT affect triggers) |
| `LifecycleConfig.review_enabled` (lifecycle.py:225) | Yes | Source from orchestrator | Already exists; receives value from orchestrator |

### Integration Approach

This feature integrates by extending existing infrastructure at every layer:

1. **Config storage**: Add `per_issue_review: CodeReviewConfig` field to `ValidationConfig` with default `CodeReviewConfig(enabled=False)`
2. **Config parsing**: Reuse `_parse_code_review_config()` to parse the root-level `per_issue_review` section in `ValidationConfig.from_dict()`. Unknown root-level keys are rejected by the existing schema validation (raises `ConfigError`), so `per_issue_review` must be explicitly added to the allowed keys list.
3. **Reviewer config extraction**: Update `_extract_reviewer_config()` to check `per_issue_review` FIRST (if enabled) before falling back to trigger configs
4. **Pipeline wiring**: Pass `per_issue_review: CodeReviewConfig | None` through `_DerivedConfig`
5. **Review enablement**: Modify `orchestrator._is_review_enabled()` to check `per_issue_review.enabled`
6. **CLI init**: Add questionary prompts before validation triggers section

No new infrastructure is created — this extends existing config/pipeline/lifecycle patterns.

### Scope Boundaries: Per-Issue vs Trigger Reviews (Critical)

**Per-issue and trigger reviews have SEPARATE enablement paths.** Modifying `_is_review_enabled()` does NOT affect trigger reviews:

| Review Type | Enablement Check | Location |
|-------------|------------------|----------|
| Per-issue | `orchestrator._is_review_enabled()` → `LifecycleConfig.review_enabled` | orchestrator.py:693, lifecycle.py:531 |
| Trigger (run_end, epic_completion, etc.) | `trigger_config.code_review.enabled` | run_coordinator.py:1361 |

**Trigger reviews check their own config**: `run_coordinator._run_trigger_code_review()` (line 1361) checks `if trigger_config.code_review is None or not trigger_config.code_review.enabled: return None`. This is completely independent of `_is_review_enabled()`.

**Implication**: When `per_issue_review` is missing/disabled, `_is_review_enabled()` returns False (per-issue reviews skip), but trigger reviews are unaffected because they use their own `code_review.enabled` flag.

**Test to add**: Verify that with `per_issue_review` absent and `validation_triggers.run_end.code_review.enabled=True`, trigger review still runs.

## Prerequisites

- [x] `CodeReviewConfig` dataclass exists and has all needed fields
- [x] `_parse_code_review_config()` handles all fields correctly
- [x] Alignment on breaking change (disabled by default per user decision)
- [x] `LifecycleConfig.review_enabled` already exists (lifecycle.py:225)
- [x] `_proceed_to_review_or_success()` already checks `review_enabled` (lifecycle.py:531)
- [x] `orchestrator._is_review_enabled()` is the canonical decision point (orchestrator.py:693)
- [x] Audit `max_review_retries` usage (per-issue only - confirmed)

## High-Level Approach

The implementation adds `per_issue_review` as a root-level key in `mala.yaml` (same level as `timeout_minutes`, `validation_triggers`, etc.). This section reuses the existing `CodeReviewConfig` dataclass, keeping the configuration schema consistent with trigger-based reviews.

**Key design decisions:**
1. **Disabled by default** — Breaking change per user decision; no migration path
2. **Independent enabled flags** — `per_issue_review.enabled` controls per-issue reviews; trigger `code_review.enabled` controls trigger reviews
3. **Shared reviewer config** — First enabled source (per_issue_review if enabled, else triggers) determines global reviewer settings
4. **Complete ignore when disabled** — When `per_issue_review.enabled=False`, the entire section is ignored (no fallback use)
5. **Reuse CodeReviewConfig** — Same fields as trigger code_review; `baseline` field is allowed syntactically but ignored in per-issue context
6. **No CLI override for enabled** — Config-only control (existing `--max-review-retries` still works)

**Implementation steps:**
1. Add `per_issue_review: CodeReviewConfig` to `ValidationConfig` with default `CodeReviewConfig(enabled=False)`
2. Call existing `_parse_code_review_config()` for the root-level `per_issue_review` key in `ValidationConfig.from_dict()`
3. Update `_extract_reviewer_config()` to check `per_issue_review` first (only if enabled), then fall back to triggers
4. Add `per_issue_review: CodeReviewConfig | None` to `_DerivedConfig`
5. Modify `orchestrator._is_review_enabled()` to return `per_issue_review.enabled` (respecting disabled_validations override)
6. Add questionary prompts for per-issue review before trigger prompts
7. Update `docs/project-config.md` with migration note and new section

## Technical Design

### Architecture

**Configuration flow:**
```
mala.yaml (root-level per_issue_review:)
    │
    ▼ (config_loader.py: ValidationConfig.from_dict)
ValidationConfig.per_issue_review: CodeReviewConfig
    │
    ▼ (factory.py: _derive_config)
_DerivedConfig.per_issue_review: CodeReviewConfig | None
    │
    ▼ (factory.py: create_orchestrator)
MalaOrchestrator stores per_issue_review config
    │
    ▼ (orchestrator.py: _is_review_enabled, line 693)
Returns per_issue_review.enabled (respecting disabled_validations)
    │
    ▼ (run_config.py → orchestration_wiring.py → agent_session_runner.py)
LifecycleConfig.review_enabled: bool
    │
    ▼ (lifecycle.py: _proceed_to_review_or_success, line 518)
if self.config.review_enabled and commit_hash and not resolution_skips_review:
    transition to RUNNING_REVIEW
else:
    transition to SUCCESS
```

**Canonical decision point**: `orchestrator._is_review_enabled()` at line 693. This is modified to check `per_issue_review.enabled`. The value flows through `run_config` → `orchestration_wiring` → `agent_session_runner` → `LifecycleConfig`.

**Reviewer config extraction:**
```
_extract_reviewer_config() priority:
1. per_issue_review (ONLY if enabled=True) → use its reviewer settings
2. First enabled trigger code_review (existing behavior)
3. Defaults (agent_sdk with timeout=600, model=sonnet)

When per_issue_review.enabled=False: Section is completely ignored.
```

### Lifecycle Transition Details

**Decision point**: `_proceed_to_review_or_success()` method in lifecycle.py (lines 518-553)

**Truth table for review decision:**

| `review_enabled` | Has commit | Resolution outcome | Result |
|------------------|------------|-------------------|--------|
| False | any | any | → SUCCESS (skip review) |
| True | False | any | → SUCCESS (no code to review) |
| True | True | NO_CHANGE/OBSOLETE/etc | → SUCCESS (resolution-based skip) |
| True | True | other/None | → RUNNING_REVIEW |

**Terminal effect/message**: When `review_enabled=False`, uses same terminal state (`SUCCESS`) and effect (`COMPLETE_SUCCESS`) as resolution-based skip. Message is "No review required".

### Data Model

**Modifications to `src/domain/validation/config.py` (ValidationConfig):**
```python
@dataclass(frozen=True)
class ValidationConfig:
    # ... existing fields (commands, preset, coverage, etc.) ...
    per_issue_review: CodeReviewConfig = field(
        default_factory=lambda: CodeReviewConfig(enabled=False)
    )
    # ... _fields_set ...
```

**Modifications to `src/orchestration/types.py` (_DerivedConfig):**
```python
@dataclass
class _DerivedConfig:
    # ... existing fields ...
    per_issue_review: CodeReviewConfig | None = None
```

**Modifications to `src/orchestration/orchestrator.py` (_is_review_enabled):**
```python
def _is_review_enabled(self) -> bool:
    """Return whether per-issue review should run."""
    # Check disabled_validations first (CLI override)
    if "review" in self._disabled_validations:
        # Allow override if reviewer supports it
        if (
            self.review_disabled_reason
            and self.review_runner.code_reviewer.overrides_disabled_setting()
        ):
            return True
        return False
    # Check per_issue_review.enabled from config
    if self._per_issue_review is not None:
        return self._per_issue_review.enabled
    # Fallback: disabled by default (breaking change)
    return False
```

### API/Interface Design

**Proposed `mala.yaml` schema (root-level key):**
```yaml
# Existing root-level keys
preset: python-uv
timeout_minutes: 60

# NEW: Per-issue review configuration
per_issue_review:
  enabled: true              # default: false (BREAKING CHANGE as of 2026-01-12)
  reviewer_type: cerberus    # default: cerberus; or "agent_sdk"
  max_retries: 3             # default: 3
  finding_threshold: P1      # default: none; P0/P1/P2/P3/none
  track_review_issues: true  # default: true
  failure_mode: continue     # default: continue; abort/continue/remediate
  cerberus:
    timeout: 300
    spawn_args: ["--mode", "fast"]
    wait_args: []
    env: []
  agent_sdk_timeout: 600     # default: 600
  agent_sdk_model: sonnet    # default: sonnet; sonnet/opus/haiku
  # baseline is allowed syntactically but ignored in per-issue context

# Existing trigger-based reviews (unchanged)
validation_triggers:
  run_end:
    code_review:
      enabled: true
      # ... trigger-specific settings ...
```

**Field names match `CodeReviewConfig` exactly (config.py:268-277).**

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/domain/validation/config.py` | Exists | Add `per_issue_review: CodeReviewConfig` field to `ValidationConfig` |
| `src/domain/validation/config_loader.py` | Exists | Parse root-level `per_issue_review` key in `ValidationConfig.from_dict()` |
| `src/orchestration/types.py` | Exists | Add `per_issue_review: CodeReviewConfig \| None` to `_DerivedConfig` |
| `src/orchestration/factory.py` | Exists | Update `_derive_config()` to pass `per_issue_review`; update `_extract_reviewer_config()` to check `per_issue_review` first (only if enabled) |
| `src/orchestration/orchestrator.py` | Exists | Modify `_is_review_enabled()` to check `per_issue_review.enabled` |
| `src/cli/cli.py` | Exists | Add `mala init` prompts for per-issue review before trigger prompts |
| `docs/project-config.md` | Exists | Document new root-level `per_issue_review` section with migration note |
| `tests/domain/validation/test_config.py` | Exists | Add tests for ValidationConfig with per_issue_review |
| `tests/domain/validation/test_config_loader.py` | Exists | Add tests for per_issue_review parsing |
| `tests/unit/orchestration/test_orchestrator.py` | Exists | Add tests for `_is_review_enabled()` with per_issue_review |

**Removed**: `src/pipeline/session_callback_factory.py` — not involved in review enablement. The path is `orchestrator._is_review_enabled()` → `run_config` → `orchestration_wiring` → `agent_session_runner` → `LifecycleConfig`.

## Risks, Edge Cases & Breaking Changes

### Breaking Changes & Compatibility

- **Breaking Change (as of 2026-01-12)**:
  - Per-issue review is now **DISABLED by default** (previously always ran)
  - Existing `mala.yaml` files without `per_issue_review` section will no longer get per-issue reviews
  - **No migration path** — per user decision during planning interview
  - **No preset enablement** — presets do not implicitly enable per_issue_review

- **Migration Note (for docs/project-config.md)**:
  > **Breaking Change (2026-01-12)**: Per-issue code review is now disabled by default. To restore the previous behavior where every issue session is reviewed, add to your `mala.yaml`:
  > ```yaml
  > per_issue_review:
  >   enabled: true
  > ```

- **`mala init` behavior**: When generating new `mala.yaml`, the wizard will prompt for per-issue review settings. If user declines, the section is omitted (matches loader defaults = disabled).

### Edge Cases & Failure Modes

- **Missing `per_issue_review` section**: Defaults to `CodeReviewConfig(enabled=False)` — review disabled
- **Partial config (only some fields)**: Use defaults for missing fields (existing `_parse_code_review_config()` behavior)
- **Invalid field values**: Validation errors raised during config loading (existing mechanism)
- **`per_issue_review.enabled=False` with other fields set**: Entire section is **completely ignored** — fields are NOT used for any purpose including reviewer fallback
- **`per_issue_review` disabled but trigger `code_review` enabled**: Trigger review runs; reviewer settings come from trigger config (per_issue_review ignored)
- **`per_issue_review` enabled but triggers disabled**: Per-issue reviews run; reviewer settings come from per_issue_review
- **Both enabled with different reviewer_type**: per_issue_review takes priority for reviewer settings (global shared reviewer)
- **In-flight issues when config changes**: Config is read at startup; mid-run changes don't affect running issues
- **`baseline` field in per_issue_review**: Allowed syntactically (same schema), ignored semantically

## Testing & Validation Strategy

### Unit Tests
- **Config Loader**:
  - Parse valid `per_issue_review` config with all fields
  - Parse minimal `per_issue_review: {enabled: true}`
  - Parse missing `per_issue_review` → defaults to `CodeReviewConfig(enabled=False)`
  - Verify root-level placement (not under validation_triggers)
  - Verify field validation (invalid values raise appropriate errors)
  - Verify `baseline` is accepted syntactically
- **Orchestrator**:
  - Verify `_is_review_enabled()` returns False when `per_issue_review.enabled=False`
  - Verify `_is_review_enabled()` returns True when `per_issue_review.enabled=True`
  - Verify `--disable-validations review` overrides `per_issue_review.enabled=True`
  - Verify missing `per_issue_review` section → `_is_review_enabled()` returns False
- **Lifecycle**:
  - Verify `_proceed_to_review_or_success()` skips to `SUCCESS` when `review_enabled=False`
  - Verify skip-review outcomes (NO_CHANGE, etc.) still work regardless of `review_enabled`
- **Factory**:
  - Verify `_extract_reviewer_config()` prioritizes `per_issue_review` when `enabled=True`
  - Verify `_extract_reviewer_config()` **ignores** `per_issue_review` when `enabled=False` and falls back to triggers
  - Verify `max_review_retries` precedence: CLI > per_issue_review > default
- **CLI Init**:
  - Verify prompts appear before trigger prompts
  - Verify omitting per-issue review results in no `per_issue_review` section

### Integration Tests
- Run a short task with `per_issue_review` disabled → verify no review phase occurs
- Run a short task with `per_issue_review` enabled → verify review phase occurs
- **Critical**: Verify trigger-based reviews run when `per_issue_review` absent but `validation_triggers.run_end.code_review.enabled=True`
- Verify CLI `--max-review-retries` overrides `per_issue_review.max_retries`
- Verify unknown root keys in mala.yaml raise `ConfigError` with helpful message

### Manual Verification
- Run `mala init` interactively and verify per-issue review prompts appear before triggers
- Run `mala run` with `per_issue_review: enabled: true` and verify review runs
- Run `mala run` with `per_issue_review` absent and verify review skips

### Acceptance Criteria Coverage

| Spec AC | Covered By |
|---------|------------|
| Per-issue review can be enabled/disabled via mala.yaml | Config parsing + orchestrator tests |
| Default is disabled | Default value in config.py, orchestrator unit tests |
| All CodeReviewConfig fields available | Reuse of dataclass, config parsing tests |
| `mala init` prompts for review settings | cli.py implementation, manual verification |
| Independent from trigger code_review | Separate enabled flags, integration test |
| Root-level YAML placement | Config loader tests |
| Reviewer config respects priority | Factory tests for `_extract_reviewer_config()` |
| Disabled section is completely ignored | Factory tests verify no fallback use |

## Open Questions

None — all questions resolved during interview and review feedback.

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
