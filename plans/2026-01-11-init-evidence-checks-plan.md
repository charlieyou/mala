# Implementation Plan: Mala Init Evidence Checks & Triggers

## Context & Goals
- **Spec**: [docs/2026-01-11-init-evidence-checks-spec.md](../docs/2026-01-11-init-evidence-checks-spec.md)
- Add `evidence_check` and `validation_triggers` configuration to `mala init` flow
- Use `questionary` library for checkbox multi-select UI
- Provide sensible defaults with opt-out capability

## Scope & Non-Goals
- **In Scope**
  - Evidence check configuration during init (R1, R2, R7, R8)
  - run_end validation trigger configuration (R3, R7, R8)
  - Checkbox multi-select UI using questionary (R5)
  - Config validation before writing (R6)
  - Trigger quick-reference table after init (R4, R9)
  - Generated YAML with correct schema (R10)
  - Non-interactive mode with CLI flags (`--yes`, `--preset`, `--skip-evidence`, `--skip-triggers`)
  - TTY detection with actionable error message for non-interactive terminals
- **Out of Scope (Non-Goals)**
  - New preset types
  - Triggers other than run_end during init
  - Granular per-command failure_mode/max_retries configuration

## Assumptions & Constraints

### Implementation Constraints
- questionary 2.x requires Python 3.8+ (we require 3.11+, so compatible)
- **Scope of questionary migration:** Only migrate prompts within init() flow (`_prompt_menu_selection()`, `_prompt_custom_commands()`). Other CLI commands' prompts (if any) remain unchanged to minimize regression risk.
- questionary is a required runtime dependency (add to main dependencies)
- Preserve existing `_write_with_backup()` logic unchanged
- Maintain `--dry-run` behavior (outputs YAML to stdout, no file write)
- New prompt helper functions are private to init() scope (prefixed with `_`)

### Testing Constraints
- Mock questionary functions in tests (patch questionary.confirm/checkbox/select to return test values)
- Do not rely on CliRunner's input parameter for questionary prompts (questionary bypasses stdin)
- Extend existing fixture with stub commands for E2E behavioral test
- All tests must be deterministic (no time-dependent assertions)

## Integration Analysis

### Existing Mechanisms Considered

| Existing Mechanism | Could Serve Feature? | Decision | Rationale |
|--------------------|---------------------|----------|-----------|
| `init()` in `cli.py:963-1017` | Yes | Extend | Main entry point - add evidence/trigger prompts after preset selection, add CLI flags |
| `_prompt_menu_selection()` in `cli.py:908-931` | Yes | Replace | Migrate to questionary.select() for consistent UX with checkboxes |
| `_prompt_custom_commands()` in `cli.py:934-952` | Yes | Replace | Migrate to questionary.text() for consistent UX |
| `validate_init_config()` in `cli_support.py` | Yes | Reuse | Already validates evidence_check and validation_triggers fields (see Validation Coverage note below) |
| `dump_config_yaml()` in `cli_support.py` | Yes | Reuse | Already serializes config dict to YAML with sort_keys=False |
| `PresetRegistry.get()` in `preset_registry.py` | Yes | Reuse | Load preset to get command names for computing defaults |
| `EvidenceCheckConfig`, `ValidationTriggersConfig` | Yes | Reuse | Config dataclasses already fully support these sections |
| Command names in `_prompt_custom_commands()` | Yes | Reuse | Use same list for questionary custom prompts |

### Validation Coverage (validate_init_config)
The existing `validate_init_config()` delegates to `ValidationConfig.from_dict()` which already validates:
- `evidence_check.required` entries must be strings (type validation)
- `validation_triggers.run_end.commands[].ref` entries must be strings
- `failure_mode` must be one of: `abort`, `continue`, `remediate`
- `max_retries` must be a non-negative integer when present

**Not validated by schema:** Command refs exist in the resolved command map. This is validated by `_validate_config()` in config_loader.py, which is called by `validate_init_config()`. The UI constraints (checkbox shows only available commands) ensure refs are always valid.

### Integration Approach
Extend the existing `init()` function by:
1. Add CLI parameters for `--yes`, `--preset`, `--skip-evidence`, `--skip-triggers`
2. Add TTY detection with actionable error for non-interactive mode without `--yes`
3. After preset/custom selection, get the list of available command names (from preset or user input)
4. Call new questionary-based prompt functions for evidence_check and validation_triggers
5. Build config dict with additional sections based on user selections
6. Existing validation and YAML generation work unchanged
7. Add trigger quick-reference table printing at the end

## Prerequisites
- [ ] Add questionary>=2.0.0 to pyproject.toml dependencies (required)
- [ ] Verify questionary has no conflicts with existing dependencies

## High-Level Approach

**Phase 1: Add questionary dependency and helper functions**
- Add questionary to pyproject.toml dependencies
- Create helper function to get command list from preset or custom dict
- Add trigger reference table printing function

**Phase 2: Migrate existing prompts and add new ones**
- Replace `_prompt_menu_selection()` with questionary.select()
- Replace `_prompt_custom_commands()` with questionary prompts
- Add `_prompt_evidence_check()` with questionary.confirm() + checkbox()
- Add `_prompt_run_end_trigger()` with questionary.confirm() + checkbox()
- Add non-interactive mode support with new CLI flags

**Phase 3: Update init() and add tests**
- Modify `init()` to orchestrate the full flow
- Add integration tests with questionary mocking
- Extend existing fixture for E2E behavioral test

## Technical Design

### Architecture

**Interactive flow (TTY detected, no --yes):**
1. Check TTY - fail with actionable error if not interactive and no --yes (unless --skip-* flags make prompts unnecessary)
2. Prompt preset selection via questionary.select() OR use --preset flag
   - Options: [go, node-npm, python-uv, rust, custom]
3. If custom: prompt for each command via questionary.text() for each builtin command name
4. Get available command list from preset/custom selection
5. If commands empty: print message, skip evidence/trigger prompts
6. If NOT --skip-evidence:
   - questionary.confirm("Configure evidence checks? [Y/n]", default=True)
   - If yes: questionary.checkbox() with defaults pre-checked (preset) or none (custom)
   - If user unchecks all: questionary.confirm("No commands selected. Skip this section?")
7. If NOT --skip-triggers:
   - questionary.confirm("Configure run_end validation trigger? [Y/n]", default=True)
   - If yes: questionary.checkbox() with defaults pre-checked (preset) or none (custom)
   - If user unchecks all: questionary.confirm("No commands selected. Skip this section?")
8. Build config dict with evidence_check and validation_triggers sections (if not skipped)
9. Validate config via validate_init_config()
10. If validation fails: questionary.select() to offer "Revise selections", "Skip section", "Abort init"
11. Write YAML via dump_config_yaml() (respects --dry-run)
12. Print trigger quick-reference table (always)

**Non-interactive flow (--yes):**
1. Require --preset flag (error if missing)
2. Load preset, get command names
3. Compute evidence defaults: intersection of {test, lint} with available commands
4. Compute trigger defaults: all commands except setup/e2e
5. Apply --skip-evidence to omit evidence_check section
6. Apply --skip-triggers to omit validation_triggers section
7. Build config dict, validate, write YAML
8. Print trigger quick-reference table

**Default computation logic:**
- **Evidence defaults (preset path):** `intersection({test, lint}, available_commands)`
- **Evidence defaults (custom path):** `[]` (empty, none pre-checked)
- **Trigger defaults (preset path):** `[cmd for cmd in available_commands if cmd not in {setup, e2e}]`
- **Trigger defaults (custom path):** `[]` (empty, none pre-checked)
- **Excluded commands for triggers:** Exact case-sensitive match on `setup` and `e2e` strings only

**Key functions to add/modify:**
- `_get_preset_command_names(preset_name: str) -> list[str]` — Load preset, return ordered command names
- `_prompt_preset_selection(presets: list[str]) -> str | None` — questionary.select(), returns preset name or None for custom
- `_prompt_custom_commands_questionary() -> dict[str, str]` — questionary.text() for each builtin command
- `_compute_evidence_defaults(commands: list[str], is_preset: bool) -> list[str]` — Pre-checked defaults for evidence
- `_compute_trigger_defaults(commands: list[str], is_preset: bool) -> list[str]` — Pre-checked defaults for trigger
- `_prompt_evidence_check(commands: list[str], is_preset: bool) -> list[str] | None` — Confirm + checkbox, returns None if skipped
- `_prompt_run_end_trigger(commands: list[str], is_preset: bool) -> list[str] | None` — Confirm + checkbox, returns None if skipped
- `_build_validation_triggers_dict(commands: list[str]) -> dict[str, Any]` — Build triggers section with ref: syntax
- `_print_trigger_reference_table() -> None` — Print quick-reference table to stdout
- Modify `init()` to orchestrate the new prompts and CLI flags

### Data Model
Uses existing config dict structure. Example output for python-uv preset with defaults:
```yaml
preset: python-uv

evidence_check:
  required: [test, lint]

validation_triggers:
  run_end:
    fire_on: success
    failure_mode: remediate
    max_retries: 3
    commands:
      - ref: test
      - ref: lint
      - ref: format
      - ref: typecheck
```

Key constraints on generated YAML:
- Command order in `evidence_check.required` and `validation_triggers.run_end.commands` matches checkbox display order (preset-defined or entry order)
- Uses `ref:` syntax for trigger commands (not bare strings)
- Omitted sections are not written (no `evidence_check: {}` or `validation_triggers: {}`)

### API/Interface Design

**New CLI flags for init():**
```python
@app.command()
def init(
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview without writing")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Accept all defaults (requires --preset)")] = False,
    preset: Annotated[str | None, typer.Option("--preset", "-p", help="Use preset non-interactively")] = None,
    skip_evidence: Annotated[bool, typer.Option("--skip-evidence", help="Omit evidence_check section")] = False,
    skip_triggers: Annotated[bool, typer.Option("--skip-triggers", help="Omit validation_triggers section")] = False,
) -> None:
```

**Flag semantics:**

| Flags | Behavior |
|-------|----------|
| (none, TTY) | Interactive flow with questionary prompts |
| (none, non-TTY) | Error: "Non-interactive terminal detected. Use `mala init --yes --preset <name>` for non-interactive setup." |
| `--yes` alone | Error: "--yes requires --preset" |
| `--yes --preset python-uv` | Accept all defaults (evidence_check + run_end trigger) |
| `--yes --preset python-uv --skip-evidence` | Accept defaults but omit evidence_check section |
| `--yes --preset python-uv --skip-triggers` | Accept defaults but omit validation_triggers section |
| `--preset python-uv` (TTY) | Use preset, still prompt for evidence/triggers interactively |
| `--preset python-uv --skip-evidence` (TTY) | Use preset, skip evidence prompt (section omitted), prompt for triggers |
| `--preset python-uv --skip-evidence --skip-triggers` (TTY) | Use preset, skip both prompts (sections omitted) |
| `--preset python-uv --skip-evidence --skip-triggers` (non-TTY) | Works without `--yes` since no prompts needed |

**Key rule:** `--skip-*` flags always take effect (skip the section), regardless of TTY status. In interactive mode, the corresponding prompt is simply not shown.

**Trigger quick-reference table format:**
```
Trigger Quick-Reference
═══════════════════════════════════════════════════════════════
Trigger            When it fires
───────────────────────────────────────────────────────────────
epic_completion    After an epic finishes all verification steps
session_end        When an agent session ends
periodic           Every N issues processed
run_end            At the end of the mala run
═══════════════════════════════════════════════════════════════
```
Note: Exact wording/formatting flexible; tests verify semantic content (key phrases present).

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/cli/cli.py` | Exists | Extend init() with new flags; replace `_prompt_menu_selection()` and `_prompt_custom_commands()` with questionary versions; add evidence/trigger prompt functions and table printing |
| `pyproject.toml` | Exists | Add `questionary>=2.0.0` to `[project.dependencies]` |
| `tests/integration/cli/test_init.py` | Exists | Add test classes for evidence/trigger prompts with questionary mocking |
| `tests/e2e/test_mala_init.py` | **New** | E2E test with stub commands fixture; verify `mala run` executes commands and fires run_end trigger |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes

| Edge Case | Expected Behavior |
|-----------|-------------------|
| No commands defined (preset bug or all custom skipped) | Print "No commands defined. Skipping evidence check and trigger configuration." and proceed |
| User unchecks all boxes in checkbox | Prompt "No commands selected. Skip this section? [Y/n]". If Y, omit section. If n, return to checkbox |
| Validation fails after user selections | Show questionary.select() with options: "Revise selections", "Skip section", "Abort init" |
| `--yes` without `--preset` | Exit with error: "--yes requires --preset to specify which preset to use" |
| Non-TTY without `--yes` | Exit with actionable error message |
| `--preset invalid-name` | Error from PresetRegistry.get(): "Unknown preset 'invalid-name'. Available presets: ..." |
| Keyboard interrupt during prompts | Exit gracefully (existing `except KeyboardInterrupt` handles this) |
| questionary import fails | Import error on `mala init`; indicates broken installation |

### Breaking Changes & Compatibility

- **Prompt appearance changes:** Users will see questionary-style prompts instead of typer.prompt(). This is a UX change, not a functional breaking change.
- **CLI flag additions:** New flags (`--yes`, `--preset`, `--skip-*`) are additive; no existing flags are removed or changed.
- **Generated YAML changes:** Configs generated after this change may include evidence_check and validation_triggers sections. These are optional fields; existing `mala run` already handles their absence.
- **No backwards incompatibility:** Users who skip the new prompts get the same config as before.

## Testing & Validation Strategy

### Unit Tests
- `_compute_evidence_defaults()`: Test intersection logic for preset path; empty list for custom path
- `_compute_trigger_defaults()`: Test exclusion of setup/e2e for preset path; empty list for custom path
- `_build_validation_triggers_dict()`: Test ref: syntax generation

### Integration Tests (questionary mocked)
Test class structure:
```python
class TestInitEvidencePrompts:
    def test_preset_evidence_defaults_shown(self, mock_questionary): ...
    def test_evidence_skip_via_confirm_no(self, mock_questionary): ...
    def test_evidence_uncheck_all_prompts_skip(self, mock_questionary): ...
    def test_custom_path_no_preselected_defaults(self, mock_questionary): ...

class TestInitTriggerPrompts:
    def test_preset_trigger_defaults_exclude_setup_e2e(self, mock_questionary): ...
    def test_trigger_skip_via_confirm_no(self, mock_questionary): ...
    def test_custom_path_no_preselected_defaults(self, mock_questionary): ...

class TestInitNonInteractive:
    def test_yes_without_preset_errors(self): ...
    def test_yes_with_preset_uses_defaults(self): ...
    def test_skip_evidence_omits_section(self): ...
    def test_skip_triggers_omits_section(self): ...
    def test_non_tty_without_yes_errors(self): ...
    def test_preset_with_skip_flags_works_non_tty(self): ...  # --preset + --skip-* works without --yes

class TestInitTriggerTable:
    def test_table_printed_on_success(self, mock_questionary): ...
    def test_table_printed_even_when_sections_skipped(self, mock_questionary): ...
    def test_table_contains_semantic_content(self, mock_questionary): ...
```

### E2E Test (behavioral verification)
**Spec-driven:** The spec's success criteria explicitly requires "running `mala run` with a single issue produces logs showing each configured command executed and `run_end` validation fired". This test verifies the complete workflow.

```python
# tests/e2e/test_mala_init.py
def test_init_with_defaults_produces_runnable_config(tmp_path, monkeypatch):
    """
    Given: Fixture repo with stub commands (test, lint, format = 'echo ok')
    When: mala init --yes --preset python-uv
    Then: mala.yaml contains evidence_check and validation_triggers
    When: mala run with single issue
    Then: Logs show commands executed and run_end validation fired
    """
```

**Separate focused test:** Additionally, a simpler test verifies just the init output:
```python
def test_init_generates_correct_yaml_structure(tmp_path):
    """Verify init --yes --preset produces expected YAML sections/ordering."""
```

### Acceptance Criteria Coverage

| Spec AC | Covered By |
|---------|------------|
| R1 — Evidence defaults + edit | `_compute_evidence_defaults()`, checkbox UI with questionary, integration tests |
| R2 — Intersection for preset path | `_compute_evidence_defaults()` logic, unit test |
| R3 — run_end trigger defaults | `_compute_trigger_defaults()` logic, fixed settings (fire_on: success, remediate, max_retries: 3), integration tests |
| R4 — Trigger table format | `_print_trigger_reference_table()`, semantic content tests |
| R5 — Checkbox multi-select | questionary.checkbox() usage, confirm + checkbox pattern |
| R6 — Config validation | Existing validate_init_config(), validation error handling flow |
| R7 — Skip sections | questionary.confirm() before each checkbox, omitted from YAML if skipped |
| R8 — Custom commands flow | `is_preset=False` branch in default computation, integration tests |
| R9 — Table always printed | Unconditional call to `_print_trigger_reference_table()` at end of init() |
| R10 — YAML schema | `_build_validation_triggers_dict()` with ref: syntax, YAML output tests |

## Open Questions

- None remaining (all resolved during interview)

## Next Steps
After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
