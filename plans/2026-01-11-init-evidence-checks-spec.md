# Mala Init: Evidence Checks and Trigger Configuration

**Tier:** M
**Owner:** Mala CLI team
**Shipped:** 2026-01-11
**Links:** [evidence_check spec](../plans/2026-01-10-evidence-check-config-spec.md), [validation-triggers.md](validation-triggers.md)

## 1. Outcome & Scope

**Problem / context**
The `mala init` command currently generates presets with command definitions (test, lint, typecheck, format) but does not configure `evidence_check` or `validation_triggers`. This forces users to manually edit `mala.yaml` after initialization to establish a working validation loop, increasing friction and the likelihood of configuration errors. Users want an opt-out, default-driven init experience that produces a valid config out of the box.

**Goal**
Enable users to configure evidence checks and validation triggers during `mala init` with sensible defaults so that new projects start with complete quality gate configuration without requiring manual YAML editing.

**Success criteria**
- Given a fixture repo with stub commands that always exit 0 (e.g., `test: "echo ok"`) and a fresh init config with defaults accepted, running `mala run` with a single issue produces logs showing each configured command executed and `run_end` validation fired (behavioral verification)
- Users can complete init without configuring evidence/trigger sections (explicit skip) while still producing a valid config that passes `mala run` without errors

**Non-goals**
- Adding new preset types beyond python-uv, go, node-npm, rust
- Changing the meaning/behavior of existing validation trigger types or validation rules
- Interactive configuration of triggers other than run_end during init (users can edit YAML later)
- Granular configuration of failure_mode or max_retries per individual command during init (global defaults will be used)

## 2. User Experience & Flows

**Primary flow**
1. User runs `mala init` and selects a preset (or chooses custom commands)
2. System prompts: "Configure evidence checks? [Y/n]"
   - If Y (default): Show checkbox multi-select with intersection of `{test, lint}` and available commands pre-checked
   - If n: Skip evidence_check section
3. User confirms defaults or modifies selection in checkbox UI (Space to toggle, Enter to confirm)
4. System prompts: "Configure run_end validation trigger? [Y/n]"
   - If Y (default): Show checkbox multi-select with validation commands pre-checked (excludes setup/e2e)
   - If n: Skip validation_triggers section
5. User confirms defaults or modifies selection in checkbox UI
6. System generates and validates the configuration
7. System writes `mala.yaml` (with backup if applicable)
8. System prints trigger quick-reference table (always, regardless of skip choices)

**Skip mechanism:** Two-step prompt pattern: (1) Y/n confirm to configure section, (2) checkbox multi-select if confirmed. This avoids keybind conflicts with questionary's text filtering. If user unchecks all boxes and presses Enter, prompt: "No commands selected. Skip this section? [Y/n]". If Y, omit section. If n, keep the section with an empty command list (meaning no evidence/trigger commands required).

**Key states**
- **Defaults accepted:** User presses Enter (Y) through all prompts; full config generated with evidence_check and run_end trigger
- **Partially customized:** User toggles some checkboxes (e.g., removes lint from evidence_check)
- **Minimal config:** User answers 'n' to skip evidence_check and/or triggers; those sections omitted from generated config
- **Empty selection kept:** User unchecks all and answers "No" to skip; section is written with an empty list
- **Empty commands state:** If no commands exist after preset/custom selection, init displays: "No commands defined. Skipping evidence check and trigger configuration." and proceeds to write config without those sections
- **Validation error state:** If validation fails (should not happen with UI constraints), init displays error and uses `questionary.select()` to offer: "Revise selections", "Skip section", or "Abort init" (consistent terminal handling)

**Schema expectations (observable contract):**
- Init must write YAML that passes the same validation as `mala run`
- `evidence_check` key is optional; absence means no evidence requirements (empty list also allowed)
- `validation_triggers` key is optional; absence means no triggers configured (empty run_end command list is allowed)
- Both can be omitted and `mala run` will execute normally without validation

**Non-interactive mode:**
- When stdin is not a TTY (CI, piping, automation), init fails with: `Error: Non-interactive mode requires --yes with --preset, or --preset with --skip-evidence and --skip-triggers`
- `--yes` flag: Accept all defaults without prompts (requires `--preset`)
- `--preset <name>`: Specify preset non-interactively
- `--skip-evidence` / `--skip-triggers`: Omit sections non-interactively (requires `--preset`)

**Interaction model:** Use `questionary` library for all interactive prompts (both Y/n confirms via `questionary.confirm()` and checkbox multi-select via `questionary.checkbox()`). This ensures consistent terminal handling via prompt-toolkit. Adding `questionary` introduces a new runtime dependency (MIT license, Python 3.8+, no conflicts with existing deps).

## 3. Requirements + Verification

**R1 — Evidence defaults + edit**
- **Requirement:** The system MUST present default `evidence_check` selections during `mala init` and allow the user to confirm or modify them using a checkbox multi-select UI.
- **Verification:** Select python-uv preset. Evidence check prompt shows `test` and `lint` pre-selected. Accept defaults. Generated `mala.yaml` contains `evidence_check: { required: [test, lint] }`.

**R2 — Evidence default set uses intersection (preset path only)**
- **Requirement:** For preset selection, the system MUST default `evidence_check.required` to the intersection of `{test, lint}` with the commands available in the selected preset. For custom commands, see R8 (no pre-selected defaults).
- **Verification:** Select python-uv preset (has test, lint). Evidence prompt shows `test` and `lint` pre-checked.

**R3 — run_end trigger defaults**
- **Requirement:** The system MUST default `validation_triggers.run_end` commands as follows:
  - **Preset path:** Include all commands except those with ref names **exactly matching** `setup` or `e2e` (case-sensitive string comparison, not semantic). Commands named `bootstrap`, `install`, `integration`, etc. are NOT excluded.
  - **Custom commands path:** No pre-selected defaults (see R8)
  - Trigger settings: `fire_on: success` (implicit default), `failure_mode: continue` (explicit), `max_retries` omitted
- **Verification:** Select go preset (defines setup, test, lint, format), accept defaults. Generated config contains run_end with commands `[test, lint, format]` (only `setup` excluded by exact name match).

**R4 — Trigger quick-reference table format**
- **Requirement:** After init completes, the system MUST print a quick-reference table with these semantic requirements:
  - Title: Contains "Available Trigger Types" (case-insensitive)
  - Columns: Trigger, Description
  - Rows (semantic inclusion, not byte-for-byte):
    - Contains `epic_completion` with description mentioning "epic" and "verification"
    - Contains `session_end` with description mentioning "session" and "ends"
    - Contains `periodic` with description mentioning "N issues"
    - Contains `run_end` with description mentioning "mala run completes"
  - Formatting: Aligned columns; border style flexible
- **Verification:** Tests assert semantic content (trigger names present, descriptions contain key phrases). Exact wording, punctuation, and formatting may vary.

**R5 — Checkbox multi-select UI**
- **Requirement:** The system MUST use `questionary` for all interactive prompts: `questionary.confirm()` for Y/n skip decisions and `questionary.checkbox()` for command selection. This requires adding `questionary` to project dependencies.
- **Verification:** At evidence check step, user can use Space to toggle, Enter to confirm. Multiple commands can be toggled before confirming.
- **Ordering (end-to-end):**
  - Checkbox options appear in preset-defined order (as listed in preset YAML) or entry order (for custom commands)
  - Generated YAML lists commands in the same order as displayed in checkbox
  - Implementation: Use ordered data structures (list, OrderedDict) throughout; do not use unordered dict/set for command storage

**R6 — Config validation checks**
- **Requirement:** The generated configuration MUST pass `validate_init_config()` (located in `src/orchestration/cli_support.py`) before writing. This function validates the in-memory config dict against the full `ValidationConfig` schema. Validation checks:
  - `evidence_check.required` entries reference commands in the resolved command map
  - `validation_triggers.run_end.commands[].ref` entries reference commands in the resolved command map
  - `failure_mode` is one of: `abort`, `continue`, `remediate`
  - `max_retries` is a non-negative integer (required when failure_mode is `remediate`)
  - Omitted sections are valid (see Schema expectations above)
- **Verification:** UI constraints prevent invalid selections; validation is a safety net. Same validator used by `mala run`.

**R7 — Skip/omit sections**
- **Requirement:** The system MUST provide Y/n prompt before each checkbox UI to skip sections. Skipped sections are omitted from generated config entirely. If the user declines to skip after selecting none, the section is written with an empty list (e.g., `evidence_check: { required: [] }`).
- **Verification:** Answer 'n' to "Configure evidence checks?" prompt. Generated config has no `evidence_check` key. Config passes validation and `mala run` works normally.

**R8 — Custom commands flow (evidence and triggers)**
- **Requirement:** When user enters custom commands (not a preset), the system MUST prompt for both evidence and trigger commands with **no pre-selected defaults**. This differs from preset path where defaults are pre-checked.
  - Rationale: Custom command names are user-defined and may not follow conventions (e.g., user might name their test command `unit` or `check`), so no assumptions about which commands are validation-related.
- **Verification:** Enter custom commands `test`, `lint`, `mypy`. Evidence prompt shows all three with none pre-checked. Trigger prompt shows all three with none pre-checked. User must explicitly select commands for each section.

**R9 — Trigger table always printed**
- **Requirement:** The trigger quick-reference table MUST be printed after every successful init, regardless of whether trigger section was configured or skipped.
- **Verification:** Skip both evidence and triggers. Table still prints with all 4 trigger types.

**R10 — Generated YAML schema**
- **Requirement:** Generated config MUST use the correct YAML structure with `ref:` syntax for trigger commands.
- **Verification:** Accept defaults for python-uv preset. Generated `mala.yaml` contains:
  ```yaml
  preset: python-uv

  evidence_check:
    required: [test, lint]

  validation_triggers:
    run_end:
      failure_mode: continue
      commands:
        - ref: test
        - ref: lint
        - ref: format
        - ref: typecheck
  ```
  Note: python-uv defines setup, test, lint, format, typecheck, e2e. Run_end excludes setup (one-time) and e2e (expensive).

## 4. Instrumentation & Release Checks

**Instrumentation**
- Events to track (optional, best-effort logging to stdout/stderr; no external telemetry required):
  - init_started, preset_selected vs custom_path, evidence_config_modified, evidence_skipped, trigger_config_modified, triggers_skipped, init_completed, init_validation_failed
- These events are for debugging/development; not a ship-blocking requirement. No analytics backend integration needed.

**Decisions made**
- evidence_check default (preset path): intersection of `{test, lint}` with available commands
- evidence_check default (custom path): no pre-selected defaults
- Trigger default (preset path): all commands except `setup` and `e2e` by **exact case-sensitive name match** (not semantic)
- Trigger default (custom path): no pre-selected defaults
- Trigger settings: fire_on defaults to success (omitted), failure_mode: continue, max_retries omitted
- UX library: Use `questionary` for all prompts (confirm, checkbox, select) for consistent terminal handling
- Non-interactive mode: Fail with actionable error; support `--yes`, `--preset`, `--skip-evidence`, `--skip-triggers` flags
- Skip mechanism: questionary.confirm() before each checkbox UI
- Empty selection behavior: Unchecking all prompts "Skip this section?"; choosing "No" writes an empty list
- Checkbox ordering: Preset-defined order for presets; entry order for custom commands; preserved to YAML output
- Trigger quick-reference table: Semantic content required (key phrases), formatting flexible; printed after every init
- failure_mode and max_retries: Fixed defaults during init (not editable); users can modify in YAML later
- YAML schema: Use `ref:` syntax for trigger commands (e.g., `- ref: test`)
- Schema optionality: Init writes YAML that passes same validation as `mala run`; evidence_check and validation_triggers are optional
- Verification fixtures: Use stub commands (exit 0) in test repos; assert logs show commands executed and run_end fired

**Open questions**
- None remaining
