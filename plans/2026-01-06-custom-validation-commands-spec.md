# Custom Validation Commands

**Tier:** M
**Owner:** [TBD]
**Target ship:** [TBD]
**Links:** [Related: mala.yaml config system, ValidationSpec, CommandsConfig]

## 1. Outcome & Scope

**Problem / context**
Mala's validation pipeline is currently limited to built-in command categories (`setup`, `format`, `lint`, `typecheck`, `test`, `e2e`). Users who need project-specific validation (e.g., import linters, architecture validators, custom scripts) cannot configure these to run as a first-class part of validation with consistent ordering, override behavior, and evidence/quality-gate integration.

**Goal**
Enable users to define and run arbitrary project-specific validation commands as part of mala's validation pipeline, with configurable advisory-only behavior and run-level overrides.

**Success criteria**
- Repos with `custom_commands` configured run those commands in the expected pipeline position and in dict insertion order.
- `allow_fail: true` custom commands do not cause overall validation to fail, while still surfacing their failure status.
- Run-level overrides for `custom_commands` take effect for that run without modifying repo config.
- Existing repos with no `custom_commands` behave the same as before.

**Non-goals**
- Supporting `custom_commands` in presets (user-only configuration).
- Custom command result parsing (beyond pass/fail exit code).
- Parallel execution, retries, caching, or sandboxing of custom commands.
- User-configurable execution order (fixed position: after lint/format/typecheck, before test).
- Defining new built-in tools; the feature runs user-provided commands as-is.

## 2. User Experience & Flows

**Primary flow**
1. User adds `custom_commands` dict to mala.yaml (with optional `allow_fail` per command).
2. User runs validation (same command as today).
3. Mala runs built-in phases (setup, format, lint, typecheck), then custom commands in insertion order, then test.
4. Results show each custom command outcome; advisory failures are visible but do not fail the overall run.

**Key states**
- Empty state: No `custom_commands` configured -> validation output/pipeline matches current behavior.
- Loading state: While a custom command runs, progress visible via same mechanism as other commands.
- Success state: Custom commands pass -> validation proceeds to test phase.
- Error state(s):
  - Invalid config -> clear ConfigError identifying the problematic entry.
  - Strict failure: command fails with `allow_fail: false` (default) -> validation aborts.
  - Advisory failure: command fails with `allow_fail: true` -> warning logged, validation continues.

**Example mala.yaml configuration:**
```yaml
preset: python-uv

custom_commands:
  # Simple string form
  import_lint: "uvx import-linter --config pyproject.toml"

  # Object form with options
  arch_check:
    command: "uvx grimp check src/"
    timeout: 120
    allow_fail: true  # Advisory only, won't block validation

run_level_commands:
  custom_commands:
    import_lint: "uvx import-linter --config pyproject.toml --verbose"
```

## 3. Requirements + Verification

**R1 — Config Schema**
- **Requirement:** The system MUST parse `custom_commands` from mala.yaml as an ordered dict where keys are command names and values are either:
  - **String shorthand:** `"command string"` (defaults: timeout=120s, allow_fail=false)
  - **Object form:** `{command: "...", timeout: <seconds>, allow_fail: <bool>}` where only `command` is required
- **Config validation:**
  - Command names (dict keys) MUST match regex `^[A-Za-z_][A-Za-z0-9_]*$` (valid Python identifiers).
  - Unknown keys in object form MUST raise ConfigError (consistent with existing validation).
  - `timeout` is in seconds (default: 120). `allow_fail` defaults to `false`.
  - `null` values in dict MUST raise ConfigError ("use run-level override to disable commands").
  - Empty or whitespace-only command strings MUST raise ConfigError.
  - Presets MUST NOT define `custom_commands`; if a preset includes `custom_commands`, raise ConfigError during preset loading ("presets cannot define custom_commands").
- **Verification:** Configure `custom_commands` with shorthand and object entries; verify defaults apply and invalid names/keys/nulls/empty-commands are rejected.

**R2 — Execution Order and Environment**
- **Requirement:** The system MUST execute custom commands after format/lint/typecheck and before test/e2e, preserving dict insertion order when multiple custom commands are defined.
- **Full pipeline order:** setup → format → lint → typecheck → **custom commands** → test → e2e
- **Phase skipping:** If any built-in phases are disabled/not configured, custom commands still run in their fixed position (after the last lint-like phase, before test/e2e). If no lint-like phases exist, custom commands run after setup.
- **Execution environment:** Custom commands run via shell (`shell=True`), from the repository root directory, inheriting the same environment as built-in validation commands.
- **Timeout behavior:** When a command exceeds its timeout, the system MUST terminate it and treat it as a failure (subject to `allow_fail` semantics). Timeout failures are logged as "timed out after {n}s".
  - **Unix:** SIGTERM to process group, then SIGKILL after 5 second grace period. Targets the process group to ensure child processes spawned by shell don't outlive the runner.
  - **Windows:** TerminateProcess equivalent (mala follows platform conventions of existing built-in command execution).
- **Verification:** Configure `custom_commands: {cmd_a: "...", cmd_b: "..."}` plus lint, test, and e2e; confirm execution order is: format → lint → typecheck → cmd_a → cmd_b → test → e2e.

**R3 — Advisory Failures (allow_fail)**
- **Requirement:** The system MUST abort validation when a custom command fails, UNLESS `allow_fail: true` is set for that command.
- **Verification:**
  - Given `allow_fail: false` (default) and command exits 1, then validation aborts and test does not run.
  - Given `allow_fail: true` and command exits 1, then warning is logged and validation proceeds to test.

**R4 — Run-Level Override (Full Replace)**
- **Requirement:** The system MUST allow custom commands to be overridden at run-level via `run_level_commands.custom_commands`. When specified, run-level custom_commands **fully replaces** (not merges with) repo-level custom_commands.
- **Override source:** `run_level_commands` is defined in `mala.yaml` (repo config). The example in Section 2 shows the YAML structure. This mechanism exists for configuring different command sets for per-issue vs run-level validation scopes (e.g., run-level test command with coverage flags). It is not an ephemeral/CLI override—it is static repo configuration that applies when validation runs in run-level scope.
- **Override semantics:**
  - Omitted `run_level_commands.custom_commands`: repo-level custom_commands run as normal.
  - `run_level_commands.custom_commands: {}` (empty dict): disables all custom commands for this run.
  - `run_level_commands.custom_commands: null`: same as omitted (repo-level applies).
- **Verification:**
  - With repo-level `custom_commands: {cmd_a: "...", cmd_b: "..."}` and run-level `custom_commands: {cmd_a: "..."}`, only cmd_a runs (cmd_b is not inherited).
  - With run-level `custom_commands: {}`, no custom commands run (repo-level disabled).

**R5 — Evidence Detection (Structured Markers)**
- **Requirement:** The validation runner MUST log structured markers to the session log for each custom command:
  - **Start marker:** `[custom:<name>:start]` logged before execution
  - **End marker:** `[custom:<name>:pass]` or `[custom:<name>:fail exit=<code>]` or `[custom:<name>:timeout]` logged after execution
- **Missing end marker:** If a configured custom command has a start marker without a corresponding end marker (e.g., run crashed/aborted, logs truncated), the gate MUST treat it as a failure for that command (subject to `allow_fail` semantics). This prevents incomplete runs from incorrectly passing.
- **QualityGate changes:** Evidence parsing MUST track custom commands by name (not just CommandKind). The `ValidationEvidence` structure MUST be extended to track which specific custom commands ran and their pass/fail status, respecting `allow_fail` semantics (advisory failures don't fail the gate).
- **QualityGate source of truth:** The gate validates against the **active run-level config** (after override resolution), not the repo-level config. If run-level overrides remove a command, the gate does not expect evidence for that command.
- **Verification:** Configure `{cmd_a: "...", cmd_b: "..."}`. Run both and verify gate detects both markers and tracks individual pass/fail status correctly. Verify that a run with only `[custom:cmd_a:start]` (no end marker) is treated as a failure.

**R6 — Agent Awareness (Prompt Integration)**
- **Requirement:** Custom commands MUST be surfaced to agents via `PromptValidationCommands` so agents know to execute them. The prompt template MUST include:
  - Custom command names and their command strings
  - `allow_fail` status for each command (so agents know advisory failures don't block)
  - Instruction that advisory failures should be noted but not fixed before proceeding
- **Verification:** Configure custom commands (including one with `allow_fail: true`); verify the implementer prompt includes instructions to run them in the correct order with advisory status indicated.

## 4. Instrumentation & Release Checks

**Instrumentation**
- Events to track: custom command start/complete, pass/fail status with exit code, `allow_fail` usage.
- Log format: `[custom:<name>:start]` before execution, `[custom:<name>:pass|fail|timeout]` after execution.
- Detection: Quality gate matches start/end marker pairs to verify each configured custom command ran and extract status.

**Implementation notes (non-normative)**
- Add `CommandKind.CUSTOM = "custom"` to enum (follows existing pattern).
- Add `custom_commands` field to `CommandsConfig` dataclass as `dict[str, CustomCommandConfig]`.
- Add `CustomCommandConfig` dataclass with fields: `command: str`, `timeout: int = 120`, `allow_fail: bool = False`.
- Extend `ValidationEvidence` to track `custom_commands_ran: dict[str, bool]` and `custom_commands_failed: dict[str, bool]`.
- Extend `PromptValidationCommands` with `custom_commands: list[tuple[str, str, bool]]` (name, command, allow_fail).
- Runner logs markers via structured logging: `[custom:<name>:start]` before, `[custom:<name>:pass|fail exit=N|timeout]` after.
- Custom command names displayed in error messages use dict key (e.g., "custom command 'import_lint' failed").
- `run_level_commands.custom_commands` uses the same schema as top-level `custom_commands`.
- Top-level `custom_commands: {}` (empty dict) is valid and results in no custom commands running (same as omitting).

**Open questions**
- None.
