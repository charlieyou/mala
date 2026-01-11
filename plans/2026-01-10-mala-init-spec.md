# mala init - Interactive CLI Config Generator

**Tier:** M
**Owner:** Mala CLI team
**Target ship:** Next release
**Links:** [docs/project-config.md](project-config.md)

## 1. Outcome & Scope

**Problem / context**
New users must manually create `mala.yaml` from scratch or copy from documentation. This creates friction during onboarding and leads to common mistakes (typos, missing required fields, invalid preset names). The docs are comprehensive but spread across multiple examples.

**Goal**
Enable users to run `mala init` to interactively generate a valid `mala.yaml` file tailored to their project's tech stack, so they can get started quickly without reading extensive documentation.

**Success criteria**
- Given a new empty directory, `mala init` with any preset produces a config that passes `mala run --dry-run` with exit code 0
- Generated configs are validated before writing (pass ValidationConfig.from_dict() + _validate_config())
- UX goal: users can complete the wizard in under 60 seconds (not a hard requirement)

**Non-goals**
- Detecting project type automatically from existing files (e.g., pyproject.toml, package.json) - users select manually
- Supporting monorepo configurations (multiple languages)
- Generating validation_triggers config (advanced feature, added later)
- Generating coverage config (can be added manually post-init)
- Introducing new dependencies (use existing typer only)

## 2. User Experience & Flows

**Primary flow**
1. User runs `mala init` (optionally with `--dry-run`)
2. CLI displays numbered menu: `1) python-uv  2) node-npm  3) go  4) rust  5) custom`
3. User enters number 1-5 to select option
4. If preset selected (1-4): generate config with `preset: <name>`
5. If "custom" selected (5): CLI prompts for each of the 6 standard commands (setup, test, lint, format, typecheck, e2e)
6. CLI validates generated config by calling `validate_generated_config(dict)` helper (wraps ValidationConfig.from_dict() + _validate_config() from `src.domain.validation.config_loader`)
7. If validation fails (ConfigError raised): show error message to stderr and exit with code 1 (no YAML preview)
8. If not `--dry-run` and `mala.yaml` exists: backup to `mala.yaml.bak` (overwriting existing .bak)
9. If not `--dry-run`: write `mala.yaml` to current directory
10. CLI displays generated file contents and "Run `mala run` to start" tip

**Key states**
- Selection state: Numbered menu, user enters 1-5
- Input state: Text prompts for custom command entry (empty = skip command)
- Dry-run state: With --dry-run, show preview without writing or backing up files
- Success state: mala.yaml written, backup created if needed, shows file contents and next steps
- Error state(s):
  - Validation failure: show error, exit code 1, no files modified
  - Backup failure: abort, do not overwrite original, exit code 1
  - Write failure after backup: keep .bak, report error with file paths, exit code 1
  - Ctrl+C: exit code 130 (SIGINT), no files modified

**Generated YAML contract**

Serialization rules (semantic equivalence required, exact formatting is not):
- Valid YAML that parses to the intended dict
- Block style (no flow style)
- Trailing newline at end of file
- PyYAML with `yaml.dump(..., default_flow_style=False, sort_keys=True)`
- Tests should assert parsed content, not exact string formatting

Preset selection (e.g., python-uv):
```yaml
preset: python-uv
```

Custom with some commands:
```yaml
commands:
  lint: "ruff check"
  test: pytest
```

Custom with all commands empty: generates empty commands dict, which fails validation with "At least one command must be defined" - CLI shows this error and suggests selecting a preset or providing at least one command.

**Invalid input handling**
- Invalid menu selection (not 1-5): show "Invalid choice, please enter 1-5:" and re-prompt
- EOF (Ctrl+D): exit with code 1, no files modified
- Whitespace-only command input: treated as empty (command skipped)

## 3. Requirements + Verification

**R1 — Command availability**
- **Requirement:** The CLI MUST provide a `mala init` command integrated into the existing typer-based CLI
- **Verification:**
  - Run `mala --help` and verify `init` is listed
  - Run `mala init --help` and verify usage shows `--dry-run` option

**R2 — Preset selection UI**
- **Requirement:** The system MUST present available presets (python-uv, node-npm, go, rust) plus "custom" option using numbered text input (1-5), and MUST NOT attempt auto-detection
- **Verification:**
  - Run `mala init` and verify all 5 options appear as numbered list
  - Enter "1" to select python-uv and verify generated config contains only `preset: python-uv`

**R3 — Custom command entry**
- **Requirement:** When "custom" is selected, the system MUST prompt for all 6 standard commands (setup, test, lint, format, typecheck, e2e) and MUST omit commands where user enters empty string
- **Verification:**
  - Select "custom", enter `pytest` for test and `ruff check` for lint, leave others empty
  - Verify generated YAML contains only the two provided commands and passes validation

**R4 — File backup on existence**
- **Requirement:** When mala.yaml exists and --dry-run is not set, the system MUST backup to mala.yaml.bak before overwriting (overwriting any existing .bak file)
- **Verification:**
  - Create mala.yaml with content "preset: go", run `mala init`
  - Verify mala.yaml.bak contains "preset: go" and mala.yaml has new content
  - Run `mala init` again, verify previous .bak is overwritten with current mala.yaml content

**R5 — Output validation**
- **Requirement:** The system MUST validate generated config via `validate_generated_config()` helper (which wraps ValidationConfig.from_dict() + _validate_config() from `src.domain.validation.config_loader`) before writing (and before printing in --dry-run)
- **Verification:**
  - If internal generation produces invalid config (e.g., empty commands), wizard exits with error to stderr and does not write mala.yaml
  - In --dry-run, validation errors are shown to stderr instead of printing YAML to stdout
- **Output contract:** On success: print YAML to stdout after write completes (or after validation in dry-run). On error: print error to stderr, stdout is empty.

**R6 — Dry-run support**
- **Requirement:** The system MUST support `--dry-run` flag to preview output without writing files or creating backups
- **Verification:**
  - Run `mala init --dry-run` with existing mala.yaml
  - Verify YAML is printed to stdout but neither mala.yaml nor mala.yaml.bak are modified

**R7 — Success output**
- **Requirement:** After successful write, the system MUST display the generated file contents and a tip to run `mala run`
- **Verification:**
  - Complete wizard flow successfully
  - Verify terminal output includes full YAML contents and contains text "mala run"

## 4. Instrumentation & Release Checks

**Testing strategy**
- Integration tests use typer.testing.CliRunner to simulate stdin and verify flows programmatically
- Use temp directories to verify filesystem effects (backup creation, file writes)
- Assert exit codes (0 for success, 1 for errors, 130 for Ctrl+C) and stderr messages
- Verify --dry-run produces no file changes via filesystem assertions

**Validation after release**
- How to confirm this worked: Run `mala init` in a new directory, select a preset, verify mala.yaml is valid via `mala run --dry-run`
- Known risks / blast radius: Low risk - new command, no changes to existing behavior

**Instrumentation**
- None required for MVP (no telemetry in mala currently)
- Future consideration: track init_completed with preset/custom attribute if telemetry added

**Decisions made**
- File existence: Backup to .bak and overwrite (overwrites existing .bak)
- Auto-detection: Disabled (user chose simplicity over magic)
- Prompt style: Numbered text input (1-5) via typer.prompt - no new dependencies
- Custom commands: All 6 standard commands prompted (setup, test, lint, format, typecheck, e2e)
- Coverage: Not prompted (user adds manually if needed)
- Output: Show file contents + next steps tip
- Dependencies: No new dependencies - use existing typer only
- Empty custom: Allow empty commands dict (validation will catch and show helpful error)
- Backup collision: Overwrite existing .bak (most recent backup wins)
- Ctrl+C: Exit code 130 (standard SIGINT)

**Implementation considerations**
- Selection: Use typer.prompt with choices parameter or manual number input validation
- YAML output: Clean and minimal using pyyaml default_flow_style=False
- Backup logic: shutil.copy(mala.yaml, mala.yaml.bak) before writing new content
- Validation: Parse generated dict through ValidationConfig.from_dict() before writing
- File location: Write to current working directory (CWD)
- Error handling: If backup fails, abort without modifying original; if write fails after backup, keep .bak and report both paths

**Open questions**
- Should wizard support --preset flag to skip interactive selection for scripted usage? (Deferred to future enhancement)
