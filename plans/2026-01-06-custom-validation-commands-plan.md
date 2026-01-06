# Implementation Plan: Custom Validation Commands

## Context & Goals
- **Spec**: `/home/cyou/mala/plans/2026-01-06-custom-validation-commands-spec.md`
- Enable users to define arbitrary project-specific validation commands (e.g., import linters, architecture validators) in mala.yaml
- Run custom commands after lint/typecheck, before test, with configurable `allow_fail` for advisory-only checks
- Integrate with quality gate evidence detection and agent prompts for complete pipeline awareness

**Important codebase findings:**
- `ValidationCommand` already has `name: str` and `allow_fail: bool` fields — no schema changes needed there
- `PromptValidationCommands` is defined in `config.py` (line 486), not `prompts.py`
- Current `DEFAULT_KILL_GRACE_SECONDS = 2.0` (not 5s as spec states)

## Scope & Non-Goals
- **In Scope**
  - `custom_commands` dict in mala.yaml (string shorthand + object form) [R1]
  - `allow_fail: true` for advisory-only checks that don't block validation [R3]
  - Run-level override via `run_level_commands.custom_commands` (full replace semantics) [R4]
  - Evidence detection with structured markers `[custom:<name>:start|pass|fail|timeout]` [R5]
  - Agent prompt integration via `PromptValidationCommands` [R6]

- **Out of Scope (Non-Goals)**
  - Preset support for custom_commands (user-only configuration)
  - Parallel execution, retries, caching, or sandboxing of custom commands
  - User-configurable execution order (fixed position: after lint/format/typecheck, before test)
  - Result parsing beyond pass/fail exit code
  - Defining new built-in tools; runs user-provided commands as-is

## Assumptions & Constraints
- **Data model**: New `CustomCommandConfig` dataclass (not extending `CommandConfig`) for cleaner separation and custom-specific defaults (allow_fail field)
- Python 3.7+ dict insertion order guarantee ensures command ordering
- Existing `shell=True` execution pattern applies to custom commands
- **Timeout grace period (P2 fix):** Spec says 5s, but `DEFAULT_KILL_GRACE_SECONDS = 2.0`. Use existing 2s default to match current behavior; update spec if needed (non-breaking since custom commands are new)

### Implementation Constraints
- Must follow existing patterns for config validation (`ConfigError`, `_fields_set` tracking)
- Evidence parsing must handle by-name tracking (extend beyond `CommandKind`-based tracking)
- Custom commands use `CommandKind.CUSTOM` but are tracked individually by name in evidence

**Config plumbing (P1 fix):**
- `custom_commands` is a **top-level field** in mala.yaml (peer to `preset`, `commands`, `coverage`, etc.)
- Must add `"custom_commands"` to `_ALLOWED_TOP_LEVEL_FIELDS` in `config_loader.py`
- Must add `custom_commands: dict[str, CustomCommandConfig]` field to `ValidationConfig` dataclass
- **YAML-to-dataclass mapping**: `ConfigLoader` already maps top-level YAML fields to `ValidationConfig` fields (e.g., `preset`, `commands`); `custom_commands` follows same pattern
- `merge_configs()` must preserve user `custom_commands` when merging with preset (presets forbidden from defining it)

**Preset prohibition (P2 fix):**
- Check must be in `PresetRegistry.get()` since presets bypass `_validate_schema()`
- Add explicit check: if preset YAML contains `custom_commands`, raise `ConfigError("presets cannot define custom_commands")`

**Marker emission path (P1 fix):**
- Markers are NOT emitted via internal `print()` — that's not captured in Claude JSONL logs
- Instead, markers are emitted **by the agent** when it executes custom commands via the Bash tool
- The `implementer_prompt.md` template instructs the agent to use wrapper pattern that:
  1. Echoes start marker
  2. Runs command with `timeout` wrapper
  3. Captures exit status
  4. Echoes pass/fail/timeout marker based on status
  5. Exits with original status (strict) or 0 (advisory)
- Markers appear in Bash tool_result content, which the log parser can extract

**Evidence parsing (P1 fix):**
- Extend `LogProvider` protocol with `extract_tool_result_content(entry) -> list[tuple[str, str]]` (tool_use_id, content)
- `parse_validation_evidence_with_spec()` extracts tool_result content and scans for custom markers
- Regex pattern: `\[custom:(\w+):(start|pass|fail exit=\d+|timeout)\]`

**Content extraction rules (P2):**
- `tool_result.content` may be non-string (`Any` type); extraction strategy:
  - If `str`: use directly
  - If `list`: concatenate string elements, skip non-strings
  - Otherwise: `str(content)` as fallback
- Scope extraction to Bash tool results only (check `tool_use.name == "Bash"`)

**Marker precedence rules (P2):**
- Terminal markers: `pass`, `fail`, `timeout` (mutually exclusive outcomes)
- Latest terminal marker wins (handles retries/re-runs)
- Any terminal marker implies "ran" even if `start` marker is absent (handles output truncation)
- If `start` marker exists but no terminal marker: treat as failure (incomplete execution)
- If custom command in spec but no markers at all: treat as "not run" (distinguishes "agent skipped" from "agent failed to emit markers")

**Name validation consistency (P2):**
- Config validation regex: `^[A-Za-z_][A-Za-z0-9_]*$` (Python identifier)
- Marker parsing regex: `\[custom:(\w+):...` where `\w` = `[a-zA-Z0-9_]`
- These are consistent: any valid config name will be captured by the marker regex
- Implementation note: ensure no divergence between validation and parsing regexes

**LogProvider protocol churn (P2):**
- Implementations to update: `SessionLogParser` (primary), `FakeLogProvider` (tests)
- Fallback behavior: return empty list `[]` for implementations that don't support content extraction
- This is additive (new method), so existing implementations continue to work for non-custom-command flows
- **Alternative considered**: Extend existing `extract_tool_results()` to return content. Rejected because it changes return type and existing callers only need `(tool_use_id, is_error)`. New method is cleaner separation.

**YAML order preservation (P2):**
- Python 3.7+ guarantees dict insertion order; PyYAML preserves mapping order
- Add regression test: parse `custom_commands: {a: ..., b: ..., c: ...}` and assert spec command order is `[a, b, c]`
- Document in config.py docstring that command execution order follows YAML key order

**`run_level_commands.custom_commands` null vs {} semantics (P2 fix):**
- `null` (explicit) = same as omitted → repo-level `custom_commands` applies
- `{}` (empty dict) = disable all custom commands for this run
- Implementation: use `_fields_set` tracking differently for `custom_commands`:
  - Track presence in `_fields_set` only when value is a non-null dict
  - In `_apply_command_overrides()`: if `custom_commands` in `overrides._fields_set`, use override (even if empty dict); else use base
  - This differs from other commands where explicit null means "disable"

### Testing Constraints
- **Integration-heavy**: Focus on end-to-end tests with real config loading and command execution
- Unit tests for parsing and validation logic
- Integration tests for command execution flow and evidence detection
- Must verify backward compatibility: repos without `custom_commands` behave identically

## Prerequisites
- [x] All required infrastructure exists (validation config, spec runner, quality gate)
- [ ] **System requirement**: Shell `timeout` command (GNU coreutils) for command timeout enforcement
  - Linux: Available by default (coreutils)
  - macOS: Not available by default; users must `brew install coreutils` and use `gtimeout`, OR the prompt template should detect platform and use Python-level timeout fallback
  - **Recommendation**: Document as macOS limitation in release notes; consider future enhancement for cross-platform timeout

## High-Level Approach

Implementation proceeds in three phases:

1. **Phase 1 - Config & Spec (Foundation)**: Add `CustomCommandConfig` dataclass, extend `ValidationConfig` with top-level `custom_commands` field, add `"custom_commands"` to `_ALLOWED_TOP_LEVEL_FIELDS`, add `CommandKind.CUSTOM` enum value. Update config parsing to handle string shorthand and object form. Validate command names (Python identifiers), reject presets with `custom_commands` (in `PresetRegistry.get()`), ensure `merge_configs()` preserves user custom_commands, and build custom `ValidationCommand` instances in the correct pipeline position.

2. **Phase 2 - Execution & Evidence (Runtime)**: Update `implementer_prompt.md` with marker-wrapped command patterns for agent execution. Extend `LogProvider` protocol with `extract_tool_result_content()`. Extend `ValidationEvidence` to track custom commands by name (ran, failed only — not allow_fail). Update quality gate to parse markers from tool_result content and check evidence against spec (looking up `allow_fail` from spec, not evidence).

3. **Phase 3 - Prompt Integration (Agent Awareness)**: Extend `PromptValidationCommands` with `custom_commands` field. Update `implementer_prompt.md` template to include custom commands in the validation sequence with `allow_fail` status indicated.

## Technical Design

### Architecture

Custom commands flow through the existing validation pipeline with targeted extensions:

```
mala.yaml → ConfigLoader → ValidationConfig.custom_commands (top-level)
                               ↓
                        build_validation_spec()
                               ↓
         ValidationCommand(kind=CUSTOM, name=<key>, allow_fail=...)
                               ↓
                    Agent reads PromptValidationCommands
                               ↓
         Agent executes: echo '[custom:<name>:start]' && <cmd> && echo '[custom:<name>:pass]'
                               ↓
                    tool_result content in JSONL log
                               ↓
         QualityGate.parse_validation_evidence_with_spec() → extract_tool_result_content()
                               ↓
                        ValidationEvidence.custom_commands_ran/failed
                               ↓
                 QualityGate check (allow_fail lookup from spec)
```

**Key design decisions:**
1. Custom commands get `CommandKind.CUSTOM` but are tracked by name in evidence (unlike built-in commands tracked only by `CommandKind`)
2. Structured markers emitted **by the agent** via echo in Bash tool calls (appears in tool_result content in JSONL logs)
3. `custom_commands` is a **top-level field** on `ValidationConfig`, not nested under `commands`
4. `_apply_command_overrides` extended to handle `custom_commands` with full-replace semantics (not merge)
5. Pipeline order: setup → format → lint → typecheck → **custom_a → custom_b** → test → e2e

### Data Model

**New dataclass in `config.py`:**
```python
@dataclass(frozen=True)
class CustomCommandConfig:
    """Configuration for a custom validation command."""
    command: str
    timeout: int = 120  # Default per spec R1
    allow_fail: bool = False

    @classmethod
    def from_value(cls, name: str, value: str | dict[str, object]) -> CustomCommandConfig:
        """Create from YAML value (string or dict).

        Validates:
        - Command name regex: ^[A-Za-z_][A-Za-z0-9_]*$
        - Non-empty command string
        - Known keys in object form
        """
        ...
```

**Extended dataclasses:**

`ValidationConfig` (in `config.py`) — **top-level field, not in CommandsConfig**:
```python
custom_commands: dict[str, CustomCommandConfig] = field(default_factory=dict)
_fields_set: frozenset[str]  # existing, now also tracks "custom_commands"
```

`ValidationEvidence` (in `quality_gate.py`):
```python
# New fields for custom command tracking (outcomes only, not config)
custom_commands_ran: dict[str, bool] = field(default_factory=dict)  # name → ran
custom_commands_failed: dict[str, bool] = field(default_factory=dict)  # name → failed
# NOTE: Do NOT store allow_fail here — gate looks up allow_fail from ValidationSpec
```

`PromptValidationCommands` (in `config.py`, line 486):
```python
# New field (includes timeout for shell wrapper)
custom_commands: list[tuple[str, str, int, bool]]  # (name, command, timeout, allow_fail)
```

**Existing dataclass (no changes needed):**

`ValidationCommand` (in `spec.py`) already has:
- `name: str` — used for custom command name
- `allow_fail: bool = False` — already exists, no schema change needed

**Enum change in `spec.py`:**
```python
class CommandKind(Enum):
    # ... existing ...
    CUSTOM = "custom"
```

### API/Interface Design

**Structured markers (R5):**
- Start: `[custom:<name>:start]`
- Success: `[custom:<name>:pass]`
- Failure: `[custom:<name>:fail exit=<code>]`
- Timeout: `[custom:<name>:timeout]`

**Marker emission mechanism:**
- Markers are emitted **by the agent** when executing custom commands via the Bash tool
- The `implementer_prompt.md` template provides command wrapper that preserves exit codes:
  ```bash
  # For allow_fail=false (strict): exit with command's status
  echo '[custom:import_lint:start]'; __status=0; timeout 120 uvx import-linter --config pyproject.toml || __status=$?; if [ $__status -eq 0 ]; then echo '[custom:import_lint:pass]'; elif [ $__status -eq 124 ]; then echo '[custom:import_lint:timeout]'; else echo "[custom:import_lint:fail exit=$__status]"; fi; exit $__status

  # For allow_fail=true (advisory): always exit 0
  echo '[custom:import_lint:start]'; __status=0; timeout 120 uvx import-linter --config pyproject.toml || __status=$?; if [ $__status -eq 0 ]; then echo '[custom:import_lint:pass]'; elif [ $__status -eq 124 ]; then echo '[custom:import_lint:timeout]'; else echo "[custom:import_lint:fail exit=$__status]"; fi; exit 0
  ```
- **Exit code semantics (P1 fix):** Strict commands (`allow_fail=false`) exit with original status so tool_result shows error; advisory commands (`allow_fail=true`) always exit 0
- **Timeout handling (P2 fix):** Use shell `timeout` command; exit code 124 = timeout marker emitted (both strict and advisory wrappers)
- **Portability note:** `timeout` is GNU coreutils; exit code 124 is timeout-specific. Document in release notes that wrapped commands returning 124 will be incorrectly treated as timeouts (rare edge case)
- Markers appear in Bash tool_result content in the Claude JSONL session log
- Evidence parser extracts tool_result content and scans for marker patterns

**Config schema (R1):**
```yaml
custom_commands:
  # String shorthand (defaults: timeout=120, allow_fail=false)
  import_lint: "uvx import-linter --config pyproject.toml"

  # Object form
  arch_check:
    command: "uvx grimp check src/"
    timeout: 120
    allow_fail: true

run_level_commands:
  custom_commands:  # Full replace, not merge
    import_lint: "uvx import-linter --config pyproject.toml --verbose"
```

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/domain/validation/config.py` | Exists | Add `CustomCommandConfig` dataclass, extend `ValidationConfig` (top-level) with `custom_commands` field, extend `PromptValidationCommands` |
| `src/domain/validation/spec.py` | Exists | Add `CommandKind.CUSTOM`, extend `build_validation_spec()` to build custom commands in correct position, extend `_apply_command_overrides()` for full-replace semantics with null-vs-{} handling |
| `src/domain/validation/config_loader.py` | Exists | Add `"custom_commands"` to `_ALLOWED_TOP_LEVEL_FIELDS`, parse top-level `custom_commands` dict, validate command names (regex), validate object form keys |
| `src/domain/validation/config_merger.py` | Exists | Ensure `merge_configs()` preserves user `custom_commands` when merging with preset |
| `src/domain/validation/preset_registry.py` | Exists | Add check in `get()`: if preset YAML contains `custom_commands`, raise `ConfigError` |
| `src/domain/quality_gate.py` | Exists | Extend `ValidationEvidence` for by-name tracking (ran, failed — not allow_fail), update evidence parsing to detect custom markers from tool_result content, update gate logic to look up `allow_fail` from spec |
| `src/infra/io/session_log_parser.py` | Exists | Add `extract_tool_result_content()` method to extract tool_result content for marker parsing |
| `src/core/protocols.py` | Exists | Extend `LogProvider` protocol with `extract_tool_result_content()`, extend `ValidationEvidenceProtocol` with `custom_commands_ran`, `custom_commands_failed` |
| `src/prompts/implementer_prompt.md` | Exists | Add custom commands section with marker-wrapped command patterns and `allow_fail` indication |
| `tests/unit/domain/test_validation_config.py` | Exists | Add tests for `CustomCommandConfig`, string shorthand, object form, validation errors |
| `tests/unit/domain/test_spec.py` | Exists | Add tests for custom command building, pipeline position, run-level override semantics, null-vs-{} handling |
| `tests/unit/domain/test_quality_gate.py` | Exists | Add tests for custom evidence parsing, by-name tracking, `allow_fail` lookup from spec |
| `tests/integration/domain/test_validation_config.py` | Exists | Add end-to-end tests for config loading → execution → evidence flow |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes

From spec:
- **Command name validation**: Names must match `^[A-Za-z_][A-Za-z0-9_]*$` (Python identifiers) — raise `ConfigError` with specific message
- **Null value in dict**: `custom_commands: {cmd: null}` — raise `ConfigError` ("use run-level override to disable commands")
- **Empty command string**: `custom_commands: {cmd: ""}` — raise `ConfigError` with specific message
- **Unknown keys in object form**: `{command: "...", typo: true}` — raise `ConfigError` with unknown key name
- **Preset with custom_commands**: Raise `ConfigError` during preset loading ("presets cannot define custom_commands")
- **Missing end marker**: If `[custom:<name>:start]` exists without corresponding end marker, gate treats as failure (subject to `allow_fail`)
- **Timeout behavior**: Emit `[custom:<name>:timeout]` marker, terminate process group, treat as failure (subject to `allow_fail`)

Implementation-specific:
- **Empty `custom_commands: {}`**: Valid, results in no custom commands running
- **Run-level `custom_commands: null`**: Same as omitted (repo-level applies)
- **Run-level `custom_commands: {}`**: Disables all custom commands for that run
- **Naming collisions**: User names a custom command `lint` or `test` — allowed but discouraged (distinct by `kind` internally)
- **Marker spoofing**: Custom command outputs text that looks like a marker — regex anchors minimize false positives; user responsibility

### Breaking Changes & Compatibility
- **Potential Breaking Changes**: None identified — repos without `custom_commands` behave identically
- **Mitigations**:
  - Default to empty dict when `custom_commands` not specified
  - All validation only runs when `custom_commands` is present
  - Existing evidence detection code paths unaffected (custom commands use separate tracking)

## Testing & Validation Strategy

### Unit Tests

**`test_validation_config.py`:**
- `CustomCommandConfig.from_value()` with string shorthand → defaults applied (timeout=120, allow_fail=false)
- `CustomCommandConfig.from_value()` with object form → all fields parsed
- Invalid command name (numeric prefix, special chars) → `ConfigError`
- Null value in dict → `ConfigError` with guidance
- Empty command string → `ConfigError`
- Unknown keys in object form → `ConfigError`
- `ValidationConfig.from_dict()` with top-level `custom_commands` → dict parsed correctly
- `_fields_set` tracking for `custom_commands` presence
- `merge_configs()` preserves user `custom_commands` when merging with preset
- YAML order preservation: `{a: ..., b: ..., c: ...}` results in spec order `[a, b, c]`

**`test_spec.py`:**
- `build_validation_spec()` with `custom_commands` → commands in correct pipeline position
- Pipeline order verification: setup → format → lint → typecheck → custom_a → custom_b → test → e2e
- `_apply_command_overrides()` full-replace semantics for `custom_commands`
- Run-level `custom_commands: {}` disables all custom commands
- Run-level `custom_commands: null` uses repo-level (treated as omitted)
- `_fields_set` tracking: null not in set, {} is in set

**`test_quality_gate.py`:**
- `extract_tool_result_content()` extracts content from tool_result entries
- Content extraction handles non-string types (list, other → str fallback)
- Parse `[custom:cmd:start]` from tool_result content → `custom_commands_ran[cmd] = True`
- Parse `[custom:cmd:pass]` → `custom_commands_failed[cmd] = False`
- Parse `[custom:cmd:fail exit=1]` → `custom_commands_failed[cmd] = True`
- Parse `[custom:cmd:timeout]` → `custom_commands_failed[cmd] = True`
- Missing end marker (start only) → treated as failure
- No markers at all for spec'd command → treated as "not run"
- Multiple markers (retry): latest terminal marker wins
- Terminal marker without start → still counts as "ran"
- Gate logic: looks up `allow_fail` from `ValidationSpec`, not from evidence
- `allow_fail=True` in spec + failure in evidence → gate passes (advisory)
- `allow_fail=False` in spec + failure in evidence → gate fails

**`test_preset_registry.py`:**
- Preset YAML with `custom_commands` → raises `ConfigError("presets cannot define custom_commands")`

### Integration Tests

**`test_validation_config.py` (integration):**
- Full config load → merge → spec build → executor run → evidence parse → gate check
- `allow_fail: true` command fails → validation continues to test
- `allow_fail: false` command fails → validation aborts
- Run-level override replaces (not merges) repo-level custom_commands
- Execution order verification with lint, custom_commands, test

### Manual Verification

- Configure `custom_commands` with both shorthand and object entries
- Verify defaults (timeout=120, allow_fail=false) apply to shorthand
- Run validation and verify markers appear in session log
- Test advisory failure: `allow_fail: true` command exits non-zero, validation proceeds
- Test strict failure: `allow_fail: false` command exits non-zero, validation aborts

### Monitoring / Observability

- Structured markers in session log enable debugging
- Event sink notifications for custom command start/pass/fail/skip (if wired up)
- Validation manifest includes custom commands in `expected_commands`

### Acceptance Criteria Coverage

| Spec AC | Covered By |
|---------|------------|
| R1 — Config Schema | `CustomCommandConfig.from_value()` unit tests, validation error tests, top-level field in `ValidationConfig` |
| R2 — Execution Order | `build_validation_spec()` unit tests, integration test for pipeline order |
| R3 — Advisory Failures | Quality gate `allow_fail` lookup from spec, integration test for advisory behavior |
| R4 — Run-Level Override | `_apply_command_overrides()` unit tests with null-vs-{} semantics, integration test for full-replace |
| R5 — Evidence Detection | `extract_tool_result_content()` for marker extraction, marker parsing tests, missing end marker handling |
| R6 — Agent Awareness | `PromptValidationCommands` with timeout in tuple, `implementer_prompt.md` marker-wrapped command patterns with exit code preservation |

## Open Questions

- None — spec and user decisions are complete.

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking
