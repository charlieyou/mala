# Implementation Plan: Inline Custom Commands (Plus-Prefix Additive Overrides)

## Context & Goals

**Problem:** Custom validation commands are currently defined in separate `custom_commands`
and `run_level_custom_commands` top-level sections, creating redundant structure when they
could live alongside built-in commands.

**Solution:** Replace the dedicated custom command groups with **inline custom commands**
inside `commands` and `run_level_commands`. Custom commands are identified as **unknown keys**
in those sections.

**Goals:**
- Simplify config by allowing custom commands inline with built-ins
- Support two run-level override modes for custom commands:
  - **Replace** (default): unprefixed custom keys fully replace repo-level customs
  - **Additive**: custom keys prefixed with `+` are merged with repo-level customs
- **Inherit repo-level customs when run-level has no custom keys** (key semantic)
- Provide clear migration path with explicit errors for old config format

## Scope & Non-Goals

**In Scope:**
- Inline custom command parsing in `commands` and `run_level_commands`
- Plus-prefix additive mode semantics
- Config validation and error messaging
- Backward-incompatible removal of `custom_commands` / `run_level_custom_commands`

**Non-Goals:**
- Changes to coverage config or preset behavior
- Changes to validation execution logic (quality gate, prompts)
- Typo detection for misspelled built-in commands (documented as known limitation)
- Changes to how custom commands are executed
- Selective disabling of individual custom commands in additive mode (use replace mode instead)
- Allowing presets to define custom commands (presets remain built-ins only)

## Assumptions & Constraints

**Assumptions:**
- Users will migrate existing configs when they encounter the explicit error
- The "inherit when no custom keys" behavior matches user expectations
- Plus-prefix syntax is intuitive for "additive merge" semantics

**Constraints:**
- No backward compatibility: old keys fail fast with migration hint
- Command name regex unchanged: `^[A-Za-z_][A-Za-z0-9_-]*$`
- Must not break existing built-in command override behavior
- Reserved key `_clear_customs` cannot be used as a custom command name
- `+`-prefixed YAML keys may require quoting in some YAML tooling (e.g., `"+import_lint":`)
- Future built-in command additions may conflict with existing custom command names (see Migration Notes)

## Prerequisites

- None (self-contained config parsing change)

## High-Level Approach

### 1) Custom Command Identification
- Any **unknown key** in `commands` or `run_level_commands` is a custom command
- Known keys (built-ins): `setup`, `format`, `lint`, `typecheck`, `test`, `e2e`
- Reserved keys: `_clear_customs` (not a custom command)
- **Known limitation:** Typos in built-in names become custom commands silently
  (e.g., `typechek:` becomes custom instead of erroring). This is documented but
  not auto-detected to keep parsing simple.

### 2) Run-Level Override Modes (Custom Commands Only)

**Critical semantic: When run-level has NO custom keys, repo-level customs are inherited.**

| Run-level custom keys | Mode | Effective customs |
|-----------------------|------|-------------------|
| None (only built-in overrides) | **Inherit** | Repo-level customs unchanged |
| `_clear_customs: true` only | **Clear** | No custom commands |
| All unprefixed | **Replace** | Only run-level customs |
| All `+`-prefixed | **Additive** | Repo + run-level (run wins conflicts) |
| Mixed prefixed/unprefixed | **Error** | Config validation fails |

This ensures that overriding a built-in (e.g., `lint: "uvx ruff check ."`) at run-level
does not silently drop repo-level custom commands like `arch-check`.

**Scope behavior:** Run-level custom command modes only apply to `ValidationScope.RUN_LEVEL`.
For `ValidationScope.PER_ISSUE`, repo-level customs are always used unchanged (existing behavior).

### 3) Explicit Clear Mechanism
To intentionally drop all repo-level custom commands without defining any run-level customs,
use the reserved key `_clear_customs: true`:
```yaml
run_level_commands:
  lint: "uvx ruff check ."
  _clear_customs: true  # Explicitly drop all repo-level custom commands
```
This is the only way to express "zero custom commands" since absence of custom keys means inherit.

**Validation rules for `_clear_customs`:**
- Only valid value is `true` (boolean). Any other value (false, null, string) → ConfigError
- Cannot be combined with custom command keys (prefixed or unprefixed) → ConfigError
- `+_clear_customs` is also invalid → ConfigError
- Not allowed in repo-level `commands` → ConfigError

### 4) Plus Prefix Validity
- `+` prefix is **only valid** for run-level custom commands
- `commands` section **must not** contain `+`-prefixed keys (error)

### 5) Null Handling
- `null` values for custom commands are **invalid** in all modes
- Validated in `CustomCommandConfig.from_dict` (existing validation reused)
- Consistent with existing "use run-level override to disable" guidance
- **Limitation:** Cannot selectively disable a single inherited custom in additive mode;
  use replace mode and re-list desired customs instead

### 6) Name Collisions with Built-ins
- Custom command names must **not** match built-in keys
- Applies even with `+` prefix (e.g., `+lint` is an error)

### 7) Preset Behavior (Unchanged)
- Presets **cannot** define custom commands (only built-ins)
- Preset validation is invoked in `src/domain/validation/preset_registry.py` during
  `PresetRegistry.load()` before the preset is registered
- If a preset contains unknown keys in `commands`, raise ConfigError during preset loading
- This prevents accidental introduction of customs via presets

### 8) Backward Compatibility (Not Supported)
- Top-level `custom_commands` and `run_level_custom_commands` are **invalid**
- Presence raises ConfigError with migration hint

### 9) Additive Merge Ordering
In additive mode, the merge uses `{**repo_customs, **run_customs}`:
- Repo-level custom commands retain their original insertion order
- Run-level customs that override repo-level commands update values in-place (position preserved)
- New run-level customs are appended at the end
- This matches Python dict merge semantics and preserves YAML key order

## Detailed Design

### Data Model Changes

**File:** `src/domain/validation/config.py`

Add `custom_commands` field and `custom_override_mode` to `CommandsConfig`. The existing
`CustomCommandConfig` type (lines 130-246 in config.py) is reused. `CustomCommandConfig.from_dict`
already validates that null values are rejected.

```python
from enum import Enum

class CustomOverrideMode(Enum):
    """Mode for run-level custom command overrides."""
    INHERIT = "inherit"    # No run-level custom keys → inherit repo-level
    CLEAR = "clear"        # _clear_customs: true → no customs
    REPLACE = "replace"    # Unprefixed custom keys → replace repo-level
    ADDITIVE = "additive"  # +prefixed custom keys → merge with repo-level

@dataclass
class CommandsConfig:
    setup: CommandConfig | None = None
    format: CommandConfig | None = None
    lint: CommandConfig | None = None
    typecheck: CommandConfig | None = None
    test: CommandConfig | None = None
    e2e: CommandConfig | None = None
    # NEW: custom commands parsed from unknown keys (stored with + prefix stripped)
    custom_commands: dict[str, CustomCommandConfig] = field(default_factory=dict)
    # NEW: mode computed during parsing (only meaningful for run-level)
    custom_override_mode: CustomOverrideMode = CustomOverrideMode.INHERIT
    # Existing: _fields_set tracks built-in fields only (for merger compatibility)
    _fields_set: frozenset[str] = field(default_factory=frozenset)
```

**Parsing logic in `CommandsConfig.from_dict(data, *, is_run_level: bool)`:**

The `is_run_level` parameter distinguishes repo vs run-level context for validation.

1. Extract known keys → built-in fields, track in `_fields_set`
2. Check for `_clear_customs` reserved key:
   - If value is not exactly `true` → ConfigError
   - If present with custom keys → ConfigError ("_clear_customs cannot be combined with custom commands")
   - If present and `is_run_level=True`: set mode to CLEAR, no custom commands
   - If present and `is_run_level=False`: ConfigError ("_clear_customs only valid at run-level")
3. Collect remaining unknown keys as raw custom keys
4. If `is_run_level=True`:
   - Count `+`-prefixed and unprefixed custom keys
   - If mixed: raise ConfigError
   - If all `+`-prefixed: set mode to ADDITIVE, strip `+` before storing
   - If all unprefixed (and at least one): set mode to REPLACE
   - If none: set mode to INHERIT
5. If `is_run_level=False`:
   - If any `+`-prefixed key: raise ConfigError (not allowed at repo-level)
   - Store custom commands directly
6. Validate command names (after stripping `+`) against regex and built-in collision

**Important:** `_fields_set` continues to track only built-in command fields (setup, test, lint,
format, typecheck, e2e). Custom commands do NOT affect `_fields_set`. The merger uses
`_fields_set` emptiness to detect "user explicitly set run_level_commands to empty/null".
Custom-only configs will still have empty `_fields_set` for built-ins, but the presence of
`custom_commands` or `custom_override_mode != INHERIT` indicates the section was not empty.

### Merger Changes

**File:** `src/domain/validation/config_merger.py`

The `_merge_commands` function uses `user._fields_set` emptiness combined with
`user_commands_explicitly_set` to detect "explicitly empty" and clear preset values.
This logic needs adjustment:

```python
def _merge_commands(
    preset: CommandsConfig,
    user: CommandsConfig,
    user_commands_explicitly_set: bool,
    clear_on_explicit_empty: bool = False,
) -> CommandsConfig:
    # Detect "explicitly empty" - user set commands to {} or null
    # NEW: only clear if no built-ins AND no customs AND not a clear directive
    is_explicitly_empty = (
        clear_on_explicit_empty
        and user_commands_explicitly_set
        and not user._fields_set  # No built-in fields set
        and not user.custom_commands  # No custom commands
        and user.custom_override_mode == CustomOverrideMode.INHERIT  # Not a clear directive
    )
    if is_explicitly_empty:
        return CommandsConfig()  # Return empty, clearing preset

    # Merge built-ins as before...
    # Custom commands: preserve user's custom_commands and custom_override_mode
    return CommandsConfig(
        setup=...,
        # ... other built-ins ...
        custom_commands=user.custom_commands,  # User's customs (may be empty)
        custom_override_mode=user.custom_override_mode,  # User's mode
        _fields_set=user._fields_set,
    )
```

### Merge Logic for Custom Commands

**File:** `src/domain/validation/spec.py`

Update `_apply_custom_commands_override()` to use the computed mode. **Critical:** This
function is only called for `ValidationScope.RUN_LEVEL`. For `PER_ISSUE` scope, the existing
code path returns repo-level customs directly (line 495-496). This behavior is preserved.

```python
def _apply_custom_commands_override(
    repo_customs: dict[str, CustomCommandConfig],
    run_level_commands: CommandsConfig,
    scope: ValidationScope,
) -> dict[str, CustomCommandConfig]:
    """Apply run-level custom command override based on computed mode.

    For PER_ISSUE scope, always returns repo_customs unchanged.
    For RUN_LEVEL scope, applies the override mode.
    """
    # PER_ISSUE always uses repo-level customs (existing behavior)
    if scope == ValidationScope.PER_ISSUE:
        return repo_customs

    mode = run_level_commands.custom_override_mode
    run_customs = run_level_commands.custom_commands

    if mode == CustomOverrideMode.INHERIT:
        return repo_customs
    elif mode == CustomOverrideMode.CLEAR:
        return {}
    elif mode == CustomOverrideMode.REPLACE:
        return run_customs
    elif mode == CustomOverrideMode.ADDITIVE:
        # Merge: repo order preserved, run-level updates in-place or appends
        return {**repo_customs, **run_customs}
    else:
        raise ValueError(f"Unknown custom override mode: {mode}")
```

### ValidationConfig Changes

**File:** `src/domain/validation/config.py`

Update `ValidationConfig.from_dict`:
- Remove `custom_commands` / `run_level_custom_commands` from allowed top-level keys
- Add explicit error if old keys present
- Pass `is_run_level=False` when parsing `commands`
- Pass `is_run_level=True` when parsing `run_level_commands`

### Preset Validation

**File:** `src/domain/validation/preset_registry.py`

Update `PresetRegistry.load()` to validate that preset `commands` contains no unknown keys.
Currently presets already block top-level `custom_commands`; this extends validation to
inline unknown keys:

```python
def _validate_preset_commands(commands_data: dict) -> None:
    """Ensure presets don't accidentally define custom commands."""
    known_keys = {"setup", "format", "lint", "typecheck", "test", "e2e"}
    unknown = set(commands_data.keys()) - known_keys
    if unknown:
        raise ConfigError(
            f"Preset commands contain unknown keys: {unknown}. "
            "Presets can only define built-in commands."
        )
```

### Prompt/Model Plumbing

**File:** `src/domain/validation/config.py` (where `PromptValidationCommands` is defined)

Update `PromptValidationCommands.from_validation_config` to source custom commands from
`commands.custom_commands` instead of the top-level `custom_commands` field. The interface
remains the same; only the source changes.

Also update `src/domain/prompts.py` if it directly references custom commands.

### File Impact Summary

| File | Changes |
|------|---------|
| `src/domain/validation/config.py` | Add `CustomOverrideMode` enum, add `custom_commands` and `custom_override_mode` fields to `CommandsConfig`, update `from_dict` with `is_run_level` param, remove top-level `custom_commands`/`run_level_custom_commands`, add validation errors, update `PromptValidationCommands.from_validation_config` |
| `src/domain/validation/config_merger.py` | Update `_merge_commands` to check `custom_commands` and `custom_override_mode` when detecting "explicitly empty", preserve user's custom fields in merge |
| `src/domain/validation/spec.py` | Update `_apply_custom_commands_override()` to use mode enum with scope parameter, update `build_validation_spec` to source customs from `commands.custom_commands` |
| `src/domain/validation/preset_registry.py` | Add `_validate_preset_commands` to reject unknown keys in preset `commands` |
| `src/domain/prompts.py` | Update any direct references to custom commands to use new source |
| `tests/unit/domain/test_validation_config.py` | Add unit tests for all modes and error cases |
| `tests/unit/domain/test_config_merger.py` | Add tests for custom-only run_level_commands not triggering "explicitly empty" |
| `tests/unit/domain/test_validation_spec.py` | Add tests for PER_ISSUE scope ignoring run-level custom override modes |
| `plans/2026-01-06-custom-validation-commands-spec.md` | Update schema documentation |

## Desired DX (Examples)

### Repo-level (commands)
```yaml
commands:
  lint: "uvx ruff check ."
  format: "uvx ruff format --check ."
  typecheck: "uvx ty check"
  test: "uv run pytest"
  # Custom commands (unknown keys)
  import_lint: "uvx import-linter --config pyproject.toml"
  arch-check:
    command: "uvx grimp check src/"
    timeout: 120
    allow_fail: true
```

### Run-level: Inherit Mode (no custom keys)
```yaml
run_level_commands:
  lint: "uvx ruff check . --select=E"
  test: "uv run pytest -q"
```
**Result:** Built-ins `lint` and `test` are overridden. Custom commands `import_lint`
and `arch-check` are **inherited** unchanged from repo-level.

### Run-level: Clear Mode (explicitly drop all customs)
```yaml
run_level_commands:
  lint: "uvx ruff check ."
  _clear_customs: true
```
**Result:** Built-in `lint` is overridden. **No custom commands** are active
(repo-level `import_lint` and `arch-check` are dropped).

### Run-level: Replace Mode (unprefixed custom keys)
```yaml
run_level_commands:
  lint: "uvx ruff check ."
  test: "uv run pytest -q"
  import_lint: "uvx import-linter --config pyproject.toml --verbose"
```
**Result:** Built-ins `lint` and `test` are overridden. Custom commands are **replaced**:
only `import_lint` is active (repo-level `arch-check` is ignored).

### Run-level: Additive Mode (all keys prefixed with `+`)
```yaml
run_level_commands:
  lint: "uvx ruff check ."
  test: "uv run pytest -q"
  +import_lint: "uvx import-linter --config pyproject.toml --verbose"
  +arch-check:
    command: "uvx grimp check src/"
    allow_fail: true
```
**Result:** Built-ins `lint` and `test` are overridden. Custom commands are **merged**:
`import_lint` and `arch-check` override repo-level versions; any other repo-level
customs would be preserved.

**Note:** Some YAML tools may require quoting `+`-prefixed keys: `"+import_lint":`.

## Risks & Edge Cases

| Risk | Mitigation |
|------|------------|
| Typos in built-in names become custom commands | Document as known limitation; users can notice via unexpected validation output |
| Users confused by replace vs inherit distinction | Clear examples in docs; inherit is the "safe default" when no custom keys |
| Mixed `+`/non-`+` keys | Fail fast with clear error message |
| Migration friction from breaking change | Explicit error with migration hint pointing to new syntax |
| Cannot selectively disable one custom in additive mode | Document as limitation; use replace mode and re-list desired customs |
| Custom-only run_level triggers "explicitly empty" | Fixed: merger checks `custom_commands` and mode before clearing |
| Presets accidentally define customs via unknown keys | Validate presets reject unknown keys in `commands` |
| `_clear_customs` naming collision | Reserved key documented; cannot be used as custom command name |
| `+` prefix YAML quoting | Document that some tooling may require quoting |
| Run-level modes applied to PER_ISSUE scope | Fixed: `_apply_custom_commands_override` checks scope, PER_ISSUE uses repo customs |
| Future built-in name collides with existing custom | Document in migration notes; users must rename customs |

## Testing & Validation

### Unit Tests (`tests/unit/domain/test_validation_config.py`)

**Parsing:**
- Unknown keys in `commands` → `custom_commands` dict, mode=INHERIT (ignored for repo)
- Unknown keys in `run_level_commands` → run-level customs with correct mode
- `+` prefix in `commands` → error
- Mixed `+` and non-`+` keys in `run_level_commands` → error
- Custom name collision with built-in → error
- Old top-level `custom_commands` key → error with migration hint
- Old top-level `run_level_custom_commands` key → error with migration hint
- `_clear_customs: true` in `commands` → error (run-level only)
- `_clear_customs: true` in `run_level_commands` → mode=CLEAR
- `_clear_customs: false` or `_clear_customs: "yes"` → error
- `_clear_customs` combined with custom keys → error
- `+_clear_customs` → error
- Preset with unknown keys in `commands` → error
- Null value for custom command → error (via `CustomCommandConfig.from_dict`)

**Override modes (in spec.py tests):**
- Inherit: run-level with only built-in overrides → repo customs inherited
- Clear: run-level with `_clear_customs: true` → no customs
- Replace: run-level with unprefixed custom keys → repo customs ignored
- Additive: run-level with `+` customs → merge with repo customs
- Additive ordering: repo order preserved, new keys appended
- Name validation with hyphens works in all modes
- **PER_ISSUE scope ignores run-level custom override modes** (regression test)

### Merger Tests (`tests/unit/domain/test_config_merger.py`)

- Run-level with only custom keys does NOT trigger "explicitly empty" clear
- Run-level with only `_clear_customs: true` does NOT trigger "explicitly empty" clear
- Run-level with `{}` (truly empty) still clears preset run-level overrides

### Integration Tests
- YAML → ValidationSpec with inline customs in `commands`
- YAML → ValidationSpec with additive `+` customs in `run_level_commands`
- YAML → ValidationSpec with built-in-only overrides (inherit mode)
- YAML → ValidationSpec with `_clear_customs: true` (clear mode)

## Migration Notes (Breaking Change)

Users must:
1. Move `custom_commands` entries into `commands` section
2. Move `run_level_custom_commands` entries into `run_level_commands` section
3. To merge run-level customs with repo-level, prefix custom keys with `+`

Any config using old keys will fail with:
```
ConfigError: 'custom_commands' is no longer supported. Move custom commands
into the 'commands' section. See plans/2026-01-06-custom-validation-commands-spec.md
for the updated schema.
```

For `run_level_custom_commands`:
```
ConfigError: 'run_level_custom_commands' is no longer supported. Move custom commands
into the 'run_level_commands' section with '+' prefix for additive merge.
See plans/2026-01-06-custom-validation-commands-spec.md for the updated schema.
```

**Forward compatibility note:** If a new built-in command is added in the future with the
same name as an existing custom command, users will receive a collision error and must
rename their custom command. Similarly, if you have a custom command named `_clear_customs`,
you must rename it before upgrading.

## Open Questions

None.
