# Separate Evidence Check Configuration

**Tier:** M
**Owner:** [TBD]
**Target ship:** [TBD]
**Links:** N/A

## 1. Outcome & Scope

**Problem / context**
The `commands:` section in `mala.yaml` is currently overloaded: it simultaneously defines which commands triggers can execute and which commands the evidence quality gate requires proof of in logs. This coupling creates friction because:
- You cannot require evidence of a command without making it available as a trigger
- You cannot exclude a command from evidence requirements while keeping it in the trigger pool
- Evidence requirements are implicitly derived from what is in the command pool, offering no explicit control

**Goal**
Enable users to configure evidence check requirements separately from run commands, so that evidence gate expectations can be defined independently of trigger execution configuration.

**Success criteria**
- Users can declare evidence requirements independently of trigger execution configuration via an `evidence_check:` section in `mala.yaml`
- When `evidence_check:` is present, the evidence gate evaluates only the explicitly listed commands
- When `evidence_check:` is absent (or has an empty `required` list), the evidence gate requires no command evidence

**Non-goals**
- Changing how triggers execute commands or how the run command pool is built
- Changing the evidence detection mechanism itself (patterns, markers, etc.)
- Maintaining current "infer from commands:" fallback behavior (explicitly a breaking change)

## 2. User Experience & Flows

**Config schema**
```yaml
evidence_check:           # Optional object; if absent, no evidence required
  required: [test, lint]  # Optional list of strings; command keys referencing resolved command map
```
- `evidence_check` MUST be an object if present; other types (boolean, string, list) are config errors
- If `required` is omitted or null, treat it as an empty list (no evidence required)
- If `required` is non-null, it MUST be a list of strings; non-string entries (e.g., `[test, 1]`) or other types are config errors
- Invalid YAML type produces a config error with the expected type

**Resolved command map and precedence**
The "resolved command map" is built by merging preset commands with project `commands:`, with project definitions taking precedence on key collisions. For example:
- Preset defines `test: "pytest"`, project defines `test: "uv run pytest"` → resolved `test` uses `"uv run pytest"`
- Preset defines `lint: "ruff check"`, project has no `lint` → resolved `lint` uses `"ruff check"`
- Project defines `import-linter: "..."`, preset has none → resolved `import-linter` uses project definition

**Primary flow**
1. User adds an `evidence_check:` section to `mala.yaml`
2. User lists required commands by reference (e.g., `required: [test, lint]`) matching keys in resolved command map
3. Evidence gate checks only for evidence of the listed commands (filtered by scope, see R7)
4. If no `evidence_check:` section exists, the gate passes without requiring command evidence

**Key states**
- **No `evidence_check:` section:** No evidence required (gate passes without command evidence)
- **`evidence_check:` present but `required` key missing or null:** Treated as `required: []` (no evidence required)
- **Empty `required: []`:** No evidence required
- **Commands listed:** Gate requires evidence of each listed command running; duplicates are ignored (treated as a set)
- **Success state:** All required command evidence is found; evidence gate passes
- **Partial evidence:** Gate fails; error message lists missing command keys and suggests running them or updating `evidence_check.required`
- **Invalid reference:** Config validation fails with error listing each invalid key and the available command keys from the resolved map

## 3. Requirements + Verification

**R1 — Separate evidence config section**
- **Requirement:** The system MUST support an `evidence_check:` config section with a `required:` list of command references that determines which command evidence is required
- **Verification:**
  - Given a mala.yaml with:
    ```yaml
    commands:
      test: "uv run pytest"
      lint: "uvx ruff check ."
      format: "uvx ruff format ."
    evidence_check:
      required: [test, lint]
    ```
  - When the evidence gate runs
  - Then it checks only for test and lint evidence, NOT format

**R2 — No evidence when section absent**
- **Requirement:** When no `evidence_check:` section exists, the gate MUST NOT require any command evidence
- **Verification:**
  - Given a mala.yaml with `commands:` but no `evidence_check:` section
  - When the evidence gate runs
  - Then the gate passes without checking for command evidence
- **Breaking change:** This changes current behavior where evidence is inferred from commands

**R3 — Validate command references against resolved command map**
- **Requirement:** The system MUST fail config load if `evidence_check.required` references a command not present in the resolved command map (which includes both preset-provided built-in commands and project-defined custom commands from `commands:`)
- **Verification:**
  - Given a mala.yaml with `evidence_check.required: [nonexistent]` where "nonexistent" is neither a preset command nor defined in `commands:`
  - When config is loaded
  - Then config validation fails with an error message identifying the invalid reference and listing available command keys
- **Note:** Built-in commands (test, lint, etc.) are available from presets even without explicit `commands:` entries; custom commands require explicit definition
- **Implementation note:** Validation requires loading the preset to resolve the command map; this may occur during config post-validation rather than initial YAML parsing

**Examples:**
- `commands:` omitted, `evidence_check.required: [test]` → valid if preset provides `test`
- `commands:` omitted, `evidence_check.required: [custom]` → config error because `custom` not in resolved map

**R4 — Detection pattern from resolved command string**
- **Requirement:** Evidence detection MUST use the pattern derived from the resolved command string for the referenced key (after preset+project merge); project overrides of a key override the pattern used for that key
- **Verification:**
  - Given preset defines `test: "pytest"` but project defines `test: "uv run pytest"`
  - And `evidence_check.required: [test]`
  - When the evidence gate parses logs
  - Then it uses the pattern derived from `"uv run pytest"` (the resolved value), not `"pytest"`
- **Note:** Detection uses the existing regex pattern-matching logic in `EvidenceCheck.parse_validation_evidence_with_spec()`; this spec does not change how patterns are derived or matched

**R5 — Built-in and custom command support**
- **Requirement:** Both preset-provided built-in commands (test, lint, format, typecheck, e2e, build, setup) and project-defined custom commands MUST be referenceable in `evidence_check.required`, provided they exist in the resolved command map
- **Verification:**
  - Given a mala.yaml with:
    ```yaml
    commands:
      import-linter: "uvx --from import-linter lint-imports"
    evidence_check:
      required: [test, import-linter]
    ```
  - When the evidence gate runs
  - Then it checks for both test (from preset) and import-linter (from `commands:`) evidence

**R6 — Duplicate and ordering semantics**
- **Requirement:** The `required` list MUST be treated as a set; duplicate entries are ignored and ordering does not affect gate behavior
- **Verification:**
  - Given `evidence_check.required: [test, lint, test]`
  - When the evidence gate runs
  - Then it checks for test and lint evidence exactly once each

**R7 — Scope filtering applies at gate evaluation time**
- **Requirement:** Each gate MUST compute `effective_required = scope_filter(evidence_check.required)` at evaluation time; commands filtered out by scope are treated as not required and are not listed as missing in error messages
- **Verification:**
  - Given `evidence_check.required: [test, e2e]` where E2E has global scope
  - When a per-session evidence gate runs
  - Then it checks only for test evidence (E2E filtered out by scope, not reported as missing)
- **Note:** R3 validation (invalid reference check) happens pre-scope against the full resolved command map; a global-only command is a valid reference even if the current gate won't require it
- **Note:** This preserves existing scope semantics from `check_evidence_against_spec()` and `get_required_evidence_kinds()`

**R8 — allow_fail does not exempt from evidence requirement**
- **Requirement:** Commands listed in `evidence_check.required` MUST have evidence of running regardless of `allow_fail` setting; `allow_fail` only affects whether failures block the gate, not whether evidence is required
- **Verification:**
  - Given a command `lint` with `allow_fail: true` and `evidence_check.required: [lint]`
  - When lint was never run (no evidence in logs)
  - Then the gate fails for missing evidence
  - But when lint ran and failed, the gate passes (failure is advisory due to `allow_fail`)
- **Messaging:** Each required command is in exactly one of three mutually exclusive states:
  1. **Missing evidence** (no evidence found) → listed in "missing" section, causes gate failure
  2. **Evidence found, success** → not listed in errors
  3. **Evidence found, failure** → if `allow_fail: false`, causes gate failure; if `allow_fail: true`, listed separately as "advisory failure" (does not block gate)
- A command cannot appear in both "missing" and "advisory failure" categories

**R9 — Evidence attribution is per-key**
- **Requirement:** Evidence MUST be attributed to each required command key independently based on pattern matching; a single log event MAY satisfy multiple required keys if it matches their respective patterns
- **Verification:**
  - Given `commands.test: "pytest"` and `commands.unit: "pytest -m unit"`
  - And `evidence_check.required: [test, unit]`
  - When the log contains only `pytest -m unit`
  - Then both `test` and `unit` are satisfied (the log line matches both patterns)
- **Note:** This follows existing detection behavior where patterns are derived from command strings and matched independently

## 4. Instrumentation & Release Checks

**Instrumentation**
- None required (internal config change with no user-facing metrics needed)

**Migration / Breaking Change Guidance**
- **Breaking change:** Projects relying on implicit evidence requirements (inferred from `commands:`) will see the evidence gate pass without checking after this change
- **User action required:** Add `evidence_check.required: [test, lint, ...]` to mala.yaml if evidence checking is desired
- **Discovery aid:** Users can inspect their current resolved command pool via `mala config show` (or similar) to see which commands were previously implicitly required before migrating
- **Release notes must include:** Clear statement that evidence is no longer inferred from `commands:`, with migration example showing how to restore previous behavior
- **No deprecation warning phase:** This is a clean break; the old behavior is removed immediately

**Testing requirements**
The following test cases MUST be automated to prevent regressions:
1. Absent `evidence_check:` → no evidence required (gate passes without command evidence)
2. `evidence_check:` present with missing/null/empty `required` → no evidence required
3. `required` with valid keys → checks only those commands
4. `required` with invalid key → config load error listing invalid key and available keys
5. Duplicates in `required` → ignored (treated as set)
6. Preset-only key in `required` (e.g., `[test]` with no project `commands.test`) → valid
7. Project custom key in `required` → valid if defined in `commands:`
8. Project override of preset key → uses project command string for pattern detection
9. Scope filtering → per-session gate excludes global-only commands, not listed as missing
10. `allow_fail` command not run → gate fails for missing evidence
11. `allow_fail` command run and failed → gate passes, listed as advisory failure
12. Overlapping patterns → single log event can satisfy multiple required keys
13. Global-only command in `required` validates successfully (R3 is pre-scope)

**Decisions made**
- Config style: Reference-based (`required: [test, lint]`) rather than independent definitions
- Fallback behavior: No evidence required when `evidence_check:` absent (breaking change accepted)
- Invalid references: Error at config load against resolved command map (fail fast)
- Detection patterns: Inherited from commands, no inline override support
- Custom commands: Both preset built-ins and project-defined custom commands are referenceable
- Schema: `evidence_check.required` missing/null treated as empty list; duplicates ignored

**Open questions**
1. Should presets be able to define `evidence_check:` that projects can override? (Deferred to future iteration)

**Implementation notes** (from review)
- `evidence_check:` is a top-level config key alongside `commands:`; preset override semantics deferred to future iteration
- Command key matching is exact/case-sensitive (consistent with existing `commands:` behavior)
- Error message ordering: preserve user order for missing keys; sort available keys alphabetically
- `evidence_check: null` is a config error (not an object); `evidence_check: {}` means no evidence required
- Consider enhancing `mala config show` to display resolved evidence requirements post-implementation
