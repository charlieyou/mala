# Evidence Check Redesign: MALA_EVIDENCE Protocol

**Date:** 2026-05-09
**Epic:** mala-3gbpn
**Status:** Implemented

## Outcome

Replace marker-based custom-command evidence with one canonical evidence protocol for every validation command. Built-in and custom commands are treated the same: each successful wrapper invocation emits exactly one summary line that the evidence gate can attribute to a command key.

```text
MALA_EVIDENCE name=<name> exit=<code> log=<path>
```

The evidence model is consolidated around a single command map:

```python
commands: dict[str, CommandEvidence]
```

Legacy split fields are removed:

- `commands_ran`
- `failed_commands`
- `custom_commands_ran`
- `custom_commands_failed`

## Scope

In scope:

- Add a reusable helper for canonical validation wrappers.
- Parse and recognize `MALA_EVIDENCE` summary lines.
- Attribute evidence by configured command key.
- Generate canonical wrappers for built-in, custom, and gate-followup commands.
- Reject duplicate configured evidence keys where duplicate names would make attribution ambiguous.
- Remove marker-era custom-command parsing from source code.
- Update documentation, prompts, and tests.

Out of scope:

- Changing the `mala.yaml` schema.
- Adding fallback support for marker-era evidence.
- Keeping compatibility properties on `ValidationEvidence`.
- Supporting hand-written `MALA_EVIDENCE` lines as authoritative evidence when the canonical wrapper was not used.

## Evidence Line Format

The only authoritative evidence line is:

```text
MALA_EVIDENCE name=<name> exit=<code> log=<path>
```

Rules:

- `name` is the validation evidence key, such as `format`, `lint`, `typecheck`, `test`, or the exact custom command name from `mala.yaml`.
- `exit` is the command exit code captured by the wrapper.
- `log` is the validation log path written by the wrapper.
- The recognizer accepts valid configured command names only.
- Exactly one valid evidence line is expected from a canonical wrapper result.
- Non-zero `exit` means the command ran and failed; `allow_fail` determines whether that failed evidence blocks the gate.

## Canonical Wrapper Requirements

All prompt-rendered validation commands must be emitted through the same wrapper builder.

The wrapper must:

1. Create the validation log directory once per wrapper invocation.
2. Run the configured command under the configured timeout.
3. Redirect command output to the command-specific log file.
4. Preserve the wrapped command exit code.
5. Print one `MALA_EVIDENCE name=<name> exit=<code> log=<path>` line.
6. Exit non-zero for strict failed commands.
7. Exit zero for advisory failed commands while still reporting the non-zero command exit in the evidence line.

Prompt instructions must tell agents to run the provided wrapper exactly as shown and not to manually echo `MALA_EVIDENCE`.

## ValidationEvidence Shape

`ValidationEvidence` stores command evidence in one map:

```python
@dataclass
class CommandEvidence:
    passed: bool
    exit_code: int | None = None
    log_path: str | None = None


@dataclass
class ValidationEvidence:
    commands: dict[str, CommandEvidence]
```

Consumers must read command evidence from `evidence.commands`. No compatibility properties or re-export shims should be added for the removed split fields.

## Implementation Tasks

### T001 - Add `validation_wrapper.py`

Add `src/domain/validation_wrapper.py` with helper functions that build canonical Bash wrappers for configured validation commands. The helper owns quoting, timeout handling, log redirection, advisory failure behavior, and the summary line format.

Verification:

- Unit tests cover strict success, strict failure, advisory failure, timeout propagation, shell quoting, and log-path rendering.

### T002 - Replace `ValidationEvidence` Shape

Replace split evidence fields with the unified `commands` map and update all callers.

Verification:

- Existing gate and orchestration tests construct and inspect `ValidationEvidence(commands=...)`.
- No source code depends on the removed split fields.

### T003 - Add Parser, Recognizer, and Duplicate-Name Validation

Add parser support for `MALA_EVIDENCE` summary lines and make evidence attribution depend on configured command keys.

Verification:

- Valid summary lines produce `CommandEvidence`.
- Invalid names, missing fields, malformed exit codes, and duplicate configured names are rejected.
- Duplicate command names cannot make one line satisfy an ambiguous command identity.

### T004 - Generate Canonical Wrappers in Prompts

Render canonical wrappers in implementer and gate-followup prompts for all required built-in and custom commands.

Verification:

- Prompt tests assert wrapper snippets contain `MALA_EVIDENCE name=%s exit=%s log=%s`.
- Built-in and custom commands use the same rendering path.
- Gate followup uses the same snippets for missing or invalid command evidence.

### T005 - Delete Marker-Era Code

Remove custom marker parsing and marker-era prompt instructions.

Verification:

- `src/` contains no references to `[custom:`.
- Tests cover the new protocol instead of marker fallback.

### T006 - Documentation and Changelog

Document the new protocol, migration impact, and command evidence shape.

Verification:

- User-facing docs describe `MALA_EVIDENCE`.
- Changelog calls out the breaking evidence protocol and removed compatibility fields.

## Rollout

All work lands in one PR. There is no marker fallback and no compatibility surface for the removed evidence fields. The `mala.yaml` schema remains unchanged.

## Acceptance Checks

Before epic closure:

- `rg -n "\\[custom:" src` returns no results.
- `rg -n "MALA_EVIDENCE" src/domain/evidence_check.py src/domain/prompts.py src/prompts/implementer_prompt.md src/prompts/gate_followup.md tests` finds the new protocol in implementation, prompts, and tests.
- Full validation suite passes with coverage at or above the project threshold.

## Notes

This plan is restored after implementation so future verification can compare the completed epic against the referenced authoritative design. It intentionally records the final accepted design, not the removed marker-era behavior.
