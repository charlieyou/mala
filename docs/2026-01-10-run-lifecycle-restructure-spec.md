# Run Lifecycle Restructure

**Tier:** M
**Owner:** Core Engineering / Mala Orchestrator
**Target ship:** 2026-01-20
**Links:** [plans/run-lifecycle-spec.md](../plans/run-lifecycle-spec.md)

## 1. Outcome & Scope

**Problem / context**
The current run lifecycle has two structural issues:
1. The `session_end` trigger fires once at the very end of a run (after all issues complete), but conceptually "session end" should mean "after each issue session ends" — operators expect per-issue validation hooks.
2. Global validation runs as a separate stage after the `run_end` trigger, creating confusing dual-stage run-level validation and duplicated concepts. These should be merged into a single unified run-level stage.

Current flow (simplified):
```
for each issue:
    agent work → gate → review → finalize
end for
session_end trigger (once, all issues done)
run_end trigger
global validation ← separate stage
finalize run
```

**Goal**
Enable operators to run per-issue validation commands after gate (before review) via `session_end`, and consolidate all run-level validation into a single `run_end` stage.

Target flow:
```
for each issue:
    agent work → gate → session_end trigger → review → finalize
end for
run_end trigger (includes global validation commands)
finalize run
```

**Success criteria**
- Per-issue `session_end` logs: `[trigger] session_end started: issue_id=X` and `[trigger] session_end completed: issue_id=X, result=pass|fail|timeout|skipped`
- Run-level logs show at most one `[trigger] run_end` stage per run (skipped runs log `[trigger] run_end skipped: reason={fire_on_not_met|run_aborted}`); no `[run] GATE` or `global_validation` entries
- Config validation rejects unknown fields (including legacy `global_validation_commands`)
- All existing tests pass after lifecycle restructure
- Behavioral: no orchestrator stage executes run-level validation commands outside `run_end` (asserted via stage graph/event stream); `session_end` evidence is attached to issue review context

**Non-goals**
- Changing the per-issue gate logic or gate command configuration
- Changing fixer agent remediation behavior (beyond scoping session_end remediation to issue)
- Modifying `epic_completion` or `periodic` trigger behavior
- Adding new trigger types

## 2. User Experience & Flows

**Execution model**
Issues are processed concurrently, each on its own per-issue executor. Each issue executes its stages in strict order: `gate → session_end → review → finalize`. Issues may interleave (issue B can be in gate while issue A is in session_end). `run_end` starts only after all issues reach finalized state.

**State machine invariant:** An issue may enter `review` only if:
- `gate.status = pass` AND
- `session_end.status ∈ {pass, fail, timeout, interrupted, skipped}`

Gate "passes" means the gate result is persisted and the state transition is committed. `session_end` runs in the same per-issue executor queue as review (sequentially, not parallel).

**issue.base_sha capture:** `issue.base_sha` is the HEAD commit on the issue branch immediately after workspace checkout/creation and before any agent or fixer writes occur. It is persisted and immutable across retries/remediation for that issue.

### Lifecycle Diagrams

**Per-issue lifecycle (with retry loops):**
```
                              ┌──────────────┐
                              │ ISSUE START  │
                              │ (base_sha    │
                              │  captured)   │
                              └──────┬───────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │   AGENT WORK    │
                            │   (commits)     │
                            └────────┬────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                              GATE (with retry loop)                            │
│                                                                                │
│    ┌──────────────────┐                                                        │
│    │  GATE COMMANDS   │◀─────────────────────────────────┐                     │
│    │  (lint, test,    │                                  │                     │
│    │   typecheck)     │                                  │                     │
│    └────────┬─────────┘                                  │                     │
│             │                                            │                     │
│        ┌────┴────┐                                       │                     │
│        │         │                                       │                     │
│       PASS      FAIL                                     │                     │
│        │         │                                       │                     │
│        │    ┌────┴────────────────────┐                  │                     │
│        │    │                         │                  │                     │
│        │  abort                 continue/remediate       │                     │
│        │    │                         │                  │                     │
│        │    ▼                         ▼                  │                     │
│        │  ┌────────┐           ┌─────────────┐           │                     │
│        │  │ ISSUE  │           │ FIXER AGENT │           │                     │
│        │  │ FAILS  │           │  (repairs)  │           │                     │
│        │  └────────┘           └──────┬──────┘           │                     │
│        │                              │                  │                     │
│        │                         ┌────┴────┐             │                     │
│        │                         │         │             │                     │
│        │                      retries   retries          │                     │
│        │                      remain    exhausted        │                     │
│        │                         │         │             │                     │
│        │                         └────┬────┘             │                     │
│        │                              │                  │                     │
│        │                              └──────────────────┘                     │
│        │                                                                       │
└────────┼───────────────────────────────────────────────────────────────────────┘
         │
         │ GATE PASSED
         ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                          SESSION_END (with retry loop)                         │
│                                                                                │
│    ┌──────────────────┐                                                        │
│    │SESSION_END CMDS  │◀─────────────────────────────────┐                     │
│    │  (additional     │                                  │                     │
│    │   validation)    │                                  │                     │
│    └────────┬─────────┘                                  │                     │
│             │                                            │                     │
│        ┌────┴────┐                                       │                     │
│        │         │                                       │                     │
│       PASS      FAIL                                     │                     │
│        │         │                                       │                     │
│        │    ┌────┴────────────────────┐                  │                     │
│        │    │         │               │                  │                     │
│        │  abort    continue      remediate               │                     │
│        │    │         │               │                  │                     │
│        │    ▼         │               ▼                  │                     │
│        │  ┌────────┐  │        ┌─────────────┐           │                     │
│        │  │  RUN   │  │        │ FIXER AGENT │           │                     │
│        │  │ ABORTS │  │        │  (repairs)  │           │                     │
│        │  └────────┘  │        └──────┬──────┘           │                     │
│        │              │               │                  │                     │
│        │              │          ┌────┴────┐             │                     │
│        │              │          │         │             │                     │
│        │              │       retries   retries          │                     │
│        │              │       remain    exhausted        │                     │
│        │              │          │         │             │                     │
│        │              │          └────┬────┘             │                     │
│        │              │               │                  │                     │
│        │              │               └──────────────────┘                     │
│        │              │               │                                        │
│        └──────────────┴───────────────┘                                        │
│                       │                                                        │
│                       ▼                                                        │
│              ┌─────────────────┐                                               │
│              │   CODE_REVIEW   │  (optional, runs once after final outcome)    │
│              │   (session_end) │                                               │
│              └────────┬────────┘                                               │
│                       │                                                        │
└───────────────────────┼────────────────────────────────────────────────────────┘
                        │
                        ▼
               ┌─────────────────┐
               │     REVIEW      │  (receives SessionEndResult evidence)
               │  (LLM reviewer) │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │    FINALIZE     │
               │  (success/fail) │
               └─────────────────┘
```

**Full run lifecycle (concurrent issues):**
```
                              ┌──────────────┐
                              │  RUN START   │
                              └──────┬───────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
   ┌───────────┐               ┌───────────┐               ┌───────────┐
   │  ISSUE A  │               │  ISSUE B  │               │  ISSUE C  │
   │           │               │           │               │           │
   │  agent    │               │  agent    │               │  agent    │
   │    ↓      │               │    ↓      │               │    ↓      │
   │  gate ◄───┼─ retry loop   │  gate ◄───┼─ retry loop   │  gate ◄───┼─ retry
   │    ↓      │               │    ↓      │               │    ↓      │
   │  session  │               │  session  │               │  session  │
   │  _end ◄───┼─ retry loop   │  _end ◄───┼─ retry loop   │  _end ◄───┼─ retry
   │    ↓      │               │    ↓      │               │    ↓      │
   │  review   │               │  review   │               │  review   │
   │    ↓      │               │    ↓      │               │    ↓      │
   │ finalize  │               │ finalize  │               │ finalize  │
   └─────┬─────┘               └─────┬─────┘               └─────┬─────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  ALL ISSUES FINALIZED │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  RUN_END TRIGGER      │◄── retry loop (if remediate)
                         │  (commands + review)  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    RUN FINALIZE       │
                         └───────────────────────┘
```

**Primary flow (per-issue with session_end)**
1. Agent completes work and commits
2. Gate check runs (validates commit, passes/fails)
3. If gate passes, `session_end` trigger fires (commands then optional code_review, sequentially)
4. Review runs (uses session_end validation evidence if available)
5. Issue finalization (success/failure recorded)

**Primary flow (run-end unified stage)**
1. All issues complete (successes and failures)
2. `run_end` trigger fires once (or logs skipped if `fire_on` condition not met or run aborted)
3. Optional run-end code review runs (if configured and not skipped by failure mode)
4. Run finalization records run-level outcome

**Key states**
- **Gate pass → session_end pending:** Issue awaiting session_end validation before review
- **Gate fail → session_end skipped:** Issue proceeds directly to failure finalization (no session_end)
- **Session_end timeout/interrupt:** Issue proceeds to review with `SessionEndResult.status=timeout|interrupted`
- **Session_end remediation:** State transitions: `session_end_failed → session_end_remediating → [fixer runs] → [re-validate] → session_end_completed|session_end_failed`. Retry loop decrements `max_retries` until pass or exhausted. On exhausted retries, `SessionEndResult.status=fail` and issue proceeds to review (does not fail before review).
- **Run_end in progress:** Unified validation running, all issue work complete
- **Run_end skipped:** `fire_on` condition not met OR run aborted; logs `[trigger] run_end skipped`
- **Run_end failure + abort mode:** Run terminates, code_review skipped
- **Run_end failure + continue/remediate mode:** Proceeds to code_review despite failure

**Run abort contract (failure_mode: abort):**
When any trigger fires `abort` (session_end or run_end command failure with `failure_mode: abort`):
1. In-flight per-issue executors receive a cancellation signal and transition to `interrupted` status
2. In-flight issues are allowed to complete their current command (not hard-killed mid-execution), then finalize as failed with `reason=run_aborted`
3. Already-passed gates remain valid (not reverted)
4. Finalize stages DO run for interrupted issues (to persist state and cleanup)
5. `run_end` is NOT executed; logs `[trigger] run_end skipped: reason=run_aborted`
6. Run-level outcome is `aborted` with the triggering issue/stage noted

**session_end failure_mode behavior:**

| failure_mode | Blocks review? | Issue outcome | Run abort? | Exhausted retries |
|--------------|----------------|---------------|------------|-------------------|
| `abort` | Yes (run stops) | Issue fails | Yes, immediately | N/A (no retries) |
| `continue` | No | Unaffected | No | N/A (no retries) |
| `remediate` | During remediation only | Unaffected | No | Proceed to review with `status=fail` |

Note: With `abort`, the run aborts immediately even if other issues are mid-flight (see Run abort contract). With `continue` or `remediate`, the issue always proceeds to review regardless of session_end outcome.

**Remediation retry accounting:**
- `attempt 1` = initial validation run (before any fixer)
- `max_retries` = number of additional fixer+re-validation cycles after initial failure
- Total validation command executions = 1 + max_retries (if all fail)
- Fixer runs before each retry (not before attempt 1)
- Code_review runs exactly once, after the final validation outcome (pass or exhausted retries)

Example with `max_retries: 2`:
1. Attempt 1: validation fails → fixer runs
2. Attempt 2 (retry 1): validation fails → fixer runs
3. Attempt 3 (retry 2): validation fails → exhausted, proceed with `status=fail`
4. Code_review runs once

**session_end command/code_review execution order:**
Commands execute first, sequentially. Code_review runs after commands complete. If commands fail:
- `failure_mode: abort` → code_review does not run (run aborts)
- `failure_mode: continue` → code_review still runs
- `failure_mode: remediate` → remediation runs; if remediation succeeds, code_review runs; if exhausted, code_review still runs

**session_end config shape:**
```yaml
validation_triggers:
  session_end:
    failure_mode: continue  # abort | continue | remediate (default: continue)
    max_retries: 3          # required when failure_mode: remediate
    commands:
      - ref: lint
      - ref: typecheck
    code_review:
      enabled: true
      reviewer_type: cerberus
      finding_threshold: P1
```
Note: `session_end` does not support `fire_on`; it always fires on gate pass only (see R2).

**Config rules:**
- `validation_triggers.session_end` is optional. If not configured, session_end is skipped for all issues (logs `reason=not_configured`).
- `session_end` with empty `commands: []` and `code_review.enabled: false` is valid (no-op, logs as `pass`).
- `validation_triggers.run_end` is optional. Defaults: `fire_on: success`, `failure_mode: continue`.
- `run_end` with empty `commands: []` and `code_review.enabled: true` is valid (review-only).
- Both triggers live under `validation_triggers` namespace.
- Config validation failures are returned to CLI with the error message AND logged once.

**Config error example:**
```
Error: Unknown field 'global_validation_commands' in mala.yaml
```

**run_end fire_on truth table:**

| success_count | failure_count | fire_on: success | fire_on: failure | fire_on: both |
|---------------|---------------|------------------|------------------|---------------|
| 0 | N (>0) | skip | fire | fire |
| N (>0) | 0 | fire | skip | fire |
| N (>0) | M (>0) | fire | fire | fire |

## 3. Requirements + Verification

**R1 — Per-issue session_end placement**
- **Requirement:** The system MUST execute `session_end` after an issue's gate check passes (gate result persisted, state committed) and before that issue's review begins. Both run on the same per-issue executor sequentially.
- **Verification:** Given an issue with a passing gate, when the lifecycle advances past gate, then `session_end` runs before review starts.

**R2 — Skip session_end on gate failure**
- **Requirement:** The system MUST NOT execute `session_end` for an issue when that issue's gate check fails.
- **Verification:** Given an issue with a failing gate, when the issue transitions to failure handling, then `session_end` is not executed for that issue.

**R3 — Unified run-level stage**
- **Requirement:** The system MUST merge "global validation" into the `run_end` trigger so that there is only one run-level validation stage.
- **Verification:** Given run-level validation is configured, when inspecting run logs after completion, then all validation appears under a single `run_end` stage with no separate `global_validation` stage.

**R4 — Reject legacy global_validation_commands key**
- **Requirement:** If the `global_validation_commands` key is present in config (regardless of whether `run_end` is also configured), the system MUST fail configuration validation with an unknown-field error. The error is returned to CLI and logged.
- **Verification:** Given a config that includes `global_validation_commands` (with or without `run_end`), when config validation runs, then it fails with an unknown-field error.

**R5 — Session_end validation evidence for review**
- **Requirement:** The review stage MUST receive a `SessionEndResult` record when session_end completes (successfully or otherwise). The `SessionEndResult` is informational: review MUST NOT auto-fail solely due to `session_end.status=fail`; instead, review surfaces the results in the evidence bundle for the reviewer prompt.
- **Data contract:**
  ```
  SessionEndResult {
    status: pass|fail|timeout|interrupted|skipped,
    started_at: timestamp|null,      # null only when status=skipped
    finished_at: timestamp|null,     # null only when status=skipped
    commands: [{ref, passed, duration_seconds, error_message}],
    code_review_result: {ran: bool, passed: bool|null, findings: [...]}|null,
    reason: string|null              # e.g., "gate_failed", "not_configured", "SIGINT", "max_retries_exhausted"
  }
  ```
- **Timestamp rules:** `started_at` and `finished_at` are non-null for all executed attempts (pass, fail, timeout, interrupted). They are null only when `status=skipped`.
- **Persistence:** `SessionEndResult` is stored in the issue record, keyed by `(run_id, issue_id)`. On crash recovery (started but not finished), status is set to `interrupted` with `reason=process_crash` on restart.
- **Partial results:** Partial command outcomes from timeout/interrupt are NOT persisted in `SessionEndResult.commands` (discarded for consistency). Partial output is available only in logs.
- **Verification:** Given session_end fails for issue X, when review runs for issue X, then review context includes the `SessionEndResult` with `status=fail` and review proceeds (does not auto-reject).

**R6 — Allow review-only run_end**
- **Requirement:** The system MUST allow `run_end` to have empty/no commands when run-end code review is enabled.
- **Verification:** Given `run_end` with no commands and code review enabled, when a run completes, then `run_end` executes and performs code review without config validation errors.

**R7 — run_end fire_on behavior**
- **Requirement:** The system MUST execute `run_end` subject to `fire_on` behavior. If `fire_on` condition is not met, `run_end` does not execute its commands/code_review but MUST log `[trigger] run_end skipped: reason=fire_on_not_met`. If run aborts before all issues finalize, `run_end` MUST log `[trigger] run_end skipped: reason=run_aborted`.
- **fire_on values:** `success` (fires if success_count > 0), `failure` (fires if failure_count > 0), `both` (always fires). Default: `success`.
- **Verification:**
  - Given all issues fail and `fire_on: both`, then `run_end` executes.
  - Given all issues fail and `fire_on: success`, then `run_end` logs skipped.
  - Given 1 success + 1 failure and `fire_on: success`, then `run_end` executes (mixed outcome).
  - Given run aborts mid-flight, then `run_end` logs `skipped: reason=run_aborted`.

**R8 — Remove RunCoordinator.run_validation**
- **Requirement:** The system MUST remove the separate run validation stage by deleting `RunCoordinator.run_validation` and ensuring no run-level validation path exists outside `run_end`.
- **Verification:** Given run-level validation is configured, when inspecting run logs, then all run-level validation appears under `[trigger] run_end` with no separate `[run] GATE` or `[global_validation]` entries.

**R9 — Session_end failure_mode and remediation**
- **Requirement:** The `session_end` trigger MUST support its own `failure_mode` configuration (`abort`, `continue`, `remediate`) with semantics per the failure_mode table in §2. When `failure_mode: remediate`, the fixer runs and validation commands are re-executed (retry loop) until pass or `max_retries` exhausted. Validation executes up to `1 + max_retries` times total. Remediation MUST block only that issue's review and MUST NOT block other issues' progression.
- **Default:** `failure_mode: continue`.
- **Verification:** Given `session_end` with `failure_mode: remediate` and `max_retries: 2`, when validation fails 3 times, then fixer runs twice (after failures 1 and 2), and issue proceeds to review with `status=fail, reason=max_retries_exhausted`.

**R10 — Session_end timeout/interrupt behavior**
- **Requirement:** If `session_end` times out or is interrupted, the system MUST proceed to review for that issue with a `SessionEndResult` having `status=timeout|interrupted` and empty command results. Partial results from completed commands are intentionally discarded for consistency (available in logs only).
- **Verification:** Given `session_end` is interrupted after 1 of 3 commands completed, when the orchestrator handles the interruption, then the issue's review runs and receives `SessionEndResult{status: interrupted, commands: [], reason: "SIGINT received"}`.

**R11 — Session_end code review baseline**
- **Requirement:** If `session_end` includes code review, it MUST use the commit SHA captured at issue session start (`issue.base_sha`) as the baseline, reviewing only commits in `[base_sha, HEAD]`.
- **Verification:** Given `session_end` code review is enabled for an issue that started at commit `abc123` and is now at `def456`, when code review runs, then only changes between `abc123` and `def456` are reviewed.

**R12 — run_end failure_mode gating for code review**
- **Requirement:** When `run_end` commands fail, whether run-end code review runs MUST follow `failure_mode`: `abort` skips review; `continue` and `remediate` proceed to review.
- **Verification:** Given a `run_end` command failure, when `failure_mode` is `abort`, then code review does not run; when `failure_mode` is `continue` or `remediate`, then code review runs.

## 4. Instrumentation & Release Checks

**Instrumentation**
- Per-issue session_end: `[trigger] session_end started: issue_id=X` and `[trigger] session_end completed: issue_id=X, result={pass|fail|timeout|interrupted|skipped}`
- Session_end skip: `[trigger] session_end skipped: issue_id=X, reason={gate_failed|not_configured}`
- Run-end unified stage: `[trigger] run_end started: success_count=N, total_count=M` and `[trigger] run_end completed: result={pass|fail}` OR `[trigger] run_end skipped: reason={fire_on_not_met|run_aborted}`
- Config validation failure: `[config] error: Unknown field 'global_validation_commands' in mala.yaml` (also returned to CLI)

**Release verification checklist**
- [ ] Run logs contain at most one `[trigger] run_end` entry per run (either started/completed or skipped)
- [ ] No `[run] GATE` or `global_validation` entries appear in run logs
- [ ] Per-issue logs show `session_end started` after `gate_passed` and before `review_start` for same issue_id
- [ ] Config with `global_validation_commands` key produces unknown-field validation error (in CLI and logs)
- [ ] When `fire_on` condition not met, logs show `run_end skipped` (not missing)
- [ ] When run aborts, logs show `run_end skipped: reason=run_aborted`
- [ ] No orchestrator stage executes run-level validation commands outside `run_end` (verified via stage graph)

**Decisions made**
- Session_end fires only on gate pass (skip if gate fails); does not support fire_on
- Reject config if legacy global_validation_commands key is present (no silent ignore)
- Session_end runs after gate, before review (additive, not replacement)
- Session_end has its own failure_mode config; default is `continue`
- SessionEndResult is informational for review (review does not auto-fail on session_end failure)
- SessionEndResult includes code_review_result field for code review evidence
- SessionEndResult persisted in issue record, keyed by (run_id, issue_id); crash recovery marks as interrupted
- Allow review-only run_end (empty commands OK if code_review enabled)
- run_end logs skipped when fire_on condition not met OR run aborted (not silent omission)
- Delete RunCoordinator.run_validation entirely (no fallback)
- Session_end remediation blocks only current issue's review, not other issues (per-issue executor)
- Session_end remediation: validation runs 1 + max_retries times; fixer runs before each retry; code_review runs once after final outcome
- Session_end timeout/interrupt → proceed to review with explicit `SessionEndResult{status: timeout|interrupted}`, partial results discarded (logs only)
- Session_end code_review baseline: commits since `issue.base_sha` (captured at workspace checkout, before any agent writes)
- Session_end commands run before code_review; code_review runs regardless of command outcome (unless abort)
- run_end fail-through: configurable via failure_mode (abort skips review, continue/remediate proceeds)
- Run abort: in-flight issues complete current command then finalize as failed; run_end skipped with reason=run_aborted
- Migration error surfaced in CLI output AND logged, includes example YAML snippet

**Open questions**
(All resolved during review iterations)
