# Epic Verification Reviewer Choice

**Tier:** M
**Owner:** [TBD]
**Target ship:** [TBD]
**Links:** Related: `src/infra/epic_verifier.py`, `src/domain/validation/config.py`

## 1. Outcome & Scope

**Problem / context**
Epic verification currently runs only via the Agent SDK path. Unlike code review—which offers `reviewer_type: "cerberus" | "agent_sdk"`—epic verification has no equivalent choice. Teams that standardize on Cerberus (for consistent tooling, operational constraints, or policy reasons) cannot use Cerberus for epic verification today.

**Goal**
Enable users to choose between Cerberus or Agent SDK for epic verification via configuration, mirroring the existing code review pattern.

**Success criteria**
- Epic verification completes successfully when configured with `reviewer_type: cerberus` (produces valid verdict with pass/fail and unmet criteria)
- Epic verification completes successfully when configured with `reviewer_type: agent_sdk` (default, maintains current behavior)
- When Cerberus is configured but unavailable, the run fails early with a clear, actionable error explaining the missing dependency
- Observability reflects which reviewer implementation was used (via `reviewer_type` field in emitted events)

**Non-goals**
- Changing what epic verification evaluates (acceptance criteria semantics, remediation/advisory issue logic, pass/fail meaning)
- Adding new verification capabilities beyond what exists today
- Deprecating the Agent SDK option
- Adding new event types beyond including `reviewer_type` in existing events

## 2. User Experience & Flows

**Primary flow**
1. User configures `epic_verification.reviewer_type: cerberus` in `mala.yaml` (new top-level section)
2. Mala discovers eligible epics (all children closed)
3. System validates Cerberus availability before starting verification
4. System invokes `review-gate spawn-epic-verify` command with epic context
5. Verification result (pass/fail + unmet criteria) returned and processed
6. Mala continues with existing downstream behavior (remediation/advisory issue creation), unchanged

**Key states**
- **Cerberus available, configured**: Use Cerberus for verification
- **Cerberus unavailable, configured**: Fail with explicit error indicating Cerberus is not installed
- **Agent SDK configured (default)**: Use existing Agent SDK implementation
- **Unrecognized reviewer_type**: Fail with error listing valid options

## 3. Requirements + Verification

**R1 — Configuration section with reviewer_type**
- **Requirement:** The system MUST accept a new top-level `epic_verification` section with `reviewer_type: "cerberus" | "agent_sdk"`, defaulting to `"agent_sdk"` when not specified.
- **Verification:**
  - Given `mala.yaml` with `epic_verification.reviewer_type: cerberus`, when epic verification runs, then Cerberus-based verification is invoked
  - Given `mala.yaml` with `epic_verification.reviewer_type: agent_sdk`, when epic verification runs, then Agent SDK verification is invoked
  - Given `mala.yaml` without `epic_verification.reviewer_type`, when epic verification runs, then Agent SDK is used (current behavior preserved)

**R2 — Configuration schema parity with code review**
- **Requirement:** The `epic_verification` section MUST support: `enabled`, `reviewer_type`, `timeout`, `max_retries`, `failure_mode`, and a nested `cerberus` block matching the structure used by code review configuration.
- **Verification:**
  - Example `mala.yaml`:
    ```yaml
    epic_verification:
      enabled: true
      reviewer_type: cerberus  # or "agent_sdk"
      timeout: 600
      max_retries: 3
      failure_mode: continue  # "abort", "continue", or "remediate"
      cerberus:
        timeout: 300
        spawn_args: []
        wait_args: []
    ```
  - Given `mala.yaml` with valid `epic_verification` settings, when configuration is loaded, then the run proceeds without config-validation errors
  - Given `mala.yaml` with invalid `epic_verification.reviewer_type`, when configuration is loaded, then a clear config error is raised indicating allowed values
- **Schema & Defaults:**
  | Field | Type | Required | Default | Notes |
  |-------|------|----------|---------|-------|
  | `enabled` | bool | No | `true` | Matches current epic verification behavior |
  | `reviewer_type` | `"cerberus"` \| `"agent_sdk"` | No | `"agent_sdk"` | Backwards compatible |
  | `timeout` | int (seconds) | No | `600` | Top-level timeout for verification |
  | `max_retries` | int | No | `3` | Retries for transient failures |
  | `failure_mode` | `"abort"` \| `"continue"` \| `"remediate"` | No | `"continue"` | Matches code review |
  | `cerberus.timeout` | int (seconds) | No | `300` | Overrides `timeout` for Cerberus path |
  | `cerberus.spawn_args` | list[str] | No | `[]` | Additional args to spawn command |
  | `cerberus.wait_args` | list[str] | No | `[]` | Additional args to wait command |
- **Timeout precedence:** `cerberus.timeout` (if set) overrides `epic_verification.timeout` when `reviewer_type: cerberus`. For Agent SDK, only `epic_verification.timeout` applies.
- **Config placement:** `epic_verification` lives at the top level of `mala.yaml`, alongside `per_issue_review` and `validation_triggers`. It does not conflict with existing `validation_triggers.epic_completion` settings, which control when verification triggers (not how it runs).
- **Implementation notes (config loading):**
  - Extend `src/domain/validation/config.py` with a new `EpicVerificationConfig` dataclass mirroring `CodeReviewConfig` structure
  - **Canonical config path:** `ValidationConfig.epic_verification: EpicVerificationConfig`. No other config keys influence verifier selection.
  - Invalid `reviewer_type` MUST fail during config load (not mid-run) with a clear error listing valid values
  - **Unknown field handling:** Unknown fields under `epic_verification` MUST raise a config validation error (strict parsing, no silent ignore)
  - Defaults (e.g., `enabled: true`, `reviewer_type: "agent_sdk"`) are applied at config parse time
  - `validation_triggers.epic_completion` controls **when** epic verification triggers; `epic_verification.*` controls **how** it runs (reviewer selection, retries, timeouts)

**R3 — Protocol abstraction with two implementations**
- **Requirement:** The system MUST define an `EpicVerifierProtocol` interface enabling two implementations (Cerberus and Agent SDK), such that orchestration selects based on configuration without changing downstream behavior. Both MUST return an `EpicVerdict` containing pass/fail status, unmet criteria list, confidence, and reasoning.
- **EpicVerdict schema:** (matches existing `src/core/protocols/validation.py:EpicVerdict`)
  | Field | Type | Required | Constraints |
  |-------|------|----------|-------------|
  | `passed` | `bool` | Yes | `True` = all acceptance criteria met |
  | `unmet_criteria` | `list[UnmetCriterion]` | Yes | Empty if passed; each has `criterion: str`, `priority: int (0-3)`, `reason: str` |
  | `confidence` | `float` | Yes | Range 0.0–1.0 |
  | `reasoning` | `str` | Yes | Explanation of verdict |
- **Verification:**
  - Given either reviewer type selected, when epic verification completes, then the orchestration receives a verdict in the same shape and semantics expected by existing epic verification flow
  - Given either reviewer type encounters an error, then error handling follows `failure_mode` configuration

**R4 — Cerberus epic verification command**
- **Requirement:** The Cerberus implementation MUST invoke `review-gate spawn-epic-verify` followed by `review-gate wait` (async spawn-wait pattern, matching code review) to perform epic verification.
- **Context passed to Cerberus:**
  - **Epic file:** Write the epic description (including acceptance criteria) to a temporary markdown file and pass its path to `spawn-epic-verify`. Review-gate extracts acceptance criteria and spec references from the epic file.
  - **Commit scope:** Prefer `--commit <sha...>` using the scoped commit list from epic child issues. If no commits are available, fall back to a range argument (if available) or `--uncommitted`.
  - **Session scoping:** Always set `CLAUDE_SESSION_ID` to a generated value with the epic ID as prefix and a random suffix (e.g., `EPIC-42-1a2b3c4d5e6f`).
- **Minimal I/O contract (v0):**
  - **Input:** `spawn-epic-verify <epic-file> [diff args]`
  - **Output:** `wait --json` returns review-gate JSON with `status`, `consensus_verdict`, `aggregated_findings`, and `parse_errors`.
    - `consensus_verdict: PASS` → `EpicVerdict.passed = true`
    - `consensus_verdict: FAIL|NEEDS_WORK` → `EpicVerdict.passed = false`
    - `aggregated_findings` map to `unmet_criteria` entries (title/body/priority)
  - **Error classification:** `status=timeout` or exit code 3 → `timeout`. `status=error|no_reviewers` or `consensus_verdict=ERROR` → `execution_error`. Invalid JSON → `parse_error`.
- **Verification:**
  - Given `reviewer_type: cerberus`, when epic verification runs, then `spawn-epic-verify` is invoked followed by `wait`
  - Given Cerberus returns valid wait JSON, when parsed, then an `EpicVerdict` with pass/fail and unmet criteria is returned

**R5 — Availability checking with explicit failure**
- **Requirement:** When `epic_verification.reviewer_type` is `"cerberus"`, the system MUST validate Cerberus availability before attempting verification and MUST fail immediately with an explicit, actionable error if Cerberus is unavailable. This availability check is a **hard fail** that bypasses `failure_mode`—misconfiguration (requesting Cerberus when unavailable) always aborts.
- **Availability check mechanism:** Use the existing `find_cerberus_bin_path()` utility (same as code review) to locate the `review-gate` binary via Claude's `installed_plugins.json`. "Unavailable" means either: (a) no Cerberus plugin entry in installed plugins, or (b) the `review-gate` binary path does not exist or is not executable.
- **Subcommand detection (normative):** Run `<review-gate> spawn-epic-verify --help` and require exit code 0. If exit code is non-zero, treat as `cerberus_unavailable`. Include the first 200 chars of stderr in the actionable error message (e.g., "Cerberus plugin does not support epic verification: <stderr snippet>. Update plugin or use reviewer_type: agent_sdk.").
- **Hard-fail categories:** `config_error` and `cerberus_unavailable` (including missing `spawn-epic-verify` subcommand) are hard-fail categories. These abort regardless of `failure_mode`, do not count toward `max_retries`, and do not emit `max_retries_exceeded`.
- **Verification:**
  - Given Cerberus binary is unavailable and `reviewer_type: cerberus` configured, when epic verification attempts to run, then an error is raised before any `spawn-epic-verify` invocation
  - Given Cerberus binary exists but fails validation, when availability check runs, then an appropriate error is raised indicating Cerberus is required
  - Given `failure_mode: continue` and Cerberus unavailable, when epic verification runs, then the run still aborts (availability failure bypasses failure_mode)

**R6 — Retry policy**
- **Requirement:** The system MUST retry verification failures up to `max_retries` times for transient failures, following the policy below.
- **Retry policy:**
  | Failure Category | Retryable | Counts Toward max_retries | failure_mode Applies |
  |------------------|-----------|---------------------------|----------------------|
  | `cerberus_unavailable` | No | No | No (hard fail) |
  | `config_error` | No | No | No (hard fail) |
  | `timeout` | Yes | Yes | Yes |
  | `execution_error` (subprocess crash) | Yes | Yes | Yes |
  | `parse_error` (invalid output) | Yes | Yes | Yes |
  | `verdict_fail` (verification failed) | No | No | Yes |
- **Backoff:** No backoff between retries (immediate retry). Implementation MAY add exponential backoff if needed.
- **Parse error retry guidance:** `parse_error` retries are best-effort for transient issues (empty/truncated output). If the same parse error recurs on retry (deterministically invalid output suggesting contract mismatch), implementation SHOULD log the raw output snippet and stop retrying early.
- **failure_mode semantics:**
  - `abort`: After retries exhausted or non-retryable error, abort the entire mala run
  - `continue`: After retries exhausted or non-retryable error, log the failure and continue the run (epic remains open)
  - `remediate`: Same as `continue`, but additionally create remediation issues for the failure
  - **verdict_fail ordering rule:** On `verdict_fail` (verification returns `passed=false`), ALWAYS run existing remediation/advisory issue creation first; THEN apply `failure_mode` to decide whether to abort or continue. This means `failure_mode: abort` aborts AFTER downstream remediation logic completes, not before. Since remediation already happens on `verdict_fail`, `failure_mode: remediate` is effectively equivalent to `continue` for this category.
- **Verification:**
  - Given a timeout on first attempt and `max_retries: 2`, when verification runs, then up to 2 additional attempts are made
  - Given `cerberus_unavailable`, when verification runs, then no retries occur (non-retryable)

**R7 — Event observability**
- **Requirement:** Epic verification events MUST include `reviewer_type` field with value `"cerberus"` or `"agent_sdk"`, matching the implementation used for that attempt.
- **Event schema (required fields):**
  | Event | Required Fields |
  |-------|-----------------|
  | `epic_verification_started` | `epic_id`, `reviewer_type`, `attempt` (1-indexed) |
  | `epic_verification_passed` | `epic_id`, `reviewer_type`, `attempt` |
  | `epic_verification_failed` | `epic_id`, `reviewer_type`, `attempt`, `failure_reason` |
- **`failure_reason` enum:** `cerberus_unavailable`, `config_error`, `timeout`, `execution_error`, `parse_error`, `verdict_fail`, `max_retries_exceeded`
- **Event emission rules:**
  - **Emitter:** The orchestration layer (e.g., `EpicVerifier`) owns event emission. Verifier implementations (Cerberus/AgentSDK) MUST NOT emit events directly.
  - **Per-attempt emission:**
    - Emit `epic_verification_started` at the start of **each** attempt
    - Emit `epic_verification_failed` for **each** failed attempt with that attempt's `failure_reason`
    - Emit `epic_verification_passed` only on the attempt that succeeds
  - **Terminal emission rules:**
    - Emit `max_retries_exceeded` only when all `max_retries + 1` attempts were made and all failed with retryable errors
    - For non-retryable failures (`verdict_fail`, `cerberus_unavailable`, `config_error`): terminate with that attempt's `epic_verification_failed` event only—do NOT emit `max_retries_exceeded`
    - Hard-fail categories (`cerberus_unavailable`, `config_error`) emit a single `failed` event and abort immediately
- **Verification:**
  - Given epic verification starts, when the "started" event is emitted, then the event includes `epic_id`, `reviewer_type`, and `attempt`
  - Given epic verification fails on attempt 1 with timeout and succeeds on attempt 2, then events emitted are: `started(attempt=1)`, `failed(attempt=1, reason=timeout)`, `started(attempt=2)`, `passed(attempt=2)`
  - Given epic verification fails all 3 attempts with timeout, then events include 3 `failed` events plus a final `failed(reason=max_retries_exceeded)`

## 4. Instrumentation & Release Checks

**Instrumentation**
- Events to track: `epic_verification_started`, `epic_verification_passed`, `epic_verification_failed`
- All events MUST include: `epic_id`, `reviewer_type`, `attempt`
- Failed events MUST include `failure_reason` from the defined enum (see R7)

**Decisions made**
- Config lives in new top-level `epic_verification` section (parallel to `per_issue_review`)
- When Cerberus unavailable but configured: fail with explicit error
- Cerberus uses new dedicated `review-gate spawn-epic-verify` command
- Config mirrors code review: `reviewer_type`, `timeout`, `max_retries`, `failure_mode`, cerberus-specific settings
- Create `EpicVerifierProtocol` with Cerberus and AgentSDK implementations
- Events include `reviewer_type` for observability
- `failure_mode` uses existing values: `abort`, `continue`, `remediate` (default: `continue`)

**External dependency**
- **Cerberus plugin requirement:** This feature requires a Cerberus plugin version that provides `spawn-epic-verify`. If the required command is missing (e.g., older plugin version), the availability check MUST treat this as `cerberus_unavailable` with an actionable error message: "Cerberus plugin does not support epic verification. Update to version X+ or use reviewer_type: agent_sdk."
- **Version detection:** The availability check MAY validate command existence via `review-gate --help` output or by attempting `spawn-epic-verify --help`. If version-specific detection is impractical, treat any missing subcommand as `cerberus_unavailable`.

**Open questions**
- None. The minimal I/O contract (v0) in R4 provides enough detail for implementation and testing. Full CLI ergonomics may evolve as the Cerberus plugin matures.
