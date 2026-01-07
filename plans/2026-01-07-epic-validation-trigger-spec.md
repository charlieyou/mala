# Epic Completion Global Validation Option

**Tier:** L
**Owner:** [TBD]
**Target ship:** [TBD]
**Links:** [TBD]

## 1. Outcome & Scope

**Problem / context**
Currently, global validations run only after ALL issues in a session complete, with no control over which validations run at what points. This creates two problems:

1. **Late feedback:** For long-running sessions with multiple epics, validation feedback comes very late—potentially hours after an epic finishes. If an epic introduces validation failures, we don't discover this until the entire session finishes.

2. **All-or-nothing validation:** Users can't run different validation subsets at different checkpoints. For example, they might want to run fast validations (type checking, linting) after each epic, while reserving expensive validations (E2E tests, comprehensive coverage) for session end.

**Goal**
Enable users to configure exactly which validation commands (standard + custom) run at specific trigger points (epic completion, session end, periodic, etc.). This allows:
- Faster feedback by running appropriate validations at each checkpoint
- Cost-effective validation by running expensive tests only when needed
- Flexible validation strategies tailored to project workflow

**Success criteria**
- **Behavioral: Fast feedback queueing:** Given epic_completion trigger configured, when epic verification completes, Then validation is queued within 10 seconds and 'queued' event is emitted with correct trigger_type and context.
- **Behavioral: Correct trigger execution:** Given periodic trigger with interval=5, when 5 non-epic issues complete, Then periodic validation queues exactly once. Epic issue completions do NOT increment counter.
- **Behavioral: Migration clarity:** Given old config with only `global_validation_commands`, when loading, Then startup fails with error message containing migration guide URL and concrete example of `validation_triggers.session_end` block.
- **Behavioral: Remediation loop correctness:** Given fixable validation failure (e.g., lint error), when `failure_mode=remediate` with `max_retries=2`, Then system spawns fixer agent, emits remediation events, re-runs validation, and aborts on exhaustion if unfixed.

**Non-goals**
- Not removing or replacing the concept of global vs per-session validation scope (per-session validations still exist and run after each issue)
- Not supporting per-session (repo-level) commands in trigger configuration (only global validation commands)
- Not adding parallel validation runs (validation runs are always sequential/queued)
- Not supporting manual user intervention on validation failures (autonomous only: abort or remediate)

**Constraints**
- **BREAKING CHANGE #1:** `global_validation_commands` behavior changes - it now defines the base command pool (not auto-run commands). Migration check applies to **effective merged config** (preset + project): if merged config has `global_validation_commands` with any commands (non-empty) AND no `validation_triggers` defined, startup MUST fail with error: "validation_triggers required. See migration guide at https://docs.mala.ai/migration/validation-triggers". Preset-only configs (e.g., `preset: python-uv` with no project `validation_triggers`) trigger this error.
- **BREAKING CHANGE #2:** `validate_every` field MUST cause hard startup error with message: "validate_every is deprecated. Use validation_triggers.periodic with interval field. See migration guide at https://docs.mala.ai/migration/validation-triggers"
- **Configuration format:** Command list structure - base pool (`global_validation_commands`) + per-trigger command list (`validation_triggers.<trigger>.commands: [{ref: name, ...}, ...]`)
- **Supported triggers:** Must support trigger types: `epic_completion`, `session_end`, `periodic` (NOT `issue_completion` - redundant with existing per-session validation)
- **Trigger execution order:** Per-session validation runs first (existing behavior), then global validation triggers queue (if configured). **Global trigger validation execution MUST block next issue from starting** to prevent workspace conflicts (validation commands operate on entire workspace). During multi-agent sessions (max_agents > 1), active agents continue running, but NO new issue assignments occur until validation completes. During remediation mode, active agents are NOT paused/drained - fixer runs concurrently with them (acceptable workspace conflict risk, constrained to remediation phase only).
- **Epic depth:** `epic_completion` trigger must support `epic_depth: 'top_level' | 'all'`
- **Epic verification condition:** `epic_completion` trigger must support `fire_on: 'success' | 'failure' | 'both'`
- **Periodic interval:** `periodic` trigger `interval` field counts completed non-epic issues using continuous counter `non_epic_completed_count`. Triggers at `interval, 2*interval, 3*interval, ...` (NOT reset after fire). Counter increments AFTER per-session validation completes (regardless of per-session validation outcome), BEFORE checking periodic trigger condition. If run aborts during per-session validation, counter does NOT increment.
- **Validation queuing:** Global validation runs execute sequentially. No parallel runs, no deduplication. Queue persists in memory only (lost on crash).
- **Abort behavior on trigger failure:** When `failure_mode=abort` triggers: (a) in-progress issue execution finishes current step then stops, (b) queued validations are discarded and emit `skipped` events with `reason=run_aborted`, (c) in-progress validation subprocess is interrupted (SIGTERM), (d) session_end trigger does NOT fire.
- **User interrupt handling (SIGINT):** When user sends SIGINT (Ctrl+C) during validation or remediation: (a) in-progress validation/remediation subprocess is interrupted (SIGTERM), (b) blocking wait is immediately released, (c) run enters abort state (no more issues processed), (d) session_end trigger does NOT fire. User interrupt is always responsive (no indefinite blocking).
- **Epic depth definition:** `epic_depth: 'top_level'` means epic has NO epic parent (parent is null or not an epic). `epic_depth: 'all'` means ANY epic closure fires trigger regardless of nesting level. Leaf epics (no children) still fire if they match depth rule.
- **Success count scope:** `success_count` counts ALL successfully completed units of work (both non-epic issues AND successfully closed epics). Used for session_end skip rule: if `success_count == 0`, session_end validation is skipped.
- **Failure mode:** `failure_mode` is REQUIRED for each trigger (no default). Schema validation MUST fail at startup if omitted.
- **Fail-fast within trigger:** Commands execute sequentially. First failure immediately stops execution of remaining commands in that trigger (fail-fast).
- **Terminal behavior after remediation:** When `failure_mode='remediate'` exhausts `max_retries`, system MUST abort run (same as `failure_mode='abort'`).
- **Config validation:** ALL config errors (malformed fields, invalid enums, missing commands) MUST fail at startup during spec building (fail-fast validation)
- **Base command pool merge:** When preset + project both define `global_validation_commands`, project commands override preset commands by name with **field-level deep merge**. If project defines `{test: {timeout: 120}}` and preset has `{test: {command: 'pytest', timeout: 300}}`, effective pool has `{test: {command: 'pytest', timeout: 120}}`. Project-only fields override; unspecified fields inherit from preset. Triggers reference the merged effective pool. Note: This field-level merge applies ONLY to `global_validation_commands`; per-trigger command list entries are override-only (no inheritance).
- **Command schema:** Base pool commands are objects with fields: `command` (string, required), `timeout` (int, optional, defaults to 120 seconds if omitted). Per-trigger command list entries have: `ref` (string, required), `command` (string, optional override), `timeout` (int, optional override). Timeout resolution: per-trigger override > base pool timeout > system default (120s). NO other fields supported (no cwd, env, shell, retries, allow_fail).

## 2. User Experience & Flows

**Primary flow**

**Epic Completion Trigger Flow:**
1. User configures mala.yaml with `validation_triggers.epic_completion` block (specifies include/override commands, failure_mode, max_retries, epic_depth, fire_on)
2. During session run, an epic becomes eligible for closure (all children closed)
3. Epic verification runs and completes (pass or fail)
4. System checks epic_completion trigger's `fire_on` setting:
   - If `fire_on: 'success'` and verification passed → queue validation
   - If `fire_on: 'failure'` and verification failed → queue validation
   - If `fire_on: 'both'` → always queue validation
   - Otherwise → skip validation
5. Validation queued and executes when queue is available (sequential execution)
6. Validation commands run in declaration order (from `include` list, with `override` applied)
7. On validation result:
   - **Pass:** Log success, continue session to next issue
   - **Fail + failure_mode='continue':** Log failure, continue session
   - **Fail + failure_mode='abort':** Stop run gracefully (no more issues processed)
   - **Fail + failure_mode='remediate':** Spawn fixer agent (like session_end), retry up to max_retries, then abort if still failing

**Session End Trigger Flow:**
1. All issues complete AND success_count > 0 AND session not aborted
2. session_end trigger (if configured) queues validation
3. Same validation execution flow as above (commands → result → failure mode handling)

**Periodic Trigger Flow:**
1. Non-epic issue completes (epic completions do NOT increment counter)
2. Per-session validation runs
3. Increment `non_epic_completed_count`
4. Check if `non_epic_completed_count % periodic.interval == 0`
5. If yes, queue periodic validation (continuous counter, triggers at interval, 2*interval, 3*interval, ...)
6. Same validation execution flow as above

**Key states**
- **Empty state (no triggers configured):** No validation runs at any checkpoint (explicit opt-in)
- **Validation queued:** Trigger fired, waiting for previous validation to complete (sequential queue)
- **Validation running:** Commands executing in order, displaying progress/logs
- **Validation passed:** All commands succeeded, session continues
- **Validation failed (continue mode):** Failure logged, session continues to next issue
- **Validation failed (abort mode):** Run stops gracefully, no more issues processed
- **Validation failed (remediate mode, attempt N):** Fixer agent spawned, attempting to fix issues
- **Remediation exhausted:** max_retries reached, all attempts failed, run aborts

**Alternate flows**

**Multiple Epics Complete Rapidly:**
- Epic 1 completes → validation queued → validation starts
- Epic 2 completes (while Epic 1 validation running) → validation queued
- Epic 1 validation completes → Epic 2 validation starts
- Sequential queue ensures no conflicts

**Validation Fails Mid-Session (remediate mode):**
1. Epic completes, validation fails, failure_mode='remediate'
2. Fixer agent spawns, receives formatted failure output
3. Fixer commits fixes (or fails to fix)
4. Validation re-runs (attempt 2)
5. If passes → session continues
6. If fails and attempts < max_retries → repeat fixer loop
7. If fails and attempts >= max_retries → abort run

**User Aborts Session (Ctrl+C):**
- session_end trigger is skipped (matches current behavior)
- In-progress validation is interrupted
- Run exits gracefully

**Epic Verification Fails:**
- If epic_completion trigger has `fire_on: 'failure'` or `fire_on: 'both'` → validation runs even though epic failed
- If `fire_on: 'success'` → validation skipped

**Trigger References Missing Command:**
- During config loading (before any issues run), spec building fails with clear error:
  - "epic_completion trigger references unknown command 'typo_test'. Available commands: test, lint, typecheck, custom_reviewer"
- Run never starts (fail-fast validation)

## 3. Requirements + Verification

**R1 — Hard Migration Error for Old Configs**
- **Requirement:** The system MUST fail at startup with a clear migration error if **effective merged config** (preset + project) has non-empty `global_validation_commands` without `validation_triggers`, OR if `validate_every` is present. Error messages MUST include migration guide URL and concrete example.
- **Verification:**
  - Given a project config with `global_validation_commands` but no `validation_triggers`, When loading config, Then system fails with error: "validation_triggers required. See migration guide at https://docs.mala.ai/migration/validation-triggers" and error includes concrete `validation_triggers.session_end` example.
  - Given a project config with `preset: python-uv` (provides `global_validation_commands`) but no project-level `validation_triggers`, When loading config, Then system fails with migration error (preset-provided commands trigger the check).
  - Given a project config with `validate_every: 5`, When loading config, Then system fails with error: "validate_every is deprecated. Use validation_triggers.periodic with interval field. See migration guide at https://docs.mala.ai/migration/validation-triggers"
  - Given a project config with neither `global_validation_commands` nor `validation_triggers` nor preset, When loading config, Then no validation runs (valid - explicit opt-out)
  - Given a project config with both `global_validation_commands` and `validation_triggers`, When loading config, Then config loads successfully and triggers execute as configured
- **Edge cases:**
  - Config has empty `validation_triggers: {}` → valid (no triggers configured, no validation runs)
  - Config has `validation_triggers` with zero commands in all triggers → valid (triggers fire but skip validation)
  - Config has `global_validation_commands: {}` (empty) with `validation_triggers` → valid (no commands in pool, triggers reference nothing)
  - Config has `global_validation_commands: {}` (empty) without `validation_triggers` → valid (empty pool, no error - check only applies to non-empty pool)
  - Custom command name shadows standard command name → custom takes precedence (defined in global_validation_commands)

**R2 — Supported Triggers and Firing Rules**
- **Requirement:** The system MUST support triggers `epic_completion`, `periodic`, and `session_end` (NOT `issue_completion`), applying trigger-specific fields and skip rules:
  - `epic_completion` supports `epic_depth` (`top_level` or `all`) and `fire_on` (`success`, `failure`, `both`).
  - `periodic` supports `interval` (integer, counts completed non-epic issues only; epics do NOT increment counter).
  - `session_end` MUST be skipped if `abort_run=True` OR `success_count==0` (success_count includes both non-epic issues AND successfully closed epics).
  - ALL triggers MUST have explicit `failure_mode` (no default; schema validation fails if omitted).
- **Verification:**
  - Given `epic_completion` with `epic_depth=top_level`, When a nested epic completes, Then no epic-completion validation runs for that nested epic.
  - Given `epic_completion` with `epic_depth=all`, When a nested epic completes, Then epic-completion validation runs for that epic.
  - Given `epic_completion` with `fire_on=success`, When an epic verification fails, Then epic-completion validation does not run; When it succeeds, Then it runs.
  - Given `periodic` with `interval: 5`, When `non_epic_completed_count` reaches 5, 10, 15, ..., Then periodic validation queues at each milestone (continuous counter). When epic issues complete, Then counter does NOT increment.
  - Given a run that is aborted (`abort_run=True`), When the session ends, Then `session_end` validation does not run even if configured.
  - Given a run where all issues failed (both epics and non-epics, `success_count=0`), When the session ends, Then `session_end` validation does not run even if configured. Given a run with only successful epics (no non-epic issues), When `success_count > 0`, Then `session_end` validation DOES run.
  - Given a trigger without `failure_mode` field, When loading config, Then system fails with error: "failure_mode required for trigger X"
- **Edge cases:**
  - Back-to-back epic completions while periodic validations are queued → both queue sequentially, no deduplication
  - Abort signal during validation execution → in-progress validation interrupted, queued validations discarded
  - `interval` set to 1 → validation runs after every non-epic issue (valid but expensive)
  - Epic with no children completes (leaf epic) → still triggers epic_completion if configured
  - `epic_depth: 'top_level'` but all epics are nested → no epic_completion validations fire
  - Session ends with `success_count > 0` but `abort_run=True` → session_end skipped (abort takes precedence)
  - Periodic interval never reached (fewer issues than interval) → periodic validation never fires (valid)

**R3 — Command Selection via Command List**
- **Requirement:** The system MUST define a base command pool via `global_validation_commands` and allow each trigger to specify an ordered list of commands via `commands: [{ref: name, ...}, ...]`. Each command entry:
  - MUST have `ref` field (references base pool command name)
  - MAY override `command` string (uses custom command instead of base pool definition)
  - MAY override `timeout` (uses custom timeout instead of base pool definition)
  - **Override resolution (field-by-field):** Unoverridden fields inherit from base pool entry. If entry has `{ref: A, command: 'custom'}`, effective command uses `command='custom'` and `timeout` from base pool A (or system default 120s if base pool A has no timeout).
  - Same `ref` MAY appear multiple times in list (allows running same command with different args)
  Missing base command references MUST fail spec building at startup with error: "trigger X references unknown command Y. Available: [list]"
- **Verification:**
  - Given base pool with commands A and B, and trigger with `commands: [{ref: A}]`, When trigger fires, Then only A executes.
  - Given trigger with `commands: [{ref: A, timeout: 300}, {ref: A, command: 'pytest -m fast', timeout: 60}]`, When trigger fires, Then A executes twice in declaration order with different configurations.
  - Given base pool `{A: {command: 'pytest', timeout: 300}}` and trigger entry `{ref: A, command: 'pytest -m fast'}`, When trigger fires, Then effective command is `'pytest -m fast'` with timeout=300 (inherited from base pool).
  - Given base pool `{A: {command: 'pytest'}}` (no timeout) and trigger entry `{ref: A, timeout: 60}`, When trigger fires, Then effective command is `'pytest'` (inherited) with timeout=60 (override).
  - Given trigger with `commands: [{ref: X}]` where X not in base pool, When loading config, Then system fails with error: "epic_completion trigger references unknown command 'X'. Available: test, lint, typecheck"
  - Given trigger with `commands: []` (empty list), When trigger fires, Then no commands execute (validation skipped with result=passed, reason='no_commands')
- **Edge cases:**
  - Base pool is empty (`global_validation_commands: {}`) while triggers reference commands → startup error (no commands available)
  - Command entry has only `ref` (no overrides) → uses base pool definition verbatim
  - Command entry overrides only `timeout` → uses base pool `command` with custom timeout
  - Command entry overrides only `command` → uses base pool `timeout` (if defined) or system default (120s)
  - Trigger omits `commands` field entirely → treated as `commands: []` (no commands, validation skipped)

**R4 — Fail-Fast Sequential Execution**
- **Requirement:** The system MUST execute commands sequentially in declaration order. First command failure MUST immediately stop execution (fail-fast); remaining commands in that trigger do NOT execute. Triggers execute sequentially (no parallelization).
- **Verification:**
  - Given trigger with commands [A, B, C], When A passes, B passes, C passes, Then all three execute in order and trigger result=passed.
  - Given trigger with commands [A, B, C], When A passes, B fails, Then C does NOT execute and trigger result=failed at command B.
  - Given trigger with commands [A, B, C], When A fails, Then B and C do NOT execute and trigger result=failed at command A.
  - Given two triggers queued, When first trigger executes, Then second trigger waits (no parallel execution).
- **Edge cases:**
  - Command times out → treated as failure, triggers fail-fast behavior
  - Command crashes (vs exits non-zero) → treated as failure, triggers fail-fast
  - Zero commands in trigger (`commands: []`) → all "pass" (nothing to fail), trigger result=passed
  - Single command triggers → fail-fast has no effect (only one command)
  - Long-running first command blocks queue → acceptable, strict sequential execution
  - Validation queue grows during long-running validation → queue unbounded (but should emit events for observability)
  - Abort requested while validations queued → in-progress validation interrupted, queued validations discarded
  - Remediation retry loop → counts as single queue slot (all attempts execute before next trigger starts)

**R5 — Trigger Failure Modes and Terminal Behavior**
- **Requirement:** Any command failure MUST mark trigger as failed (due to fail-fast in R4). The trigger's `failure_mode` (REQUIRED field) controls run behavior:
  - `abort`: Immediately stop run after trigger fails. No more issues processed.
  - `continue`: Log failure, record result, continue run. Later triggers can still fire.
  - `remediate`: Spawn fixer agent (using `RunCoordinator._run_fixer_agent()`), retry trigger up to `max_retries` times. Remediation MUST block next issue execution until remediation completes (to prevent git conflicts/workspace races). If all retries exhausted, MUST abort run (terminal behavior).
  `max_retries` semantics: Counts fixer agent invocation attempts. `max_retries=0` means validation runs once with NO fixer (aborts immediately on failure). `max_retries=1` means validation + 1 fixer attempt if fails.
- **Verification:**
  - Given `failure_mode=continue`, When trigger fails, Then run continues to next issue and later triggers can fire.
  - Given `failure_mode=abort`, When trigger fails, Then run stops immediately and later triggers (including `session_end`) do NOT execute.
  - Given `failure_mode=remediate, max_retries=2`, When trigger fails and fixer succeeds on attempt 1, Then trigger re-runs and (if passed) run continues.
  - Given `failure_mode=remediate, max_retries=2`, When trigger fails and fixer fails on attempts 1 and 2, Then system aborts run (terminal behavior after exhaustion).
  - Given `failure_mode=remediate, max_retries=0`, When trigger fails, Then NO fixer spawns and run aborts immediately.
  - Given `failure_mode=remediate` active on epic_completion, When next issue queued, Then next issue execution MUST wait until remediation completes (blocks to prevent workspace conflicts).
  - Given trigger without `failure_mode`, When loading config, Then startup fails with error: "failure_mode required for trigger X"
- **Edge cases:**
  - Fixer succeeds but trigger re-run fails on different command → counts as failed attempt, continues retry loop
  - Multiple triggers fail with mixed failure_modes → each respects own mode (first abort stops run)
  - Fixer agent crashes/errors → counts as failed attempt
  - `max_retries` not specified → startup error: "max_retries required when failure_mode=remediate"
  - `max_retries` specified for `failure_mode=abort` or `continue` → ignored (no effect)
  - Remediation exhaustion during epic_completion → run aborts, session_end skipped

**R6 — Deprecation, Presets, and Migration**
- **Requirement:** The system MUST support the migration constraints:
  - `global_validation_commands` becomes a base command pool (not auto-run).
  - `validate_every` is deprecated in favor of a `periodic` trigger with `interval`.
  - Presets MUST define only the base pool (no trigger definitions), and triggers MUST be configured explicitly per project.
  - Base pool merge: Project commands override preset commands by name with **field-level deep merge** (project-specified fields override; unspecified fields inherit from preset).
- **Verification:**
  - Given preset with `test: {command: 'pytest', timeout: 300}` and project with `test: {timeout: 120}`, When loading config, Then effective pool has `test: {command: 'pytest', timeout: 120}` (field-level merge: command inherited from preset, timeout overridden by project).
  - Given preset with `test: {command: 'pytest'}` (no timeout) and project with `test: {timeout: 120}`, When loading config, Then effective pool has `test: {command: 'pytest', timeout: 120}` (both fields present in effective config).
  - Given preset with `test` and project with `lint`, When triggers fire, Then trigger can reference both `test` (from preset) and `lint` (from project) in merged effective pool.
  - Given a configuration using `validate_every: 5`, When loading config, Then startup fails with error: "validate_every is deprecated. Use validation_triggers.periodic with interval field. See migration guide at https://docs.mala.ai/migration/validation-triggers"
  - Given a configuration with only `global_validation_commands` (no `validation_triggers`), When loading config, Then startup fails with error including migration guide URL and concrete `validation_triggers.session_end` example.
- **Edge cases:**
  - Mixed configurations containing both deprecated and new fields → error message: "Cannot use both validate_every and validation_triggers.periodic. Remove validate_every."
  - Users expecting triggers to be inherited from presets (must not happen) → presets only provide base pool
  - Conflicting preset merges for base command names → field-level merge rules apply (project fields win)
  - Config has empty `global_validation_commands: {}` with `validation_triggers` → valid (triggers just won't have any commands to include)
  - Preset provides `global_validation_commands` but user config adds `validation_triggers` → valid (user triggers use preset's command pool)
  - Project overrides preset command with only `command` field (no timeout) → effective entry has project's command + preset's timeout (if defined) or system default (120s)

## 4. Instrumentation & Release Checks

**Instrumentation** *(M/L)*
- **Events to track** (via MalaEventSink protocol):
  - `on_trigger_validation_queued(trigger_type: str, trigger_context: str)` - When a trigger fires and queues validation (e.g., trigger_type="epic_completion", trigger_context="epic-123")
  - `on_trigger_validation_started(trigger_type: str, commands: list[str])` - When queued validation begins executing
  - `on_trigger_command_started(trigger_type: str, command_ref: str, index: int)` - When individual command within trigger starts
  - `on_trigger_command_completed(trigger_type: str, command_ref: str, index: int, passed: bool, duration_seconds: float)` - When individual command completes
  - `on_trigger_validation_passed(trigger_type: str, duration_seconds: float)` - When all commands in trigger pass
  - `on_trigger_validation_failed(trigger_type: str, failed_command: str, failure_mode: str)` - When any command fails
  - `on_trigger_remediation_started(trigger_type: str, attempt: int, max_retries: int)` - When fixer agent spawns for remediation
  - `on_trigger_remediation_succeeded(trigger_type: str, attempt: int)` - When remediation fixes the issue
  - `on_trigger_remediation_exhausted(trigger_type: str, attempts: int)` - When all retry attempts fail
  - Trigger lifecycle: fired/queued, started, completed (pass/fail), skipped (with reason)
  - Command lifecycle within a trigger: started (index), completed (passed, duration), allows per-command metrics

- **Metrics to derive:**
  - Time-to-feedback: time from `epic_completion` to validation result
  - Adoption: % of projects/runs with `validation_triggers` configured; breakdown by trigger type usage
  - Reliability: trigger failure rate and command failure rate by command and trigger type
  - Remediation effectiveness: % of failed triggers that pass within `max_retries`; average attempts to success
  - Cost proxy: total validation time per run and by trigger type (to validate "expensive at session end only" strategies)
  - Validation overhead: Time spent in validation vs. time spent in agent logic
  - Fix rate: % of remediations that successfully fixed the validation error

**Launch checklist** *(L)*
- [ ] All MUST requirements (R1-R6) are testable and verified with end-to-end scenarios for each trigger type
- [ ] Migration guidance is published (including `validate_every` → `periodic.interval`, and `global_validation_commands` repurpose)
- [ ] Clear failure messaging exists for missing base-command references and invalid trigger configs
- [ ] Telemetry confirms trigger firing, outcomes, and remediation attempts; dashboards exist for time-to-feedback and remediation success
- [ ] Rollback condition defined (e.g., disable trigger execution and revert to session-end-only behavior or disable validations entirely)
- [ ] Docs updated for project configuration and validation behavior (including "no implicit defaults" and "empty config runs nothing")
- [ ] Schema validation rejects invalid triggers or missing command references
- [ ] Verify `epic_completion` fires correctly for nested vs top-level epics
- [ ] Verify `periodic` trigger respects interval accurately
- [ ] Verify `remediate` mode correctly invokes the existing Fixer Agent and resumes on success
- [ ] Verify `abort` mode actually stops the run (no zombie processes)
- [ ] Key error states covered (missing command, invalid config, remediation exhausted)
- [ ] Metrics pipeline captures all instrumentation events
- [ ] CLI `--help` updated with new validation trigger options (if any)

**Decisions made** *(S/M/L)*

**Core Decisions:**
1. **Tier: L** - Full spec with constraints, alternate flows, edge cases (complexity score: 10+)
2. **Primary problem:** Faster validation feedback + flexible validation strategies (fast checks at epic completion, comprehensive checks at session end)
3. **Config structure:** Command list structure - base pool + per-trigger `commands: [{ref, command?, timeout?}, ...]` (supports multiple variants of same command)

**Breaking Changes:**
4. **BREAKING CHANGE #1:** Repurpose `global_validation_commands` - now defines base command pool (not auto-run). Configs with only `global_validation_commands` (no `validation_triggers`) fail with migration error.
5. **BREAKING CHANGE #2:** Replace `validate_every` with `periodic` trigger (has `interval` field)
6. **Migration requirement:** Existing mala.yaml files must add `validation_triggers` block to specify when commands run

**Trigger Configuration:**
7. **Supported triggers:** `epic_completion`, `session_end`, `periodic` (NOT `issue_completion` - redundant with per-session validation)
8. **Epic depth:** Configurable per-trigger via `epic_depth: 'top_level' | 'all'` field on epic_completion trigger
9. **Epic verification condition:** Configurable per-trigger via `fire_on: 'success' | 'failure' | 'both'` on epic_completion trigger
10. **Periodic interval:** Counts completed non-epic issues (NOT wall-clock). Epics do not increment counter.
11. **Session abort:** `session_end` trigger skipped if `abort_run=True` OR `success_count==0` (success_count includes both epics and non-epic issues)
11b. **Trigger execution blocking:** Per-session validation runs first, then global triggers queue. **Global validation execution BLOCKS next issue** to prevent workspace conflicts (validation operates on entire workspace). During multi-agent sessions, active agents continue but no new issue assignments until validation completes.

**Failure Handling:**
12. **Failure behavior:** Per-trigger `failure_mode: 'abort' | 'remediate' | 'continue'` (REQUIRED field, no default)
13. **Max retries:** Per-trigger `max_retries` (REQUIRED when failure_mode='remediate'; counts fixer attempts)
14. **Fail-fast within trigger:** First command failure stops execution (remaining commands NOT executed)
15. **Terminal behavior:** After remediation exhaustion, system MUST abort run (same as failure_mode='abort')
16. **Remediation mechanism:** Use existing `RunCoordinator._run_fixer_agent()` - spawns inline SDK agent, commits fixes, re-runs validation

**Execution:**
17. **Validation queuing:** Sequential execution only - queue validation runs, no parallel, no deduplication
18. **Command list structure:** Each trigger has `commands: [{ref, command?, timeout?}, ...]` allowing same ref multiple times with different configs
19. **Execution order:** Commands run in list declaration order
20. **Base command pool:** Defined in `global_validation_commands` (repurposed field)
21. **Missing command:** Startup error with clear message if trigger references non-existent command: "trigger X references unknown command Y. Available: [list]"
22. **Config validation:** ALL config errors fail at startup (fail-fast validation)

**Presets & Defaults:**
23. **Preset behavior:** Presets (python-uv, go, etc.) define only `global_validation_commands` (base pool), NOT `validation_triggers`. Users must configure triggers themselves.
24. **Empty config:** If mala.yaml has neither `global_validation_commands` nor `validation_triggers`, no validation runs (explicit opt-in)
25. **Unconfigured triggers:** Only explicitly configured triggers run. No implicit defaults (e.g., if `session_end` not configured, no validation at session end)

**Open questions** *(S/M/L)*

1. **Validation queue persistence:** If the process crashes mid-queue, should queued validations persist and resume? Or are they lost (simpler, matches current behavior)? Current decision: queue is in-memory only (lost on crash).

**Resolved (now in Constraints/Requirements):**
- ~~Default failure_mode~~ → RESOLVED: Required field, no default (R2, startup error if omitted)
- ~~Issue completion trigger~~ → RESOLVED: Removed (redundant with per-session validation)
- ~~Periodic interval units~~ → RESOLVED: Counts completed non-epic issues (R2)
- ~~Fail-fast within trigger~~ → RESOLVED: Fail-fast (R4, first failure stops execution)
- ~~Terminal behavior after remediation~~ → RESOLVED: Abort run (R5)
- ~~Zero retries semantics~~ → RESOLVED: max_retries=0 means no fixer, abort on failure (R5)
- ~~Empty command list~~ → RESOLVED: Trigger fires, no commands execute, result=passed (R3)
- ~~Config validation timing~~ → RESOLVED: Startup, fail-fast (Constraints)
- ~~Migration signal severity~~ → RESOLVED: Hard error at startup (R1)

## Appendix: Configuration Example

```yaml
# mala.yaml - Per-trigger validation configuration example

preset: python-uv  # Provides base global_validation_commands

# Base command pool (preset provides: test, lint, typecheck, format)
# Add custom commands to the pool
global_validation_commands:
  import-linter:
    command: "uv run lint-imports"
    timeout: 60
  security-scan:
    command: "uv run bandit -r src/"
    timeout: 120

# Per-trigger configuration (NEW SCHEMA: command list)
validation_triggers:
  epic_completion:
    epic_depth: top_level      # Only top-level epics, not nested
    fire_on: success           # Only when epic verification passes
    failure_mode: continue     # REQUIRED: log failures but keep going
    commands:
      - ref: typecheck
        timeout: 60            # Override timeout for fast feedback
      - ref: lint
      - ref: import-linter

  session_end:
    failure_mode: remediate    # REQUIRED: try to fix failures
    max_retries: 3             # REQUIRED for remediate mode
    commands:
      - ref: test
        command: "uv run pytest --cov --cov-report=html"
        timeout: 600           # Full test suite with coverage
      - ref: lint
      - ref: typecheck
      - ref: security-scan

  periodic:
    interval: 10               # Every 10 non-epic issues
    failure_mode: continue     # REQUIRED
    commands:
      - ref: lint
      - ref: typecheck

  # Uncomment to run critical security scan with abort on failure:
  # epic_completion:
  #   epic_depth: top_level
  #   fire_on: success
  #   failure_mode: abort      # REQUIRED: stop run on security issues
  #   commands:
  #     - ref: security-scan
```

## Appendix: Migration Guide Sketch

**Before (old config):**
```yaml
preset: python-uv
global_validation_commands:
  test:
    command: "uv run pytest"
  lint:
    command: "uvx ruff check ."
# validate_every: 5  # If using periodic
```

**After (new config):**
```yaml
preset: python-uv
# global_validation_commands now just defines command pool (preset handles it)

validation_triggers:
  session_end:
    failure_mode: remediate  # REQUIRED
    max_retries: 3           # REQUIRED for remediate mode
    commands:
      - ref: test
      - ref: lint
      - ref: typecheck
      - ref: format

  # If you had validate_every: 5, add:
  periodic:
    interval: 5              # Every 5 non-epic issues
    failure_mode: continue   # REQUIRED
    commands:
      - ref: lint
      - ref: typecheck
```

**Migration steps:**
1. If you have `validate_every: N`, convert to `validation_triggers.periodic` with `interval: N`
2. Add a `validation_triggers.session_end` block to specify which commands run at session end (previously all ran automatically)
3. Optionally add `epic_completion` or other triggers for faster feedback
4. Remove any `validate_every` field (deprecated)
