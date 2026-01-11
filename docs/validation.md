# Validation System

The validation module (`src/domain/validation/`) provides structured validation with policy-based configuration.

## Quality Gate

After an agent completes an issue, the orchestrator runs a quality gate that verifies:

1. **Commit exists**: One or more git commits with `bd-<issue_id>` in the message, created during the current run (stale commits from previous runs are rejected via a baseline timestamp)
2. **Evidence (optional)**: If `evidence_check.required` is configured, the gate requires evidence in JSONL session logs for those command names
3. **Command success**: Required evidence must show successful command execution (failures fail the gate; custom commands with `allow_fail: true` do not fail the gate)

Evidence requirements are **opt-in**. If `evidence_check` is omitted or `required: []`, the gate does **not** require validation evidence (it still requires a commit).
Required evidence names must exist in the resolved command pool (built-in or custom); invalid names fail fast at startup.

### Resolution Markers

The gate handles special resolutions:

- **`ISSUE_NO_CHANGE` / `ISSUE_OBSOLETE`**: Requires a rationale; commit and evidence checks are skipped
- **`ISSUE_ALREADY_COMPLETE`**: Requires a rationale and a matching commit; baseline timestamp is ignored; evidence checks are skipped
- **`ISSUE_DOCS_ONLY`**: Requires a rationale and a commit; evidence checks are skipped only if the commit does **not** touch files that trigger validation (based on `code_patterns`, `config_files`, `setup_files`) and does not modify `mala.yaml`

### Same-Session Re-entry

If the gate fails, the orchestrator **resumes the same Claude session** with a follow-up prompt containing:
- The specific failure reasons (missing commit, missing validations)
- Instructions to fix and re-run validations
- The current attempt number (e.g., "Attempt 2/3")

This continues for up to `max_gate_retries` attempts (default: 3). The orchestrator tracks:
- **Log offset**: Only evidence from the current attempt is considered
- **Previous commit hash**: Detects "no progress" when commit is unchanged
- **No-progress detection**: Stops retries early if agent makes no meaningful changes

### Idle Timeout Retry

When a Claude CLI subprocess hangs (no output for an extended period), the orchestrator automatically recovers:

1. **Detection**: If no SDK message arrives within the idle timeout (derived from agent timeout, clamped to 5-15 minutes), an idle timeout is triggered
2. **Disconnect**: The orchestrator calls `disconnect()` to cleanly terminate the hung subprocess
3. **Resume strategy**:
   - If a session ID exists: Resume the same session with a prompt explaining the timeout
   - If no session ID but no tool calls yet: Retry fresh (no side effects to lose)
   - If tool calls occurred without session context: Fail immediately (potential data loss)
4. **Retry limits**: Up to `max_idle_retries` (default: 2) attempts with exponential backoff

This prevents hung agents from blocking issue processing indefinitely.

## Session-End Validation (Trigger)

After the gate passes, mala can run a **session_end** trigger (if configured) to execute additional commands and/or code review for the completed issue. These commands run in the repository root via the orchestrator, not inside the agent session.

- **failure_mode** controls what happens on failure (`abort`, `continue`, `remediate`)
- **remediate** spawns a fixer agent and retries the session_end commands
- Results are recorded and passed to the external review as **informational context**

## External Review Gate (Per Issue)

After session_end (or immediately after the gate if session_end is not configured), an external review gate runs for the issue:

- Reviewer implementation is selected from the **first enabled** `validation_triggers.<trigger>.code_review` block:
  - `reviewer_type: agent_sdk` (default if none configured)
  - `reviewer_type: cerberus` (requires Cerberus review-gate)
- If the configured reviewer is unavailable (e.g., Cerberus not installed), review is disabled for that run
- P0/P1 findings fail the review; P2/P3 findings are tracked (see below)
- Reviews are retried on parse errors or no-progress conditions, up to the configured retry limit

## Trigger-Based Code Review

Each trigger can optionally include a `code_review` block. These reviews run **after trigger commands** and use the same reviewer types as the external review gate. For full configuration details, see [validation-triggers.md](validation-triggers.md#code-review).

### Code Review Configuration

The `code_review` block configures automated code review for each trigger:
The first enabled block also determines the reviewer implementation and settings
used by the per-issue external review gate.

| Field | Required | Values | Default | Description |
|-------|----------|--------|---------|-------------|
| `enabled` | No | Boolean | `false` | Whether to run code review |
| `reviewer_type` | No | `cerberus`, `agent_sdk` | `cerberus` | Which reviewer to use |
| `failure_mode` | No | `abort`, `continue`, `remediate` | `continue` | How to handle review failures |
| `max_retries` | No | Integer | `3` | Retry attempts for remediation |
| `finding_threshold` | No | `P0`, `P1`, `P2`, `P3`, `none` | `none` | Minimum severity to report |
| `baseline` | Required for cumulative* | `since_run_start`, `since_last_review` | - | What code to include |
| `cerberus` | No | Object | - | Cerberus-specific settings |

*If omitted for `epic_completion` or `run_end`, mala logs a warning and defaults to `since_run_start`.

### Trigger Types and Code Review

| Trigger | When | Baseline | Use Case |
|---------|------|----------|----------|
| `session_end` | After each agent session | Not applicable (per-issue) | Review individual issue commits |
| `epic_completion` | When epic completes | Required | Cumulative review of epic work |
| `run_end` | After all issues complete | Required | Final cumulative review |

**Example configuration:**

```yaml
validation_triggers:
  session_end:
    failure_mode: continue
    code_review:
      enabled: true
      reviewer_type: cerberus
      finding_threshold: P1

  run_end:
    fire_on: success
    failure_mode: continue
    code_review:
      enabled: true
      baseline: since_last_review
      finding_threshold: P0
```

### Review Flow (Per-Issue Review Gate)

1. **Review spawns**: The configured reviewer (`cerberus` or `agent_sdk`) reviews issue commits
2. **Scope verification**: Reviewers check commits against issue description and acceptance criteria
3. **Consensus**: All available reviewers must unanimously pass
4. **Review failure handling**: If any reviewer finds errors, orchestrator resumes the SAME session with:
   - List of issues (file, line, priority, message) from all reviewers
   - Instructions to fix errors and re-run validations
   - Commit list for the issue (includes all work across retry attempts)
5. **Re-gating**: After fixes, runs both the gate and review again

Trigger-based code_review uses fixer remediation (when configured) rather than resuming the original implementer session.

Trigger code_review remediation retries are capped by `code_review.max_retries` (default: 3). Per-issue external review retries are capped by an internal default (currently 3) and are not configured in `mala.yaml`.

**Skipped for no-work resolutions**: Issues resolved with `ISSUE_NO_CHANGE`, `ISSUE_OBSOLETE`, `ISSUE_ALREADY_COMPLETE`, or `ISSUE_DOCS_ONLY` skip code review entirely since there's no new code to review.

### Low-Priority Review Findings (P2/P3)

When code review passes but includes P2/P3 findings, mala can automatically create tracking issues:

1. **Collection**: P2/P3 findings are collected from the review result (P0/P1 block the review)
2. **Issue creation**: After the issue is successfully closed, beads issues are created for each finding
3. **Issue format**: Each tracking issue includes:
   - Title: `[Review] {finding title}`
   - File and line references
   - Original finding description
   - Link to the source issue

This ensures low-priority review findings are tracked and not forgotten, without blocking the current issue from completing.

### Migration from Legacy Config

Root-level `reviewer_type`, `agent_sdk_review_timeout`, and `agent_sdk_reviewer_model` fields are not supported. Move review configuration into `validation_triggers.<trigger>.code_review`:

**Before:**
```yaml
preset: python-uv
reviewer_type: cerberus
```

**After:**
```yaml
preset: python-uv
validation_triggers:
  session_end:
    failure_mode: remediate
    code_review:
      enabled: true
      reviewer_type: cerberus
      finding_threshold: P1
```

For complete migration instructions, see [validation-triggers.md](validation-triggers.md#migration-guide).

## Trigger Validation

Validation triggers (`session_end`, `periodic`, `epic_completion`, `run_end`) execute configured commands from `mala.yaml`:

- Commands run **sequentially** and **fail-fast**
- Validation runs **block new issue assignments** while they execute
- `failure_mode: remediate` spawns a fixer agent and retries the trigger commands
- `run_end` fires based on `fire_on` (success/failure/both)

Trigger commands execute in the repository root (no worktree isolation).

## ValidationSpec

Defines validation commands and evidence expectations per scope. If `mala.yaml` is missing, mala returns a default Python/uv spec with **no evidence requirements**.

```python
from src.domain.validation import build_validation_spec, ValidationScope

spec = build_validation_spec(
    scope=ValidationScope.PER_SESSION,
    disable_validations={"integration-tests"},  # Optional disable flags
)
```

## Code vs Docs Classification

Changes are classified to determine whether `ISSUE_DOCS_ONLY` is allowed:

| Category | Paths/Files | Validation |
|----------|-------------|------------|
| **Code** | Matches `code_patterns`, `config_files`, or `setup_files` (plus `mala.yaml`) | Docs-only resolution **rejected** |
| **Docs** | Does not match validation trigger patterns | Docs-only resolution allowed |

Note: For documentation-only commits, use `ISSUE_DOCS_ONLY` to skip evidence checks when files do not match validation trigger patterns.

## Worktree Validation (Programmatic)

The `SpecValidationRunner` uses isolated git worktrees when run directly in code
or tests. The orchestrator's trigger validation runs in the repository root.

## Parallel Validation

Agents run validation commands in parallel using **isolated cache directories** to prevent conflicts. For `preset: python-uv`:

```bash
pytest -o cache_dir=/tmp/pytest-$AGENT_ID        # Isolated pytest cache
ruff check . --cache-dir=/tmp/ruff-$AGENT_ID     # Isolated ruff cache (via RUFF_CACHE_DIR)
ruff format .                                     # No cache conflicts
ty check                                          # Type check (read-only)
uv sync                                           # Has internal locking
```

This approach avoids deadlocks that occurred when agents held file locks while waiting for a global test mutex. File locks prevent concurrent edits; isolated caches prevent validation conflicts.

## Failure Handling

After all retries are exhausted (gate or review), the orchestrator:
- Marks the issue with `needs-followup` label
- Records error summary and log path in issue notes
- Does NOT close the issue (leaves it for manual intervention)

When an agent fails (including quality gate failures after all retries), the orchestrator records context in the beads issue notes:
- Error summary (gate failures, review issues, timeout, etc.)
- Path to the JSONL session log (in `~/.config/mala/logs/`)
- Attempt counts (gate attempts, review attempts)

The next agent (or human) can read the issue notes with `bd show <issue_id>` and grep the log file for context.
