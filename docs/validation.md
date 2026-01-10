# Validation System

The validation module (`src/domain/validation/`) provides structured validation with policy-based configuration.

## Quality Gate

After an agent completes an issue, the orchestrator runs a quality gate that verifies:

1. **Commit exists**: One or more git commits with `bd-<issue_id>` in the message, created during the current run (stale commits from previous runs are rejected via baseline timestamp)
2. **Validation evidence**: The agent ran ALL required checks defined in `mala.yaml` (parsed from JSONL logs). For Python projects using `preset: python-uv`:
   - `pytest` - tests
   - `ruff check` - linting
   - `ruff format` - formatting
   - `ty check` - type checking
3. **Commands succeeded**: All validation commands must exit with zero status (non-zero exits fail the gate)

All validation commands must run AND pass for the gate to pass. Partial validation (e.g., only tests) is rejected. The required commands are spec-driven via `ValidationSpec` built from `mala.yaml`. Multiple commits per issue are supported; reviews use the full commit list for the issue when available.

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

## Code Review

Code review runs after the deterministic gate passes. Configure it in `validation_triggers.<trigger>.code_review`. For complete configuration details, see [validation-triggers.md](validation-triggers.md#code-review).

### Code Review Configuration

The `code_review` block configures automated code review for each trigger:

| Field | Required | Values | Default | Description |
|-------|----------|--------|---------|-------------|
| `enabled` | No | Boolean | `false` | Whether to run code review |
| `reviewer_type` | No | `cerberus`, `agent_sdk` | `cerberus` | Which reviewer to use |
| `failure_mode` | No | `abort`, `continue`, `remediate` | `continue` | How to handle review failures |
| `max_retries` | No | Integer | `3` | Retry attempts for remediation |
| `finding_threshold` | No | `P0`, `P1`, `P2`, `P3`, `none` | `none` | Minimum severity to report |
| `baseline` | Required for cumulative | `since_run_start`, `since_last_review` | - | What code to include |
| `cerberus` | No | Object | - | Cerberus-specific settings |

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

### Review Flow

1. **Review spawns**: The configured reviewer (`cerberus` or `agent_sdk`) reviews issue commits
2. **Scope verification**: Reviewers check commits against issue description and acceptance criteria
3. **Consensus**: All available reviewers must unanimously pass
4. **Review failure handling**: If any reviewer finds errors, orchestrator resumes the SAME session with:
   - List of issues (file, line, priority, message) from all reviewers
   - Instructions to fix errors and re-run validations
   - Commit list for the issue (includes all work across retry attempts)
5. **Re-gating**: After fixes, runs both deterministic gate AND code review again

Review retries are capped at `max_retries` in the `code_review` block (default: 3). Use `--disable review` to disable all code review globally.

**Skipped for no-work resolutions**: Issues resolved with `ISSUE_NO_CHANGE`, `ISSUE_OBSOLETE`, or `ISSUE_ALREADY_COMPLETE` skip code review entirely since there's no new code to review.

### Low-Priority Review Findings (P2/P3)

When code review passes but includes P2/P3 priority findings, the orchestrator automatically creates tracking issues:

1. **Collection**: P2/P3 findings are collected from the review result (P0/P1 block the review)
2. **Issue creation**: After the issue is successfully closed, beads issues are created for each finding
3. **Issue format**: Each tracking issue includes:
   - Title: `[Review] {finding title}`
   - File and line references
   - Original finding description
   - Link to the source issue

This ensures low-priority review findings are tracked and not forgotten, without blocking the current issue from completing.

### Migration from Legacy Config

Root-level `reviewer_type`, `agent_sdk_review_timeout`, and `agent_sdk_reviewer_model` fields are deprecated. Move review configuration into `validation_triggers.<trigger>.code_review`:

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

## Global Validation

After all issues complete, the orchestrator runs a final validation pass. This catches issues that only manifest when all changes are combined:

1. **Triggers**: After all per-session work completes (with at least one success)
2. **Worktree validation**: Runs tests in isolated worktree at HEAD commit
3. **Fixer agent**: On failure, spawns a dedicated fixer agent with the failure output
4. **Retry loop**: Continues up to `max_gate_retries` attempts

Global validations can override the base `commands` via `global_validation_commands` in `mala.yaml`. Any omitted fields inherit from `commands`.

### Validation Scopes

| Scope | When | What runs |
|-------|------|-----------|
| **Per-session** | After each issue completes | pytest, ruff, ty |
| **Global** | After all issues complete | pytest, ruff, ty, + E2E fixture test |

### Test Flags

| Flag | Default | Effect |
|------|---------|--------|
| `integration-tests` | included | Pytest tests marked `@pytest.mark.integration` are skipped when this flag is in the disable list |
| `e2e` | enabled (global only) | E2E fixture test runs only during global validation |

### Disable Flags

- `--disable global-validate`: Skip global validation entirely
- `--disable integration-tests`: Exclude integration-marked pytest tests
- `--disable e2e`: Disable E2E fixture test (only affects global)
- `--disable followup-on-run-validate-fail`: Don't mark issues on global validation failure

## Repo Type Detection

mala automatically detects the repository type and adjusts validation accordingly:

| Repo Type | Detection | Validation |
|-----------|-----------|------------|
| **Python** | Has `pyproject.toml`, `uv.lock`, or `requirements.txt` | Full Python toolchain (pytest, ruff, ty) |
| **Generic** | No Python project markers | Minimal validation (no Python-specific tools) |

This allows mala to process issues in non-Python repositories without failing on missing Python tooling.

## ValidationSpec

Defines what validations run per scope (per-session vs global):

```python
from src.domain.validation import build_validation_spec, ValidationScope

spec = build_validation_spec(
    scope=ValidationScope.PER_SESSION,
    disable_validations={"integration-tests"},  # Optional disable flags
)
```

## Code vs Docs Classification

Changes are classified to determine validation requirements:

| Category | Paths/Files | Validation |
|----------|-------------|------------|
| **Code** | `src/**`, `tests/**`, `commands/**`, `.py`, `.sh`, `.toml` | Full suite (tests + coverage) |
| **Docs** | `.md`, `.rst`, `.txt` outside code paths | Full suite (tests still run) |

Note: For docs-only issues that need no changes, agents can use `ISSUE_NO_CHANGE` to skip validation entirely.

## Worktree Validation

Clean-room validation runs in isolated git worktrees:

```
/tmp/mala-worktrees/{run_id}/{issue_id}/{attempt}/
```

- Commits are tested in isolation from the working tree
- Failed validations can keep worktrees for debugging (`--keep-worktrees`)
- Stale worktrees from crashed runs are auto-cleaned

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
