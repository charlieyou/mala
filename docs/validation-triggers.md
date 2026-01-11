# Validation Triggers

Validation triggers let you run validation commands at specific checkpoints during a mala session. Instead of running all validations only at the end, you can configure fast checks after each issue or epic and comprehensive checks at run end.

## Overview

The trigger system provides:

- **Faster feedback**: Run appropriate validations at each checkpoint (e.g., lint after each epic)
- **Cost-effective validation**: Run expensive tests only when needed (e.g., E2E tests at run end)
- **Flexible strategies**: Different failure modes per trigger (abort, continue, or auto-fix)

## Configuration

Triggers are configured in `mala.yaml` using two sections:

1. **Base command pool**: Built from `commands` (preset or explicit), including custom commands
2. **Trigger configuration** (`validation_triggers`): Specifies when and how to run them

### Base Pool Construction

The base command pool determines which commands can be referenced in triggers:

- **Built-in commands** (`setup`, `build`, `test`, `lint`, `format`, `typecheck`, `e2e`): Use `commands` (from preset or explicit config)
- **Custom commands**: Any additional keys defined under `commands`

```yaml
preset: python-uv  # Provides test, lint, typecheck, format in commands

# When to run which commands
validation_triggers:
  epic_completion:
    epic_depth: top_level
    fire_on: success
    failure_mode: continue
    commands:
      - ref: lint
      - ref: typecheck

  session_end:
    failure_mode: remediate
    max_retries: 3
    commands:
      - ref: test
      - ref: lint
      - ref: typecheck

  periodic:
    interval: 10
    failure_mode: continue
    commands:
      - ref: lint
```

## Trigger Types

### epic_completion

Fires when an epic (story/milestone) completes verification.

| Field | Required | Values | Description |
|-------|----------|--------|-------------|
| `epic_depth` | Yes | `top_level`, `all` | Which epics trigger validation |
| `fire_on` | Yes | `success`, `failure`, `both` | When to fire based on verification result |
| `failure_mode` | Yes | `abort`, `continue`, `remediate` | How to handle validation failures |
| `max_retries` | When remediate | Integer | Retry attempts for remediation |
| `max_epic_verification_retries` | No | Integer | Maximum retries for epic verification loop (default: 3) |
| `epic_verify_lock_timeout_seconds` | No | Integer | Timeout in seconds for acquiring epic verification lock (default: 300) |
| `commands` | No | List | Commands to run (empty = no validation) |

**epic_depth values:**
- `top_level`: Only epics with no epic parent (root-level epics)
- `all`: All epics including nested ones

**fire_on values:**
- `success`: Fire only when epic verification passes
- `failure`: Fire only when epic verification fails
- `both`: Fire regardless of verification result

```yaml
validation_triggers:
  epic_completion:
    epic_depth: top_level
    fire_on: success
    failure_mode: continue
    commands:
      - ref: typecheck
        timeout: 60
      - ref: lint
```

### session_end

Fires after each **agent session** completes (per-issue), after the quality gate passes.

Commands run in the repository root via the orchestrator (not inside the agent session).

| Field | Required | Values | Description |
|-------|----------|--------|-------------|
| `failure_mode` | Yes | `abort`, `continue`, `remediate` | How to handle validation failures |
| `max_retries` | When remediate | Integer | Retry attempts for remediation |
| `commands` | No | List | Commands to run (empty = no validation) |

**Notes:**
- Runs after each issue's gate pass (not after the whole run)
- Session_end failures do **not** block the external review stage (unless `failure_mode: abort`)

```yaml
validation_triggers:
  session_end:
    failure_mode: remediate
    max_retries: 3
    commands:
      - ref: test
        command: "uv run pytest --cov --cov-report=html"
        timeout: 600
      - ref: lint
      - ref: security-scan
```

### periodic

Fires after a specified number of non-epic issues complete.

| Field | Required | Values | Description |
|-------|----------|--------|-------------|
| `interval` | Yes | Integer | Number of issues between validations |
| `failure_mode` | Yes | `abort`, `continue`, `remediate` | How to handle validation failures |
| `max_retries` | When remediate | Integer | Retry attempts for remediation |
| `commands` | No | List | Commands to run (empty = no validation) |

The counter increments only for non-epic issues (epic completions don't count). Validations fire at `interval`, `2*interval`, `3*interval`, etc.

```yaml
validation_triggers:
  periodic:
    interval: 5
    failure_mode: continue
    commands:
      - ref: lint
      - ref: typecheck
```

### run_end

Fires once at the end of the mala run, after all issue work completes (including any per-issue `session_end` validations).

| Field | Required | Values | Description |
|-------|----------|--------|-------------|
| `fire_on` | No | `success`, `failure`, `both` | When to fire (default: `success`) |
| `failure_mode` | Yes | `abort`, `continue`, `remediate` | How to handle validation failures |
| `max_retries` | When remediate | Integer | Retry attempts for remediation |
| `commands` | No | List | Commands to run (empty = no validation) |

**fire_on values:**
- `success`: Fire only when all issues succeeded
- `failure`: Fire only when any issue failed
- `both`: Fire regardless of issue results

```yaml
validation_triggers:
  run_end:
    fire_on: success
    failure_mode: continue
    commands:
      - ref: test
      - ref: lint
```

## Failure Modes

Each trigger must specify a `failure_mode`:

| Mode | Behavior |
|------|----------|
| `abort` | Stop the run immediately on validation failure |
| `continue` | Log the failure and continue processing issues |
| `remediate` | Spawn a fixer agent to attempt repairs, retry up to `max_retries` times |

### Remediation

When `failure_mode: remediate` is set:

1. Validation runs
2. If it fails, a fixer agent spawns with the failure output
3. Fixer attempts to commit fixes
4. Validation re-runs
5. Steps 2-4 repeat up to `max_retries` times
6. If still failing after all retries, the run aborts

`max_retries` counts fixer attempts:
- `max_retries: 0` - No fixer, abort immediately on failure
- `max_retries: 1` - One fixer attempt after initial failure
- `max_retries: 3` - Three fixer attempts

```yaml
validation_triggers:
  session_end:
    failure_mode: remediate
    max_retries: 3
    commands:
      - ref: test
      - ref: lint
```

## Command List Structure

Each trigger has a `commands` list that references the base command pool.

### Command Entry Fields

| Field | Required | Description |
|-------|----------|-------------|
| `ref` | Yes | Name of command in base pool |
| `command` | No | Override the command string |
| `timeout` | No | Override the timeout in seconds |

### String Shorthand

For simple references without overrides:

```yaml
commands:
  - lint
  - typecheck
```

### Full Object Form

For overrides:

```yaml
commands:
  - ref: test
    command: "uv run pytest --cov"
    timeout: 600
  - ref: lint
    timeout: 120
```

### Override Resolution

When overriding, unspecified fields inherit from the base pool. The base pool is built from `commands` (preset or explicit):

```yaml
preset: python-uv  # Provides commands.test = "uv run pytest"

# Override the preset's test command
commands:
  test:
    command: "uv run pytest --cov"
    timeout: 300

# Trigger with per-trigger override
validation_triggers:
  session_end:
    failure_mode: continue
    commands:
      - ref: test
        timeout: 600  # Overrides timeout, inherits command from commands.test
```

Result: `command="uv run pytest --cov"`, `timeout=600`

If `commands.test` were not defined, the trigger would use the preset's `commands.test` command.

### Running Same Command Multiple Times

The same `ref` can appear multiple times with different configurations:

```yaml
commands:
  - ref: test
    command: "uv run pytest -m fast"
    timeout: 60
  - ref: test
    command: "uv run pytest -m slow"
    timeout: 600
```

### Empty Commands

An empty or omitted `commands` list means no validation runs, but the trigger still fires:

```yaml
validation_triggers:
  epic_completion:
    epic_depth: all
    fire_on: both
    failure_mode: continue
    commands: []  # No validation, just marks the checkpoint
```

## Execution Behavior

### Sequential Execution

Commands within a trigger execute sequentially in declaration order. If a command fails, remaining commands are skipped (fail-fast).

### Trigger Queuing

Multiple triggers queue and execute sequentially. No parallel validation runs.

### Blocking

Trigger validation **blocks new issue assignments** to prevent workspace conflicts. Active agents continue running, but no new issues start until validation completes.

## Code Review

Each trigger can optionally include a `code_review` block to run automated code reviews after validation commands pass.

### Configuration

| Field | Required | Values | Default | Description |
|-------|----------|--------|---------|-------------|
| `enabled` | No | Boolean | `false` | Whether to run code review |
| `reviewer_type` | No | `cerberus`, `agent_sdk` | `cerberus` | Which reviewer to use |
| `failure_mode` | No | `abort`, `continue`, `remediate` | `continue` | How to handle review failures |
| `max_retries` | No | Integer | `3` | Retry attempts for remediation |
| `finding_threshold` | No | `P0`, `P1`, `P2`, `P3`, `none` | `none` | Minimum severity to report |
| `baseline` | Yes for `epic_completion`, `run_end` | `since_run_start`, `since_last_review` | - | What code to include |
| `cerberus` | No | Object | - | Cerberus-specific settings |
| `agent_sdk_timeout` | No | Integer | `600` | Timeout in seconds for Agent SDK reviews |
| `agent_sdk_model` | No | `sonnet`, `opus`, `haiku` | `sonnet` | Model for Agent SDK reviewer |
| `track_review_issues` | No | Boolean | `true` | Create beads issues for P2/P3 review findings |

**baseline requirement:** The `baseline` field is required for `epic_completion` and `run_end` triggers because they review accumulated changes across multiple issues. If omitted, mala logs a warning and defaults to `since_run_start`. It is not used for `session_end`, which reviews only the single issue's changes.

### Per-Issue Code Review (session_end)

Reviews code changes from a single completed issue (no `baseline` needed):

```yaml
validation_triggers:
  session_end:
    failure_mode: continue
    commands:
      - ref: lint
      - ref: test
    code_review:
      enabled: true
      reviewer_type: cerberus
      failure_mode: continue
      finding_threshold: P1
```

### Cumulative Code Review (epic_completion, run_end)

Reviews accumulated changes across multiple issues. The `baseline` field is required (if omitted, mala defaults to `since_run_start` with a warning):

- `since_run_start`: Review all changes since the run began
- `since_last_review`: Review changes since the last successful review at this trigger point

```yaml
validation_triggers:
  epic_completion:
    epic_depth: top_level
    fire_on: success
    failure_mode: continue
    commands:
      - ref: lint
    code_review:
      enabled: true
      reviewer_type: cerberus
      failure_mode: continue
      baseline: since_run_start
      finding_threshold: P1

  run_end:
    fire_on: success
    failure_mode: continue
    commands:
      - ref: test
    code_review:
      enabled: true
      baseline: since_last_review
      finding_threshold: P0
```

### Cerberus-Specific Settings

When using `reviewer_type: cerberus`, additional settings are available:

```yaml
code_review:
  enabled: true
  reviewer_type: cerberus
  cerberus:
    timeout: 300
    spawn_args: ["--verbose"]
    wait_args: []
    env: [["API_KEY", "xxx"]]
```

| Field | Description |
|-------|-------------|
| `timeout` | Review timeout in seconds (default: 300) |
| `spawn_args` | Additional arguments when spawning reviewer |
| `wait_args` | Additional arguments when waiting for results |
| `env` | Environment variables as key-value pairs |

## Migration Guide

### From Root-Level `reviewer_type` (Removed)

The root-level `reviewer_type`, `agent_sdk_review_timeout`, and `agent_sdk_reviewer_model` fields have been removed. Review configuration is now done via `validation_triggers.<trigger>.code_review`:

**Old (no longer supported):**
```yaml
preset: python-uv
reviewer_type: cerberus  # Error: Unknown field
```

**New:**
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

### From `MALA_TRACK_REVIEW_ISSUES` Environment Variable

The `MALA_TRACK_REVIEW_ISSUES` environment variable is deprecated. Use `track_review_issues` in the `code_review` block instead:

**Before (deprecated):**
```bash
export MALA_TRACK_REVIEW_ISSUES=false
mala run
```

**After:**
```yaml
validation_triggers:
  session_end:
    code_review:
      enabled: true
      track_review_issues: false
```

The env var remains supported for backwards compatibility but `code_review.track_review_issues` takes precedence when configured.

### From `validate_every`

The `validate_every` field is not supported. Replace with `periodic` trigger:

**Before:**
```yaml
preset: python-uv
validate_every: 5
```

**After:**
```yaml
preset: python-uv
validation_triggers:
  periodic:
    interval: 5
    failure_mode: continue
    commands:
      - ref: lint
      - ref: typecheck
```

### Opting Out of Validation

To explicitly disable all validation triggers (no validation at any checkpoint):

```yaml
preset: python-uv
validation_triggers: {}  # Empty triggers block
```

## Example Configurations

### Fast Feedback at Epic Completion

Run quick checks after each epic, comprehensive tests at session end:

```yaml
preset: python-uv

validation_triggers:
  epic_completion:
    epic_depth: top_level
    fire_on: success
    failure_mode: continue
    commands:
      - ref: typecheck
        timeout: 60
      - ref: lint
        timeout: 60

  session_end:
    failure_mode: remediate
    max_retries: 3
    commands:
      - ref: test
        timeout: 600
      - ref: lint
      - ref: typecheck
```

### Periodic Lint Checks

Keep code clean during long sessions:

```yaml
preset: python-uv

validation_triggers:
  periodic:
    interval: 5
    failure_mode: continue
    commands:
      - ref: lint
      - ref: typecheck

  session_end:
    failure_mode: remediate
    max_retries: 2
    commands:
      - ref: test
```

### Strict Security Checks

Abort on any security issues:

```yaml
preset: python-uv

commands:
  security-scan:
    command: "uv run bandit -r src/"
    timeout: 120

validation_triggers:
  epic_completion:
    epic_depth: top_level
    fire_on: success
    failure_mode: abort  # Stop immediately on security issues
    commands:
      - ref: security-scan

  session_end:
    failure_mode: remediate
    max_retries: 3
    commands:
      - ref: test
      - ref: lint
      - ref: security-scan
```

### Custom Commands Only

Define and use only custom commands:

```yaml
commands:
  lint: "uvx ruff check ."
  test: "uv run pytest"
  import-linter:
    command: "uvx --from import-linter lint-imports"
    timeout: 60
  mypy:
    command: "uv run mypy src/"
    timeout: 120

validation_triggers:
  session_end:
    failure_mode: remediate
    max_retries: 2
    commands:
      - ref: test
      - ref: lint
      - ref: import-linter
      - ref: mypy
```

## Troubleshooting

### Error: `validate_every is not supported`

```
validate_every is not supported. Use validation_triggers.periodic with interval field.
See migration guide at https://docs.mala.ai/migration/validation-triggers
```

**Cause:** Using the old `validate_every` field.

**Solution:** Replace with `validation_triggers.periodic`:

```yaml
validation_triggers:
  periodic:
    interval: 5
    failure_mode: continue
    commands:
      - ref: lint
```

### Error: `failure_mode required`

```
failure_mode required for trigger epic_completion
```

**Cause:** Missing required `failure_mode` field.

**Solution:** Add `failure_mode: abort`, `continue`, or `remediate`.

### Error: `max_retries required when failure_mode=remediate`

```
max_retries required when failure_mode=remediate for trigger session_end
```

**Cause:** Using `failure_mode: remediate` without specifying `max_retries`.

**Solution:** Add `max_retries` with an integer value.

### Error: `trigger X references unknown command`

```
epic_completion trigger references unknown command 'typo_test'. Available: test, lint, typecheck
```

**Cause:** A command `ref` doesn't match any command in the base pool.

**Solution:** Check the command name matches one in `commands`.

### Error: `epic_depth required` / `fire_on required`

```
epic_depth required for trigger epic_completion
```

**Cause:** Missing required field for `epic_completion` trigger.

**Solution:** Add both `epic_depth` and `fire_on` fields.

### Error: `interval required`

```
interval required for trigger periodic
```

**Cause:** Missing `interval` field for `periodic` trigger.

**Solution:** Add `interval` with an integer value (number of issues between validations).
