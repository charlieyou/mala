# CLI Reference

> For project-level validation configuration (`mala.yaml`), see [Project Configuration](project-config.md).

## CLI Options

### Execution Limits

| Flag | Default | Description |
|------|---------|-------------|
| `--max-agents`, `-n` | unlimited | Maximum concurrent agents |
| `--timeout`, `-t` | 60 | Timeout per agent in minutes |
| `--max-issues`, `-i` | unlimited | Maximum total issues to process |

### Scope & Ordering

| Flag | Default | Description |
|------|---------|-------------|
| `--scope`, `-s` | `all` | Scope filter: `all`, `epic:<id>`, `ids:<id,...>`, `orphans` |
| `--order` | `epic-priority` | Issue ordering mode (see [Order Modes](#order-modes)) |
| `--resume`, `-r` | false | Include in_progress issues alongside open issues |

### Order Modes

The `--order` flag controls how issues are sorted and processed:

| Mode | Description |
|------|-------------|
| `focus` | **Single-epic mode**: Only process issues from one epic at a time. Picks the highest-priority epic and returns only its issues. Other epics are queued for later. |
| `epic-priority` | **Default**: Group issues by epic, then order groups by priority. All epics are processed, but issues from the same epic are kept together. |
| `issue-priority` | **Global priority**: Sort all issues by priority regardless of epic. Issues from different epics may be interleaved. |
| `input` | **Preserve order**: Keep issues in the order specified by `--scope ids:<id,...>`. Requires explicit ID list. |

**Examples:**

```bash
# Default: group by epic, process all epics
mala run /path/to/repo

# Focus on one epic at a time (strict single-epic)
mala run --order focus /path/to/repo

# Global priority ordering (ignore epic grouping)
mala run --order issue-priority /path/to/repo

# Process specific issues in exact order
mala run --scope ids:T-123,T-456,T-789 --order input /path/to/repo
```

### Quality Gates

| Flag | Default | Description |
|------|---------|-------------|
| `--max-gate-retries` | 3 | Maximum quality gate retry attempts per issue |
| `--max-review-retries` | 3 | Maximum external review retry attempts per issue |
| `--max-epic-verification-retries` | 3 | Maximum retries for epic verification loop |
| `--disable` | - | Validations to skip (see [Disable Flags](#disable-validation-flags)) |

### Review Backend (Cerberus)

These flags apply when using `reviewer_type: cerberus` in `validation_triggers.<trigger>.code_review`:

| Flag | Default | Description |
|------|---------|-------------|
| `--review-timeout` | 1200 | Timeout in seconds for Cerberus review operations |
| `--review-spawn-args` | - | Extra args appended to `review-gate spawn-code-review` |
| `--review-wait-args` | - | Extra args appended to `review-gate wait` |
| `--review-env` | - | Extra env for review-gate (JSON object or comma KEY=VALUE list) |

### Watch Mode

| Flag | Default | Description |
|------|---------|-------------|
| `--watch` | false | Keep running and poll for new issues instead of exiting when idle |
| `--validate-every` | 10 | Run validation after every N issues (watch mode only) |

### Debugging

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run`, `-d` | false | Preview task order without processing |
| `--verbose`, `-v` | false | Enable verbose output; shows full tool arguments |
| `--epic-override` | - | Repeatable epic IDs to close without verification (human bypass) |

### Disable Validation Flags

Use `--disable` with one or more values.
Repeat the flag or pass comma-separated lists (e.g., `--disable coverage --disable review` or `--disable coverage,review`):

| Value | Description |
|-------|-------------|
| `post-validate` | Skip all per-session validation (tests, lint, typecheck) |
| `global-validate` | Skip global validation |
| `integration-tests` | Exclude integration tests from pytest |
| `coverage` | Disable coverage checking |
| `e2e` | Disable E2E fixture repo test |
| `review` | Disable code review (configured via `validation_triggers.<trigger>.code_review`) |
| `followup-on-run-validate-fail` | Don't mark issues with `needs-followup` on global validation failure |

## Global Configuration

mala uses a global config directory at `~/.config/mala/`:

```
~/.config/mala/
├── .env          # API keys (ANTHROPIC_API_KEY) and config overrides
├── logs/         # JSONL session logs
└── runs/         # Run metadata (repo-segmented directories)
    └── -home-user-repo/
```

Environment variables are loaded from `~/.config/mala/.env` (global config).
Precedence: CLI flags override global config, which overrides program defaults.

### Directory Overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `MALA_RUNS_DIR` | `~/.config/mala/runs` | Base directory for run metadata (per-repo subdirs) |
| `MALA_LOCK_DIR` | `/tmp/mala-locks` | Directory for filesystem locks |
| `MALA_REVIEW_TIMEOUT` | `1200` | Review-gate wait timeout in seconds |
| `MALA_DISABLE_DEBUG_LOG` | - | Set to `1` to disable debug file logging (for performance or disk space) |
| `CLAUDE_CONFIG_DIR` | `~/.claude` | Claude SDK config directory (plugins, sessions) |

### Epic Verification

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | - | API key for LLM calls (falls back to `ANTHROPIC_API_KEY`) |
| `LLM_BASE_URL` | - | Base URL for LLM API (for proxy/routing) |
| `MALA_MAX_EPIC_VERIFICATION_RETRIES` | `3` | Number of retries after first verification attempt fails |

### Cerberus Review Overrides

When using `reviewer_type: cerberus` in `validation_triggers.<trigger>.code_review`:

| Variable | Description |
|----------|-------------|
| `MALA_CERBERUS_SPAWN_ARGS` | Extra args for `review-gate spawn-code-review` |
| `MALA_CERBERUS_WAIT_ARGS` | Extra args for `review-gate wait` |
| `MALA_CERBERUS_ENV` | Extra env for review-gate (JSON object or comma KEY=VALUE list) |

Note: The repo's `.env` file is for testing only and is not loaded by the program.

## Logs

Agent logs are written in JSONL format to `~/.config/mala/logs/`:

```
<session-uuid>.jsonl
```

Check log status with:
```bash
mala status    # Shows recent logs and their timestamps
```

### Output Verbosity

mala supports two output modes controlled by `--verbose`:

| Mode | Flag | Description |
|------|------|-------------|
| **Normal** | (default) | Single line per tool call |
| **Verbose** | `--verbose` / `-v` | Full tool arguments in key=value format |

```bash
mala run /path/to/repo          # Normal output (default)
mala run -v /path/to/repo       # Verbose mode - full tool args
```

## Terminal Output

Agent output uses color-coded prefixes to distinguish concurrent agents:
- Each agent gets a unique bright color (cyan, yellow, magenta, green, blue, white)
- Log lines are prefixed with `[issue-id]` in the agent's color
- Tool usage, text output, and completion status are all color-coded
