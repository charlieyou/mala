# CLI Reference

> For project-level validation configuration (`mala.yaml`), see [Project Configuration](project-config.md).

## Commands

### `mala run`

The main command to run the mala orchestrator. See [CLI Options](#cli-options) below.

### `mala init`

Generate a `mala.yaml` configuration file (interactive by default).

```bash
mala init              # Create mala.yaml interactively
mala init --dry-run    # Preview config without writing
mala init --preset python-uv --yes   # Non-interactive with defaults
mala init --preset python-uv --skip-evidence --skip-triggers  # Non-interactive, minimal
```

**Options:**

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview the generated config without writing to disk |
| `--yes`, `-y` | Accept defaults (requires `--preset`) |
| `--preset`, `-p` | Preset to use in non-interactive or interactive mode |
| `--skip-evidence` | Omit the `evidence_check` section |
| `--skip-triggers` | Omit the `validation_triggers` section |

**Workflow:**

1. Select a preset (python-uv, node-npm, go, rust) or choose custom
2. For custom: enter commands for built-in command names (setup, build, test, lint, format, typecheck, e2e)
3. Optionally configure `evidence_check` and `validation_triggers` (checkboxes)
4. Config is validated and written to `mala.yaml` (or printed for `--dry-run`)
5. If `mala.yaml` exists, a backup is created at `mala.yaml.bak`
6. YAML is printed to stdout (even in non-dry-run mode) and a trigger reference table is printed to stderr

**Non-interactive mode:** requires either `--preset --yes` or `--preset --skip-evidence --skip-triggers`.

### `mala status`

Show running mala instances, locks, and recent run metadata.

```bash
mala status
mala status --all
```

### `mala clean`

Clean up stale lock files.

```bash
mala clean
mala clean --force   # Clean even if a mala instance is running
```

### `mala logs`

Search and inspect run metadata.

```bash
mala logs list
mala logs list --all --json
mala logs sessions --issue ISSUE-123
mala logs sessions --issue ISSUE-123 --all
mala logs show <run_id_or_prefix>
```

### `mala epic-verify`

Verify (and optionally close) a single epic without running issues.

```bash
mala epic-verify EPIC-123 /path/to/repo
mala epic-verify EPIC-123 --no-close
mala epic-verify EPIC-123 --force
mala epic-verify EPIC-123 --human-override --close
```

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
| `--resume`, `-r` | false | Include in_progress issues and attempt to resume their Claude sessions |
| `--strict` | false | Fail if `--resume` finds no prior session for an issue (requires `--resume`) |
| `--fresh/--no-fresh` | false | Start new SDK session instead of resuming (requires `--resume`, conflicts with `--strict`) |

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

### Session Handling

When using `--resume`, you can control how sessions are handled:

```bash
# Resume existing sessions (default behavior)
mala run --resume /path/to/repo

# Start fresh sessions while keeping WIP scope and review feedback
mala run --resume --fresh /path/to/repo
```

The `--fresh` flag starts a new SDK session instead of resuming the previous one. This is useful when you want to clear context/token history while still including in-progress issues and their review feedback in scope.

### Watch Mode

| Flag | Default | Description |
|------|---------|-------------|
| `--watch` | false | Keep running and poll for new issues instead of exiting when idle |

### Debugging

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run`, `-d` | false | Preview task order without processing |
| `--verbose`, `-v` | false | Enable verbose output; shows full tool arguments |

### Claude Settings

| Flag | Default | Description |
|------|---------|-------------|
| `--claude-settings-sources` | `local,project` | Comma-separated list of settings sources: `local`, `project`, `user` |

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
| `MALA_DISABLE_DEBUG_LOG` | - | Set to `1` to disable debug file logging (for performance or disk space) |
| `CLAUDE_CONFIG_DIR` | `~/.claude` | Claude SDK config directory (plugins, sessions) |
| `MALA_CLAUDE_SETTINGS_SOURCES` | `local,project` | Comma-separated Claude settings sources |

### Epic Verification

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | - | API key for LLM calls (falls back to `ANTHROPIC_API_KEY`) |
| `LLM_BASE_URL` | - | Base URL for LLM API (for proxy/routing) |

Note: The repo's `.env` file is for testing only and is not loaded by the program.

### Deprecated Environment Variables

The following environment variables are deprecated and will be removed in a future release.
Configure these settings in `mala.yaml` instead:

| Deprecated Variable | Replacement in mala.yaml |
|---------------------|--------------------------|
| `MALA_REVIEW_TIMEOUT` | `validation_triggers.<trigger>.code_review.cerberus.timeout` |
| `MALA_CERBERUS_SPAWN_ARGS` | `validation_triggers.<trigger>.code_review.cerberus.spawn_args` |
| `MALA_CERBERUS_WAIT_ARGS` | `validation_triggers.<trigger>.code_review.cerberus.wait_args` |
| `MALA_CERBERUS_ENV` | `validation_triggers.<trigger>.code_review.cerberus.env` |
| `MALA_MAX_EPIC_VERIFICATION_RETRIES` | `validation_triggers.epic_completion.max_epic_verification_retries` |
| `MALA_MAX_DIFF_SIZE_KB` | `max_diff_size_kb` (root level) |
| `MALA_TRACK_REVIEW_ISSUES` | `track_review_issues` (root level) or `validation_triggers.<trigger>.code_review.track_review_issues` |

## Logs

Agent logs are written in JSONL format to `~/.config/mala/logs/`:

```
<session-uuid>.jsonl
```

Check log status with:
```bash
mala status     # Shows running instances, locks, and recent runs
mala logs list  # Lists recent runs with counts
mala logs show <run_id>  # Shows a specific run in detail
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

## Removed Flags

`--reviewer-type` is no longer supported. Configure reviewer type in
`validation_triggers.<trigger>.code_review.reviewer_type` in `mala.yaml`.

## Terminal Output

Agent output uses color-coded prefixes to distinguish concurrent agents:
- Each agent gets a unique bright color (cyan, yellow, magenta, green, blue, white)
- Log lines are prefixed with `[issue-id]` in the agent's color
- Tool usage, text output, and completion status are all color-coded
