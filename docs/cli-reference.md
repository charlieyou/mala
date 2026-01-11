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

### Watch Mode

| Flag | Default | Description |
|------|---------|-------------|
| `--watch` | false | Keep running and poll for new issues instead of exiting when idle |

### Debugging

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run`, `-d` | false | Preview task order without processing |
| `--verbose`, `-v` | false | Enable verbose output; shows full tool arguments |

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

## Deprecated Flags

These flags are deprecated and will be removed in a future version. Use the recommended alternatives.

### `--reviewer-type`

**Status**: Deprecated

**Migration**: Use `validation_triggers.<trigger>.code_review.reviewer_type` in `mala.yaml` instead.

**Before:**
```bash
mala run --reviewer-type cerberus /path/to/repo
```

**After:**
```yaml
# mala.yaml
validation_triggers:
  session_end:
    failure_mode: continue
    code_review:
      enabled: true
      reviewer_type: cerberus  # or "agent_sdk"
```

The per-trigger configuration provides more flexibility, allowing different reviewer types for different triggers (e.g., `agent_sdk` for per-issue review, `cerberus` for cumulative run-end review).

## Terminal Output

Agent output uses color-coded prefixes to distinguish concurrent agents:
- Each agent gets a unique bright color (cyan, yellow, magenta, green, blue, white)
- Log lines are prefixed with `[issue-id]` in the agent's color
- Tool usage, text output, and completion status are all color-coded
