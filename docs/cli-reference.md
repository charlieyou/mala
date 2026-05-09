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
| `--timeout`, `-t` | 30 | Timeout per agent in minutes |
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

### Coder Selection

Mala can drive its per-issue implementation agent on Claude, Sourcegraph's
Amp, or OpenAI's Codex (`codex app-server`). The choice is global to a run
— every issue in a single run uses the same coder, and fixer agents follow
the main coder.

| Flag | Default | Description |
|------|---------|-------------|
| `--coder` | `claude` | Coder backend: `claude`, `amp`, or `codex`. Validated at parse time. |
| `--amp-mode` | `deep` | Amp execution mode: `smart`, `rush`, or `deep`. Only consulted when `coder=amp`. |
| `--codex-model` | `gpt-5.5` | Codex model identifier (e.g., `gpt-5.5`). Only consulted when `coder=codex`. |
| `--codex-effort` | `medium` | Codex reasoning effort. Validated against the SDK's `ReasoningEffort` enum at parse time. Only consulted when `coder=codex`. |
| `--codex-approval-policy` | `never` | Codex approval policy: `never`, `on-request`, `on-failure`, `untrusted`. Only consulted when `coder=codex`. |
| `--codex-sandbox` | `danger-full-access` | Codex sandbox mode: `read-only`, `workspace-write`, `danger-full-access`. Only consulted when `coder=codex`. |

#### Precedence

All coder flags follow the same **CLI > env > yaml > default** precedence as
`claude_settings_sources`:

| Setting | CLI | Env | YAML | Default |
|---------|-----|-----|------|---------|
| Coder | `--coder claude` | `MALA_CODER=claude` | `coder: claude` | `claude` |
| Amp mode | `--amp-mode rush` | `MALA_AMP_MODE=rush` | `amp_mode: rush` | `deep` |
| Codex model | `--codex-model gpt-5.5` | `MALA_CODEX_MODEL=gpt-5.5` | `coder_options.codex.model: gpt-5.5` | `gpt-5.5` |
| Codex effort | `--codex-effort high` | `MALA_CODEX_EFFORT=high` | `coder_options.codex.effort: high` | `medium` |
| Codex approval policy | `--codex-approval-policy never` | `MALA_CODEX_APPROVAL_POLICY=never` | `coder_options.codex.approval_policy: never` | `never` |
| Codex sandbox | `--codex-sandbox danger-full-access` | `MALA_CODEX_SANDBOX=danger-full-access` | `coder_options.codex.sandbox: danger-full-access` | `danger-full-access` |

Invalid values fail validation **before** any agent process starts.

#### Cross-Coder Flag Behavior

The CLI never errors when a coder-specific flag is passed against a different
coder — flags are logged as ignored at info-level so switching `--coder` does
not require pruning unrelated flags from your invocation:

| Flag/setting | `coder=claude` | `coder=amp` | `coder=codex` |
|--------------|----------------|-------------|---------------|
| `--claude-settings-sources` / `MALA_CLAUDE_SETTINGS_SOURCES` | applied | logged as ignored (info) | logged as ignored (info) |
| `--amp-mode` / `MALA_AMP_MODE` / `amp_mode` | logged as ignored (info) | applied | logged as ignored (info) |
| `--codex-*` / `MALA_CODEX_*` / `coder_options.codex.*` | logged as ignored (info) | logged as ignored (info) | applied |
| `MALA_DISALLOWED_TOOLS` | applied (Claude hooks) | **no-op**, warned once at run start (MVP limitation) | applied (Codex `PreToolUse` hook) |

**Examples:**

```bash
# Run with Amp (GPT-5 reasoning via deep mode)
mala run --coder amp /path/to/repo

# Run with Amp in rush mode (Haiku) for cheaper iteration
mala run --coder amp --amp-mode rush /path/to/repo

# Run with Codex (gpt-5.5)
mala run --coder codex /path/to/repo

# Run with Codex at high reasoning effort
mala run --coder codex --codex-model gpt-5.5 --codex-effort high /path/to/repo

# Same via env (CI-friendly)
MALA_CODER=amp MALA_AMP_MODE=deep mala run /path/to/repo
MALA_CODER=codex MALA_CODEX_MODEL=gpt-5.5 mala run /path/to/repo
```

**Amp prerequisites:** binary install (npm install is unsupported), Bun runtime
via the Amp binary, writable `~/.config/amp/plugins/`. See the
[Amp prerequisites in README](../README.md#amp-optional-for-coder-amp).

**Codex prerequisites:** `openai-codex-app-server-sdk` (Python SDK, install via
`uv sync --extra codex` or `uv add openai-codex-app-server-sdk`),
`openai-codex-cli-bin` runtime (pulled in by the SDK), Codex auth configured
locally, and writable `~/.codex/plugins/` for the bundled `mala-safety`
plugin. See the [Codex prerequisites in README](../README.md#codex-optional-for-coder-codex)
for install steps and [Codex Prerequisites](#codex-prerequisites) below for
the auto-trust mechanism, the interactive-trust fallback, and per-error
remediation. Missing SDK / runtime / auth raises `CodexNotInstalledError`;
a missing or untrusted bundled hook raises `CodexHookNotActiveError` — both
fail closed before any issue agent runs.

## Codex Prerequisites

This section documents the runtime and config preconditions for
`--coder codex`, the auto-trust mechanism mala uses to enable Codex's bundled
`mala-safety` plugin, the documented one-time interactive-trust fallback when
auto-trust is unavailable, and remediation steps for each fail-closed
`CodexHookNotActiveError` / `CodexNotInstalledError` reason. The error class's
`docs_url` attribute points at this section.

### Auto-trust (default)

When `mala run --coder codex` starts, `install_prerequisites()` writes the
following entries to `$CODEX_HOME/config.toml` (default `~/.codex/config.toml`)
so Codex loads and trusts the bundled `mala-safety` plugin without an
interactive prompt:

- `[features]` with `plugins = true`, `plugin_hooks = true`, and `hooks = true`
  (the three feature gates that must all be on for plugin-bundled hooks to be
  discovered, registered, and executed).
- `[plugins."mala-safety@local"]` with `enabled = true` (so Codex's
  `configured_plugins_from_stack` enumerates the plugin).
- `[hooks.state."mala-safety@local:.codex-plugin/hooks.json:pre_tool_use:0:0"]`
  with `enabled = true` and a computed `trusted_hash` (so Codex marks the
  `PreToolUse` handler Trusted without prompting).
- The same `[hooks.state."..."]` block for the `session_start` event the
  selftest probe drives.

The writes are idempotent — reruns against an already-trusted hook short-circuit
without rewriting the file. Other keys inside `[features]` and unrelated tables
in `config.toml` are preserved.

The runtime self-test verifies both trusted handlers: `SessionStart` must fire
when the Codex turn starts, and `PreToolUse` must fire before a real tool call.
Both handlers write event-specific marker files, and mala only proceeds when
both markers contain the expected hook version hash.

### Interactive trust fallback (one-time)

Auto-trust requires `$CODEX_HOME` to be writable. On systems where `~/.codex/`
or `~/.codex/config.toml` is read-only (NFS, immutable filesystems, sandboxed
container roots, restrictive ACLs), `install_prerequisites()` fails closed
with `CodexHookNotActiveError(TRUSTED_HASH_MISMATCH)` rather than silently
running Codex without the safety hook. To recover, do **one** of:

1. **Make `$CODEX_HOME` writable** (preferred): adjust permissions so
   `chmod u+w ~/.codex ~/.codex/config.toml` succeeds, then rerun mala.
   Auto-trust then takes over and the run proceeds unattended.
2. **Trust the hook once interactively.** Run `codex` directly in the same
   shell environment (with the same `CODEX_HOME`) and accept the trust
   prompt for the `mala-safety` plugin's `PreToolUse` and `SessionStart`
   handlers. Codex persists the trust state into `config.toml`; subsequent
   `mala run --coder codex` invocations short-circuit auto-trust because
   the entries already match.
3. **Pre-write the trust entries by hand.** Add the auto-trust blocks
   listed in [Auto-trust (default)](#auto-trust-default) to
   `$CODEX_HOME/config.toml`. The easiest way to obtain the exact
   `trusted_hash` value is to run `mala run --coder codex` once against a
   writable sandbox `CODEX_HOME` (e.g., `CODEX_HOME=$(mktemp -d) mala run …`)
   and copy the value from the resulting `config.toml`; the hash is a
   pure function of the bundled hook's `hooks.json` payload, so it is
   stable across machines for a given mala version.

The fallback is one-time per `mala-safety` version: a mala upgrade that
changes the hook's normalized identity recomputes `trusted_hash`, so a
read-only `CODEX_HOME` requires repeating step 2 or step 3 after the upgrade.

### Remediation by error reason

`mala run --coder codex` aborts before any issue agent spawns if the
preconditions are not met. The structured `reason` field on
`CodexHookNotActiveError` (and the message body of `CodexNotInstalledError`)
maps to one of the rows below.

| Reason | What it means | How to recover |
|--------|---------------|----------------|
| `CodexNotInstalledError` (SDK/auth) | The `codex_app_server` Python SDK is not importable, or no Codex credential is detectable (no `OPENAI_API_KEY` / `CODEX_API_KEY` / `CODEX_ACCESS_TOKEN`, no `auth.json`, no keyring config). | Install the SDK extra: `uv sync --extra codex`. Then run `codex login` (Sign in with ChatGPT) or set `OPENAI_API_KEY`. See [OpenAI Codex auth docs](https://developers.openai.com/codex/auth). |
| `CODEX_BINARY_MISSING` | No `codex` binary on `PATH`, no `CODEX_BINARY` override, and the bundled `codex_cli_bin` runtime package is not importable. | Reinstall the runtime extra: `uv sync --extra codex` (pulls in `openai-codex-cli-bin`). Verify with `uv run python -c "import codex_cli_bin"`. As an escape hatch, set `CODEX_BINARY=/path/to/codex` to an explicit binary on disk. |
| `SCRIPT_MISSING` | The `mala-codex-pre-tool-use` console script is not on `PATH`, or one of the hook's dependency modules cannot be located on disk (the per-module identity hash uses on-disk source bytes). | Reinstall mala so `[project.scripts]` is wired up: `uv tool install mala-agent` (CLI install) or `uv sync` (development install). Verify with `which mala-codex-pre-tool-use`. If you installed mala into a venv, ensure that venv is active when `mala run --coder codex` starts. |
| `PLUGIN_DISABLED` | Codex's `plugin/list` does not surface `mala-safety@local`, reports it as not installed, or reports it as not enabled — or the upstream `marketplace_load_errors` field was non-empty during the live selftest probe. | Confirm `~/.codex/plugins/` is writable. Delete the cached plugin tree at `$CODEX_HOME/plugins/cache/local/mala-safety/` and rerun mala; the installer is idempotent and recreates the cache. If `marketplace_load_errors` cites a parse failure, inspect `$CODEX_HOME/config.toml` for hand-edits that broke the `[plugins."mala-safety@local"]` table or the `[features]` block. |
| `TRUSTED_HASH_MISMATCH` | Mala could not create / read / write `$CODEX_HOME/config.toml`, the trust entry is missing on read, or the on-disk `trusted_hash` no longer matches the bundled hook's identity (typical after a mala upgrade against a read-only `CODEX_HOME`). | Make `$CODEX_HOME` writable and rerun, or use the [interactive trust fallback](#interactive-trust-fallback-one-time) above. Do **not** delete `config.toml` if it has unrelated user settings — mala merges into existing tables, so a manual rewrite that drops user keys is a regression. |
| `HOOK_MARKER_MISSING` / `VERSION_MISMATCH` | The selftest probe ran the hook but did not see the expected per-event marker, or the marker's version hash did not match the running provider's expected hash. Indicates the installed hook script and the provider drifted. | Reinstall mala so the bundled hook script and provider stay in lockstep: `uv sync` (development) or `uv tool upgrade mala-agent` (CLI install). Reruns are idempotent. If the mismatch persists after a clean reinstall, file a bug — the hook's identity hash is pinned to its module bytes, so a real mismatch indicates source drift. |

After resolving the underlying cause, rerun `mala run --coder codex`; the
selftest re-runs every invocation and short-circuits cleanly when all
preconditions hold.

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
| `MALA_VALIDATION_LOG_DIR` | `/tmp/mala-validation-logs` | Base directory for implementer validation command logs |
| `MALA_DISABLE_DEBUG_LOG` | - | Set to `1` to disable debug file logging (for performance or disk space) |
| `CLAUDE_CONFIG_DIR` | `~/.claude` | Claude SDK config directory (plugins, sessions) |
| `MALA_CLAUDE_SETTINGS_SOURCES` | `local,project` | Comma-separated Claude settings sources |
| `MALA_CODER` | `claude` | Coder backend: `claude`, `amp`, or `codex`. Overridden by `--coder`; falls back to `coder:` in `mala.yaml`. |
| `MALA_AMP_MODE` | `deep` | Amp execution mode: `smart`, `rush`, or `deep`. Overridden by `--amp-mode`; falls back to `amp_mode` in `mala.yaml`. Only consulted when coder is `amp`. |
| `MALA_CODEX_MODEL` | `gpt-5.5` | Codex model id. Overridden by `--codex-model`; falls back to `coder_options.codex.model`. Only consulted when coder is `codex`. |
| `MALA_CODEX_EFFORT` | `medium` | Codex reasoning effort (validated against the SDK's `ReasoningEffort` enum). Overridden by `--codex-effort`; falls back to `coder_options.codex.effort`. Only consulted when coder is `codex`. |
| `MALA_CODEX_APPROVAL_POLICY` | `never` | Codex approval policy: `never`, `on-request`, `on-failure`, `untrusted`. Overridden by `--codex-approval-policy`; falls back to `coder_options.codex.approval_policy`. Only consulted when coder is `codex`. |
| `MALA_CODEX_SANDBOX` | `danger-full-access` | Codex sandbox mode: `read-only`, `workspace-write`, `danger-full-access`. Overridden by `--codex-sandbox`; falls back to `coder_options.codex.sandbox`. Only consulted when coder is `codex`. |

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
