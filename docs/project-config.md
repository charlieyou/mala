# Project Configuration

> For CLI flags and runtime settings, see [CLI Reference](cli-reference.md).

By default, Mala reads project configuration from `mala.yaml` in the root of
the target repository. The file defines your project's validation commands,
code patterns, coverage settings, coder backend, reviewers, and validation
triggers.

You can keep multiple config files for different workflows and select one at
runtime with `--config`:

```bash
mala run --config mala.codex.yaml .
mala epic-verify --config mala.strict.yaml EPIC-123 .
```

When `--config` is omitted, Mala preserves the default `<repo>/mala.yaml`
lookup. When `--config PATH` is provided, `PATH` may have any filename and
relative paths are resolved from the current working directory, not from the
`repo_path` argument. `mala init` still writes `mala.yaml`.

## Quick Start

Use a preset for common stacks:

```yaml
# Python project
preset: python-uv
```

```yaml
# Node.js project
preset: node-npm
```

```yaml
# Go project
preset: go
```

```yaml
# Rust project
preset: rust
```

## Configuration Schema

```yaml
# mala.yaml

preset: string           # Optional. Preset to extend (python-uv, node-npm, go, rust)

commands:                   # Optional. Command definitions
  setup: string | object | null     # Environment setup (e.g., "uv sync", "npm install")
  build: string | object | null     # Build step (e.g., "npm run build")
  test: string | object | null      # Test runner (e.g., "uv run pytest", "go test ./...")
  lint: string | object | null      # Linter (e.g., "uvx ruff check .", "golangci-lint run")
  format: string | object | null    # Formatter command (e.g., "uvx ruff format .")
  typecheck: string | object | null # Type checker (e.g., "uvx ty check", "tsc --noEmit")
  e2e: string | object | null       # End-to-end tests (e.g., "uv run pytest -m e2e")
  <custom>: string | object         # Custom commands (e.g., "security-scan")

code_patterns:           # Optional. Glob patterns for code files
  - "*.py"
  - "src/**/*.ts"

config_files:            # Optional. Tool config files (invalidate lint/format cache)
  - "pyproject.toml"
  - ".eslintrc*"

setup_files:             # Optional. Dependency files (invalidate setup cache)
  - "uv.lock"
  - "package-lock.json"

coverage:                # Optional. Omit to disable coverage
  command: string        # Optional. Command to generate coverage (uses test if omitted)
  format: string         # Required. Format: "xml" (Cobertura)
  file: string           # Required. Path to coverage report
  threshold: number      # Required. Minimum coverage percentage (0-100)

evidence_check:          # Optional. Gate evidence requirements
  required: list         # Command names that must have evidence in session logs

claude_settings_sources: list     # Optional. SDK settings sources (default: [local, project])

coder: string                    # Optional. Coder backend: "claude", "amp", or "codex" (default: claude)
amp_mode: string                 # Optional. Amp mode: "smart", "rush", "deep" (default: deep). Only consulted when coder: amp.
model: string                    # Optional. Model for Claude/Codex (defaults: claude=opus[1m], codex=gpt-5.5). Ignored by Amp.
effort: string | null            # Optional. Reasoning effort for Claude/Codex/supported Amp modes.

coder_options:                   # Optional. Per-coder option blocks.
  codex:                         # Only consulted when coder: codex.
    approval_policy: string      # never | on-request | on-failure | untrusted (default: "never")
    sandbox: string              # read-only | workspace-write | danger-full-access (default: "danger-full-access")
    mcp_servers: object          # Extra MCP servers; merged with bundled mala-locking (default: {})

timeout_minutes: int             # Optional. Agent timeout in minutes (default: 30)

max_idle_retries: int            # Optional. Maximum idle timeout retries (default: 2)
idle_timeout_seconds: float      # Optional. Idle timeout in seconds (default: derived from timeout_minutes)

max_diff_size_kb: int            # Optional. Maximum diff size in KB for epic verification

epic_verification:               # Optional. Epic verification backend selection
  enabled: bool                  # Enable/disable epic verification (default: true)
  reviewer_type: string          # "cerberus" or "agent_sdk" (default: agent_sdk)
  timeout: int                   # Top-level timeout in seconds (default: 600)
  max_retries: int               # Maximum retry attempts (default: 3)
  failure_mode: string           # continue | abort | remediate (default: continue)
  cerberus: object               # Cerberus-specific settings
  agent_sdk_timeout: int         # Agent SDK timeout in seconds (default: 300)
  agent_sdk_model: string        # sonnet | opus | haiku (default: opus)
  retry_policy: object           # Per-category retry limits

per_issue_review:                # Optional. Per-issue code review (disabled by default)
  enabled: bool                  # Enable/disable per-issue review (default: false)
  reviewer_type: string          # "cerberus" or "agent_sdk" (default: cerberus)
  max_retries: int               # Maximum review retry attempts (default: 3)
  finding_threshold: string      # P0 | P1 | P2 | P3 | none (default: none)
  track_review_issues: bool      # Create beads issues for findings (default: true)
  failure_mode: string           # continue | abort | remediate (default: continue)
  cerberus: object               # Cerberus-specific settings
  agent_sdk_timeout: int         # Agent SDK timeout in seconds (default: 300)
  agent_sdk_model: string        # sonnet | opus | haiku (default: opus)

validation_triggers:             # Optional. See validation-triggers.md
  epic_completion: object        # Run validation when epics complete
  session_end: object            # Run validation at session end
  periodic: object               # Run validation periodically
  run_end: object                # Run validation at end of run
```

## Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `preset` | string | No | Preset name to extend |
| `commands` | object | No | Map of command kind to shell command or object |
| `commands.<kind>` | string, object, or null | No | Shell command or `{command, timeout}`. `null` explicitly disables |
| `commands.build` | string, object, or null | No | Build step (optional) |
| `commands.<custom>` | string or object | No | Custom command definition (name must be a valid identifier) |
| `code_patterns` | list | No | Glob patterns for code files |
| `config_files` | list | No | Tool config files that trigger re-lint |
| `setup_files` | list | No | Lock files that trigger setup re-run |
| `coverage` | object | No | Coverage configuration |
| `coverage.command` | string | No | Coverage-enabled test command |
| `coverage.format` | string | Yes* | Format: `xml` |
| `coverage.file` | string | Yes* | Path to coverage report |
| `coverage.threshold` | number | Yes* | Minimum coverage % |
| `claude_settings_sources` | list | No | SDK settings sources: `local`, `project`, `user` (default: `[local, project]`) |
| `coder` | string | No | Coder backend: `claude`, `amp`, or `codex` (default: `claude`). Overridden by `--coder` / `MALA_CODER`. See [Coder Selection](#coder-selection). |
| `amp_mode` | string | No | Amp execution mode: `smart`, `rush`, or `deep` (default: `deep`). Only consulted when `coder: amp`. |
| `model` | string | No | Model for Claude and Codex coders. Defaults: `claude=opus[1m]`, `codex=gpt-5.5`. Overridden by `--model` / `MALA_MODEL`. Ignored by Amp. |
| `effort` | string | No | Reasoning effort for Claude, Codex, and supported Amp modes. Overridden by `--effort` / `MALA_EFFORT`. |
| `coder_options` | object | No | Per-coder option block. See [Codex Coder Options](#codex-coder-options). Only consulted when the matching coder is selected. |
| `coder_options.codex.approval_policy` | string | No | Codex approval policy: `never`, `on-request`, `on-failure`, `untrusted` (default: `never`). Overridden by `--codex-approval-policy` / `MALA_CODEX_APPROVAL_POLICY`. |
| `coder_options.codex.sandbox` | string | No | Codex sandbox mode: `read-only`, `workspace-write`, `danger-full-access` (default: `danger-full-access`). Overridden by `--codex-sandbox` / `MALA_CODEX_SANDBOX`. |
| `coder_options.codex.mcp_servers` | object | No | Extra MCP servers (merged with the bundled `mala-locking` server; the bundled server cannot be replaced). YAML-only. Default: `{}`. |
| `timeout_minutes` | integer | No | Agent timeout in minutes (default: 30). Can be overridden by CLI `--timeout` |
| `max_idle_retries` | integer | No | Maximum idle timeout retries before aborting (default: 2). Set to 0 to disable retries. |
| `idle_timeout_seconds` | float | No | Seconds to wait for SDK activity before triggering idle recovery (default: derived from `timeout_minutes`). Set to 0 to disable idle timeout. |
| `max_diff_size_kb` | integer | No | Maximum diff size in KB for epic verification. Diffs larger than this limit will skip verification. |
| `epic_verification` | object | No | Epic verification backend selection and retries |
| `epic_verification.enabled` | boolean | No | Enable epic verification (default: `true`) |
| `epic_verification.reviewer_type` | string | No | Reviewer type: `cerberus` or `agent_sdk` (default: `agent_sdk`) |
| `epic_verification.timeout` | integer | No | Top-level timeout in seconds (default: 600) |
| `epic_verification.max_retries` | integer | No | Maximum retry attempts (default: 3) |
| `epic_verification.failure_mode` | string | No | Failure handling: `continue`, `abort`, `remediate` (default: `continue`) |
| `epic_verification.cerberus` | object | No | Cerberus-specific settings (see below) |
| `epic_verification.cerberus.timeout` | integer | No | Timeout in seconds (default: 600) |
| `epic_verification.cerberus.spawn_args` | list | No | Additional arguments for spawn command |
| `epic_verification.cerberus.wait_args` | list | No | Additional arguments for wait command |
| `epic_verification.cerberus.env` | object | No | Environment variables as key-value pairs |
| `epic_verification.agent_sdk_timeout` | integer | No | Agent SDK timeout in seconds (default: 300) |
| `epic_verification.agent_sdk_model` | string | No | Agent SDK model: `sonnet`, `opus`, `haiku` (default: `opus`) |
| `epic_verification.retry_policy` | object | No | Per-category retry limits |
| `epic_verification.retry_policy.timeout_retries` | integer | No | Retry limit for timeouts (default: 3) |
| `epic_verification.retry_policy.execution_retries` | integer | No | Retry limit for execution errors (default: 2) |
| `epic_verification.retry_policy.parse_retries` | integer | No | Retry limit for parse errors (default: 1) |
| `validation_triggers` | object | No | Trigger configuration. See [validation-triggers.md](validation-triggers.md) |
| `evidence_check` | object | No | Evidence requirements for the quality gate |
| `evidence_check.required` | list | No | List of command names that must appear in session logs |
| `per_issue_review` | object | No | Per-issue code review configuration |
| `per_issue_review.enabled` | boolean | No | Enable per-issue review (default: `false`) |
| `per_issue_review.reviewer_type` | string | No | Reviewer type: `cerberus` or `agent_sdk` (default: `cerberus`) |
| `per_issue_review.max_retries` | integer | No | Maximum review retry attempts (default: 3) |
| `per_issue_review.finding_threshold` | string | No | Minimum severity to fail: `P0`, `P1`, `P2`, `P3`, `none` (default: `none`) |
| `per_issue_review.track_review_issues` | boolean | No | Create beads issues for P2/P3 findings (default: `true`) |
| `per_issue_review.failure_mode` | string | No | Failure handling: `continue`, `abort`, `remediate` (default: `continue`) |
| `per_issue_review.cerberus` | object | No | Cerberus-specific settings (see below) |
| `per_issue_review.cerberus.timeout` | integer | No | Timeout in seconds (default: 600) |
| `per_issue_review.cerberus.spawn_args` | list | No | Additional arguments for spawn command |
| `per_issue_review.cerberus.wait_args` | list | No | Additional arguments for wait command |
| `per_issue_review.cerberus.env` | object | No | Environment variables as key-value pairs |
| `per_issue_review.agent_sdk_timeout` | integer | No | Agent SDK timeout in seconds (default: 300) |
| `per_issue_review.agent_sdk_model` | string | No | Agent SDK model: `sonnet`, `opus`, `haiku` (default: `opus`) |

*Required when `coverage` section is present.

## Context Management

The Claude Agent SDK handles context management automatically via prompt caching and auto-compaction. No manual configuration is required.

## Command Object Form

Each command can be defined as a string or an object with an optional timeout:

```yaml
commands:
  test:
    command: "uv run pytest"
    timeout: 600
```

## Custom Commands

Define additional validation commands under `commands` using custom names:

```yaml
commands:
  security-scan:
    command: "uv run bandit -r src/"
    timeout: 120
    allow_fail: true
```

Notes:
- Custom command names must match `[A-Za-z_][A-Za-z0-9_-]*`
- `allow_fail: true` means a failed custom command does **not** fail the evidence gate
- Custom commands default to a 120s timeout when not specified
  - `allow_fail` does **not** bypass trigger command failures (triggers always fail-fast)

## Evidence Check

`evidence_check.required` controls which commands must appear (and pass) in agent session logs.
If omitted or an empty list, the quality gate does **not** require validation evidence.
Required commands can be built-ins or custom commands; both use the same canonical wrapper and emit `MALA_EVIDENCE name=<name> exit=<code> log=<path>`.

```yaml
commands:
  security-scan:
    command: "uv run bandit -r src/"
    timeout: 120
    allow_fail: true

evidence_check:
  required: [test, lint, security-scan]
```

Built-in commands now use the same evidence wrapper as custom commands. The `name=` value in the `MALA_EVIDENCE` line is the command key from `evidence_check.required`, so `test`, `lint`, and `security-scan` are reported uniformly.

## Built-in Presets

### python-uv

```yaml
commands:
  setup: "uv sync"
  test: "uv run pytest -o cache_dir=/tmp/pytest-${AGENT_ID:-default}"
  lint: "RUFF_CACHE_DIR=/tmp/ruff-${AGENT_ID:-default} uvx ruff check ."
  format: "RUFF_CACHE_DIR=/tmp/ruff-${AGENT_ID:-default} uvx ruff format ."
  typecheck: "uvx ty check"
  e2e: "uv run pytest -m e2e -o cache_dir=/tmp/pytest-${AGENT_ID:-default}"

code_patterns:
  - "**/*.py"
  - "pyproject.toml"

config_files:
  - "pyproject.toml"
  - "ruff.toml"
  - ".ruff.toml"

setup_files:
  - "uv.lock"
  - "pyproject.toml"
```

### node-npm

```yaml
commands:
  setup: "npm install"
  test: "npm test"
  lint: "npx eslint ."
  format: "npx prettier --check ."
  typecheck: "npx tsc --noEmit"

code_patterns:
  - "**/*.js"
  - "**/*.ts"
  - "**/*.jsx"
  - "**/*.tsx"

config_files:
  - "package.json"
  - "tsconfig.json"
  - ".eslintrc*"
  - ".prettierrc*"

setup_files:
  - "package-lock.json"
  - "package.json"
```

### go

```yaml
commands:
  setup: "go mod download"
  test: "go test ./..."
  lint: "golangci-lint run"
  format: 'test -z "$(gofmt -l .)"'

code_patterns:
  - "**/*.go"

config_files:
  - "go.mod"
  - "go.sum"
  - ".golangci.yml"

setup_files:
  - "go.mod"
  - "go.sum"
```

### rust

```yaml
commands:
  setup: "cargo fetch"
  test: "cargo test"
  lint: "cargo clippy -- -D warnings"
  format: "cargo fmt --check"

code_patterns:
  - "**/*.rs"

config_files:
  - "Cargo.toml"
  - "Cargo.lock"
  - "clippy.toml"

setup_files:
  - "Cargo.toml"
  - "Cargo.lock"
```

## Extending Presets

Override specific commands while inheriting the rest:

```yaml
preset: python-uv
commands:
  test: "uv run pytest -x --tb=short"  # Custom test command
```

Disable a command from the preset:

```yaml
preset: python-uv
commands:
  typecheck: null  # Disable type checking
```

Override patterns (replaces preset patterns entirely):

```yaml
preset: node-npm
code_patterns:
  - "src/**/*.ts"
  - "lib/**/*.ts"
```

## Merge Rules

When using a preset:

- **Commands**: User value replaces preset value. `null` disables. Omitted inherits from preset.
- **Lists** (`code_patterns`, `config_files`, `setup_files`): User list **replaces** preset list entirely.
- **Coverage**: User config **replaces** preset coverage. `coverage: null` disables.
- **Evidence check**: User config **replaces** preset (presets do not define evidence checks).
- **Validation triggers**: User config **replaces** preset (presets do not define triggers).

## Code Patterns

Glob patterns determine which file changes trigger validation.

| Pattern | Matches |
|---------|---------|
| `*.py` | Any `.py` file (basename match) |
| `src/*.py` | `.py` files directly in `src/` |
| `src/**/*.py` | `.py` files anywhere under `src/` |
| `**/*.py` | Any `.py` file in any directory |

If `code_patterns` is empty or omitted, all files trigger validation.

## Coverage Configuration

Coverage is optional. When enabled, all three fields are required:

```yaml
coverage:
  format: xml          # Cobertura XML format
  file: coverage.xml   # Path relative to repo root
  threshold: 80        # Minimum 80% line coverage
```

Use a separate coverage command if test coverage is slow:

```yaml
preset: go
coverage:
  command: "go test -coverprofile=coverage.out ./... && gocover-cobertura < coverage.out > coverage.xml"
  format: xml
  file: coverage.xml
  threshold: 80
```

This keeps `go test ./...` (from preset) fast for regular validation.

## Examples

### Minimal Python

```yaml
preset: python-uv
```

### Python with Coverage

```yaml
preset: python-uv
coverage:
  command: "uv run pytest --cov --cov-report=xml"
  format: xml
  file: coverage.xml
  threshold: 85
```

### Node.js with Custom Test

```yaml
preset: node-npm
commands:
  test: "npm run test:ci"
```

### Full Custom (No Preset)

```yaml
commands:
  setup: "make deps"
  test: "make test"
  lint: "make lint"
  format: "make fmt-check"

code_patterns:
  - "**/*.go"
  - "**/*.c"
  - "Makefile"
```

### Partial Commands (Skip Validation Steps)

```yaml
commands:
  setup: "npm install"
  test: "npm test"
# lint, format, typecheck, e2e are skipped
```

## Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `mala.yaml not found` | Missing default config file | Create `mala.yaml` in repo root or pass `--config PATH` |
| `Configuration file not found: <path>` | Missing explicit config selected with `--config` | Create the selected file or fix the path |
| `Unknown preset 'foo'` | Invalid preset name | Use: python-uv, node-npm, go, rust |
| `Unknown field 'bar'` | Typo or invalid field | Check field names against schema |
| `At least one command must be defined` | No commands and no preset | Add a preset or define commands |
| `Command cannot be empty string` | Used `""` instead of `null` | Use `null` to disable a command |
| `Unsupported coverage format` | Used non-XML format | Use `format: xml` |
| `Coverage file not found` | Coverage file missing after test | Verify coverage command produces the file |

## Cache Invalidation

| File Changed | Re-runs |
|--------------|---------|
| Matches `code_patterns` | lint, format, typecheck |
| Matches `config_files` | lint, format, typecheck |
| Matches `setup_files` | setup, lint, format, typecheck |
| Selected Mala config (`mala.yaml` by default, or `--config PATH`) | All commands |

## Claude Settings Sources

Control which Claude Code settings files the Claude Agent SDK uses during validation. This allows you to define mala-specific settings (e.g., timeouts, models) that differ from your interactive development sessions.

### Configuration Methods

Configure sources via mala.yaml, environment variable, or CLI flag:

```yaml
# mala.yaml (top-level key)
preset: python-uv
claude_settings_sources: [local, project]
```

```bash
# Environment variable
export MALA_CLAUDE_SETTINGS_SOURCES=local,project
```

```bash
# CLI flag
mala run --claude-settings-sources local,project,user
```

### Precedence

Settings sources are resolved in this order (highest wins):

1. CLI flag (`--claude-settings-sources`)
2. Environment variable (`MALA_CLAUDE_SETTINGS_SOURCES`)
3. Selected project config (`claude_settings_sources`)
4. Default: `[local, project]`

### Valid Sources

| Source | File Path | Description |
|--------|-----------|-------------|
| `local` | `.claude/settings.local.json` | Repository root, typically for validation-specific settings |
| `project` | `.claude/settings.json` | Repository root, shared project settings |
| `user` | `~/.claude/settings.json` | User's home directory, personal settings |

The SDK merges settings with local > project > user precedence (first source wins for conflicts).

### Breaking Change

**Default changed from `[project, user]` to `[local, project]`.**

This prioritizes validation-specific settings over user settings, ensuring reproducible validation across CI and developer machines.

**Migration**: To restore old behavior, explicitly set:

```yaml
claude_settings_sources: [project, user]
```

### Recommendation

For reproducible validation environments, commit `.claude/settings.local.json` to version control:

```json
// .claude/settings.local.json
{
  "timeout": 300
}
```

This file is typically gitignored for interactive Claude Code use, but committing it ensures consistent validation across CI and all developers.

## Coder Selection

Mala can drive its per-issue implementation agent on Claude, Sourcegraph's
Amp, or OpenAI's Codex (`codex app-server`). Selection is global to a run —
every issue uses the same coder, and fixer agents follow the main coder.

### Configuration

```yaml
# Select the Amp coder explicitly
coder: amp
amp_mode: deep    # deep (GPT-5 reasoning, default), smart (Opus), rush (Haiku)
```

```yaml
# Select the Codex coder
coder: codex
model: gpt-5.5
effort: high
coder_options:
  codex:
    approval_policy: never        # default
    sandbox: danger-full-access   # default
```

Existing config files **without `coder:` remain valid** and default to
`claude`; the field is purely additive.

### Precedence

`coder`, `amp_mode`, `model`, `effort`, and `coder_options.codex.*` all follow the same
**CLI > env > yaml > default** precedence used by `claude_settings_sources`:

1. CLI flag (`--coder`, `--amp-mode`, `--model`, `--effort`, `--codex-approval-policy`, `--codex-sandbox`)
2. Environment variable (`MALA_CODER`, `MALA_AMP_MODE`, `MALA_MODEL`, `MALA_EFFORT`, `MALA_CODEX_APPROVAL_POLICY`, `MALA_CODEX_SANDBOX`)
3. Selected project config (`coder:`, `amp_mode:`, `model:`, `effort:`, `coder_options.codex.*`)
4. Defaults: `coder=claude`, `amp_mode=deep`, `claude.model=opus[1m]`, `codex.model=gpt-5.5`, `codex.approval_policy=never`, `codex.sandbox=danger-full-access`

### Validation

Invalid values fail validation at config-load time, before any agent process
starts:

| Field | Allowed values |
|-------|----------------|
| `coder` | `claude`, `amp`, `codex` |
| `amp_mode` | `smart`, `rush`, `deep` |
| `model` | non-empty string (e.g., `opus[1m]` or `gpt-5.5`); confirmed against the live backend at run time |
| `effort` | `low`, `medium`, `high`, `xhigh`, `max` (Amp modes further restrict supported values) |
| `coder_options.codex.approval_policy` | `never`, `on-request`, `on-failure`, `untrusted` |
| `coder_options.codex.sandbox` | `read-only`, `workspace-write`, `danger-full-access` |

### Cross-Coder Behavior

- `amp_mode` is **only consulted when `coder: amp`**. Set with another
  coder, it is ignored and does not error.
- `coder_options.codex.*` is **only consulted when `coder: codex`**.
  Set with another coder, it is ignored.
- `claude_settings_sources` is ignored when `coder: amp` or `coder: codex`.
- `MALA_DISALLOWED_TOOLS` is a **no-op under `coder: amp`** in MVP and warns
  once on run start; tracked as a follow-up. Under `coder: codex` it is
  enforced inside the bundled `PreToolUse` command hook.

### Amp Prerequisites

Selecting `coder: amp` requires the official Amp **binary** install, a working
Bun runtime, and a writable `~/.config/amp/plugins/`. The npm-distributed Amp
is **unsupported** and will fail mala's runtime plugin self-test before any
issue agent runs. See the [Amp prerequisites in README](../README.md#amp-optional-for-coder-amp)
for the full list.

### Codex Prerequisites

Selecting `coder: codex` requires the `openai-codex-app-server-sdk` Python
SDK (installed with mala by default), the `openai-codex-cli-bin` runtime
(pulled in by the SDK), Codex auth configured locally, and a readable Codex
auth source (`auth.json`, keyring config, or auth env vars).
Mala installs the bundled `mala-safety` plugin into a per-run temporary
`CODEX_HOME` and passes that home only to mala-launched `codex app-server`
subprocesses; it does not mutate the user's normal `~/.codex` plugin cache or
`config.toml`, and it only copies auth-scoped Codex settings into the temporary
home. User plugins, hooks, MCP servers, feature flags, and shell-environment
policies are not inherited by Mala workers. Mala runs a fail-closed prerequisite
check + plugin self-test before any issue agent is spawned: missing SDK/runtime/auth raises
`CodexNotInstalledError`, and a missing/untrusted/disabled bundled
`SessionStart` or `PreToolUse` hook raises `CodexHookNotActiveError`. See the
[Codex prerequisites in README](../README.md#codex-for-coder-codex)
for the full list.

### Codex Coder Options

`coder_options.codex` configures Codex-only safety/runtime options. Use the
top-level `model` and `effort` settings for Codex model and reasoning effort.
All fields are optional; omitted fields fall through to the documented defaults.

```yaml
coder: codex
model: gpt-5.5                    # default; latest gpt-5.5 family release
effort: high                      # passed through to Codex per turn
coder_options:
  codex:
    approval_policy: never         # default; see warnings below
    sandbox: danger-full-access    # default; see warnings below
    mcp_servers:                   # merged with bundled mala-locking server (cannot replace)
      myservice:
        command: my-mcp-bin
        args: ["--port", "1234"]
```

**Defaults rationale.** `approval_policy: never` + `sandbox: danger-full-access`
puts Codex in the same unattended posture as Amp's `--dangerously-allow-all`:
the bundled `PreToolUse` command hook (`mala-codex-pre-tool-use`) plus the
bundled `mala-locking` MCP server are the safety surface. If the hook isn't
loaded, the run aborts before any issue agent is spawned. Tightening the
defaults (e.g., `sandbox: workspace-write`, `approval_policy: on-request`)
is supported but may stall unattended runs that hit approval prompts.

**Bundled MCP cannot be replaced.** The bundled `mala-locking` MCP server
is shipped inside the Mala Codex plugin (`plugins/codex/mala-safety/.codex-plugin/.mcp.json`)
and is mandatory for lock-ownership enforcement. User-supplied
`coder_options.codex.mcp_servers` entries are **merged** with the bundled
spec; the bundled `mala-locking` key wins on conflict (you cannot point
the locking name at a different binary).

### Example

```yaml
preset: python-uv
coder: amp
amp_mode: rush     # Haiku — cheaper for batch issue work
evidence_check:
  required: [test, lint]
```

```yaml
preset: python-uv
coder: codex
model: gpt-5.5
effort: high
evidence_check:
  required: [test, lint]
```

## Code Review Configuration

Code review is configured per validation trigger using the `code_review` block. See [validation-triggers.md](validation-triggers.md#code-review) for full documentation.

```yaml
validation_triggers:
  session_end:
    failure_mode: continue
    commands:
      - ref: lint
      - ref: test
    code_review:
      enabled: true
      reviewer_type: cerberus    # "cerberus" (default) or "agent_sdk"
      finding_threshold: P1
      track_review_issues: true  # Create beads issues for P2/P3 findings (default)
```

### Reviewer Types

| Type | Description |
|------|-------------|
| `cerberus` (default) | Invokes the Cerberus v2 `cerberus` binary on `$PATH`. |
| `agent_sdk` | Uses Claude agents for interactive code review. |

## Per-Issue Review

> **Breaking Change (2026-01-12)**: Per-issue code review is now disabled by default.
> To restore the previous behavior where every issue session is reviewed, add:
> ```yaml
> per_issue_review:
>   enabled: true
> ```

Per-issue review runs code review at the end of each issue session, reviewing only the commits made for that specific issue. This is separate from trigger-based reviews (configured under `validation_triggers`).

### Minimal Configuration

Enable per-issue review with defaults:

```yaml
per_issue_review:
  enabled: true
```

### Full Configuration

All available options with their defaults:

```yaml
per_issue_review:
  enabled: true
  reviewer_type: cerberus        # "cerberus" or "agent_sdk"
  max_retries: 3                 # Retry attempts on execution error
  finding_threshold: none        # P0 | P1 | P2 | P3 | none
  track_review_issues: true      # Create beads issues for findings
  failure_mode: continue         # continue | abort | remediate

  # Cerberus-specific settings (when reviewer_type: cerberus)
  cerberus:
    timeout: 600
    spawn_args: []
    wait_args: []
    env: {}

  # Agent SDK settings (when reviewer_type: agent_sdk)
  agent_sdk_timeout: 300
  agent_sdk_model: opus          # sonnet | opus | haiku
```

### Cerberus v2 Runtime

When `reviewer_type: cerberus`, mala invokes the v2 `cerberus` binary from
`$PATH`. It does not auto-discover Claude plugin cache installs.

Cerberus settings are available under both
`epic_verification.cerberus`,
`validation_triggers.<trigger>.code_review.cerberus`, and
`per_issue_review.cerberus`:

| Field | Description |
|-------|-------------|
| `timeout` | Review timeout in seconds (default: 600) |
| `spawn_args` | Additional arguments for `cerberus spawn-code-review` |
| `wait_args` | Additional arguments for `cerberus wait` |
| `env` | Extra environment variables passed to `cerberus` |

Use `cerberus.env.CERBERUS_ROOT` when the install does not follow mala's
derived layout. The automatic derivation expects:

```text
<CERBERUS_ROOT>/bin/cerberus
<CERBERUS_ROOT>/prompts/reviewers/
```

Bare `go install`, distro packages that split `bin` and `share`, and custom
copies such as `~/bin/cerberus` must set `CERBERUS_ROOT` explicitly.

Use `cerberus.env.CERBERUS_PROJECT_KEY` only when you need a stable project key
that differs from mala's repo-path-derived default.

### Reviewer Selection Priority

> **Reviewer Selection**: When `per_issue_review` is enabled, it also determines the reviewer type
> (Cerberus or Agent SDK) used for **all reviews**, including trigger-based reviews. To use
> different reviewer settings for triggers, leave `per_issue_review.enabled: false` so your
> trigger configs take priority.

The reviewer selection priority is:
1. `per_issue_review` settings (if `enabled: true`)
2. Individual trigger `code_review` settings
3. Built-in defaults

### Finding Threshold Behavior

| `finding_threshold` | Behavior |
|---------------------|----------|
| `none` (default) | Never fail on findings; create beads issues and continue |
| `P3` | Fail if any finding (P0-P3) |
| `P2` | Fail if any P0, P1, or P2 finding |
| `P1` | Fail if any P0 or P1 finding |
| `P0` | Fail only if P0 (critical) finding |

### Failure Mode Behavior

| `failure_mode` | On execution error | On findings exceeding threshold |
|----------------|-------------------|--------------------------------|
| `continue` (default) | Log warning, skip review, continue | Create beads issues, continue |
| `abort` | Fail the run | Create beads issues, fail the run |
| `remediate` | Retry reviewer up to `max_retries` | Attempt to fix findings, then fail if unresolved |

## Limitations

- **Windows**: Shell commands use POSIX syntax. Avoid `export`, shell built-ins, etc.
- **Monorepos**: Single `mala.yaml` per repo. Multiple language support not yet available.
- **Coverage formats**: Only Cobertura XML supported. JSON/LCOV planned for future.
