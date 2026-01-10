# Project Configuration

> For CLI flags and runtime settings, see [CLI Reference](cli-reference.md).

`mala.yaml` is the required configuration file for mala. It defines your project's validation commands, code patterns, and coverage settings. Place it in the root of your repository.

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
  test: string | object | null      # Test runner (e.g., "uv run pytest", "go test ./...")
  lint: string | object | null      # Linter (e.g., "uvx ruff check .", "golangci-lint run")
  format: string | object | null    # Formatter command (e.g., "uvx ruff format .")
  typecheck: string | object | null # Type checker (e.g., "uvx ty check", "tsc --noEmit")
  e2e: string | object | null       # End-to-end tests (e.g., "uv run pytest -m e2e")

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

claude_settings_sources: list     # Optional. SDK settings sources (default: [local, project])

timeout_minutes: int             # Optional. Agent timeout in minutes (default: 60)

context_restart_threshold: float # Optional. Ratio (0.0-1.0) to restart agent on context pressure (default: 0.70)
context_limit: int               # Optional. Maximum context tokens (default: 200000)

max_idle_retries: int            # Optional. Maximum idle timeout retries (default: 2)
idle_timeout_seconds: float      # Optional. Idle timeout in seconds (default: derived from timeout_minutes)

max_diff_size_kb: int            # Optional. Maximum diff size in KB for epic verification

validation_triggers:             # Optional. See validation-triggers.md
  epic_completion: object        # Run validation when epics complete
  session_end: object            # Run validation at session end
  periodic: object               # Run validation periodically
```

## Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `preset` | string | No | Preset name to extend |
| `commands` | object | No | Map of command kind to shell command or object |
| `commands.<kind>` | string, object, or null | No | Shell command or `{command, timeout}`. `null` explicitly disables |
| `code_patterns` | list | No | Glob patterns for code files |
| `config_files` | list | No | Tool config files that trigger re-lint |
| `setup_files` | list | No | Lock files that trigger setup re-run |
| `coverage` | object | No | Coverage configuration |
| `coverage.command` | string | No | Coverage-enabled test command |
| `coverage.format` | string | Yes* | Format: `xml` |
| `coverage.file` | string | Yes* | Path to coverage report |
| `coverage.threshold` | number | Yes* | Minimum coverage % |
| `claude_settings_sources` | list | No | SDK settings sources: `local`, `project`, `user` (default: `[local, project]`) |
| `timeout_minutes` | integer | No | Agent timeout in minutes (default: 60). Can be overridden by CLI `--timeout` |
| `context_restart_threshold` | float | No | Ratio (0.0-1.0) at which to restart agent on context pressure (default: 0.70) |
| `context_limit` | integer | No | Maximum context tokens (default: 200000) |
| `max_idle_retries` | integer | No | Maximum idle timeout retries before aborting (default: 2). Set to 0 to disable retries. |
| `idle_timeout_seconds` | float | No | Seconds to wait for SDK activity before triggering idle recovery (default: derived from `timeout_minutes`). Set to 0 to disable idle timeout. |
| `max_diff_size_kb` | integer | No | Maximum diff size in KB for epic verification. Diffs larger than this limit will skip verification. |
| `validation_triggers` | object | No | Trigger configuration. See [validation-triggers.md](validation-triggers.md) |

*Required when `coverage` section is present.

## Command Object Form

Each command can be defined as a string or an object with an optional timeout:

```yaml
commands:
  test:
    command: "uv run pytest"
    timeout: 600
```

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
- **Global commands**: User value overrides base commands for global validation only. Omitted inherits from `commands`.
- **Lists** (`code_patterns`, `config_files`, `setup_files`): User list **replaces** preset list entirely.
- **Coverage**: User config **replaces** preset coverage. `coverage: null` disables.

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
| `mala.yaml not found` | Missing config file | Create `mala.yaml` in repo root |
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
| `mala.yaml` | All commands |

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
3. mala.yaml (`claude_settings_sources`)
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
| `cerberus` (default) | Uses the Cerberus CLI plugin for review. Requires plugin installation. |
| `agent_sdk` | Uses Claude agents for interactive code review. |

## Limitations

- **Windows**: Shell commands use POSIX syntax. Avoid `export`, shell built-ins, etc.
- **Monorepos**: Single `mala.yaml` per repo. Multiple language support not yet available.
- **Coverage formats**: Only Cobertura XML supported. JSON/LCOV planned for future.
