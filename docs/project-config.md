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

commands:                # Optional. Command definitions
  setup: string | null   # Environment setup (e.g., "uv sync", "npm install")
  test: string | null    # Test runner (e.g., "uv run pytest", "go test ./...")
  lint: string | null    # Linter (e.g., "uvx ruff check .", "golangci-lint run")
  format: string | null  # Formatter check (e.g., "uvx ruff format --check .")
  typecheck: string | null  # Type checker (e.g., "uvx ty check", "tsc --noEmit")
  e2e: string | null     # End-to-end tests (e.g., "uv run pytest -m e2e")

run_level_commands:      # Optional. Overrides for run-level validation
  test: string | null    # Run-level test command (e.g., with coverage)

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
```

## Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `preset` | string | No | Preset name to extend |
| `commands` | object | No | Map of command kind to shell command |
| `commands.<kind>` | string or null | No | Shell command. `null` explicitly disables |
| `run_level_commands` | object | No | Overrides for run-level validation commands |
| `run_level_commands.<kind>` | string or null | No | Run-level command override. `null` disables |
| `code_patterns` | list | No | Glob patterns for code files |
| `config_files` | list | No | Tool config files that trigger re-lint |
| `setup_files` | list | No | Lock files that trigger setup re-run |
| `coverage` | object | No | Coverage configuration |
| `coverage.command` | string | No | Coverage-enabled test command |
| `coverage.format` | string | Yes* | Format: `xml` |
| `coverage.file` | string | Yes* | Path to coverage report |
| `coverage.threshold` | number | Yes* | Minimum coverage % |

*Required when `coverage` section is present.

## Built-in Presets

### python-uv

```yaml
commands:
  setup: "uv sync"
  test: "uv run pytest"
  lint: "uvx ruff check ."
  format: "uvx ruff format --check ."
  typecheck: "uvx ty check"
  e2e: "uv run pytest -m e2e"

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
- **Run-level commands**: User value overrides base commands for run-level validation only. Omitted inherits from `commands`.
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

### Python Run-Level Coverage Only

```yaml
preset: python-uv
run_level_commands:
  test: "uv run pytest --cov=src --cov-report=xml:coverage.xml --cov-fail-under=0"
coverage:
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

## Limitations

- **Windows**: Shell commands use POSIX syntax. Avoid `export`, shell built-ins, etc.
- **Monorepos**: Single `mala.yaml` per repo. Multiple language support not yet available.
- **Coverage formats**: Only Cobertura XML supported. JSON/LCOV planned for future.
