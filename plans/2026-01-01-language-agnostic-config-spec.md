# Language-Agnostic Configuration via `mala.yaml`

## Overview

Mala currently has Python-specific commands and patterns hard-coded throughout the codebase (pytest, ruff, ty, uv). This feature introduces a `mala.yaml` configuration file that allows users to define their own setup, test, lint, format, typecheck, and e2e commands for any programming language or toolchain. Users can optionally extend built-in presets for common stacks (Python, Node.js, Go, Rust) and override specific commands as needed.

## Goals

- Enable Mala to work with any programming language by externalizing all language-specific configuration to `mala.yaml`
- Provide built-in presets for common stacks (python-uv, node-npm, go, rust) to minimize configuration burden
- Allow users to extend presets with project-specific overrides
- Make coverage reporting optional and configurable (Cobertura XML format for MVP; JSON and LCOV in future iterations)
- Support user-defined file patterns to determine what constitutes "code changes"

## Non-Goals (Out of Scope)

- Auto-detection of project type or language without explicit configuration
- Backwards compatibility fallback to Python defaults when no config exists
- Migration tooling or `mala init` command (users must manually create config)
- Monorepo support with multiple languages or nested configurations
- Versioning presets separately from Mala releases
- Custom command types beyond the six built-in kinds (setup, test, lint, format, typecheck, e2e)
- Remote/dynamic preset fetching at runtime
- JSON and LCOV coverage formats (deferred to follow-up)

## Acceptance Criteria

- Given a Go project with a valid `mala.yaml` specifying Go commands, when Mala runs validation, then it executes the configured Go toolchain commands (`go mod download`, `go test`, `golangci-lint`, `gofmt`)

- Given a Node.js project with a valid `mala.yaml` specifying npm commands, when Mala runs validation, then it executes the configured Node.js toolchain commands (`npm install`, `npm test`, `npx eslint .`, `npx prettier --check .`)

- Given a `mala.yaml` that specifies `preset: python-uv` with a custom `test` command override, when Mala runs validation, then it uses the preset's commands except for the overridden test command

- Given a `mala.yaml` with only `setup` and `test` commands defined (no lint/format/typecheck), when Mala runs validation, then it skips the undefined validation steps without error

- Given a project with no `mala.yaml` file, when Mala attempts to run, then it fails fast with a clear error message indicating the configuration file is required

- Given a `mala.yaml` with invalid syntax or unknown fields, when Mala parses the config, then it fails fast with a specific error message identifying the problem

- Given a `mala.yaml` with coverage enabled specifying `format: xml`, `file: coverage.xml`, and `threshold: 80`, when tests complete, then Mala parses the coverage file and enforces the configured threshold

- Given a `mala.yaml` without a `coverage` section, when validation completes, then no coverage evaluation is performed

- Given a `mala.yaml` with custom `code_patterns: ["*.go", "cmd/**"]`, when determining if changes require validation, then only files matching those patterns (relative to repo root) trigger validation

- Given a command string `"npx eslint ."`, when extracting the tool name, then Mala skips the known wrapper `npx` and extracts `eslint` for logging and evidence detection

## Configuration Schema

### Top-Level Structure

```yaml
# mala.yaml - all fields optional except when noted

preset: string           # Optional. Name of preset to extend (e.g., "python-uv", "go")

commands:                # Optional. Command definitions (strings)
  setup: string | null   # Environment setup (e.g., "uv sync", "npm install")
  test: string | null    # Test runner (e.g., "uv run pytest", "go test ./...")
  lint: string | null    # Linter (e.g., "uvx ruff check .", "golangci-lint run")
  format: string | null  # Formatter check (e.g., "uvx ruff format --check .", "gofmt -l .")
  typecheck: string | null  # Type checker (e.g., "uvx ty check", "tsc --noEmit")
  e2e: string | null     # End-to-end tests (e.g., "uv run pytest -m e2e")

code_patterns:           # Optional. List of glob patterns for code files
  - string               # Patterns relative to repo root (e.g., "*.py", "src/**/*.ts")

config_files:            # Optional. Tool config files that invalidate lint/format cache
  - string               # Patterns relative to repo root (e.g., ".eslintrc*", "ruff.toml")

setup_files:             # Optional. Lock/dependency files that invalidate setup cache
  - string               # Patterns relative to repo root (e.g., "package-lock.json", "uv.lock")

coverage:                # Optional. Omit entirely to disable coverage
  command: string        # Optional. Separate command to run tests WITH coverage (overrides test command for coverage runs)
  format: string         # Required when coverage enabled. Values: "xml" (MVP only)
  file: string           # Required when coverage enabled. Path relative to repo root
  threshold: number      # Required when coverage enabled. Minimum coverage percentage (0-100)
```

### Field Semantics

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `preset` | string | No | none | Preset name to extend. Must match a file in presets/ |
| `commands` | object | No | {} | Map of command kind to shell command string |
| `commands.<kind>` | string or null | No | (from preset or skip) | Shell command to execute. `null` explicitly disables |
| `code_patterns` | list[string] | No | (from preset or ["**/*"]) | Glob patterns for code change detection |
| `config_files` | list[string] | No | (from preset or []) | Tool config files that invalidate lint/format/typecheck cache |
| `setup_files` | list[string] | No | (from preset or []) | Lock/dependency files that invalidate setup cache |
| `coverage` | object | No | (disabled) | Coverage configuration. Omit to disable |
| `coverage.command` | string | No | (uses test command) | Command to run tests with coverage. If omitted, uses `commands.test` |
| `coverage.format` | string | Yes* | - | Coverage report format. MVP supports: "xml" |
| `coverage.file` | string | Yes* | - | Path to coverage report file, relative to repo root |
| `coverage.threshold` | number | Yes* | - | Minimum coverage percentage (0-100). Coverage >= threshold passes. |

*Required when `coverage` section is present. Setting any required field to `null` inside coverage is a validation error. Use `coverage: null` to disable coverage entirely.

### Coverage Execution

When `coverage` is configured:
- **Regular validation**: Runs `commands.test` (fast, no coverage overhead)
- **Coverage check**: Runs `coverage.command` (or `commands.test` if not specified) to generate the coverage report
- **When coverage runs**: Coverage is checked on every validation run where tests pass. The coverage file must exist after test execution.
- **Threshold enforcement**: Uses root `<coverage line-rate>` attribute from Cobertura XML, multiplied by 100. If `line-rate` is missing, validation fails with an error. Branch coverage is ignored for MVP.

### Example Configurations

**Minimal Go project:**
```yaml
preset: go
```

**Node.js with custom test command:**
```yaml
preset: node-npm
commands:
  test: "npm run test:ci"
```

**Full custom configuration (no preset):**
```yaml
commands:
  setup: "go mod download"
  test: "go test -v ./..."
  lint: "golangci-lint run"
  format: "gofmt -l ."

code_patterns:
  - "*.go"
  - "go.mod"
  - "go.sum"

coverage:
  format: xml
  file: coverage.xml
  threshold: 80
```

**Disable specific command from preset:**
```yaml
preset: python-uv
commands:
  typecheck: null  # Explicitly disable type checking
```

## Preset Format and Discovery

### Preset File Structure

Presets are YAML files with the same schema as `mala.yaml` (excluding the `preset` field). They are stored in the `src/domain/validation/presets/` directory within the Mala package.

```yaml
# presets/python-uv.yaml
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

### Discovery Mechanism

1. Presets are bundled as package data in `src/domain/validation/presets/`
2. At runtime, the Preset Registry uses `importlib.resources` to discover presets (works in all installation scenarios including wheels)
3. Preset names are derived from filenames (e.g., `python-uv.yaml` -> `python-uv`)
4. Available presets for MVP: `python-uv`, `node-npm`, `go`, `rust`
5. **Loading timing**: Presets are loaded lazily when first referenced by a `mala.yaml`. Only the referenced preset is validated, not all presets. This allows adding new presets without affecting existing users.
6. **Validation**: If a referenced preset has invalid YAML or schema errors, config loading fails with a clear error message.

### Built-in Presets (MVP)

| Preset | Setup | Test | Lint | Format | Typecheck | E2E |
|--------|-------|------|------|--------|-----------|-----|
| `python-uv` | `uv sync` | `uv run pytest` | `uvx ruff check .` | `uvx ruff format --check .` | `uvx ty check` | `uv run pytest -m e2e` |
| `node-npm` | `npm install` | `npm test` | `npx eslint .` | `npx prettier --check .` | `npx tsc --noEmit` | (none) |
| `go` | `go mod download` | `go test ./...` | `golangci-lint run` | `test -z "$(gofmt -l .)"` | (none) | (none) |
| `rust` | `cargo fetch` | `cargo test` | `cargo clippy` | `cargo fmt --check` | (none) | (none) |

**Note on Coverage**: Presets do not include `coverage` configuration. Coverage is opt-in and requires the user to:
1. Configure their test command to generate Cobertura XML output
2. Add a `coverage` section specifying format, file path, and threshold

Example for Go with coverage:
```yaml
preset: go
coverage:
  command: "go test -coverprofile=coverage.out ./... && gocover-cobertura < coverage.out > coverage.xml"
  format: xml
  file: coverage.xml
  threshold: 80
```

This keeps `commands.test` fast (`go test ./...` from preset) for regular validation, while `coverage.command` runs the slower coverage-enabled tests only when coverage threshold checking is needed.

## Merge Rules

When a user config specifies a preset, the merger applies these rules:

### Scalar Fields (preset, coverage.format, coverage.file, coverage.threshold)
- User value replaces preset value entirely
- `null` in user config removes the field (disables it)

### Command Fields (commands.setup, commands.test, etc.)
- User value replaces preset value for that specific command
- `null` explicitly disables the command even if preset defines it
- Omitted commands inherit from preset

### List Fields (code_patterns, config_files, setup_files)
- User list **replaces** preset list entirely (no merge/append)
- To extend preset patterns, user must include all desired patterns

### Coverage Object
- If user defines `coverage`, it **replaces** preset coverage entirely
- If user omits `coverage`, preset coverage is used (if any)
- If user sets `coverage: null`, coverage is disabled even if preset defines it

### Merge Examples

```yaml
# Preset: python-uv
commands:
  test: "uv run pytest"
  lint: "uvx ruff check ."
code_patterns: ["*.py"]

# User config
preset: python-uv
commands:
  test: "uv run pytest -x"  # Override
  lint: null                 # Disable
code_patterns: ["*.py", "tests/**"]  # Replace entirely

# Result
commands:
  setup: "uv sync"           # From preset
  test: "uv run pytest -x"   # User override
  lint: null                 # Disabled
  format: "uvx ruff format --check ."  # From preset
  typecheck: "uvx ty check"  # From preset
code_patterns: ["*.py", "tests/**"]  # User (replaced preset)
```

## Code Pattern Matching

### Semantics

- Patterns are evaluated relative to the repository root
- Uses `fnmatch` for simple patterns and `glob`-style matching for `**` patterns (compatible with Python 3.10+)
- **Filename-only patterns** (no `/`): matched against `os.path.basename(path)` using `fnmatch.fnmatch()`
- **Path patterns** (contain `/`): matched against full relative path; leading `./` is stripped before matching
- **Recursive patterns** (`**`): use `fnmatch.fnmatch()` with `**` replaced by a regex that matches any path segment sequence
- Path separators are normalized to `/` before matching (cross-platform)
- Note: Brace expansion (e.g., `*.{js,ts}`) is **not** supported; use multiple patterns instead

### Pattern Syntax

| Pattern | Matches | Explanation |
|---------|---------|-------------|
| `*.py` | `main.py`, `src/utils.py`, `tests/test_main.py` | Matches basename only (no `/` in pattern) |
| `src/*.py` | `src/main.py`, `src/utils.py` | Files directly in src/ |
| `src/**/*.py` | `src/main.py`, `src/sub/deep/file.py` | All .py files under src/ |
| `**/test_*.py` | `test_main.py`, `tests/test_utils.py` | Test files anywhere |

To match multiple extensions, use separate patterns:
```yaml
code_patterns:
  - "**/*.js"
  - "**/*.ts"
```

### Matching Algorithm

```python
import re

def glob_to_regex(pattern: str) -> str:
    """Convert glob pattern to regex. * matches non-slash, ** matches anything."""
    pattern = pattern.lstrip('./')
    result = []
    i = 0
    while i < len(pattern):
        if pattern[i:i+2] == '**':
            result.append('.*')  # ** matches anything including /
            i += 2
            if i < len(pattern) and pattern[i] == '/':
                i += 1  # Skip trailing / after **
        elif pattern[i] == '*':
            result.append('[^/]*')  # * matches anything EXCEPT /
            i += 1
        elif pattern[i] == '?':
            result.append('[^/]')  # ? matches single non-slash char
            i += 1
        elif pattern[i] in '.^$+{}[]|()\\':
            result.append('\\' + pattern[i])  # Escape regex special chars
            i += 1
        else:
            result.append(pattern[i])
            i += 1
    return '^' + ''.join(result) + '$'

def matches_pattern(path: str, pattern: str) -> bool:
    if '/' not in pattern and '**' not in pattern:
        # Filename-only pattern: match against basename
        regex = glob_to_regex(pattern)
        return re.match(regex, os.path.basename(path)) is not None
    else:
        # Path pattern: match against full relative path
        regex = glob_to_regex(pattern)
        return re.match(regex, path) is not None

for changed_file in changed_files:
    relative_path = normalize_separators(changed_file)  # Use forward slashes

    for pattern in code_patterns + config_files + setup_files:
        if matches_pattern(relative_path, pattern):
            return TRIGGER_VALIDATION
```

**Pattern matching behavior**:
- `src/*.py` matches `src/main.py` but NOT `src/sub/file.py` (single `*` doesn't cross `/`)
- `src/**/*.py` matches `src/main.py` AND `src/sub/file.py` (`**` crosses directories)
- `*.py` matches any `.py` file by basename (no `/` in pattern)

### Evaluation

1. When changes are detected, extract list of changed file paths (relative to repo root)
2. If `mala.yaml` itself changed, validation is always triggered and all caches are invalidated
3. For each changed file, check if it matches any pattern in `code_patterns`, `config_files`, or `setup_files`
4. If any file matches any list, validation is triggered
5. If `code_patterns` is empty or not defined, all files trigger validation
6. Changes to `config_files` invalidate lint/format/typecheck caches
7. Changes to `setup_files` invalidate setup cache (forcing environment re-setup)

## Tool Name Extraction

To derive friendly tool names from command strings for logging and evidence detection:

### Algorithm

1. If command contains shell operators (`&&`, `||`, `|`, `;`), try each command segment in order until a valid tool is found
2. Parse the command segment into tokens (shell-style word splitting via `shlex.split`). If parsing fails (e.g., unbalanced quotes), use the first whitespace-delimited word as the tool name
3. Skip leading environment variable assignments (tokens containing `=` before the command)
4. Skip shell built-ins: `export`, `set`, `unset`, `source`, `.`, `eval`, `exec`, `cd`, `pushd`, `popd`, `alias`, `unalias`, `declare`, `local`, `readonly`, `typeset`
5. Strip path prefixes from first token (e.g., `/usr/bin/eslint` -> `eslint`)
6. A **valid tool** is any token that: (a) is not an env var assignment, (b) is not a shell built-in, and (c) does not start with `-` (not a flag)
7. Apply wrapper rules in order:

**Single-token wrappers** (skip wrapper, skip flags and their values, use next positional token):
- `npx`, `bunx` (Node.js package runners)
- `uvx`, `pipx` (Python package runners)

**Multi-token wrappers** (skip sequence, skip flags and their values, use next positional token):
- `python -m`, `python3 -m` (Python module execution)
- `uv run`, `poetry run` (Python environment runners)

**Flag handling**: When skipping flags after wrappers:
- Skip tokens starting with `-` or `--`
- Known flags that consume an argument value: `-p`, `--package`, `--from`, `--extra`, `--with`
- When encountering these flags, skip both the flag AND the next token
- Continue until finding a token that doesn't start with `-` and isn't a flag value

**Compound commands with script extraction** (include subcommand AND script name for `run`):
- `npm run <script>` → tool name is `npm run:<script>` (e.g., `npm run lint` → `npm run:lint`)
- `yarn run <script>` → tool name is `yarn run:<script>`
- `pnpm run <script>` → tool name is `pnpm run:<script>`
- `npm test`, `npm start` → tool name is `npm test`, `npm start` (no script extraction)
- `yarn test`, `yarn lint` → tool name is `yarn test`, `yarn lint`

**Compound commands** (include subcommand in tool name):
- `go` followed by `test`, `build`, `run`, `vet` → tool name is `go <subcommand>`
- `cargo` followed by `test`, `build`, `run`, `clippy`, `fmt` → tool name is `cargo <subcommand>`

6. If no wrapper rule matches, use the first token as tool name

### Examples

| Command | Extracted Tool Name | Rule Applied |
|---------|---------------------|--------------|
| `pytest` | `pytest` | No wrapper |
| `npx eslint .` | `eslint` | Single-token wrapper |
| `npx -p eslint eslint .` | `eslint` | Wrapper + skip flags |
| `uvx ruff check .` | `ruff` | Single-token wrapper |
| `uvx --from ruff ruff check` | `ruff` | Wrapper + skip flags |
| `NODE_ENV=test jest` | `jest` | Env var skipped |
| `python -m pytest` | `pytest` | Multi-token wrapper |
| `python3 -m black --check .` | `black` | Multi-token wrapper |
| `uv run pytest` | `pytest` | Multi-token wrapper |
| `uv run --extra dev pytest` | `pytest` | Multi-token wrapper + skip flags |
| `poetry run ruff check .` | `ruff` | Multi-token wrapper |
| `go test ./...` | `go test` | Compound command |
| `go build -o bin/app` | `go build` | Compound command |
| `cargo test` | `cargo test` | Compound command |
| `cargo clippy` | `cargo clippy` | Compound command |
| `npm test` | `npm test` | Compound command |
| `npm run lint` | `npm run:lint` | Script extraction |
| `npm run test:ci` | `npm run:test:ci` | Script extraction |
| `yarn lint` | `yarn lint` | Compound command |
| `yarn run build` | `yarn run:build` | Script extraction |
| `golangci-lint run` | `golangci-lint` | No wrapper |
| `/usr/local/bin/ruff check` | `ruff` | Path stripped |
| `go test ./... && gocover-cobertura` | `go test` | First valid command |
| `export NODE_ENV=test && npm test` | `npm test` | Shell built-in skipped |
| `set -e && pytest` | `pytest` | Shell built-in skipped |

### Limitations

- **Heuristic-based**: The tool name extraction relies on hardcoded lists of wrappers, flags, and shell built-ins. Custom wrappers with non-standard flag syntax may produce incorrect tool names.
- **Best effort**: If extraction fails or produces an unexpected result, the tool name is used for logging only and doesn't affect validation behavior. Users can override by ensuring their commands start with the actual tool.
- **Security note**: Commands are executed with `shell=True` from trusted `mala.yaml` content. Never inject unsanitized user input into command strings. Use `shlex.quote()` if dynamic values are ever added in future.

## Technical Context

### Architecture

The configuration system introduces a new layer between Mala's orchestration logic and the command execution infrastructure. Currently, `build_validation_spec()` constructs a `ValidationSpec` with hard-coded Python commands. This will be replaced by a configuration loader that reads `mala.yaml`, optionally merges it with a preset, and produces the same `ValidationSpec` structure.

The existing `CommandRunner`, `ValidationResult`, and quality gate logic remain unchanged - they already operate on abstract commands. The key change is moving from static Python-specific definitions to configuration-driven command generation.

### Command Format Migration

**Current implementation**: `ValidationCommand.command` is `list[str]`, executed via `subprocess.run(cmd, shell=False)`.

**New implementation**: Commands are stored as shell strings in `mala.yaml` and executed via `subprocess.run(cmd, shell=True)`.

The config loader will:
1. Store the original shell string for display/logging
2. Execute commands with `shell=True` to support pipes, redirections, and env vars
3. Security note: `mala.yaml` is trusted user configuration, not untrusted input

### Cache Invalidation Rules

Mala uses caching to avoid redundant work. There are two distinct cache types:

1. **Setup cache**: Tracks whether `setup` command needs to re-run (e.g., `npm install`)
2. **Lint evidence cache**: Tracks whether lint/format/typecheck commands were run in the current session (not persistent across git state changes)

**Current behavior (preserved)**: The existing `LintCache` invalidates whenever git state changes (HEAD moves or working tree changes). This means lint/format/typecheck ALWAYS re-run when any file changes. The cache only prevents re-running the same commands multiple times within a single validation session.

| Changed File | Triggers Validation | Re-runs Lint/Format/Typecheck | Re-runs Setup |
|--------------|---------------------|-------------------------------|---------------|
| Matches `code_patterns` | Yes | Yes (code changed) | No |
| Matches `config_files` | Yes | Yes (config changed) | No |
| Matches `setup_files` | Yes | Yes | Yes (deps changed) |
| `mala.yaml` itself | Yes | Yes | Yes |

**The `config_files` distinction**: While all file changes trigger lint re-runs, `config_files` changes are specifically logged to explain WHY the lint might produce different results (e.g., "ruff.toml changed, re-running lint with new rules").

### Key Components

- **Configuration Loader**: Reads and parses `mala.yaml` from the repository root, validates against the schema, and produces a normalized configuration object

- **Preset Registry**: Scans `src/domain/validation/presets/` at import time, loads all `*.yaml` files, and provides lookup by preset name

- **Configuration Merger**: Combines a base preset with user overrides using the merge rules defined above

- **Validation Spec Builder**: Transforms the merged configuration into a `ValidationSpec` with `ValidationCommand` instances for each defined command kind

- **Coverage Parser**: Parses Cobertura XML format (MVP). Architecture should allow adding JSON/LCOV parsers in future

- **Baseline Coverage Service**: Must be generalized to use configured setup and test commands instead of hard-coded Python commands when refreshing coverage baselines

- **Code Pattern Matcher**: Evaluates file paths against user-defined glob patterns to determine if changes are code-relevant

- **Tool Name Extractor**: Implements the extraction algorithm to derive tool names from command strings

### Integration Points

- **Validation spec construction**: Replace hard-coded command building in `build_validation_spec()` with configuration-driven construction
- **Quality gate messaging**: Derive tool names from configured commands rather than the hard-coded `KIND_TO_NAME` mapping
- **Lint cache detection**: Auto-derive tool detection patterns from extracted tool names instead of hard-coded `LINT_COMMAND_PATTERNS`
- **RepoType detection**: Replace Python-specific file detection with configuration presence check
- **Coverage parsing**: Use configured format to select parser (XML only for MVP)
- **Baseline coverage refresh**: Generalize `BaselineCoverageService` to execute configured commands

## User Experience

### Primary Flow

1. User creates `mala.yaml` in their repository root
2. User either specifies a preset to extend (e.g., `preset: go`) or defines commands directly
3. User optionally configures `code_patterns` and `coverage` settings
4. Mala reads the configuration on startup and builds the validation spec
5. Validation runs using the configured commands
6. Results are reported using tool names derived from the configuration

### Error States

- **Missing config file**: Mala terminates with error: "mala.yaml not found. Mala requires a configuration file to run. See documentation for setup."
- **Invalid YAML syntax**: Mala terminates with error identifying the syntax error location (line/column)
- **Unknown preset**: Mala terminates with error: "Unknown preset 'foo'. Available presets: python-uv, node-npm, go, rust"
- **Unknown command kind**: Mala terminates with error listing valid command kinds (setup, test, lint, format, typecheck, e2e)
- **Invalid coverage config**: Mala terminates with error: "Coverage enabled but missing required field 'format'" (or file, threshold)
- **Unsupported coverage format**: Mala terminates with error: "Unsupported coverage format 'lcov'. Supported formats: xml"
- **Coverage file not found**: Mala reports coverage evaluation error when enabled but file is missing after test run
- **Command execution failure**: Failure is reported using the tool name extracted from the command string

### Edge Cases

- **Empty commands section with preset**: Use all preset commands unchanged
- **Command set to null**: Explicitly disable that command even if preset defines it
- **Command omitted entirely**: Inherit from preset, or skip if no preset
- **No commands at all (no preset, empty/missing commands)**: Config validation error: "At least one command must be defined. Specify a preset or define commands directly."
- **All commands explicitly set to null**: Config validation error (same as above - no runnable commands)
- **Command set to empty string `""`**: Config validation error: "Command cannot be empty string. Use null to disable."
- **No code_patterns defined**: Use preset's patterns if available, otherwise treat all files as code
- **Empty code_patterns list**: Treat all files as code (same as no patterns)
- **Coverage section present but empty**: Error - if coverage key exists, all required fields must be specified
- **Coverage enabled but test command is null/missing**: Config validation error at startup: "Coverage requires a test command to generate coverage data"

## Open Questions

- How should projects with multiple languages (monorepos) be handled in future iterations?
- Should the preset directory be configurable or always relative to the package?

## Decisions Made

- **Backwards compatibility**: Require `mala.yaml` - no auto-fallback to Python defaults. No migration tooling provided. This avoids hidden assumptions and makes configuration explicit.

- **Presets**: Provide built-in presets (python-uv, node-npm, go, rust) stored as external YAML files in `src/domain/validation/presets/`. Users can extend and override.

- **Coverage MVP scope**: Support only Cobertura XML format for MVP. JSON and LCOV formats deferred to follow-up iteration.

- **Coverage configuration**: Optional. When `coverage` section is present, all three fields (format, file, threshold) are required. Omit section entirely to disable.

- **Config location**: `mala.yaml` in repository root for discoverability and simplicity.

- **Command format**: Simple strings executed via shell (`subprocess.run(shell=True)`). Supports pipes, redirections, and environment variable assignments. Easy to author and read.

- **Command execution context**: All commands run with working directory set to the repository root. Mala does not inject environment variables beyond the inherited environment.

- **Shell and OS support**: Commands are executed using the system default shell (`/bin/sh` on POSIX, `cmd.exe` on Windows). The tool name extraction algorithm uses POSIX `shlex` parsing. **Windows support is limited**: shell built-ins like `export` don't exist, and quoting rules differ. For cross-platform configs, avoid POSIX-specific syntax or provide platform-specific command variants.

- **Empty string commands**: An empty string (`""`) for a command is treated as a validation error, not as disabling the command. Use `null` to explicitly disable a command.

- **MVP scope**: Full support for all 6 command types (setup, test, lint, format, typecheck, e2e) for parity with current validation flow.

- **Merge semantics**: User values replace preset values at the field level. Lists replace entirely (no append). `null` explicitly disables.

- **Code patterns**: User-defined glob patterns relative to repo root. Uses standard glob syntax with `**` support. List replaces preset list.

- **Tool name extraction**: Skip known wrappers (npx, uvx, yarn, etc.) to extract actual tool name. Maintains list of known wrappers.

- **Error handling**: Fail fast with clear, specific error messages on invalid configuration.

- **Preset storage**: External YAML files in `src/domain/validation/presets/` directory, bundled as package data.
