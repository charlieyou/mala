# Implementation Plan: Language-Agnostic Configuration via mala.yaml

## Context & Goals
- **Spec**: `docs/2026-01-01-language-agnostic-config-spec.md`
- Replace hard-coded Python commands (pytest, ruff, ty, uv) with user-configurable `mala.yaml`
- Provide built-in presets for common stacks: python-uv, node-npm, go, rust
- Support 6 command types (setup, test, lint, format, typecheck, e2e), code pattern matching, and Cobertura XML coverage
- Enable mala to work with any language/toolchain without code changes

## Scope & Non-Goals

### In Scope
- Configuration schema and dataclasses for `mala.yaml`
- Strict config loading with fail-fast validation (eager at startup)
- Preset registry with 4 built-in presets using `importlib.resources`
- Config merging (preset as base, user overrides on top)
- Tool name extraction for quality gate messaging
- Code pattern matching for file-triggered validation
- Per-command timeout support (optional, default 120s)
- Cobertura XML coverage parsing with configurable threshold
- Direct migration from hard-coded Python commands (breaking change)

### Out of Scope (Non-Goals)
- Backwards compatibility fallback to Python defaults when no config exists
- Windows shell compatibility (document as limitation only)
- Hot reload of config during session
- CLI `--config` flag for alternate config paths
- Feature flag or gradual rollout
- Non-Cobertura coverage formats (JSON, LCOV deferred)
- Migration tooling or `mala init` command
- Monorepo support with multiple languages

## Assumptions & Constraints

- `mala.yaml` is trusted configuration; shell commands executed as-is via `shell=True`
- Users have existing validation tooling installed (preset commands reference external tools)
- Config parsed once at startup; `ValidationSpec` cached for session duration
- Breaking change: existing Python repos must add `mala.yaml` to continue working
- PyYAML is available as a dependency (verify in `pyproject.toml`)

### Implementation Constraints
- Extend `src/domain/validation/spec.py` rather than replacing the module
- Use frozen dataclasses with factory methods (pattern from `src/infra/io/config.py`)
- Use `importlib.resources` for preset discovery (works in wheels)
- Shell commands as strings executed with `subprocess.run(cmd, shell=True)`
- No backwards-compatibility shims (per CLAUDE.md)

### Testing Constraints
- 85% coverage threshold enforced at quality gate
- Preset tests: unit tests (mock `importlib.resources`) AND integration tests (real package install)
- Test markers: `unit`, `integration` per project convention
- Test fixtures in `tests/fixtures/mala-configs/`

## Prerequisites

- [ ] Spec reviewed and approved (`docs/2026-01-01-language-agnostic-config-spec.md`)
- [ ] PyYAML available as dependency (verify in `pyproject.toml`)
- [ ] Review existing `ValidationSpec` and `build_validation_spec()` implementation
- [ ] Review existing `CoverageResult` and `parse_coverage_xml()` implementation

## High-Level Approach

1. **Foundation**: Create config dataclasses and YAML loader with strict validation
2. **Presets**: Build preset registry with `importlib.resources` and 4 preset files
3. **Merge Logic**: Implement config merger (preset as base, user overrides on top)
4. **Utilities**: Add tool name extractor and code pattern matcher
5. **Integration**: Update `spec.py` to build `ValidationSpec` from config
6. **Coverage**: Generalize coverage parsing for config-driven thresholds
7. **Quality Gate**: Update lint cache and quality gate for dynamic tool names
8. **Testing**: Comprehensive unit and integration tests

## Detailed Plan

### Task 1: Create Configuration Schema and Dataclasses
- **Goal**: Define the core data structures for validation configuration
- **Covers**: Foundation for AC 1-10
- **Depends on**: Prerequisites
- **Changes**:
  - **New**: `src/domain/validation/config.py`
    - `CommandConfig`: frozen dataclass with `command: str`, `timeout: int | None = None`
      - Supports both string shorthand (`test: "uv run pytest"`) and object form:
        ```yaml
        test:
          command: "uv run pytest"
          timeout: 300
        ```
    - `YamlCoverageConfig`: frozen dataclass with `command: str | None`, `format: str`, `file: str`, `threshold: int`, `timeout: int | None = None` (**Note**: Named `YamlCoverageConfig` to avoid collision with existing `CoverageConfig` in `spec.py` which has different fields: `enabled`, `min_percent`, `branch`, `report_path`)
    - `CommandsConfig`: frozen dataclass with optional fields `setup`, `test`, `lint`, `format`, `typecheck`, `e2e` (each `str | CommandConfig | None`)
    - `ValidationConfig`: frozen dataclass with:
      - `preset: str | None` — preset name to extend
      - `commands: CommandsConfig | None` — nested commands wrapper matching YAML structure
      - `coverage: YamlCoverageConfig | None`
      - `code_patterns: list[str] | None`
      - `config_files: list[str] | None`
      - `setup_files: list[str] | None`
    - `ConfigError`, `PresetNotFoundError`: custom exceptions defined in this module
    - **Note**: The YAML schema uses nested `commands:` wrapper; the loader preserves this structure in `ValidationConfig.commands`
- **Verification**:
  - **New**: `tests/test_validation_config.py`
    - Test dataclass instantiation with all field combinations
    - Test immutability (frozen)
    - Test default values
  - Run: `uv run pytest tests/test_validation_config.py -v`
- **Rollback**: Delete `src/domain/validation/config.py` and `tests/test_validation_config.py`

### Task 2: Implement YAML Config Loader
- **Goal**: Load and validate `mala.yaml` with strict schema checking
- **Covers**: AC 5 (missing config error), AC 6 (invalid YAML error)
- **Depends on**: Task 1
- **Changes**:
  - **New**: `src/domain/validation/config_loader.py`
    - `load_config(repo_path: Path) -> ValidationConfig`: Load `mala.yaml` from repo root
    - `_parse_yaml(content: str) -> dict`: Parse YAML with error handling
    - `_validate_schema(data: dict) -> None`: Validate against expected schema, raise on unknown fields
    - `_build_config(data: dict) -> ValidationConfig`: Convert dict to dataclass
    - `_validate_config(config: ValidationConfig) -> None`: Post-build validation
    - Raise `ConfigError` (new exception from `config.py`) with specific messages for:
      - Missing file: `"mala.yaml not found in {repo_path}. Mala requires a configuration file to run."`
      - Invalid YAML syntax: `"Invalid YAML syntax in mala.yaml: {details}"`
      - Unknown field: `"Unknown field '{field}' in mala.yaml"`
      - Invalid type: `"Field '{field}' must be {expected_type}, got {actual_type}"`
      - No commands: `"At least one command must be defined. Specify a preset or define commands directly."` (enforced in `_validate_config()`)
      - Empty command string: `"Command cannot be empty string. Use null to disable."`
      - Unsupported coverage format: `"Unsupported coverage format '{format}'. Supported formats: xml"` (enforce `coverage.format == "xml"` for MVP)
- **Verification**:
  - **New**: `tests/test_config_loader.py`
    - Test successful load with valid config
    - Test missing file error (exact message)
    - Test invalid YAML syntax error
    - Test unknown field rejection
    - Test type validation
    - Test "no commands defined" error (no preset, no commands)
    - Test "all commands null" error
    - Test empty command string error
    - Test unsupported coverage format error (e.g., "lcov")
  - **New**: `tests/fixtures/mala-configs/valid-minimal.yaml`
  - **New**: `tests/fixtures/mala-configs/valid-full.yaml`
  - **New**: `tests/fixtures/mala-configs/invalid-syntax.yaml`
  - **New**: `tests/fixtures/mala-configs/invalid-unknown-field.yaml`
  - Run: `uv run pytest tests/test_config_loader.py -v`
- **Rollback**: Delete `src/domain/validation/config_loader.py`, `tests/test_config_loader.py`, and fixture files

### Task 3: Create Preset Registry
- **Goal**: Provide built-in presets discoverable via `importlib.resources`
- **Covers**: AC 1 (Go preset), AC 2 (Node preset), AC 3 (preset + override)
- **Depends on**: Task 1, Task 2 (for `_build_config()` function)
- **Changes**:
  - **New**: `src/domain/validation/preset_registry.py`
    - `PresetRegistry` class:
      - `get(name: str) -> ValidationConfig`: Load and return preset config
        - Uses `_load_preset_yaml()` to get raw dict
        - Reuses `_build_config()` from `config_loader.py` to convert dict to `ValidationConfig`
      - `list_presets() -> list[str]`: Return available preset names
      - `_load_preset_yaml(name: str) -> dict`: Use `importlib.resources` to read preset file
    - Raise `PresetNotFoundError` (imported from `config.py`) for unknown presets with message: `"Unknown preset '{name}'. Available presets: python-uv, node-npm, go, rust"`
  - **New**: `src/domain/validation/presets/__init__.py` (empty, for package)
  - **New**: `src/domain/validation/presets/python-uv.yaml`
    ```yaml
    # Python project using uv package manager
    # Commands: uv sync, pytest, ruff check, ruff format --check, ty check
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
  - **New**: `src/domain/validation/presets/node-npm.yaml`
    ```yaml
    # Node.js project using npm
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
  - **New**: `src/domain/validation/presets/go.yaml`
    ```yaml
    # Go project using go modules
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
  - **New**: `src/domain/validation/presets/rust.yaml`
    ```yaml
    # Rust project using cargo
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
  - Modify: `pyproject.toml` — Add package data inclusion for presets (use hatch build configuration since hatchling is the build backend):
    ```toml
    [tool.hatch.build.targets.wheel]
    packages = ["src"]
    include = ["src/domain/validation/presets/*.yaml"]
    ```
  - Also add PyYAML dependency if not already present:
    ```toml
    dependencies = [
      ...
      "pyyaml>=6.0",
    ]
    ```
- **Verification**:
  - **New**: `tests/test_preset_registry.py`
    - Unit tests (mock `importlib.resources`):
      - Test `get()` returns valid `ValidationConfig`
      - Test `list_presets()` returns all 4 presets
      - Test `PresetNotFoundError` for unknown preset
    - Integration tests (real package):
      - Test actual preset file loading after package install
      - Test preset YAML syntax validity
  - Run: `uv run pytest tests/test_preset_registry.py -v`
- **Rollback**: Delete `preset_registry.py`, `presets/` directory, tests, and revert `pyproject.toml`

### Task 4: Implement Config Merger
- **Goal**: Merge preset config with user overrides (user wins)
- **Covers**: AC 3 (preset + custom override), AC 4 (partial config)
- **Depends on**: Task 1, Task 3
- **Changes**:
  - **New**: `src/domain/validation/config_merger.py`
    - `merge_configs(preset: ValidationConfig | None, user: ValidationConfig) -> ValidationConfig`
    - Merge rules (per spec):
      - If no preset, return user config as-is
      - For command fields (setup, test, lint, format, typecheck, e2e): user value replaces preset entirely if present; `null` disables even if preset defines it; omitted commands inherit from preset
      - For coverage: user replaces preset entirely if present
      - For list fields (code_patterns, config_files, setup_files): user **replaces** preset entirely (no merge/append)
      - `None`/missing fields in user config inherit from preset
- **Verification**:
  - **New**: `tests/test_config_merger.py`
    - Test no preset (user config returned unchanged)
    - Test preset with no user overrides (preset returned)
    - Test user override replaces preset command
    - Test user `null` disables preset command
    - Test user `None`/omitted inherits preset value
    - Test list fields replace (not extend)
    - Test coverage override replaces entirely
  - Run: `uv run pytest tests/test_config_merger.py -v`
- **Rollback**: Delete `src/domain/validation/config_merger.py` and `tests/test_config_merger.py`

### Task 5: Implement Tool Name Extractor
- **Goal**: Extract human-readable tool name from shell command for quality gate messaging
- **Covers**: AC 10 (npx eslint -> eslint)
- **Depends on**: None (utility module)
- **Changes**:
  - **New**: `src/domain/validation/tool_name_extractor.py`
    - `extract_tool_name(command: str) -> str`
    - Algorithm (per spec):
      1. If command contains shell operators (`&&`, `||`, `|`, `;`), try each segment until valid tool found
      2. Parse segment via `shlex.split`; if parsing fails, use first whitespace-delimited word
      3. Skip env var assignments (tokens containing `=` before command)
      4. Skip shell built-ins (`export`, `set`, `cd`, etc.)
      5. Strip path prefixes (e.g., `/usr/bin/eslint` -> `eslint`)
      6. Apply wrapper rules:
         - Single-token wrappers: `npx`, `bunx`, `uvx`, `pipx` — skip wrapper and flags, use next positional
         - Multi-token wrappers: `python -m`, `python3 -m`, `uv run`, `poetry run` — skip sequence and flags
         - Compound commands: `go test`, `cargo clippy`, `npm test` — include subcommand
         - Script extraction: `npm run lint` -> `npm run:lint`
      7. Fallback: return first token (best effort), log warning
    - Examples from spec:
      - `"npx eslint ."` -> `"eslint"`
      - `"uvx ruff check ."` -> `"ruff"`
      - `"uv run pytest"` -> `"pytest"`
      - `"go test ./..."` -> `"go test"`
      - `"cargo clippy"` -> `"cargo clippy"`
      - `"npm run lint"` -> `"npm run:lint"`
- **Verification**:
  - **New**: `tests/test_tool_name_extractor.py`
    - Test all spec examples (full table)
    - Test path stripping
    - Test unknown wrapper (fallback behavior)
    - Test empty/malformed commands (best-effort, log warning)
  - Run: `uv run pytest tests/test_tool_name_extractor.py -v`
- **Rollback**: Delete `src/domain/validation/tool_name_extractor.py` and `tests/test_tool_name_extractor.py`

### Task 6: Implement Code Pattern Matcher
- **Goal**: Match file paths against glob patterns to determine if validation should run
- **Covers**: AC 9 (custom code_patterns)
- **Depends on**: None (utility module)
- **Changes**:
  - **New**: `src/domain/validation/code_pattern_matcher.py`
    - `glob_to_regex(pattern: str) -> re.Pattern`
      - Convert glob pattern to regex (`*` matches non-slash, `**` matches anything including `/`)
      - On invalid pattern, treat as literal string (per user decision 15), log warning
    - `matches_pattern(path: str, pattern: str) -> bool`
      - Filename-only patterns (no `/`): match against `os.path.basename(path)`
      - Path patterns (contain `/`): match against full relative path
    - `filter_matching_files(files: list[str], patterns: list[str]) -> list[str]`
      - Return files matching any pattern
      - Empty patterns list -> matches everything
- **Verification**:
  - **New**: `tests/test_code_pattern_matcher.py`
    - Test `*.py` matches `foo.py`, not `foo.js`
    - Test `src/*.py` matches `src/foo.py`, not `src/sub/foo.py`
    - Test `src/**/*.py` matches `src/foo.py` and `src/sub/deep/file.py`
    - Test `**/test_*.py` matches `test_main.py` and `tests/test_utils.py`
    - Test empty patterns matches everything
    - Test invalid pattern treated as literal
    - Test `filter_matching_files`
  - Run: `uv run pytest tests/test_code_pattern_matcher.py -v`
- **Rollback**: Delete `src/domain/validation/code_pattern_matcher.py` and `tests/test_code_pattern_matcher.py`

### Task 7: Create Config-Driven Spec Builder
- **Goal**: Replace hard-coded Python commands with config-driven `ValidationSpec` construction
- **Covers**: AC 1-4 (all language command execution)
- **Depends on**: Tasks 1-6
- **Changes**:
  - Modify: `src/domain/validation/spec.py`
    - Add imports for config modules
    - Modify `build_validation_spec(repo_path: Path) -> ValidationSpec`:
      - Call `load_config(repo_path)` eagerly at startup
      - If `preset` specified, load via `PresetRegistry.get()` and merge
      - Build `ValidationCommand` instances from merged config
      - Use 120s default timeout, override with per-command timeout if specified
      - Set `code_patterns`, `config_files`, `setup_files` on `ValidationSpec`
    - Extend `ValidationSpec` dataclass:
      - Add `code_patterns: list[str] = field(default_factory=list)`
      - Add `config_files: list[str] = field(default_factory=list)`
      - Add `setup_files: list[str] = field(default_factory=list)`
    - Modify `ValidationCommand`:
      - Change `command: list[str]` to `command: str` (shell string)
      - Add `shell: bool = True` field
      - Add `timeout: int = 120` field
    - Remove hard-coded Python command logic
    - Remove `RepoType` enum and detection (no longer needed—config determines commands)
  - Modify: `src/domain/validation/__init__.py`
    - Remove `RepoType` from re-exports (lines 50, 87) since it's being removed from `spec.py`
- **Verification**:
  - Update: `tests/test_spec.py`
    - Remove tests for `RepoType` detection
    - Add tests for config-driven spec building
    - Test `ValidationSpec` with `code_patterns`, `config_files`, `setup_files`
    - Test `ValidationCommand` with `shell=True` and timeout
  - Update all other tests constructing `ValidationCommand`:
    - `tests/test_validation.py` — Update ValidationCommand construction
    - `tests/test_orchestrator.py` — Update ValidationCommand construction
    - `tests/test_quality_gate.py` — Update ValidationCommand construction
    - `tests/test_spec_workspace.py` — Update ValidationCommand construction
    - `tests/test_gate_runner.py` — Update ValidationCommand construction
    - Search for any test helper fixtures/builders that construct ValidationCommand
  - **New**: `tests/fixtures/mala-configs/go-project.yaml` (for AC 1)
  - **New**: `tests/fixtures/mala-configs/node-project.yaml` (for AC 2)
  - **New**: `tests/fixtures/mala-configs/preset-override.yaml` (for AC 3)
  - **New**: `tests/fixtures/mala-configs/partial-config.yaml` (for AC 4)
  - **New**: `tests/fixtures/mala-configs/command-with-timeout.yaml` — Test per-command timeout
    ```yaml
    commands:
      test:
        command: "go test -v ./..."
        timeout: 300  # 5 minute timeout
      lint: "golangci-lint run"  # String shorthand, uses default 120s
    ```
  - Run: `uv run pytest tests/test_spec.py -v`
- **Rollback**: Revert `src/domain/validation/spec.py`, revert `tests/test_spec.py`, delete fixture files

### Task 8: Update Coverage Parsing for Config
- **Goal**: Use coverage config for file path, format, and threshold
- **Covers**: AC 7 (coverage parsing), AC 8 (no coverage section)
- **Depends on**: Task 7
- **Changes**:
  - Modify: `src/domain/validation/coverage.py`
    - Update `parse_coverage_xml` signature: `parse_coverage_xml(coverage_config: YamlCoverageConfig, repo_path: Path) -> CoverageResult`
      - Use `coverage_config.file` for XML path (instead of hardcoded `coverage.xml`)
      - Use `coverage_config.threshold` for pass/fail determination
      - Keep existing Cobertura XML parsing logic (uses `<coverage line-rate>` attribute)
    - Update all call sites of `parse_coverage_xml`:
      - `src/domain/validation/coverage.py:228` — `parse_and_check_coverage()` wrapper
      - `src/domain/validation/coverage.py:249` — `get_baseline_coverage()`
      - `src/domain/validation/spec_result_builder.py` — `_check_coverage()` method uses `CoverageConfig`; update to use `YamlCoverageConfig`
      - `tests/test_coverage.py` — All test calls to this function
    - Handle missing coverage config: skip coverage evaluation entirely (return `None`)
    - If coverage file not found after test run, validation fails with clear error
- **Verification**:
  - Update: `tests/test_coverage.py`
    - Test with `YamlCoverageConfig` specifying file and threshold
    - Test threshold enforcement (pass at threshold, fail below)
    - Test behavior when no coverage config (return `None` or skip)
    - Test missing coverage file error
    - Update all existing tests to use new signature
  - Run: `uv run pytest tests/test_coverage.py -v`
- **Rollback**: Revert `src/domain/validation/coverage.py` and `tests/test_coverage.py`

### Task 9: Generalize BaselineCoverageService
- **Goal**: Support config-driven coverage command for baseline refresh
- **Covers**: AC 7 (coverage command execution)
- **Depends on**: Task 8
- **Changes**:
  - Modify: `src/domain/validation/coverage.py`
    - Update `BaselineCoverageService` to accept `YamlCoverageConfig` (not the existing `CoverageConfig` in spec.py which has different fields)
    - Use `coverage_config.command` for running coverage (required for baseline refresh per user decision 14)
    - Use `coverage_config.timeout` (or default 120s)
    - If no coverage config, baseline refresh is not available
    - If `coverage.command` execution fails, overall validation fails (per user decision 7)
- **Verification**:
  - Update: `tests/test_coverage.py`
    - Test baseline refresh with custom coverage command
    - Test failure when `coverage.command` fails
    - Test behavior without coverage config (baseline refresh unavailable)
  - Run: `uv run pytest tests/test_coverage.py -v`
- **Rollback**: Revert `BaselineCoverageService` changes

### Task 10: Update Lint Cache Detection Patterns
- **Goal**: Use tool name extractor for dynamic lint tool detection
- **Covers**: Supports AC 1, 2 (lint tools from any language)
- **Depends on**: Task 5, Task 7
- **Changes**:
  - Modify: `src/infra/hooks/lint_cache.py`
    - Remove hard-coded `LINT_COMMAND_PATTERNS` (ruff, ty only)
    - Update `LintCache` to accept lint/typecheck tool names from `ValidationSpec`
    - Use `extract_tool_name()` to determine cache key from command
    - Cache invalidation based on extracted tool name
- **Verification**:
  - Update: `tests/test_lint_cache.py`
    - Test cache with eslint command
    - Test cache with golangci-lint command
    - Test cache key extraction
  - Run: `uv run pytest tests/test_lint_cache.py -v`
- **Rollback**: Revert `src/infra/hooks/lint_cache.py` and `tests/test_lint_cache.py`

### Task 11: Update Quality Gate Messaging
- **Goal**: Use extracted tool names in quality gate output
- **Covers**: AC 10 (tool name display)
- **Depends on**: Task 5, Task 7
- **Changes**:
  - Modify: `src/domain/quality_gate.py`
    - Remove hard-coded `KIND_TO_NAME` mapping
    - Use `extract_tool_name(command)` to get display name
    - Update `ValidationEvidence` to store tool name
    - Update quality gate messages to use extracted names
- **Verification**:
  - Update: `tests/test_quality_gate.py`
    - Test quality gate output shows `"eslint"` not `"npx eslint ."`
    - Test quality gate output shows `"pytest"` not `"uv run pytest"`
  - Run: `uv run pytest tests/test_quality_gate.py -v`
- **Rollback**: Revert `src/domain/quality_gate.py` and `tests/test_quality_gate.py`

### Task 12: Update Command Execution for Shell Mode
- **Goal**: Execute shell string commands with `shell=True`
- **Covers**: AC 1-4 (command execution)
- **Depends on**: Task 7
- **Changes**:
  - Modify: `src/infra/tools/command_runner.py` — Primary command execution site
    - Update `CommandRunner.run()` and `CommandRunner.run_async()` to accept shell string commands
    - Add `shell: bool = False` parameter (default False for backwards compatibility)
    - When `shell=True`, pass command as string directly to `subprocess.Popen(cmd, shell=True, ...)`
    - Ensure stdout/stderr capture works with shell mode (already uses text mode)
    - Timeout handling already implemented via process group termination
  - Modify: `src/domain/validation/spec_executor.py` — Uses CommandRunner
    - Update to pass `shell=True` when executing `ValidationCommand` with shell string
    - Pass `timeout=cmd.timeout` to CommandRunner
  - Update call sites in `src/domain/validation/coverage.py` (`BaselineCoverageService` uses CommandRunner)
- **Verification**:
  - Update: `tests/test_command_runner.py`
    - Test shell string execution with `shell=True`
    - Test timeout enforcement with shell commands
    - Test exit code capture from shell commands
  - Run existing tests: `uv run pytest tests/test_command_runner.py tests/test_spec.py -v`
- **Rollback**: Revert changes to command_runner.py, spec_executor.py, coverage.py

### Task 13: Wire Code Patterns into Validation Gating
- **Goal**: Integrate code pattern matcher into change detection to trigger/skip validation
- **Covers**: AC 9 (custom code_patterns filter files correctly)
- **Depends on**: Task 6, Task 7
- **Changes**:
  - Identify change detection location (likely in orchestrator or validation runner)
  - Modify: `src/domain/validation/spec_runner.py` or `src/orchestration/` — wherever file changes are evaluated
    - Import `filter_matching_files` from `code_pattern_matcher.py`
    - When evaluating changed files, use `spec.code_patterns` to filter
    - If no files match any pattern, skip validation (unless `code_patterns` is empty, in which case all files trigger)
    - Changes to `mala.yaml` itself always trigger validation
  - Add: Cache invalidation logic based on `config_files` and `setup_files`:
    - Changes to `config_files` trigger lint/format/typecheck re-run
    - Changes to `setup_files` trigger setup command re-run
- **Verification**:
  - **New**: `tests/test_validation_gating.py`
    - Test: Only `.py` files trigger validation when `code_patterns: ["*.py"]`
    - Test: Non-matching files (e.g., `.md`) don't trigger validation
    - Test: Empty `code_patterns` matches all files
    - Test: `mala.yaml` change always triggers validation
    - Test: `config_files` change triggers lint cache invalidation
    - Test: `setup_files` change triggers setup re-run
  - Run: `uv run pytest tests/test_validation_gating.py -v`
- **Rollback**: Revert changes to validation runner, delete test file

### Task 14: Integration Testing
- **Goal**: End-to-end validation of config-driven workflow
- **Covers**: AC 1-10 (all acceptance criteria)
- **Depends on**: Tasks 1-13
- **Changes**:
  - **New**: `tests/test_validation_config_integration.py`
    - Test: Go project with `mala.yaml` using go preset (AC 1)
    - Test: Node.js project with node-npm preset (AC 2)
    - Test: Python project with python-uv preset + custom test override (AC 3)
    - Test: Minimal config (only setup + test) skips lint/format/typecheck (AC 4)
    - Test: Missing `mala.yaml` fails with clear error (AC 5)
    - Test: Invalid YAML syntax fails with specific error (AC 6)
    - Test: Coverage config with xml format, custom file, threshold (AC 7)
    - Test: No coverage section skips coverage (AC 8)
    - Test: Custom `code_patterns` filter files correctly (AC 9)
    - Test: Tool name extraction in quality gate output (AC 10)
  - Create test fixtures with actual project structures
- **Verification**:
  - Run: `uv run pytest tests/test_validation_config_integration.py -v -m integration`
  - Run full test suite: `uv run pytest -m "unit or integration"`
  - Verify 85% coverage: `uv run pytest --cov --cov-fail-under=85`
- **Rollback**: Delete `tests/test_validation_config_integration.py` and associated fixtures

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes
- **Missing `mala.yaml`**: Fail fast with clear error message: "mala.yaml not found in {repo_path}. Mala requires a configuration file to run." (AC 5)
- **Invalid YAML syntax**: Fail fast with YAML parser error details including line/column (AC 6)
- **Unknown fields in YAML**: Fail fast listing the unknown field name (AC 6)
- **Missing preset**: Fail fast with "Unknown preset '{name}'. Available presets: python-uv, node-npm, go, rust"
- **Empty command string**: Config validation error: "Command cannot be empty string. Use null to disable."
- **No commands at all**: Config validation error: "At least one command must be defined. Specify a preset or define commands directly."
- **All commands explicitly set to null**: Same error as above
- **Coverage command failure**: Overall validation fails (not skipped) per user decision 7
- **Coverage file not found**: Validation fails with error: "Coverage file '{file}' not found after test execution"
- **Invalid glob pattern**: Treat as literal string, log warning, don't fail (per user decision 15)
- **Tool extraction uncertainty**: Best-effort with fallback to raw command, log warning (per user decision 10)
- **Timeout exceeded**: Command fails with timeout error, overall validation fails
- **Coverage without coverage.command**: Baseline refresh not available (per user decision 14)

### Breaking Changes & Compatibility
- **Potential Breaking Changes**:
  - Existing Python repos without `mala.yaml` will fail (previously worked with auto-detection)
  - `ValidationCommand.command` changes from `list[str]` to `str`
  - `ValidationCommand` adds `shell: bool = True` and `timeout: int = 120` fields
  - `ValidationSpec` adds `code_patterns`, `config_files`, `setup_files` fields
  - `RepoType` enum removed

- **Mitigations**:
  - Document migration in release notes with example `mala.yaml` for Python projects
  - Provide clear error message pointing to documentation
  - `python-uv.yaml` preset matches previous hard-coded behavior exactly

- **Rollout Strategy**:
  - Direct replacement (per user decision 8)
  - No feature flag
  - Individual task rollbacks possible via git revert

## Testing & Validation

### Unit Tests
- `tests/test_validation_config.py`: Dataclass construction, immutability, defaults
- `tests/test_config_loader.py`: YAML parsing, schema validation, error messages
- `tests/test_preset_registry.py`: Preset loading (mocked), error handling
- `tests/test_config_merger.py`: Merge rules, inheritance, override semantics
- `tests/test_tool_name_extractor.py`: All spec examples, edge cases
- `tests/test_code_pattern_matcher.py`: Glob conversion, matching logic

### Integration Tests
- `tests/test_preset_registry.py`: Real preset file loading via installed package
- `tests/test_validation_config_integration.py`: Full workflow tests for AC 1-10

### Regression Tests
- Existing `tests/test_spec.py` updated for new behavior
- Existing `tests/test_coverage.py` updated for config-driven coverage
- Existing `tests/test_quality_gate.py` updated for dynamic tool names
- Existing `tests/test_lint_cache.py` updated for dynamic detection

### Manual Verification
- Create test repo with Go project, verify all commands execute
- Create test repo with Node.js project, verify all commands execute
- Test error messages for missing/invalid config
- Verify quality gate output shows correct tool names

### Monitoring / Observability
- Existing orchestrator logging captures command execution
- Config parsing errors logged with full context
- Tool name extraction warnings logged

### Acceptance Criteria Coverage
| Spec AC | Covered By |
|---------|------------|
| AC 1: Go project execution | Task 3 (Preset), Task 7 (Spec Builder), Task 14 (Integration Test) |
| AC 2: Node.js project execution | Task 3 (Preset), Task 7 (Spec Builder), Task 14 (Integration Test) |
| AC 3: Preset + override | Task 4 (Merger), Task 7 (Spec Builder), Task 14 (Integration Test) |
| AC 4: Partial config (skip undefined) | Task 7 (Spec Builder), Task 14 (Integration Test) |
| AC 5: Missing mala.yaml error | Task 2 (Loader), Task 14 (Integration Test) |
| AC 6: Invalid YAML error | Task 2 (Loader), Task 14 (Integration Test) |
| AC 7: Coverage parsing + threshold | Task 8 (Coverage), Task 9 (Baseline), Task 14 (Integration Test) |
| AC 8: No coverage section | Task 8 (Coverage), Task 14 (Integration Test) |
| AC 9: Custom code_patterns | Task 6 (Matcher), Task 7 (Spec Builder), Task 13 (Validation Gating), Task 14 (Integration Test) |
| AC 10: Tool name extraction | Task 5 (Extractor), Task 11 (Quality Gate), Task 14 (Integration Test) |

## Rollback Strategy (Plan-Level)

- **Full rollback steps**:
  1. Revert all commits from this implementation in reverse order
  2. Verify hard-coded Python commands restored in `spec.py`
  3. Verify existing tests pass
  4. No data migration required (config is read-only)

- **Verification of rollback**:
  - Run full test suite: `uv run pytest -m "unit or integration"`
  - Verify Python repo validation works without `mala.yaml`
  - Verify coverage baseline refresh works with hard-coded pytest

- **Cleanup**:
  - Remove any created `mala.yaml` files from test repos
  - No database or persistent state changes

## Open Questions

1. **Coverage format extensibility**: The spec mentions JSON and LCOV as follow-up. Should the coverage parser interface be designed with this in mind now, or defer until those formats are implemented?

2. **Shell detection on Windows**: The spec mentions Windows as a limitation. Should we detect Windows at runtime and emit a warning, or silently proceed and let commands fail naturally?

## Appendix: Key Decisions Made During Synthesis

All key decisions were resolved in the interview phase:

| Decision | Resolution | Source |
|----------|------------|--------|
| Transition strategy | Breaking change, no fallback to Python defaults | User decision 1 |
| Config loading timing | Eager at startup, fail fast on errors | User decision 2 |
| Per-command timeout | Optional field, default 120s | User decision 3, 12 |
| CLI --config flag | Not supported, always use mala.yaml in repo root | User decision 4 |
| ValidationSpec API | Extend with code_patterns, config_files, setup_files | User decision 9 |
| Tool extraction errors | Best-effort with fallback, log warning | User decision 10 |
| Preset style | Commented YAML with documentation | User decision 11 |
| Config hot reload | Not supported, config loaded once at startup | User decision 13 |
| Baseline coverage | Require coverage.command for baseline refresh | User decision 14 |
| Invalid glob patterns | Treat as literals, don't fail | User decision 15 |
| Rollout | Direct replacement, no feature flag | User decision 8 |
| Windows support | Document as limitation only | User decision 6 |
| Coverage failure | Overall validation fails | User decision 7 |
