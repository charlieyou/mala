"""ValidationSpec and related types for mala validation.

This module provides:
- ValidationSpec: defines what validations to run for a given scope
- ValidationCommand: a single command in the validation pipeline
- ValidationContext: immutable context for a validation run
- ValidationArtifacts: record of validation outputs (re-exported from models)
- IssueResolution: how an issue was resolved (re-exported from models)
- Code vs docs classification helper
- Disable list handling for spec omissions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

# Re-export shared types from models for backward compatibility
# Note: Using explicit aliases (X as X) marks these as intentional re-exports
from src.core.models import (
    IssueResolution as IssueResolution,
    ResolutionOutcome as ResolutionOutcome,
    ValidationArtifacts as ValidationArtifacts,
)
from src.domain.validation.config import CommandsConfig

if TYPE_CHECKING:
    from re import Pattern

    from src.domain.validation.config import (
        CommandConfig,
        CustomCommandConfig,
        YamlCoverageConfig,
    )


class ValidationScope(Enum):
    """Scope for validation: per-issue or run-level."""

    PER_ISSUE = "per_issue"
    RUN_LEVEL = "run_level"


class CommandKind(Enum):
    """Kind of validation command."""

    SETUP = "setup"  # Environment setup (uv sync, etc.)
    LINT = "lint"
    FORMAT = "format"
    TYPECHECK = "typecheck"
    TEST = "test"
    E2E = "e2e"
    CUSTOM = "custom"  # User-defined custom commands


# Default timeout for validation commands in seconds
DEFAULT_COMMAND_TIMEOUT = 120

# Command kinds that represent lint-like tools for LintCache
LINT_COMMAND_KINDS: frozenset[CommandKind] = frozenset(
    {CommandKind.LINT, CommandKind.FORMAT, CommandKind.TYPECHECK}
)


@dataclass(frozen=True)
class ValidationCommand:
    """A single command in the validation pipeline.

    Attributes:
        name: Human-readable name (e.g., "pytest", "ruff check").
        command: The command as a shell string.
        kind: Classification of the command type.
        shell: Whether to run command in shell mode. Defaults to True.
        timeout: Command timeout in seconds. Defaults to 120.
        detection_pattern: Compiled regex to detect this command in log evidence.
            If None, falls back to hardcoded patterns in QualityGate.
        use_test_mutex: Whether to wrap with test mutex.
        allow_fail: If True, failure doesn't stop the pipeline.
    """

    name: str
    command: str
    kind: CommandKind
    shell: bool = True
    timeout: int = DEFAULT_COMMAND_TIMEOUT
    detection_pattern: Pattern[str] | None = None
    use_test_mutex: bool = False
    allow_fail: bool = False


@dataclass
class CoverageConfig:
    """Configuration for coverage validation.

    Attributes:
        enabled: Whether coverage is enabled.
        min_percent: Minimum coverage percentage. If None, uses "no decrease"
            mode where coverage must not drop below the baseline stored in
            .coverage-baseline.json.
        branch: Whether to include branch coverage.
        report_path: Path to coverage report file.
    """

    enabled: bool = True
    min_percent: float | None = None
    branch: bool = True
    report_path: Path | None = None


@dataclass
class E2EConfig:
    """Configuration for E2E fixture repo validation.

    Attributes:
        enabled: Whether E2E is enabled.
        fixture_root: Root directory for fixture repos.
        command: Command to run for E2E.
        required_env: Environment variables required for E2E.
    """

    enabled: bool = True
    fixture_root: Path | None = None
    command: list[str] = field(default_factory=list)
    required_env: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationContext:
    """Immutable context for a single validation run.

    Attributes:
        issue_id: The issue ID (None for run-level).
        repo_path: Path to the repository.
        commit_hash: The commit being validated.
        log_path: Path to the Claude session log.
        changed_files: List of files changed in this commit.
        scope: Whether this is per-issue or run-level.
    """

    issue_id: str | None
    repo_path: Path
    commit_hash: str
    changed_files: list[str]
    scope: ValidationScope
    log_path: Path | None = None


@dataclass
class ValidationSpec:
    """Defines what validations to run for a given scope.

    Attributes:
        commands: List of validation commands to run.
        require_clean_git: Whether git working tree must be clean.
        require_pytest_for_code_changes: If code changed, pytest is required.
        coverage: Coverage configuration.
        e2e: E2E configuration.
        scope: The validation scope.
        code_patterns: Glob patterns for code files that trigger validation.
        config_files: Tool config files that invalidate lint/format cache.
        setup_files: Lock/dependency files that invalidate setup cache.
        yaml_coverage_config: Optional YamlCoverageConfig from mala.yaml for
            baseline refresh (contains command and file path).
    """

    commands: list[ValidationCommand]
    scope: ValidationScope
    require_clean_git: bool = True
    require_pytest_for_code_changes: bool = True
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    e2e: E2EConfig = field(default_factory=E2EConfig)
    code_patterns: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    setup_files: list[str] = field(default_factory=list)
    yaml_coverage_config: YamlCoverageConfig | None = None

    def commands_by_kind(self, kind: CommandKind) -> list[ValidationCommand]:
        """Return commands matching the given kind."""
        return [cmd for cmd in self.commands if cmd.kind == kind]

    def extract_lint_tools(self) -> frozenset[str]:
        """Extract lint tool names from this ValidationSpec.

        Extracts tool names from commands with LINT, FORMAT, or TYPECHECK kinds.
        These are the tools that LintCache should recognize and cache.

        Returns:
            Frozenset of lint tool names. Empty frozenset if no lint commands.
        """
        from src.core.tool_name_extractor import extract_tool_name

        lint_tools: set[str] = set()
        for cmd in self.commands:
            if cmd.kind in LINT_COMMAND_KINDS:
                tool_name = extract_tool_name(cmd.command)
                if tool_name:
                    lint_tools.add(tool_name)

        return frozenset(lint_tools)


# Code classification constants
# Paths that are considered code changes
CODE_PATHS = frozenset({"src/", "tests/", "commands/", "src/scripts/"})

# Specific files that are considered code changes
CODE_FILES = frozenset({"pyproject.toml", "uv.lock"})

# Extensions that are considered code changes
CODE_EXTENSIONS = frozenset({".py", ".sh", ".toml", ".yml", ".yaml", ".json"})

# Extensions that are considered documentation
DOC_EXTENSIONS = frozenset({".md", ".rst", ".txt"})


def classify_change(file_path: str) -> Literal["code", "docs"]:
    """Classify a file change as code or docs.

    Code changes require tests + coverage. Doc changes may skip tests.

    Classification rules (per quality-hardening-plan.md):
    - Paths: src/**, tests/**, commands/**, src/scripts/** => code
    - Files: pyproject.toml, uv.lock, .env templates => code
    - Extensions: .py, .sh, .toml, .yml, .yaml, .json => code
    - Extensions: .md, .rst, .txt => docs
    - Unknown extensions outside code paths => docs

    Args:
        file_path: The file path to classify.

    Returns:
        "code" if the file is code, "docs" if documentation.
    """
    path = Path(file_path)

    # Check if it's a known code file
    if path.name in CODE_FILES:
        return "code"

    # Check for .env templates
    if path.name.startswith(".env"):
        return "code"

    # Check if it's under a code path
    path_str = str(path)
    for code_path in CODE_PATHS:
        if path_str.startswith(code_path):
            return "code"

    # Check extension
    suffix = path.suffix.lower()
    if suffix in CODE_EXTENSIONS:
        return "code"
    if suffix in DOC_EXTENSIONS:
        return "docs"

    # Default to docs for unknown types
    return "docs"


def build_validation_spec(
    repo_path: Path,
    scope: ValidationScope | None = None,
    disable_validations: set[str] | None = None,
) -> ValidationSpec:
    """Build a ValidationSpec from config files.

    This function loads the mala.yaml configuration from the repository,
    merges it with any preset configuration, and builds a ValidationSpec.

    Disable values:
    - "post-validate": Skip test commands entirely
    - "run-level-validate": (handled elsewhere, not here)
    - "integration-tests": (handled by config, not here)
    - "coverage": Disable coverage checking
    - "e2e": Disable E2E fixture repo test

    Args:
        repo_path: Path to the repository root directory.
        scope: The validation scope. Defaults to PER_ISSUE.
        disable_validations: Set of validation types to disable.

    Returns:
        A ValidationSpec configured according to the config files.
    """
    from src.domain.validation.config import ConfigError
    from src.domain.validation.config_loader import ConfigMissingError, load_config
    from src.domain.validation.config_merger import merge_configs
    from src.domain.validation.preset_registry import PresetRegistry

    # Use default scope if not specified
    if scope is None:
        scope = ValidationScope.PER_ISSUE

    disable = disable_validations or set()

    # Determine if we should skip tests
    skip_tests = "post-validate" in disable

    # Load config from repo
    # - ConfigMissingError: gracefully return empty spec (optional config)
    # - ConfigError: fail fast (invalid syntax, unknown fields, etc.)
    try:
        user_config = load_config(repo_path)
    except ConfigMissingError:
        # No config file - return empty spec with all validations disabled
        return ValidationSpec(
            commands=[],
            scope=scope,
            require_clean_git=True,
            require_pytest_for_code_changes=True,
            coverage=CoverageConfig(enabled=False),
            e2e=E2EConfig(enabled=False),
            code_patterns=[],
            config_files=[],
            setup_files=[],
        )

    # Load and merge preset if specified
    if user_config.preset is not None:
        registry = PresetRegistry()
        preset_config = registry.get(user_config.preset)
        merged_config = merge_configs(preset_config, user_config)
    else:
        merged_config = user_config

    # Ensure at least one command is defined after merge
    if not merged_config.has_any_command():
        raise ConfigError(
            "At least one command must be defined. "
            "Specify a preset or define commands directly."
        )

    # Resolve commands for scope (run-level may override base commands)
    commands_config = merged_config.commands
    if scope == ValidationScope.RUN_LEVEL:
        commands_config = _apply_command_overrides(
            merged_config.commands, merged_config.run_level_commands
        )

    # Resolve custom_commands for scope (run-level full-replace semantics)
    custom_commands = _apply_custom_commands_override(
        merged_config.custom_commands,
        merged_config.run_level_custom_commands,
        merged_config._fields_set,
        scope,
    )

    # Coverage requires a test command in the effective commands_config
    # This check must happen after _apply_command_overrides to catch cases where
    # run_level_commands.test is explicitly null (disabling the base test command)
    # However, we only raise an error if there's no run_level_commands.test set either -
    # if run_level_commands.test is set, coverage will be generated at run-level
    if merged_config.coverage is not None and commands_config.test is None:
        # Check if run_level_commands.test provides a test command for coverage
        run_level_test_set = (
            merged_config.run_level_commands._fields_set
            and "test" in merged_config.run_level_commands._fields_set
            and merged_config.run_level_commands.test is not None
        )
        if not run_level_test_set:
            raise ConfigError(
                "Coverage requires a test command to generate coverage data."
            )

    # Build commands list from config
    commands: list[ValidationCommand] = []

    if not skip_tests:
        commands = _build_commands_from_config(commands_config, custom_commands)

    # Determine if coverage is enabled
    # Coverage is disabled for PER_ISSUE scope when run_level_commands.test provides
    # a different test command (e.g., with --cov flags). In this pattern, only the
    # run-level test command generates coverage.xml, so per-issue validation should
    # not check coverage.
    # Note: If run_level_commands.test is explicitly null, we don't disable coverage
    # for PER_ISSUE - the intent is to skip tests at run level, not to move coverage
    # to run level.
    run_level_has_different_test_command = (
        merged_config.run_level_commands._fields_set
        and "test" in merged_config.run_level_commands._fields_set
        and merged_config.run_level_commands.test is not None
    )
    coverage_only_at_run_level = (
        run_level_has_different_test_command and scope == ValidationScope.PER_ISSUE
    )
    coverage_enabled = (
        merged_config.coverage is not None
        and "coverage" not in disable
        and not skip_tests
        and not coverage_only_at_run_level
    )

    # Build coverage config
    coverage_config = CoverageConfig(
        enabled=coverage_enabled,
        min_percent=merged_config.coverage.threshold
        if merged_config.coverage
        else None,
        branch=True,
        report_path=(
            Path(merged_config.coverage.file) if merged_config.coverage else None
        ),
    )

    # Configure E2E
    # Use commands_config which has run_level_commands overrides applied
    e2e_enabled = (
        scope == ValidationScope.RUN_LEVEL
        and "e2e" not in disable
        and commands_config.e2e is not None
    )
    e2e_config = E2EConfig(
        enabled=e2e_enabled,
        required_env=[],
    )

    return ValidationSpec(
        commands=commands,
        scope=scope,
        require_clean_git=True,
        require_pytest_for_code_changes=True,
        coverage=coverage_config,
        e2e=e2e_config,
        code_patterns=list(merged_config.code_patterns),
        config_files=list(merged_config.config_files),
        setup_files=list(merged_config.setup_files),
        yaml_coverage_config=merged_config.coverage if coverage_enabled else None,
    )


def _apply_command_overrides(
    base: CommandsConfig, overrides: CommandsConfig
) -> CommandsConfig:
    """Apply run-level command overrides on top of base commands."""

    def is_explicit(field_name: str, value: CommandConfig | None) -> bool:
        if overrides._fields_set:
            return field_name in overrides._fields_set
        return value is not None

    def pick(
        field_name: str,
        base_cmd: CommandConfig | None,
        override_cmd: CommandConfig | None,
    ) -> CommandConfig | None:
        return override_cmd if is_explicit(field_name, override_cmd) else base_cmd

    return CommandsConfig(
        setup=pick("setup", base.setup, overrides.setup),
        test=pick("test", base.test, overrides.test),
        lint=pick("lint", base.lint, overrides.lint),
        format=pick("format", base.format, overrides.format),
        typecheck=pick("typecheck", base.typecheck, overrides.typecheck),
        e2e=pick("e2e", base.e2e, overrides.e2e),
        _fields_set=overrides._fields_set,
    )


def _apply_custom_commands_override(
    base: dict[str, CustomCommandConfig],
    override: dict[str, CustomCommandConfig] | None,
    fields_set: frozenset[str],
    scope: ValidationScope,
) -> dict[str, CustomCommandConfig]:
    """Apply run-level custom_commands override with full-replace semantics.

    Run-level override behavior:
    - If run_level_custom_commands not in fields_set: use base (repo-level)
    - If run_level_custom_commands is explicitly set to {} (empty dict): no custom commands
    - If run_level_custom_commands is explicitly set to {cmd_a: ...}: only run cmd_a
      (does NOT inherit cmd_b from base - full replace, not merge)
    - If run_level_custom_commands is explicitly set to null: same as not set (use base)

    For PER_ISSUE scope, always use base (run-level override only applies to RUN_LEVEL).

    For programmatic ValidationConfig instances where _fields_set is empty,
    a non-None override value is treated as explicit (consistent with
    _apply_command_overrides behavior).

    Args:
        base: Repo-level custom_commands dict.
        override: Run-level custom_commands dict (None if not set or explicitly null).
        fields_set: Set of field names explicitly set in config.
        scope: The validation scope.

    Returns:
        The effective custom_commands dict for this scope.
    """
    # For PER_ISSUE scope, always use repo-level custom_commands
    if scope == ValidationScope.PER_ISSUE:
        return base

    # For RUN_LEVEL scope, check if run_level_custom_commands was explicitly set.
    # When fields_set is non-empty (from_dict), check membership.
    # When fields_set is empty (programmatic), fall back to checking if override is not None.
    if fields_set:
        is_explicit = "run_level_custom_commands" in fields_set
    else:
        is_explicit = override is not None

    if not is_explicit:
        # Not set - use repo-level
        return base

    # Explicitly set - could be {} (empty), null, or {cmd_a: ...}
    # If override is None, it means explicitly null - treat as "use base"
    # If override is {} (empty dict), it means disable all custom commands
    # If override is {cmd_a: ...}, use only those commands (full replace)
    if override is None:
        return base

    return override


def _build_commands_from_config(
    config: CommandsConfig,
    custom_commands: dict[str, CustomCommandConfig] | None = None,
) -> list[ValidationCommand]:
    """Build ValidationCommand list from a CommandsConfig.

    Pipeline order: setup → format → lint → typecheck → custom_a → custom_b → test → e2e

    Args:
        config: The command configuration.
        custom_commands: Optional dict of custom commands to insert after typecheck.

    Returns:
        List of ValidationCommand instances.
    """
    from src.core.tool_name_extractor import extract_tool_name

    commands: list[ValidationCommand] = []
    cmds = config

    # Setup command
    if cmds.setup is not None:
        commands.append(
            ValidationCommand(
                name="setup",
                command=cmds.setup.command,
                kind=CommandKind.SETUP,
                timeout=cmds.setup.timeout or DEFAULT_COMMAND_TIMEOUT,
                detection_pattern=re.compile(
                    _tool_name_to_pattern(extract_tool_name(cmds.setup.command)),
                    re.IGNORECASE,
                ),
            )
        )

    # Format command
    if cmds.format is not None:
        commands.append(
            ValidationCommand(
                name="format",
                command=cmds.format.command,
                kind=CommandKind.FORMAT,
                timeout=cmds.format.timeout or DEFAULT_COMMAND_TIMEOUT,
                detection_pattern=re.compile(
                    _tool_name_to_pattern(extract_tool_name(cmds.format.command)),
                    re.IGNORECASE,
                ),
            )
        )

    # Lint command
    if cmds.lint is not None:
        commands.append(
            ValidationCommand(
                name="lint",
                command=cmds.lint.command,
                kind=CommandKind.LINT,
                timeout=cmds.lint.timeout or DEFAULT_COMMAND_TIMEOUT,
                detection_pattern=re.compile(
                    _tool_name_to_pattern(extract_tool_name(cmds.lint.command)),
                    re.IGNORECASE,
                ),
            )
        )

    # Typecheck command
    if cmds.typecheck is not None:
        commands.append(
            ValidationCommand(
                name="typecheck",
                command=cmds.typecheck.command,
                kind=CommandKind.TYPECHECK,
                timeout=cmds.typecheck.timeout or DEFAULT_COMMAND_TIMEOUT,
                detection_pattern=re.compile(
                    _tool_name_to_pattern(extract_tool_name(cmds.typecheck.command)),
                    re.IGNORECASE,
                ),
            )
        )

    # Custom commands (inserted after typecheck, before test)
    # Dict iteration order is preserved in Python 3.7+
    if custom_commands:
        for name, custom_cmd in custom_commands.items():
            commands.append(
                ValidationCommand(
                    name=name,
                    command=custom_cmd.command,
                    kind=CommandKind.CUSTOM,
                    timeout=custom_cmd.timeout or DEFAULT_COMMAND_TIMEOUT,
                    allow_fail=custom_cmd.allow_fail,
                    detection_pattern=re.compile(
                        _tool_name_to_pattern(extract_tool_name(custom_cmd.command)),
                        re.IGNORECASE,
                    ),
                )
            )

    # Test command
    if cmds.test is not None:
        commands.append(
            ValidationCommand(
                name="test",
                command=cmds.test.command,
                kind=CommandKind.TEST,
                timeout=cmds.test.timeout or DEFAULT_COMMAND_TIMEOUT,
                detection_pattern=re.compile(
                    _tool_name_to_pattern(extract_tool_name(cmds.test.command)),
                    re.IGNORECASE,
                ),
            )
        )

    # E2E command (not added to regular command list, handled separately)
    # E2E is controlled by E2EConfig.enabled, not by command presence

    return commands


def _tool_name_to_pattern(tool_name: str) -> str:
    """Convert a tool name to a regex pattern.

    Args:
        tool_name: The tool name to convert.

    Returns:
        Regex pattern string to match the tool name.
    """
    if not tool_name:
        return r"$^"  # Matches nothing
    # Escape special regex characters in the tool name
    escaped = re.escape(tool_name)
    return rf"\b{escaped}\b"


def extract_lint_tools_from_spec(spec: ValidationSpec | None) -> frozenset[str] | None:
    """Extract lint tool names from a ValidationSpec.

    Extracts tool names from commands with LINT, FORMAT, or TYPECHECK kinds.
    These are the tools that LintCache should recognize and cache.

    This is a convenience wrapper around ValidationSpec.extract_lint_tools()
    that handles None specs.

    Args:
        spec: The ValidationSpec to extract from. If None, returns None.

    Returns:
        Frozenset of lint tool names, or None if spec is None or has no
        lint commands (allowing LintCache to use defaults).
    """
    if spec is None:
        return None

    lint_tools = spec.extract_lint_tools()
    return lint_tools if lint_tools else None
