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

if TYPE_CHECKING:
    from re import Pattern

    from src.domain.validation.config import ValidationConfig, YamlCoverageConfig


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


# Default timeout for validation commands in seconds
DEFAULT_COMMAND_TIMEOUT = 120


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

    If no mala.yaml is found, returns an empty spec with no commands.

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
    from src.domain.validation.config_loader import load_config
    from src.domain.validation.config_merger import merge_configs
    from src.domain.validation.preset_registry import PresetRegistry

    # Use default scope if not specified
    if scope is None:
        scope = ValidationScope.PER_ISSUE

    disable = disable_validations or set()

    # Determine if we should skip tests
    skip_tests = "post-validate" in disable

    # Try to load config from repo
    try:
        user_config = load_config(repo_path)
    except ConfigError:
        # No config file found or invalid - return empty spec
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

    # Build commands list from config
    commands: list[ValidationCommand] = []

    if not skip_tests:
        commands = _build_commands_from_config(merged_config)

    # Determine if coverage is enabled
    coverage_enabled = (
        merged_config.coverage is not None
        and "coverage" not in disable
        and not skip_tests
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
    e2e_enabled = (
        scope == ValidationScope.RUN_LEVEL
        and "e2e" not in disable
        and merged_config.commands.e2e is not None
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
        yaml_coverage_config=merged_config.coverage,
    )


def _build_commands_from_config(config: ValidationConfig) -> list[ValidationCommand]:
    """Build ValidationCommand list from a merged ValidationConfig.

    Args:
        config: The merged configuration.

    Returns:
        List of ValidationCommand instances.
    """
    from src.domain.validation.tool_name_extractor import extract_tool_name

    commands: list[ValidationCommand] = []
    cmds = config.commands

    # Setup command
    if cmds.setup is not None:
        commands.append(
            ValidationCommand(
                name="setup",
                command=cmds.setup.command,
                kind=CommandKind.SETUP,
                timeout=cmds.setup.timeout or DEFAULT_COMMAND_TIMEOUT,
                detection_pattern=re.compile(
                    _tool_name_to_pattern(extract_tool_name(cmds.setup.command))
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
                    _tool_name_to_pattern(extract_tool_name(cmds.format.command))
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
                    _tool_name_to_pattern(extract_tool_name(cmds.lint.command))
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
                    _tool_name_to_pattern(extract_tool_name(cmds.typecheck.command))
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
                    _tool_name_to_pattern(extract_tool_name(cmds.test.command))
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
