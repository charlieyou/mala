"""ValidationSpec and related types for mala validation.

This module provides:
- ValidationSpec: defines what validations to run for a given scope
- ValidationCommand: a single command in the validation pipeline
- ValidationContext: immutable context for a validation run
- ValidationArtifacts: record of validation outputs
- IssueResolution: how an issue was resolved
- Code vs docs classification helper
- Disable list handling for spec omissions
- Repo type detection for conditional validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from re import Pattern


class RepoType(Enum):
    """Type of repository for validation purposes."""

    PYTHON = "python"  # Python project with uv/pyproject.toml
    GENERIC = "generic"  # Non-Python or unknown project type


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


class ResolutionOutcome(Enum):
    """Outcome of issue resolution."""

    SUCCESS = "success"
    NO_CHANGE = "no_change"
    OBSOLETE = "obsolete"
    ALREADY_COMPLETE = "already_complete"


@dataclass(frozen=True)
class ValidationCommand:
    """A single command in the validation pipeline.

    Attributes:
        name: Human-readable name (e.g., "pytest", "ruff check").
        command: The command as a list of strings.
        kind: Classification of the command type.
        detection_pattern: Compiled regex to detect this command in log evidence.
            If None, falls back to hardcoded patterns in QualityGate.
        use_test_mutex: Whether to wrap with test mutex.
        allow_fail: If True, failure doesn't stop the pipeline.
    """

    name: str
    command: list[str]
    kind: CommandKind
    detection_pattern: Pattern[str] | None = None
    use_test_mutex: bool = False
    allow_fail: bool = False


@dataclass
class CoverageConfig:
    """Configuration for coverage validation.

    Attributes:
        enabled: Whether coverage is enabled.
        min_percent: Minimum coverage percentage (default 85.0).
        branch: Whether to include branch coverage.
        report_path: Path to coverage report file.
    """

    enabled: bool = True
    min_percent: float = 85.0
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


@dataclass
class ValidationArtifacts:
    """Record of validation outputs for observability.

    Attributes:
        log_dir: Directory containing validation logs.
        worktree_path: Path to the worktree (if any).
        worktree_state: State of the worktree ("kept" or "removed").
        coverage_report: Path to coverage report.
        e2e_fixture_path: Path to E2E fixture directory.
    """

    log_dir: Path
    worktree_path: Path | None = None
    worktree_state: Literal["kept", "removed"] | None = None
    coverage_report: Path | None = None
    e2e_fixture_path: Path | None = None


@dataclass(frozen=True)
class IssueResolution:
    """Records how an issue was resolved.

    Attributes:
        outcome: The resolution outcome (success, no_change, obsolete).
        rationale: Explanation for the resolution.
    """

    outcome: ResolutionOutcome
    rationale: str


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
    """

    commands: list[ValidationCommand]
    scope: ValidationScope
    require_clean_git: bool = True
    require_pytest_for_code_changes: bool = True
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    e2e: E2EConfig = field(default_factory=E2EConfig)

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


def detect_repo_type(repo_path: Path) -> RepoType:
    """Detect the type of repository for validation purposes.

    Checks for Python project markers to determine if Python-specific
    validation commands (uv sync, ruff, ty, pytest) should be required.

    Detection rules:
    - pyproject.toml exists => Python
    - uv.lock exists => Python
    - requirements.txt exists => Python (pip-based)
    - Otherwise => Generic

    Args:
        repo_path: Path to the repository root.

    Returns:
        RepoType indicating the detected project type.
    """
    # Check for uv/modern Python project
    if (repo_path / "pyproject.toml").exists():
        return RepoType.PYTHON

    # Check for uv lock file
    if (repo_path / "uv.lock").exists():
        return RepoType.PYTHON

    # Check for pip-based Python project
    if (repo_path / "requirements.txt").exists():
        return RepoType.PYTHON

    # Default to generic (no Python validation required)
    return RepoType.GENERIC


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
    scope: ValidationScope,
    disable_validations: set[str] | None = None,
    coverage_threshold: float = 85.0,
    repo_path: Path | None = None,
) -> ValidationSpec:
    """Build a ValidationSpec from CLI inputs.

    This implements the strict-by-default policy from quality-hardening-plan.md:
    - All validations are ON by default
    - Disable flags explicitly turn off validations

    Repo-type aware: Python-specific commands (ruff, ty, pytest) are
    only included when the target repo is detected as a Python project.

    Disable values:
    - "post-validate": Skip test commands entirely
    - "run-level-validate": (handled elsewhere, not here)
    - "slow-tests": Exclude slow tests from pytest
    - "coverage": Disable coverage checking
    - "e2e": Disable E2E fixture repo test
    - "codex-review": (handled elsewhere, not here)
    - "followup-on-run-validate-fail": (handled elsewhere, not here)

    Args:
        scope: Whether this is per-issue or run-level validation.
        disable_validations: Set of validation types to disable.
        coverage_threshold: Minimum coverage percentage.
        repo_path: Path to the target repository. If provided, enables
            repo-type detection to conditionally include Python commands.

    Returns:
        A ValidationSpec configured according to the inputs.
    """
    disable = disable_validations or set()

    # Detect repo type if path provided
    is_python_repo = True  # Default to Python for backward compatibility
    if repo_path is not None:
        repo_type = detect_repo_type(repo_path)
        is_python_repo = repo_type == RepoType.PYTHON

    # Determine if we should skip tests
    skip_tests = "post-validate" in disable

    # Determine if coverage is enabled early so we can use it for pytest flags
    coverage_enabled = "coverage" not in disable and not skip_tests

    # Build commands list
    commands: list[ValidationCommand] = []

    # Include setup for Python repos (unless skip_tests)
    # uv sync --all-extras is required to install dependencies in worktree
    if is_python_repo and not skip_tests:
        commands.append(
            ValidationCommand(
                name="uv sync",
                command=["uv", "sync", "--all-extras"],
                kind=CommandKind.SETUP,
                detection_pattern=re.compile(r"\buv\s+sync\b"),
            )
        )

    # Include format/lint/typecheck for Python repos (unless skip_tests)
    if is_python_repo and not skip_tests:
        commands.append(
            ValidationCommand(
                name="ruff format",
                command=["uvx", "ruff", "format", "--check", "."],
                kind=CommandKind.FORMAT,
                detection_pattern=re.compile(r"\b(uvx\s+)?ruff\s+format\b"),
            )
        )

        commands.append(
            ValidationCommand(
                name="ruff check",
                command=["uvx", "ruff", "check", "."],
                kind=CommandKind.LINT,
                detection_pattern=re.compile(r"\b(uvx\s+)?ruff\s+check\b"),
            )
        )

        commands.append(
            ValidationCommand(
                name="ty check",
                command=["uvx", "ty", "check"],
                kind=CommandKind.TYPECHECK,
                detection_pattern=re.compile(r"\b(uvx\s+)?ty\s+check\b"),
            )
        )

    # Add pytest for Python repos (unless skipping tests)
    if is_python_repo and not skip_tests:
        pytest_cmd = ["uv", "run", "pytest"]

        # Add coverage flags when coverage is enabled
        if coverage_enabled:
            pytest_cmd.extend(
                [
                    "--cov=src",
                    "--cov-report=xml",
                    "--cov-branch",
                    f"--cov-fail-under={coverage_threshold}",
                ]
            )

        # Add slow tests unless disabled
        if "slow-tests" not in disable:
            pytest_cmd.extend(["-m", "slow or not slow"])

        commands.append(
            ValidationCommand(
                name="pytest",
                command=pytest_cmd,
                kind=CommandKind.TEST,
                detection_pattern=re.compile(r"\b(uv\s+run\s+)?pytest\b"),
            )
        )

    # Configure coverage
    coverage_config = CoverageConfig(
        enabled=coverage_enabled,
        min_percent=coverage_threshold,
        branch=True,
    )

    # Configure E2E
    # E2E is only for run-level, and only if not disabled
    e2e_enabled = scope == ValidationScope.RUN_LEVEL and "e2e" not in disable
    e2e_config = E2EConfig(
        enabled=e2e_enabled,
        # Note: MORPH_API_KEY is not required here since MorphLLM is optional.
        # The e2e runner handles missing Morph gracefully.
        required_env=[],
    )

    return ValidationSpec(
        commands=commands,
        scope=scope,
        require_clean_git=True,
        require_pytest_for_code_changes=True,
        coverage=coverage_config,
        e2e=e2e_config,
    )
