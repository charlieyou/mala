"""Validation package for mala post-commit validation.

This package provides validation runners for clean-room validation
in temporary git worktrees:

- SpecValidationRunner: Modern API using ValidationSpec + ValidationContext (RECOMMENDED)

For new code, use SpecValidationRunner with ValidationSpec:

    from src.validation import SpecValidationRunner, build_validation_spec

    runner = SpecValidationRunner(repo_path)
    spec = build_validation_spec(scope=ValidationScope.PER_ISSUE, ...)
    result = await runner.run_spec(spec, context)
"""

from .coverage import (
    CoverageResult,
    CoverageStatus,
    check_coverage_threshold,
    parse_and_check_coverage,
    parse_coverage_xml,
)
from .e2e import (
    E2EConfig as E2ERunnerConfig,
    E2EPrereqResult,
    E2EResult,
    E2ERunner,
    E2EStatus,
    check_e2e_prereqs,
)
from .helpers import (
    annotate_issue,
    decode_timeout_output,
    format_step_output,
    get_ready_issue_id,
    init_fixture_repo,
    tail,
    write_fixture_repo,
)
from .legacy_runner import ValidationConfig
from .result import ValidationResult, ValidationStepResult
from .runner import (
    SpecValidationRunner,
)
from .spec import (
    CommandKind,
    CoverageConfig,
    E2EConfig,
    IssueResolution,
    RepoType,
    ResolutionOutcome,
    ValidationArtifacts,
    ValidationCommand,
    ValidationContext,
    ValidationScope,
    ValidationSpec,
    build_validation_spec,
    classify_change,
    detect_repo_type,
)
from .worktree import (
    WorktreeConfig,
    WorktreeContext,
    WorktreeResult,
    WorktreeState,
    cleanup_stale_worktrees,
    create_worktree,
    remove_worktree,
)

__all__ = [
    # Spec types
    "CommandKind",
    "CoverageConfig",
    # Coverage
    "CoverageResult",
    "CoverageStatus",
    # E2E
    "E2EConfig",
    "E2EPrereqResult",
    "E2EResult",
    "E2ERunner",
    "E2ERunnerConfig",
    "E2EStatus",
    "IssueResolution",
    # Repo type detection
    "RepoType",
    "ResolutionOutcome",
    # Runners
    "SpecValidationRunner",
    "ValidationArtifacts",
    "ValidationCommand",
    "ValidationConfig",
    "ValidationContext",
    # Result types
    "ValidationResult",
    "ValidationScope",
    "ValidationSpec",
    "ValidationStepResult",
    # Worktree
    "WorktreeConfig",
    "WorktreeContext",
    "WorktreeResult",
    "WorktreeState",
    # Helpers (public)
    "annotate_issue",
    "build_validation_spec",
    "check_coverage_threshold",
    "check_e2e_prereqs",
    "classify_change",
    "cleanup_stale_worktrees",
    "create_worktree",
    "decode_timeout_output",
    "detect_repo_type",
    "format_step_output",
    "get_ready_issue_id",
    "init_fixture_repo",
    "parse_and_check_coverage",
    "parse_coverage_xml",
    "remove_worktree",
    "tail",
    "write_fixture_repo",
]
