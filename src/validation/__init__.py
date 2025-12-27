"""Validation package for mala post-commit validation.

This package provides validation runners for clean-room validation
in temporary git worktrees:

- ValidationRunner: Facade that delegates to specialized runners
- SpecValidationRunner: Modern API using ValidationSpec + ValidationContext
- LegacyValidationRunner: Legacy API using ValidationConfig

For new code, prefer SpecValidationRunner with ValidationSpec.
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
from .legacy_runner import LegacyValidationRunner, ValidationConfig
from .result import ValidationResult, ValidationStepResult
from .runner import (
    ValidationRunner,
    _check_e2e_prereqs,
    _format_step_output,
    _tail,
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
from .spec_runner import SpecValidationRunner
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
    # Runners
    "LegacyValidationRunner",
    # Repo type detection
    "RepoType",
    "ResolutionOutcome",
    "SpecValidationRunner",
    "ValidationArtifacts",
    "ValidationCommand",
    "ValidationConfig",
    "ValidationContext",
    # Result types
    "ValidationResult",
    "ValidationRunner",
    "ValidationScope",
    "ValidationSpec",
    "ValidationStepResult",
    # Worktree
    "WorktreeConfig",
    "WorktreeContext",
    "WorktreeResult",
    "WorktreeState",
    # Backwards compatibility (private)
    "_check_e2e_prereqs",
    "_format_step_output",
    "_tail",
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
