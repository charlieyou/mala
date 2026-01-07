"""Validation package for mala post-commit validation.

This package provides validation runners for clean-room validation
in temporary git worktrees:

- SpecValidationRunner: Modern API using ValidationSpec + ValidationContext (RECOMMENDED)

For new code, use SpecValidationRunner with ValidationSpec and injected dependencies:

    from src.domain.validation import SpecValidationRunner, build_validation_spec

    runner = SpecValidationRunner(
        repo_path,
        env_config=env_config,
        command_runner=command_runner,
        lock_manager=lock_manager,
    )
    spec = build_validation_spec(repo_path, scope=ValidationScope.PER_SESSION, ...)
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
from .result import ValidationResult, ValidationStepResult
from .runner import (
    SpecValidationRunner,
)
from .spec import (
    CommandKind,
    CoverageConfig,
    E2EConfig,
    IssueResolution,
    ResolutionOutcome,
    ValidationArtifacts,
    ValidationCommand,
    ValidationContext,
    ValidationScope,
    ValidationSpec,
    build_validation_spec,
    classify_change,
)
from .validation_gating import (
    get_config_files_changed,
    get_matching_code_files,
    get_setup_files_changed,
    should_invalidate_lint_cache,
    should_invalidate_setup_cache,
    should_trigger_validation,
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
    "ResolutionOutcome",
    # Runners
    "SpecValidationRunner",
    "ValidationArtifacts",
    "ValidationCommand",
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
    "format_step_output",
    # Validation gating
    "get_config_files_changed",
    "get_matching_code_files",
    "get_ready_issue_id",
    "get_setup_files_changed",
    "init_fixture_repo",
    "parse_and_check_coverage",
    "parse_coverage_xml",
    "remove_worktree",
    "should_invalidate_lint_cache",
    "should_invalidate_setup_cache",
    "should_trigger_validation",
    "tail",
    "write_fixture_repo",
]
