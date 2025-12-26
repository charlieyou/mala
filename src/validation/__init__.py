"""Validation package for mala post-commit validation.

This package provides a clean-room validation runner that operates in
temporary git worktrees, ensuring commits are tested in isolation.
"""

from .runner import (
    ValidationConfig,
    ValidationResult,
    ValidationRunner,
    ValidationStepResult,
    _check_e2e_prereqs,
    _format_step_output,
    _tail,
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
    "CommandKind",
    "CoverageConfig",
    "E2EConfig",
    "IssueResolution",
    "ResolutionOutcome",
    "ValidationArtifacts",
    "ValidationCommand",
    "ValidationConfig",
    "ValidationContext",
    "ValidationResult",
    "ValidationRunner",
    "ValidationScope",
    "ValidationSpec",
    "ValidationStepResult",
    "WorktreeConfig",
    "WorktreeContext",
    "WorktreeResult",
    "WorktreeState",
    "_check_e2e_prereqs",
    "_format_step_output",
    "_tail",
    "build_validation_spec",
    "classify_change",
    "cleanup_stale_worktrees",
    "create_worktree",
    "remove_worktree",
]
