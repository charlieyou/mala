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
    "_check_e2e_prereqs",
    "_format_step_output",
    "_tail",
    "build_validation_spec",
    "classify_change",
]
