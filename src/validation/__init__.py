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

__all__ = [
    "ValidationConfig",
    "ValidationResult",
    "ValidationRunner",
    "ValidationStepResult",
    "_check_e2e_prereqs",
    "_format_step_output",
    "_tail",
]
