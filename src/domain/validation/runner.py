"""Post-commit validation runner for mala.

This module re-exports the validation runner:

- SpecValidationRunner: Modern API using ValidationSpec (RECOMMENDED)

For new code, use SpecValidationRunner directly with ValidationSpec:

    from src.domain.validation import SpecValidationRunner, build_validation_spec

    runner = SpecValidationRunner(repo_path)
    spec = build_validation_spec(scope=ValidationScope.PER_ISSUE, ...)
    result = await runner.run_spec(spec, context)
"""

from __future__ import annotations

# Re-export result types
from .result import ValidationResult, ValidationStepResult

# Re-export spec runner
from .spec_runner import SpecValidationRunner

# Silence unused import warnings for re-exports
__all__ = [
    "SpecValidationRunner",
    "ValidationResult",
    "ValidationStepResult",
]
