"""Post-commit validation runner for mala.

This module re-exports the validation runner:

- SpecValidationRunner: Modern API using ValidationSpec (RECOMMENDED)

For new code, use SpecValidationRunner directly with ValidationSpec and injected dependencies:

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
