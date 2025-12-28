"""Legacy validation configuration for mala.

This module provides ValidationConfig which is a legacy configuration
dataclass. LegacyValidationRunner has been removed in favor of
SpecValidationRunner with ValidationSpec.

For new code, use SpecValidationRunner with ValidationSpec:

    from src.validation import SpecValidationRunner, build_validation_spec

    runner = SpecValidationRunner(repo_path)
    spec = build_validation_spec(scope=ValidationScope.PER_ISSUE, ...)
    result = await runner.run_spec(spec, context)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValidationConfig:
    """Configuration for post-commit validation (legacy API).

    For new code, prefer ValidationSpec with SpecValidationRunner.
    """

    run_slow_tests: bool = True
    run_e2e: bool = True
    coverage: bool = True
    coverage_min: int = 85
    coverage_source: str = "."
    use_test_mutex: bool = (
        False  # Disabled to avoid deadlocks; isolated caches prevent conflicts
    )
    step_timeout_seconds: float | None = None
