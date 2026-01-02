"""Validation result types.

These are used by SpecValidationRunner to return validation outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .coverage import CoverageResult
    from .e2e import E2EResult
    from .spec import ValidationArtifacts


@dataclass
class ValidationStepResult:
    """Result of a single validation step."""

    name: str
    command: str
    ok: bool
    returncode: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_seconds: float = 0.0


@dataclass
class ValidationResult:
    """Result of a validation run."""

    passed: bool
    steps: list[ValidationStepResult] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    retriable: bool = True
    artifacts: ValidationArtifacts | None = None
    coverage_result: CoverageResult | None = None
    e2e_result: E2EResult | None = None  # E2E result if E2E was executed

    def short_summary(self) -> str:
        """One-line summary for logs/prompts."""
        if self.passed:
            return "passed"
        if not self.failure_reasons:
            return "failed"
        return "; ".join(self.failure_reasons)
