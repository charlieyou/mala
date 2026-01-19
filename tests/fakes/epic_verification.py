"""Fake epic verification coordinator components for testing.

Provides in-memory implementations for testing EpicVerificationCoordinator
at the coordinator level (EpicVerificationResult), complementing the model-level
FakeEpicVerificationModel (EpicVerdict) in epic_model.py.

Observable state:
- FakeVerificationResults.attempts: list of VerificationAttempt recording all calls
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.models import EpicVerdict, EpicVerificationResult


@dataclass
class CoordinatorVerificationAttempt:
    """Record of a verification attempt for observable assertions.

    Attributes:
        epic_id: Identifier of the epic being verified.
        force: Whether force verification was requested.
        result: The result returned for this attempt.
    """

    epic_id: str
    force: bool
    result: EpicVerificationResult


@dataclass
class FakeVerificationResults:
    """Returns EpicVerificationResult from a pre-configured sequence.

    Each call to verify() returns the next result. If exhausted, returns
    the last result (or a default passing result if sequence was empty).

    This is the coordinator-level fake (returns EpicVerificationResult).
    For model-level testing (returns EpicVerdict), use FakeEpicVerificationModel.

    Observable state:
        attempts: list of CoordinatorVerificationAttempt recording all calls

    Example:
        >>> results = FakeVerificationResults(results=[
        ...     make_failing_verification_result(remediation_issues=["rem-1"]),
        ...     make_passing_verification_result(),
        ... ])
        >>> await results.verify("epic-1", force=False)
        EpicVerificationResult(passed_count=0, ...)
        >>> results.attempts[0].result.passed_count
        0
        >>> await results.verify("epic-1", force=True)
        EpicVerificationResult(passed_count=1, ...)
        >>> len(results.attempts)
        2
    """

    results: list[EpicVerificationResult] = field(default_factory=list)
    attempts: list[CoordinatorVerificationAttempt] = field(default_factory=list)
    _call_index: int = field(default=0, repr=False)

    async def verify(self, epic_id: str, force: bool) -> EpicVerificationResult:
        """Return next result from sequence, recording the attempt."""
        if self._call_index < len(self.results):
            result = self.results[self._call_index]
            self._call_index += 1
        elif self.results:
            # Return last result if exhausted
            result = self.results[-1]
        else:
            # Default passing result
            result = make_passing_verification_result()

        self.attempts.append(CoordinatorVerificationAttempt(epic_id, force, result))
        return result


def make_passing_verification_result(epic_id: str = "epic-1") -> EpicVerificationResult:
    """Create a passing EpicVerificationResult."""
    return EpicVerificationResult(
        verified_count=1,
        passed_count=1,
        failed_count=0,
        verdicts={
            epic_id: EpicVerdict(
                passed=True,
                unmet_criteria=[],
                reasoning="All criteria met",
            )
        },
        remediation_issues_created=[],
    )


def make_failing_verification_result(
    epic_id: str = "epic-1",
    remediation_issues: list[str] | None = None,
) -> EpicVerificationResult:
    """Create a failing EpicVerificationResult."""
    return EpicVerificationResult(
        verified_count=1,
        passed_count=0,
        failed_count=1,
        verdicts={
            epic_id: EpicVerdict(
                passed=False,
                unmet_criteria=[],
                reasoning="Criteria not met",
            )
        },
        remediation_issues_created=remediation_issues or [],
    )


def make_not_eligible_verification_result(
    reason: str | None = "2 of 5 child issues still open",
) -> EpicVerificationResult:
    """Create a result indicating epic not eligible (children still open)."""
    return EpicVerificationResult(
        verified_count=0,
        passed_count=0,
        failed_count=0,
        verdicts={},
        remediation_issues_created=[],
        ineligibility_reason=reason,
    )
