"""Fake epic verification model for testing.

Provides an in-memory implementation of EpicVerificationModel with
configurable verdict sequences and observable state for testing
verification retry logic and verdict handling.
"""

from dataclasses import dataclass, field

from src.core.models import EpicVerdict, UnmetCriterion


@dataclass
class VerificationAttempt:
    """Record of a verification attempt for observable assertions.

    Attributes:
        epic_id: Identifier of the epic being verified (extracted from context).
        epic_context: Full context string passed to verify().
        verdict: The verdict returned for this attempt.
    """

    epic_id: str
    epic_context: str
    verdict: EpicVerdict


@dataclass
class FakeEpicVerificationModel:
    """In-memory epic verification model implementing EpicVerificationModel.

    Returns verdicts from a pre-configured sequence. Each call to verify()
    returns the next verdict in the sequence. If the sequence is exhausted,
    returns a passing verdict by default.

    Observable state:
        attempts: list of VerificationAttempt recording all verify() calls

    Example:
        >>> failing = EpicVerdict(passed=False, unmet_criteria=[], reasoning="failed")
        >>> passing = EpicVerdict(passed=True, unmet_criteria=[], reasoning="ok")
        >>> model = FakeEpicVerificationModel(verdicts=[failing, passing])
        >>> await model.verify("epic-1: ...", None)
        EpicVerdict(passed=False, ...)
        >>> model.attempts[0].verdict.passed
        False
        >>> await model.verify("epic-1: ...", None)
        EpicVerdict(passed=True, ...)
        >>> len(model.attempts)
        2
    """

    verdicts: list[EpicVerdict] = field(default_factory=list)
    attempts: list[VerificationAttempt] = field(default_factory=list)

    _call_index: int = field(default=0, repr=False)

    async def verify(
        self,
        epic_context: str,
    ) -> EpicVerdict:
        """Verify if the epic's acceptance criteria are met.

        Returns the next verdict from the configured sequence, or a default
        passing verdict if the sequence is exhausted.

        The epic_id is extracted from the first line of epic_context (before colon)
        for recording in attempts.
        """
        # Extract epic_id from context (e.g., "epic-123: Description" -> "epic-123")
        epic_id = epic_context.split(":")[0].strip() if epic_context else "unknown"

        if self._call_index < len(self.verdicts):
            verdict = self.verdicts[self._call_index]
            self._call_index += 1
        else:
            # Default to passing if sequence exhausted
            verdict = EpicVerdict(
                passed=True,
                unmet_criteria=[],
                reasoning="Default passing verdict (sequence exhausted)",
            )

        self.attempts.append(
            VerificationAttempt(
                epic_id=epic_id,
                epic_context=epic_context,
                verdict=verdict,
            )
        )
        return verdict


def make_failing_verdict(
    criteria: list[tuple[str, str, int]] | None = None,
    reasoning: str = "Verification failed",
) -> EpicVerdict:
    """Factory for creating failing verdicts with unmet criteria.

    Args:
        criteria: List of (criterion, evidence, priority) tuples.
            If None, creates a single generic unmet criterion.
        reasoning: Explanation of failure.

    Returns:
        EpicVerdict with passed=False and specified unmet criteria.
    """
    import hashlib

    if criteria is None:
        criteria = [("Generic criterion not met", "No evidence provided", 1)]

    unmet = [
        UnmetCriterion(
            criterion=c,
            evidence=e,
            priority=p,
            criterion_hash=hashlib.sha256(c.encode()).hexdigest()[:16],
        )
        for c, e, p in criteria
    ]

    return EpicVerdict(
        passed=False,
        unmet_criteria=unmet,
        reasoning=reasoning,
    )


def make_passing_verdict(
    reasoning: str = "All criteria satisfied",
) -> EpicVerdict:
    """Factory for creating passing verdicts.

    Args:
        reasoning: Explanation of success.

    Returns:
        EpicVerdict with passed=True and empty unmet_criteria.
    """
    return EpicVerdict(
        passed=True,
        unmet_criteria=[],
        reasoning=reasoning,
    )


# Protocol compliance note:
# FakeEpicVerificationModel implements EpicVerificationModel structurally.
# Static assertion is omitted because ty is strict about return type variance
# (EpicVerdict vs EpicVerdictProtocol), but duck typing works correctly at runtime.
