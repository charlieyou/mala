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
        epic_id: Identifier of the epic being verified (extracted from criteria).
        epic_criteria: Full criteria string passed to verify().
        commit_range: Commit range passed to verify().
        commit_list: Commit list passed to verify().
        spec_content: Spec content passed to verify().
        verdict: The verdict returned for this attempt.
    """

    epic_id: str
    epic_criteria: str
    commit_range: str
    commit_list: str
    spec_content: str | None
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
        >>> failing = EpicVerdict(passed=False, unmet_criteria=[], confidence=0.9, reasoning="failed")
        >>> passing = EpicVerdict(passed=True, unmet_criteria=[], confidence=0.95, reasoning="ok")
        >>> model = FakeEpicVerificationModel(verdicts=[failing, passing])
        >>> await model.verify("epic-1: ...", "abc..def", "abc,def", None)
        EpicVerdict(passed=False, ...)
        >>> model.attempts[0].verdict.passed
        False
        >>> await model.verify("epic-1: ...", "abc..def", "abc,def", None)
        EpicVerdict(passed=True, ...)
        >>> len(model.attempts)
        2
    """

    verdicts: list[EpicVerdict] = field(default_factory=list)
    attempts: list[VerificationAttempt] = field(default_factory=list)

    _call_index: int = field(default=0, repr=False)

    async def verify(
        self,
        epic_criteria: str,
        commit_range: str,
        commit_list: str,
        spec_content: str | None,
    ) -> EpicVerdict:
        """Verify if the commit scope satisfies the epic's acceptance criteria.

        Returns the next verdict from the configured sequence, or a default
        passing verdict if the sequence is exhausted.

        The epic_id is extracted from the first line of epic_criteria (before colon)
        for recording in attempts.
        """
        # Extract epic_id from criteria (e.g., "epic-123: Description" -> "epic-123")
        epic_id = epic_criteria.split(":")[0].strip() if epic_criteria else "unknown"

        if self._call_index < len(self.verdicts):
            verdict = self.verdicts[self._call_index]
            self._call_index += 1
        else:
            # Default to passing if sequence exhausted
            verdict = EpicVerdict(
                passed=True,
                unmet_criteria=[],
                confidence=1.0,
                reasoning="Default passing verdict (sequence exhausted)",
            )

        self.attempts.append(
            VerificationAttempt(
                epic_id=epic_id,
                epic_criteria=epic_criteria,
                commit_range=commit_range,
                commit_list=commit_list,
                spec_content=spec_content,
                verdict=verdict,
            )
        )
        return verdict


def make_failing_verdict(
    criteria: list[tuple[str, str, int]] | None = None,
    confidence: float = 0.9,
    reasoning: str = "Verification failed",
) -> EpicVerdict:
    """Factory for creating failing verdicts with unmet criteria.

    Args:
        criteria: List of (criterion, evidence, priority) tuples.
            If None, creates a single generic unmet criterion.
        confidence: Model confidence (0.0-1.0).
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
        confidence=confidence,
        reasoning=reasoning,
    )


def make_passing_verdict(
    confidence: float = 0.95,
    reasoning: str = "All criteria satisfied",
) -> EpicVerdict:
    """Factory for creating passing verdicts.

    Args:
        confidence: Model confidence (0.0-1.0).
        reasoning: Explanation of success.

    Returns:
        EpicVerdict with passed=True and empty unmet_criteria.
    """
    return EpicVerdict(
        passed=True,
        unmet_criteria=[],
        confidence=confidence,
        reasoning=reasoning,
    )


# Protocol compliance note:
# FakeEpicVerificationModel implements EpicVerificationModel structurally.
# Static assertion is omitted because ty is strict about return type variance
# (EpicVerdict vs EpicVerdictProtocol), but duck typing works correctly at runtime.
