"""Contract tests for EpicVerificationModel implementations.

Ensures FakeEpicVerificationModel implements all methods of EpicVerificationModel
protocol and exhibits correct behavioral parity.

Note: Protocol type compliance test is intentionally omitted because
FakeEpicVerificationModel returns EpicVerdict while the protocol expects
EpicVerdictProtocol. This is due to return type variance in Python's Protocol
system. See tests/fakes/epic_model.py:150-153 for details.
"""

import pytest

from src.core.protocols.validation import EpicVerificationModel
from tests.contracts import get_protocol_members
from tests.fakes.epic_model import (
    FakeEpicVerificationModel,
    make_failing_verdict,
    make_passing_verdict,
)


@pytest.mark.unit
def test_fake_epic_verification_model_implements_all_protocol_methods() -> None:
    """FakeEpicVerificationModel must implement all public methods of EpicVerificationModel."""
    protocol_methods = get_protocol_members(EpicVerificationModel)
    fake_methods = {
        name for name in dir(FakeEpicVerificationModel) if not name.startswith("_")
    }

    missing = protocol_methods - fake_methods
    assert not missing, (
        f"FakeEpicVerificationModel missing protocol methods: {sorted(missing)}"
    )


class TestFakeEpicVerificationModelBehavior:
    """Behavioral tests for FakeEpicVerificationModel."""

    @pytest.mark.unit
    async def test_verify_returns_configured_verdict(self) -> None:
        """verify() returns verdicts from configured sequence."""
        failing = make_failing_verdict()
        passing = make_passing_verdict()
        model = FakeEpicVerificationModel(verdicts=[failing, passing])

        result1 = await model.verify("epic-1: Test")
        assert result1.passed is False

        result2 = await model.verify("epic-1: Test")
        assert result2.passed is True

    @pytest.mark.unit
    async def test_verify_returns_default_passing_when_exhausted(self) -> None:
        """verify() returns passing verdict when sequence exhausted."""
        model = FakeEpicVerificationModel(verdicts=[])
        result = await model.verify("epic-1: Test")
        assert result.passed is True

    @pytest.mark.unit
    async def test_verify_records_attempts(self) -> None:
        """verify() records all attempts with epic_id extracted from epic_context."""
        model = FakeEpicVerificationModel(verdicts=[make_passing_verdict()])
        await model.verify("epic-123: Some description")

        assert len(model.attempts) == 1
        assert model.attempts[0].epic_id == "epic-123"
        assert model.attempts[0].epic_context == "epic-123: Some description"
        assert model.attempts[0].verdict.passed is True

    @pytest.mark.unit
    async def test_verify_with_full_context(self) -> None:
        """verify() accepts epic_context with embedded spec content."""
        model = FakeEpicVerificationModel(verdicts=[make_passing_verdict()])
        result = await model.verify(
            epic_context="epic-1: Test\n\n# Spec\n- Must do X",
        )
        assert result.passed is True

    @pytest.mark.unit
    def test_make_failing_verdict_creates_unmet_criteria(self) -> None:
        """make_failing_verdict() creates verdict with unmet criteria."""
        verdict = make_failing_verdict(
            criteria=[("Criterion 1", "No evidence", 1), ("Criterion 2", "Partial", 2)]
        )
        assert verdict.passed is False
        assert len(verdict.unmet_criteria) == 2
        assert verdict.unmet_criteria[0].criterion == "Criterion 1"
        assert verdict.unmet_criteria[1].priority == 2

    @pytest.mark.unit
    def test_make_passing_verdict_has_empty_criteria(self) -> None:
        """make_passing_verdict() creates verdict with no unmet criteria."""
        verdict = make_passing_verdict()
        assert verdict.passed is True
        assert verdict.unmet_criteria == []
