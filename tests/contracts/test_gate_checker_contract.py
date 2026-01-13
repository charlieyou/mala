"""Contract tests for GateChecker implementations.

Ensures FakeGateChecker implements all methods of GateChecker protocol
and exhibits correct behavioral parity.
"""

from pathlib import Path

import pytest

from src.core.protocols.validation import GateChecker
from src.domain.evidence_check import CommitResult, GateResult, ValidationEvidence
from tests.contracts import get_protocol_members
from tests.fakes.gate_checker import FakeGateChecker


@pytest.mark.unit
def test_fake_gate_checker_implements_all_protocol_methods() -> None:
    """FakeGateChecker must implement all public methods of GateChecker."""
    protocol_methods = get_protocol_members(GateChecker)
    fake_methods = {name for name in dir(FakeGateChecker) if not name.startswith("_")}

    missing = protocol_methods - fake_methods
    assert not missing, f"FakeGateChecker missing protocol methods: {sorted(missing)}"


class TestFakeGateCheckerBehavior:
    """Behavioral tests for FakeGateChecker."""

    @pytest.mark.unit
    def test_check_with_resolution_returns_configured_result(self) -> None:
        """check_with_resolution() returns pre-configured gate_result."""
        checker = FakeGateChecker(
            gate_result=GateResult(passed=False, failure_reasons=["Test failed"])
        )
        result = checker.check_with_resolution(
            issue_id="test-1",
            log_path=Path("/tmp/log.jsonl"),
            baseline_timestamp=None,
            log_offset=0,
            spec=None,
        )
        assert result.passed is False
        assert "Test failed" in result.failure_reasons

    @pytest.mark.unit
    def test_check_with_resolution_records_calls(self) -> None:
        """check_with_resolution() records all call arguments."""
        checker = FakeGateChecker()
        checker.check_with_resolution(
            issue_id="issue-42",
            log_path=Path("/logs/session.jsonl"),
            baseline_timestamp=1234567890,
            log_offset=500,
            spec=None,
        )
        assert len(checker.check_with_resolution_calls) == 1
        call = checker.check_with_resolution_calls[0]
        assert call["issue_id"] == "issue-42"
        assert call["log_path"] == Path("/logs/session.jsonl")
        assert call["baseline_timestamp"] == 1234567890
        assert call["log_offset"] == 500

    @pytest.mark.unit
    def test_get_log_end_offset_returns_configured_value(self) -> None:
        """get_log_end_offset() returns pre-configured log_end_offset."""
        checker = FakeGateChecker(log_end_offset=2500)
        offset = checker.get_log_end_offset(Path("/tmp/log.jsonl"), start_offset=100)
        assert offset == 2500

    @pytest.mark.unit
    def test_get_log_end_offset_records_calls(self) -> None:
        """get_log_end_offset() records all call arguments."""
        checker = FakeGateChecker()
        checker.get_log_end_offset(Path("/tmp/log.jsonl"), start_offset=200)
        assert len(checker.get_log_end_offset_calls) == 1
        call = checker.get_log_end_offset_calls[0]
        assert call["log_path"] == Path("/tmp/log.jsonl")
        assert call["start_offset"] == 200

    @pytest.mark.unit
    def test_check_no_progress_returns_configured_result(self) -> None:
        """check_no_progress() returns pre-configured no_progress_result."""
        checker = FakeGateChecker(no_progress_result=True)
        result = checker.check_no_progress(
            log_path=Path("/tmp/log.jsonl"),
            log_offset=100,
            previous_commit_hash="abc123",
            current_commit_hash="abc123",
            spec=None,
        )
        assert result is True

    @pytest.mark.unit
    def test_check_no_progress_records_calls(self) -> None:
        """check_no_progress() records all call arguments."""
        checker = FakeGateChecker()
        checker.check_no_progress(
            log_path=Path("/tmp/log.jsonl"),
            log_offset=100,
            previous_commit_hash="abc123",
            current_commit_hash="def456",
            spec=None,
            check_validation_evidence=False,
        )
        assert len(checker.check_no_progress_calls) == 1
        call = checker.check_no_progress_calls[0]
        assert call["log_path"] == Path("/tmp/log.jsonl")
        assert call["log_offset"] == 100
        assert call["previous_commit_hash"] == "abc123"
        assert call["current_commit_hash"] == "def456"
        assert call["check_validation_evidence"] is False

    @pytest.mark.unit
    def test_no_progress_calls_alias(self) -> None:
        """no_progress_calls property aliases check_no_progress_calls."""
        checker = FakeGateChecker()
        checker.check_no_progress(
            log_path=Path("/tmp/log.jsonl"),
            log_offset=0,
            previous_commit_hash=None,
            current_commit_hash=None,
        )
        # Alias should return the same list
        assert checker.no_progress_calls is checker.check_no_progress_calls
        assert len(checker.no_progress_calls) == 1

    @pytest.mark.unit
    def test_parse_validation_evidence_returns_configured_value(self) -> None:
        """parse_validation_evidence_with_spec() returns pre-configured evidence."""
        from src.domain.validation.spec import (
            CommandKind,
            ValidationScope,
            ValidationSpec,
        )

        evidence = ValidationEvidence(
            commands_ran={CommandKind.TEST: True, CommandKind.LINT: True}
        )
        checker = FakeGateChecker(validation_evidence=evidence)
        spec = ValidationSpec(commands=[], scope=ValidationScope.PER_SESSION)
        result = checker.parse_validation_evidence_with_spec(
            log_path=Path("/tmp/log.jsonl"),
            spec=spec,
            offset=0,
        )
        # Test via protocol-compliant methods
        assert result.has_any_evidence() is True
        evidence_dict = result.to_evidence_dict()
        assert evidence_dict.get("test") is True
        assert evidence_dict.get("lint") is True

    @pytest.mark.unit
    def test_check_commit_exists_returns_configured_result(self) -> None:
        """check_commit_exists() returns pre-configured commit_result."""
        checker = FakeGateChecker(
            commit_result=CommitResult(exists=True, commit_hash="abc123def")
        )
        result = checker.check_commit_exists("issue-1", baseline_timestamp=None)
        assert result.exists is True
        assert result.commit_hash == "abc123def"

    @pytest.mark.unit
    def test_check_commit_exists_not_found(self) -> None:
        """check_commit_exists() can be configured to return not found."""
        checker = FakeGateChecker(
            commit_result=CommitResult(exists=False, commit_hash=None)
        )
        result = checker.check_commit_exists("nonexistent-issue")
        assert result.exists is False
        assert result.commit_hash is None
