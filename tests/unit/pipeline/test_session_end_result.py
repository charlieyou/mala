"""Unit tests for SessionEndResult and SessionEndRetryState dataclasses."""

from datetime import UTC, datetime

from src.core.session_end_result import (
    CodeReviewResult,
    CommandOutcome,
    SessionEndResult,
    SessionEndRetryState,
)


class TestSessionEndRetryState:
    """Tests for SessionEndRetryState dataclass."""

    def test_incremented_attempt(self) -> None:
        """Track retry attempts through state."""
        state = SessionEndRetryState(attempt=1, max_retries=2)
        # Simulate retry progression
        state.attempt = 2
        assert state.attempt == 2
        state.attempt = 3
        assert state.attempt == 3


class TestCommandOutcome:
    """Tests for CommandOutcome dataclass."""

    def test_to_dict(self) -> None:
        """CommandOutcome serializes to dict correctly."""
        outcome = CommandOutcome(
            ref="test",
            passed=False,
            duration_seconds=2.5,
            error_message="Test failed",
        )
        data = outcome.to_dict()
        assert data == {
            "ref": "test",
            "passed": False,
            "duration_seconds": 2.5,
            "error_message": "Test failed",
        }


class TestCodeReviewResult:
    """Tests for CodeReviewResult dataclass."""

    def test_review_failed_with_findings(self) -> None:
        """Code review that failed with findings."""
        findings = [
            {"file": "a.py", "issue": "bug"},
            {"file": "b.py", "issue": "style"},
        ]
        result = CodeReviewResult(ran=True, passed=False, findings=findings)  # ty:ignore[invalid-argument-type]
        assert result.ran
        assert result.passed is False
        assert len(result.findings) == 2


class TestSerialization:
    """Tests for dataclass serialization (for persistence)."""

    def test_session_end_result_to_dict_json_serializable(self) -> None:
        """SessionEndResult.to_dict() produces JSON-serializable output."""
        import json

        now = datetime.now(tz=UTC)
        result = SessionEndResult(
            status="pass",
            started_at=now,
            finished_at=now,
            reason=None,
        )
        data = result.to_dict()
        # Must not raise TypeError
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["status"] == "pass"
        assert parsed["started_at"] == now.isoformat()
        assert parsed["finished_at"] == now.isoformat()
        assert parsed["commands"] == []

    def test_retry_state_to_dict_json_serializable(self) -> None:
        """SessionEndRetryState.to_dict() produces JSON-serializable output."""
        import json

        state = SessionEndRetryState(attempt=2, max_retries=3)
        data = state.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["attempt"] == 2
        assert parsed["max_retries"] == 3

    def test_code_review_result_to_dict_json_serializable(self) -> None:
        """CodeReviewResult.to_dict() produces JSON-serializable output."""
        import json

        review = CodeReviewResult(ran=True, passed=True, findings=[])
        data = review.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["ran"] is True
        assert parsed["passed"] is True

    def test_session_end_result_with_commands_json_serializable(self) -> None:
        """SessionEndResult with commands serializes to JSON cleanly."""
        import json

        now = datetime.now(tz=UTC)
        cmd_outcome = CommandOutcome(
            ref="test",
            passed=True,
            duration_seconds=1.5,
        )
        result = SessionEndResult(
            status="pass",
            started_at=now,
            finished_at=now,
            commands=[cmd_outcome],
        )
        data = result.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert len(parsed["commands"]) == 1
        assert parsed["commands"][0]["ref"] == "test"
        assert parsed["commands"][0]["passed"] is True
        assert parsed["commands"][0]["duration_seconds"] == 1.5
        assert parsed["commands"][0]["error_message"] is None

    def test_session_end_result_with_code_review_json_serializable(self) -> None:
        """SessionEndResult with code_review_result serializes to JSON cleanly."""
        import json

        now = datetime.now(tz=UTC)
        review = CodeReviewResult(ran=True, passed=False, findings=[{"issue": "test"}])
        result = SessionEndResult(
            status="pass",
            started_at=now,
            finished_at=now,
            code_review_result=review,
        )
        data = result.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["code_review_result"]["ran"] is True
        assert parsed["code_review_result"]["passed"] is False
        assert parsed["code_review_result"]["findings"] == [{"issue": "test"}]

    def test_session_end_result_skipped_json_serializable(self) -> None:
        """Skipped SessionEndResult (null timestamps) serializes to JSON cleanly."""
        import json

        result = SessionEndResult(
            status="skipped",
            reason="not_configured",
        )
        data = result.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["status"] == "skipped"
        assert parsed["started_at"] is None
        assert parsed["finished_at"] is None
        assert parsed["reason"] == "not_configured"
