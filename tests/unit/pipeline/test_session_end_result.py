"""Unit tests for SessionEndResult and SessionEndRetryState dataclasses."""

from datetime import UTC, datetime

from src.infra.tools.command_runner import CommandResult
from src.pipeline.session_end_result import (
    CodeReviewResult,
    SessionEndResult,
    SessionEndRetryState,
)


class TestSessionEndResult:
    """Tests for SessionEndResult dataclass."""

    def test_create_with_pass_status(self) -> None:
        """Create SessionEndResult with pass status."""
        now = datetime.now(tz=UTC)
        result = SessionEndResult(
            status="pass",
            started_at=now,
            finished_at=now,
        )
        assert result.status == "pass"
        assert result.started_at == now
        assert result.finished_at == now
        assert result.commands == []
        assert result.code_review_result is None
        assert result.reason is None

    def test_create_with_fail_status(self) -> None:
        """Create SessionEndResult with fail status."""
        now = datetime.now(tz=UTC)
        result = SessionEndResult(
            status="fail",
            started_at=now,
            finished_at=now,
            reason="max_retries_exhausted",
        )
        assert result.status == "fail"
        assert result.reason == "max_retries_exhausted"

    def test_create_with_timeout_status(self) -> None:
        """Create SessionEndResult with timeout status."""
        now = datetime.now(tz=UTC)
        result = SessionEndResult(
            status="timeout",
            started_at=now,
            finished_at=now,
            reason="command_timeout",
        )
        assert result.status == "timeout"

    def test_create_with_interrupted_status(self) -> None:
        """Create SessionEndResult with interrupted status."""
        now = datetime.now(tz=UTC)
        result = SessionEndResult(
            status="interrupted",
            started_at=now,
            finished_at=now,
            reason="SIGINT",
        )
        assert result.status == "interrupted"
        assert result.reason == "SIGINT"

    def test_create_with_skipped_status_null_timestamps(self) -> None:
        """Create SessionEndResult with skipped status has null timestamps."""
        result = SessionEndResult(
            status="skipped",
            started_at=None,
            finished_at=None,
            reason="gate_failed",
        )
        assert result.status == "skipped"
        assert result.started_at is None
        assert result.finished_at is None
        assert result.reason == "gate_failed"

    def test_skipped_not_configured(self) -> None:
        """Skipped status when session_end is not configured."""
        result = SessionEndResult(
            status="skipped",
            reason="not_configured",
        )
        assert result.status == "skipped"
        assert result.reason == "not_configured"

    def test_with_command_results(self) -> None:
        """SessionEndResult with command execution results."""
        now = datetime.now(tz=UTC)
        cmd_result = CommandResult(
            command=["pytest"],
            returncode=0,
            stdout="All tests passed",
            stderr="",
            duration_seconds=5.2,
        )
        result = SessionEndResult(
            status="pass",
            started_at=now,
            finished_at=now,
            commands=[cmd_result],
        )
        assert len(result.commands) == 1
        assert result.commands[0].ok

    def test_with_code_review_result(self) -> None:
        """SessionEndResult with code review result."""
        now = datetime.now(tz=UTC)
        review = CodeReviewResult(
            ran=True,
            passed=True,
            findings=[],
        )
        result = SessionEndResult(
            status="pass",
            started_at=now,
            finished_at=now,
            code_review_result=review,
        )
        assert result.code_review_result is not None
        assert result.code_review_result.ran
        assert result.code_review_result.passed

    def test_code_review_with_findings(self) -> None:
        """Code review result with findings."""
        now = datetime.now(tz=UTC)
        review = CodeReviewResult(
            ran=True,
            passed=False,
            findings=[{"file": "main.py", "line": 42, "message": "Issue found"}],
        )
        result = SessionEndResult(
            status="pass",
            started_at=now,
            finished_at=now,
            code_review_result=review,
        )
        assert result.code_review_result is not None
        assert not result.code_review_result.passed
        assert len(result.code_review_result.findings) == 1


class TestSessionEndRetryState:
    """Tests for SessionEndRetryState dataclass."""

    def test_default_values(self) -> None:
        """Create SessionEndRetryState with default values."""
        state = SessionEndRetryState()
        assert state.attempt == 1
        assert state.max_retries == 0
        assert state.log_offset == 0
        assert state.previous_commit_hash is None

    def test_custom_values(self) -> None:
        """Create SessionEndRetryState with custom values."""
        state = SessionEndRetryState(
            attempt=2,
            max_retries=3,
            log_offset=1024,
            previous_commit_hash="abc123def456",
        )
        assert state.attempt == 2
        assert state.max_retries == 3
        assert state.log_offset == 1024
        assert state.previous_commit_hash == "abc123def456"

    def test_incremented_attempt(self) -> None:
        """Track retry attempts through state."""
        state = SessionEndRetryState(attempt=1, max_retries=2)
        # Simulate retry progression
        state.attempt = 2
        assert state.attempt == 2
        state.attempt = 3
        assert state.attempt == 3


class TestCodeReviewResult:
    """Tests for CodeReviewResult dataclass."""

    def test_review_not_run(self) -> None:
        """Code review that did not run."""
        result = CodeReviewResult(ran=False, passed=None)
        assert not result.ran
        assert result.passed is None
        assert result.findings == []

    def test_review_passed(self) -> None:
        """Code review that passed."""
        result = CodeReviewResult(ran=True, passed=True)
        assert result.ran
        assert result.passed is True

    def test_review_failed_with_findings(self) -> None:
        """Code review that failed with findings."""
        findings = [
            {"file": "a.py", "issue": "bug"},
            {"file": "b.py", "issue": "style"},
        ]
        result = CodeReviewResult(ran=True, passed=False, findings=findings)
        assert result.ran
        assert result.passed is False
        assert len(result.findings) == 2


class TestSerialization:
    """Tests for dataclass serialization (for persistence)."""

    def test_session_end_result_to_dict(self) -> None:
        """SessionEndResult can be converted to dict for JSON serialization."""
        from dataclasses import asdict

        now = datetime.now(tz=UTC)
        result = SessionEndResult(
            status="pass",
            started_at=now,
            finished_at=now,
            reason=None,
        )
        data = asdict(result)
        assert data["status"] == "pass"
        assert data["started_at"] == now
        assert data["commands"] == []

    def test_retry_state_to_dict(self) -> None:
        """SessionEndRetryState can be converted to dict for JSON serialization."""
        from dataclasses import asdict

        state = SessionEndRetryState(attempt=2, max_retries=3)
        data = asdict(state)
        assert data["attempt"] == 2
        assert data["max_retries"] == 3

    def test_code_review_result_to_dict(self) -> None:
        """CodeReviewResult can be converted to dict for JSON serialization."""
        from dataclasses import asdict

        review = CodeReviewResult(ran=True, passed=True, findings=[])
        data = asdict(review)
        assert data["ran"] is True
        assert data["passed"] is True

    def test_session_end_result_json_serializable(self) -> None:
        """SessionEndResult with datetime fields can be JSON-serialized via isoformat."""
        import json
        from dataclasses import asdict

        now = datetime.now(tz=UTC)
        result = SessionEndResult(
            status="pass",
            started_at=now,
            finished_at=now,
            reason=None,
        )
        data = asdict(result)
        # Convert datetime to ISO format string for JSON serialization
        data["started_at"] = (
            data["started_at"].isoformat() if data["started_at"] else None
        )
        data["finished_at"] = (
            data["finished_at"].isoformat() if data["finished_at"] else None
        )
        # This should not raise TypeError
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["status"] == "pass"
        assert parsed["started_at"] == now.isoformat()

    def test_session_end_result_skipped_json_serializable(self) -> None:
        """Skipped SessionEndResult (null timestamps) serializes to JSON cleanly."""
        import json
        from dataclasses import asdict

        result = SessionEndResult(
            status="skipped",
            reason="not_configured",
        )
        data = asdict(result)
        # No datetime conversion needed when timestamps are None
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["status"] == "skipped"
        assert parsed["started_at"] is None
        assert parsed["finished_at"] is None
