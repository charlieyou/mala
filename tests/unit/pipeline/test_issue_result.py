"""Unit tests for IssueResult dataclass."""

from datetime import UTC, datetime
from pathlib import Path

from src.pipeline.issue_result import IssueResult
from src.core.session_end_result import SessionEndResult


class TestIssueResult:
    """Tests for IssueResult dataclass."""

    def test_create_minimal(self) -> None:
        """Create IssueResult with only required fields."""
        result = IssueResult(
            issue_id="test-1",
            agent_id="agent-1",
            success=True,
            summary="Test summary",
        )
        assert result.issue_id == "test-1"
        assert result.agent_id == "agent-1"
        assert result.success is True
        assert result.summary == "Test summary"
        # Defaults
        assert result.base_sha is None
        assert result.session_end_result is None

    def test_base_sha_field(self) -> None:
        """Create IssueResult with base_sha field."""
        result = IssueResult(
            issue_id="test-2",
            agent_id="agent-2",
            success=True,
            summary="Test",
            base_sha="abc123def456789",
        )
        assert result.base_sha == "abc123def456789"

    def test_session_end_result_field(self) -> None:
        """Create IssueResult with session_end_result field."""
        now = datetime.now(tz=UTC)
        session_end = SessionEndResult(
            status="pass",
            started_at=now,
            finished_at=now,
        )
        result = IssueResult(
            issue_id="test-3",
            agent_id="agent-3",
            success=True,
            summary="Test",
            session_end_result=session_end,
        )
        assert result.session_end_result is not None
        assert result.session_end_result.status == "pass"

    def test_both_new_fields(self) -> None:
        """Create IssueResult with both base_sha and session_end_result."""
        now = datetime.now(tz=UTC)
        session_end = SessionEndResult(
            status="fail",
            started_at=now,
            finished_at=now,
            reason="gate_failed",
        )
        result = IssueResult(
            issue_id="test-4",
            agent_id="agent-4",
            success=False,
            summary="Failed test",
            base_sha="fedcba987654321",
            session_end_result=session_end,
        )
        assert result.base_sha == "fedcba987654321"
        assert result.session_end_result is not None
        assert result.session_end_result.status == "fail"
        assert result.session_end_result.reason == "gate_failed"

    def test_backward_compatible_creation(self) -> None:
        """Existing IssueResult creation patterns still work."""
        # Replicate a common creation pattern without new fields
        result = IssueResult(
            issue_id="issue-123",
            agent_id="bd-mala-vk19",
            success=True,
            summary="Implementation complete",
            duration_seconds=120.5,
            session_id="sess-abc",
            gate_attempts=2,
            review_attempts=1,
            session_log_path=Path("/tmp/session.log"),
        )
        assert result.issue_id == "issue-123"
        assert result.duration_seconds == 120.5
        assert result.session_id == "sess-abc"
        assert result.gate_attempts == 2
        assert result.review_attempts == 1
        # New fields default to None
        assert result.base_sha is None
        assert result.session_end_result is None


class TestIssueResultSerialization:
    """Tests for IssueResult serialization with new fields."""

    def test_session_end_result_serializes(self) -> None:
        """SessionEndResult in IssueResult can be serialized via to_dict()."""
        import json

        now = datetime.now(tz=UTC)
        session_end = SessionEndResult(
            status="pass",
            started_at=now,
            finished_at=now,
        )
        result = IssueResult(
            issue_id="test-ser",
            agent_id="agent-ser",
            success=True,
            summary="Serialization test",
            base_sha="sha123",
            session_end_result=session_end,
        )
        # SessionEndResult has to_dict, so we can serialize it
        assert result.session_end_result is not None
        session_end_dict = result.session_end_result.to_dict()
        json_str = json.dumps(session_end_dict)
        parsed = json.loads(json_str)
        assert parsed["status"] == "pass"
