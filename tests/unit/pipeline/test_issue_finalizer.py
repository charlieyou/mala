"""Unit tests for IssueFinalizer.

Tests for last_review_issues passthrough and persistence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, AsyncMock

from src.pipeline.issue_finalizer import (
    IssueFinalizer,
    IssueFinalizeInput,
    IssueFinalizeCallbacks,
    IssueFinalizeConfig,
)
from src.pipeline.issue_result import IssueResult

if TYPE_CHECKING:
    from pathlib import Path

    from src.infra.io.log_output.run_metadata import IssueRun


def make_minimal_result(
    issue_id: str = "test-issue",
    last_review_issues: list[dict[str, Any]] | None = None,
) -> IssueResult:
    """Create a minimal IssueResult for testing."""
    return IssueResult(
        issue_id=issue_id,
        agent_id="test-agent",
        success=False,
        summary="Test failed",
        duration_seconds=10.0,
        session_id="session-123",
        gate_attempts=1,
        review_attempts=1,
        resolution=None,
        low_priority_review_issues=None,
        session_log_path=None,
        review_log_path=None,
        baseline_timestamp=None,
        last_review_issues=last_review_issues,
    )


def make_callbacks() -> IssueFinalizeCallbacks:
    """Create mock callbacks for testing."""
    return IssueFinalizeCallbacks(
        close_issue=AsyncMock(return_value=True),
        mark_needs_followup=AsyncMock(return_value=True),
        on_issue_closed=MagicMock(),
        on_issue_completed=MagicMock(),
        trigger_epic_closure=AsyncMock(),
        create_tracking_issues=AsyncMock(),
    )


class TestIssueFinalizer:
    """Test IssueFinalizer persistence of last_review_issues."""

    def test_record_issue_run_persists_last_review_issues_with_values(
        self, tmp_path: Path
    ) -> None:
        """IssueRun should include last_review_issues when present in IssueResult."""
        review_issues = [
            {"file": "src/foo.py", "title": "Missing docstring", "priority": "P0"},
            {"file": "src/bar.py", "title": "Type error", "priority": "P1"},
        ]
        result = make_minimal_result(last_review_issues=review_issues)

        run_metadata = MagicMock()
        recorded_issue_run: IssueRun | None = None

        def capture_issue_run(issue_run: IssueRun) -> None:
            nonlocal recorded_issue_run
            recorded_issue_run = issue_run

        run_metadata.record_issue = capture_issue_run

        input = IssueFinalizeInput(
            issue_id="test-issue",
            result=result,
            run_metadata=run_metadata,
            log_path=None,
            stored_gate_result=None,
            review_log_path=None,
        )

        callbacks = make_callbacks()
        config = IssueFinalizeConfig(track_review_issues=False)
        finalizer = IssueFinalizer(
            config=config,
            callbacks=callbacks,
            evidence_check=None,
            per_session_spec=None,
        )

        # Call _record_issue_run directly to test the persistence logic
        finalizer._record_issue_run(input, MagicMock())

        assert recorded_issue_run is not None
        assert recorded_issue_run.last_review_issues == review_issues

    def test_record_issue_run_persists_none_when_no_review_issues(
        self, tmp_path: Path
    ) -> None:
        """IssueRun should have last_review_issues=None when not in IssueResult."""
        result = make_minimal_result(last_review_issues=None)

        run_metadata = MagicMock()
        recorded_issue_run: IssueRun | None = None

        def capture_issue_run(issue_run: IssueRun) -> None:
            nonlocal recorded_issue_run
            recorded_issue_run = issue_run

        run_metadata.record_issue = capture_issue_run

        input = IssueFinalizeInput(
            issue_id="test-issue",
            result=result,
            run_metadata=run_metadata,
            log_path=None,
            stored_gate_result=None,
            review_log_path=None,
        )

        callbacks = make_callbacks()
        config = IssueFinalizeConfig(track_review_issues=False)
        finalizer = IssueFinalizer(
            config=config,
            callbacks=callbacks,
            evidence_check=None,
            per_session_spec=None,
        )

        finalizer._record_issue_run(input, MagicMock())

        assert recorded_issue_run is not None
        assert recorded_issue_run.last_review_issues is None

    def test_record_issue_run_preserves_empty_list(self, tmp_path: Path) -> None:
        """IssueRun should preserve empty list (not convert to None)."""
        result = make_minimal_result(last_review_issues=[])

        run_metadata = MagicMock()
        recorded_issue_run: IssueRun | None = None

        def capture_issue_run(issue_run: IssueRun) -> None:
            nonlocal recorded_issue_run
            recorded_issue_run = issue_run

        run_metadata.record_issue = capture_issue_run

        input = IssueFinalizeInput(
            issue_id="test-issue",
            result=result,
            run_metadata=run_metadata,
            log_path=None,
            stored_gate_result=None,
            review_log_path=None,
        )

        callbacks = make_callbacks()
        config = IssueFinalizeConfig(track_review_issues=False)
        finalizer = IssueFinalizer(
            config=config,
            callbacks=callbacks,
            evidence_check=None,
            per_session_spec=None,
        )

        finalizer._record_issue_run(input, MagicMock())

        assert recorded_issue_run is not None
        assert recorded_issue_run.last_review_issues == []
