"""Unit tests for IssueFinalizer.

Tests for last_review_issues passthrough and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.pipeline.issue_finalizer import (
    IssueFinalizer,
    IssueFinalizeInput,
    IssueFinalizeConfig,
)
from src.pipeline.issue_result import IssueResult
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.issue_lifecycle_port import FakeIssueLifecyclePort
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

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


def make_finalizer(config: IssueFinalizeConfig) -> IssueFinalizer:
    """Create an IssueFinalizer with fake ports."""
    return IssueFinalizer(
        config=config,
        issue_provider=FakeIssueProvider({"test-issue": FakeIssue("test-issue")}),
        event_sink=FakeEventSink(),
        issue_lifecycle=FakeIssueLifecyclePort(),
        epic_verification_coordinator=MagicMock(check_epic_closure=AsyncMock()),
        evidence_check=None,
        per_session_spec=None,
    )


@dataclass(frozen=True)
class FakeReviewIssue:
    """Fake ReviewIssueProtocol for tracking issue tests."""

    file: str
    line_start: int
    line_end: int
    priority: int | None
    title: str
    body: str
    reviewer: str


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

        config = IssueFinalizeConfig(track_review_issues=False)
        finalizer = make_finalizer(config)

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

        config = IssueFinalizeConfig(track_review_issues=False)
        finalizer = make_finalizer(config)

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

        config = IssueFinalizeConfig(track_review_issues=False)
        finalizer = make_finalizer(config)

        finalizer._record_issue_run(input, MagicMock())

        assert recorded_issue_run is not None
        assert recorded_issue_run.last_review_issues == []

    @pytest.mark.asyncio
    async def test_finalize_reports_created_follow_up_work(self) -> None:
        """Finalization signals when review tracking creates a new issue."""
        provider = FakeIssueProvider({"test-issue": FakeIssue("test-issue")})
        finalizer = IssueFinalizer(
            config=IssueFinalizeConfig(track_review_issues=True),
            issue_provider=provider,
            event_sink=FakeEventSink(),
            issue_lifecycle=FakeIssueLifecyclePort(),
            epic_verification_coordinator=MagicMock(check_epic_closure=AsyncMock()),
            evidence_check=None,
            per_session_spec=None,
        )
        result = IssueResult(
            issue_id="test-issue",
            agent_id="test-agent",
            success=True,
            summary="done",
            low_priority_review_issues=[
                FakeReviewIssue(
                    file="src/foo.py",
                    line_start=10,
                    line_end=10,
                    priority=2,
                    title="Consider refactoring",
                    body="details",
                    reviewer="reviewer",
                )
            ],
        )

        output = await finalizer.finalize(
            IssueFinalizeInput(
                issue_id="test-issue",
                result=result,
                run_metadata=MagicMock(),
            )
        )

        assert output.created_follow_up_work is True
        assert len(provider.created_issues) == 1
