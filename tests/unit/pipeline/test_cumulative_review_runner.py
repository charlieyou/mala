"""Unit tests for CumulativeReviewRunner baseline determination.

Tests _get_baseline_commit() logic for all trigger types and baseline modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from src.domain.validation.config import TriggerType
from src.pipeline.cumulative_review_runner import CumulativeReviewRunner

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from src.core.protocols import ReviewIssueProtocol


@dataclass
class FakeCodeReviewConfig:
    """Fake CodeReviewConfig for testing."""

    baseline: str | None = None


@dataclass
class FakeRunMetadata:
    """Fake RunMetadata for testing baseline determination."""

    run_start_commit: str | None = None
    last_cumulative_review_commits: dict[str, str] = field(default_factory=dict)


@dataclass
class FakeGitUtils:
    """Fake GitUtils for testing.

    Configurable responses for baseline/HEAD lookups and reachability checks.
    """

    baseline_for_issue: dict[str, str | None] = field(default_factory=dict)
    head_commit: str = "abc1234"
    reachable_commits: set[str] = field(default_factory=set)
    get_baseline_for_issue_calls: list[str] = field(default_factory=list)
    get_head_commit_calls: int = 0
    is_commit_reachable_calls: list[str] = field(default_factory=list)

    async def get_baseline_for_issue(self, issue_id: str) -> str | None:
        """Return configured baseline for issue_id."""
        self.get_baseline_for_issue_calls.append(issue_id)
        return self.baseline_for_issue.get(issue_id)

    async def get_head_commit(self) -> str:
        """Return configured HEAD commit."""
        self.get_head_commit_calls += 1
        return self.head_commit

    async def is_commit_reachable(self, commit: str) -> bool:
        """Return True if commit is in reachable_commits, or all_reachable."""
        self.is_commit_reachable_calls.append(commit)
        # If reachable_commits is empty, treat all commits as reachable (default)
        if not self.reachable_commits:
            return True
        return commit in self.reachable_commits


@dataclass
class FakeReviewResult:
    """Fake ReviewResultProtocol for testing."""

    passed: bool = True
    issues: Sequence[ReviewIssueProtocol] = field(default_factory=list)
    parse_error: str | None = None
    fatal_error: bool = False
    review_log_path: Path | None = None


class FakeReviewRunner:
    """Fake ReviewRunnerProtocol for testing."""

    async def run_review(
        self,
        input: object,
        interrupt_event: object = None,
    ) -> FakeReviewResult:
        return FakeReviewResult()


class FakeBeadsClient:
    """Fake IssueProvider for testing."""

    pass


def make_runner(
    git_utils: FakeGitUtils | None = None,
) -> CumulativeReviewRunner:
    """Create a CumulativeReviewRunner with fake dependencies."""
    return CumulativeReviewRunner(
        review_runner=FakeReviewRunner(),  # type: ignore[arg-type]
        git_utils=git_utils or FakeGitUtils(),  # type: ignore[arg-type]
        beads_client=FakeBeadsClient(),  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
    )


class TestGetBaselineCommitSessionEnd:
    """Tests for session_end trigger type."""

    @pytest.mark.asyncio
    async def test_session_end_with_issue_commits_returns_baseline(self) -> None:
        """session_end with existing issue commits returns parent of first commit."""
        git_utils = FakeGitUtils(baseline_for_issue={"mala-123": "parent-sha"})
        runner = make_runner(git_utils)

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.SESSION_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=FakeRunMetadata(),  # type: ignore[arg-type]
            issue_id="mala-123",
        )

        assert result.commit == "parent-sha"
        assert result.skip_reason is None
        assert git_utils.get_baseline_for_issue_calls == ["mala-123"]

    @pytest.mark.asyncio
    async def test_session_end_no_commits_returns_none_with_skip_reason(self) -> None:
        """session_end with no issue commits returns None with skip_reason."""
        git_utils = FakeGitUtils(baseline_for_issue={})
        runner = make_runner(git_utils)

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.SESSION_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=FakeRunMetadata(),  # type: ignore[arg-type]
            issue_id="mala-456",
        )

        assert result.commit is None
        assert result.skip_reason == "no commits found for issue mala-456"
        assert git_utils.get_baseline_for_issue_calls == ["mala-456"]

    @pytest.mark.asyncio
    async def test_session_end_without_issue_id_returns_none_with_skip_reason(
        self,
    ) -> None:
        """session_end without issue_id returns None with skip_reason."""
        git_utils = FakeGitUtils()
        runner = make_runner(git_utils)

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.SESSION_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=FakeRunMetadata(),  # type: ignore[arg-type]
        )

        assert result.commit is None
        assert result.skip_reason == "session_end trigger missing issue_id"
        assert git_utils.get_baseline_for_issue_calls == []


class TestGetBaselineCommitSinceRunStart:
    """Tests for since_run_start baseline mode."""

    @pytest.mark.asyncio
    async def test_run_end_since_run_start_returns_run_start_commit(self) -> None:
        """run_end with since_run_start uses run_metadata.run_start_commit."""
        runner = make_runner()
        metadata = FakeRunMetadata(run_start_commit="run-start-sha")

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(baseline="since_run_start"),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
        )

        assert result.commit == "run-start-sha"
        assert result.skip_reason is None

    @pytest.mark.asyncio
    async def test_epic_completion_since_run_start_returns_run_start_commit(
        self,
    ) -> None:
        """epic_completion with since_run_start uses run_metadata.run_start_commit."""
        runner = make_runner()
        metadata = FakeRunMetadata(run_start_commit="epic-start-sha")

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.EPIC_COMPLETION,
            config=FakeCodeReviewConfig(baseline="since_run_start"),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            epic_id="epic-001",
        )

        assert result.commit == "epic-start-sha"
        assert result.skip_reason is None

    @pytest.mark.asyncio
    async def test_default_baseline_mode_is_since_run_start(self) -> None:
        """When baseline is None, defaults to since_run_start behavior."""
        runner = make_runner()
        metadata = FakeRunMetadata(run_start_commit="default-start-sha")

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(baseline=None),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
        )

        assert result.commit == "default-start-sha"
        assert result.skip_reason is None


class TestGetBaselineCommitSinceLastReview:
    """Tests for since_last_review baseline mode."""

    @pytest.mark.asyncio
    async def test_run_end_since_last_review_uses_stored_baseline(self) -> None:
        """run_end with since_last_review looks up last_cumulative_review_commits."""
        runner = make_runner()
        metadata = FakeRunMetadata(
            run_start_commit="run-start-sha",
            last_cumulative_review_commits={"run_end": "last-review-sha"},
        )

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(baseline="since_last_review"),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
        )

        assert result.commit == "last-review-sha"
        assert result.skip_reason is None

    @pytest.mark.asyncio
    async def test_epic_completion_since_last_review_uses_stored_baseline(
        self,
    ) -> None:
        """epic_completion with since_last_review looks up epic-specific key."""
        runner = make_runner()
        metadata = FakeRunMetadata(
            run_start_commit="run-start-sha",
            last_cumulative_review_commits={
                "epic_completion:epic-001": "epic-review-sha"
            },
        )

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.EPIC_COMPLETION,
            config=FakeCodeReviewConfig(baseline="since_last_review"),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            epic_id="epic-001",
        )

        assert result.commit == "epic-review-sha"
        assert result.skip_reason is None

    @pytest.mark.asyncio
    async def test_since_last_review_fallback_to_run_start(self) -> None:
        """since_last_review falls back to run_start_commit if no stored baseline."""
        runner = make_runner()
        metadata = FakeRunMetadata(
            run_start_commit="fallback-sha",
            last_cumulative_review_commits={},
        )

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(baseline="since_last_review"),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
        )

        assert result.commit == "fallback-sha"
        assert result.skip_reason is None

    @pytest.mark.asyncio
    async def test_epic_completion_without_epic_id_falls_back(self) -> None:
        """epic_completion without epic_id falls back to since_run_start."""
        runner = make_runner()
        metadata = FakeRunMetadata(
            run_start_commit="fallback-sha",
            last_cumulative_review_commits={
                "epic_completion:epic-001": "epic-review-sha"
            },
        )

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.EPIC_COMPLETION,
            config=FakeCodeReviewConfig(baseline="since_last_review"),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            # No epic_id provided
        )

        assert result.commit == "fallback-sha"
        assert result.skip_reason is None


class TestGetBaselineCommitFallbacks:
    """Tests for fallback behavior when run_start_commit is missing."""

    @pytest.mark.asyncio
    async def test_missing_run_start_commit_captures_head(self) -> None:
        """When run_start_commit is None, captures current HEAD."""
        git_utils = FakeGitUtils(head_commit="current-head-sha")
        runner = make_runner(git_utils)
        metadata = FakeRunMetadata(run_start_commit=None)

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(baseline="since_run_start"),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
        )

        assert result.commit == "current-head-sha"
        assert result.skip_reason is None
        assert git_utils.get_head_commit_calls == 1

    @pytest.mark.asyncio
    async def test_missing_run_start_and_empty_head_returns_none_with_skip_reason(
        self,
    ) -> None:
        """When run_start_commit is None and HEAD is empty, returns None with reason."""
        git_utils = FakeGitUtils(head_commit="")
        runner = make_runner(git_utils)
        metadata = FakeRunMetadata(run_start_commit=None)

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(baseline="since_run_start"),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
        )

        assert result.commit is None
        assert result.skip_reason == "could not determine baseline commit"

    @pytest.mark.asyncio
    async def test_since_last_review_fallback_to_head_when_no_run_start(self) -> None:
        """since_last_review with no stored baseline and no run_start captures HEAD."""
        git_utils = FakeGitUtils(head_commit="head-fallback-sha")
        runner = make_runner(git_utils)
        metadata = FakeRunMetadata(
            run_start_commit=None,
            last_cumulative_review_commits={},
        )

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(baseline="since_last_review"),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
        )

        assert result.commit == "head-fallback-sha"
        assert result.skip_reason is None


class TestGetBaselineCommitReachability:
    """Tests for baseline commit reachability checks (shallow clone handling)."""

    @pytest.mark.asyncio
    async def test_unreachable_baseline_returns_skip_reason(self) -> None:
        """When baseline commit is unreachable, returns None with skip_reason."""
        git_utils = FakeGitUtils(
            head_commit="abc1234",
            reachable_commits={"abc1234"},  # Only HEAD is reachable
        )
        runner = make_runner(git_utils)
        metadata = FakeRunMetadata(run_start_commit="unreachable-sha")

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(baseline="since_run_start"),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
        )

        assert result.commit is None
        assert result.skip_reason is not None
        assert "unreachable-sha" in result.skip_reason
        assert "not reachable" in result.skip_reason
        assert git_utils.is_commit_reachable_calls == ["unreachable-sha"]

    @pytest.mark.asyncio
    async def test_reachable_baseline_returns_commit(self) -> None:
        """When baseline commit is reachable, returns the commit."""
        git_utils = FakeGitUtils(
            head_commit="abc1234",
            reachable_commits={"run-start-sha", "abc1234"},
        )
        runner = make_runner(git_utils)
        metadata = FakeRunMetadata(run_start_commit="run-start-sha")

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(baseline="since_run_start"),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
        )

        assert result.commit == "run-start-sha"
        assert result.skip_reason is None
        assert git_utils.is_commit_reachable_calls == ["run-start-sha"]

    @pytest.mark.asyncio
    async def test_session_end_unreachable_baseline_returns_skip_reason(self) -> None:
        """session_end with unreachable baseline returns None with skip_reason."""
        git_utils = FakeGitUtils(
            baseline_for_issue={"mala-123": "issue-baseline-sha"},
            reachable_commits={"abc1234"},  # Baseline not in set
        )
        runner = make_runner(git_utils)

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.SESSION_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=FakeRunMetadata(),  # type: ignore[arg-type]
            issue_id="mala-123",
        )

        assert result.commit is None
        assert result.skip_reason is not None
        assert "issue-baseline-sha" in result.skip_reason
        assert "not reachable" in result.skip_reason

    @pytest.mark.asyncio
    async def test_since_last_review_unreachable_stored_baseline(self) -> None:
        """since_last_review with unreachable stored baseline returns skip_reason."""
        git_utils = FakeGitUtils(
            head_commit="abc1234",
            reachable_commits={"abc1234"},  # Only HEAD is reachable
        )
        runner = make_runner(git_utils)
        metadata = FakeRunMetadata(
            run_start_commit="also-unreachable",
            last_cumulative_review_commits={"run_end": "stored-baseline-sha"},
        )

        result = await runner._get_baseline_commit(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(baseline="since_last_review"),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
        )

        assert result.commit is None
        assert result.skip_reason is not None
        assert "stored-baseline-sha" in result.skip_reason
        assert "not reachable" in result.skip_reason
