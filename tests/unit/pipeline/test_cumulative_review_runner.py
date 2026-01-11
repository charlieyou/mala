"""Unit tests for CumulativeReviewRunner.

Tests _get_baseline_commit() logic for all trigger types and baseline modes,
and run_review() flow including diff handling, review execution, and baseline updates.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from src.domain.validation.config import FailureMode, TriggerType
from src.pipeline.cumulative_review_runner import CumulativeReviewRunner

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass
class FakeCodeReviewConfig:
    """Fake CodeReviewConfig for testing."""

    baseline: str | None = None
    failure_mode: FailureMode = FailureMode.ABORT
    track_review_issues: bool = True


@dataclass
class FakeRunMetadata:
    """Fake RunMetadata for testing baseline determination."""

    run_id: str = "run-123"
    run_start_commit: str | None = None
    last_cumulative_review_commits: dict[str, str] = field(default_factory=dict)


@dataclass
class FakeDiffStat:
    """Fake DiffStat for testing."""

    total_lines: int = 100
    files_changed: list[str] = field(default_factory=list)


@dataclass
class FakeGitUtils:
    """Fake GitUtils for testing.

    Configurable responses for baseline/HEAD lookups, reachability, and diff stats.
    """

    baseline_for_issue: dict[str, str | None] = field(default_factory=dict)
    head_commit: str = "abc1234"
    reachable_commits: set[str] = field(default_factory=set)
    diff_stat: FakeDiffStat = field(default_factory=FakeDiffStat)
    diff_content: str = "diff --git a/file.py b/file.py\n+added line"
    get_baseline_for_issue_calls: list[str] = field(default_factory=list)
    get_head_commit_calls: int = 0
    is_commit_reachable_calls: list[str] = field(default_factory=list)
    get_diff_stat_calls: list[tuple[str, str]] = field(default_factory=list)
    get_diff_content_calls: list[tuple[str, str]] = field(default_factory=list)

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

    async def get_diff_stat(
        self, from_commit: str, to_commit: str = "HEAD"
    ) -> FakeDiffStat:
        """Return configured diff stat."""
        self.get_diff_stat_calls.append((from_commit, to_commit))
        return self.diff_stat

    async def get_diff_content(self, from_commit: str, to_commit: str = "HEAD") -> str:
        """Return configured diff content."""
        self.get_diff_content_calls.append((from_commit, to_commit))
        return self.diff_content


@dataclass
class FakeReviewIssue:
    """Fake ReviewIssueProtocol for testing."""

    file: str = "unknown"
    line_start: int = 0
    line_end: int = 0
    priority: int = 2
    title: str = "Test issue"
    body: str = "Test body"


@dataclass
class FakeReviewResult:
    """Fake ReviewResultProtocol for testing."""

    passed: bool = True
    issues: Sequence[FakeReviewIssue] = field(default_factory=list)
    parse_error: str | None = None
    fatal_error: bool = False
    review_log_path: Path | None = None


@dataclass
class FakeReviewOutput:
    """Fake ReviewOutputProtocol for testing."""

    result: FakeReviewResult = field(default_factory=FakeReviewResult)
    session_log_path: str | None = None
    interrupted: bool = False


@dataclass
class FakeReviewRunner:
    """Fake ReviewRunnerProtocol for testing."""

    output: FakeReviewOutput = field(default_factory=FakeReviewOutput)
    should_raise: Exception | None = None
    run_review_calls: list[object] = field(default_factory=list)

    async def run_review(
        self,
        input: object,
        interrupt_event: object = None,
    ) -> FakeReviewOutput:
        self.run_review_calls.append(input)
        if self.should_raise:
            raise self.should_raise
        return self.output


@dataclass
class FakeBeadsClient:
    """Fake IssueProvider for testing."""

    create_issue_calls: list[dict[str, object]] = field(default_factory=list)
    find_issue_by_tag_calls: list[str] = field(default_factory=list)
    existing_tags: dict[str, str] = field(default_factory=dict)
    should_raise: Exception | None = None

    async def find_issue_by_tag_async(self, tag: str) -> str | None:
        """Find an issue by tag. Returns existing_tags[tag] if present."""
        self.find_issue_by_tag_calls.append(tag)
        return self.existing_tags.get(tag)

    async def create_issue_async(
        self,
        title: str,
        description: str,
        priority: str,
        tags: list[str] | None = None,
        parent_id: str | None = None,
    ) -> str | None:
        self.create_issue_calls.append(
            {
                "title": title,
                "description": description,
                "priority": priority,
                "tags": tags or [],
                "parent_id": parent_id,
            }
        )
        if self.should_raise:
            raise self.should_raise
        return None


def make_runner(
    git_utils: FakeGitUtils | None = None,
    review_runner: FakeReviewRunner | None = None,
    beads_client: FakeBeadsClient | None = None,
) -> tuple[CumulativeReviewRunner, FakeGitUtils, FakeReviewRunner, FakeBeadsClient]:
    """Create a CumulativeReviewRunner with fake dependencies.

    Returns tuple of (runner, git_utils, review_runner, beads_client) for inspection.
    """
    git = git_utils or FakeGitUtils()
    reviewer = review_runner or FakeReviewRunner()
    beads = beads_client or FakeBeadsClient()
    runner = CumulativeReviewRunner(
        review_runner=reviewer,  # type: ignore[arg-type]
        git_utils=git,  # type: ignore[arg-type]
        beads_client=beads,  # type: ignore[arg-type]
        logger=logging.getLogger("test"),
    )
    return runner, git, reviewer, beads


def make_runner_simple(
    git_utils: FakeGitUtils | None = None,
) -> CumulativeReviewRunner:
    """Create a CumulativeReviewRunner with fake dependencies (simple version)."""
    runner, _, _, _ = make_runner(git_utils=git_utils)
    return runner


class TestGetBaselineCommitSessionEnd:
    """Tests for session_end trigger type."""

    @pytest.mark.asyncio
    async def test_session_end_with_issue_commits_returns_baseline(self) -> None:
        """session_end with existing issue commits returns parent of first commit."""
        git_utils = FakeGitUtils(baseline_for_issue={"mala-123": "parent-sha"})
        runner = make_runner_simple(git_utils)

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
        runner = make_runner_simple(git_utils)

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
        runner = make_runner_simple(git_utils)

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
        runner = make_runner_simple()
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
        runner = make_runner_simple()
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
        runner = make_runner_simple()
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
        runner = make_runner_simple()
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
        runner = make_runner_simple()
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
        runner = make_runner_simple()
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
        runner = make_runner_simple()
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
        runner = make_runner_simple(git_utils)
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
        runner = make_runner_simple(git_utils)
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
        runner = make_runner_simple(git_utils)
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
        runner = make_runner_simple(git_utils)
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
        runner = make_runner_simple(git_utils)
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
        runner = make_runner_simple(git_utils)

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
        runner = make_runner_simple(git_utils)
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


class TestRunReviewEmptyDiff:
    """Tests for empty diff handling in run_review()."""

    @pytest.mark.asyncio
    async def test_empty_diff_returns_skipped(self, tmp_path: Path) -> None:
        """Empty diff returns skipped status without calling reviewer."""
        git_utils = FakeGitUtils(
            head_commit="abc1234",
            diff_stat=FakeDiffStat(total_lines=0, files_changed=[]),
        )
        runner, _, reviewer, _ = make_runner(git_utils=git_utils)
        metadata = FakeRunMetadata(run_start_commit="baseline-sha")

        result = await runner.run_review(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
        )

        assert result.status == "skipped"
        assert result.skip_reason == "empty_diff"
        assert result.findings == ()
        assert result.new_baseline_commit is None
        # Verify reviewer was NOT called
        assert reviewer.run_review_calls == []


class TestRunReviewLargeDiff:
    """Tests for large diff warning in run_review()."""

    @pytest.mark.asyncio
    async def test_large_diff_logs_warning_and_proceeds(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Large diff logs warning but proceeds with review."""
        git_utils = FakeGitUtils(
            head_commit="abc1234",
            diff_stat=FakeDiffStat(total_lines=6000, files_changed=["a.py", "b.py"]),
        )
        runner, _, reviewer, _ = make_runner(git_utils=git_utils)
        metadata = FakeRunMetadata(run_start_commit="baseline-sha")

        with caplog.at_level(logging.WARNING):
            result = await runner.run_review(
                trigger_type=TriggerType.RUN_END,
                config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
                run_metadata=metadata,  # type: ignore[arg-type]
                repo_path=tmp_path,
                interrupt_event=asyncio.Event(),
            )

        assert result.status == "success"
        assert "Large diff" in caplog.text
        assert "6000 lines" in caplog.text
        # Verify reviewer WAS called
        assert len(reviewer.run_review_calls) == 1


class TestRunReviewExecution:
    """Tests for review execution via ReviewRunner."""

    @pytest.mark.asyncio
    async def test_successful_review_returns_success(self, tmp_path: Path) -> None:
        """Successful review returns success status."""
        git_utils = FakeGitUtils(head_commit="abc1234")
        runner, _, reviewer, _ = make_runner(git_utils=git_utils)
        metadata = FakeRunMetadata(run_start_commit="baseline-sha")

        result = await runner.run_review(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
        )

        assert result.status == "success"
        assert result.new_baseline_commit == "abc1234"
        assert len(reviewer.run_review_calls) == 1

    @pytest.mark.asyncio
    async def test_diff_content_passed_to_reviewer(self, tmp_path: Path) -> None:
        """diff_content is fetched and passed to ReviewRunner."""
        expected_diff = "diff --git a/test.py b/test.py\n+new code"
        git_utils = FakeGitUtils(
            head_commit="abc1234",
            diff_content=expected_diff,
        )
        runner, git, reviewer, _ = make_runner(git_utils=git_utils)
        metadata = FakeRunMetadata(run_start_commit="baseline-sha")

        await runner.run_review(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
        )

        # Verify get_diff_content was called with correct args
        assert len(git.get_diff_content_calls) == 1
        assert git.get_diff_content_calls[0] == ("baseline-sha", "abc1234")

        # Verify diff_content was passed to reviewer
        assert len(reviewer.run_review_calls) == 1
        review_input = reviewer.run_review_calls[0]
        assert hasattr(review_input, "diff_content")
        assert review_input.diff_content == expected_diff  # type: ignore[union-attr]
        assert review_input.claude_session_id == "cumulative-run_end-run-123"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_execution_error_returns_failed_no_baseline_update(
        self, tmp_path: Path
    ) -> None:
        """Execution error returns failed status and does NOT update baseline."""
        git_utils = FakeGitUtils(head_commit="abc1234")
        review_runner = FakeReviewRunner(should_raise=RuntimeError("Review failed"))
        runner, _, _, _ = make_runner(git_utils=git_utils, review_runner=review_runner)
        metadata = FakeRunMetadata(
            run_start_commit="baseline-sha",
            last_cumulative_review_commits={},
        )

        result = await runner.run_review(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
        )

        assert result.status == "failed"
        assert "execution_error" in (result.skip_reason or "")
        assert result.new_baseline_commit is None
        # Baseline should NOT be updated
        assert "run_end" not in metadata.last_cumulative_review_commits

    @pytest.mark.asyncio
    async def test_execution_error_with_continue_failure_mode_returns_skipped(
        self, tmp_path: Path
    ) -> None:
        """Execution error with failure_mode=CONTINUE returns skipped status."""
        git_utils = FakeGitUtils(head_commit="abc1234")
        review_runner = FakeReviewRunner(should_raise=RuntimeError("Review failed"))
        runner, _, _, _ = make_runner(git_utils=git_utils, review_runner=review_runner)
        metadata = FakeRunMetadata(
            run_start_commit="baseline-sha",
            last_cumulative_review_commits={},
        )

        result = await runner.run_review(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(failure_mode=FailureMode.CONTINUE),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
        )

        assert result.status == "skipped"
        assert "execution_error" in (result.skip_reason or "")
        assert result.new_baseline_commit is None
        # Baseline should NOT be updated
        assert "run_end" not in metadata.last_cumulative_review_commits


class TestRunReviewBaselineUpdate:
    """Tests for baseline update on completion."""

    @pytest.mark.asyncio
    async def test_run_end_updates_baseline_on_success(self, tmp_path: Path) -> None:
        """run_end trigger updates baseline on successful review."""
        git_utils = FakeGitUtils(head_commit="new-head-sha")
        runner, _, _, _ = make_runner(git_utils=git_utils)
        metadata = FakeRunMetadata(
            run_start_commit="baseline-sha",
            last_cumulative_review_commits={},
        )

        result = await runner.run_review(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
        )

        assert result.status == "success"
        assert result.new_baseline_commit == "new-head-sha"
        assert metadata.last_cumulative_review_commits["run_end"] == "new-head-sha"

    @pytest.mark.asyncio
    async def test_epic_completion_updates_epic_specific_baseline(
        self, tmp_path: Path
    ) -> None:
        """epic_completion trigger updates epic-specific baseline."""
        git_utils = FakeGitUtils(head_commit="epic-head-sha")
        runner, _, _, _ = make_runner(git_utils=git_utils)
        metadata = FakeRunMetadata(
            run_start_commit="baseline-sha",
            last_cumulative_review_commits={},
        )

        result = await runner.run_review(
            trigger_type=TriggerType.EPIC_COMPLETION,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
            epic_id="epic-001",
        )

        assert result.status == "success"
        assert metadata.last_cumulative_review_commits["epic_completion:epic-001"] == (
            "epic-head-sha"
        )

    @pytest.mark.asyncio
    async def test_skipped_review_does_not_update_baseline(
        self, tmp_path: Path
    ) -> None:
        """Skipped review (empty diff) does NOT update baseline."""
        git_utils = FakeGitUtils(
            head_commit="abc1234",
            diff_stat=FakeDiffStat(total_lines=0, files_changed=[]),
        )
        runner, _, _, _ = make_runner(git_utils=git_utils)
        metadata = FakeRunMetadata(
            run_start_commit="baseline-sha",
            last_cumulative_review_commits={},
        )

        result = await runner.run_review(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
        )

        assert result.status == "skipped"
        assert "run_end" not in metadata.last_cumulative_review_commits


class TestRunReviewFindings:
    """Tests for finding extraction and beads issue creation."""

    @pytest.mark.asyncio
    async def test_findings_extracted_from_review_result(self, tmp_path: Path) -> None:
        """Findings are extracted from review result issues."""
        git_utils = FakeGitUtils(head_commit="abc1234")
        # Create a review result with findings
        review_result = FakeReviewResult(
            passed=False,
            issues=[
                FakeReviewIssue(
                    file="src/main.py",
                    line_start=10,
                    line_end=15,
                    priority=1,
                    title="Security issue",
                    body="Found SQL injection",
                ),
            ],
        )
        review_runner = FakeReviewRunner(output=FakeReviewOutput(result=review_result))
        runner, _, _, _ = make_runner(git_utils=git_utils, review_runner=review_runner)
        metadata = FakeRunMetadata(run_start_commit="baseline-sha")

        result = await runner.run_review(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
        )

        assert result.status == "success"
        assert len(result.findings) == 1
        assert result.findings[0].file == "src/main.py"
        assert result.findings[0].title == "Security issue"

    @pytest.mark.asyncio
    async def test_beads_issues_created_for_findings(self, tmp_path: Path) -> None:
        """Beads issues are created for each finding."""
        git_utils = FakeGitUtils(head_commit="abc1234")
        review_result = FakeReviewResult(
            passed=False,
            issues=[
                FakeReviewIssue(
                    file="src/main.py",
                    line_start=10,
                    line_end=15,
                    priority=1,
                    title="Security issue",
                    body="Found SQL injection",
                ),
            ],
        )
        review_runner = FakeReviewRunner(output=FakeReviewOutput(result=review_result))
        runner, _, _, beads = make_runner(
            git_utils=git_utils, review_runner=review_runner
        )
        metadata = FakeRunMetadata(run_start_commit="baseline-sha")

        await runner.run_review(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
        )

        assert len(beads.create_issue_calls) == 1
        assert beads.create_issue_calls[0]["title"] == "Security issue"
        assert beads.create_issue_calls[0]["priority"] == "P1"
        tags = beads.create_issue_calls[0]["tags"]
        assert isinstance(tags, list)
        # Verify new tag format
        assert "trigger:run_end" in tags
        assert "cumulative-review" in tags
        assert "review_finding" in tags
        assert "auto_generated" in tags
        # Verify fingerprint tag is present
        assert any(str(t).startswith("fp:") for t in tags)

    @pytest.mark.asyncio
    async def test_beads_issue_creation_failure_is_logged_not_fatal(
        self, tmp_path: Path
    ) -> None:
        """Beads issue creation failure is logged but doesn't fail the review."""
        git_utils = FakeGitUtils(head_commit="abc1234")
        review_result = FakeReviewResult(
            passed=False,
            issues=[
                FakeReviewIssue(
                    file="src/main.py",
                    line_start=10,
                    line_end=15,
                    priority=1,
                    title="Security issue",
                    body="Found SQL injection",
                ),
            ],
        )
        review_runner = FakeReviewRunner(output=FakeReviewOutput(result=review_result))
        beads_client = FakeBeadsClient(should_raise=RuntimeError("Beads API error"))
        runner, _, _, _ = make_runner(
            git_utils=git_utils,
            review_runner=review_runner,
            beads_client=beads_client,
        )
        metadata = FakeRunMetadata(run_start_commit="baseline-sha")

        result = await runner.run_review(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
        )

        # Review should still succeed even if beads creation failed
        assert result.status == "success"
        assert len(result.findings) == 1

    @pytest.mark.asyncio
    async def test_beads_issues_skipped_when_track_review_issues_false(
        self, tmp_path: Path
    ) -> None:
        """Beads issues are NOT created when track_review_issues=False."""
        git_utils = FakeGitUtils(head_commit="abc1234")
        review_result = FakeReviewResult(
            passed=False,
            issues=[
                FakeReviewIssue(
                    file="src/main.py",
                    line_start=10,
                    line_end=15,
                    priority=1,
                    title="Security issue",
                    body="Found SQL injection",
                ),
            ],
        )
        review_runner = FakeReviewRunner(output=FakeReviewOutput(result=review_result))
        runner, _, _, beads = make_runner(
            git_utils=git_utils, review_runner=review_runner
        )
        metadata = FakeRunMetadata(run_start_commit="baseline-sha")

        result = await runner.run_review(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(track_review_issues=False),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
        )

        # Review should succeed but NO beads issues created
        assert result.status == "success"
        assert len(result.findings) == 1  # findings still extracted
        assert len(beads.create_issue_calls) == 0  # but no beads issues created


class TestLargeDiffThreshold:
    """Test that LARGE_DIFF_THRESHOLD is configurable and respected."""

    @pytest.mark.asyncio
    async def test_threshold_boundary_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Diff at exactly threshold does not trigger warning."""
        git_utils = FakeGitUtils(
            head_commit="abc1234",
            diff_stat=FakeDiffStat(total_lines=5000, files_changed=["a.py"]),
        )
        runner, _, _, _ = make_runner(git_utils=git_utils)
        metadata = FakeRunMetadata(run_start_commit="baseline-sha")

        with caplog.at_level(logging.WARNING):
            await runner.run_review(
                trigger_type=TriggerType.RUN_END,
                config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
                run_metadata=metadata,  # type: ignore[arg-type]
                repo_path=tmp_path,
                interrupt_event=asyncio.Event(),
            )

        # At threshold (5000) should NOT warn
        assert "Large diff" not in caplog.text

    @pytest.mark.asyncio
    async def test_threshold_exceeded_triggers_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Diff exceeding threshold triggers warning."""
        git_utils = FakeGitUtils(
            head_commit="abc1234",
            diff_stat=FakeDiffStat(total_lines=5001, files_changed=["a.py"]),
        )
        runner, _, _, _ = make_runner(git_utils=git_utils)
        metadata = FakeRunMetadata(run_start_commit="baseline-sha")

        with caplog.at_level(logging.WARNING):
            await runner.run_review(
                trigger_type=TriggerType.RUN_END,
                config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
                run_metadata=metadata,  # type: ignore[arg-type]
                repo_path=tmp_path,
                interrupt_event=asyncio.Event(),
            )

        # Over threshold (5001) should warn
        assert "Large diff" in caplog.text


class TestFindingDedup:
    """Tests for fingerprint-based finding deduplication."""

    @pytest.mark.asyncio
    async def test_duplicate_finding_skipped(self, tmp_path: Path) -> None:
        """Finding with existing fingerprint tag is not created again."""
        git_utils = FakeGitUtils(head_commit="abc1234")
        review_result = FakeReviewResult(
            passed=False,
            issues=[
                FakeReviewIssue(
                    file="src/main.py",
                    line_start=10,
                    line_end=15,
                    priority=1,
                    title="Security issue",
                    body="Found SQL injection",
                ),
            ],
        )
        review_runner = FakeReviewRunner(output=FakeReviewOutput(result=review_result))
        # Pre-populate existing tag to simulate existing issue
        from src.pipeline.cumulative_review_runner import (
            ReviewFinding,
            _get_finding_fingerprint,
        )

        finding = ReviewFinding(
            file="src/main.py",
            line_start=10,
            line_end=15,
            priority=1,
            title="Security issue",
            body="Found SQL injection",
            reviewer="cumulative_review",
        )
        fingerprint = _get_finding_fingerprint(finding)
        beads_client = FakeBeadsClient(
            existing_tags={f"fp:{fingerprint}": "existing-issue-123"}
        )

        runner, _, _, beads = make_runner(
            git_utils=git_utils,
            review_runner=review_runner,
            beads_client=beads_client,
        )
        metadata = FakeRunMetadata(run_start_commit="baseline-sha")

        await runner.run_review(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
        )

        # Should have called find_issue_by_tag to check for dupe
        assert len(beads.find_issue_by_tag_calls) == 1
        assert beads.find_issue_by_tag_calls[0].startswith("fp:")
        # Should NOT have created issue (it was a duplicate)
        assert len(beads.create_issue_calls) == 0

    @pytest.mark.asyncio
    async def test_unique_finding_created(self, tmp_path: Path) -> None:
        """Finding with no existing fingerprint tag is created."""
        git_utils = FakeGitUtils(head_commit="abc1234")
        review_result = FakeReviewResult(
            passed=False,
            issues=[
                FakeReviewIssue(
                    file="src/main.py",
                    line_start=10,
                    line_end=15,
                    priority=1,
                    title="Security issue",
                    body="Found SQL injection",
                ),
            ],
        )
        review_runner = FakeReviewRunner(output=FakeReviewOutput(result=review_result))
        beads_client = FakeBeadsClient(existing_tags={})  # No existing issues

        runner, _, _, beads = make_runner(
            git_utils=git_utils,
            review_runner=review_runner,
            beads_client=beads_client,
        )
        metadata = FakeRunMetadata(run_start_commit="baseline-sha")

        await runner.run_review(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
        )

        # Should have checked for dupe
        assert len(beads.find_issue_by_tag_calls) == 1
        # Should have created issue (not a duplicate)
        assert len(beads.create_issue_calls) == 1


class TestFindingMetadata:
    """Tests for rich metadata in finding descriptions."""

    @pytest.mark.asyncio
    async def test_description_includes_baseline_and_trigger(
        self, tmp_path: Path
    ) -> None:
        """Description includes baseline commit range and trigger type."""
        git_utils = FakeGitUtils(head_commit="abc1234")
        review_result = FakeReviewResult(
            passed=False,
            issues=[
                FakeReviewIssue(
                    file="src/main.py",
                    line_start=10,
                    line_end=15,
                    priority=1,
                    title="Security issue",
                    body="Found SQL injection",
                ),
            ],
        )
        review_runner = FakeReviewRunner(output=FakeReviewOutput(result=review_result))
        runner, _, _, beads = make_runner(
            git_utils=git_utils, review_runner=review_runner
        )
        metadata = FakeRunMetadata(run_start_commit="baseline-sha")

        await runner.run_review(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
        )

        assert len(beads.create_issue_calls) == 1
        desc = beads.create_issue_calls[0]["description"]
        assert isinstance(desc, str)
        assert "**Trigger:** run_end" in desc
        assert "**Baseline:** baseline-sha..abc1234" in desc
        assert "**Location:** src/main.py:10-15" in desc
        assert "**Priority:** P1" in desc

    @pytest.mark.asyncio
    async def test_epic_completion_includes_epic_metadata(self, tmp_path: Path) -> None:
        """Epic completion findings include epic ID in metadata and tags."""
        git_utils = FakeGitUtils(head_commit="abc1234")
        review_result = FakeReviewResult(
            passed=False,
            issues=[
                FakeReviewIssue(
                    file="src/main.py",
                    line_start=10,
                    line_end=10,
                    priority=2,
                    title="Code smell",
                    body="Consider refactoring",
                ),
            ],
        )
        review_runner = FakeReviewRunner(output=FakeReviewOutput(result=review_result))
        runner, _, _, beads = make_runner(
            git_utils=git_utils, review_runner=review_runner
        )
        metadata = FakeRunMetadata(run_start_commit="baseline-sha")

        await runner.run_review(
            trigger_type=TriggerType.EPIC_COMPLETION,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
            epic_id="epic-001",
        )

        assert len(beads.create_issue_calls) == 1
        desc = beads.create_issue_calls[0]["description"]
        tags = beads.create_issue_calls[0]["tags"]
        parent_id = beads.create_issue_calls[0]["parent_id"]

        assert isinstance(desc, str)
        assert "**Epic:** epic-001" in desc
        assert "**Trigger:** epic_completion" in desc
        assert isinstance(tags, list)
        assert "epic:epic-001" in tags
        assert "trigger:epic_completion" in tags
        # Parent should be set to epic
        assert parent_id == "epic-001"

    @pytest.mark.asyncio
    async def test_single_line_location_format(self, tmp_path: Path) -> None:
        """Single line findings show simple line number format."""
        git_utils = FakeGitUtils(head_commit="abc1234")
        review_result = FakeReviewResult(
            passed=False,
            issues=[
                FakeReviewIssue(
                    file="src/main.py",
                    line_start=42,
                    line_end=42,  # Same as start = single line
                    priority=3,
                    title="Minor issue",
                    body="Just a warning",
                ),
            ],
        )
        review_runner = FakeReviewRunner(output=FakeReviewOutput(result=review_result))
        runner, _, _, beads = make_runner(
            git_utils=git_utils, review_runner=review_runner
        )
        metadata = FakeRunMetadata(run_start_commit="baseline-sha")

        await runner.run_review(
            trigger_type=TriggerType.RUN_END,
            config=FakeCodeReviewConfig(),  # type: ignore[arg-type]
            run_metadata=metadata,  # type: ignore[arg-type]
            repo_path=tmp_path,
            interrupt_event=asyncio.Event(),
        )

        assert len(beads.create_issue_calls) == 1
        desc = beads.create_issue_calls[0]["description"]
        assert isinstance(desc, str)
        # Single line should be "file:line" not "file:line-line"
        assert "**Location:** src/main.py:42" in desc
        assert "42-42" not in desc
