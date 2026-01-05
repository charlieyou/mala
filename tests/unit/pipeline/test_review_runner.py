"""Unit tests for ReviewRunner pipeline stage.

Tests the extracted code review orchestration logic using fake code reviewers,
without actual Cerberus CLI or subprocess dependencies.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from src.pipeline.review_runner import (
    NoProgressInput,
    ReviewInput,
    ReviewOutput,
    ReviewRunner,
    ReviewRunnerConfig,
)
from src.core.protocols import (
    ReviewIssueProtocol,
    ReviewResultProtocol,
)
from tests.fakes.gate_checker import FakeGateChecker

if TYPE_CHECKING:
    from src.core.protocols import CodeReviewer, GateChecker


@dataclass
class FakeReviewIssue:
    """Fake review issue for testing that satisfies ReviewIssueProtocol."""

    file: str
    line_start: int
    line_end: int
    priority: int | None
    title: str
    body: str
    reviewer: str


@dataclass
class FakeReviewResult(ReviewResultProtocol):
    """Fake review result for testing that satisfies ReviewResultProtocol."""

    passed: bool
    issues: Sequence[ReviewIssueProtocol] = field(default_factory=list)
    parse_error: str | None = None
    fatal_error: bool = False
    review_log_path: Path | None = None


@dataclass
class FakeCodeReviewer:
    """Fake code reviewer for testing.

    Returns predetermined results without invoking Cerberus CLI.
    """

    result: FakeReviewResult = field(
        default_factory=lambda: FakeReviewResult(passed=True, issues=[])
    )
    calls: list[dict] = field(default_factory=list)

    async def __call__(
        self,
        diff_range: str,
        context_file: Path | None = None,
        timeout: int = 300,
        claude_session_id: str | None = None,
        *,
        commit_shas: Sequence[str] | None = None,
    ) -> FakeReviewResult:
        """Record call and return configured result."""
        self.calls.append(
            {
                "diff_range": diff_range,
                "context_file": context_file,
                "timeout": timeout,
                "claude_session_id": claude_session_id,
                "commit_shas": commit_shas,
            }
        )
        return self.result


class TestReviewRunnerBasics:
    """Test basic ReviewRunner functionality."""

    @pytest.fixture
    def fake_reviewer(self) -> FakeCodeReviewer:
        """Create a fake code reviewer."""
        return FakeCodeReviewer()

    @pytest.fixture
    def config(self) -> ReviewRunnerConfig:
        """Create a default config."""
        return ReviewRunnerConfig()

    @pytest.mark.asyncio
    async def test_run_review_returns_output(
        self,
        fake_reviewer: FakeCodeReviewer,
        config: ReviewRunnerConfig,
        tmp_path: Path,
    ) -> None:
        """Runner should return ReviewOutput with result."""
        runner = ReviewRunner(
            code_reviewer=cast("CodeReviewer", fake_reviewer),
            config=config,
        )

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
        )

        output = await runner.run_review(review_input)

        assert isinstance(output, ReviewOutput)
        assert output.result.passed is True
        assert output.session_log_path is None

    @pytest.mark.asyncio
    async def test_run_review_passes_parameters(
        self,
        fake_reviewer: FakeCodeReviewer,
        tmp_path: Path,
    ) -> None:
        """Runner should pass all parameters to code reviewer."""
        config = ReviewRunnerConfig(
            review_timeout=600,
        )
        runner = ReviewRunner(
            code_reviewer=cast("CodeReviewer", fake_reviewer),
            config=config,
        )

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
            issue_description="Fix the bug",
            baseline_commit="def456",
            commit_shas=["commit1", "commit2"],
            claude_session_id="session-123",
        )

        await runner.run_review(review_input)

        assert len(fake_reviewer.calls) == 1
        call = fake_reviewer.calls[0]
        # New signature: diff_range, context_file, timeout
        assert call["diff_range"] == "def456..abc123"
        assert call["timeout"] == 600
        # context_file should be set when issue_description is provided
        assert call["context_file"] is not None
        assert call["claude_session_id"] == "session-123"
        assert call["commit_shas"] == ["commit1", "commit2"]

    @pytest.mark.asyncio
    async def test_run_review_captures_review_log(
        self,
        tmp_path: Path,
    ) -> None:
        """Runner should capture review log path when available."""
        result = FakeReviewResult(
            passed=True,
            issues=[],
            review_log_path=Path("/path/to/review.jsonl"),
        )
        fake_reviewer = FakeCodeReviewer(result=result)

        config = ReviewRunnerConfig()
        runner = ReviewRunner(
            code_reviewer=cast("CodeReviewer", fake_reviewer),
            config=config,
        )

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
        )

        output = await runner.run_review(review_input)

        assert output.session_log_path == "/path/to/review.jsonl"

    @pytest.mark.asyncio
    async def test_run_review_no_context_file_without_description(
        self,
        fake_reviewer: FakeCodeReviewer,
        config: ReviewRunnerConfig,
        tmp_path: Path,
    ) -> None:
        """Runner should not create context file when no issue_description."""
        runner = ReviewRunner(
            code_reviewer=cast("CodeReviewer", fake_reviewer),
            config=config,
        )

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
            issue_description=None,  # No description
        )

        await runner.run_review(review_input)

        assert len(fake_reviewer.calls) == 1
        call = fake_reviewer.calls[0]
        assert call["context_file"] is None

    @pytest.mark.asyncio
    async def test_run_review_diff_range_without_baseline(
        self,
        fake_reviewer: FakeCodeReviewer,
        config: ReviewRunnerConfig,
        tmp_path: Path,
    ) -> None:
        """Runner should use commit's parent as baseline when not provided."""
        runner = ReviewRunner(
            code_reviewer=cast("CodeReviewer", fake_reviewer),
            config=config,
        )

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
            baseline_commit=None,  # No baseline
        )

        await runner.run_review(review_input)

        assert len(fake_reviewer.calls) == 1
        call = fake_reviewer.calls[0]
        # Uses commit's own parent, not HEAD~1, for correct historical reviews
        assert call["diff_range"] == "abc123~1..abc123"


class TestReviewRunnerResults:
    """Test different review result scenarios."""

    @pytest.mark.asyncio
    async def test_review_passed(self, tmp_path: Path) -> None:
        """Runner should return passed result correctly."""
        result = FakeReviewResult(passed=True, issues=[])
        fake_reviewer = FakeCodeReviewer(result=result)
        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", fake_reviewer))

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
        )

        output = await runner.run_review(review_input)

        assert output.result.passed is True
        assert output.result.issues == []

    @pytest.mark.asyncio
    async def test_review_failed_with_issues(self, tmp_path: Path) -> None:
        """Runner should return failed result with issues."""
        issues = [
            FakeReviewIssue(
                title="[P1] Bug found",
                body="Description",
                priority=1,
                file="src/main.py",
                line_start=10,
                line_end=15,
                reviewer="cerberus",
            )
        ]
        result = FakeReviewResult(passed=False, issues=issues)
        fake_reviewer = FakeCodeReviewer(result=result)
        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", fake_reviewer))

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
        )

        output = await runner.run_review(review_input)

        assert output.result.passed is False
        assert len(output.result.issues) == 1
        assert output.result.issues[0].title == "[P1] Bug found"

    @pytest.mark.asyncio
    async def test_review_failed_with_parse_error(self, tmp_path: Path) -> None:
        """Runner should return failed result with parse error."""
        result = FakeReviewResult(
            passed=False,
            issues=[],
            parse_error="Invalid JSON output",
        )
        fake_reviewer = FakeCodeReviewer(result=result)
        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", fake_reviewer))

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
        )

        output = await runner.run_review(review_input)

        assert output.result.passed is False
        assert output.result.parse_error == "Invalid JSON output"


class TestReviewRunnerNoProgress:
    """Test no-progress detection."""

    @pytest.fixture
    def fake_gate_checker(self) -> FakeGateChecker:
        """Create a fake gate checker."""
        return FakeGateChecker()

    @pytest.fixture
    def tmp_log_path(self, tmp_path: Path) -> Path:
        """Create a temporary log file."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")
        return log_path

    def test_check_no_progress_returns_true(
        self,
        fake_gate_checker: FakeGateChecker,
        tmp_log_path: Path,
    ) -> None:
        """Runner should return True when no progress detected."""
        fake_gate_checker.no_progress_result = True
        fake_reviewer = FakeCodeReviewer()
        runner = ReviewRunner(
            code_reviewer=cast("CodeReviewer", fake_reviewer),
            gate_checker=cast("GateChecker", fake_gate_checker),
        )

        no_progress_input = NoProgressInput(
            log_path=tmp_log_path,
            log_offset=1000,
            previous_commit_hash="abc123",
            current_commit_hash="abc123",
        )

        result = runner.check_no_progress(no_progress_input)

        assert result is True

    def test_check_no_progress_returns_false(
        self,
        fake_gate_checker: FakeGateChecker,
        tmp_log_path: Path,
    ) -> None:
        """Runner should return False when progress detected."""
        fake_gate_checker.no_progress_result = False
        fake_reviewer = FakeCodeReviewer()
        runner = ReviewRunner(
            code_reviewer=cast("CodeReviewer", fake_reviewer),
            gate_checker=cast("GateChecker", fake_gate_checker),
        )

        no_progress_input = NoProgressInput(
            log_path=tmp_log_path,
            log_offset=1000,
            previous_commit_hash="abc123",
            current_commit_hash="def456",
        )

        result = runner.check_no_progress(no_progress_input)

        assert result is False

    def test_check_no_progress_passes_parameters(
        self,
        fake_gate_checker: FakeGateChecker,
        tmp_log_path: Path,
    ) -> None:
        """Runner should pass all parameters to gate checker."""
        fake_reviewer = FakeCodeReviewer()
        runner = ReviewRunner(
            code_reviewer=cast("CodeReviewer", fake_reviewer),
            gate_checker=cast("GateChecker", fake_gate_checker),
        )

        no_progress_input = NoProgressInput(
            log_path=tmp_log_path,
            log_offset=500,
            previous_commit_hash="abc123",
            current_commit_hash="def456",
            spec=None,
        )

        runner.check_no_progress(no_progress_input)

        assert len(fake_gate_checker.no_progress_calls) == 1
        call = fake_gate_checker.no_progress_calls[0]
        assert call["log_path"] == tmp_log_path
        assert call["log_offset"] == 500
        assert call["previous_commit_hash"] == "abc123"
        assert call["current_commit_hash"] == "def456"
        assert call["check_validation_evidence"] is False

    def test_check_no_progress_raises_without_gate_checker(
        self,
        tmp_log_path: Path,
    ) -> None:
        """Runner should raise when gate_checker not set."""
        fake_reviewer = FakeCodeReviewer()
        runner = ReviewRunner(
            code_reviewer=cast("CodeReviewer", fake_reviewer),
            gate_checker=None,  # Not set
        )

        no_progress_input = NoProgressInput(
            log_path=tmp_log_path,
            log_offset=1000,
            previous_commit_hash="abc123",
            current_commit_hash="abc123",
        )

        with pytest.raises(ValueError, match="gate_checker must be set"):
            runner.check_no_progress(no_progress_input)


class TestReviewRunnerConfig:
    """Test configuration handling."""

    def test_config_with_custom_values(self) -> None:
        """Config should accept custom values via flags."""
        config = ReviewRunnerConfig(
            max_review_retries=4,
            review_timeout=600,
        )

        assert config.max_review_retries == 4
        assert config.review_timeout == 600

    def test_config_defaults(self) -> None:
        """Config should have sensible defaults."""
        config = ReviewRunnerConfig()

        assert config.max_review_retries == 3
        assert config.review_timeout == 1200

    def test_config_deprecated_fields_still_accepted(self) -> None:
        """Config should accept deprecated fields for backward compatibility."""
        config = ReviewRunnerConfig(
            thinking_mode="high",
            capture_session_log=True,
        )

        # These fields are deprecated but still accepted for backward compat
        assert config.thinking_mode == "high"
        assert config.capture_session_log is True


class TestReviewInput:
    """Test input data handling."""

    def test_input_required_fields(self, tmp_path: Path) -> None:
        """Input should require issue_id, repo_path, and commit_sha."""
        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
        )

        assert review_input.issue_id == "test-123"
        assert review_input.repo_path == tmp_path
        assert review_input.commit_sha == "abc123"
        assert review_input.issue_description is None
        assert review_input.baseline_commit is None
        assert review_input.claude_session_id is None

    def test_input_with_optional_fields(self, tmp_path: Path) -> None:
        """Input should accept optional fields."""
        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
            issue_description="Fix the bug",
            baseline_commit="def456",
            claude_session_id="session-456",
        )

        assert review_input.issue_description == "Fix the bug"
        assert review_input.baseline_commit == "def456"
        assert review_input.claude_session_id == "session-456"


class TestReviewOutput:
    """Test output data handling."""

    def test_output_required_fields(self) -> None:
        """Output should have required fields with defaults."""
        result = FakeReviewResult(passed=True, issues=[])
        output = ReviewOutput(result=result)

        assert output.result.passed is True
        assert output.session_log_path is None

    def test_output_with_session_log(self) -> None:
        """Output should include session log path."""
        result = FakeReviewResult(passed=True, issues=[])
        output = ReviewOutput(
            result=result,
            session_log_path="/path/to/log.jsonl",
        )

        assert output.session_log_path == "/path/to/log.jsonl"


class TestContextFileCleanup:
    """Test context file cleanup after review completes."""

    @pytest.mark.asyncio
    async def test_context_file_cleaned_up_after_success(
        self,
        tmp_path: Path,
    ) -> None:
        """Context file should be deleted after successful review."""
        context_file_path: Path | None = None

        @dataclass
        class CapturingReviewer:
            """Reviewer that captures the context file path."""

            async def __call__(
                self,
                diff_range: str,
                context_file: Path | None = None,
                timeout: int = 300,
                claude_session_id: str | None = None,
                *,
                commit_shas: Sequence[str] | None = None,
            ) -> FakeReviewResult:
                nonlocal context_file_path
                context_file_path = context_file
                # Verify file exists during review
                assert context_file is not None
                assert context_file.exists()
                return FakeReviewResult(passed=True, issues=[])

        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", CapturingReviewer()))

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
            issue_description="Fix the bug",
        )

        await runner.run_review(review_input)

        # Context file should be cleaned up after review
        assert context_file_path is not None
        assert not context_file_path.exists()

    @pytest.mark.asyncio
    async def test_context_file_cleaned_up_after_failure(
        self,
        tmp_path: Path,
    ) -> None:
        """Context file should be deleted even when review raises exception."""
        context_file_path: Path | None = None

        @dataclass
        class FailingReviewer:
            """Reviewer that raises an exception."""

            async def __call__(
                self,
                diff_range: str,
                context_file: Path | None = None,
                timeout: int = 300,
                claude_session_id: str | None = None,
                *,
                commit_shas: Sequence[str] | None = None,
            ) -> FakeReviewResult:
                nonlocal context_file_path
                context_file_path = context_file
                # Verify file exists during review
                assert context_file is not None
                assert context_file.exists()
                raise RuntimeError("Review failed")

        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", FailingReviewer()))

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
            issue_description="Fix the bug",
        )

        with pytest.raises(RuntimeError, match="Review failed"):
            await runner.run_review(review_input)

        # Context file should still be cleaned up even after exception
        assert context_file_path is not None
        assert not context_file_path.exists()

    @pytest.mark.asyncio
    async def test_no_cleanup_needed_without_description(
        self,
        tmp_path: Path,
    ) -> None:
        """No cleanup needed when no issue_description provided."""
        fake_reviewer = FakeCodeReviewer()
        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", fake_reviewer))

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
            issue_description=None,  # No description
        )

        # Should not raise any errors
        await runner.run_review(review_input)

        assert fake_reviewer.calls[0]["context_file"] is None
