"""Unit tests for ReviewRunner pipeline stage.

Tests the extracted code review orchestration logic using fake code reviewers,
without actual Cerberus CLI or subprocess dependencies.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.cerberus_review import ReviewIssue, ReviewResult
from src.pipeline.review_runner import (
    NoProgressInput,
    ReviewInput,
    ReviewOutput,
    ReviewRunner,
    ReviewRunnerConfig,
)
from src.quality_gate import CommitResult, GateResult, ValidationEvidence
from src.validation.spec import ValidationSpec


@dataclass
class FakeCodeReviewer:
    """Fake code reviewer for testing.

    Returns predetermined results without invoking Cerberus CLI.
    """

    result: ReviewResult = field(
        default_factory=lambda: ReviewResult(passed=True, issues=[])
    )
    calls: list[dict] = field(default_factory=list)

    async def __call__(
        self,
        diff_range: str,
        context_file: Path | None = None,
        timeout: int = 300,
        claude_session_id: str | None = None,
    ) -> ReviewResult:
        """Record call and return configured result."""
        self.calls.append(
            {
                "diff_range": diff_range,
                "context_file": context_file,
                "timeout": timeout,
                "claude_session_id": claude_session_id,
            }
        )
        return self.result


@dataclass
class FakeGateChecker:
    """Fake gate checker for no-progress testing.

    Implements the GateChecker protocol with stub methods except for
    check_no_progress which is used by ReviewRunner.
    """

    no_progress_result: bool = False
    no_progress_calls: list[dict] = field(default_factory=list)
    _gate_result: GateResult = field(
        default_factory=lambda: GateResult(passed=True, failure_reasons=[])
    )
    _commit_result: CommitResult = field(
        default_factory=lambda: CommitResult(exists=True, commit_hash="abc123")
    )
    _validation_evidence: ValidationEvidence = field(default_factory=ValidationEvidence)

    def check_with_resolution(
        self,
        issue_id: str,
        log_path: Path,
        baseline_timestamp: int | None = None,
        log_offset: int = 0,
        spec: ValidationSpec | None = None,
    ) -> GateResult:
        """Return pre-configured gate result."""
        return self._gate_result

    def get_log_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        """Stub - not used by ReviewRunner."""
        return start_offset

    def check_no_progress(
        self,
        log_path: Path,
        log_offset: int,
        previous_commit_hash: str | None,
        current_commit_hash: str | None,
        spec: ValidationSpec | None = None,
        check_validation_evidence: bool = True,
    ) -> bool:
        """Record call and return configured result."""
        self.no_progress_calls.append(
            {
                "log_path": log_path,
                "log_offset": log_offset,
                "previous_commit_hash": previous_commit_hash,
                "current_commit_hash": current_commit_hash,
                "spec": spec,
                "check_validation_evidence": check_validation_evidence,
            }
        )
        return self.no_progress_result

    def parse_validation_evidence_with_spec(
        self, log_path: Path, spec: ValidationSpec, offset: int = 0
    ) -> ValidationEvidence:
        """Return pre-configured validation evidence."""
        return self._validation_evidence

    def check_commit_exists(
        self, issue_id: str, baseline_timestamp: int | None = None
    ) -> CommitResult:
        """Return pre-configured commit result."""
        return self._commit_result


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
            code_reviewer=fake_reviewer,
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
            code_reviewer=fake_reviewer,
            config=config,
        )

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
            issue_description="Fix the bug",
            baseline_commit="def456",
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

    @pytest.mark.asyncio
    async def test_run_review_captures_review_log(
        self,
        tmp_path: Path,
    ) -> None:
        """Runner should capture review log path when available."""
        result = ReviewResult(
            passed=True,
            issues=[],
            review_log_path=Path("/path/to/review.jsonl"),
        )
        fake_reviewer = FakeCodeReviewer(result=result)

        config = ReviewRunnerConfig()
        runner = ReviewRunner(
            code_reviewer=fake_reviewer,
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
            code_reviewer=fake_reviewer,
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
            code_reviewer=fake_reviewer,
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
        result = ReviewResult(passed=True, issues=[])
        fake_reviewer = FakeCodeReviewer(result=result)
        runner = ReviewRunner(code_reviewer=fake_reviewer)

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
            ReviewIssue(
                title="[P1] Bug found",
                body="Description",
                priority=1,
                file="src/main.py",
                line_start=10,
                line_end=15,
                reviewer="cerberus",
            )
        ]
        result = ReviewResult(passed=False, issues=issues)
        fake_reviewer = FakeCodeReviewer(result=result)
        runner = ReviewRunner(code_reviewer=fake_reviewer)

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
        result = ReviewResult(
            passed=False,
            issues=[],
            parse_error="Invalid JSON output",
        )
        fake_reviewer = FakeCodeReviewer(result=result)
        runner = ReviewRunner(code_reviewer=fake_reviewer)

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
            code_reviewer=fake_reviewer,
            gate_checker=fake_gate_checker,
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
            code_reviewer=fake_reviewer,
            gate_checker=fake_gate_checker,
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
            code_reviewer=fake_reviewer,
            gate_checker=fake_gate_checker,
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
            code_reviewer=fake_reviewer,
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
        result = ReviewResult(passed=True, issues=[])
        output = ReviewOutput(result=result)

        assert output.result.passed is True
        assert output.session_log_path is None

    def test_output_with_session_log(self) -> None:
        """Output should include session log path."""
        result = ReviewResult(passed=True, issues=[])
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
            ) -> ReviewResult:
                nonlocal context_file_path
                context_file_path = context_file
                # Verify file exists during review
                assert context_file is not None
                assert context_file.exists()
                return ReviewResult(passed=True, issues=[])

        runner = ReviewRunner(code_reviewer=CapturingReviewer())

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
            ) -> ReviewResult:
                nonlocal context_file_path
                context_file_path = context_file
                # Verify file exists during review
                assert context_file is not None
                assert context_file.exists()
                raise RuntimeError("Review failed")

        runner = ReviewRunner(code_reviewer=FailingReviewer())

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
        runner = ReviewRunner(code_reviewer=fake_reviewer)

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
            issue_description=None,  # No description
        )

        # Should not raise any errors
        await runner.run_review(review_input)

        assert fake_reviewer.calls[0]["context_file"] is None
