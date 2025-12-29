"""Unit tests for ReviewRunner pipeline stage.

Tests the extracted code review orchestration logic using fake code reviewers,
without actual Codex CLI or subprocess dependencies.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.codex_review import CodexReviewResult, ReviewIssue
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

    Returns predetermined results without invoking Codex CLI.
    """

    result: CodexReviewResult = field(
        default_factory=lambda: CodexReviewResult(passed=True, issues=[])
    )
    calls: list[dict] = field(default_factory=list)

    async def __call__(
        self,
        repo_path: Path,
        commit_sha: str,
        max_retries: int = 2,
        issue_description: str | None = None,
        baseline_commit: str | None = None,
        capture_session_log: bool = False,
        thinking_mode: str | None = None,
    ) -> CodexReviewResult:
        """Record call and return configured result."""
        self.calls.append(
            {
                "repo_path": repo_path,
                "commit_sha": commit_sha,
                "max_retries": max_retries,
                "issue_description": issue_description,
                "baseline_commit": baseline_commit,
                "capture_session_log": capture_session_log,
                "thinking_mode": thinking_mode,
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
            thinking_mode="high",
            capture_session_log=True,
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
        )

        await runner.run_review(review_input)

        assert len(fake_reviewer.calls) == 1
        call = fake_reviewer.calls[0]
        assert call["repo_path"] == tmp_path
        assert call["commit_sha"] == "abc123"
        assert call["issue_description"] == "Fix the bug"
        assert call["baseline_commit"] == "def456"
        assert call["capture_session_log"] is True
        assert call["thinking_mode"] == "high"

    @pytest.mark.asyncio
    async def test_run_review_captures_session_log(
        self,
        tmp_path: Path,
    ) -> None:
        """Runner should capture session log path when available."""
        result = CodexReviewResult(
            passed=True,
            issues=[],
            session_log_path="/path/to/session.jsonl",
        )
        fake_reviewer = FakeCodeReviewer(result=result)

        config = ReviewRunnerConfig(capture_session_log=True)
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

        assert output.session_log_path == "/path/to/session.jsonl"


class TestReviewRunnerResults:
    """Test different review result scenarios."""

    @pytest.mark.asyncio
    async def test_review_passed(self, tmp_path: Path) -> None:
        """Runner should return passed result correctly."""
        result = CodexReviewResult(passed=True, issues=[])
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
                confidence_score=0.9,
                priority=1,
                file="src/main.py",
                line_start=10,
                line_end=15,
            )
        ]
        result = CodexReviewResult(passed=False, issues=issues)
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
        result = CodexReviewResult(
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

    def test_config_default_values(self) -> None:
        """Config should have sensible defaults."""
        config = ReviewRunnerConfig()

        assert config.max_review_retries == 2
        assert config.thinking_mode is None
        assert config.capture_session_log is False

    def test_config_with_custom_values(self) -> None:
        """Config should accept custom values."""
        config = ReviewRunnerConfig(
            max_review_retries=4,
            thinking_mode="xhigh",
            capture_session_log=True,
        )

        assert config.max_review_retries == 4
        assert config.thinking_mode == "xhigh"
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

    def test_input_with_optional_fields(self, tmp_path: Path) -> None:
        """Input should accept optional fields."""
        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_sha="abc123",
            issue_description="Fix the bug",
            baseline_commit="def456",
        )

        assert review_input.issue_description == "Fix the bug"
        assert review_input.baseline_commit == "def456"


class TestReviewOutput:
    """Test output data handling."""

    def test_output_required_fields(self) -> None:
        """Output should have required fields with defaults."""
        result = CodexReviewResult(passed=True, issues=[])
        output = ReviewOutput(result=result)

        assert output.result.passed is True
        assert output.session_log_path is None

    def test_output_with_session_log(self) -> None:
        """Output should include session log path."""
        result = CodexReviewResult(passed=True, issues=[])
        output = ReviewOutput(
            result=result,
            session_log_path="/path/to/log.jsonl",
        )

        assert output.session_log_path == "/path/to/log.jsonl"
