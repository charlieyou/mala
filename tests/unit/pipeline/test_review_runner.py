"""Unit tests for ReviewRunner pipeline stage.

Tests the extracted code review orchestration logic using fake code reviewers,
without actual Cerberus CLI or subprocess dependencies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from src.core.models import ReviewInput
from src.core.protocols.review import ReviewResultProtocol
from src.pipeline.review_runner import (
    NoProgressInput,
    ReviewOutput,
    ReviewRunner,
    ReviewRunnerConfig,
)
from tests.fakes.gate_checker import FakeGateChecker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.protocols.review import (
        CodeReviewer,
        ReviewIssueProtocol,
    )
    from src.core.protocols.validation import GateChecker


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

    def overrides_disabled_setting(self) -> bool:
        """Return True; test fakes override the disabled setting."""
        return True

    async def __call__(
        self,
        context_file: Path | None = None,
        timeout: int = 300,
        claude_session_id: str | None = None,
        author_context: str | None = None,
        *,
        commit_shas: Sequence[str],
        interrupt_event: asyncio.Event | None = None,
    ) -> FakeReviewResult:
        """Record call and return configured result."""
        self.calls.append(
            {
                "context_file": context_file,
                "timeout": timeout,
                "claude_session_id": claude_session_id,
                "author_context": author_context,
                "commit_shas": commit_shas,
                "interrupt_event": interrupt_event,
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
            commit_shas=["commit1"],
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
            issue_description="Fix the bug",
            commit_shas=["commit1", "commit2"],
            claude_session_id="session-123",
        )

        await runner.run_review(review_input)

        assert len(fake_reviewer.calls) == 1
        call = fake_reviewer.calls[0]
        # New signature: commit_shas, context_file, timeout
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
            commit_shas=["commit1"],
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
            issue_description=None,  # No description
            commit_shas=["commit1"],
        )

        await runner.run_review(review_input)

        assert len(fake_reviewer.calls) == 1
        call = fake_reviewer.calls[0]
        assert call["context_file"] is None

    @pytest.mark.asyncio
    async def test_run_review_skips_when_no_commits(
        self,
        fake_reviewer: FakeCodeReviewer,
        config: ReviewRunnerConfig,
        tmp_path: Path,
    ) -> None:
        """Runner should skip review when commit list is empty."""
        runner = ReviewRunner(
            code_reviewer=cast("CodeReviewer", fake_reviewer),
            config=config,
        )

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_shas=[],
        )

        output = await runner.run_review(review_input)
        assert output.result.passed is True
        assert fake_reviewer.calls == []


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
            commit_shas=["commit1"],
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
            commit_shas=["commit1"],
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
            commit_shas=["commit1"],
        )

        output = await runner.run_review(review_input)

        assert output.result.passed is False
        assert output.result.parse_error == "Invalid JSON output"

    @pytest.mark.asyncio
    async def test_review_exception_returns_fatal_error(self, tmp_path: Path) -> None:
        """Runner should convert reviewer exceptions into fatal errors."""

        class ExplodingReviewer:
            def overrides_disabled_setting(self) -> bool:
                return True

            async def __call__(
                self,
                context_file: Path | None = None,
                timeout: int = 300,
                claude_session_id: str | None = None,
                author_context: str | None = None,
                *,
                commit_shas: Sequence[str],
                interrupt_event: asyncio.Event | None = None,
            ) -> FakeReviewResult:
                raise RuntimeError("boom")

        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", ExplodingReviewer()))

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_shas=["commit1"],
        )

        output = await runner.run_review(review_input)

        assert output.result.passed is False
        assert output.result.fatal_error is True
        assert output.result.parse_error == "boom"


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

    def test_config_deprecated_fields_still_accepted(self) -> None:
        """Config should accept deprecated fields for backward compatibility."""
        config = ReviewRunnerConfig(
            thinking_mode="high",
            capture_session_log=True,
        )

        # These fields are deprecated but still accepted for backward compat
        assert config.thinking_mode == "high"
        assert config.capture_session_log is True


class TestContextFileCleanup:
    """Test context file cleanup after review completes."""

    @pytest.mark.asyncio
    async def test_author_context_in_context_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Author context should be appended to the context file."""
        captured_text: str | None = None

        @dataclass
        class CapturingReviewer:
            """Reviewer that captures the context file contents."""

            def overrides_disabled_setting(self) -> bool:
                return True

            async def __call__(
                self,
                context_file: Path | None = None,
                timeout: int = 300,
                claude_session_id: str | None = None,
                author_context: str | None = None,
                *,
                commit_shas: Sequence[str],
                interrupt_event: asyncio.Event | None = None,
            ) -> FakeReviewResult:
                nonlocal captured_text
                assert context_file is not None
                captured_text = context_file.read_text()
                return FakeReviewResult(passed=True, issues=[])

        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", CapturingReviewer()))

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            issue_description="Fix the bug",
            author_context="This is a false positive because X.",
            commit_shas=["commit1"],
        )

        await runner.run_review(review_input)

        assert captured_text is not None
        assert "Fix the bug" in captured_text
        # Check for the new prominent header and guidance
        assert "## Implementer's Response to Previous Review Findings" in captured_text
        assert "false positives or disputed findings" in captured_text
        assert "This is a false positive because X." in captured_text

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

            def overrides_disabled_setting(self) -> bool:
                return True

            async def __call__(
                self,
                context_file: Path | None = None,
                timeout: int = 300,
                claude_session_id: str | None = None,
                author_context: str | None = None,
                *,
                commit_shas: Sequence[str],
                interrupt_event: asyncio.Event | None = None,
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
            issue_description="Fix the bug",
            commit_shas=["commit1"],
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

            def overrides_disabled_setting(self) -> bool:
                return True

            async def __call__(
                self,
                context_file: Path | None = None,
                timeout: int = 300,
                claude_session_id: str | None = None,
                author_context: str | None = None,
                *,
                commit_shas: Sequence[str],
                interrupt_event: asyncio.Event | None = None,
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
            issue_description="Fix the bug",
            commit_shas=["commit1"],
        )

        output = await runner.run_review(review_input)

        assert output.result.fatal_error is True
        assert output.result.parse_error == "Review failed"

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
            issue_description=None,  # No description
            commit_shas=["commit1"],
        )

        # Should not raise any errors
        await runner.run_review(review_input)

        assert fake_reviewer.calls[0]["context_file"] is None


class TestReviewRunnerInterruptHandling:
    """Test interrupt handling in ReviewRunner."""

    @pytest.mark.asyncio
    async def test_returns_interrupted_when_event_set_before_start(
        self,
        tmp_path: Path,
    ) -> None:
        """Runner should return interrupted=True when event is set before starting."""
        fake_reviewer = FakeCodeReviewer()
        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", fake_reviewer))

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_shas=["commit1"],
        )

        # Set interrupt event before starting
        interrupt_event = asyncio.Event()
        interrupt_event.set()

        output = await runner.run_review(review_input, interrupt_event=interrupt_event)

        # Should return interrupted output without calling reviewer
        assert output.interrupted is True
        assert output.result.passed is False
        assert len(fake_reviewer.calls) == 0

    @pytest.mark.asyncio
    async def test_passes_interrupt_event_to_reviewer(
        self,
        tmp_path: Path,
    ) -> None:
        """Runner should pass interrupt_event to the code reviewer."""
        fake_reviewer = FakeCodeReviewer()
        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", fake_reviewer))

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_shas=["commit1"],
        )

        interrupt_event = asyncio.Event()

        await runner.run_review(review_input, interrupt_event=interrupt_event)

        assert len(fake_reviewer.calls) == 1
        assert fake_reviewer.calls[0]["interrupt_event"] is interrupt_event

    @pytest.mark.asyncio
    async def test_returns_interrupted_when_event_set_during_review(
        self,
        tmp_path: Path,
    ) -> None:
        """Runner should return interrupted=True when event is set during review."""

        class InterruptSettingReviewer:
            """Reviewer that sets interrupt event during call."""

            def overrides_disabled_setting(self) -> bool:
                return True

            async def __call__(
                self,
                context_file: Path | None = None,
                timeout: int = 300,
                claude_session_id: str | None = None,
                author_context: str | None = None,
                *,
                commit_shas: Sequence[str],
                interrupt_event: asyncio.Event | None = None,
            ) -> FakeReviewResult:
                # Simulate interrupt being set during review
                if interrupt_event:
                    interrupt_event.set()
                return FakeReviewResult(passed=True, issues=[])

        runner = ReviewRunner(
            code_reviewer=cast("CodeReviewer", InterruptSettingReviewer())
        )

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_shas=["commit1"],
        )

        interrupt_event = asyncio.Event()

        output = await runner.run_review(review_input, interrupt_event=interrupt_event)

        assert output.interrupted is True
        # Result should still be from the reviewer
        assert output.result.passed is True

    @pytest.mark.asyncio
    async def test_not_interrupted_when_event_not_set(
        self,
        tmp_path: Path,
    ) -> None:
        """Runner should return interrupted=False when event is never set."""
        fake_reviewer = FakeCodeReviewer()
        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", fake_reviewer))

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_shas=["commit1"],
        )

        interrupt_event = asyncio.Event()

        output = await runner.run_review(review_input, interrupt_event=interrupt_event)

        assert output.interrupted is False
        assert output.result.passed is True

    @pytest.mark.asyncio
    async def test_not_interrupted_when_no_event_provided(
        self,
        tmp_path: Path,
    ) -> None:
        """Runner should return interrupted=False when no event provided."""
        fake_reviewer = FakeCodeReviewer()
        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", fake_reviewer))

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            commit_shas=["commit1"],
        )

        # No interrupt_event provided (uses default None)
        output = await runner.run_review(review_input)

        assert output.interrupted is False
        assert output.result.passed is True


class TestSessionEndEvidence:
    """Tests for session_end_result evidence passing (per spec R5)."""

    @pytest.mark.asyncio
    async def test_session_end_result_included_in_context(
        self,
        tmp_path: Path,
    ) -> None:
        """Session end result should appear in context file when present."""
        from src.core.session_end_result import SessionEndResult

        captured_text: str | None = None

        @dataclass
        class CapturingReviewer:
            """Reviewer that captures the context file contents."""

            def overrides_disabled_setting(self) -> bool:
                return True

            async def __call__(
                self,
                context_file: Path | None = None,
                timeout: int = 300,
                claude_session_id: str | None = None,
                author_context: str | None = None,
                *,
                commit_shas: Sequence[str],
                interrupt_event: asyncio.Event | None = None,
            ) -> FakeReviewResult:
                nonlocal captured_text
                assert context_file is not None
                captured_text = context_file.read_text()
                return FakeReviewResult(passed=True, issues=[])

        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", CapturingReviewer()))

        session_end = SessionEndResult(status="pass", reason="all checks passed")
        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            issue_description="Fix the bug",
            commit_shas=["commit1"],
            session_end_result=session_end,
        )

        await runner.run_review(review_input)

        assert captured_text is not None
        assert "## Session End Validation Results (Informational)" in captured_text
        assert "**Status**: pass" in captured_text
        assert "**Reason**: all checks passed" in captured_text

    @pytest.mark.asyncio
    async def test_session_end_failed_does_not_auto_fail_review(
        self,
        tmp_path: Path,
    ) -> None:
        """Review should NOT auto-fail when session_end.status=fail (per spec R5)."""
        from src.core.session_end_result import SessionEndResult

        fake_reviewer = FakeCodeReviewer(
            result=FakeReviewResult(passed=True, issues=[])
        )
        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", fake_reviewer))

        session_end = SessionEndResult(status="fail", reason="gate_failed")
        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            issue_description="Fix the bug",
            commit_shas=["commit1"],
            session_end_result=session_end,
        )

        output = await runner.run_review(review_input)

        # Review should pass - session_end failure doesn't auto-fail review
        assert output.result.passed is True

    @pytest.mark.asyncio
    async def test_session_end_skipped_not_included_in_context(
        self,
        tmp_path: Path,
    ) -> None:
        """Session end with status=skipped should not appear in context."""
        from src.core.session_end_result import SessionEndResult

        captured_text: str | None = None

        @dataclass
        class CapturingReviewer:
            """Reviewer that captures the context file contents."""

            def overrides_disabled_setting(self) -> bool:
                return True

            async def __call__(
                self,
                context_file: Path | None = None,
                timeout: int = 300,
                claude_session_id: str | None = None,
                author_context: str | None = None,
                *,
                commit_shas: Sequence[str],
                interrupt_event: asyncio.Event | None = None,
            ) -> FakeReviewResult:
                nonlocal captured_text
                if context_file is not None:
                    captured_text = context_file.read_text()
                return FakeReviewResult(passed=True, issues=[])

        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", CapturingReviewer()))

        session_end = SessionEndResult(status="skipped", reason="not_configured")
        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            issue_description="Fix the bug",
            commit_shas=["commit1"],
            session_end_result=session_end,
        )

        await runner.run_review(review_input)

        # Context file should only contain issue_description, not session_end
        assert captured_text is not None
        assert "Fix the bug" in captured_text
        assert "Session End Validation" not in captured_text

    @pytest.mark.asyncio
    async def test_session_end_none_proceeds_normally(
        self,
        tmp_path: Path,
    ) -> None:
        """Review should proceed normally when session_end_result is None."""
        fake_reviewer = FakeCodeReviewer(
            result=FakeReviewResult(passed=True, issues=[])
        )
        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", fake_reviewer))

        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            issue_description="Fix the bug",
            commit_shas=["commit1"],
            session_end_result=None,  # Explicitly None
        )

        output = await runner.run_review(review_input)

        assert output.result.passed is True

    @pytest.mark.asyncio
    async def test_session_end_with_command_results(
        self,
        tmp_path: Path,
    ) -> None:
        """Session end evidence should include command results when present."""
        from src.core.session_end_result import CommandOutcome, SessionEndResult

        captured_text: str | None = None

        @dataclass
        class CapturingReviewer:
            """Reviewer that captures the context file contents."""

            def overrides_disabled_setting(self) -> bool:
                return True

            async def __call__(
                self,
                context_file: Path | None = None,
                timeout: int = 300,
                claude_session_id: str | None = None,
                author_context: str | None = None,
                *,
                commit_shas: Sequence[str],
                interrupt_event: asyncio.Event | None = None,
            ) -> FakeReviewResult:
                nonlocal captured_text
                assert context_file is not None
                captured_text = context_file.read_text()
                return FakeReviewResult(passed=True, issues=[])

        runner = ReviewRunner(code_reviewer=cast("CodeReviewer", CapturingReviewer()))

        cmd_pass = CommandOutcome(
            ref="test",
            passed=True,
            duration_seconds=1.5,
        )
        cmd_fail = CommandOutcome(
            ref="lint",
            passed=False,
            duration_seconds=0.5,
            error_message="error: lint failed",
        )
        session_end = SessionEndResult(
            status="fail",
            reason="gate_failed",
            commands=[cmd_pass, cmd_fail],
        )
        review_input = ReviewInput(
            issue_id="test-123",
            repo_path=tmp_path,
            issue_description="Fix the bug",
            commit_shas=["commit1"],
            session_end_result=session_end,
        )

        await runner.run_review(review_input)

        assert captured_text is not None
        assert "### Command Results" in captured_text
        assert "`test`" in captured_text
        assert "[PASS]" in captured_text
        assert "`lint`" in captured_text
        assert "[FAIL]" in captured_text
        assert "error: lint failed" in captured_text
