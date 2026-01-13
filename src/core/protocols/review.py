"""Review protocols for code review operations.

This module defines protocols for code review, enabling dependency injection
and testability for the orchestrator's post-commit code review functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from collections.abc import Sequence  # noqa: TC003 - runtime needed

if TYPE_CHECKING:
    import asyncio
    from pathlib import Path

    from src.core.session_end_result import SessionEndResult

    from .validation import RetryStateProtocol


@runtime_checkable
class ReviewIssueProtocol(Protocol):
    """Protocol for review issues found during code review.

    Matches the shape of cerberus_review.ReviewIssue for structural typing.
    """

    file: str
    """File path where the issue was found."""

    line_start: int
    """Starting line number."""

    line_end: int
    """Ending line number."""

    priority: int | None
    """Issue priority (0=P0, 1=P1, etc.)."""

    title: str
    """Issue title."""

    body: str
    """Issue body/description."""

    reviewer: str
    """Which reviewer found this issue."""


@runtime_checkable
class ReviewResultProtocol(Protocol):
    """Protocol for code review results.

    Matches the shape of cerberus_review.ReviewResult for structural typing.
    """

    passed: bool
    """Whether the review passed."""

    issues: Sequence[ReviewIssueProtocol]
    """List of issues found during review."""

    parse_error: str | None
    """Parse error message if JSON parsing failed."""

    fatal_error: bool
    """Whether this is a fatal error (should not retry)."""

    review_log_path: Path | None
    """Path to review session logs."""

    interrupted: bool
    """Whether the review was interrupted by SIGINT."""


@runtime_checkable
class CodeReviewer(Protocol):
    """Protocol for code review operations.

    Provides a callable interface for reviewing commits and returning
    structured results. The orchestrator uses this to run post-commit
    code reviews via the Cerberus review-gate CLI.

    The canonical implementation is DefaultReviewer in cerberus_review.py.
    Test implementations can return predetermined results for isolation.
    """

    async def __call__(
        self,
        context_file: Path | None = None,
        timeout: int = 300,
        claude_session_id: str | None = None,
        author_context: str | None = None,
        *,
        commit_shas: Sequence[str],
        interrupt_event: asyncio.Event | None = None,
    ) -> ReviewResultProtocol:
        """Run code review on a specific set of commits.

        Args:
            context_file: Optional path to file with issue description context.
            timeout: Timeout in seconds for the review operation.
            claude_session_id: Optional Claude session ID for review attribution.
            author_context: Optional author-provided context for the reviewer.
            commit_shas: List of commit SHAs to review directly.
            interrupt_event: Optional event to check for SIGINT interruption.
                When set, reviewers should abort gracefully and return
                a result with interrupted=True.

        Returns:
            ReviewResultProtocol with review outcome. On parse failure,
            returns passed=False with parse_error set.
        """
        ...

    def overrides_disabled_setting(self) -> bool:
        """Return True if this reviewer should run even when review is disabled.

        DefaultReviewer returns False (respects disabled settings).
        Custom/injected reviewers return True (user explicitly provided them).
        """
        ...


@runtime_checkable
class ReviewInputProtocol(Protocol):
    """Protocol for review input data.

    Matches the shape of review_runner.ReviewInput for structural typing.
    """

    issue_id: str
    repo_path: Path
    commit_shas: list[str]
    issue_description: str | None
    claude_session_id: str | None
    author_context: str | None
    previous_findings: Sequence[ReviewIssueProtocol] | None
    diff_content: str | None


@runtime_checkable
class ReviewOutputProtocol(Protocol):
    """Protocol for review output data.

    Matches the shape of review_runner.ReviewOutput for structural typing.
    """

    result: ReviewResultProtocol
    session_log_path: str | None
    interrupted: bool


@runtime_checkable
class ReviewRunnerProtocol(Protocol):
    """Protocol for review runner operations.

    Provides an interface for running code reviews on commits. This allows
    modules to depend on the protocol rather than the concrete ReviewRunner
    implementation, avoiding circular imports in the pipeline layer.

    The canonical implementation is ReviewRunner in src/pipeline/review_runner.py.
    """

    async def run_review(
        self,
        input: ReviewInputProtocol,
        interrupt_event: asyncio.Event | None = None,
    ) -> ReviewOutputProtocol:
        """Run code review on the given input.

        Args:
            input: ReviewInput with commit_sha, issue_description, etc.
            interrupt_event: Optional event to check for SIGINT interruption.

        Returns:
            ReviewOutput with result and optional session log path.
        """
        ...


@runtime_checkable
class ReviewOutcomeProtocol(Protocol):
    """Protocol for review outcome.

    Matches the shape of domain.lifecycle.ReviewOutcome for structural typing.
    This allows IReviewRunner to reference the return type without importing
    from domain.
    """

    @property
    def passed(self) -> bool:
        """Whether the review passed."""
        ...

    @property
    def parse_error(self) -> str | None:
        """Parse error message if JSON parsing failed."""
        ...

    @property
    def fatal_error(self) -> bool:
        """Whether the review failure is unrecoverable."""
        ...

    @property
    def issues(self) -> Sequence[ReviewIssueProtocol]:
        """List of issues found during review."""
        ...

    @property
    def interrupted(self) -> bool:
        """Whether the review was interrupted by SIGINT."""
        ...


@runtime_checkable
class IReviewRunner(Protocol):
    """Protocol for review operations.

    This protocol defines methods for running code reviews and checking
    progress. It replaces the on_review_check and on_review_no_progress
    callbacks from SessionCallbacks.

    The canonical implementation is SessionCallbackFactory in
    src/pipeline/session_callback_factory.py.
    """

    async def run_review(
        self,
        issue_id: str,
        description: str | None,
        session_id: str | None,
        retry_state: RetryStateProtocol,
        author_context: str | None,
        previous_findings: Sequence[ReviewIssueProtocol] | None,
        session_end_result: SessionEndResult | None,
    ) -> ReviewOutcomeProtocol:
        """Run code review.

        Args:
            issue_id: The issue ID being reviewed.
            description: Optional issue description for context.
            session_id: Optional Claude session ID for review attribution.
            retry_state: Current retry state with attempt counts.
            author_context: Optional author-provided context for the reviewer.
            previous_findings: Optional list of findings from previous review.
            session_end_result: Optional session-end validation result.

        Returns:
            ReviewOutcome indicating pass/fail/retry.
        """
        ...

    def check_no_progress(
        self,
        log_path: Path,
        log_offset: int,
        prev_commit: str | None,
        curr_commit: str | None,
    ) -> bool:
        """Check if no progress was made since the last attempt.

        Args:
            log_path: Path to the JSONL log file.
            log_offset: Byte offset marking the end of the previous attempt.
            prev_commit: Commit hash from the previous attempt.
            curr_commit: Commit hash from this attempt.

        Returns:
            True if no progress was made, False if progress was detected.
        """
        ...
