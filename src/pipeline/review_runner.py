"""ReviewRunner: Code review pipeline stage.

Extracted from MalaOrchestrator to separate review orchestration from main
orchestration logic. This module handles:
- Running code reviews via injected CodeReviewer protocol
- Retry decisions and no-progress detection
- Session log path tracking

The ReviewRunner receives explicit inputs and returns explicit outputs,
making it testable without SDK or subprocess dependencies.

Design principles:
- Protocol-based dependencies for testability
- Explicit input/output types for clarity
- Pure functions where possible
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from src.infra.sigint_guard import InterruptGuard

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import asyncio

    from src.core.protocols import (
        CodeReviewer,
        GateChecker,
        ReviewResultProtocol,
        ValidationSpecProtocol,
    )
    from src.domain.validation.spec import ValidationSpec


@dataclass
class _InterruptedReviewResult:
    """Minimal ReviewResultProtocol implementation for interrupted reviews.

    This local class avoids importing from src.infra.clients which would
    violate the layered architecture (orchestration cannot import infra.clients).
    """

    passed: bool
    issues: list[object]
    parse_error: str | None
    fatal_error: bool
    review_log_path: Path | None
    interrupted: bool = True


@dataclass
class _FatalReviewResult:
    """Minimal ReviewResultProtocol implementation for fatal review errors."""

    passed: bool
    issues: list[object]
    parse_error: str | None
    fatal_error: bool
    review_log_path: Path | None
    interrupted: bool = False


@dataclass
class ReviewRunnerConfig:
    """Configuration for ReviewRunner behavior.

    Attributes:
        max_review_retries: Maximum number of review retry attempts.
        review_timeout: Timeout in seconds for review operations.
        thinking_mode: Deprecated, kept for backward compatibility.
        capture_session_log: Deprecated, kept for backward compatibility.
    """

    max_review_retries: int = 3
    review_timeout: int = 1200
    # Deprecated fields (kept for backward compatibility with orchestrator)
    thinking_mode: str | None = None
    capture_session_log: bool = False


@dataclass
class ReviewInput:
    """Input for running a code review.

    Bundles all data needed to run a single review check.

    Attributes:
        issue_id: The issue being reviewed.
        repo_path: Path to the repository.
        issue_description: Issue description for scope verification.
        commit_shas: List of commit SHAs to review directly.
        claude_session_id: Optional Claude session ID for external review context.
    """

    issue_id: str
    repo_path: Path
    commit_shas: list[str]
    issue_description: str | None = None
    claude_session_id: str | None = None


@dataclass
class ReviewOutput:
    """Output from a code review check.

    Attributes:
        result: The ReviewResultProtocol from the review.
        session_log_path: Path to the review session log (if captured).
        interrupted: Whether the review was interrupted by SIGINT.
    """

    result: ReviewResultProtocol
    session_log_path: str | None = None
    interrupted: bool = False


@dataclass
class NoProgressInput:
    """Input for no-progress check before review retry.

    Attributes:
        log_path: Path to the session log file.
        log_offset: Byte offset marking the end of the previous attempt.
        previous_commit_hash: Commit hash from the previous attempt.
        current_commit_hash: Commit hash from this attempt.
        spec: Optional ValidationSpec for evidence detection.
    """

    log_path: Path
    log_offset: int
    previous_commit_hash: str | None
    current_commit_hash: str | None
    spec: ValidationSpec | None = None


@dataclass
class ReviewRunner:
    """Code review runner for post-gate validation.

    This class encapsulates the review orchestration logic that was previously
    inline in MalaOrchestrator. It receives a CodeReviewer (protocol) for
    actual review execution.

    The ReviewRunner is responsible for:
    - Running code reviews via the injected CodeReviewer
    - Checking no-progress conditions for retry termination
    - Tracking session log paths

    Usage:
        runner = ReviewRunner(
            code_reviewer=reviewer,
            config=ReviewRunnerConfig(max_review_retries=3),
        )
        output = await runner.run_review(input)

    Attributes:
        code_reviewer: CodeReviewer implementation for running reviews.
        config: Configuration for review behavior.
        gate_checker: Optional GateChecker for no-progress detection.
    """

    code_reviewer: CodeReviewer
    config: ReviewRunnerConfig = field(default_factory=ReviewRunnerConfig)
    gate_checker: GateChecker | None = None

    async def run_review(
        self,
        input: ReviewInput,
        interrupt_event: asyncio.Event | None = None,
    ) -> ReviewOutput:
        """Run code review on the given commit.

        This method invokes the injected CodeReviewer with the appropriate
        parameters derived from the input.

        Args:
            input: ReviewInput with commit_sha, issue_description, etc.
            interrupt_event: Optional event to check for SIGINT interruption.

        Returns:
            ReviewOutput with result and optional session log path.
        """
        import tempfile

        # Check for early interrupt before starting
        guard = InterruptGuard(interrupt_event)
        if guard.is_interrupted():
            logger.info(
                "Review interrupted before starting: issue_id=%s", input.issue_id
            )
            return ReviewOutput(
                result=cast(
                    "ReviewResultProtocol",
                    _InterruptedReviewResult(
                        passed=False,
                        issues=[],
                        parse_error=None,
                        fatal_error=False,
                        review_log_path=None,
                    ),
                ),
                interrupted=True,
            )

        if not input.commit_shas:
            logger.info("Review skipped (no commits): issue_id=%s", input.issue_id)
            return ReviewOutput(
                result=cast(
                    "ReviewResultProtocol",
                    _FatalReviewResult(
                        passed=True,
                        issues=[],
                        parse_error=None,
                        fatal_error=False,
                        review_log_path=None,
                    ),
                ),
                interrupted=False,
            )
        logger.info(
            "Review started: issue_id=%s commits=%d",
            input.issue_id,
            len(input.commit_shas),
        )

        # Create context file if issue_description provided
        # Use NamedTemporaryFile to avoid permission issues on shared systems
        context_file: Path | None = None
        temp_file = None
        if input.issue_description:
            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                prefix=f"review-context-{input.issue_id}-",
                suffix=".txt",
                delete=False,
            )
            # Set context_file immediately so cleanup happens even if write/close fails
            context_file = Path(temp_file.name)

        try:
            # Write and close inside try block to ensure cleanup on failure
            if temp_file is not None and input.issue_description is not None:
                temp_file.write(input.issue_description)
                temp_file.close()
            try:
                result = await self.code_reviewer(
                    context_file=context_file,
                    timeout=self.config.review_timeout,
                    claude_session_id=input.claude_session_id,
                    commit_shas=input.commit_shas,
                    interrupt_event=interrupt_event,
                )
            except Exception as exc:
                logger.exception(
                    "Review failed: issue_id=%s error=%s",
                    input.issue_id,
                    exc,
                )
                return ReviewOutput(
                    result=cast(
                        "ReviewResultProtocol",
                        _FatalReviewResult(
                            passed=False,
                            issues=[],
                            parse_error=str(exc),
                            fatal_error=True,
                            review_log_path=None,
                        ),
                    ),
                    interrupted=guard.is_interrupted(),
                )

            # Check if interrupted during review
            was_interrupted = guard.is_interrupted()

            session_log_path = None
            if result.review_log_path:
                session_log_path = str(result.review_log_path)

            logger.info(
                "Review result: issue_id=%s passed=%s issues=%d interrupted=%s",
                input.issue_id,
                result.passed,
                len(result.issues),
                was_interrupted,
            )

            return ReviewOutput(
                result=result,
                session_log_path=session_log_path,
                interrupted=was_interrupted,
            )
        finally:
            # Clean up context file after review completes (success or failure)
            if context_file is not None and context_file.exists():
                context_file.unlink()

    def check_no_progress(self, input: NoProgressInput) -> bool:
        """Check if no progress was made since the last review attempt.

        No progress is detected when the commit hash hasn't changed and
        there are no uncommitted changes in the working tree.

        This should be called before running a review retry to avoid
        wasting resources on a review that will likely fail again.

        Args:
            input: NoProgressInput with log_path, offsets, and commit hashes.

        Returns:
            True if no progress was made, False if progress was detected.

        Raises:
            ValueError: If gate_checker is not set (required for no-progress).
        """
        if self.gate_checker is None:
            raise ValueError("gate_checker must be set for no-progress detection")

        return self.gate_checker.check_no_progress(
            log_path=input.log_path,
            log_offset=input.log_offset,
            previous_commit_hash=input.previous_commit_hash,
            current_commit_hash=input.current_commit_hash,
            spec=cast("ValidationSpecProtocol | None", input.spec),
            check_validation_evidence=False,  # Only commit/working-tree for reviews
        )
