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
from src.pipeline.review_formatter import format_review_issues

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import asyncio

    from src.core.models import ReviewInput
    from src.core.protocols.review import CodeReviewer, ReviewResultProtocol
    from src.core.protocols.validation import GateChecker, ValidationSpecProtocol
    from src.core.session_end_result import SessionEndResult
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


def _format_session_end_evidence(session_end_result: SessionEndResult) -> str:
    """Format SessionEndResult as evidence for reviewer context.

    Creates a human-readable summary of session_end trigger results.
    Per spec R5, this is informational - review does NOT auto-fail
    based on session_end status.

    Args:
        session_end_result: The session_end outcome to format.

    Returns:
        Formatted markdown string for the context file.
    """
    parts = [
        "## Session End Validation Results (Informational)",
        "",
        "The following validation checks ran after the implementation completed.",
        "**Note**: These results are for your information only. Review should NOT",
        "auto-fail based on session_end status - consider these as additional context.",
        "",
        f"**Status**: {session_end_result.status}",
    ]

    if session_end_result.reason:
        parts.append(f"**Reason**: {session_end_result.reason}")

    # Format command results if present
    if session_end_result.commands:
        parts.append("")
        parts.append("### Command Results")
        for i, cmd in enumerate(session_end_result.commands, 1):
            status = "PASS" if cmd.passed else "FAIL"
            parts.append(f"- **Command {i}**: `{cmd.ref}` [{status}]")
            if not cmd.passed and cmd.error_message:
                # Truncate long error messages
                error_msg = (
                    cmd.error_message[:500] + "..."
                    if len(cmd.error_message) > 500
                    else cmd.error_message
                )
                parts.append(f"  - Error: {error_msg}")

    # Format code review result if present
    if (
        session_end_result.code_review_result
        and session_end_result.code_review_result.ran
    ):
        cr = session_end_result.code_review_result
        parts.append("")
        parts.append("### Code Review Check")
        passed_str = (
            "PASSED" if cr.passed else "FAILED" if cr.passed is False else "N/A"
        )
        parts.append(f"- **Result**: {passed_str}")
        if cr.findings:
            parts.append(f"- **Findings**: {len(cr.findings)} issue(s)")

    return "\n".join(parts)


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

        # Create context file if issue_description, previous_findings, or author_context provided
        # Use NamedTemporaryFile to avoid permission issues on shared systems
        context_file: Path | None = None
        temp_file = None
        context_parts: list[str] = []
        if input.issue_description:
            context_parts.append(input.issue_description)
        # Include previous findings so reviewer knows what was flagged before
        if input.previous_findings:
            formatted_findings = format_review_issues(
                input.previous_findings, base_path=input.repo_path
            )
            context_parts.append(
                "## Previous Review Findings (for context)\n\n"
                "These issues were flagged in the previous review attempt. "
                "The implementer may have addressed them or disputed them below.\n\n"
                f"{formatted_findings}"
            )
        if input.author_context:
            context_parts.append(
                "## Implementer's Response to Previous Review Findings\n\n"
                "The implementer provided the following context about their changes. "
                "Pay close attention to any claims about false positives or disputed findings - "
                "verify these claims before re-flagging the same issues.\n\n"
                f"{input.author_context}"
            )
        # Include session_end evidence when available (per spec R5: informational only)
        if input.session_end_result and input.session_end_result.status != "skipped":
            context_parts.append(_format_session_end_evidence(input.session_end_result))
        context_text = "\n\n".join(context_parts).strip()
        # Log warning if author_context was provided but didn't make it to context
        # This shouldn't happen in practice but helps debug if wiring is broken
        if input.author_context and not context_text:
            logger.warning(
                "author_context provided but context_text is empty: issue_id=%s",
                input.issue_id,
            )
        if context_text:
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
            if temp_file is not None:
                temp_file.write(context_text)
                temp_file.close()

            result = await self.code_reviewer(
                context_file=context_file,
                timeout=self.config.review_timeout,
                claude_session_id=input.claude_session_id,
                author_context=input.author_context,
                commit_shas=input.commit_shas,
                interrupt_event=interrupt_event,
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
