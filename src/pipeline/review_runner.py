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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cerberus_review import ReviewResult
    from src.protocols import CodeReviewer, GateChecker
    from src.validation.spec import ValidationSpec


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
    review_timeout: int = 300
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
        commit_sha: Current commit SHA to review.
        issue_description: Issue description for scope verification.
        baseline_commit: Optional baseline commit for cumulative diff.
    """

    issue_id: str
    repo_path: Path
    commit_sha: str
    issue_description: str | None = None
    baseline_commit: str | None = None


@dataclass
class ReviewOutput:
    """Output from a code review check.

    Attributes:
        result: The ReviewResult from the review.
        session_log_path: Path to the review session log (if captured).
    """

    result: ReviewResult
    session_log_path: str | None = None


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
            config=ReviewRunnerConfig(max_review_retries=2),
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

    async def run_review(self, input: ReviewInput) -> ReviewOutput:
        """Run code review on the given commit.

        This method invokes the injected CodeReviewer with the appropriate
        parameters derived from the input.

        Args:
            input: ReviewInput with commit_sha, issue_description, etc.

        Returns:
            ReviewOutput with result and optional session log path.
        """
        import tempfile

        # Build diff range
        baseline = input.baseline_commit or "HEAD~1"
        diff_range = f"{baseline}..{input.commit_sha}"

        # Create context file if issue_description provided
        context_file: Path | None = None
        if input.issue_description:
            context_dir = Path(tempfile.gettempdir()) / "claude"
            context_dir.mkdir(parents=True, exist_ok=True)
            context_file = context_dir / f"review-context-{input.issue_id}.txt"
            context_file.write_text(input.issue_description)

        result = await self.code_reviewer(
            diff_range=diff_range,
            context_file=context_file,
            timeout=self.config.review_timeout,
        )

        session_log_path = None
        if result.review_log_path:
            session_log_path = str(result.review_log_path)

        return ReviewOutput(
            result=result,
            session_log_path=session_log_path,
        )

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
            spec=input.spec,
            check_validation_evidence=False,  # Only commit/working-tree for reviews
        )
