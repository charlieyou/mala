"""Protocol definitions for pipeline stage abstractions.

This module defines Protocol classes that enable dependency injection and
testability for the MalaOrchestrator's pipeline stages. Each protocol
represents a stage boundary that the orchestrator interacts with.

Design principles:
- Protocols use structural typing (typing.Protocol) for flexibility
- Methods match exactly what the orchestrator actually calls
- Result types are referenced from existing modules, not redefined
- BeadsClient, run_codex_review, and QualityGate conform to these protocols

Usage:
    These protocols enable:
    1. Mock implementations for unit testing the orchestrator
    2. Alternative implementations (e.g., in-memory issue tracker)
    3. Clear contracts between orchestrator and its dependencies
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from .codex_review import CodexReviewResult
    from .quality_gate import CommitResult, GateResult, ValidationEvidence
    from .validation.spec import ValidationSpec


@runtime_checkable
class IssueProvider(Protocol):
    """Protocol for issue tracking operations.

    Provides methods for fetching, claiming, closing, and marking issues.
    The orchestrator uses this to manage issue lifecycle during parallel
    processing.

    The canonical implementation is BeadsClient, which wraps the bd CLI.
    Test implementations can use in-memory state for isolation.

    Methods match BeadsClient's async API exactly so BeadsClient conforms
    to this protocol without adaptation.
    """

    async def get_ready_async(
        self,
        exclude_ids: set[str] | None = None,
        epic_id: str | None = None,
        only_ids: set[str] | None = None,
        suppress_warn_ids: set[str] | None = None,
        prioritize_wip: bool = False,
        focus: bool = True,
    ) -> list[str]:
        """Get list of ready issue IDs, sorted by priority.

        Args:
            exclude_ids: Set of issue IDs to exclude from results.
            epic_id: Optional epic ID to filter by - only return children.
            only_ids: Optional set of issue IDs to include exclusively.
            suppress_warn_ids: Set of issue IDs to suppress from warnings.
            prioritize_wip: If True, sort in_progress issues first.
            focus: If True, group tasks by parent epic.

        Returns:
            List of issue IDs sorted by priority (lower = higher priority).
        """
        ...

    async def claim_async(self, issue_id: str) -> bool:
        """Claim an issue by setting status to in_progress.

        Args:
            issue_id: The issue ID to claim.

        Returns:
            True if successfully claimed, False otherwise.
        """
        ...

    async def close_async(self, issue_id: str) -> bool:
        """Close an issue by setting status to closed.

        Args:
            issue_id: The issue ID to close.

        Returns:
            True if successfully closed, False otherwise.
        """
        ...

    async def mark_needs_followup_async(
        self, issue_id: str, reason: str, log_path: Path | None = None
    ) -> bool:
        """Mark an issue as needing follow-up.

        Called when the quality gate fails and the issue needs manual
        intervention or a follow-up task.

        Args:
            issue_id: The issue ID to mark.
            reason: Description of why the quality gate failed.
            log_path: Optional path to the JSONL log file from the attempt.

        Returns:
            True if successfully marked, False otherwise.
        """
        ...

    async def get_issue_description_async(self, issue_id: str) -> str | None:
        """Get the description of an issue.

        Args:
            issue_id: The issue ID to get description for.

        Returns:
            The issue description string, or None if not found.
        """
        ...

    async def close_eligible_epics_async(self) -> bool:
        """Auto-close epics where all children are complete.

        Returns:
            True if any epics were closed, False otherwise.
        """
        ...

    async def commit_issues_async(self) -> bool:
        """Commit .beads/issues.jsonl if it has changes.

        Returns:
            True if commit succeeded, False otherwise.
        """
        ...


@runtime_checkable
class CodeReviewer(Protocol):
    """Protocol for code review operations.

    Provides a callable interface for reviewing commits and returning
    structured results. The orchestrator uses this to run post-commit
    code reviews.

    The canonical implementation is the run_codex_review function, which
    conforms to this protocol as a callable with matching signature.
    Test implementations can return predetermined results for isolation.
    """

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
        """Run code review on a commit with JSON output and retry logic.

        Args:
            repo_path: Path to the git repository.
            commit_sha: The commit SHA to review.
            max_retries: Maximum number of attempts (default 2).
            issue_description: Issue description for scope verification.
            baseline_commit: Optional baseline commit for cumulative diff.
            capture_session_log: If True, capture session log path.
            thinking_mode: Optional reasoning effort level for reviewer.

        Returns:
            CodexReviewResult with review outcome. On parse failure after
            all retries, returns passed=False (fail-closed).
        """
        ...


@runtime_checkable
class GateChecker(Protocol):
    """Protocol for quality gate checking.

    Provides methods for verifying agent work meets quality requirements.
    The orchestrator uses this after each agent attempt to determine if
    the issue was successfully resolved.

    The canonical implementation is QualityGate, which conforms to this
    protocol. Test implementations can verify specific conditions for isolation.

    Methods match QualityGate's API exactly so QualityGate conforms to this
    protocol without adaptation.
    """

    def check_with_resolution(
        self,
        issue_id: str,
        log_path: Path,
        baseline_timestamp: int | None = None,
        log_offset: int = 0,
        spec: ValidationSpec | None = None,
    ) -> GateResult:
        """Run quality gate check with support for no-op/obsolete resolutions.

        This method is scope-aware and handles special resolution outcomes:
        - ISSUE_NO_CHANGE: Issue already addressed, no commit needed
        - ISSUE_OBSOLETE: Issue no longer relevant, no commit needed
        - ISSUE_ALREADY_COMPLETE: Work done in previous run

        Args:
            issue_id: The issue ID to verify.
            log_path: Path to the JSONL log file from agent session.
            baseline_timestamp: Unix timestamp for commit freshness check.
            log_offset: Byte offset to start parsing from.
            spec: ValidationSpec for scope-aware evidence checking (required).

        Returns:
            GateResult with pass/fail, failure reasons, and resolution.

        Raises:
            ValueError: If spec is not provided.
        """
        ...

    def get_log_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        """Get the byte offset at the end of a log file.

        This is a lightweight method for getting the current file position
        after reading from a given offset. Use this when you only need the
        offset for retry scoping, not the evidence itself.

        Args:
            log_path: Path to the JSONL log file.
            start_offset: Byte offset to start from (default 0).

        Returns:
            The byte offset at the end of the file, or start_offset if file
            doesn't exist or can't be read.
        """
        ...

    def check_no_progress(
        self,
        log_path: Path,
        log_offset: int,
        previous_commit_hash: str | None,
        current_commit_hash: str | None,
        spec: ValidationSpec | None = None,
    ) -> bool:
        """Check if no progress was made since the last attempt.

        No progress is detected when:
        - The commit hash hasn't changed (or both are None)
        - No new validation evidence was found after the log offset

        Args:
            log_path: Path to the JSONL log file from agent session.
            log_offset: Byte offset marking the end of the previous attempt.
            previous_commit_hash: Commit hash from the previous attempt.
            current_commit_hash: Commit hash from this attempt.
            spec: Optional ValidationSpec for spec-driven evidence detection.

        Returns:
            True if no progress was made, False if progress was detected.
        """
        ...

    def parse_validation_evidence_with_spec(
        self, log_path: Path, spec: ValidationSpec, offset: int = 0
    ) -> ValidationEvidence:
        """Parse JSONL log for validation evidence using spec-defined patterns.

        Args:
            log_path: Path to the JSONL log file.
            spec: ValidationSpec defining detection patterns.
            offset: Byte offset to start parsing from (default 0).

        Returns:
            ValidationEvidence with flags indicating which validations ran.
        """
        ...

    def check_commit_exists(
        self, issue_id: str, baseline_timestamp: int | None = None
    ) -> CommitResult:
        """Check if a commit with bd-<issue_id> exists in recent history.

        Searches commits from the last 30 days to accommodate long-running
        work that may span multiple days.

        Args:
            issue_id: The issue ID to search for (without bd- prefix).
            baseline_timestamp: Unix timestamp. If provided, only accepts
                commits created after this time.

        Returns:
            CommitResult indicating whether a matching commit exists.
        """
        ...
