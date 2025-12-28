"""Protocol definitions for pipeline stage abstractions.

This module defines Protocol classes that enable dependency injection and
testability for the MalaOrchestrator's pipeline stages. Each protocol
represents a stage boundary that the orchestrator interacts with.

Design principles:
- Protocols use structural typing (typing.Protocol) for flexibility
- Methods are minimal - only what the orchestrator actually calls
- Result types are referenced from existing modules, not redefined
- All methods are async to match the orchestrator's async architecture

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
    from .quality_gate import GateResult
    from .validation.spec import ValidationSpec


# Type alias for review result (references existing CodexReviewResult)
# This allows protocols to use a cleaner name while maintaining compatibility
ReviewResult = "CodexReviewResult"


@runtime_checkable
class IssueProvider(Protocol):
    """Protocol for issue tracking operations.

    Provides methods for fetching, claiming, closing, and marking issues.
    The orchestrator uses this to manage issue lifecycle during parallel
    processing.

    The canonical implementation is BeadsClient, which wraps the bd CLI.
    Test implementations can use in-memory state for isolation.

    Methods:
        get_ready: Fetch list of ready issue IDs, sorted by priority.
        claim: Claim an issue by setting status to in_progress.
        close: Close an issue by setting status to closed.
        mark_needs_followup: Mark an issue as needing follow-up.
    """

    async def get_ready(
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

    async def claim(self, issue_id: str) -> bool:
        """Claim an issue by setting status to in_progress.

        Args:
            issue_id: The issue ID to claim.

        Returns:
            True if successfully claimed, False otherwise.
        """
        ...

    async def close(self, issue_id: str) -> bool:
        """Close an issue by setting status to closed.

        Args:
            issue_id: The issue ID to close.

        Returns:
            True if successfully closed, False otherwise.
        """
        ...

    async def mark_needs_followup(
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


@runtime_checkable
class CodeReviewer(Protocol):
    """Protocol for code review operations.

    Provides a method for reviewing commits and returning structured results.
    The orchestrator uses this to run post-commit code reviews.

    The canonical implementation wraps run_codex_review(), which uses Codex CLI.
    Test implementations can return predetermined results for isolation.

    Methods:
        review: Run code review on a commit and return structured results.
    """

    async def review(
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
            ReviewResult (CodexReviewResult) with review outcome. On parse
            failure after all retries, returns passed=False (fail-closed).
        """
        ...


@runtime_checkable
class GateChecker(Protocol):
    """Protocol for quality gate checking.

    Provides a method for verifying agent work meets quality requirements.
    The orchestrator uses this after each agent attempt to determine if
    the issue was successfully resolved.

    The canonical implementation is QualityGate.
    Test implementations can verify specific conditions for isolation.

    The gate checks:
    - Commit exists with correct issue ID
    - Validation commands ran (parsed from JSONL logs)
    - No-change/obsolete resolutions have rationale and clean tree

    Methods:
        check: Run quality gate check and return results.
    """

    def check(
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
