"""Protocol definitions for pipeline stage abstractions.

This module defines Protocol classes that enable dependency injection and
testability for the MalaOrchestrator's pipeline stages. Each protocol
represents a stage boundary that the orchestrator interacts with.

Design principles:
- Protocols use structural typing (typing.Protocol) for flexibility
- Methods match exactly what the orchestrator actually calls
- Result types are referenced from existing modules, not redefined
- BeadsClient, ReviewRunner, and QualityGate conform to these protocols

Usage:
    These protocols enable:
    1. Mock implementations for unit testing the orchestrator
    2. Alternative implementations (e.g., in-memory issue tracker)
    3. Clear contracts between orchestrator and its dependencies
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from .cerberus_review import ReviewResult
    from .models import EpicVerdict
    from .quality_gate import CommitResult, GateResult, ValidationEvidence
    from .session_log_parser import JsonlEntry
    from .validation.spec import ValidationSpec


@runtime_checkable
class LogProvider(Protocol):
    """Protocol for abstracting SDK log storage and schema.

    Provides methods for accessing session logs without hardcoding filesystem
    paths or Claude SDK's internal log format. This enables:
    - Testing with mock log providers that return synthetic events
    - Future support for remote log storage or SDK API access
    - Isolation from SDK log format changes

    The canonical implementation is FileSystemLogProvider, which reads JSONL
    logs from the Claude SDK's ~/.claude/projects/{encoded-path}/ directory.
    Test implementations can return in-memory events for isolation.

    Methods:
        get_log_path: Get the filesystem path for a session log.
        iter_events: Iterate over typed log entries from a session.
        get_end_offset: Get the byte offset at the end of a log file.
    """

    def get_log_path(self, repo_path: Path, session_id: str) -> Path:
        """Get the filesystem path for a session's log file.

        This method computes the expected log file path based on the repo
        and session. The path may or may not exist yet.

        Args:
            repo_path: Path to the repository the session was run in.
            session_id: Claude SDK session ID (UUID from ResultMessage).

        Returns:
            Path to the JSONL log file.
        """
        ...

    def iter_events(self, log_path: Path, offset: int = 0) -> Iterator[JsonlEntry]:
        """Iterate over parsed JSONL entries from a log file.

        Reads the file starting from the given byte offset and yields
        structured entries. This enables incremental parsing across
        retry attempts.

        Args:
            log_path: Path to the JSONL log file.
            offset: Byte offset to start reading from (default 0).

        Yields:
            JsonlEntry objects for each successfully parsed JSON line.
            The entry field contains the typed LogEntry if parsing succeeded.

        Note:
            - Lines that fail UTF-8 decoding are silently skipped
            - Empty lines are silently skipped
            - Lines that fail JSON parsing are silently skipped
            - If file doesn't exist, yields nothing
        """
        ...

    def get_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        """Get the byte offset at the end of a log file.

        This is a lightweight method for getting the current file position.
        Use this when you only need the offset for retry scoping, not the
        parsed entries themselves.

        Args:
            log_path: Path to the JSONL log file.
            start_offset: Byte offset to start from (default 0).

        Returns:
            The byte offset at the end of the file, or start_offset if file
            doesn't exist or can't be read.
        """
        ...


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

    async def reset_async(
        self, issue_id: str, log_path: Path | None = None, error: str | None = None
    ) -> bool:
        """Reset an issue back to ready status.

        Called when an implementation attempt fails and the issue should be
        made available for retry.

        Args:
            issue_id: The issue ID to reset.
            log_path: Optional path to the JSONL log file from the attempt.
            error: Optional error message describing the failure.

        Returns:
            True if successfully reset, False otherwise.
        """
        ...

    async def get_epic_children_async(self, epic_id: str) -> set[str]:
        """Get all child issue IDs of an epic.

        Args:
            epic_id: The epic ID to get children for.

        Returns:
            Set of child issue IDs, or empty set if not found or on error.
        """
        ...

    async def get_parent_epic_async(self, issue_id: str) -> str | None:
        """Get the parent epic ID for an issue.

        Args:
            issue_id: The issue ID to find the parent epic for.

        Returns:
            The parent epic ID, or None if no parent epic exists (orphan).
        """
        ...


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
        diff_range: str,
        context_file: Path | None = None,
        timeout: int = 300,
    ) -> ReviewResult:
        """Run code review on a diff range.

        Args:
            diff_range: Git diff range to review (e.g., "baseline..HEAD").
            context_file: Optional path to file with issue description context.
            timeout: Timeout in seconds for the review operation.

        Returns:
            ReviewResult with review outcome. On parse failure,
            returns passed=False with parse_error set.
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
        check_validation_evidence: bool = True,
    ) -> bool:
        """Check if no progress was made since the last attempt.

        No progress is detected when ALL of these are true:
        - The commit hash hasn't changed (or both are None)
        - No uncommitted changes in the working tree
        - (Optionally) No new validation evidence was found after the log offset

        Args:
            log_path: Path to the JSONL log file from agent session.
            log_offset: Byte offset marking the end of the previous attempt.
            previous_commit_hash: Commit hash from the previous attempt.
            current_commit_hash: Commit hash from this attempt.
            spec: Optional ValidationSpec for spec-driven evidence detection.
            check_validation_evidence: If True (default), also check for new validation
                evidence. Set to False for review retries where only commit/working-tree
                changes should gate progress.

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


@runtime_checkable
class EpicVerificationModel(Protocol):
    """Protocol for model-agnostic epic verification.

    Provides an interface for verifying whether code changes satisfy
    an epic's acceptance criteria. The initial implementation uses
    Claude via SDK, but this protocol allows swapping to other models
    (Codex, Gemini, local models) without changing the verifier.

    The canonical implementation is ClaudeEpicVerificationModel in
    src/epic_verifier.py. Test implementations can return predetermined
    verdicts for isolation.
    """

    async def verify(
        self,
        epic_criteria: str,
        diff: str,
        spec_content: str | None,
    ) -> EpicVerdict:
        """Verify if the diff satisfies the epic's acceptance criteria.

        Args:
            epic_criteria: The epic's acceptance criteria text.
            diff: Scoped git diff of child issue commits only.
            spec_content: Optional content of linked spec file.

        Returns:
            Structured verdict with pass/fail and unmet criteria details.
        """
        ...
