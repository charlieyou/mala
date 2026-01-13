"""Issue provider protocol for issue tracking operations.

This module defines protocols for issue tracking, enabling dependency injection
and testability for the orchestrator's issue lifecycle management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.core.models import OrderPreference

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class IssueResolutionProtocol(Protocol):
    """Protocol for issue resolution records.

    Matches the shape of models.IssueResolution for structural typing.
    """

    outcome: Any
    """The resolution outcome (success, no_change, obsolete, etc.)."""

    rationale: str
    """Explanation for the resolution."""


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
        only_ids: list[str] | None = None,
        suppress_warn_ids: set[str] | None = None,
        include_wip: bool = False,
        focus: bool = True,
        orphans_only: bool = False,
        order_preference: OrderPreference = OrderPreference.EPIC_PRIORITY,
    ) -> list[str]:
        """Get list of ready issue IDs, sorted by priority.

        Args:
            exclude_ids: Set of issue IDs to exclude from results.
            epic_id: Optional epic ID to filter by - only return children.
            only_ids: Optional list of issue IDs to include exclusively.
            suppress_warn_ids: Set of issue IDs to suppress from warnings.
            include_wip: If True, include in_progress issues in the scope.
            focus: If True, group tasks by parent epic.
            orphans_only: If True, only return issues with no parent epic.
            order_preference: Issue ordering (focus, epic-priority, issue-priority, or input).

        Returns:
            List of issue IDs sorted by order_preference (lower priority = higher).
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

    async def reopen_issue_async(self, issue_id: str) -> bool:
        """Reopen an issue by setting status to ready.

        Used by deadlock resolution to reset victim issues so they can be
        picked up again after the blocker completes.

        Args:
            issue_id: The issue ID to reopen.

        Returns:
            True if successfully reopened, False otherwise.
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

    async def add_dependency_async(self, issue_id: str, depends_on_id: str) -> bool:
        """Add a dependency between two issues.

        Creates a "blocks" relationship where depends_on_id blocks issue_id.
        Used by deadlock resolution to record that a victim issue depends on
        the blocker's issue.

        Args:
            issue_id: The issue that depends on another.
            depends_on_id: The issue that blocks issue_id.

        Returns:
            True if dependency added successfully, False otherwise.
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

    async def create_issue_async(
        self,
        title: str,
        description: str,
        priority: str,
        tags: list[str] | None = None,
        parent_id: str | None = None,
    ) -> str | None:
        """Create a new issue for tracking.

        Used to create tracking issues for low-priority review findings (P2/P3)
        that should be addressed later but don't block the current work.

        Args:
            title: Issue title.
            description: Issue description (supports markdown).
            priority: Priority string (P1, P2, P3, etc.).
            tags: Optional list of tags to apply.
            parent_id: Optional parent epic ID to attach this issue to.

        Returns:
            Created issue ID, or None on failure.
        """
        ...

    async def find_issue_by_tag_async(self, tag: str) -> str | None:
        """Find an existing issue with the given tag.

        Used for deduplication when creating tracking issues.

        Args:
            tag: The tag to search for.

        Returns:
            Issue ID if found, None otherwise.
        """
        ...

    async def update_issue_description_async(
        self, issue_id: str, description: str
    ) -> bool:
        """Update an issue's description.

        Used for appending new findings to existing tracking issues.

        Args:
            issue_id: The issue ID to update.
            description: New description content (replaces existing).

        Returns:
            True if successfully updated, False otherwise.
        """
        ...

    async def update_issue_async(
        self,
        issue_id: str,
        *,
        title: str | None = None,
        priority: str | None = None,
    ) -> bool:
        """Update an issue's title and/or priority.

        Used for updating tracking issues when new findings change
        the count or highest priority.

        Args:
            issue_id: The issue ID to update.
            title: New title (optional).
            priority: New priority string like "P2" (optional).

        Returns:
            True if successfully updated, False otherwise.
        """
        ...

    async def get_blocked_count_async(self) -> int | None:
        """Get count of issues that exist but aren't ready.

        Used by watch mode to report how many issues are blocked on
        dependencies or other conditions.

        Returns:
            Count of blocked issues, or None if unknown/unsupported.
        """
        ...
