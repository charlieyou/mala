"""Fake IssueProvider for testing.

FakeIssueProvider implements the IssueProvider protocol with in-memory state,
enabling behavior-based testing without real bd CLI or file system operations.

Observable state:
- claimed: Set of issue IDs that have been claimed
- closed: Set of issue IDs that have been closed
- reset_calls: List of (issue_id, log_path, error) tuples for reset_async calls
- created_issues: List of created issue dicts
- followup_calls: List of (issue_id, reason, log_path) tuples
- reopened: Set of issue IDs that have been reopened
- dependency_calls: List of (issue_id, depends_on_id) tuples
- updated_descriptions: List of (issue_id, description) tuples
- updated_issues: List of (issue_id, title, priority) tuples
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.core.models import OrderPreference


@dataclass
class FakeIssue:
    """In-memory representation of an issue for FakeIssueProvider."""

    id: str
    status: str = "open"
    priority: int = 0  # Match IssueManager.get_priority default
    parent_epic: str | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    title: str = ""
    updated_at: str = ""
    issue_type: str = (
        "task"  # "task" or "epic" - epics are filtered from get_ready_async
    )


class FakeIssueProvider:
    """In-memory IssueProvider implementation for testing.

    Implements all IssueProvider protocol methods with simple in-memory logic.
    State changes are observable via public attributes for test assertions.
    """

    def __init__(self, issues: dict[str, FakeIssue] | None = None) -> None:
        """Initialize with optional pre-populated issues.

        Args:
            issues: Dict mapping issue ID to FakeIssue. Defaults to empty.
        """
        self.issues: dict[str, FakeIssue] = issues or {}

        # Observable state for assertions
        self.claimed: set[str] = set()
        self.closed: set[str] = set()
        self.reset_calls: list[tuple[str, Path | None, str | None]] = []
        self.created_issues: list[dict[str, object]] = []
        self.followup_calls: list[tuple[str, str, Path | None]] = []
        self.reopened: set[str] = set()
        self.dependency_calls: list[tuple[str, str]] = []
        self.updated_descriptions: list[tuple[str, str]] = []
        self.updated_issues: list[tuple[str, str | None, str | None]] = []
        self.commit_calls: int = 0
        self.close_epics_calls: int = 0

        # Tag index for find_issue_by_tag_async
        self._tag_to_issue: dict[str, str] = {}
        for issue_id, issue in self.issues.items():
            for tag in issue.tags:
                self._tag_to_issue[tag] = issue_id

    async def get_ready_async(
        self,
        exclude_ids: set[str] | None = None,
        epic_id: str | None = None,
        only_ids: list[str] | None = None,
        suppress_warn_ids: set[str] | None = None,
        include_wip: bool = False,
        focus: bool = True,
        orphans_only: bool = False,
        order_preference: OrderPreference = OrderPreference.FOCUS,
    ) -> list[str]:
        """Get list of ready issue IDs, sorted by priority.

        Uses IssueManager.sort_issues for sorting to ensure parity with real impl.
        """
        from src.infra.issue_manager import IssueManager

        exclude = exclude_ids or set()
        # Collect eligible issues - include in_progress when include_wip=True
        # "open" status = ready for pickup (matches `bd ready` which returns open+unblocked)
        eligible_statuses = {"open", "in_progress"} if include_wip else {"open"}
        candidates: list[dict[str, object]] = []

        for issue_id, issue in self.issues.items():
            if issue_id in exclude:
                continue
            if issue.status not in eligible_statuses:
                continue
            if only_ids is not None and issue_id not in only_ids:
                continue
            if epic_id is not None and issue.parent_epic != epic_id:
                continue
            if orphans_only and issue.parent_epic is not None:
                continue
            # Filter out epics - matches IssueManager.apply_filters behavior
            if issue.issue_type == "epic":
                continue
            # Convert to dict format expected by IssueManager
            candidates.append(
                {
                    "id": issue_id,
                    "status": issue.status,
                    "priority": issue.priority,
                    "parent_epic": issue.parent_epic,
                    "updated_at": issue.updated_at,
                    "tags": issue.tags,
                }
            )

        # Delegate sorting to IssueManager for behavioral parity
        sorted_issues = IssueManager.sort_issues(
            candidates, focus, include_wip, only_ids, order_preference
        )
        return [str(i["id"]) for i in sorted_issues]

    async def claim_async(self, issue_id: str) -> bool:
        """Claim an issue by setting status to in_progress."""
        if issue_id not in self.issues:
            return False
        self.issues[issue_id].status = "in_progress"
        self.claimed.add(issue_id)
        return True

    async def close_async(self, issue_id: str) -> bool:
        """Close an issue by setting status to closed."""
        if issue_id not in self.issues:
            return False
        self.issues[issue_id].status = "closed"
        self.closed.add(issue_id)
        return True

    async def mark_needs_followup_async(
        self, issue_id: str, reason: str, log_path: Path | None = None
    ) -> bool:
        """Mark an issue as needing follow-up."""
        if issue_id not in self.issues:
            return False
        self.issues[issue_id].status = "needs_followup"
        self.followup_calls.append((issue_id, reason, log_path))
        return True

    async def reopen_issue_async(self, issue_id: str) -> bool:
        """Reopen an issue by setting status to open."""
        if issue_id not in self.issues:
            return False
        self.issues[issue_id].status = "open"
        self.reopened.add(issue_id)
        return True

    async def add_dependency_async(self, issue_id: str, depends_on_id: str) -> bool:
        """Add a dependency between two issues."""
        self.dependency_calls.append((issue_id, depends_on_id))
        return True

    async def get_issue_description_async(self, issue_id: str) -> str | None:
        """Get the description of an issue."""
        if issue_id not in self.issues:
            return None
        return self.issues[issue_id].description

    async def close_eligible_epics_async(self) -> bool:
        """Auto-close epics where all children are complete."""
        self.close_epics_calls += 1
        # Find epics and check if all children are closed
        closed_any = False
        for issue_id, issue in self.issues.items():
            if issue.status == "closed":
                continue
            # Check if this is an epic (has children)
            children = [
                iid
                for iid, i in self.issues.items()
                if i.parent_epic == issue_id and iid != issue_id
            ]
            if children and all(self.issues[c].status == "closed" for c in children):
                issue.status = "closed"
                self.closed.add(issue_id)
                closed_any = True
        return closed_any

    async def commit_issues_async(self) -> bool:
        """Commit .beads/issues.jsonl if it has changes."""
        self.commit_calls += 1
        return True

    async def reset_async(
        self, issue_id: str, log_path: Path | None = None, error: str | None = None
    ) -> bool:
        """Reset an issue back to open status (ready for pickup)."""
        if issue_id not in self.issues:
            return False
        self.issues[issue_id].status = "open"
        self.reset_calls.append((issue_id, log_path, error))
        # Remove from claimed if it was claimed
        self.claimed.discard(issue_id)
        return True

    async def get_epic_children_async(self, epic_id: str) -> set[str]:
        """Get all child issue IDs of an epic."""
        return {
            issue_id
            for issue_id, issue in self.issues.items()
            if issue.parent_epic == epic_id
        }

    async def get_parent_epic_async(self, issue_id: str) -> str | None:
        """Get the parent epic ID for an issue."""
        if issue_id not in self.issues:
            return None
        return self.issues[issue_id].parent_epic

    async def create_issue_async(
        self,
        title: str,
        description: str,
        priority: str,
        tags: list[str] | None = None,
        parent_id: str | None = None,
    ) -> str | None:
        """Create a new issue for tracking."""
        issue_id = f"fake-{len(self.created_issues) + 1}"
        tags_list = tags or []

        new_issue = FakeIssue(
            id=issue_id,
            title=title,
            description=description,
            priority=_priority_to_int(priority),
            tags=tags_list,
            parent_epic=parent_id,
        )
        self.issues[issue_id] = new_issue

        # Update tag index
        for tag in tags_list:
            self._tag_to_issue[tag] = issue_id

        self.created_issues.append(
            {
                "id": issue_id,
                "title": title,
                "description": description,
                "priority": priority,
                "tags": tags_list,
                "parent_id": parent_id,
            }
        )
        return issue_id

    async def find_issue_by_tag_async(self, tag: str) -> str | None:
        """Find an existing issue with the given tag."""
        return self._tag_to_issue.get(tag)

    async def update_issue_description_async(
        self, issue_id: str, description: str
    ) -> bool:
        """Update an issue's description."""
        if issue_id not in self.issues:
            return False
        self.issues[issue_id].description = description
        self.updated_descriptions.append((issue_id, description))
        return True

    async def update_issue_async(
        self,
        issue_id: str,
        *,
        title: str | None = None,
        priority: str | None = None,
    ) -> bool:
        """Update an issue's title and/or priority."""
        if issue_id not in self.issues:
            return False
        if title is not None:
            self.issues[issue_id].title = title
        if priority is not None:
            self.issues[issue_id].priority = _priority_to_int(priority)
        self.updated_issues.append((issue_id, title, priority))
        return True

    async def get_blocked_count_async(self) -> int | None:
        """Get count of blocked task issues.

        Returns the count of task issues with status="blocked".
        Mirrors BeadsClient which runs: bd list --status blocked -t task
        """
        blocked = 0
        for issue in self.issues.values():
            if issue.issue_type == "task" and issue.status == "blocked":
                blocked += 1
        return blocked


def _priority_to_int(priority: str) -> int:
    """Convert priority string like 'P1' to integer."""
    if priority.upper().startswith("P"):
        try:
            return int(priority[1:])
        except ValueError:
            pass
    return 0  # Match IssueManager.get_priority default
