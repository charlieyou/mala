"""BeadsClient: Wrapper for bd CLI calls used by MalaOrchestrator."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class BeadsClient:
    """Client for interacting with beads via the bd CLI."""

    def __init__(
        self,
        repo_path: Path,
        log_warning: Callable[[str], None] | None = None,
    ):
        """Initialize BeadsClient.

        Args:
            repo_path: Path to the repository with beads issues.
            log_warning: Optional callback for logging warnings.
        """
        self.repo_path = repo_path
        self._log_warning = log_warning or (lambda msg: None)

    def get_epic_children(self, epic_id: str) -> set[str]:
        """Get IDs of all children of an epic.

        Args:
            epic_id: The epic ID to get children for.

        Returns:
            Set of issue IDs that are children of the epic.
        """
        result = subprocess.run(
            ["bd", "dep", "tree", epic_id, "--direction=up", "--json"],
            capture_output=True,
            text=True,
            cwd=self.repo_path,
        )
        if result.returncode != 0:
            self._log_warning(f"bd dep tree failed for {epic_id}: {result.stderr}")
            return set()
        try:
            tree = json.loads(result.stdout)
            # First entry is the epic itself (depth 0), children have depth > 0
            return {item["id"] for item in tree if item.get("depth", 0) > 0}
        except json.JSONDecodeError:
            return set()

    def get_ready(
        self,
        exclude_ids: set[str] | None = None,
        epic_id: str | None = None,
        only_ids: set[str] | None = None,
    ) -> list[str]:
        """Get list of ready issue IDs via bd CLI, sorted by priority.

        Args:
            exclude_ids: Set of issue IDs to exclude from results.
            epic_id: Optional epic ID to filter by - only return children of this epic.
            only_ids: Optional set of issue IDs to include exclusively. If provided,
                only these IDs will be returned (if they exist and are ready).

        Returns:
            List of issue IDs sorted by priority (lower = higher priority).
        """
        exclude_ids = exclude_ids or set()

        # Get epic children if filtering by epic
        epic_children: set[str] | None = None
        if epic_id:
            epic_children = self.get_epic_children(epic_id)
            if not epic_children:
                self._log_warning(f"No children found for epic {epic_id}")
                return []

        result = subprocess.run(
            ["bd", "ready", "--json"],
            capture_output=True,
            text=True,
            cwd=self.repo_path,
        )
        if result.returncode != 0:
            self._log_warning(f"bd ready failed: {result.stderr}")
            return []
        try:
            issues = json.loads(result.stdout)
            # Get set of ready issue IDs for validation
            ready_issue_ids = {i["id"] for i in issues}

            # Validate only_ids - warn about IDs that aren't ready
            if only_ids:
                not_ready = only_ids - ready_issue_ids
                if not_ready:
                    self._log_warning(
                        f"Specified IDs not ready: {', '.join(sorted(not_ready))}"
                    )

            # Filter and sort by priority (lower number = higher priority)
            filtered = [
                i
                for i in issues
                if i["id"] not in exclude_ids
                and i.get("issue_type") != "epic"  # Skip epics
                and (epic_children is None or i["id"] in epic_children)
                and (only_ids is None or i["id"] in only_ids)
            ]
            filtered.sort(key=lambda i: i.get("priority", 999))
            return [i["id"] for i in filtered]
        except json.JSONDecodeError:
            return []

    def claim(self, issue_id: str) -> bool:
        """Claim an issue by setting status to in_progress.

        Args:
            issue_id: The issue ID to claim.

        Returns:
            True if successfully claimed, False otherwise.
        """
        result = subprocess.run(
            ["bd", "update", issue_id, "--status", "in_progress"],
            capture_output=True,
            text=True,
            cwd=self.repo_path,
        )
        return result.returncode == 0

    def reset(self, issue_id: str) -> None:
        """Reset failed issue to ready status.

        Args:
            issue_id: The issue ID to reset.
        """
        subprocess.run(
            ["bd", "update", issue_id, "--status", "ready"],
            capture_output=True,
            text=True,
            cwd=self.repo_path,
        )

    def get_issue_status(self, issue_id: str) -> str | None:
        """Get the current status of an issue.

        Args:
            issue_id: The issue ID to check.

        Returns:
            The issue status string, or None if not found.
        """
        result = subprocess.run(
            ["bd", "show", issue_id, "--json"],
            capture_output=True,
            text=True,
            cwd=self.repo_path,
        )
        if result.returncode != 0:
            return None
        try:
            issue_data = json.loads(result.stdout)
            # Handle both list and dict responses
            if isinstance(issue_data, list) and issue_data:
                issue_data = issue_data[0]
            if isinstance(issue_data, dict):
                return issue_data.get("status")
        except json.JSONDecodeError:
            pass
        return None

    def commit_issues(self) -> bool:
        """Commit .beads/issues.jsonl if it has changes.

        Returns:
            True if commit succeeded, False otherwise.
        """
        result = subprocess.run(
            ["git", "add", ".beads/issues.jsonl"],
            cwd=self.repo_path,
            capture_output=True,
        )
        if result.returncode != 0:
            return False

        commit_result = subprocess.run(
            ["git", "commit", "-m", "beads: close completed issues"],
            cwd=self.repo_path,
            capture_output=True,
        )
        return commit_result.returncode == 0

    def close_eligible_epics(self) -> bool:
        """Auto-close epics where all children are complete.

        Returns:
            True if any epics were closed, False otherwise.
        """
        result = subprocess.run(
            ["bd", "epic", "close-eligible"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
