"""BeadsClient: Wrapper for br CLI calls used by MalaOrchestrator.

This module provides an async-only client for interacting with the beads issue
tracker via the br CLI. All public methods are async to support non-blocking
concurrent execution in the orchestrator.

Design note: This client intentionally provides only async implementations.
Sync versions were removed to eliminate duplication and ensure consistent
behavior across the codebase.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from src.core.models import OrderPreference
from src.infra.issue_manager import IssueManager
from src.infra.tools.command_runner import CommandResult, CommandRunner

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

# Default timeout for bd/git subprocess calls (seconds)
DEFAULT_COMMAND_TIMEOUT = 30.0


class BeadsClient:
    """Client for interacting with beads via the br CLI."""

    def __init__(
        self,
        repo_path: Path,
        log_warning: Callable[[str], None] | None = None,
        timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT,
    ):
        """Initialize BeadsClient.

        Args:
            repo_path: Path to the repository with beads issues.
            log_warning: Optional callback for logging warnings.
            timeout_seconds: Timeout for bd/git subprocess calls.
        """
        self.repo_path = repo_path
        self._log_warning = log_warning or (lambda msg: None)
        self.timeout_seconds = timeout_seconds
        self._runner = CommandRunner(cwd=repo_path, timeout_seconds=timeout_seconds)
        # Cache for parent epic lookups to avoid repeated subprocess calls
        self._parent_epic_cache: dict[str, str | None] = {}
        # Locks for in-flight lookups to prevent duplicate concurrent calls
        self._parent_epic_locks: dict[str, asyncio.Lock] = {}
        # Cache for epic blocked status lookups
        self._blocked_epic_cache: dict[str, bool] = {}
        # Locks for in-flight blocked epic lookups
        self._blocked_epic_locks: dict[str, asyncio.Lock] = {}
        # Cache for epic priority lookups (epic_id -> priority)
        self._epic_priority_cache: dict[str, int] = {}

    async def _run_subprocess_async(
        self,
        cmd: list[str],
        timeout: float | None = None,
    ) -> CommandResult:
        """Run subprocess asynchronously with timeout and proper termination.

        Uses CommandRunner for consistent subprocess handling with process-group
        termination. When a timeout occurs, the entire process group is terminated
        (SIGTERM) and then killed (SIGKILL) if it doesn't exit promptly.

        Args:
            cmd: Command to run.
            timeout: Override timeout (uses self.timeout_seconds if None).

        Returns:
            CommandResult with execution details. On timeout, returns
            returncode=1 and stderr="timeout" for backward compatibility.
        """
        effective_timeout = timeout if timeout is not None else self.timeout_seconds
        result = await self._runner.run_async(cmd, timeout=effective_timeout)

        if result.timed_out:
            self._log_warning(
                f"Command timed out after {effective_timeout}s: {' '.join(cmd)}"
            )
            # Return backward-compatible timeout result
            return CommandResult(
                command=cmd,
                returncode=1,
                stdout="",
                stderr="timeout",
                duration_seconds=result.duration_seconds,
                timed_out=True,
            )

        if not result.ok and result.stderr:
            self._log_warning(f"Command failed: {' '.join(cmd)}: {result.stderr}")

        return result

    # --- Async methods (non-blocking, use in async context) ---

    async def get_epic_children_async(self, epic_id: str) -> set[str]:
        """Get IDs of all children of an epic (async version).

        Args:
            epic_id: The epic ID to get children for.

        Returns:
            Set of issue IDs that are children of the epic.
        """
        result = await self._run_subprocess_async(
            ["br", "list", "--parent", epic_id, "--all", "--json"]
        )
        if result.returncode != 0:
            self._log_warning(f"br list --parent failed for {epic_id}: {result.stderr}")
            return set()
        try:
            issues = json.loads(result.stdout)
            return {item["id"] for item in issues}
        except json.JSONDecodeError:
            return set()

    async def _sort_by_epic_groups(
        self, issues: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Sort issues by epic groups for focus mode.

        Groups issues by parent epic, then sorts:
        1. Groups by (epic_priority, max_updated DESC) - falls back to min_priority
           of issues in group if epic_priority is unavailable
        2. Within groups by (priority, updated DESC)

        Orphan tasks (no parent epic) form a virtual group using min_priority.

        Note: This async version fetches parent epics first (populating epic_priority),
        then delegates to IssueManager.sort_by_epic_groups for the actual sorting.

        Args:
            issues: List of issue dicts to sort.

        Returns:
            Sorted list of issue dicts.
        """
        if not issues:
            return issues

        # Enrich issues with parent_epic and epic_priority
        enriched = await self.enrich_with_epics_async(issues)

        # Delegate to IssueManager for pure sorting logic
        return IssueManager.sort_by_epic_groups(enriched)

    def _negate_timestamp(self, timestamp: object) -> str:
        """Negate a timestamp string for descending sort.

        Delegates to IssueManager.negate_timestamp for actual logic.
        """
        return IssueManager.negate_timestamp(timestamp)

    # ───────────────────────────────────────────────────────────────────────────
    # Pipeline steps for _fetch_and_filter_issues
    #
    # Design: Pipeline has two types of steps:
    # - I/O steps (fetch_ready_issues_async, fetch_wip_issues_async, enrich_with_epics_async):
    #   Perform async/subprocess calls, return raw data
    # - Pure transformation steps (in IssueManager):
    #   Static methods with no I/O, directly testable with fixture data
    #
    # BeadsClient provides raw I/O methods; IssueManager handles all processing.
    # ───────────────────────────────────────────────────────────────────────────

    async def fetch_ready_issues_async(self) -> tuple[list[dict[str, object]], bool]:
        """Fetch ready issues from br CLI (raw I/O, no processing).

        Returns:
            Tuple of (issues list, success flag). Returns ([], False) on error.
        """
        result = await self._run_subprocess_async(
            ["br", "ready", "--json", "-t", "task", "--limit", "0"]
        )
        if result.returncode != 0:
            self._log_warning(f"br ready failed: {result.stderr}")
            return [], False
        try:
            issues = json.loads(result.stdout)
            return (list(issues), True) if isinstance(issues, list) else ([], True)
        except json.JSONDecodeError:
            return [], False

    async def fetch_wip_issues_async(self) -> list[dict[str, object]]:
        """Fetch in_progress issues from br CLI (raw I/O, no processing)."""
        result = await self._run_subprocess_async(
            [
                "br",
                "list",
                "--status",
                "in_progress",
                "--json",
                "-t",
                "task",
                "--limit",
                "0",
            ]
        )
        if result.returncode != 0:
            return []
        try:
            wip = json.loads(result.stdout)
            if not isinstance(wip, list):
                return []
            return list(wip)
        except json.JSONDecodeError:
            return []

    async def enrich_with_epics_async(
        self, issues: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Add parent_epic and epic_priority info (I/O step).

        Returns a new list with enriched copies; does not mutate input.
        Each issue gets:
        - parent_epic: The ID of the parent epic (or None for orphans)
        - epic_priority: The priority of the parent epic (or None for orphans)
        """
        if not issues:
            return issues
        ids = [str(i["id"]) for i in issues]
        epics = await self.get_parent_epics_async(ids)
        result = []
        for i in issues:
            epic_id = epics.get(str(i["id"]))
            epic_priority = self._epic_priority_cache.get(epic_id) if epic_id else None
            result.append({**i, "parent_epic": epic_id, "epic_priority": epic_priority})
        return result

    # ───────────────────────────────────────────────────────────────────────────
    # Legacy pipeline methods (delegate to public methods or IssueManager)
    # ───────────────────────────────────────────────────────────────────────────

    async def _fetch_base_issues(self) -> tuple[list[dict[str, object]], bool]:
        """Fetch ready issues from br CLI (pipeline step 1).

        Delegates to fetch_ready_issues_async for actual I/O.
        """
        return await self.fetch_ready_issues_async()

    @staticmethod
    def _merge_wip_issues(
        base_issues: list[dict[str, object]], wip_issues: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Merge WIP issues into base list (pipeline step 2, pure function).

        Delegates to IssueManager.merge_wip_issues for actual logic.
        """
        return IssueManager.merge_wip_issues(base_issues, wip_issues)

    @staticmethod
    def _apply_filters(
        issues: list[dict[str, object]],
        exclude_ids: set[str],
        epic_children: set[str] | None,
        only_ids: list[str] | None,
    ) -> list[dict[str, object]]:
        """Apply only_ids and epic filters (pipeline step 3, pure function).

        Delegates to IssueManager.apply_filters for actual logic.
        """
        return IssueManager.apply_filters(issues, exclude_ids, epic_children, only_ids)

    async def _enrich_with_epics(
        self, issues: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Add parent_epic info and filter blocked epics (pipeline step 4).

        Delegates I/O to enrich_with_epics_async and filtering to IssueManager.
        """
        enriched = await self.enrich_with_epics_async(issues)
        if not enriched:
            return enriched
        epic_ids = {str(i["parent_epic"]) for i in enriched if i.get("parent_epic")}
        blocked = await self._get_blocked_epics_async(epic_ids)
        return IssueManager.filter_blocked_epics(enriched, blocked)

    def _sort_issues(
        self,
        issues: list[dict[str, object]],
        focus: bool,
        include_wip: bool,
        only_ids: list[str] | None = None,
        order_preference: OrderPreference = OrderPreference.EPIC_PRIORITY,
    ) -> list[dict[str, object]]:
        """Sort issues by focus mode vs priority (pipeline step 5, pure function).

        Delegates to IssueManager.sort_issues for actual logic.
        """
        return IssueManager.sort_issues(
            issues, focus, include_wip, only_ids, order_preference
        )

    def _sort_by_epic_groups_sync(
        self, issues: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Sort issues by epic groups for focus mode.

        Delegates to IssueManager.sort_by_epic_groups for actual logic.
        """
        return IssueManager.sort_by_epic_groups(issues)

    async def _fetch_wip_issues(self) -> list[dict[str, object]]:
        """Fetch in_progress issues from br CLI."""
        return await self.fetch_wip_issues_async()

    def _warn_missing_ids(
        self,
        only_ids: list[str] | None,
        issues: list[dict[str, object]],
        suppress_ids: set[str],
    ) -> None:
        """Log warning for specified IDs not found in issues.

        Delegates to IssueManager.find_missing_ids for actual logic.
        """
        bad = IssueManager.find_missing_ids(only_ids, issues, suppress_ids)
        if bad:
            self._log_warning(f"Specified IDs not ready: {', '.join(sorted(bad))}")

    async def _resolve_epic_children(self, epic_id: str | None) -> set[str] | None:
        """Resolve epic children, logging warning if epic has none. Returns None to abort."""
        if not epic_id:
            return None  # Sentinel: no epic filter
        children = await self.get_epic_children_async(epic_id)
        if not children:
            self._log_warning(f"No children found for epic {epic_id}")
        return children  # Empty set signals abort, non-empty signals filter

    async def _fetch_and_filter_issues(
        self,
        exclude_ids: set[str] | None = None,
        epic_id: str | None = None,
        only_ids: list[str] | None = None,
        suppress_warn_ids: set[str] | None = None,
        include_wip: bool = False,
        focus: bool = True,
        orphans_only: bool = False,
        order_preference: OrderPreference = OrderPreference.EPIC_PRIORITY,
    ) -> list[dict[str, object]]:
        """Fetch, filter, enrich, and sort ready issues."""
        exclude_ids = exclude_ids or set()
        epic_children = await self._resolve_epic_children(epic_id)
        if epic_id and not epic_children:
            return []
        issues, ok = await self._fetch_base_issues()
        if not ok and not include_wip:
            # WIP fallback: when include_wip=True, continue even if br ready fails
            # so we can still return in-progress issues (intentional design)
            return []
        if include_wip:
            wip = await self._fetch_wip_issues()
            wip = IssueManager.filter_blocked_wip(wip)
            issues = self._merge_wip_issues(issues, wip)
        else:
            # Filter out in_progress issues from br ready when --resume not passed
            # br ready returns both open and in_progress issues by default
            issues = IssueManager.filter_wip_issues(issues)
        self._warn_missing_ids(only_ids, issues, suppress_warn_ids or set())
        filtered = self._apply_filters(issues, exclude_ids, epic_children, only_ids)
        enriched = await self._enrich_with_epics(filtered)
        # Apply orphans_only filter after enrichment (needs parent_epic info)
        if orphans_only:
            enriched = IssueManager.filter_orphans_only(enriched)
        return self._sort_issues(
            enriched, focus, include_wip, only_ids, order_preference
        )

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
        """Get list of ready issue IDs via br CLI, sorted by priority (async version).

        Args:
            exclude_ids: Set of issue IDs to exclude from results.
            epic_id: Optional epic ID to filter by - only return children of this epic.
            only_ids: Optional list of issue IDs to include exclusively.
            suppress_warn_ids: Optional set of issue IDs to suppress from warnings.
            include_wip: If True, include in_progress issues in the scope.
            focus: If True, group tasks by parent epic and complete one epic at a time.
                When focus=True, groups are sorted by (epic_priority, max_updated DESC)
                and within groups by (priority, updated DESC). Falls back to min_priority
                of issues in group if epic_priority is unavailable. Orphan tasks form a
                virtual group using min_priority.
            orphans_only: If True, only return issues with no parent epic.
            order_preference: Issue ordering (focus, epic-priority, issue-priority, or input).

        Returns:
            List of issue IDs sorted by priority (lower = higher priority).
            When include_wip is True, in_progress issues are included in results.
        """
        filtered = await self._fetch_and_filter_issues(
            exclude_ids=exclude_ids,
            epic_id=epic_id,
            only_ids=only_ids,
            suppress_warn_ids=suppress_warn_ids,
            include_wip=include_wip,
            focus=focus,
            orphans_only=orphans_only,
            order_preference=order_preference,
        )
        return [str(i["id"]) for i in filtered]

    async def get_ready_issues_async(
        self,
        exclude_ids: set[str] | None = None,
        epic_id: str | None = None,
        only_ids: list[str] | None = None,
        suppress_warn_ids: set[str] | None = None,
        include_wip: bool = False,
        focus: bool = True,
        orphans_only: bool = False,
        order_preference: OrderPreference = OrderPreference.EPIC_PRIORITY,
    ) -> list[dict[str, object]]:
        """Get list of ready issues with full metadata, sorted by priority (async version).

        Similar to get_ready_async but returns full issue dicts with parent epic info.
        Used for dry-run preview to display task details before processing.

        Args:
            exclude_ids: Set of issue IDs to exclude from results.
            epic_id: Optional epic ID to filter by - only return children of this epic.
            only_ids: Optional list of issue IDs to include exclusively.
            suppress_warn_ids: Optional set of issue IDs to suppress from warnings.
            include_wip: If True, include in_progress issues in the scope.
            focus: Legacy flag for epic grouping (use order_preference instead).
            orphans_only: If True, only return issues with no parent epic.
            order_preference: Issue ordering (focus, epic-priority, issue-priority, or input).

        Returns:
            List of issue dicts with id, title, priority, status, and parent_epic fields.
            Sorted by priority (lower = higher priority) with optional epic grouping.
        """
        return await self._fetch_and_filter_issues(
            exclude_ids=exclude_ids,
            epic_id=epic_id,
            only_ids=only_ids,
            suppress_warn_ids=suppress_warn_ids,
            include_wip=include_wip,
            focus=focus,
            orphans_only=orphans_only,
            order_preference=order_preference,
        )

    async def claim_async(self, issue_id: str) -> bool:
        """Claim an issue by setting status to in_progress (async version).

        Args:
            issue_id: The issue ID to claim.

        Returns:
            True if successfully claimed, False otherwise.
        """
        result = await self._run_subprocess_async(
            ["br", "update", issue_id, "--status", "in_progress"]
        )
        return result.returncode == 0

    async def reset_async(
        self, issue_id: str, log_path: Path | None = None, error: str = ""
    ) -> None:
        """Reset failed issue to ready status with failure context (async version).

        Args:
            issue_id: The issue ID to reset.
            log_path: Optional path to the JSONL log file from the failed attempt.
            error: Optional error summary describing the failure.
        """
        args = ["br", "update", issue_id, "--status", "ready"]
        if log_path or error:
            notes_parts = []
            if error:
                notes_parts.append(f"Failed: {error}")
            if log_path:
                notes_parts.append(f"Log: {log_path}")
            args.extend(["--notes", "\n".join(notes_parts)])
        await self._run_subprocess_async(args)

    async def get_issue_status_async(self, issue_id: str) -> str | None:
        """Get the current status of an issue (async version).

        Args:
            issue_id: The issue ID to check.

        Returns:
            The issue status string, or None if not found.
        """
        result = await self._run_subprocess_async(["br", "show", issue_id, "--json"])
        if result.returncode != 0:
            return None
        try:
            issue_data = json.loads(result.stdout)
            if isinstance(issue_data, list) and issue_data:
                issue_data = issue_data[0]
            if isinstance(issue_data, dict):
                return issue_data.get("status")
        except json.JSONDecodeError:
            pass
        return None

    async def get_issue_description_async(self, issue_id: str) -> str | None:
        """Get the description of an issue (async version).

        Args:
            issue_id: The issue ID to get description for.

        Returns:
            The issue description string, or None if not found.
        """
        result = await self._run_subprocess_async(["br", "show", issue_id, "--json"])
        if result.returncode != 0:
            return None
        try:
            issue_data = json.loads(result.stdout)
            if isinstance(issue_data, list) and issue_data:
                issue_data = issue_data[0]
            if isinstance(issue_data, dict):
                # Build a comprehensive description including title and scope
                parts = []
                if title := issue_data.get("title"):
                    parts.append(f"Title: {title}")
                if desc := issue_data.get("description"):
                    parts.append(f"\n{desc}")
                if acceptance := issue_data.get("acceptance"):
                    parts.append(f"\nAcceptance Criteria:\n{acceptance}")
                return "\n".join(parts) if parts else None
        except json.JSONDecodeError:
            pass
        return None

    async def commit_issues_async(self) -> bool:
        """Export and commit .beads/issues.jsonl if it has changes (async version).

        Uses `bd sync --no-pull --no-push` to ensure the JSONL is exported from
        SQLite before committing. This handles cases where epic closures are
        persisted in SQLite but not yet exported to JSONL (e.g., when the bd
        daemon is slow or not running).

        Returns:
            True if sync succeeded (or no changes to commit), False otherwise.
        """
        result = await self._run_subprocess_async(["br", "sync", "--flush-only"])
        if result.returncode != 0:
            return False

        # Commit the exported JSONL to git
        await self._run_subprocess_async(["git", "add", ".beads/"])
        await self._run_subprocess_async(
            ["git", "commit", "-m", "sync beads issues"]
        )
        # Return True even if commit fails (nothing to commit is ok)
        return True

    async def close_eligible_epics_async(self) -> bool:
        """Auto-close epics where all children are complete (async version).

        Returns:
            True if any epics were closed, False otherwise.
        """
        result = await self._run_subprocess_async(["br", "epic", "close-eligible"])
        return result.returncode == 0 and bool(result.stdout.strip())

    async def mark_needs_followup_async(
        self, issue_id: str, reason: str, log_path: Path | None = None
    ) -> bool:
        """Mark an issue as needing follow-up (async version).

        Args:
            issue_id: The issue ID to mark.
            reason: Description of why the quality gate failed.
            log_path: Optional path to the JSONL log file from the attempt.

        Returns:
            True if successfully marked, False otherwise.
        """
        notes = f"Quality gate failed: {reason}"
        if log_path:
            notes += f"\nLog: {log_path}"
        result = await self._run_subprocess_async(
            [
                "br",
                "update",
                issue_id,
                "--add-label",
                "needs-followup",
                "--notes",
                notes,
            ]
        )
        return result.returncode == 0

    async def close_async(self, issue_id: str) -> bool:
        """Close an issue by setting status to closed (async version).

        Args:
            issue_id: The issue ID to close.

        Returns:
            True if successfully closed, False otherwise.
        """
        result = await self._run_subprocess_async(["br", "close", issue_id])
        return result.returncode == 0

    async def reopen_issue_async(self, issue_id: str) -> bool:
        """Reopen an issue by setting status to open.

        Used by deadlock resolution to reset victim issues so they can be
        picked up again after the blocker completes.

        Args:
            issue_id: The issue ID to reopen.

        Returns:
            True if successfully reopened, False otherwise.
        """
        result = await self._run_subprocess_async(
            ["br", "update", issue_id, "--status", "open"]
        )
        return result.returncode == 0

    async def add_dependency_async(self, issue_id: str, depends_on_id: str) -> bool:
        """Add a dependency between two issues.

        Creates a "blocks" relationship where depends_on_id blocks issue_id.
        Uses `bd dep add <issue_id> <depends_on_id>`.

        Args:
            issue_id: The issue that depends on another.
            depends_on_id: The issue that blocks issue_id.

        Returns:
            True if dependency added successfully, False otherwise.
        """
        result = await self._run_subprocess_async(
            ["br", "dep", "add", issue_id, depends_on_id]
        )
        return result.returncode == 0

    async def create_issue_async(
        self,
        title: str,
        description: str,
        priority: str,
        tags: list[str] | None = None,
        parent_id: str | None = None,
    ) -> str | None:
        """Create a new issue via br CLI (async version).

        Args:
            title: Issue title.
            description: Issue description (supports markdown).
            priority: Priority string (P1, P2, P3, etc.).
            tags: Optional list of tags to apply.
            parent_id: Optional parent epic ID to attach this issue to.

        Returns:
            Created issue ID, or None on failure.
        """
        cmd = [
            "br",
            "create",
            "--description",
            description,
            "--priority",
            priority,
            "--silent",
        ]
        if parent_id:
            cmd.extend(["--parent", parent_id])
        if tags:
            cmd.extend(["--labels", ",".join(tags)])
        cmd.append(title)

        result = await self._run_subprocess_async(cmd)
        if result.returncode != 0:
            self._log_warning(f"bd create failed: {result.stderr}")
            return None

        # Parse issue ID from output (typically "Created issue: <id>" or silent id)
        match = re.search(r"Created issue:\s*(\S+)", result.stdout)
        if match:
            return match.group(1)

        # Try parsing as JSON if the CLI returns JSON
        try:
            data = json.loads(result.stdout)
            if isinstance(data, dict):
                issue_id = data.get("id")
                if issue_id:
                    return str(issue_id)
        except json.JSONDecodeError:
            pass

        # Fallback: try using stripped output as bare ID
        issue_id = result.stdout.strip()
        return issue_id if issue_id else None

    async def find_issue_by_tag_async(self, tag: str) -> str | None:
        """Find an existing issue with the given tag.

        Args:
            tag: The tag to search for.

        Returns:
            Issue ID if found, None otherwise.
        """
        result = await self._run_subprocess_async(
            ["br", "list", "--label", tag, "--json"]
        )
        if result.returncode != 0:
            return None
        try:
            issues = json.loads(result.stdout)
            if isinstance(issues, list) and issues:
                # Return first matching issue (should be only one due to dedup)
                return str(issues[0].get("id", ""))
            return None
        except json.JSONDecodeError:
            return None

    async def update_issue_description_async(
        self, issue_id: str, description: str
    ) -> bool:
        """Update an issue's description.

        Args:
            issue_id: The issue ID to update.
            description: New description content (replaces existing).

        Returns:
            True if successfully updated, False otherwise.
        """
        result = await self._run_subprocess_async(
            ["br", "update", issue_id, "--description", description]
        )
        return result.returncode == 0

    async def update_issue_async(
        self,
        issue_id: str,
        *,
        title: str | None = None,
        priority: str | None = None,
    ) -> bool:
        """Update an issue's title and/or priority.

        Args:
            issue_id: The issue ID to update.
            title: New title (optional).
            priority: New priority string like "P2" (optional).

        Returns:
            True if successfully updated, False otherwise.
        """
        if title is None and priority is None:
            return True  # Nothing to update

        cmd = ["br", "update", issue_id]
        if title is not None:
            cmd.extend(["--title", title])
        if priority is not None:
            cmd.extend(["--priority", priority])

        result = await self._run_subprocess_async(cmd)
        return result.returncode == 0

    async def get_parent_epic_async(self, issue_id: str) -> str | None:
        """Get the parent epic ID for an issue.

        Uses `br dep tree <id>` to get the ancestor chain.
        The parent epic is the first ancestor with issue_type == "epic".
        Results are cached to avoid repeated subprocess calls.

        Args:
            issue_id: The issue ID to find the parent epic for.

        Returns:
            The parent epic ID, or None if no parent epic exists (orphan).
        """
        # Check cache first (no lock needed for read)
        if issue_id in self._parent_epic_cache:
            return self._parent_epic_cache[issue_id]

        # Get or create lock for this issue to prevent duplicate concurrent calls
        if issue_id not in self._parent_epic_locks:
            self._parent_epic_locks[issue_id] = asyncio.Lock()
        lock = self._parent_epic_locks[issue_id]

        async with lock:
            # Check cache again after acquiring lock (another coroutine may have populated it)
            if issue_id in self._parent_epic_cache:
                return self._parent_epic_cache[issue_id]

            result = await self._run_subprocess_async(
                ["br", "dep", "tree", issue_id, "--json"]
            )
            if result.returncode != 0:
                self._log_warning(f"br dep tree failed for {issue_id}: {result.stderr}")
                self._parent_epic_cache[issue_id] = None
                return None
            try:
                tree = json.loads(result.stdout)
                # Find the first ancestor (depth > 0) with issue_type == "epic"
                parent_epic: str | None = None
                for item in tree:
                    if item.get("depth", 0) > 0 and item.get("issue_type") == "epic":
                        epic_id = item.get("id")
                        if epic_id is None:
                            continue  # Skip malformed item
                        parent_epic = str(epic_id)
                        # Cache epic priority for focus mode ordering
                        # Priority may be int or "P1" string format
                        epic_priority = item.get("priority")
                        if epic_priority is not None:
                            try:
                                prio_str = str(epic_priority)
                                # Handle "P1" format by stripping leading P
                                if prio_str.upper().startswith("P"):
                                    prio_str = prio_str[1:]
                                self._epic_priority_cache[parent_epic] = int(prio_str)
                            except (ValueError, IndexError):
                                pass  # Skip if priority is unparseable
                        break

                # Cache for this issue only
                # Note: We don't cache all children of the epic here because
                # get_epic_children_async returns all descendants under --parent,
                # which would incorrectly cache nested epic children as belonging
                # to the top-level epic.
                self._parent_epic_cache[issue_id] = parent_epic

                return parent_epic
            except json.JSONDecodeError:
                self._parent_epic_cache[issue_id] = None
                return None

    async def get_parent_epics_async(
        self, issue_ids: list[str]
    ) -> dict[str, str | None]:
        """Get parent epic IDs for multiple issues efficiently.

        Processes unique issues concurrently with caching:
        - Duplicate issue IDs in the input are deduped before processing
        - Previously looked-up issues return immediately from cache
        - Concurrent lookups for the same issue are deduplicated via locks

        Args:
            issue_ids: List of issue IDs to find parent epics for.

        Returns:
            Dict mapping each issue ID to its parent epic ID (or None for orphans).
        """
        # Dedupe issue_ids to minimize subprocess calls
        unique_ids = list(dict.fromkeys(issue_ids))

        # Process unique issues concurrently (locks prevent duplicate subprocess calls)
        tasks = [self.get_parent_epic_async(issue_id) for issue_id in unique_ids]
        parent_epics = await asyncio.gather(*tasks)

        # Build result mapping from unique lookups
        unique_results = dict(zip(unique_ids, parent_epics, strict=True))

        # Return mapping for all input issue_ids (including duplicates)
        return {issue_id: unique_results[issue_id] for issue_id in issue_ids}

    async def get_epic_blockers_async(self, epic_id: str) -> set[str]:
        """Get the set of issue IDs that are blocking an epic.

        Retrieves the blocked_by field from the epic and returns it as a set.
        These are typically remediation issues created by epic verification
        that must be resolved before the epic can be closed.

        Args:
            epic_id: The epic ID to get blockers for.

        Returns:
            Set of issue IDs that are blocking the epic.
        """
        result = await self._run_subprocess_async(["br", "show", epic_id, "--json"])
        if result.returncode != 0:
            return set()

        try:
            issue_data = json.loads(result.stdout)
            if isinstance(issue_data, list) and issue_data:
                issue_data = issue_data[0]
            if isinstance(issue_data, dict):
                blocked_by = issue_data.get("blocked_by")
                if blocked_by is None:
                    return set()
                # blocked_by can be a string or a list
                if isinstance(blocked_by, str):
                    return {blocked_by} if blocked_by else set()
                if isinstance(blocked_by, list):
                    return {str(b) for b in blocked_by if b}
        except json.JSONDecodeError:
            pass

        return set()

    async def _is_epic_blocked_async(self, epic_id: str) -> bool:
        """Check if an epic is blocked (has blocked_by or status=blocked).

        An epic is considered blocked if:
        - It has status "blocked", OR
        - It has a non-empty blocked_by field (unmet dependencies)

        Results are cached to avoid repeated subprocess calls.

        Args:
            epic_id: The epic ID to check.

        Returns:
            True if the epic is blocked, False otherwise.
        """
        # Check cache first
        if epic_id in self._blocked_epic_cache:
            return self._blocked_epic_cache[epic_id]

        # Get or create lock for this epic
        if epic_id not in self._blocked_epic_locks:
            self._blocked_epic_locks[epic_id] = asyncio.Lock()
        lock = self._blocked_epic_locks[epic_id]

        async with lock:
            # Check cache again after acquiring lock
            if epic_id in self._blocked_epic_cache:
                return self._blocked_epic_cache[epic_id]

            result = await self._run_subprocess_async(["br", "show", epic_id, "--json"])
            if result.returncode != 0:
                # If we can't get epic info, assume not blocked to avoid hiding tasks
                self._blocked_epic_cache[epic_id] = False
                return False

            try:
                issue_data = json.loads(result.stdout)
                if isinstance(issue_data, list) and issue_data:
                    issue_data = issue_data[0]
                if isinstance(issue_data, dict):
                    status = issue_data.get("status")
                    blocked_by = issue_data.get("blocked_by")
                    is_blocked = status == "blocked" or bool(blocked_by)
                    self._blocked_epic_cache[epic_id] = is_blocked
                    return is_blocked
            except json.JSONDecodeError:
                pass

            self._blocked_epic_cache[epic_id] = False
            return False

    async def _get_blocked_epics_async(self, epic_ids: set[str]) -> set[str]:
        """Get the set of epics that are blocked.

        Args:
            epic_ids: Set of epic IDs to check.

        Returns:
            Set of epic IDs that are blocked.
        """
        if not epic_ids:
            return set()

        # Convert to list once to ensure consistent iteration order
        epic_ids_list = list(epic_ids)
        tasks = [self._is_epic_blocked_async(epic_id) for epic_id in epic_ids_list]
        results = await asyncio.gather(*tasks)
        return {
            epic_id
            for epic_id, is_blocked in zip(epic_ids_list, results, strict=True)
            if is_blocked
        }

    async def get_blocked_count_async(self) -> int | None:
        """Get count of issues that exist but aren't ready.

        Used by watch mode to report how many issues are blocked on
        dependencies or other conditions. Returns None if the count
        cannot be determined (e.g., br CLI failure).

        Returns:
            Count of blocked issues, or None on error.
        """
        result = await self._run_subprocess_async(
            ["br", "list", "--status", "blocked", "--json", "-t", "task"]
        )
        if result.returncode != 0:
            return None
        try:
            data = json.loads(result.stdout)
            return len(data) if isinstance(data, list) else None
        except json.JSONDecodeError:
            return None
