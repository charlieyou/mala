"""BeadsClient: Wrapper for bd CLI calls used by MalaOrchestrator.

This module provides an async-only client for interacting with the beads issue
tracker via the bd CLI. All public methods are async to support non-blocking
concurrent execution in the orchestrator.

Design note: This client intentionally provides only async implementations.
Sync versions were removed to eliminate duplication and ensure consistent
behavior across the codebase.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

# Default timeout for bd/git subprocess calls (seconds)
DEFAULT_COMMAND_TIMEOUT = 30.0


class SubprocessResult:
    """Result of a subprocess execution, compatible with subprocess.CompletedProcess."""

    def __init__(
        self,
        args: list[str],
        returncode: int,
        stdout: str = "",
        stderr: str = "",
    ):
        self.args = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class BeadsClient:
    """Client for interacting with beads via the bd CLI."""

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
        # Cache for parent epic lookups to avoid repeated subprocess calls
        self._parent_epic_cache: dict[str, str | None] = {}
        # Locks for in-flight lookups to prevent duplicate concurrent calls
        self._parent_epic_locks: dict[str, asyncio.Lock] = {}
        # Cache for epic blocked status lookups
        self._blocked_epic_cache: dict[str, bool] = {}
        # Locks for in-flight blocked epic lookups
        self._blocked_epic_locks: dict[str, asyncio.Lock] = {}

    async def _run_subprocess_async(
        self,
        cmd: list[str],
        timeout: float | None = None,
    ) -> SubprocessResult:
        """Run subprocess asynchronously with timeout and proper termination.

        Uses asyncio subprocess to enable proper process termination on timeout.
        When a timeout occurs, the entire process group is terminated (SIGTERM)
        and then killed (SIGKILL) if it doesn't exit promptly. This ensures that
        any child processes spawned by the command are also terminated.

        Args:
            cmd: Command to run.
            timeout: Override timeout (uses self.timeout_seconds if None).

        Returns:
            SubprocessResult, or a failed result on timeout/error.
        """
        effective_timeout = timeout if timeout is not None else self.timeout_seconds

        # On Unix, use start_new_session to create a new process group
        # This allows us to kill all child processes on timeout
        use_process_group = sys.platform != "win32"

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.repo_path,
                start_new_session=use_process_group,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=effective_timeout,
                )
                return SubprocessResult(
                    args=cmd,
                    returncode=proc.returncode or 0,
                    stdout=stdout_bytes.decode() if stdout_bytes else "",
                    stderr=stderr_bytes.decode() if stderr_bytes else "",
                )
            except TimeoutError:
                # Terminate the subprocess and all its children on timeout
                self._log_warning(
                    f"Command timed out after {effective_timeout}s: {' '.join(cmd)}"
                )
                await self._terminate_process(proc, use_process_group)
                return SubprocessResult(
                    args=cmd, returncode=1, stdout="", stderr="timeout"
                )
        except Exception as e:
            self._log_warning(f"Command failed: {' '.join(cmd)}: {e}")
            return SubprocessResult(args=cmd, returncode=1, stdout="", stderr=str(e))

    async def _terminate_process(
        self, proc: asyncio.subprocess.Process, use_process_group: bool
    ) -> None:
        """Terminate a process and all its children.

        First sends SIGTERM to the process group (or just the process on Windows),
        waits up to 2 seconds for graceful exit, then sends SIGKILL to the entire
        process group to ensure all children are killed (even if the parent exited).

        Args:
            proc: The process to terminate.
            use_process_group: Whether to kill the entire process group.
        """
        if proc.returncode is not None:
            # Process already exited
            return

        pgid = proc.pid if use_process_group else None

        try:
            if pgid is not None:
                # Kill the entire process group
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            else:
                proc.terminate()

            # Give it a short time to terminate gracefully
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except TimeoutError:
                pass  # Will force kill below

            # Always SIGKILL the process group after grace period to ensure
            # all children are killed, even if the parent exited quickly
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass  # Group already gone
            elif proc.returncode is None:
                proc.kill()

            # Ensure we wait for the main process to fully exit
            if proc.returncode is None:
                await proc.wait()
        except ProcessLookupError:
            # Process already exited
            pass

    # --- Async methods (non-blocking, use in async context) ---

    async def get_epic_children_async(self, epic_id: str) -> set[str]:
        """Get IDs of all children of an epic (async version).

        Args:
            epic_id: The epic ID to get children for.

        Returns:
            Set of issue IDs that are children of the epic.
        """
        result = await self._run_subprocess_async(
            ["bd", "dep", "tree", epic_id, "--direction=up", "--json"]
        )
        if result.returncode != 0:
            self._log_warning(f"bd dep tree failed for {epic_id}: {result.stderr}")
            return set()
        try:
            tree = json.loads(result.stdout)
            return {item["id"] for item in tree if item.get("depth", 0) > 0}
        except json.JSONDecodeError:
            return set()

    async def _sort_by_epic_groups(
        self, issues: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Sort issues by epic groups for focus mode.

        Groups issues by parent epic, then sorts:
        1. Groups by (min_priority, max_updated DESC)
        2. Within groups by (priority, updated DESC)

        Orphan tasks (no parent epic) form a virtual group with the same rules.

        Args:
            issues: List of issue dicts to sort.

        Returns:
            Sorted list of issue dicts.
        """
        if not issues:
            return issues

        # Get parent epics for all issues
        issue_ids = [str(i["id"]) for i in issues]
        parent_epics = await self.get_parent_epics_async(issue_ids)

        # Group issues by parent epic (None for orphans)
        groups: dict[str | None, list[dict[str, object]]] = {}
        for issue in issues:
            epic = parent_epics.get(str(issue["id"]))
            if epic not in groups:
                groups[epic] = []
            groups[epic].append(issue)

        def get_priority(issue: dict[str, object]) -> int:
            """Extract priority as int, defaulting to 0."""
            prio = issue.get("priority")
            if prio is None:
                return 0
            if isinstance(prio, int):
                return prio
            return int(str(prio))

        def get_updated_at(issue: dict[str, object]) -> str:
            """Extract updated_at as string, defaulting to empty."""
            val = issue.get("updated_at")
            return str(val) if val is not None else ""

        # Sort within each group by (priority, updated DESC)
        for epic, group_issues in groups.items():
            group_issues.sort(
                key=lambda i: (
                    get_priority(i),
                    self._negate_timestamp(get_updated_at(i)),
                )
            )

        # Compute group sort key: (min_priority, -max_updated)
        def group_sort_key(epic: str | None) -> tuple[int, str]:
            group_issues = groups[epic]
            min_priority = min(get_priority(i) for i in group_issues)
            # Find max updated_at (most recent first)
            max_updated = max(get_updated_at(i) for i in group_issues)
            # Negate by prepending ~ to reverse sort (~ > any alphanum)
            # We want max_updated DESC, so higher updated = earlier in sort
            return (min_priority, self._negate_timestamp(max_updated))

        # Sort groups and flatten
        sorted_epics = sorted(groups.keys(), key=group_sort_key)
        result: list[dict[str, object]] = []
        for epic in sorted_epics:
            result.extend(groups[epic])

        return result

    def _negate_timestamp(self, timestamp: object) -> str:
        """Negate a timestamp string for descending sort.

        Converts each character to its complement relative to 'z'/'9'
        so that lexicographic sort becomes descending.

        Args:
            timestamp: ISO timestamp string or empty string.

        Returns:
            Negated string for descending sort.
        """
        if not timestamp or not isinstance(timestamp, str):
            # Empty/missing timestamps sort last (after all real timestamps)
            return "~"  # ~ sorts after alphanumeric
        # Negate each character: higher chars become lower
        # This works for ISO timestamps like "2025-01-15T10:30:00Z"
        return "".join(chr(255 - ord(c)) for c in timestamp)

    # ───────────────────────────────────────────────────────────────────────────
    # Pure pipeline functions for issue processing (no I/O, no side effects)
    # ───────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_wip_issues(
        base_issues: list[dict[str, object]],
        wip_issues: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Merge WIP issues into base issue list (pure function).

        Args:
            base_issues: List of issues from bd ready.
            wip_issues: List of in_progress issues.

        Returns:
            Combined list with WIP issues appended (avoiding duplicates).
        """
        ready_ids: set[str] = {str(i["id"]) for i in base_issues}
        merged = list(base_issues)
        for wip in wip_issues:
            if str(wip["id"]) not in ready_ids:
                merged.append(wip)
        return merged

    @staticmethod
    def _filter_issues(
        issues: list[dict[str, object]],
        exclude_ids: set[str],
        epic_children: set[str] | None,
        only_ids: set[str] | None,
    ) -> list[dict[str, object]]:
        """Filter issue list by various criteria (pure function).

        Args:
            issues: List of issues to filter.
            exclude_ids: Set of issue IDs to exclude.
            epic_children: Optional set of epic child IDs to include.
            only_ids: Optional set of issue IDs to include exclusively.

        Returns:
            Filtered list of issues.
        """
        return [
            i
            for i in issues
            if str(i["id"]) not in exclude_ids
            and i.get("issue_type") != "epic"
            and (epic_children is None or str(i["id"]) in epic_children)
            and (only_ids is None or str(i["id"]) in only_ids)
        ]

    @staticmethod
    def _filter_blocked_epics(
        issues: list[dict[str, object]],
        parent_epics: dict[str, str | None],
        blocked_epics: set[str],
    ) -> list[dict[str, object]]:
        """Filter out issues with blocked parent epics (pure function).

        Args:
            issues: List of issues to filter.
            parent_epics: Mapping of issue_id -> parent_epic_id.
            blocked_epics: Set of blocked epic IDs.

        Returns:
            Filtered list excluding issues with blocked parent epics.
        """
        return [
            i for i in issues if parent_epics.get(str(i["id"])) not in blocked_epics
        ]

    @staticmethod
    def _add_parent_epic_field(
        issues: list[dict[str, object]],
        parent_epics: dict[str, str | None],
    ) -> list[dict[str, object]]:
        """Add parent_epic field to each issue (pure function).

        Args:
            issues: List of issues to enrich.
            parent_epics: Mapping of issue_id -> parent_epic_id.

        Returns:
            List of issues with parent_epic field added.
        """
        for issue in issues:
            issue["parent_epic"] = parent_epics.get(str(issue["id"]))
        return issues

    def _sort_issues(
        self,
        issues: list[dict[str, object]],
        by_epic_groups: bool,
        prioritize_wip: bool,
    ) -> list[dict[str, object]]:
        """Sort issues by priority with optional grouping (pure function).

        Args:
            issues: List of issues with parent_epic field if by_epic_groups=True.
            by_epic_groups: If True, group by parent epic before sorting.
            prioritize_wip: If True, sort in_progress issues first.

        Returns:
            Sorted list of issues.
        """
        if not issues:
            return issues
        sorted_issues = list(issues)
        if by_epic_groups:
            sorted_issues = self._sort_by_epic_groups_sync(sorted_issues)
        else:
            sorted_issues.sort(key=lambda i: i.get("priority") or 0)
        if prioritize_wip:
            sorted_issues.sort(
                key=lambda i: (0 if i.get("status") == "in_progress" else 1,)
            )
        return sorted_issues

    def _sort_by_epic_groups_sync(
        self, issues: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Sort issues by epic groups for focus mode (pure function).

        Groups issues by parent_epic field, then sorts:
        1. Groups by (min_priority, max_updated DESC)
        2. Within groups by (priority, updated DESC)

        Args:
            issues: List of issue dicts with parent_epic field.

        Returns:
            Sorted list of issue dicts.
        """
        if not issues:
            return issues
        groups: dict[str | None, list[dict[str, object]]] = {}
        for issue in issues:
            epic = issue.get("parent_epic")
            epic_key: str | None = str(epic) if epic is not None else None
            if epic_key not in groups:
                groups[epic_key] = []
            groups[epic_key].append(issue)

        def get_priority(issue: dict[str, object]) -> int:
            prio = issue.get("priority")
            return int(str(prio)) if prio is not None else 0

        def get_updated_at(issue: dict[str, object]) -> str:
            val = issue.get("updated_at")
            return str(val) if val is not None else ""

        for group_issues in groups.values():
            group_issues.sort(
                key=lambda i: (
                    get_priority(i),
                    self._negate_timestamp(get_updated_at(i)),
                )
            )

        def group_sort_key(epic: str | None) -> tuple[int, str]:
            group_issues = groups[epic]
            min_priority = min(get_priority(i) for i in group_issues)
            max_updated = max(get_updated_at(i) for i in group_issues)
            return (min_priority, self._negate_timestamp(max_updated))

        sorted_epics = sorted(groups.keys(), key=group_sort_key)
        result: list[dict[str, object]] = []
        for epic in sorted_epics:
            result.extend(groups[epic])
        return result

    # ───────────────────────────────────────────────────────────────────────────
    # Async I/O helpers (not pure - they do subprocess/network calls)
    # ───────────────────────────────────────────────────────────────────────────

    async def _fetch_ready_issues(self) -> tuple[list[dict[str, object]], bool]:
        """Fetch ready issues from bd CLI.

        Returns:
            Tuple of (issues list, success flag). On failure, returns ([], False).
        """
        result = await self._run_subprocess_async(["bd", "ready", "--json"])
        if result.returncode != 0:
            self._log_warning(f"bd ready failed: {result.stderr}")
            return [], False
        try:
            issues = json.loads(result.stdout)
            return (list(issues), True) if isinstance(issues, list) else ([], True)
        except json.JSONDecodeError:
            return [], False

    async def _fetch_wip_issues(self) -> list[dict[str, object]]:
        """Fetch unblocked in_progress issues from bd CLI."""
        result = await self._run_subprocess_async(
            ["bd", "list", "--status", "in_progress", "--json"]
        )
        if result.returncode != 0:
            return []
        try:
            wip_issues = json.loads(result.stdout)
            if not isinstance(wip_issues, list):
                return []
            return [wip for wip in wip_issues if wip.get("blocked_by") is None]
        except json.JSONDecodeError:
            return []

    async def _fetch_and_filter_issues(
        self,
        exclude_ids: set[str] | None = None,
        epic_id: str | None = None,
        only_ids: set[str] | None = None,
        suppress_warn_ids: set[str] | None = None,
        prioritize_wip: bool = False,
        focus: bool = True,
    ) -> list[dict[str, object]]:
        """Fetch, filter, enrich, and sort ready issues."""
        exclude_ids = exclude_ids or set()
        suppress_warn_ids = suppress_warn_ids or set()
        epic_children: set[str] | None = None
        if epic_id:
            epic_children = await self.get_epic_children_async(epic_id)
            if not epic_children:
                self._log_warning(f"No children found for epic {epic_id}")
                return []
        base, ok = await self._fetch_ready_issues()
        if not ok:
            return []
        wip = await self._fetch_wip_issues() if prioritize_wip else []
        issues = self._merge_wip_issues(base, wip) if prioritize_wip else base
        if only_ids:
            bad = only_ids - {str(i["id"]) for i in issues} - suppress_warn_ids
            if bad:
                self._log_warning(f"Specified IDs not ready: {', '.join(sorted(bad))}")
        issues = self._filter_issues(issues, exclude_ids, epic_children, only_ids)
        if not issues:
            return []
        epics = await self.get_parent_epics_async([str(i["id"]) for i in issues])
        blocked = await self._get_blocked_epics_async({e for e in epics.values() if e})
        issues = self._add_parent_epic_field(
            self._filter_blocked_epics(issues, epics, blocked), epics
        )
        return self._sort_issues(
            issues, by_epic_groups=focus, prioritize_wip=prioritize_wip
        )

    async def get_ready_async(
        self,
        exclude_ids: set[str] | None = None,
        epic_id: str | None = None,
        only_ids: set[str] | None = None,
        suppress_warn_ids: set[str] | None = None,
        prioritize_wip: bool = False,
        focus: bool = True,
    ) -> list[str]:
        """Get list of ready issue IDs via bd CLI, sorted by priority (async version).

        Args:
            exclude_ids: Set of issue IDs to exclude from results.
            epic_id: Optional epic ID to filter by - only return children of this epic.
            only_ids: Optional set of issue IDs to include exclusively.
            suppress_warn_ids: Optional set of issue IDs to suppress from warnings.
            prioritize_wip: If True, sort in_progress issues before open issues.
            focus: If True, group tasks by parent epic and complete one epic at a time.
                When focus=True, groups are sorted by (min_priority, max_updated DESC)
                and within groups by (priority, updated DESC). Orphan tasks form a
                virtual group with the same sorting rules.

        Returns:
            List of issue IDs sorted by priority (lower = higher priority).
            When prioritize_wip is True, in_progress issues come first.
            When focus is True, tasks are grouped by epic.
        """
        filtered = await self._fetch_and_filter_issues(
            exclude_ids=exclude_ids,
            epic_id=epic_id,
            only_ids=only_ids,
            suppress_warn_ids=suppress_warn_ids,
            prioritize_wip=prioritize_wip,
            focus=focus,
        )
        return [str(i["id"]) for i in filtered]

    async def get_ready_issues_async(
        self,
        exclude_ids: set[str] | None = None,
        epic_id: str | None = None,
        only_ids: set[str] | None = None,
        suppress_warn_ids: set[str] | None = None,
        prioritize_wip: bool = False,
        focus: bool = True,
    ) -> list[dict[str, object]]:
        """Get list of ready issues with full metadata, sorted by priority (async version).

        Similar to get_ready_async but returns full issue dicts with parent epic info.
        Used for dry-run preview to display task details before processing.

        Args:
            exclude_ids: Set of issue IDs to exclude from results.
            epic_id: Optional epic ID to filter by - only return children of this epic.
            only_ids: Optional set of issue IDs to include exclusively.
            suppress_warn_ids: Optional set of issue IDs to suppress from warnings.
            prioritize_wip: If True, sort in_progress issues before open issues.
            focus: If True, group tasks by parent epic and complete one epic at a time.

        Returns:
            List of issue dicts with id, title, priority, status, and parent_epic fields.
            Sorted by priority (lower = higher priority) with optional epic grouping.
        """
        return await self._fetch_and_filter_issues(
            exclude_ids=exclude_ids,
            epic_id=epic_id,
            only_ids=only_ids,
            suppress_warn_ids=suppress_warn_ids,
            prioritize_wip=prioritize_wip,
            focus=focus,
        )

    async def claim_async(self, issue_id: str) -> bool:
        """Claim an issue by setting status to in_progress (async version).

        Args:
            issue_id: The issue ID to claim.

        Returns:
            True if successfully claimed, False otherwise.
        """
        result = await self._run_subprocess_async(
            ["bd", "update", issue_id, "--status", "in_progress"]
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
        args = ["bd", "update", issue_id, "--status", "ready"]
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
        result = await self._run_subprocess_async(["bd", "show", issue_id, "--json"])
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
        result = await self._run_subprocess_async(["bd", "show", issue_id, "--json"])
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
        """Commit .beads/issues.jsonl if it has changes (async version).

        Returns:
            True if commit succeeded, False otherwise.
        """
        result = await self._run_subprocess_async(["git", "add", ".beads/issues.jsonl"])
        if result.returncode != 0:
            return False

        commit_result = await self._run_subprocess_async(
            ["git", "commit", "-m", "beads: close completed issues"]
        )
        return commit_result.returncode == 0

    async def close_eligible_epics_async(self) -> bool:
        """Auto-close epics where all children are complete (async version).

        Returns:
            True if any epics were closed, False otherwise.
        """
        result = await self._run_subprocess_async(["bd", "epic", "close-eligible"])
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
                "bd",
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
        result = await self._run_subprocess_async(["bd", "close", issue_id])
        return result.returncode == 0

    async def get_parent_epic_async(self, issue_id: str) -> str | None:
        """Get the parent epic ID for an issue.

        Uses `bd dep tree <id> --direction=down` to get the ancestor chain.
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
                ["bd", "dep", "tree", issue_id, "--direction=down", "--json"]
            )
            if result.returncode != 0:
                self._log_warning(f"bd dep tree failed for {issue_id}: {result.stderr}")
                self._parent_epic_cache[issue_id] = None
                return None
            try:
                tree = json.loads(result.stdout)
                # Find the first ancestor (depth > 0) with issue_type == "epic"
                parent_epic: str | None = None
                for item in tree:
                    if item.get("depth", 0) > 0 and item.get("issue_type") == "epic":
                        parent_epic = item["id"]
                        break

                # Cache for this issue only
                # Note: We don't cache all children of the epic here because
                # get_epic_children_async returns ALL descendants (not just direct
                # children), which would incorrectly cache nested epic children
                # as belonging to the top-level epic.
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

            result = await self._run_subprocess_async(["bd", "show", epic_id, "--json"])
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
