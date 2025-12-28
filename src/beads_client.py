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

    async def get_ready_async(
        self,
        exclude_ids: set[str] | None = None,
        epic_id: str | None = None,
        only_ids: set[str] | None = None,
        suppress_warn_ids: set[str] | None = None,
        prioritize_wip: bool = False,
    ) -> list[str]:
        """Get list of ready issue IDs via bd CLI, sorted by priority (async version).

        Args:
            exclude_ids: Set of issue IDs to exclude from results.
            epic_id: Optional epic ID to filter by - only return children of this epic.
            only_ids: Optional set of issue IDs to include exclusively.
            suppress_warn_ids: Optional set of issue IDs to suppress from warnings.
            prioritize_wip: If True, sort in_progress issues before open issues.

        Returns:
            List of issue IDs sorted by priority (lower = higher priority).
            When prioritize_wip is True, in_progress issues come first.
        """
        exclude_ids = exclude_ids or set()
        suppress_warn_ids = suppress_warn_ids or set()

        # Get epic children if filtering by epic
        epic_children: set[str] | None = None
        if epic_id:
            epic_children = await self.get_epic_children_async(epic_id)
            if not epic_children:
                self._log_warning(f"No children found for epic {epic_id}")
                return []

        result = await self._run_subprocess_async(["bd", "ready", "--json"])
        if result.returncode != 0:
            self._log_warning(f"bd ready failed: {result.stderr}")
            return []
        try:
            issues = json.loads(result.stdout)

            # Only include in_progress issues when --wip flag is set
            if prioritize_wip:
                # bd ready doesn't include in_progress issues, fetch them separately
                wip_result = await self._run_subprocess_async(
                    ["bd", "list", "--status", "in_progress", "--json"]
                )
                if wip_result.returncode == 0:
                    try:
                        wip_issues = json.loads(wip_result.stdout)
                        # Filter to unblocked WIP issues not already in ready list
                        ready_ids = {i["id"] for i in issues}
                        for wip in wip_issues:
                            if (
                                wip["id"] not in ready_ids
                                and wip.get("blocked_by") is None
                            ):
                                issues.append(wip)
                    except json.JSONDecodeError:
                        pass

            ready_issue_ids = {i["id"] for i in issues}

            if only_ids:
                not_ready = only_ids - ready_issue_ids - suppress_warn_ids
                if not_ready:
                    self._log_warning(
                        f"Specified IDs not ready: {', '.join(sorted(not_ready))}"
                    )

            filtered = [
                i
                for i in issues
                if i["id"] not in exclude_ids
                and i.get("issue_type") != "epic"
                and (epic_children is None or i["id"] in epic_children)
                and (only_ids is None or i["id"] in only_ids)
            ]

            # Sort by priority, and optionally by status (in_progress first)
            # Default to 0 (P0) for issues without priority to match bd CLI behavior
            if prioritize_wip:
                # in_progress issues get status_order=0, others get 1
                filtered.sort(
                    key=lambda i: (
                        0 if i.get("status") == "in_progress" else 1,
                        i.get("priority") or 0,
                    )
                )
            else:
                filtered.sort(key=lambda i: i.get("priority") or 0)

            return [i["id"] for i in filtered]
        except json.JSONDecodeError:
            return []

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
        When a parent epic is found, all children of that epic are cached
        to minimize subprocess calls for sibling tasks.

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

                # Cache for this issue
                self._parent_epic_cache[issue_id] = parent_epic

                # If we found a parent epic, cache it for all children of that epic
                # This minimizes subprocess calls for sibling tasks
                if parent_epic is not None:
                    children = await self.get_epic_children_async(parent_epic)
                    for child_id in children:
                        if child_id not in self._parent_epic_cache:
                            self._parent_epic_cache[child_id] = parent_epic

                return parent_epic
            except json.JSONDecodeError:
                self._parent_epic_cache[issue_id] = None
                return None

    async def get_parent_epics_async(
        self, issue_ids: list[str]
    ) -> dict[str, str | None]:
        """Get parent epic IDs for multiple issues efficiently.

        Processes unique issues concurrently with epic-level caching:
        - When a task's parent epic is discovered, all children of that epic
          are cached, so sibling tasks don't require additional subprocess calls
        - Duplicate issue IDs in the input are deduped before processing
        - Previously looked-up issues return immediately from cache

        This results in O(unique epics) subprocess calls, not O(unique tasks).

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
