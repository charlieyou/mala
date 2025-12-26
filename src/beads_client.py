"""BeadsClient: Wrapper for bd CLI calls used by MalaOrchestrator."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# Default timeout for bd/git subprocess calls (seconds)
DEFAULT_COMMAND_TIMEOUT = 30.0


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

    async def _run_subprocess_async(
        self,
        cmd: list[str],
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run subprocess in thread pool with timeout.

        Args:
            cmd: Command to run.
            timeout: Override timeout (uses self.timeout_seconds if None).

        Returns:
            CompletedProcess result, or a failed result on timeout/error.
        """
        effective_timeout = timeout if timeout is not None else self.timeout_seconds

        def run_blocking() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            )

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(run_blocking),
                timeout=effective_timeout,
            )
        except TimeoutError:
            self._log_warning(
                f"Command timed out after {effective_timeout}s: {' '.join(cmd)}"
            )
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="timeout"
            )
        except Exception as e:
            self._log_warning(f"Command failed: {' '.join(cmd)}: {e}")
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr=str(e)
            )

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
    ) -> list[str]:
        """Get list of ready issue IDs via bd CLI, sorted by priority (async version).

        Args:
            exclude_ids: Set of issue IDs to exclude from results.
            epic_id: Optional epic ID to filter by - only return children of this epic.
            only_ids: Optional set of issue IDs to include exclusively.
            suppress_warn_ids: Optional set of issue IDs to suppress from warnings.

        Returns:
            List of issue IDs sorted by priority (lower = higher priority).
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
            filtered.sort(key=lambda i: i.get("priority", 999))
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
