"""Git utility functions for mala.

Provides helpers for getting git repository information.
"""

import asyncio
import subprocess
from pathlib import Path

# Default timeout for git commands (seconds)
DEFAULT_GIT_TIMEOUT = 5.0


async def get_git_commit_async(cwd: Path, timeout: float = DEFAULT_GIT_TIMEOUT) -> str:
    """Get the current git commit hash (short) - async version."""

    def run_blocking() -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(run_blocking),
            timeout=timeout,
        )
    except TimeoutError:
        return ""


async def get_git_branch_async(cwd: Path, timeout: float = DEFAULT_GIT_TIMEOUT) -> str:
    """Get the current git branch name - async version."""

    def run_blocking() -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(run_blocking),
            timeout=timeout,
        )
    except TimeoutError:
        return ""


async def get_baseline_for_issue(
    repo_path: Path, issue_id: str, timeout: float = DEFAULT_GIT_TIMEOUT
) -> str | None:
    """Get the baseline commit for an issue from git history.

    Finds the first commit with "bd-{issue_id}:" prefix and returns its parent.
    This allows accurate cumulative diff calculation across resumed sessions.

    Args:
        repo_path: Path to the git repository.
        issue_id: The issue ID (e.g., "mala-123").
        timeout: Timeout in seconds for git operations.

    Returns:
        The commit hash of the parent of the first issue commit, or None if:
        - No commits exist for this issue (fresh issue)
        - The first commit is the root commit (no parent)
        - Git commands fail or timeout
    """

    def run_blocking() -> str | None:
        try:
            # Find first commit with "bd-{issue_id}:" prefix
            # Using --reverse to get chronological order (oldest first)
            log_result = subprocess.run(
                [
                    "git",
                    "log",
                    "--oneline",
                    "--reverse",
                    f"--grep=^bd-{issue_id}:",
                ],
                capture_output=True,
                text=True,
                cwd=repo_path,
            )
            if log_result.returncode != 0 or not log_result.stdout.strip():
                return None  # No commits for this issue

            # Get first commit hash (first line, first word)
            first_line = log_result.stdout.strip().split("\n")[0]
            first_commit = first_line.split()[0]

            # Get parent of first commit
            parent_result = subprocess.run(
                ["git", "rev-parse", f"{first_commit}^"],
                capture_output=True,
                text=True,
                cwd=repo_path,
            )
            if parent_result.returncode != 0:
                return None  # Root commit (no parent)

            return parent_result.stdout.strip()
        except Exception:
            return None

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(run_blocking),
            timeout=timeout,
        )
    except TimeoutError:
        return None
