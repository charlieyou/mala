"""Git utility functions for mala.

Provides helpers for getting git repository information.
"""

import re
from pathlib import Path

from src.validation.command_runner import CommandRunner, run_command_async

# Default timeout for git commands (seconds)
DEFAULT_GIT_TIMEOUT = 5.0


async def get_git_commit_async(cwd: Path, timeout: float = DEFAULT_GIT_TIMEOUT) -> str:
    """Get the current git commit hash (short) - async version."""
    result = await run_command_async(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=cwd,
        timeout_seconds=timeout,
    )
    if result.ok:
        return result.stdout.strip()
    return ""


async def get_git_branch_async(cwd: Path, timeout: float = DEFAULT_GIT_TIMEOUT) -> str:
    """Get the current git branch name - async version."""
    result = await run_command_async(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=cwd,
        timeout_seconds=timeout,
    )
    if result.ok:
        return result.stdout.strip()
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
    runner = CommandRunner(cwd=repo_path, timeout_seconds=timeout)

    # Find first commit with "bd-{issue_id}:" prefix
    # Using --reverse to get chronological order (oldest first)
    # Escape regex metacharacters in issue_id to avoid matching wrong issues
    # (e.g., "mala-g3h.1" should not match "mala-g3hX1")
    escaped_issue_id = re.escape(issue_id)
    log_result = await runner.run_async(
        [
            "git",
            "log",
            "--oneline",
            "--reverse",
            f"--grep=^bd-{escaped_issue_id}:",
        ],
    )

    if not log_result.ok or not log_result.stdout.strip():
        return None  # No commits for this issue

    # Get first commit hash (first line, first word)
    first_line = log_result.stdout.strip().split("\n")[0]
    first_commit = first_line.split()[0]

    # Get parent of first commit
    parent_result = await runner.run_async(
        ["git", "rev-parse", f"{first_commit}^"],
    )

    if not parent_result.ok:
        return None  # Root commit (no parent)

    return parent_result.stdout.strip()
