"""Git utility functions for mala.

Provides helpers for getting git repository information.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.infra.tools.command_runner import CommandRunner, run_command_async

logger = logging.getLogger(__name__)

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

    baseline = parent_result.stdout.strip()
    logger.debug("Baseline resolved: issue_id=%s commit=%s", issue_id, baseline)
    return baseline


async def get_issue_commits_async(
    repo_path: Path,
    issue_id: str,
    *,
    since_timestamp: int | None = None,
    timeout: float = DEFAULT_GIT_TIMEOUT,
) -> list[str]:
    """Get commit SHAs for an issue, optionally filtered by timestamp.

    Finds commits with "bd-{issue_id}:" prefix, ordered oldest -> newest.

    Args:
        repo_path: Path to the git repository.
        issue_id: The issue ID (e.g., "mala-123").
        since_timestamp: Optional Unix timestamp (seconds). If provided,
            only commits after this time are returned.
        timeout: Timeout in seconds for git operations.

    Returns:
        List of commit SHAs (full length). Empty if none found or git fails.
    """
    runner = CommandRunner(cwd=repo_path, timeout_seconds=timeout)
    escaped_issue_id = re.escape(issue_id)

    cmd = [
        "git",
        "log",
        "--format=%H",
        "--reverse",
        f"--grep=^bd-{escaped_issue_id}:",
    ]
    if since_timestamp is not None and since_timestamp > 0:
        cmd.append(f"--since=@{since_timestamp}")

    log_result = await runner.run_async(cmd)
    if not log_result.ok:
        return []

    return [line.strip() for line in log_result.stdout.splitlines() if line.strip()]


@dataclass(frozen=True)
class DiffStat:
    """Statistics about a git diff."""

    total_lines: int
    files_changed: list[str]


async def get_diff_stat(
    repo_path: Path,
    from_commit: str,
    to_commit: str = "HEAD",
    timeout: float = DEFAULT_GIT_TIMEOUT,
) -> DiffStat:
    """Get diff statistics between commits.

    Uses: git diff --stat <from_commit> <to_commit>

    Args:
        repo_path: Path to the git repository.
        from_commit: The base commit.
        to_commit: The target commit (default: HEAD).
        timeout: Timeout in seconds for git operations.

    Returns:
        DiffStat with total lines changed and list of changed files.
        Returns DiffStat(total_lines=0, files_changed=[]) for empty diff.

    Raises:
        ValueError: If git command fails (e.g., invalid commit).
    """
    runner = CommandRunner(cwd=repo_path, timeout_seconds=timeout)

    result = await runner.run_async(
        ["git", "diff", "--stat", from_commit, to_commit],
    )

    if not result.ok:
        raise ValueError(f"git diff --stat failed: {result.stderr}")

    stdout = result.stdout.strip()
    if not stdout:
        return DiffStat(total_lines=0, files_changed=[])

    lines = stdout.split("\n")
    files_changed: list[str] = []
    total_lines = 0

    for line in lines:
        # File lines look like: "src/foo.py | 10 ++++---"
        # Summary line looks like: "3 files changed, 10 insertions(+), 5 deletions(-)"
        if "|" in line:
            # Extract filename (before the |)
            filename = line.split("|")[0].strip()
            files_changed.append(filename)
        elif "changed" in line:
            # Parse summary line for total lines
            # Matches: "X insertions(+)" and/or "Y deletions(-)"
            insertions_match = re.search(r"(\d+) insertion", line)
            deletions_match = re.search(r"(\d+) deletion", line)
            if insertions_match:
                total_lines += int(insertions_match.group(1))
            if deletions_match:
                total_lines += int(deletions_match.group(1))

    return DiffStat(total_lines=total_lines, files_changed=files_changed)


async def get_diff_content(
    repo_path: Path,
    from_commit: str,
    to_commit: str = "HEAD",
    timeout: float = DEFAULT_GIT_TIMEOUT,
) -> str:
    """Get unified diff content for review.

    Uses: git diff <from_commit> <to_commit>
    Note: Two-argument form, NOT range syntax.

    Args:
        repo_path: Path to the git repository.
        from_commit: The base commit.
        to_commit: The target commit (default: HEAD).
        timeout: Timeout in seconds for git operations.

    Returns:
        Unified diff string. Empty string for no changes.

    Raises:
        ValueError: If git command fails (e.g., invalid commit).
    """
    runner = CommandRunner(cwd=repo_path, timeout_seconds=timeout)

    result = await runner.run_async(
        ["git", "diff", from_commit, to_commit],
    )

    if not result.ok:
        raise ValueError(f"git diff failed: {result.stderr}")

    return result.stdout
