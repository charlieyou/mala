"""Git utility functions for mala.

Provides helpers for getting git repository information.
"""

import codecs
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


async def is_commit_reachable(
    repo_path: Path, commit: str, timeout: float = DEFAULT_GIT_TIMEOUT
) -> bool:
    """Check if a commit is reachable in the local repository.

    Uses `git cat-file -e` to verify the commit object exists locally.
    This is important for shallow clones where a baseline commit may
    exist logically but not be present in the local clone.

    Args:
        repo_path: Path to the git repository.
        commit: The commit SHA to check.
        timeout: Timeout in seconds for git operations.

    Returns:
        True if the commit is reachable locally, False otherwise.
    """
    result = await run_command_async(
        ["git", "cat-file", "-e", commit],
        cwd=repo_path,
        timeout_seconds=timeout,
    )
    return result.ok


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

    Uses: git diff --numstat <from_commit> <to_commit>

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
        ["git", "diff", "--numstat", from_commit, to_commit],
    )

    if not result.ok:
        raise ValueError(f"git diff --numstat failed: {result.stderr}")

    stdout = result.stdout.strip()
    if not stdout:
        return DiffStat(total_lines=0, files_changed=[])

    files_changed: list[str] = []
    total_lines = 0

    for line in stdout.split("\n"):
        # numstat format: "added\tremoved\tfilename"
        # For binary files: "-\t-\tfilename"
        parts = line.split("\t")
        if len(parts) >= 3:
            added, removed, filename = parts[0], parts[1], parts[2]
            # Handle renames: "old_name -> new_name" -> use new_name
            if " => " in filename:
                # Format: "{prefix/}{old => new}{/suffix}" or "old => new"
                filename = _parse_rename_path(filename)
            # Git quotes paths containing spaces or special characters
            if filename.startswith('"') and filename.endswith('"'):
                filename = _unquote_git_path(filename[1:-1])
            files_changed.append(filename)
            # Binary files show "-" for added/removed
            if added != "-":
                total_lines += int(added)
            if removed != "-":
                total_lines += int(removed)

    return DiffStat(total_lines=total_lines, files_changed=files_changed)


def _unquote_git_path(path: str) -> str:
    """Unescape special characters in a git-quoted path.

    Git escapes special characters inside quoted paths using backslash sequences:
    \" for ", \\\\ for \\, \\t for tab, etc. Uses unicode_escape decoder to handle this.
    """
    return codecs.decode(path, "unicode_escape")


def _parse_rename_path(path: str) -> str:
    """Parse git rename path to extract the new filename.

    Handles formats like:
    - "old.py => new.py"
    - "{old.py => new.py}"
    - "dir/{old.py => new.py}"
    - "{dir1 => dir2}/file.py"
    """
    # Simple case: "old => new"
    if " => " in path and "{" not in path:
        return path.split(" => ")[1]

    # Complex case with braces: extract and reconstruct
    # e.g., "dir/{old.py => new.py}" -> "dir/new.py"
    result = []
    i = 0
    while i < len(path):
        if path[i] == "{":
            # Find closing brace
            end = path.index("}", i)
            inner = path[i + 1 : end]
            # Extract the "new" part after " => "
            if " => " in inner:
                new_part = inner.split(" => ")[1]
                result.append(new_part)
            else:
                # Brace segment without rename arrow - preserve it
                result.append(inner)
            i = end + 1
        else:
            result.append(path[i])
            i += 1
    return "".join(result)


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


@dataclass
class GitUtils:
    """Wrapper class for git operations needed by CumulativeReviewRunner.

    This class wraps the module-level async functions to enable dependency
    injection for testing. It holds the repo_path as instance state.

    Attributes:
        repo_path: Path to the git repository.
    """

    repo_path: Path

    async def get_diff_stat(
        self,
        from_commit: str,
        to_commit: str = "HEAD",
    ) -> DiffStat:
        """Get diff statistics between commits.

        Args:
            from_commit: The base commit.
            to_commit: The target commit (default: HEAD).

        Returns:
            DiffStat with total lines changed and list of changed files.
        """
        return await get_diff_stat(self.repo_path, from_commit, to_commit)

    async def get_diff_content(
        self,
        from_commit: str,
        to_commit: str = "HEAD",
    ) -> str:
        """Get unified diff content between commits.

        Args:
            from_commit: The base commit.
            to_commit: The target commit (default: HEAD).

        Returns:
            Unified diff string.
        """
        return await get_diff_content(self.repo_path, from_commit, to_commit)

    async def get_baseline_for_issue(self, issue_id: str) -> str | None:
        """Get the baseline commit for an issue from git history.

        Finds the first commit with "bd-{issue_id}:" prefix and returns its parent.
        Delegates to the module-level get_baseline_for_issue() function.

        Args:
            issue_id: The issue ID (e.g., "mala-123").

        Returns:
            The commit hash of the parent of the first issue commit, or None if:
            - No commits exist for this issue (fresh issue)
            - The first commit is the root commit (no parent)
            - Git commands fail
        """
        return await get_baseline_for_issue(self.repo_path, issue_id)

    async def get_head_commit(self) -> str:
        """Get the current HEAD commit hash (short form).

        Returns:
            Short commit hash of HEAD (uses git's default abbreviation length),
            or empty string if git command fails (e.g., not a git repository).
        """
        return await get_git_commit_async(self.repo_path)

    async def is_commit_reachable(self, commit: str) -> bool:
        """Check if a commit is reachable in the local repository.

        Verifies the commit object exists locally. Important for shallow clones
        where a baseline commit may exist logically but not be present locally.

        Args:
            commit: The commit SHA to check.

        Returns:
            True if the commit is reachable locally, False otherwise.
        """
        return await is_commit_reachable(self.repo_path, commit)
