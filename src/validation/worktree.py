"""Git worktree utilities for clean-room validation.

Provides deterministic worktree creation and cleanup with state tracking.
Worktree paths follow the format: {base_dir}/{run_id}/{issue_id}/{attempt}/
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class WorktreeState(Enum):
    """State of a validation worktree."""

    PENDING = "pending"  # Not yet created
    CREATED = "created"  # Successfully created
    REMOVED = "removed"  # Successfully removed
    FAILED = "failed"  # Creation or removal failed
    KEPT = "kept"  # Kept after failure (--keep-worktrees)


@dataclass
class WorktreeConfig:
    """Configuration for worktree operations."""

    base_dir: Path
    """Base directory for all worktrees (e.g., /tmp/mala-worktrees)."""

    keep_on_failure: bool = False
    """If True, keep worktrees on validation failure for debugging."""

    force_remove: bool = True
    """If True, use --force when removing worktrees."""


@dataclass
class WorktreeResult:
    """Result of a worktree operation."""

    path: Path
    """Path to the worktree directory."""

    state: WorktreeState
    """Current state of the worktree."""

    error: str | None = None
    """Error message if operation failed."""


@dataclass
class WorktreeContext:
    """Context for a validation worktree with state tracking."""

    config: WorktreeConfig
    repo_path: Path
    run_id: str
    issue_id: str
    attempt: int

    state: WorktreeState = field(default=WorktreeState.PENDING)
    error: str | None = None
    _path: Path | None = field(default=None, repr=False)

    @property
    def path(self) -> Path:
        """Get the deterministic worktree path."""
        if self._path is None:
            self._path = (
                self.config.base_dir / self.run_id / self.issue_id / str(self.attempt)
            )
        return self._path

    def to_result(self) -> WorktreeResult:
        """Convert to a WorktreeResult for external use."""
        return WorktreeResult(path=self.path, state=self.state, error=self.error)


def create_worktree(
    repo_path: Path,
    commit_sha: str,
    config: WorktreeConfig,
    run_id: str,
    issue_id: str,
    attempt: int,
) -> WorktreeContext:
    """Create a git worktree for validation.

    Args:
        repo_path: Path to the main git repository.
        commit_sha: Commit SHA to checkout in the worktree.
        config: Worktree configuration.
        run_id: Unique identifier for this validation run.
        issue_id: Issue identifier being validated.
        attempt: Attempt number (1-indexed).

    Returns:
        WorktreeContext with state tracking.
    """
    ctx = WorktreeContext(
        config=config,
        repo_path=repo_path.resolve(),
        run_id=run_id,
        issue_id=issue_id,
        attempt=attempt,
    )

    # Ensure parent directories exist
    ctx.path.parent.mkdir(parents=True, exist_ok=True)

    # Remove any existing directory at the path (stale from crashed run)
    if ctx.path.exists():
        shutil.rmtree(ctx.path, ignore_errors=True)

    result = subprocess.run(
        ["git", "worktree", "add", "--detach", str(ctx.path), commit_sha],
        cwd=ctx.repo_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        ctx.state = WorktreeState.FAILED
        ctx.error = _format_git_error("git worktree add", result)
        # Clean up any partial directory
        if ctx.path.exists():
            shutil.rmtree(ctx.path, ignore_errors=True)
        return ctx

    ctx.state = WorktreeState.CREATED
    return ctx


def remove_worktree(
    ctx: WorktreeContext,
    validation_passed: bool,
) -> WorktreeContext:
    """Remove a git worktree, respecting keep_on_failure setting.

    Args:
        ctx: Worktree context from create_worktree.
        validation_passed: Whether validation succeeded.

    Returns:
        Updated WorktreeContext with new state.
    """
    # Honor keep_on_failure for debugging failed validations
    if not validation_passed and ctx.config.keep_on_failure:
        ctx.state = WorktreeState.KEPT
        return ctx

    # Skip removal if worktree was never created
    if ctx.state not in (WorktreeState.CREATED, WorktreeState.KEPT):
        return ctx

    # Try git worktree remove first
    cmd = ["git", "worktree", "remove"]
    if ctx.config.force_remove:
        cmd.append("--force")
    cmd.append(str(ctx.path))

    result = subprocess.run(
        cmd,
        cwd=ctx.repo_path,
        capture_output=True,
        text=True,
    )

    # Even if git worktree remove fails, try to clean up the directory
    if ctx.path.exists():
        try:
            shutil.rmtree(ctx.path)
        except OSError:
            pass

    # Prune the worktree list to clean up the git repo
    subprocess.run(
        ["git", "worktree", "prune"],
        cwd=ctx.repo_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 and ctx.path.exists():
        ctx.state = WorktreeState.FAILED
        ctx.error = _format_git_error("git worktree remove", result)
        return ctx

    ctx.state = WorktreeState.REMOVED
    return ctx


def cleanup_stale_worktrees(
    repo_path: Path,
    config: WorktreeConfig,
    run_id: str | None = None,
) -> int:
    """Clean up stale worktrees from previous runs.

    Args:
        repo_path: Path to the main git repository.
        config: Worktree configuration.
        run_id: If provided, only clean up worktrees for this run.
                If None, clean up all worktrees under base_dir.

    Returns:
        Number of worktrees cleaned up.
    """
    cleaned = 0
    base = config.base_dir

    if not base.exists():
        return 0

    if run_id:
        # Clean up specific run
        run_dir = base / run_id
        if run_dir.exists():
            cleaned += _cleanup_run_dir(repo_path, run_dir)
    else:
        # Clean up all runs
        for run_dir in base.iterdir():
            if run_dir.is_dir():
                cleaned += _cleanup_run_dir(repo_path, run_dir)

    # Prune the worktree list
    subprocess.run(
        ["git", "worktree", "prune"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    return cleaned


def _cleanup_run_dir(repo_path: Path, run_dir: Path) -> int:
    """Clean up all worktrees in a run directory."""
    cleaned = 0

    for issue_dir in run_dir.iterdir():
        if not issue_dir.is_dir():
            continue

        for attempt_dir in issue_dir.iterdir():
            if not attempt_dir.is_dir():
                continue

            # Try git worktree remove, then force delete
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(attempt_dir)],
                cwd=repo_path,
                capture_output=True,
                text=True,
            )

            if attempt_dir.exists():
                try:
                    shutil.rmtree(attempt_dir)
                except OSError:
                    continue

            cleaned += 1

    # Remove empty run/issue directories
    try:
        for issue_dir in run_dir.iterdir():
            if issue_dir.is_dir() and not any(issue_dir.iterdir()):
                issue_dir.rmdir()
        if run_dir.exists() and not any(run_dir.iterdir()):
            run_dir.rmdir()
    except OSError:
        pass

    return cleaned


def _format_git_error(cmd_name: str, result: subprocess.CompletedProcess[str]) -> str:
    """Format a git command error message."""
    msg = f"{cmd_name} exited {result.returncode}"
    stderr = result.stderr.strip()
    if stderr:
        # Truncate long stderr
        if len(stderr) > 200:
            stderr = stderr[:200] + "..."
        msg = f"{msg}: {stderr}"
    return msg
