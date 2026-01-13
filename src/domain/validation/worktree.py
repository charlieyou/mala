"""Git worktree utilities for clean-room validation.

Provides deterministic worktree creation and cleanup with state tracking.
Worktree paths follow the format: {base_dir}/{run_id}/{issue_id}/{attempt}/
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols.infra import CommandResultProtocol, CommandRunnerPort


# Pattern for valid path components (alphanumeric, dash, underscore, dot)
# Must not start with dot to prevent hidden files/directories
_SAFE_PATH_COMPONENT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")


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
    _validated: bool = field(default=False, repr=False)

    def _validate_path_components(self) -> None:
        """Validate that path components are safe and don't escape base_dir.

        Raises:
            ValueError: If any path component is unsafe.
        """
        if self._validated:
            return

        # Validate run_id
        if not _SAFE_PATH_COMPONENT.match(self.run_id):
            raise ValueError(
                f"Invalid run_id '{self.run_id}': must be alphanumeric with ._- allowed"
            )

        # Validate issue_id
        if not _SAFE_PATH_COMPONENT.match(self.issue_id):
            raise ValueError(
                f"Invalid issue_id '{self.issue_id}': must be alphanumeric with ._- allowed"
            )

        # Validate attempt is positive
        if self.attempt < 1:
            raise ValueError(f"Invalid attempt '{self.attempt}': must be >= 1")

        self._validated = True

    @property
    def path(self) -> Path:
        """Get the deterministic worktree path.

        Raises:
            ValueError: If path components are unsafe.
        """
        if self._path is None:
            # Validate before constructing path
            self._validate_path_components()

            # Construct path
            candidate = (
                self.config.base_dir / self.run_id / self.issue_id / str(self.attempt)
            )

            # Resolve and verify it's within base_dir
            resolved_base = self.config.base_dir.resolve()
            resolved_path = candidate.resolve()

            if not str(resolved_path).startswith(str(resolved_base) + "/"):
                raise ValueError(
                    f"Computed path '{resolved_path}' escapes base_dir '{resolved_base}'"
                )

            self._path = resolved_path
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
    command_runner: CommandRunnerPort,
) -> WorktreeContext:
    """Create a git worktree for validation.

    Args:
        repo_path: Path to the main git repository.
        commit_sha: Commit SHA to checkout in the worktree.
        config: Worktree configuration.
        run_id: Unique identifier for this validation run.
        issue_id: Issue identifier being validated.
        attempt: Attempt number (1-indexed).
        command_runner: Command runner for executing git commands.

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

    # Validate and get path (may raise ValueError for unsafe inputs)
    try:
        worktree_path = ctx.path
    except ValueError as e:
        ctx.state = WorktreeState.FAILED
        ctx.error = str(e)
        return ctx

    # Ensure parent directories exist
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove any existing path (stale from crashed run)
    if worktree_path.exists():
        if worktree_path.is_file():
            ctx.state = WorktreeState.FAILED
            ctx.error = f"Path exists as file, not directory: {worktree_path}"
            return ctx
        try:
            shutil.rmtree(worktree_path)
        except OSError as e:
            ctx.state = WorktreeState.FAILED
            ctx.error = f"Failed to remove stale worktree: {e}"
            return ctx

    result = command_runner.run(
        ["git", "worktree", "add", "--detach", str(worktree_path), commit_sha],
        cwd=ctx.repo_path,
    )

    if result.returncode != 0:
        ctx.state = WorktreeState.FAILED
        ctx.error = _format_git_error("git worktree add", result)
        # Clean up any partial directory
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)
        return ctx

    ctx.state = WorktreeState.CREATED
    return ctx


def remove_worktree(
    ctx: WorktreeContext,
    validation_passed: bool,
    command_runner: CommandRunnerPort,
) -> WorktreeContext:
    """Remove a git worktree, respecting keep_on_failure setting.

    Args:
        ctx: Worktree context from create_worktree.
        validation_passed: Whether validation succeeded.
        command_runner: Command runner for executing git commands.

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

    result = command_runner.run(
        cmd,
        cwd=ctx.repo_path,
    )

    # Track git command failure
    git_failed = result.returncode != 0
    git_error = _format_git_error("git worktree remove", result) if git_failed else None

    # Only attempt directory cleanup if:
    # 1. Git worktree remove succeeded, OR
    # 2. force_remove is True (user explicitly requested forced cleanup)
    # This protects uncommitted changes when force_remove=False and git remove fails
    dir_cleanup_failed = False
    should_cleanup_dir = not git_failed or ctx.config.force_remove

    if should_cleanup_dir and ctx.path.exists():
        try:
            shutil.rmtree(ctx.path)
        except OSError as e:
            dir_cleanup_failed = True
            if git_error:
                git_error = f"{git_error}; directory cleanup also failed: {e}"
            else:
                git_error = f"Directory cleanup failed: {e}"

    # Clean up empty parent directories up to base_dir
    if not dir_cleanup_failed:
        _cleanup_empty_parents(ctx.path, ctx.config.base_dir)

    # Prune the worktree list to clean up stale git metadata
    command_runner.run(
        ["git", "worktree", "prune"],
        cwd=ctx.repo_path,
    )

    # Report failure if git command failed or directory cleanup failed
    if git_failed or dir_cleanup_failed:
        ctx.state = WorktreeState.FAILED
        ctx.error = git_error
        return ctx

    ctx.state = WorktreeState.REMOVED
    return ctx


def cleanup_stale_worktrees(
    repo_path: Path,
    config: WorktreeConfig,
    command_runner: CommandRunnerPort,
    run_id: str | None = None,
) -> int:
    """Clean up stale worktrees from previous runs.

    Args:
        repo_path: Path to the main git repository.
        config: Worktree configuration.
        command_runner: Command runner for executing git commands.
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
            cleaned += _cleanup_run_dir(repo_path, run_dir, command_runner)
    else:
        # Clean up all runs
        for run_dir in base.iterdir():
            if run_dir.is_dir():
                cleaned += _cleanup_run_dir(repo_path, run_dir, command_runner)

    # Prune the worktree list
    command_runner.run(
        ["git", "worktree", "prune"],
        cwd=repo_path,
    )

    return cleaned


def _cleanup_empty_parents(worktree_path: Path, base_dir: Path) -> None:
    """Remove empty parent directories between worktree_path and base_dir.

    Walks up from worktree_path.parent, removing each empty directory,
    stopping at base_dir (which is not removed).

    Args:
        worktree_path: The removed worktree path.
        base_dir: Stop directory (not removed).
    """
    try:
        base = base_dir.resolve()
        current = worktree_path.resolve().parent

        # Safety: only proceed if current is inside base
        if not current.is_relative_to(base):
            return

        while current != base and current.is_relative_to(base):
            if not current.exists():
                current = current.parent
                continue
            if not current.is_dir():
                break
            # Check if empty
            try:
                next(current.iterdir())
                break  # Not empty
            except StopIteration:
                pass  # Empty, continue
            # Remove empty directory
            current.rmdir()
            current = current.parent
    except (OSError, ValueError):
        pass  # Ignore errors - cleanup is best-effort


def _cleanup_run_dir(
    repo_path: Path, run_dir: Path, command_runner: CommandRunnerPort
) -> int:
    """Clean up all worktrees in a run directory."""
    cleaned = 0

    for issue_dir in run_dir.iterdir():
        if not issue_dir.is_dir():
            continue

        for attempt_dir in issue_dir.iterdir():
            if not attempt_dir.is_dir():
                continue

            # Try git worktree remove, then force delete
            command_runner.run(
                ["git", "worktree", "remove", "--force", str(attempt_dir)],
                cwd=repo_path,
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


def _format_git_error(cmd_name: str, result: CommandResultProtocol) -> str:
    """Format a git command error message."""
    msg = f"{cmd_name} exited {result.returncode}"
    stderr = result.stderr.strip()
    if stderr:
        # Truncate long stderr
        if len(stderr) > 200:
            stderr = stderr[:200] + "..."
        msg = f"{msg}: {stderr}"
    return msg
