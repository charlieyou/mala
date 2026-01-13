"""Workspace setup for spec-based validation.

This module provides helpers for managing the workspace context during
validation runs, including:
- Log directory setup and run ID generation
- Baseline coverage refresh before worktree creation
- Worktree creation and cleanup lifecycle

The SpecRunWorkspace dataclass captures all workspace state needed for
validation, and the setup/cleanup functions manage the lifecycle.
"""

from __future__ import annotations

import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .coverage import BaselineCoverageService
from .spec import ValidationArtifacts
from .worktree import (
    WorktreeConfig,
    WorktreeState,
    create_worktree,
    remove_worktree,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from src.core.protocols.infra import (
        CommandRunnerPort,
        EnvConfigPort,
        LockManagerPort,
    )

    from .spec import ValidationContext, ValidationSpec
    from .worktree import WorktreeContext


class SetupError(Exception):
    """Raised when workspace setup fails.

    Attributes:
        reason: Human-readable failure reason.
        retriable: Whether the failure is potentially retriable.
    """

    def __init__(self, reason: str, retriable: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retriable = retriable


@dataclass
class SpecRunWorkspace:
    """Workspace context for a validation run.

    This dataclass captures all the state needed to run validation:
    - Where to run commands (validation_cwd)
    - Where to store artifacts (artifacts, log_dir)
    - Baseline coverage for "no decrease" mode
    - Optional worktree context for cleanup

    Attributes:
        validation_cwd: Working directory for running validation commands.
            This is either the main repo (in-place validation) or a worktree.
        artifacts: ValidationArtifacts for tracking logs and outputs.
        baseline_percent: Baseline coverage percentage for "no decrease" mode.
            None if using explicit threshold or coverage disabled.
        run_id: Unique identifier for this validation run.
        log_dir: Directory for logs and artifacts.
        worktree_ctx: Optional worktree context if validation uses a worktree.
            Used for cleanup after validation completes.
    """

    validation_cwd: Path
    artifacts: ValidationArtifacts
    baseline_percent: float | None
    run_id: str
    log_dir: Path
    worktree_ctx: WorktreeContext | None


def setup_workspace(
    spec: ValidationSpec,
    context: ValidationContext,
    log_dir: Path | None,
    step_timeout_seconds: float | None,
    command_runner: CommandRunnerPort,
    env_config: EnvConfigPort,
    lock_manager: LockManagerPort,
) -> SpecRunWorkspace:
    """Set up workspace for a validation run.

    This function:
    1. Creates/uses log directory
    2. Generates unique run ID
    3. Initializes artifacts tracking
    4. Refreshes baseline coverage if in "no decrease" mode
    5. Creates worktree if validating a specific commit

    Args:
        spec: What validations to run.
        context: Immutable context for the validation run.
        log_dir: Directory for logs/artifacts. Uses temp dir if None.
        step_timeout_seconds: Optional timeout for baseline refresh commands.
        command_runner: Command runner for executing git commands.
        env_config: Environment configuration for paths.
        lock_manager: Lock manager for file locking.

    Returns:
        SpecRunWorkspace with all context needed for validation.

    Raises:
        SetupError: If baseline refresh or worktree creation fails.
    """
    # Set up log directory
    if log_dir is None:
        log_dir = Path(tempfile.mkdtemp(prefix="mala-validation-logs-"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique run ID
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    issue_id = context.issue_id or "global"

    # Initialize artifacts
    artifacts = ValidationArtifacts(log_dir=log_dir)

    # Check/refresh baseline coverage BEFORE worktree creation
    # This captures baseline from main repo state
    baseline_percent: float | None = None
    if spec.coverage.enabled and spec.coverage.min_percent is None:
        # "No decrease" mode - need to get baseline via service
        baseline_service = BaselineCoverageService(
            context.repo_path,
            env_config=env_config,
            command_runner=command_runner,
            lock_manager=lock_manager,
            coverage_config=spec.yaml_coverage_config,
            step_timeout_seconds=step_timeout_seconds,
        )
        result = baseline_service.refresh_if_stale(spec)
        if not result.success:
            raise SetupError(
                result.error or "Baseline refresh failed",
                retriable=False,
            )
        baseline_percent = result.percent

    # Set up worktree if we have a commit to validate
    worktree_ctx: WorktreeContext | None = None
    validation_cwd: Path

    if context.commit_hash:
        worktree_config = WorktreeConfig(
            base_dir=log_dir / "worktrees",
            keep_on_failure=False,  # Always clean up worktrees
        )
        worktree_ctx = create_worktree(
            repo_path=context.repo_path,
            commit_sha=context.commit_hash,
            config=worktree_config,
            run_id=run_id,
            issue_id=issue_id,
            attempt=1,
            command_runner=command_runner,
        )

        if worktree_ctx.state == WorktreeState.FAILED:
            raise SetupError(
                f"Worktree creation failed: {worktree_ctx.error}",
                retriable=False,
            )

        validation_cwd = worktree_ctx.path
        artifacts.worktree_path = worktree_ctx.path
    else:
        # No commit specified, validate in place
        validation_cwd = context.repo_path

    return SpecRunWorkspace(
        validation_cwd=validation_cwd,
        artifacts=artifacts,
        baseline_percent=baseline_percent,
        run_id=run_id,
        log_dir=log_dir,
        worktree_ctx=worktree_ctx,
    )


def cleanup_workspace(
    workspace: SpecRunWorkspace,
    validation_passed: bool,
    command_runner: CommandRunnerPort,
) -> None:
    """Clean up workspace after validation completes.

    Handles worktree removal with proper pass/fail status handling:
    - On success: removes the worktree
    - On failure: keeps the worktree for debugging

    Args:
        workspace: The workspace to clean up.
        validation_passed: Whether validation succeeded.
        command_runner: Command runner for executing git commands.
    """
    if workspace.worktree_ctx is None:
        return

    worktree_ctx = remove_worktree(
        workspace.worktree_ctx,
        validation_passed=validation_passed,
        command_runner=command_runner,
    )

    # Update artifacts with worktree state
    if worktree_ctx.state == WorktreeState.KEPT:
        workspace.artifacts.worktree_state = "kept"
    elif worktree_ctx.state == WorktreeState.REMOVED:
        workspace.artifacts.worktree_state = "removed"


@contextmanager
def workspace_context(
    spec: ValidationSpec,
    context: ValidationContext,
    log_dir: Path | None,
    step_timeout_seconds: float | None,
    command_runner: CommandRunnerPort,
    env_config: EnvConfigPort,
    lock_manager: LockManagerPort,
) -> Generator[SpecRunWorkspace, None, None]:
    """Context manager for workspace setup and cleanup.

    Ensures cleanup is called even if validation raises an exception.
    On exception, validation_passed=False is used for cleanup.

    Args:
        spec: What validations to run.
        context: Immutable context for the validation run.
        log_dir: Directory for logs/artifacts. Uses temp dir if None.
        step_timeout_seconds: Optional timeout for baseline refresh commands.
        command_runner: Command runner for executing git commands.
        env_config: Environment configuration for paths.
        lock_manager: Lock manager for file locking.

    Yields:
        SpecRunWorkspace with all context needed for validation.

    Raises:
        SetupError: If baseline refresh or worktree creation fails.
    """
    workspace = setup_workspace(
        spec,
        context,
        log_dir,
        step_timeout_seconds,
        command_runner,
        env_config,
        lock_manager,
    )
    validation_passed = False
    try:
        yield workspace
        validation_passed = True
    finally:
        cleanup_workspace(workspace, validation_passed, command_runner)
