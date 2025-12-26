"""Unit tests for src/validation/worktree.py - git worktree utilities.

These tests mock subprocess calls to test worktree logic without actually
creating git worktrees.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.validation.worktree import (
    WorktreeConfig,
    WorktreeContext,
    WorktreeResult,
    WorktreeState,
    cleanup_stale_worktrees,
    create_worktree,
    remove_worktree,
)


class TestWorktreeState:
    """Test WorktreeState enum."""

    def test_state_values(self) -> None:
        assert WorktreeState.PENDING.value == "pending"
        assert WorktreeState.CREATED.value == "created"
        assert WorktreeState.REMOVED.value == "removed"
        assert WorktreeState.FAILED.value == "failed"
        assert WorktreeState.KEPT.value == "kept"


class TestWorktreeConfig:
    """Test WorktreeConfig dataclass."""

    def test_defaults(self, tmp_path: Path) -> None:
        config = WorktreeConfig(base_dir=tmp_path)
        assert config.base_dir == tmp_path
        assert config.keep_on_failure is False
        assert config.force_remove is True

    def test_custom_values(self, tmp_path: Path) -> None:
        config = WorktreeConfig(
            base_dir=tmp_path,
            keep_on_failure=True,
            force_remove=False,
        )
        assert config.keep_on_failure is True
        assert config.force_remove is False


class TestWorktreeResult:
    """Test WorktreeResult dataclass."""

    def test_basic_result(self, tmp_path: Path) -> None:
        result = WorktreeResult(
            path=tmp_path / "worktree",
            state=WorktreeState.CREATED,
        )
        assert result.path == tmp_path / "worktree"
        assert result.state == WorktreeState.CREATED
        assert result.error is None

    def test_failed_result(self, tmp_path: Path) -> None:
        result = WorktreeResult(
            path=tmp_path / "worktree",
            state=WorktreeState.FAILED,
            error="git worktree add failed",
        )
        assert result.state == WorktreeState.FAILED
        assert result.error == "git worktree add failed"


class TestWorktreeContext:
    """Test WorktreeContext dataclass."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> WorktreeConfig:
        return WorktreeConfig(base_dir=tmp_path / "worktrees")

    def test_deterministic_path(self, config: WorktreeConfig, tmp_path: Path) -> None:
        ctx = WorktreeContext(
            config=config,
            repo_path=tmp_path / "repo",
            run_id="run-123",
            issue_id="mala-42",
            attempt=1,
        )

        expected = config.base_dir / "run-123" / "mala-42" / "1"
        assert ctx.path == expected

    def test_path_caching(self, config: WorktreeConfig, tmp_path: Path) -> None:
        ctx = WorktreeContext(
            config=config,
            repo_path=tmp_path / "repo",
            run_id="run-123",
            issue_id="mala-42",
            attempt=2,
        )

        # Access path twice - should return same object
        path1 = ctx.path
        path2 = ctx.path
        assert path1 is path2

    def test_to_result(self, config: WorktreeConfig, tmp_path: Path) -> None:
        ctx = WorktreeContext(
            config=config,
            repo_path=tmp_path / "repo",
            run_id="run-123",
            issue_id="mala-42",
            attempt=1,
            state=WorktreeState.CREATED,
        )

        result = ctx.to_result()
        assert isinstance(result, WorktreeResult)
        assert result.path == ctx.path
        assert result.state == WorktreeState.CREATED
        assert result.error is None

    def test_to_result_with_error(self, config: WorktreeConfig, tmp_path: Path) -> None:
        ctx = WorktreeContext(
            config=config,
            repo_path=tmp_path / "repo",
            run_id="run-123",
            issue_id="mala-42",
            attempt=1,
            state=WorktreeState.FAILED,
            error="test error",
        )

        result = ctx.to_result()
        assert result.state == WorktreeState.FAILED
        assert result.error == "test error"


class TestCreateWorktree:
    """Test create_worktree function."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> WorktreeConfig:
        return WorktreeConfig(base_dir=tmp_path / "worktrees")

    def test_create_success(self, config: WorktreeConfig, tmp_path: Path) -> None:
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        )

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            ctx = create_worktree(
                repo_path=tmp_path / "repo",
                commit_sha="abc123",
                config=config,
                run_id="run-1",
                issue_id="mala-10",
                attempt=1,
            )

        assert ctx.state == WorktreeState.CREATED
        assert ctx.error is None
        assert ctx.path == config.base_dir / "run-1" / "mala-10" / "1"

        # Verify git worktree add was called correctly
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[:3] == ["git", "worktree", "add"]
        assert "--detach" in cmd
        assert "abc123" in cmd

    def test_create_failure(self, config: WorktreeConfig, tmp_path: Path) -> None:
        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        )

        with patch("subprocess.run", return_value=mock_result):
            with patch("shutil.rmtree"):
                ctx = create_worktree(
                    repo_path=tmp_path / "repo",
                    commit_sha="abc123",
                    config=config,
                    run_id="run-1",
                    issue_id="mala-10",
                    attempt=1,
                )

        assert ctx.state == WorktreeState.FAILED
        assert ctx.error is not None
        assert "128" in ctx.error
        assert "not a git repository" in ctx.error

    def test_create_cleans_stale_directory(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        # Create a stale directory at the worktree path
        expected_path = config.base_dir / "run-1" / "mala-10" / "1"
        expected_path.mkdir(parents=True)
        (expected_path / "stale_file").touch()

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        with patch("subprocess.run", return_value=mock_result):
            ctx = create_worktree(
                repo_path=tmp_path / "repo",
                commit_sha="abc123",
                config=config,
                run_id="run-1",
                issue_id="mala-10",
                attempt=1,
            )

        assert ctx.state == WorktreeState.CREATED
        # The stale file should be gone (directory was removed before create)
        assert not (expected_path / "stale_file").exists()

    def test_create_increments_attempt(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        with patch("subprocess.run", return_value=mock_result):
            ctx1 = create_worktree(
                repo_path=tmp_path / "repo",
                commit_sha="abc123",
                config=config,
                run_id="run-1",
                issue_id="mala-10",
                attempt=1,
            )
            ctx2 = create_worktree(
                repo_path=tmp_path / "repo",
                commit_sha="def456",
                config=config,
                run_id="run-1",
                issue_id="mala-10",
                attempt=2,
            )

        assert ctx1.path == config.base_dir / "run-1" / "mala-10" / "1"
        assert ctx2.path == config.base_dir / "run-1" / "mala-10" / "2"


class TestRemoveWorktree:
    """Test remove_worktree function."""

    @pytest.fixture
    def created_ctx(self, tmp_path: Path) -> WorktreeContext:
        config = WorktreeConfig(base_dir=tmp_path / "worktrees")
        ctx = WorktreeContext(
            config=config,
            repo_path=tmp_path / "repo",
            run_id="run-1",
            issue_id="mala-10",
            attempt=1,
            state=WorktreeState.CREATED,
        )
        return ctx

    def test_remove_success(self, created_ctx: WorktreeContext) -> None:
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        with patch("subprocess.run", return_value=mock_result):
            ctx = remove_worktree(created_ctx, validation_passed=True)

        assert ctx.state == WorktreeState.REMOVED

    def test_remove_honors_keep_on_failure(self, tmp_path: Path) -> None:
        config = WorktreeConfig(base_dir=tmp_path / "worktrees", keep_on_failure=True)
        ctx = WorktreeContext(
            config=config,
            repo_path=tmp_path / "repo",
            run_id="run-1",
            issue_id="mala-10",
            attempt=1,
            state=WorktreeState.CREATED,
        )

        # Validation failed and keep_on_failure=True
        with patch("subprocess.run") as mock_run:
            result_ctx = remove_worktree(ctx, validation_passed=False)

        assert result_ctx.state == WorktreeState.KEPT
        # Should NOT call git worktree remove
        mock_run.assert_not_called()

    def test_remove_deletes_on_failure_when_not_kept(
        self, created_ctx: WorktreeContext
    ) -> None:
        # keep_on_failure is False by default
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        with patch("subprocess.run", return_value=mock_result):
            ctx = remove_worktree(created_ctx, validation_passed=False)

        assert ctx.state == WorktreeState.REMOVED

    def test_remove_skips_pending_worktree(self, tmp_path: Path) -> None:
        config = WorktreeConfig(base_dir=tmp_path / "worktrees")
        ctx = WorktreeContext(
            config=config,
            repo_path=tmp_path / "repo",
            run_id="run-1",
            issue_id="mala-10",
            attempt=1,
            state=WorktreeState.PENDING,
        )

        with patch("subprocess.run") as mock_run:
            result_ctx = remove_worktree(ctx, validation_passed=True)

        assert result_ctx.state == WorktreeState.PENDING
        mock_run.assert_not_called()

    def test_remove_uses_force_flag(self, created_ctx: WorktreeContext) -> None:
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            remove_worktree(created_ctx, validation_passed=True)

        # First call should be git worktree remove --force
        first_call = mock_run.call_args_list[0]
        cmd = first_call[0][0]
        assert "--force" in cmd

    def test_remove_without_force_flag(self, tmp_path: Path) -> None:
        config = WorktreeConfig(base_dir=tmp_path / "worktrees", force_remove=False)
        ctx = WorktreeContext(
            config=config,
            repo_path=tmp_path / "repo",
            run_id="run-1",
            issue_id="mala-10",
            attempt=1,
            state=WorktreeState.CREATED,
        )

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            remove_worktree(ctx, validation_passed=True)

        # First call should NOT have --force
        first_call = mock_run.call_args_list[0]
        cmd = first_call[0][0]
        assert "--force" not in cmd

    def test_remove_prunes_worktree_list(self, created_ctx: WorktreeContext) -> None:
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            remove_worktree(created_ctx, validation_passed=True)

        # Should call git worktree prune after remove
        calls = mock_run.call_args_list
        prune_call = [c for c in calls if "prune" in c[0][0]]
        assert len(prune_call) == 1


class TestCleanupStaleWorktrees:
    """Test cleanup_stale_worktrees function."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> WorktreeConfig:
        return WorktreeConfig(base_dir=tmp_path / "worktrees")

    def test_cleanup_nonexistent_base_dir(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        # Base dir doesn't exist
        cleaned = cleanup_stale_worktrees(tmp_path / "repo", config)
        assert cleaned == 0

    def test_cleanup_specific_run(self, config: WorktreeConfig, tmp_path: Path) -> None:
        # Create some worktree directories
        (config.base_dir / "run-1" / "mala-10" / "1").mkdir(parents=True)
        (config.base_dir / "run-1" / "mala-10" / "2").mkdir(parents=True)
        (config.base_dir / "run-2" / "mala-20" / "1").mkdir(parents=True)

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        with patch("subprocess.run", return_value=mock_result):
            cleaned = cleanup_stale_worktrees(tmp_path / "repo", config, run_id="run-1")

        # Should clean 2 worktrees from run-1 only
        assert cleaned == 2
        # run-2 should still exist
        assert (config.base_dir / "run-2" / "mala-20" / "1").exists()

    def test_cleanup_all_runs(self, config: WorktreeConfig, tmp_path: Path) -> None:
        # Create worktrees in multiple runs
        (config.base_dir / "run-1" / "mala-10" / "1").mkdir(parents=True)
        (config.base_dir / "run-2" / "mala-20" / "1").mkdir(parents=True)
        (config.base_dir / "run-3" / "mala-30" / "1").mkdir(parents=True)

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        with patch("subprocess.run", return_value=mock_result):
            cleaned = cleanup_stale_worktrees(tmp_path / "repo", config)

        assert cleaned == 3

    def test_cleanup_removes_empty_parent_dirs(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        # Create a single worktree
        (config.base_dir / "run-1" / "mala-10" / "1").mkdir(parents=True)

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        with patch("subprocess.run", return_value=mock_result):
            cleanup_stale_worktrees(tmp_path / "repo", config, run_id="run-1")

        # Parent directories should be removed if empty
        assert not (config.base_dir / "run-1").exists()

    def test_cleanup_calls_git_worktree_prune(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        config.base_dir.mkdir(parents=True)

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            cleanup_stale_worktrees(tmp_path / "repo", config)

        # Should call git worktree prune
        calls = [c[0][0] for c in mock_run.call_args_list]
        prune_calls = [c for c in calls if "prune" in c]
        assert len(prune_calls) == 1


class TestErrorFormatting:
    """Test error message formatting."""

    def test_long_stderr_truncated(self, tmp_path: Path) -> None:
        config = WorktreeConfig(base_dir=tmp_path / "worktrees")
        long_stderr = "x" * 500

        mock_result = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr=long_stderr,
        )

        with patch("subprocess.run", return_value=mock_result):
            with patch("shutil.rmtree"):
                ctx = create_worktree(
                    repo_path=tmp_path / "repo",
                    commit_sha="abc123",
                    config=config,
                    run_id="run-1",
                    issue_id="mala-10",
                    attempt=1,
                )

        assert ctx.error is not None
        # Error should be truncated
        assert len(ctx.error) < 250  # 200 chars of stderr + prefix
        assert "..." in ctx.error
