"""Unit tests for src/validation/worktree.py - git worktree utilities.

These tests mock run_command calls to test worktree logic without actually
creating git worktrees.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.infra.tools.command_runner import CommandResult
from src.domain.validation.worktree import (
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
        mock_result = CommandResult(
            command=[],
            returncode=0,
            stdout="",
            stderr="",
        )

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_result
        ) as mock_run:
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
        mock_result = CommandResult(
            command=[],
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        )

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_result
        ):
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

        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_result
        ):
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
        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_result
        ):
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
        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_result
        ):
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
        with patch("src.domain.validation.worktree.run_command") as mock_run:
            result_ctx = remove_worktree(ctx, validation_passed=False)

        assert result_ctx.state == WorktreeState.KEPT
        # Should NOT call git worktree remove
        mock_run.assert_not_called()

    def test_remove_deletes_on_failure_when_not_kept(
        self, created_ctx: WorktreeContext
    ) -> None:
        # keep_on_failure is False by default
        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_result
        ):
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

        with patch("src.domain.validation.worktree.run_command") as mock_run:
            result_ctx = remove_worktree(ctx, validation_passed=True)

        assert result_ctx.state == WorktreeState.PENDING
        mock_run.assert_not_called()

    def test_remove_uses_force_flag(self, created_ctx: WorktreeContext) -> None:
        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_result
        ) as mock_run:
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

        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_result
        ) as mock_run:
            remove_worktree(ctx, validation_passed=True)

        # First call should NOT have --force
        first_call = mock_run.call_args_list[0]
        cmd = first_call[0][0]
        assert "--force" not in cmd

    def test_remove_prunes_worktree_list(self, created_ctx: WorktreeContext) -> None:
        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_result
        ) as mock_run:
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

        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_result
        ):
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

        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_result
        ):
            cleaned = cleanup_stale_worktrees(tmp_path / "repo", config)

        assert cleaned == 3

    def test_cleanup_removes_empty_parent_dirs(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        # Create a single worktree
        (config.base_dir / "run-1" / "mala-10" / "1").mkdir(parents=True)

        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_result
        ):
            cleanup_stale_worktrees(tmp_path / "repo", config, run_id="run-1")

        # Parent directories should be removed if empty
        assert not (config.base_dir / "run-1").exists()

    def test_cleanup_calls_git_worktree_prune(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        config.base_dir.mkdir(parents=True)

        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_result
        ) as mock_run:
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

        mock_result = CommandResult(
            command=[],
            returncode=1,
            stdout="",
            stderr=long_stderr,
        )

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_result
        ):
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


class TestPathValidation:
    """Test path validation for security."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> WorktreeConfig:
        return WorktreeConfig(base_dir=tmp_path / "worktrees")

    def test_rejects_run_id_with_slash(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        """run_id containing '/' should be rejected."""
        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="run/escape",
            issue_id="mala-10",
            attempt=1,
        )
        assert ctx.state == WorktreeState.FAILED
        assert "Invalid run_id" in (ctx.error or "")

    def test_rejects_issue_id_with_dotdot(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        """issue_id containing '..' should be rejected."""
        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="run-1",
            issue_id="../escape",
            attempt=1,
        )
        assert ctx.state == WorktreeState.FAILED
        assert "Invalid issue_id" in (ctx.error or "")

    def test_rejects_absolute_path_in_run_id(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        """Absolute path in run_id should be rejected."""
        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="/etc/passwd",
            issue_id="mala-10",
            attempt=1,
        )
        assert ctx.state == WorktreeState.FAILED
        assert "Invalid run_id" in (ctx.error or "")

    def test_rejects_hidden_directory_run_id(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        """run_id starting with dot should be rejected."""
        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id=".hidden",
            issue_id="mala-10",
            attempt=1,
        )
        assert ctx.state == WorktreeState.FAILED
        assert "Invalid run_id" in (ctx.error or "")

    def test_rejects_negative_attempt(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        """Negative attempt number should be rejected."""
        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="run-1",
            issue_id="mala-10",
            attempt=-1,
        )
        assert ctx.state == WorktreeState.FAILED
        assert "Invalid attempt" in (ctx.error or "")

    def test_accepts_valid_path_components(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        """Valid path components should be accepted."""
        mock_result = CommandResult(command=[], returncode=0, stdout="", stderr="")

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_result
        ):
            ctx = create_worktree(
                repo_path=tmp_path / "repo",
                commit_sha="abc123",
                config=config,
                run_id="run-123_test.v1",
                issue_id="mala-42",
                attempt=1,
            )
        assert ctx.state == WorktreeState.CREATED

    def test_context_path_raises_on_unsafe_input(self, tmp_path: Path) -> None:
        """WorktreeContext.path should raise ValueError for unsafe inputs."""
        config = WorktreeConfig(base_dir=tmp_path / "worktrees")
        ctx = WorktreeContext(
            config=config,
            repo_path=tmp_path / "repo",
            run_id="../escape",
            issue_id="mala-10",
            attempt=1,
        )

        with pytest.raises(ValueError, match="Invalid run_id"):
            _ = ctx.path


class TestRemoveWorktreeFailurePropagation:
    """Test that remove_worktree properly propagates failures."""

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
        # Force path validation
        ctx._validated = True
        ctx._path = config.base_dir / "run-1" / "mala-10" / "1"
        return ctx

    def test_reports_failure_even_when_directory_deleted(
        self, created_ctx: WorktreeContext
    ) -> None:
        """git worktree remove failure should be reported even if directory was deleted."""
        # Git command fails
        mock_git_fail = CommandResult(
            command=[], returncode=1, stdout="", stderr="worktree not found"
        )
        # But directory doesn't exist (already deleted somehow)
        with (
            patch(
                "src.domain.validation.worktree.run_command", return_value=mock_git_fail
            ),
            patch.object(Path, "exists", return_value=False),
        ):
            ctx = remove_worktree(created_ctx, validation_passed=True)

        assert ctx.state == WorktreeState.FAILED
        assert "worktree not found" in (ctx.error or "")

    def test_reports_directory_cleanup_failure(
        self, created_ctx: WorktreeContext, tmp_path: Path
    ) -> None:
        """Should report failure when directory cleanup fails."""
        # Create the directory so exists() returns True
        created_ctx._path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]

        mock_git_success = CommandResult(command=[], returncode=0, stdout="", stderr="")

        def mock_rmtree(path: Path) -> None:
            raise OSError("Permission denied")

        with (
            patch(
                "src.domain.validation.worktree.run_command",
                return_value=mock_git_success,
            ),
            patch("shutil.rmtree", side_effect=mock_rmtree),
        ):
            ctx = remove_worktree(created_ctx, validation_passed=True)

        assert ctx.state == WorktreeState.FAILED
        assert "Permission denied" in (ctx.error or "")

    def test_preserves_directory_on_git_failure_when_force_remove_false(
        self, tmp_path: Path
    ) -> None:
        """Should NOT delete directory when git fails and force_remove=False.

        This protects uncommitted changes in dirty worktrees.
        """
        config = WorktreeConfig(
            base_dir=tmp_path / "worktrees",
            force_remove=False,  # Key: force_remove is False
        )
        ctx = WorktreeContext(
            config=config,
            repo_path=tmp_path / "repo",
            run_id="run-1",
            issue_id="mala-10",
            attempt=1,
            state=WorktreeState.CREATED,
        )
        ctx._validated = True
        ctx._path = config.base_dir / "run-1" / "mala-10" / "1"

        # Create the directory with some content (simulating uncommitted changes)
        ctx._path.mkdir(parents=True, exist_ok=True)
        (ctx._path / "uncommitted.txt").write_text("precious data")

        # Git command fails (e.g., dirty worktree without --force)
        mock_git_fail = CommandResult(
            command=[],
            returncode=1,
            stdout="",
            stderr="fatal: worktree has uncommitted changes",
        )

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_git_fail
        ):
            result_ctx = remove_worktree(ctx, validation_passed=True)

        # Should fail
        assert result_ctx.state == WorktreeState.FAILED
        assert "uncommitted changes" in (result_ctx.error or "")
        # Directory should still exist (preserving uncommitted changes)
        assert ctx._path.exists()
        assert (ctx._path / "uncommitted.txt").exists()

    def test_deletes_directory_on_git_failure_when_force_remove_true(
        self, tmp_path: Path
    ) -> None:
        """Should delete directory when git fails but force_remove=True.

        User explicitly requested forced cleanup, so we clean up anyway.
        """
        config = WorktreeConfig(
            base_dir=tmp_path / "worktrees",
            force_remove=True,  # Key: force_remove is True
        )
        ctx = WorktreeContext(
            config=config,
            repo_path=tmp_path / "repo",
            run_id="run-1",
            issue_id="mala-10",
            attempt=1,
            state=WorktreeState.CREATED,
        )
        ctx._validated = True
        ctx._path = config.base_dir / "run-1" / "mala-10" / "1"

        # Create the directory
        ctx._path.mkdir(parents=True, exist_ok=True)
        (ctx._path / "some_file.txt").write_text("data")

        # Git command fails but force_remove=True means we still cleanup
        mock_git_fail = CommandResult(
            command=[], returncode=1, stdout="", stderr="git worktree remove failed"
        )

        with patch(
            "src.domain.validation.worktree.run_command", return_value=mock_git_fail
        ):
            result_ctx = remove_worktree(ctx, validation_passed=True)

        # Should fail (git command failed)
        assert result_ctx.state == WorktreeState.FAILED
        # But directory should be deleted (force_remove=True)
        assert not ctx._path.exists()


class TestStalePathCleanup:
    """Test stale path cleanup error handling."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> WorktreeConfig:
        return WorktreeConfig(base_dir=tmp_path / "worktrees")

    def test_fails_if_path_is_file(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        """Should fail if worktree path exists as a file."""
        # Create parent and a file at the worktree path
        worktree_path = config.base_dir / "run-1" / "mala-10" / "1"
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        worktree_path.touch()  # Create as file, not directory

        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="run-1",
            issue_id="mala-10",
            attempt=1,
        )

        assert ctx.state == WorktreeState.FAILED
        assert "exists as file" in (ctx.error or "")

    def test_fails_if_stale_cleanup_fails(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        """Should fail fast if stale worktree cleanup fails."""
        # Create a directory at the worktree path
        worktree_path = config.base_dir / "run-1" / "mala-10" / "1"
        worktree_path.mkdir(parents=True, exist_ok=True)

        def mock_rmtree(path: Path, **kwargs: object) -> None:
            raise OSError("Permission denied")

        with patch("shutil.rmtree", side_effect=mock_rmtree):
            ctx = create_worktree(
                repo_path=tmp_path / "repo",
                commit_sha="abc123",
                config=config,
                run_id="run-1",
                issue_id="mala-10",
                attempt=1,
            )

        assert ctx.state == WorktreeState.FAILED
        assert "Failed to remove stale worktree" in (ctx.error or "")
