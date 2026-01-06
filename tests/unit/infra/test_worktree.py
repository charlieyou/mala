"""Unit tests for src/validation/worktree.py - git worktree utilities.

These tests use FakeCommandRunner to test worktree logic without actually
creating git worktrees.
"""

from pathlib import Path

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
from tests.fakes.command_runner import FakeCommandRunner


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
        runner = FakeCommandRunner(allow_unregistered=True)

        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="run-1",
            issue_id="mala-10",
            attempt=1,
            command_runner=runner,
        )

        assert ctx.state == WorktreeState.CREATED
        assert ctx.error is None
        assert ctx.path == config.base_dir / "run-1" / "mala-10" / "1"

        # Verify git worktree add was called with expected arguments
        worktree_add_calls = runner.get_calls_with_prefix(["git", "worktree", "add"])
        assert len(worktree_add_calls) == 1
        cmd_tuple, _ = worktree_add_calls[0]
        assert "--detach" in cmd_tuple
        assert "abc123" in cmd_tuple

    def test_create_failure(self, config: WorktreeConfig, tmp_path: Path) -> None:
        # Create worktree path so cleanup can be tested with real filesystem
        worktree_path = config.base_dir / "run-1" / "mala-10" / "1"
        worktree_path.mkdir(parents=True, exist_ok=True)

        runner = FakeCommandRunner(allow_unregistered=True)
        # Register the exact command that will be called
        runner.responses[
            ("git", "worktree", "add", "--detach", str(worktree_path), "abc123")
        ] = CommandResult(
            command=[],
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        )

        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="run-1",
            issue_id="mala-10",
            attempt=1,
            command_runner=runner,
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

        runner = FakeCommandRunner(allow_unregistered=True)

        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="run-1",
            issue_id="mala-10",
            attempt=1,
            command_runner=runner,
        )

        assert ctx.state == WorktreeState.CREATED
        # The stale file should be gone (directory was removed before create)
        assert not (expected_path / "stale_file").exists()

    def test_create_increments_attempt(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner(allow_unregistered=True)

        ctx1 = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="run-1",
            issue_id="mala-10",
            attempt=1,
            command_runner=runner,
        )
        ctx2 = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="def456",
            config=config,
            run_id="run-1",
            issue_id="mala-10",
            attempt=2,
            command_runner=runner,
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
        runner = FakeCommandRunner(allow_unregistered=True)

        ctx = remove_worktree(
            created_ctx, validation_passed=True, command_runner=runner
        )

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
        runner = FakeCommandRunner(allow_unregistered=True)

        # Validation failed and keep_on_failure=True
        result_ctx = remove_worktree(
            ctx, validation_passed=False, command_runner=runner
        )

        assert result_ctx.state == WorktreeState.KEPT
        # Should NOT call git worktree remove
        assert not runner.has_call_with_prefix(["git", "worktree", "remove"])

    def test_remove_deletes_on_failure_when_not_kept(
        self, created_ctx: WorktreeContext
    ) -> None:
        # keep_on_failure is False by default
        runner = FakeCommandRunner(allow_unregistered=True)

        ctx = remove_worktree(
            created_ctx, validation_passed=False, command_runner=runner
        )

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
        runner = FakeCommandRunner(allow_unregistered=True)

        result_ctx = remove_worktree(ctx, validation_passed=True, command_runner=runner)

        assert result_ctx.state == WorktreeState.PENDING
        # Should not call any git commands for pending worktree
        assert len(runner.calls) == 0

    def test_remove_uses_force_flag(self, created_ctx: WorktreeContext) -> None:
        runner = FakeCommandRunner(allow_unregistered=True)

        remove_worktree(created_ctx, validation_passed=True, command_runner=runner)

        # First call should be git worktree remove --force
        remove_calls = runner.get_calls_with_prefix(["git", "worktree", "remove"])
        assert len(remove_calls) >= 1
        cmd_tuple, _ = remove_calls[0]
        assert "--force" in cmd_tuple

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

        runner = FakeCommandRunner(allow_unregistered=True)

        remove_worktree(ctx, validation_passed=True, command_runner=runner)

        # First call should NOT have --force
        remove_calls = runner.get_calls_with_prefix(["git", "worktree", "remove"])
        assert len(remove_calls) >= 1
        cmd_tuple, _ = remove_calls[0]
        assert "--force" not in cmd_tuple

    def test_remove_prunes_worktree_list(self, created_ctx: WorktreeContext) -> None:
        runner = FakeCommandRunner(allow_unregistered=True)

        remove_worktree(created_ctx, validation_passed=True, command_runner=runner)

        # Should call git worktree prune after remove
        assert runner.has_call_with_prefix(["git", "worktree", "prune"])


class TestCleanupStaleWorktrees:
    """Test cleanup_stale_worktrees function."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> WorktreeConfig:
        return WorktreeConfig(base_dir=tmp_path / "worktrees")

    def test_cleanup_nonexistent_base_dir(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        # Base dir doesn't exist
        runner = FakeCommandRunner(allow_unregistered=True)
        cleaned = cleanup_stale_worktrees(
            tmp_path / "repo", config, command_runner=runner
        )
        assert cleaned == 0

    def test_cleanup_specific_run(self, config: WorktreeConfig, tmp_path: Path) -> None:
        # Create some worktree directories
        (config.base_dir / "run-1" / "mala-10" / "1").mkdir(parents=True)
        (config.base_dir / "run-1" / "mala-10" / "2").mkdir(parents=True)
        (config.base_dir / "run-2" / "mala-20" / "1").mkdir(parents=True)

        runner = FakeCommandRunner(allow_unregistered=True)

        cleaned = cleanup_stale_worktrees(
            tmp_path / "repo", config, command_runner=runner, run_id="run-1"
        )

        # Should clean 2 worktrees from run-1 only
        assert cleaned == 2
        # run-2 should still exist
        assert (config.base_dir / "run-2" / "mala-20" / "1").exists()

    def test_cleanup_all_runs(self, config: WorktreeConfig, tmp_path: Path) -> None:
        # Create worktrees in multiple runs
        (config.base_dir / "run-1" / "mala-10" / "1").mkdir(parents=True)
        (config.base_dir / "run-2" / "mala-20" / "1").mkdir(parents=True)
        (config.base_dir / "run-3" / "mala-30" / "1").mkdir(parents=True)

        runner = FakeCommandRunner(allow_unregistered=True)

        cleaned = cleanup_stale_worktrees(
            tmp_path / "repo", config, command_runner=runner
        )

        assert cleaned == 3

    def test_cleanup_removes_empty_parent_dirs(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        # Create a single worktree
        (config.base_dir / "run-1" / "mala-10" / "1").mkdir(parents=True)

        runner = FakeCommandRunner(allow_unregistered=True)

        cleanup_stale_worktrees(
            tmp_path / "repo", config, command_runner=runner, run_id="run-1"
        )

        # Parent directories should be removed if empty
        assert not (config.base_dir / "run-1").exists()

    def test_cleanup_calls_git_worktree_prune(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        config.base_dir.mkdir(parents=True)

        runner = FakeCommandRunner(allow_unregistered=True)

        cleanup_stale_worktrees(tmp_path / "repo", config, command_runner=runner)

        # Should call git worktree prune
        assert runner.has_call_with_prefix(["git", "worktree", "prune"])


class TestErrorFormatting:
    """Test error message formatting."""

    def test_long_stderr_truncated(self, tmp_path: Path) -> None:
        config = WorktreeConfig(base_dir=tmp_path / "worktrees")
        long_stderr = "x" * 500

        # Create worktree path so cleanup uses real filesystem
        worktree_path = config.base_dir / "run-1" / "mala-10" / "1"
        worktree_path.mkdir(parents=True, exist_ok=True)

        runner = FakeCommandRunner(allow_unregistered=True)
        # Register the exact command that will be called
        runner.responses[
            ("git", "worktree", "add", "--detach", str(worktree_path), "abc123")
        ] = CommandResult(
            command=[],
            returncode=1,
            stdout="",
            stderr=long_stderr,
        )

        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="run-1",
            issue_id="mala-10",
            attempt=1,
            command_runner=runner,
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

    @pytest.fixture
    def runner(self) -> FakeCommandRunner:
        """Provide a fake command runner for tests."""
        return FakeCommandRunner(allow_unregistered=True)

    def test_rejects_run_id_with_slash(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        """run_id containing '/' should be rejected."""
        runner = FakeCommandRunner(allow_unregistered=True)
        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="run/escape",
            issue_id="mala-10",
            attempt=1,
            command_runner=runner,
        )
        assert ctx.state == WorktreeState.FAILED
        assert "Invalid run_id" in (ctx.error or "")

    def test_rejects_issue_id_with_dotdot(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        """issue_id containing '..' should be rejected."""
        runner = FakeCommandRunner(allow_unregistered=True)
        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="run-1",
            issue_id="../escape",
            attempt=1,
            command_runner=runner,
        )
        assert ctx.state == WorktreeState.FAILED
        assert "Invalid issue_id" in (ctx.error or "")

    def test_rejects_absolute_path_in_run_id(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        """Absolute path in run_id should be rejected."""
        runner = FakeCommandRunner(allow_unregistered=True)
        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="/etc/passwd",
            issue_id="mala-10",
            attempt=1,
            command_runner=runner,
        )
        assert ctx.state == WorktreeState.FAILED
        assert "Invalid run_id" in (ctx.error or "")

    def test_rejects_hidden_directory_run_id(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        """run_id starting with dot should be rejected."""
        runner = FakeCommandRunner(allow_unregistered=True)
        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id=".hidden",
            issue_id="mala-10",
            attempt=1,
            command_runner=runner,
        )
        assert ctx.state == WorktreeState.FAILED
        assert "Invalid run_id" in (ctx.error or "")

    def test_rejects_negative_attempt(
        self, config: WorktreeConfig, tmp_path: Path
    ) -> None:
        """Negative attempt number should be rejected."""
        runner = FakeCommandRunner(allow_unregistered=True)
        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="run-1",
            issue_id="mala-10",
            attempt=-1,
            command_runner=runner,
        )
        assert ctx.state == WorktreeState.FAILED
        assert "Invalid attempt" in (ctx.error or "")

    def test_accepts_valid_path_components(
        self, config: WorktreeConfig, tmp_path: Path, runner: FakeCommandRunner
    ) -> None:
        """Valid path components should be accepted."""
        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="run-123_test.v1",
            issue_id="mala-42",
            attempt=1,
            command_runner=runner,
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
        # Ensure directory doesn't exist (simulating already-deleted directory)
        # Note: created_ctx._path is set but directory not created
        assert created_ctx._path is not None
        assert not created_ctx._path.exists()

        # Git command fails - register the exact command (with --force since default)
        runner = FakeCommandRunner(allow_unregistered=True)
        runner.responses[
            ("git", "worktree", "remove", "--force", str(created_ctx._path))
        ] = CommandResult(
            command=[], returncode=1, stdout="", stderr="worktree not found"
        )

        ctx = remove_worktree(
            created_ctx, validation_passed=True, command_runner=runner
        )

        assert ctx.state == WorktreeState.FAILED
        assert "worktree not found" in (ctx.error or "")

    def test_reports_directory_cleanup_failure(
        self,
        created_ctx: WorktreeContext,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should report failure when directory cleanup fails."""
        import shutil

        # Create the directory so exists() returns True
        created_ctx._path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]

        runner = FakeCommandRunner(allow_unregistered=True)

        def mock_rmtree(path: Path, **kwargs: object) -> None:
            raise OSError("Permission denied")

        monkeypatch.setattr(shutil, "rmtree", mock_rmtree)

        ctx = remove_worktree(
            created_ctx, validation_passed=True, command_runner=runner
        )

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
        # Note: no --force flag since force_remove=False
        runner = FakeCommandRunner(allow_unregistered=True)
        runner.responses[("git", "worktree", "remove", str(ctx._path))] = CommandResult(
            command=[],
            returncode=1,
            stdout="",
            stderr="fatal: worktree has uncommitted changes",
        )

        result_ctx = remove_worktree(ctx, validation_passed=True, command_runner=runner)

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
        runner = FakeCommandRunner(allow_unregistered=True)
        runner.responses[("git", "worktree", "remove", "--force", str(ctx._path))] = (
            CommandResult(
                command=[], returncode=1, stdout="", stderr="git worktree remove failed"
            )
        )

        result_ctx = remove_worktree(ctx, validation_passed=True, command_runner=runner)

        # Should fail (git command failed)
        assert result_ctx.state == WorktreeState.FAILED
        # But directory should be deleted (force_remove=True)
        assert not ctx._path.exists()


class TestStalePathCleanup:
    """Test stale path cleanup error handling."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> WorktreeConfig:
        return WorktreeConfig(base_dir=tmp_path / "worktrees")

    @pytest.fixture
    def runner(self) -> FakeCommandRunner:
        """Provide a fake command runner for tests."""
        return FakeCommandRunner(allow_unregistered=True)

    def test_fails_if_path_is_file(
        self, config: WorktreeConfig, tmp_path: Path, runner: FakeCommandRunner
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
            command_runner=runner,
        )

        assert ctx.state == WorktreeState.FAILED
        assert "exists as file" in (ctx.error or "")

    def test_fails_if_stale_cleanup_fails(
        self,
        config: WorktreeConfig,
        tmp_path: Path,
        runner: FakeCommandRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should fail fast if stale worktree cleanup fails."""
        import shutil

        # Create a directory at the worktree path
        worktree_path = config.base_dir / "run-1" / "mala-10" / "1"
        worktree_path.mkdir(parents=True, exist_ok=True)

        def mock_rmtree(path: Path, **kwargs: object) -> None:
            raise OSError("Permission denied")

        monkeypatch.setattr(shutil, "rmtree", mock_rmtree)

        ctx = create_worktree(
            repo_path=tmp_path / "repo",
            commit_sha="abc123",
            config=config,
            run_id="run-1",
            issue_id="mala-10",
            attempt=1,
            command_runner=runner,
        )

        assert ctx.state == WorktreeState.FAILED
        assert "Failed to remove stale worktree" in (ctx.error or "")


class TestCleanupEmptyParents:
    """Test _cleanup_empty_parents helper function."""

    def test_removes_empty_parent_chain(self, tmp_path: Path) -> None:
        """Should remove empty parent directories up to base_dir."""
        from src.domain.validation.worktree import _cleanup_empty_parents

        base_dir = tmp_path / "worktrees"
        worktree_path = base_dir / "run-1" / "issue-1" / "attempt-1"
        worktree_path.mkdir(parents=True)

        # Simulate: worktree content already removed, only empty dirs remain
        worktree_path.rmdir()

        _cleanup_empty_parents(worktree_path, base_dir)

        # All empty parents should be removed
        assert not (base_dir / "run-1" / "issue-1").exists()
        assert not (base_dir / "run-1").exists()
        # base_dir itself should remain
        assert base_dir.exists()

    def test_stops_at_non_empty_dir(self, tmp_path: Path) -> None:
        """Should stop when encountering a non-empty directory."""
        from src.domain.validation.worktree import _cleanup_empty_parents

        base_dir = tmp_path / "worktrees"
        worktree_path = base_dir / "run-1" / "issue-1" / "attempt-1"
        other_file = base_dir / "run-1" / "other.txt"

        worktree_path.mkdir(parents=True)
        other_file.write_text("keep me")

        worktree_path.rmdir()
        (base_dir / "run-1" / "issue-1").rmdir()

        _cleanup_empty_parents(worktree_path, base_dir)

        # run-1 should remain (has other.txt)
        assert (base_dir / "run-1").exists()
        assert other_file.exists()

    def test_ignores_path_outside_base_dir(self, tmp_path: Path) -> None:
        """Should not remove anything if path is outside base_dir."""
        from src.domain.validation.worktree import _cleanup_empty_parents

        base_dir = tmp_path / "worktrees"
        base_dir.mkdir()
        outside_path = tmp_path / "other" / "path"
        outside_path.mkdir(parents=True)

        _cleanup_empty_parents(outside_path, base_dir)

        # outside_path should still exist
        assert outside_path.exists()

    def test_handles_nonexistent_path(self, tmp_path: Path) -> None:
        """Should handle case where worktree path doesn't exist."""
        from src.domain.validation.worktree import _cleanup_empty_parents

        base_dir = tmp_path / "worktrees"
        base_dir.mkdir()
        nonexistent = base_dir / "run-1" / "issue-1" / "attempt-1"

        # Should not raise
        _cleanup_empty_parents(nonexistent, base_dir)
