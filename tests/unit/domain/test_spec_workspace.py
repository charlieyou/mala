"""Unit tests for SpecRunWorkspace helper.

These tests verify that the workspace helper correctly:
1. Sets up log directories and generates run/issue IDs
2. Refreshes baseline coverage when needed
3. Creates and manages worktrees for commit-based validation
4. Cleans up resources with proper pass/fail handling

TDD approach: tests written first to define the expected behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from tests.fakes import FakeEnvConfig

from src.domain.validation.config import YamlCoverageConfig
from src.domain.validation.spec import (
    CommandKind,
    CoverageConfig,
    E2EConfig,
    ValidationArtifacts,
    ValidationCommand,
    ValidationContext,
    ValidationScope,
    ValidationSpec,
)
from src.domain.validation.worktree import WorktreeContext, WorktreeState
from src.infra.tools.command_runner import CommandRunner
from tests.fakes.lock_manager import FakeLockManager

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a temporary repository directory."""
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


@pytest.fixture
def command_runner(tmp_repo: Path) -> CommandRunner:
    """Create a command runner for tests."""
    return CommandRunner(cwd=tmp_repo)


@pytest.fixture
def mock_env_config() -> FakeEnvConfig:
    """Create a fake EnvConfigPort for tests."""
    return FakeEnvConfig()


@pytest.fixture
def fake_lock_manager() -> FakeLockManager:
    """Create a FakeLockManager for tests."""
    return FakeLockManager()


@pytest.fixture
def basic_spec() -> ValidationSpec:
    """Create a basic validation spec without coverage or E2E."""
    return ValidationSpec(
        commands=[
            ValidationCommand(
                name="pytest",
                command="pytest",
                kind=CommandKind.TEST,
            ),
        ],
        scope=ValidationScope.PER_SESSION,
        coverage=CoverageConfig(enabled=False),
        e2e=E2EConfig(enabled=False),
    )


@pytest.fixture
def context_in_place(tmp_repo: Path) -> ValidationContext:
    """Create a context for in-place validation (no worktree)."""
    return ValidationContext(
        issue_id="test-123",
        repo_path=tmp_repo,
        commit_hash="",  # Empty = validate in place
        changed_files=["src/test.py"],
        scope=ValidationScope.PER_SESSION,
    )


@pytest.fixture
def context_with_commit(tmp_repo: Path) -> ValidationContext:
    """Create a context with a commit hash (requires worktree)."""
    return ValidationContext(
        issue_id="test-456",
        repo_path=tmp_repo,
        commit_hash="abc123",
        changed_files=["src/test.py"],
        scope=ValidationScope.PER_SESSION,
    )


class TestSpecRunWorkspaceDataclass:
    """Test SpecRunWorkspace dataclass fields."""

    def test_workspace_has_required_fields(
        self, tmp_repo: Path, basic_spec: ValidationSpec
    ) -> None:
        """SpecRunWorkspace should have validation_cwd, artifacts, and baseline_percent."""
        from src.domain.validation.spec_workspace import SpecRunWorkspace

        # Create a minimal workspace for in-place validation
        workspace = SpecRunWorkspace(
            validation_cwd=tmp_repo,
            artifacts=ValidationArtifacts(log_dir=tmp_repo / "logs"),
            baseline_percent=85.0,
            run_id="run-abc123",
            log_dir=tmp_repo / "logs",
            worktree_ctx=None,
        )

        assert workspace.validation_cwd == tmp_repo
        assert workspace.artifacts.log_dir == tmp_repo / "logs"
        assert workspace.baseline_percent == 85.0
        assert workspace.run_id == "run-abc123"
        assert workspace.worktree_ctx is None

    def test_workspace_with_worktree_context(self, tmp_repo: Path) -> None:
        """SpecRunWorkspace should include optional worktree context."""
        from src.domain.validation.spec_workspace import SpecRunWorkspace

        worktree_path = tmp_repo / "worktree"
        worktree_path.mkdir()

        mock_worktree = MagicMock(spec=WorktreeContext)
        mock_worktree.state = WorktreeState.CREATED
        mock_worktree.path = worktree_path

        workspace = SpecRunWorkspace(
            validation_cwd=worktree_path,
            artifacts=ValidationArtifacts(log_dir=tmp_repo / "logs"),
            baseline_percent=None,
            run_id="run-xyz789",
            log_dir=tmp_repo / "logs",
            worktree_ctx=mock_worktree,
        )

        assert workspace.worktree_ctx is not None
        assert workspace.worktree_ctx.path == worktree_path


class TestSetupWorkspace:
    """Test setup_workspace function."""

    def test_setup_creates_log_dir(
        self,
        tmp_repo: Path,
        basic_spec: ValidationSpec,
        context_in_place: ValidationContext,
        command_runner: CommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
    ) -> None:
        """setup_workspace should create log directory if not provided."""
        from src.domain.validation.spec_workspace import setup_workspace

        workspace = setup_workspace(
            spec=basic_spec,
            context=context_in_place,
            log_dir=None,
            step_timeout_seconds=None,
            command_runner=command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
        )

        # Log dir should be created
        assert workspace.log_dir is not None
        assert workspace.log_dir.exists()
        assert workspace.artifacts.log_dir == workspace.log_dir

    def test_setup_uses_provided_log_dir(
        self,
        tmp_repo: Path,
        basic_spec: ValidationSpec,
        context_in_place: ValidationContext,
        command_runner: CommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
    ) -> None:
        """setup_workspace should use provided log_dir."""
        from src.domain.validation.spec_workspace import setup_workspace

        log_dir = tmp_repo / "custom-logs"
        log_dir.mkdir()

        workspace = setup_workspace(
            spec=basic_spec,
            context=context_in_place,
            log_dir=log_dir,
            step_timeout_seconds=None,
            command_runner=command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
        )

        assert workspace.log_dir == log_dir
        assert workspace.artifacts.log_dir == log_dir

    def test_setup_generates_unique_run_id(
        self,
        tmp_repo: Path,
        basic_spec: ValidationSpec,
        context_in_place: ValidationContext,
        command_runner: CommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
    ) -> None:
        """setup_workspace should generate a unique run ID."""
        from src.domain.validation.spec_workspace import setup_workspace

        workspace1 = setup_workspace(
            spec=basic_spec,
            context=context_in_place,
            log_dir=None,
            step_timeout_seconds=None,
            command_runner=command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
        )
        workspace2 = setup_workspace(
            spec=basic_spec,
            context=context_in_place,
            log_dir=None,
            step_timeout_seconds=None,
            command_runner=command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
        )

        assert workspace1.run_id.startswith("run-")
        assert workspace2.run_id.startswith("run-")
        assert workspace1.run_id != workspace2.run_id

    def test_setup_in_place_validation(
        self,
        tmp_repo: Path,
        basic_spec: ValidationSpec,
        context_in_place: ValidationContext,
        command_runner: CommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
    ) -> None:
        """For in-place validation, validation_cwd should be repo_path."""
        from src.domain.validation.spec_workspace import setup_workspace

        workspace = setup_workspace(
            spec=basic_spec,
            context=context_in_place,
            log_dir=None,
            step_timeout_seconds=None,
            command_runner=command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
        )

        # No commit hash = validate in place
        assert workspace.validation_cwd == context_in_place.repo_path
        assert workspace.worktree_ctx is None

    def test_setup_with_worktree(
        self,
        tmp_repo: Path,
        basic_spec: ValidationSpec,
        context_with_commit: ValidationContext,
        command_runner: CommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
    ) -> None:
        """For commit-based validation, setup should create a worktree."""
        from src.domain.validation.spec_workspace import setup_workspace

        mock_worktree = MagicMock(spec=WorktreeContext)
        mock_worktree.state = WorktreeState.CREATED
        mock_worktree.path = tmp_repo / "worktree"
        mock_worktree.path.mkdir()

        with patch(
            "src.domain.validation.spec_workspace.create_worktree",
            return_value=mock_worktree,
        ):
            workspace = setup_workspace(
                spec=basic_spec,
                context=context_with_commit,
                log_dir=None,
                step_timeout_seconds=None,
                command_runner=command_runner,
                env_config=mock_env_config,
                lock_manager=fake_lock_manager,
            )

        assert workspace.worktree_ctx is not None
        assert workspace.validation_cwd == mock_worktree.path
        assert workspace.artifacts.worktree_path == mock_worktree.path


class TestSetupWorkspaceBaseline:
    """Test baseline coverage handling in setup_workspace."""

    def test_setup_refreshes_baseline_when_no_decrease_mode(
        self,
        tmp_repo: Path,
        context_in_place: ValidationContext,
        command_runner: CommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
    ) -> None:
        """When coverage.min_percent is None, setup should refresh baseline."""
        from src.domain.validation.spec_workspace import setup_workspace
        from src.infra.tools.command_runner import CommandResult

        yaml_coverage_config = YamlCoverageConfig(
            format="xml",
            file="coverage.xml",
            threshold=0.0,
            command="uv run pytest --cov=src --cov-report=xml",
        )
        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command="pytest",
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_SESSION,
            coverage=CoverageConfig(enabled=True, min_percent=None),
            e2e=E2EConfig(enabled=False),
            yaml_coverage_config=yaml_coverage_config,
        )

        # Create baseline file
        baseline_xml = tmp_repo / "coverage.xml"
        baseline_xml.write_text(
            '<?xml version="1.0"?>\n<coverage line-rate="0.85" branch-rate="0.80" />'
        )
        import os

        future_time = 4102444800
        os.utime(baseline_xml, (future_time, future_time))

        def mock_git_run(args: list[str], **kwargs: object) -> CommandResult:
            if "status" in args and "--porcelain" in args:
                return CommandResult(command=args, returncode=0, stdout="", stderr="")
            elif "log" in args:
                return CommandResult(
                    command=args, returncode=0, stdout="1700000000\n", stderr=""
                )
            return CommandResult(command=args, returncode=0, stdout="", stderr="")

        with patch(
            "src.domain.validation.coverage.is_baseline_stale", return_value=False
        ):
            workspace = setup_workspace(
                spec=spec,
                context=context_in_place,
                log_dir=None,
                step_timeout_seconds=None,
                command_runner=command_runner,
                env_config=mock_env_config,
                lock_manager=fake_lock_manager,
            )

        assert workspace.baseline_percent == 85.0

    def test_setup_skips_baseline_when_threshold_explicit(
        self,
        tmp_repo: Path,
        context_in_place: ValidationContext,
        command_runner: CommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
    ) -> None:
        """When coverage.min_percent is explicit, baseline refresh is skipped."""
        from src.domain.validation.spec_workspace import setup_workspace

        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command="pytest",
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_SESSION,
            coverage=CoverageConfig(enabled=True, min_percent=80.0),
            e2e=E2EConfig(enabled=False),
        )

        workspace = setup_workspace(
            spec=spec,
            context=context_in_place,
            log_dir=None,
            step_timeout_seconds=None,
            command_runner=command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
        )

        # No baseline refresh needed when explicit threshold
        assert workspace.baseline_percent is None

    def test_setup_skips_baseline_when_coverage_disabled(
        self,
        tmp_repo: Path,
        basic_spec: ValidationSpec,
        context_in_place: ValidationContext,
        command_runner: CommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
    ) -> None:
        """When coverage is disabled, baseline refresh is skipped."""
        from src.domain.validation.spec_workspace import setup_workspace

        workspace = setup_workspace(
            spec=basic_spec,  # coverage disabled
            context=context_in_place,
            log_dir=None,
            step_timeout_seconds=None,
            command_runner=command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
        )

        assert workspace.baseline_percent is None


class TestSetupWorkspaceErrors:
    """Test error handling in setup_workspace."""

    def test_setup_returns_error_on_baseline_failure(
        self,
        tmp_repo: Path,
        context_in_place: ValidationContext,
        command_runner: CommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
    ) -> None:
        """When baseline refresh fails, setup should return an error result."""
        from src.domain.validation.spec_workspace import (
            SetupError,
            setup_workspace,
        )
        from src.infra.tools.command_runner import CommandResult

        yaml_coverage_config = YamlCoverageConfig(
            format="xml",
            file="coverage.xml",
            threshold=0.0,
            command="uv run pytest --cov=src --cov-report=xml",
        )
        spec = ValidationSpec(
            commands=[
                ValidationCommand(
                    name="pytest",
                    command="pytest",
                    kind=CommandKind.TEST,
                ),
            ],
            scope=ValidationScope.PER_SESSION,
            coverage=CoverageConfig(enabled=True, min_percent=None),
            e2e=E2EConfig(enabled=False),
            yaml_coverage_config=yaml_coverage_config,
        )

        # No baseline file, and refresh will fail
        def mock_git_run(args: list[str], **kwargs: object) -> CommandResult:
            if "status" in args and "--porcelain" in args:
                # Uncommitted changes - but no worktree to run tests
                return CommandResult(
                    command=args, returncode=0, stdout="M dirty.py\n", stderr=""
                )
            elif "log" in args:
                return CommandResult(
                    command=args, returncode=0, stdout="1700000000\n", stderr=""
                )
            return CommandResult(command=args, returncode=0, stdout="", stderr="")

        # Mock worktree creation to fail
        mock_worktree = MagicMock(spec=WorktreeContext)
        mock_worktree.state = WorktreeState.FAILED
        mock_worktree.error = "Baseline refresh failed"

        with (
            patch(
                "src.domain.validation.coverage.is_baseline_stale", return_value=False
            ),
            patch(
                "src.domain.validation.worktree.create_worktree",
                return_value=mock_worktree,
            ),
            patch("src.infra.tools.locking.try_lock", return_value=True),
        ):
            with pytest.raises(SetupError) as exc_info:
                setup_workspace(
                    spec=spec,
                    context=context_in_place,
                    log_dir=None,
                    step_timeout_seconds=None,
                    command_runner=command_runner,
                    env_config=mock_env_config,
                    lock_manager=fake_lock_manager,
                )

        assert "Baseline refresh failed" in str(exc_info.value)

    def test_setup_returns_error_on_worktree_failure(
        self,
        tmp_repo: Path,
        basic_spec: ValidationSpec,
        context_with_commit: ValidationContext,
        command_runner: CommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
    ) -> None:
        """When worktree creation fails, setup should return an error result."""
        from src.domain.validation.spec_workspace import (
            SetupError,
            setup_workspace,
        )

        mock_worktree = MagicMock(spec=WorktreeContext)
        mock_worktree.state = WorktreeState.FAILED
        mock_worktree.error = "git worktree add failed"

        with patch(
            "src.domain.validation.spec_workspace.create_worktree",
            return_value=mock_worktree,
        ):
            with pytest.raises(SetupError) as exc_info:
                setup_workspace(
                    spec=basic_spec,
                    context=context_with_commit,
                    log_dir=None,
                    step_timeout_seconds=None,
                    command_runner=command_runner,
                    env_config=mock_env_config,
                    lock_manager=fake_lock_manager,
                )

        assert "Worktree creation failed" in str(exc_info.value)


class TestCleanupWorkspace:
    """Test cleanup_workspace function."""

    def test_cleanup_removes_worktree_on_success(
        self, tmp_repo: Path, command_runner: CommandRunner
    ) -> None:
        """On validation success, cleanup should remove the worktree."""
        from src.domain.validation.spec_workspace import (
            SpecRunWorkspace,
            cleanup_workspace,
        )

        worktree_path = tmp_repo / "worktree"
        worktree_path.mkdir()

        mock_worktree = MagicMock(spec=WorktreeContext)
        mock_worktree.state = WorktreeState.CREATED
        mock_worktree.path = worktree_path

        mock_worktree_removed = MagicMock(spec=WorktreeContext)
        mock_worktree_removed.state = WorktreeState.REMOVED

        workspace = SpecRunWorkspace(
            validation_cwd=worktree_path,
            artifacts=ValidationArtifacts(log_dir=tmp_repo / "logs"),
            baseline_percent=None,
            run_id="run-test",
            log_dir=tmp_repo / "logs",
            worktree_ctx=mock_worktree,
        )

        with patch(
            "src.domain.validation.spec_workspace.remove_worktree",
            return_value=mock_worktree_removed,
        ):
            cleanup_workspace(
                workspace, validation_passed=True, command_runner=command_runner
            )

        assert workspace.artifacts.worktree_state == "removed"

    def test_cleanup_keeps_worktree_on_failure(
        self, tmp_repo: Path, command_runner: CommandRunner
    ) -> None:
        """On validation failure, cleanup should keep the worktree."""
        from src.domain.validation.spec_workspace import (
            SpecRunWorkspace,
            cleanup_workspace,
        )

        worktree_path = tmp_repo / "worktree"
        worktree_path.mkdir()

        mock_worktree = MagicMock(spec=WorktreeContext)
        mock_worktree.state = WorktreeState.CREATED
        mock_worktree.path = worktree_path

        mock_worktree_kept = MagicMock(spec=WorktreeContext)
        mock_worktree_kept.state = WorktreeState.KEPT

        workspace = SpecRunWorkspace(
            validation_cwd=worktree_path,
            artifacts=ValidationArtifacts(log_dir=tmp_repo / "logs"),
            baseline_percent=None,
            run_id="run-test",
            log_dir=tmp_repo / "logs",
            worktree_ctx=mock_worktree,
        )

        with patch(
            "src.domain.validation.spec_workspace.remove_worktree",
            return_value=mock_worktree_kept,
        ):
            cleanup_workspace(
                workspace, validation_passed=False, command_runner=command_runner
            )

        assert workspace.artifacts.worktree_state == "kept"

    def test_cleanup_noop_without_worktree(
        self, tmp_repo: Path, command_runner: CommandRunner
    ) -> None:
        """Cleanup should be a no-op when there's no worktree."""
        from src.domain.validation.spec_workspace import (
            SpecRunWorkspace,
            cleanup_workspace,
        )

        workspace = SpecRunWorkspace(
            validation_cwd=tmp_repo,
            artifacts=ValidationArtifacts(log_dir=tmp_repo / "logs"),
            baseline_percent=None,
            run_id="run-test",
            log_dir=tmp_repo / "logs",
            worktree_ctx=None,  # No worktree
        )

        # Should not raise or modify anything
        cleanup_workspace(
            workspace, validation_passed=True, command_runner=command_runner
        )

        assert workspace.artifacts.worktree_state is None


class TestWorkspaceContextManager:
    """Test workspace_context context manager."""

    def test_context_manager_yields_workspace(
        self,
        tmp_repo: Path,
        basic_spec: ValidationSpec,
        context_in_place: ValidationContext,
        command_runner: CommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
    ) -> None:
        """workspace_context should yield a SpecRunWorkspace."""
        from src.domain.validation.spec_workspace import workspace_context

        with workspace_context(
            spec=basic_spec,
            context=context_in_place,
            log_dir=None,
            step_timeout_seconds=None,
            command_runner=command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
        ) as workspace:
            assert workspace.validation_cwd == context_in_place.repo_path
            assert workspace.artifacts is not None

    def test_context_manager_cleans_up_on_success(
        self,
        tmp_repo: Path,
        basic_spec: ValidationSpec,
        context_with_commit: ValidationContext,
        command_runner: CommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
    ) -> None:
        """workspace_context should cleanup worktree on normal exit."""
        from src.domain.validation.spec_workspace import workspace_context

        mock_worktree = MagicMock(spec=WorktreeContext)
        mock_worktree.state = WorktreeState.CREATED
        mock_worktree.path = tmp_repo / "worktree"
        mock_worktree.path.mkdir()

        mock_worktree_removed = MagicMock(spec=WorktreeContext)
        mock_worktree_removed.state = WorktreeState.REMOVED

        remove_called_with: list[tuple[object, bool]] = []

        def mock_remove(
            ctx: object, validation_passed: bool, command_runner: object
        ) -> MagicMock:
            remove_called_with.append((ctx, validation_passed))
            return mock_worktree_removed

        with (
            patch(
                "src.domain.validation.spec_workspace.create_worktree",
                return_value=mock_worktree,
            ),
            patch(
                "src.domain.validation.spec_workspace.remove_worktree",
                side_effect=mock_remove,
            ),
        ):
            with workspace_context(
                spec=basic_spec,
                context=context_with_commit,
                log_dir=None,
                step_timeout_seconds=None,
                command_runner=command_runner,
                env_config=mock_env_config,
                lock_manager=fake_lock_manager,
            ) as workspace:
                # Normal execution
                _ = workspace

        # Cleanup should be called with validation_passed=True (no exception)
        assert len(remove_called_with) == 1
        assert remove_called_with[0][1] is True

    def test_context_manager_cleans_up_on_exception(
        self,
        tmp_repo: Path,
        basic_spec: ValidationSpec,
        context_with_commit: ValidationContext,
        command_runner: CommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
    ) -> None:
        """workspace_context should cleanup worktree on exception."""
        from src.domain.validation.spec_workspace import workspace_context

        mock_worktree = MagicMock(spec=WorktreeContext)
        mock_worktree.state = WorktreeState.CREATED
        mock_worktree.path = tmp_repo / "worktree"
        mock_worktree.path.mkdir()

        mock_worktree_kept = MagicMock(spec=WorktreeContext)
        mock_worktree_kept.state = WorktreeState.KEPT

        remove_called_with: list[tuple[object, bool]] = []

        def mock_remove(
            ctx: object, validation_passed: bool, command_runner: object
        ) -> MagicMock:
            remove_called_with.append((ctx, validation_passed))
            return mock_worktree_kept

        with (
            patch(
                "src.domain.validation.spec_workspace.create_worktree",
                return_value=mock_worktree,
            ),
            patch(
                "src.domain.validation.spec_workspace.remove_worktree",
                side_effect=mock_remove,
            ),
        ):
            with pytest.raises(ValueError, match="test error"):
                with workspace_context(
                    spec=basic_spec,
                    context=context_with_commit,
                    log_dir=None,
                    step_timeout_seconds=None,
                    command_runner=command_runner,
                    env_config=mock_env_config,
                    lock_manager=fake_lock_manager,
                ) as workspace:
                    _ = workspace
                    raise ValueError("test error")

        # Cleanup should be called with validation_passed=False (exception)
        assert len(remove_called_with) == 1
        assert remove_called_with[0][1] is False
