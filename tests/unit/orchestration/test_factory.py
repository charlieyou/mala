"""Unit tests for orchestration factory functions.

Tests for:
- _check_review_availability: Review availability checking by reviewer_type
- _build_dependencies: RuntimeDeps assembly with explicit DI overrides

Precedence/timeout/reviewer-type tests live in test_config_resolution.py.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.infra.io.config import MalaConfig
from src.orchestration.factory import (
    _check_review_availability,
    _with_cerberus_bin_path,
)
from src.orchestration.types import OrchestratorConfig


class TestCheckReviewAvailability:
    """Tests for _check_review_availability function."""

    @pytest.fixture
    def mala_config(self) -> MalaConfig:
        """Create a minimal MalaConfig for testing."""
        return MalaConfig(
            runs_dir=Path("/tmp/runs"),
            lock_dir=Path("/tmp/locks"),
        )

    def test_agent_sdk_reviewer_is_available(self, mala_config: MalaConfig) -> None:
        """agent_sdk reviewer should always be available (no external deps)."""
        result = _check_review_availability(
            mala_config=mala_config,
            disabled_validations=set(),
            reviewer_type="agent_sdk",
        )
        assert result is None

    def test_explicitly_disabled_review_returns_none(
        self, mala_config: MalaConfig
    ) -> None:
        """Explicitly disabled review should return None (no warning needed)."""
        result = _check_review_availability(
            mala_config=mala_config,
            disabled_validations={"review"},
            reviewer_type="agent_sdk",
        )
        assert result is None

    def test_unknown_reviewer_type_returns_reason(
        self, mala_config: MalaConfig
    ) -> None:
        """Unknown reviewer_type should disable review with warning."""
        result = _check_review_availability(
            mala_config=mala_config,
            disabled_validations=set(),
            reviewer_type="unknown_type",
        )
        assert result is not None
        assert "unknown reviewer_type" in result
        assert "unknown_type" in result

    def test_cerberus_without_binary_returns_reason(
        self, mala_config: MalaConfig
    ) -> None:
        """cerberus reviewer without binary should disable review."""
        # Patch shutil.which to return None (no binary found)
        with patch("shutil.which", return_value=None):
            result = _check_review_availability(
                mala_config=mala_config,
                disabled_validations=set(),
                reviewer_type="cerberus",
            )
        assert result is not None
        assert "review-gate" in result

    def test_cerberus_with_binary_is_available(self, mala_config: MalaConfig) -> None:
        """cerberus reviewer with binary available should return None."""
        # Patch shutil.which to return a path (binary found)
        with patch("shutil.which", return_value="/usr/bin/review-gate"):
            result = _check_review_availability(
                mala_config=mala_config,
                disabled_validations=set(),
                reviewer_type="cerberus",
            )
        assert result is None

    def test_cerberus_with_explicit_bin_path_existing(self) -> None:
        """cerberus reviewer with explicit bin_path to existing binary is available."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir)
            # Create the review-gate binary
            review_gate = bin_path / "review-gate"
            review_gate.touch()
            review_gate.chmod(0o755)

            config = MalaConfig(
                runs_dir=Path("/tmp/runs"),
                lock_dir=Path("/tmp/locks"),
                cerberus_bin_path=bin_path,
            )
            result = _check_review_availability(
                mala_config=config,
                disabled_validations=set(),
                reviewer_type="cerberus",
            )
        assert result is None

    def test_cerberus_with_explicit_bin_path_missing_binary(self) -> None:
        """cerberus reviewer with explicit bin_path but missing binary is disabled."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir)
            # Do NOT create the review-gate binary

            config = MalaConfig(
                runs_dir=Path("/tmp/runs"),
                lock_dir=Path("/tmp/locks"),
                cerberus_bin_path=bin_path,
            )
            result = _check_review_availability(
                mala_config=config,
                disabled_validations=set(),
                reviewer_type="cerberus",
            )
        assert result is not None
        assert "review-gate" in result


class TestWithCerberusBinPath:
    """Tests for Cerberus runtime PATH injection."""

    def test_preserves_system_path_when_env_has_no_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Configured Cerberus bin prepends to current PATH when env omits PATH."""
        bin_path = tmp_path / "cerberus-bin"
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        config = MalaConfig(
            runs_dir=Path("/tmp/runs"),
            lock_dir=Path("/tmp/locks"),
            cerberus_bin_path=bin_path,
        )

        result = _with_cerberus_bin_path({"CERBERUS_TOKEN": "secret"}, config)

        assert result["PATH"] == f"{bin_path}:/usr/bin:/bin"
        assert result["CERBERUS_TOKEN"] == "secret"


class TestBuildDependenciesRuntimeDeps:
    """Tests for RuntimeDeps construction in _build_dependencies."""

    def test_constructs_defaults_when_deps_none(self, tmp_path: Path) -> None:
        """Factory constructs CommandRunner, EnvConfig, LockManager when deps=None."""
        from src.infra.tools.command_runner import CommandRunner
        from src.infra.tools.env import EnvConfig
        from src.infra.tools.locking import LockManager
        from src.orchestration.config_resolution import _ReviewerConfig
        from src.orchestration.factory import _build_dependencies
        from src.orchestration.types import _DerivedConfig

        config = OrchestratorConfig(repo_path=tmp_path)
        mala_config = MalaConfig.from_env(validate=False)
        derived = _DerivedConfig(
            timeout_seconds=600,
            disabled_validations=set(),
            max_idle_retries=3,
            idle_timeout_seconds=None,
        )
        reviewer_config = _ReviewerConfig()

        result = _build_dependencies(
            config, mala_config, derived, None, reviewer_config
        )

        # Verify types are concrete implementations
        assert isinstance(result.command_runner, CommandRunner)
        assert isinstance(result.env_config, EnvConfig)
        assert isinstance(result.lock_manager, LockManager)

    def test_uses_provided_command_runner(self, tmp_path: Path) -> None:
        """Factory uses provided command_runner instead of creating default."""
        from tests.fakes.command_runner import FakeCommandRunner
        from src.orchestration.config_resolution import _ReviewerConfig
        from src.orchestration.factory import _build_dependencies
        from src.orchestration.types import OrchestratorDependencies, _DerivedConfig

        fake_runner = FakeCommandRunner(allow_unregistered=True)
        deps = OrchestratorDependencies(command_runner=fake_runner)
        config = OrchestratorConfig(repo_path=tmp_path)
        mala_config = MalaConfig.from_env(validate=False)
        derived = _DerivedConfig(
            timeout_seconds=600,
            disabled_validations=set(),
            max_idle_retries=3,
            idle_timeout_seconds=None,
        )
        reviewer_config = _ReviewerConfig()

        result = _build_dependencies(
            config, mala_config, derived, deps, reviewer_config
        )

        assert result.command_runner is fake_runner

    def test_uses_provided_env_config(self, tmp_path: Path) -> None:
        """Factory uses provided env_config instead of creating default."""
        from tests.fakes.env_config import FakeEnvConfig
        from src.orchestration.config_resolution import _ReviewerConfig
        from src.orchestration.factory import _build_dependencies
        from src.orchestration.types import OrchestratorDependencies, _DerivedConfig

        fake_env = FakeEnvConfig()
        deps = OrchestratorDependencies(env_config=fake_env)
        config = OrchestratorConfig(repo_path=tmp_path)
        mala_config = MalaConfig.from_env(validate=False)
        derived = _DerivedConfig(
            timeout_seconds=600,
            disabled_validations=set(),
            max_idle_retries=3,
            idle_timeout_seconds=None,
        )
        reviewer_config = _ReviewerConfig()

        result = _build_dependencies(
            config, mala_config, derived, deps, reviewer_config
        )

        assert result.env_config is fake_env

    def test_uses_provided_lock_manager(self, tmp_path: Path) -> None:
        """Factory uses provided lock_manager instead of creating default."""
        from tests.fakes.lock_manager import FakeLockManager
        from src.orchestration.config_resolution import _ReviewerConfig
        from src.orchestration.factory import _build_dependencies
        from src.orchestration.types import OrchestratorDependencies, _DerivedConfig

        fake_manager = FakeLockManager()
        deps = OrchestratorDependencies(lock_manager=fake_manager)
        config = OrchestratorConfig(repo_path=tmp_path)
        mala_config = MalaConfig.from_env(validate=False)
        derived = _DerivedConfig(
            timeout_seconds=600,
            disabled_validations=set(),
            max_idle_retries=3,
            idle_timeout_seconds=None,
        )
        reviewer_config = _ReviewerConfig()

        result = _build_dependencies(
            config, mala_config, derived, deps, reviewer_config
        )

        assert result.lock_manager is fake_manager

    def test_fills_gaps_with_defaults(self, tmp_path: Path) -> None:
        """Factory fills None fields with defaults while respecting provided ones."""
        from tests.fakes.command_runner import FakeCommandRunner
        from tests.fakes.env_config import FakeEnvConfig
        from src.infra.tools.locking import LockManager
        from src.orchestration.config_resolution import _ReviewerConfig
        from src.orchestration.factory import _build_dependencies
        from src.orchestration.types import OrchestratorDependencies, _DerivedConfig

        # Provide only command_runner and env_config, leave lock_manager as None
        fake_runner = FakeCommandRunner(allow_unregistered=True)
        fake_env = FakeEnvConfig()
        deps = OrchestratorDependencies(
            command_runner=fake_runner,
            env_config=fake_env,
            lock_manager=None,  # Should be filled with default
        )
        config = OrchestratorConfig(repo_path=tmp_path)
        mala_config = MalaConfig.from_env(validate=False)
        derived = _DerivedConfig(
            timeout_seconds=600,
            disabled_validations=set(),
            max_idle_retries=3,
            idle_timeout_seconds=None,
        )
        reviewer_config = _ReviewerConfig()

        result = _build_dependencies(
            config, mala_config, derived, deps, reviewer_config
        )

        # Provided values are used
        assert result.command_runner is fake_runner
        assert result.env_config is fake_env
        # Missing value is filled with default
        assert isinstance(result.lock_manager, LockManager)
