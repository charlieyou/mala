"""Unit tests for orchestration factory functions.

Tests for:
- _check_review_availability: Review availability checking by reviewer_type
- _build_dependencies: RuntimeDeps assembly with explicit DI overrides

Precedence/timeout/reviewer-type tests live in test_config_resolution.py.
"""

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from src.orchestration.factory import (
    _check_epic_verifier_availability,
    _check_review_availability,
)
from src.orchestration.types import OrchestratorConfig
from src.infra.io.config import MalaConfig

if TYPE_CHECKING:
    from src.core.protocols.events import MalaEventSink


def _write_fake_cerberus(root: Path) -> Path:
    """Create a fake Cerberus v2 root with binary and reviewer prompts."""
    bin_dir = root / "bin"
    bin_dir.mkdir()
    (root / "prompts" / "reviewers").mkdir(parents=True)
    binary = bin_dir / "cerberus"
    binary.write_text("#!/usr/bin/env sh\nexit 0\n")
    binary.chmod(0o755)
    return bin_dir


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
        with patch("shutil.which", return_value=None):
            result = _check_review_availability(
                mala_config=mala_config,
                disabled_validations=set(),
                reviewer_type="cerberus",
            )
        assert result == "cerberus binary not found in PATH"

    def test_cerberus_with_binary_and_prompts_is_available(
        self, tmp_path: Path
    ) -> None:
        """cerberus reviewer is available when binary and prompts exist."""
        bin_dir = _write_fake_cerberus(tmp_path)
        config = MalaConfig(cerberus_env=(("PATH", str(bin_dir)),))
        result = _check_review_availability(
            mala_config=config,
            disabled_validations=set(),
            reviewer_type="cerberus",
        )
        assert result is None

    def test_cerberus_with_missing_prompts_returns_reason(self, tmp_path: Path) -> None:
        """cerberus reviewer is disabled when root lacks reviewer prompts."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        binary = bin_dir / "cerberus"
        binary.write_text("#!/usr/bin/env sh\nexit 0\n")
        binary.chmod(0o755)
        config = MalaConfig(cerberus_env=(("PATH", str(bin_dir)),))
        result = _check_review_availability(
            mala_config=config,
            disabled_validations=set(),
            reviewer_type="cerberus",
        )
        assert result == f"cerberus root {tmp_path} missing prompts/reviewers/"

    def test_availability_checks_do_not_invoke_cerberus_subprocess(
        self, tmp_path: Path
    ) -> None:
        """Availability checks must not run cerberus subcommands."""
        bin_dir = _write_fake_cerberus(tmp_path)
        config = MalaConfig(cerberus_env=(("PATH", str(bin_dir)),))
        with patch("subprocess.run") as run:
            assert (
                _check_review_availability(
                    mala_config=config,
                    disabled_validations=set(),
                    reviewer_type="cerberus",
                )
                is None
            )
            assert (
                _check_epic_verifier_availability(
                    reviewer_type="cerberus",
                    mala_config=config,
                )
                is None
            )
        run.assert_not_called()

    def test_cerberus_uses_code_review_cerberus_env_for_preflight(
        self, tmp_path: Path
    ) -> None:
        """code_review.cerberus.env is honored before disabling review."""
        from src.domain.validation.config_types import CerberusConfig

        bin_dir = _write_fake_cerberus(tmp_path)
        result = _check_review_availability(
            mala_config=MalaConfig(cerberus_env=(("PATH", "/missing"),)),
            disabled_validations=set(),
            reviewer_type="cerberus",
            cerberus_config=CerberusConfig(env=(("PATH", str(bin_dir)),)),
        )

        assert result is None


class TestCerberusFactoryWiring:
    """Tests for Cerberus adapter construction."""

    def test_create_code_reviewer_passes_state_root_and_project_key(
        self, tmp_path: Path
    ) -> None:
        from src.infra.clients.cerberus_review import DefaultReviewer
        from src.orchestration.config_resolution import _ReviewerConfig
        from src.orchestration.factory import _create_code_reviewer

        state_root = tmp_path / "runs" / "run-id" / "cerberus"
        config = MalaConfig(
            cerberus_state_root=state_root,
            cerberus_project_key="project-key",
            cerberus_env=(("PATH", "/cerberus/bin"),),
        )

        reviewer = _create_code_reviewer(
            repo_path=tmp_path,
            mala_config=config,
            event_sink=cast("MalaEventSink", None),
            reviewer_config=_ReviewerConfig(reviewer_type="cerberus"),
        )

        assert isinstance(reviewer, DefaultReviewer)
        assert reviewer.state_root == state_root
        assert reviewer.project_key == "project-key"
        assert reviewer.env == {"PATH": "/cerberus/bin"}

    def test_create_code_reviewer_derives_project_key(self, tmp_path: Path) -> None:
        from src.infra.clients.cerberus_review import DefaultReviewer
        from src.orchestration.config_resolution import _ReviewerConfig
        from src.orchestration.factory import _create_code_reviewer

        config = MalaConfig(runs_dir=tmp_path / "runs")
        expected = hashlib.sha256(os.path.abspath(tmp_path).encode()).hexdigest()[:16]

        reviewer = _create_code_reviewer(
            repo_path=tmp_path,
            mala_config=config,
            event_sink=cast("MalaEventSink", None),
            reviewer_config=_ReviewerConfig(reviewer_type="cerberus"),
        )

        assert isinstance(reviewer, DefaultReviewer)
        assert reviewer.state_root == tmp_path / "runs" / "cerberus"
        assert reviewer.project_key == expected

    def test_create_epic_verifier_passes_state_root_and_project_key(
        self, tmp_path: Path
    ) -> None:
        from src.infra.clients.cerberus_epic_verifier import CerberusEpicVerifier
        from src.orchestration.factory import _create_epic_verification_model

        state_root = tmp_path / "runs" / "run-id" / "cerberus"
        config = MalaConfig(
            cerberus_state_root=state_root,
            cerberus_project_key="project-key",
        )

        model = _create_epic_verification_model(
            reviewer_type="cerberus",
            repo_path=tmp_path,
            timeout_ms=60000,
            mala_config=config,
        )

        assert isinstance(model, CerberusEpicVerifier)
        assert model.state_root == state_root
        assert model.project_key == "project-key"

    def test_cerberus_env_project_key_overrides_derived_key(
        self, tmp_path: Path
    ) -> None:
        from src.domain.validation.config_types import CerberusConfig
        from src.infra.clients.cerberus_epic_verifier import CerberusEpicVerifier
        from src.orchestration.factory import _create_epic_verification_model

        model = _create_epic_verification_model(
            reviewer_type="cerberus",
            repo_path=tmp_path,
            timeout_ms=60000,
            mala_config=MalaConfig(cerberus_project_key="default-project"),
            cerberus_config=CerberusConfig(
                env=(("CERBERUS_PROJECT_KEY", "pinned-project"),)
            ),
        )

        assert isinstance(model, CerberusEpicVerifier)
        assert model.project_key == "pinned-project"


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
