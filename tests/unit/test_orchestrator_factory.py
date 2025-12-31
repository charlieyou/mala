"""Unit tests for orchestrator_factory module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import MalaConfig
from src.orchestrator_factory import (
    OrchestratorConfig,
    OrchestratorDependencies,
    create_orchestrator,
    _resolve_feature_flags,
    _check_review_availability,
)


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a temporary repository with basic structure."""
    # Create a pyproject.toml to make it look like a Python project
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    return tmp_path


@pytest.fixture
def basic_config(tmp_repo: Path) -> OrchestratorConfig:
    """Create a basic OrchestratorConfig for testing."""
    return OrchestratorConfig(
        repo_path=tmp_repo,
        max_agents=2,
        timeout_minutes=30,
    )


@pytest.fixture
def mock_mala_config() -> MalaConfig:
    """Create a mock MalaConfig for testing."""
    return MalaConfig(
        runs_dir=Path("/tmp/test-runs"),
        lock_dir=Path("/tmp/test-locks"),
        claude_config_dir=Path("/tmp/test-claude"),
        braintrust_enabled=False,
        morph_enabled=False,
    )


@pytest.fixture
def mock_issue_provider() -> MagicMock:
    """Create a mock IssueProvider."""
    provider = MagicMock()
    provider.get_ready_async = AsyncMock(return_value=[])
    provider.claim_async = AsyncMock(return_value=True)
    provider.close_async = AsyncMock(return_value=True)
    provider.mark_needs_followup_async = AsyncMock(return_value=True)
    provider.get_issue_description_async = AsyncMock(return_value="Test issue")
    provider.close_eligible_epics_async = AsyncMock(return_value=False)
    provider.commit_issues_async = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def mock_gate_checker() -> MagicMock:
    """Create a mock GateChecker."""
    return MagicMock()


@pytest.fixture
def mock_event_sink() -> MagicMock:
    """Create a mock MalaEventSink."""
    sink = MagicMock()
    sink.on_warning = MagicMock()
    return sink


@pytest.mark.unit
class TestOrchestratorConfig:
    """Tests for OrchestratorConfig dataclass."""

    def test_minimal_config(self, tmp_repo: Path) -> None:
        """Test creating config with only required field."""
        config = OrchestratorConfig(repo_path=tmp_repo)
        assert config.repo_path == tmp_repo
        assert config.max_agents is None
        assert config.timeout_minutes is None
        assert config.max_gate_retries == 3
        assert config.focus is True

    def test_full_config(self, tmp_repo: Path) -> None:
        """Test creating config with all fields."""
        config = OrchestratorConfig(
            repo_path=tmp_repo,
            max_agents=4,
            timeout_minutes=60,
            max_issues=10,
            epic_id="epic-123",
            only_ids={"issue-1", "issue-2"},
            braintrust_enabled=True,
            max_gate_retries=5,
            max_review_retries=2,
            disable_validations={"coverage"},
            coverage_threshold=80.0,
            morph_enabled=True,
            prioritize_wip=True,
            focus=False,
            cli_args={"verbose": True},
            epic_override_ids={"epic-456"},
        )
        assert config.max_agents == 4
        assert config.epic_id == "epic-123"
        assert config.only_ids == {"issue-1", "issue-2"}
        assert config.braintrust_enabled is True
        assert config.prioritize_wip is True
        assert config.focus is False


@pytest.mark.unit
class TestOrchestratorDependencies:
    """Tests for OrchestratorDependencies dataclass."""

    def test_empty_deps(self) -> None:
        """Test creating dependencies with no fields."""
        deps = OrchestratorDependencies()
        assert deps.issue_provider is None
        assert deps.code_reviewer is None
        assert deps.gate_checker is None
        assert deps.log_provider is None
        assert deps.telemetry_provider is None
        assert deps.event_sink is None

    def test_partial_deps(
        self, mock_issue_provider: MagicMock, mock_gate_checker: MagicMock
    ) -> None:
        """Test creating dependencies with some fields."""
        deps = OrchestratorDependencies(
            issue_provider=mock_issue_provider,
            gate_checker=mock_gate_checker,
        )
        assert deps.issue_provider is mock_issue_provider
        assert deps.gate_checker is mock_gate_checker
        assert deps.code_reviewer is None


@pytest.mark.unit
class TestResolveFeatureFlags:
    """Tests for _resolve_feature_flags function."""

    def test_explicit_config_overrides_mala_config(
        self, tmp_repo: Path, mock_mala_config: MalaConfig
    ) -> None:
        """Test that explicit OrchestratorConfig values override MalaConfig."""
        config = OrchestratorConfig(
            repo_path=tmp_repo,
            braintrust_enabled=True,
            morph_enabled=True,
        )
        bt, morph = _resolve_feature_flags(config, mock_mala_config)
        assert bt is True
        assert morph is True

    def test_falls_back_to_mala_config(self, tmp_repo: Path) -> None:
        """Test that None in OrchestratorConfig falls back to MalaConfig."""
        config = OrchestratorConfig(repo_path=tmp_repo)
        mala_config = MalaConfig(
            runs_dir=Path("/tmp/test-runs"),
            lock_dir=Path("/tmp/test-locks"),
            claude_config_dir=Path("/tmp/test-claude"),
            braintrust_enabled=True,
            morph_enabled=True,
            braintrust_api_key="test-key",
            morph_api_key="test-key",
        )
        bt, morph = _resolve_feature_flags(config, mala_config)
        assert bt is True
        assert morph is True


@pytest.mark.unit
class TestCheckReviewAvailability:
    """Tests for _check_review_availability function."""

    def test_review_already_disabled(self, mock_mala_config: MalaConfig) -> None:
        """Test that review disabled in validations returns None."""
        result = _check_review_availability(mock_mala_config, {"review"})
        assert result is None

    def test_review_gate_in_path(self, mock_mala_config: MalaConfig) -> None:
        """Test that review-gate in PATH returns None (available)."""
        with patch("shutil.which", return_value="/usr/bin/review-gate"):
            result = _check_review_availability(mock_mala_config, set())
            assert result is None

    def test_review_gate_not_found(self, mock_mala_config: MalaConfig) -> None:
        """Test that missing review-gate returns disabled reason."""
        with patch("shutil.which", return_value=None):
            result = _check_review_availability(mock_mala_config, set())
            assert result == "cerberus plugin not detected (review-gate unavailable)"


@pytest.mark.unit
class TestCreateOrchestrator:
    """Tests for create_orchestrator factory function."""

    def test_creates_orchestrator_with_minimal_config(
        self,
        basic_config: OrchestratorConfig,
        mock_mala_config: MalaConfig,
        mock_issue_provider: MagicMock,
        mock_gate_checker: MagicMock,
        mock_event_sink: MagicMock,
    ) -> None:
        """Test creating orchestrator with minimal config and mocked deps."""
        deps = OrchestratorDependencies(
            issue_provider=mock_issue_provider,
            gate_checker=mock_gate_checker,
            event_sink=mock_event_sink,
        )

        orchestrator = create_orchestrator(
            config=basic_config,
            mala_config=mock_mala_config,
            deps=deps,
        )

        assert orchestrator.repo_path == basic_config.repo_path.resolve()
        assert orchestrator.max_agents == 2
        assert orchestrator.timeout_seconds == 30 * 60
        assert orchestrator.beads is mock_issue_provider
        assert orchestrator.quality_gate is mock_gate_checker
        assert orchestrator.event_sink is mock_event_sink

    def test_uses_env_config_when_mala_config_none(
        self,
        basic_config: OrchestratorConfig,
        mock_issue_provider: MagicMock,
        mock_gate_checker: MagicMock,
        mock_event_sink: MagicMock,
    ) -> None:
        """Test that MalaConfig.from_env is called when mala_config is None."""
        deps = OrchestratorDependencies(
            issue_provider=mock_issue_provider,
            gate_checker=mock_gate_checker,
            event_sink=mock_event_sink,
        )

        with patch.object(MalaConfig, "from_env") as mock_from_env:
            mock_from_env.return_value = MalaConfig(
                runs_dir=Path("/tmp/test-runs"),
                lock_dir=Path("/tmp/test-locks"),
                claude_config_dir=Path("/tmp/test-claude"),
            )

            orchestrator = create_orchestrator(
                config=basic_config,
                mala_config=None,
                deps=deps,
            )

            mock_from_env.assert_called_once_with(validate=False)
            assert orchestrator is not None

    def test_feature_flags_propagated(
        self,
        tmp_repo: Path,
        mock_issue_provider: MagicMock,
        mock_gate_checker: MagicMock,
        mock_event_sink: MagicMock,
    ) -> None:
        """Test that feature flags are correctly propagated to orchestrator."""
        config = OrchestratorConfig(
            repo_path=tmp_repo,
            braintrust_enabled=True,
            morph_enabled=True,
        )
        mala_config = MalaConfig(
            runs_dir=Path("/tmp/test-runs"),
            lock_dir=Path("/tmp/test-locks"),
            claude_config_dir=Path("/tmp/test-claude"),
            braintrust_api_key="test-key",
            morph_api_key="test-key",
        )
        deps = OrchestratorDependencies(
            issue_provider=mock_issue_provider,
            gate_checker=mock_gate_checker,
            event_sink=mock_event_sink,
        )

        orchestrator = create_orchestrator(
            config=config,
            mala_config=mala_config,
            deps=deps,
        )

        assert orchestrator.braintrust_enabled is True
        assert orchestrator.morph_enabled is True

    def test_disabled_validations_propagated(
        self,
        tmp_repo: Path,
        mock_issue_provider: MagicMock,
        mock_gate_checker: MagicMock,
        mock_event_sink: MagicMock,
        mock_mala_config: MalaConfig,
    ) -> None:
        """Test that disabled validations are propagated."""
        config = OrchestratorConfig(
            repo_path=tmp_repo,
            disable_validations={"coverage", "integration-tests"},
        )
        deps = OrchestratorDependencies(
            issue_provider=mock_issue_provider,
            gate_checker=mock_gate_checker,
            event_sink=mock_event_sink,
        )

        orchestrator = create_orchestrator(
            config=config,
            mala_config=mock_mala_config,
            deps=deps,
        )

        assert "coverage" in orchestrator._disabled_validations
        assert "integration-tests" in orchestrator._disabled_validations

    def test_cli_args_propagated(
        self,
        tmp_repo: Path,
        mock_issue_provider: MagicMock,
        mock_gate_checker: MagicMock,
        mock_event_sink: MagicMock,
        mock_mala_config: MalaConfig,
    ) -> None:
        """Test that CLI args are propagated."""
        cli_args = {"verbose": True, "dry_run": False}
        config = OrchestratorConfig(
            repo_path=tmp_repo,
            cli_args=cli_args,
        )
        deps = OrchestratorDependencies(
            issue_provider=mock_issue_provider,
            gate_checker=mock_gate_checker,
            event_sink=mock_event_sink,
        )

        orchestrator = create_orchestrator(
            config=config,
            mala_config=mock_mala_config,
            deps=deps,
        )

        assert orchestrator.cli_args == cli_args


@pytest.mark.unit
class TestLegacyCompatibility:
    """Tests for backward compatibility with legacy initialization."""

    def test_legacy_init_still_works(
        self,
        tmp_repo: Path,
        mock_issue_provider: MagicMock,
        mock_gate_checker: MagicMock,
        mock_event_sink: MagicMock,
    ) -> None:
        """Test that legacy __init__ still works."""
        from src.orchestrator import MalaOrchestrator

        mala_config = MalaConfig(
            runs_dir=Path("/tmp/test-runs"),
            lock_dir=Path("/tmp/test-locks"),
            claude_config_dir=Path("/tmp/test-claude"),
        )

        orchestrator = MalaOrchestrator(
            repo_path=tmp_repo,
            max_agents=2,
            timeout_minutes=30,
            issue_provider=mock_issue_provider,
            gate_checker=mock_gate_checker,
            event_sink=mock_event_sink,
            config=mala_config,
        )

        assert orchestrator.repo_path == tmp_repo.resolve()
        assert orchestrator.max_agents == 2
        assert orchestrator.timeout_seconds == 30 * 60

    def test_legacy_init_requires_repo_path(self) -> None:
        """Test that legacy init without repo_path raises ValueError."""
        from src.orchestrator import MalaOrchestrator

        with pytest.raises(ValueError, match="repo_path is required"):
            MalaOrchestrator()

    def test_factory_and_legacy_produce_equivalent_orchestrators(
        self,
        tmp_repo: Path,
        mock_issue_provider: MagicMock,
        mock_gate_checker: MagicMock,
        mock_event_sink: MagicMock,
    ) -> None:
        """Test that factory and legacy initialization produce equivalent orchestrators."""
        from src.orchestrator import MalaOrchestrator

        mala_config = MalaConfig(
            runs_dir=Path("/tmp/test-runs"),
            lock_dir=Path("/tmp/test-locks"),
            claude_config_dir=Path("/tmp/test-claude"),
        )

        # Factory initialization
        config = OrchestratorConfig(
            repo_path=tmp_repo,
            max_agents=2,
            timeout_minutes=30,
            max_gate_retries=5,
        )
        deps = OrchestratorDependencies(
            issue_provider=mock_issue_provider,
            gate_checker=mock_gate_checker,
            event_sink=mock_event_sink,
        )
        factory_orch = create_orchestrator(
            config=config,
            mala_config=mala_config,
            deps=deps,
        )

        # Legacy initialization
        legacy_orch = MalaOrchestrator(
            repo_path=tmp_repo,
            max_agents=2,
            timeout_minutes=30,
            max_gate_retries=5,
            issue_provider=mock_issue_provider,
            gate_checker=mock_gate_checker,
            event_sink=mock_event_sink,
            config=mala_config,
        )

        # Compare key attributes
        assert factory_orch.repo_path == legacy_orch.repo_path
        assert factory_orch.max_agents == legacy_orch.max_agents
        assert factory_orch.timeout_seconds == legacy_orch.timeout_seconds
        assert factory_orch.max_gate_retries == legacy_orch.max_gate_retries
