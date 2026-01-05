"""Unit tests for orchestration wiring dataclasses and build functions.

Tests the frozen dataclasses (RuntimeDeps, PipelineConfig, IssueFilterConfig)
and the build functions that construct pipeline components.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.domain.prompts import PromptProvider
from src.domain.validation.config import PromptValidationCommands
from src.infra.io.config import MalaConfig
from src.orchestration.orchestration_wiring import (
    build_gate_runner,
    build_issue_coordinator,
    build_review_runner,
    build_run_coordinator,
    build_session_config,
)
from src.orchestration.types import (
    IssueFilterConfig,
    PipelineConfig,
    RuntimeDeps,
)
from src.pipeline.gate_runner import AsyncGateRunner, GateRunner
from src.pipeline.issue_execution_coordinator import IssueExecutionCoordinator
from src.pipeline.review_runner import ReviewRunner
from src.pipeline.run_coordinator import RunCoordinator


@pytest.fixture
def mock_prompt_provider() -> PromptProvider:
    """Create a mock PromptProvider with minimal string values."""
    return PromptProvider(
        implementer_prompt="implementer",
        review_followup_prompt="review",
        gate_followup_prompt="gate",
        fixer_prompt="fixer",
        idle_resume_prompt="idle",
        checkpoint_request_prompt="checkpoint",
        continuation_prompt="continuation",
    )


@pytest.fixture
def mock_prompt_validation_commands() -> PromptValidationCommands:
    """Create a mock PromptValidationCommands."""
    return PromptValidationCommands(
        lint="echo lint",
        format="echo format",
        typecheck="echo typecheck",
        test="echo test",
    )


@pytest.fixture
def mock_runtime_deps() -> RuntimeDeps:
    """Create RuntimeDeps with mock protocol implementations."""
    mock_config = MalaConfig(
        runs_dir=Path("/tmp/runs"),
        lock_dir=Path("/tmp/locks"),
        claude_config_dir=Path("/tmp/claude"),
        review_timeout=300,
    )
    return RuntimeDeps(
        quality_gate=MagicMock(),
        code_reviewer=MagicMock(),
        beads=MagicMock(),
        event_sink=MagicMock(),
        command_runner=MagicMock(),
        env_config=MagicMock(),
        lock_manager=MagicMock(),
        mala_config=mock_config,
    )


@pytest.fixture
def mock_pipeline_config(
    mock_prompt_provider: PromptProvider,
    mock_prompt_validation_commands: PromptValidationCommands,
    tmp_path: Path,
) -> PipelineConfig:
    """Create PipelineConfig with test values."""
    return PipelineConfig(
        repo_path=tmp_path,
        timeout_seconds=3600,
        max_gate_retries=3,
        max_review_retries=3,
        coverage_threshold=85.0,
        disabled_validations={"lint"},
        context_restart_threshold=0.90,
        context_limit=200_000,
        prompts=mock_prompt_provider,
        prompt_validation_commands=mock_prompt_validation_commands,
        deadlock_monitor=None,
    )


@pytest.fixture
def mock_issue_filter_config() -> IssueFilterConfig:
    """Create IssueFilterConfig with test values."""
    return IssueFilterConfig(
        max_agents=4,
        max_issues=10,
        epic_id="test-epic",
        only_ids=["issue-1", "issue-2"],
        prioritize_wip=True,
        focus=True,
        orphans_only=False,
        epic_override_ids={"epic-1"},
    )


@pytest.mark.unit
class TestRuntimeDepsDataclass:
    """Tests for RuntimeDeps frozen dataclass."""

    def test_instantiation_with_valid_values(
        self, mock_runtime_deps: RuntimeDeps
    ) -> None:
        """RuntimeDeps can be instantiated with valid protocol implementations."""
        assert mock_runtime_deps.quality_gate is not None
        assert mock_runtime_deps.code_reviewer is not None
        assert mock_runtime_deps.beads is not None
        assert mock_runtime_deps.event_sink is not None
        assert mock_runtime_deps.command_runner is not None
        assert mock_runtime_deps.env_config is not None
        assert mock_runtime_deps.lock_manager is not None
        assert mock_runtime_deps.mala_config is not None

    def test_immutability_raises_frozen_instance_error(
        self, mock_runtime_deps: RuntimeDeps
    ) -> None:
        """Attempting to modify a frozen RuntimeDeps raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            mock_runtime_deps.quality_gate = MagicMock()  # type: ignore[misc]

    def test_all_fields_required(self) -> None:
        """RuntimeDeps requires all fields (no defaults)."""
        with pytest.raises(TypeError, match=r"missing .* required"):
            RuntimeDeps()  # type: ignore[call-arg]


@pytest.mark.unit
class TestPipelineConfigDataclass:
    """Tests for PipelineConfig frozen dataclass."""

    def test_instantiation_with_valid_values(
        self,
        mock_pipeline_config: PipelineConfig,
        tmp_path: Path,
    ) -> None:
        """PipelineConfig can be instantiated with valid values."""
        assert mock_pipeline_config.repo_path == tmp_path
        assert mock_pipeline_config.timeout_seconds == 3600
        assert mock_pipeline_config.max_gate_retries == 3
        assert mock_pipeline_config.max_review_retries == 3
        assert mock_pipeline_config.coverage_threshold == 85.0
        assert mock_pipeline_config.disabled_validations == {"lint"}
        assert mock_pipeline_config.context_restart_threshold == 0.90
        assert mock_pipeline_config.context_limit == 200_000
        assert mock_pipeline_config.prompts is not None
        assert mock_pipeline_config.prompt_validation_commands is not None
        assert mock_pipeline_config.deadlock_monitor is None

    def test_immutability_raises_frozen_instance_error(
        self, mock_pipeline_config: PipelineConfig
    ) -> None:
        """Attempting to modify a frozen PipelineConfig raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            mock_pipeline_config.timeout_seconds = 7200  # type: ignore[misc]

    def test_deadlock_monitor_default_none(
        self,
        mock_prompt_provider: PromptProvider,
        mock_prompt_validation_commands: PromptValidationCommands,
        tmp_path: Path,
    ) -> None:
        """PipelineConfig.deadlock_monitor defaults to None."""
        config = PipelineConfig(
            repo_path=tmp_path,
            timeout_seconds=3600,
            max_gate_retries=3,
            max_review_retries=3,
            coverage_threshold=None,
            disabled_validations=None,
            context_restart_threshold=0.90,
            context_limit=200_000,
            prompts=mock_prompt_provider,
            prompt_validation_commands=mock_prompt_validation_commands,
        )
        assert config.deadlock_monitor is None


@pytest.mark.unit
class TestIssueFilterConfigDataclass:
    """Tests for IssueFilterConfig frozen dataclass."""

    def test_instantiation_with_valid_values(
        self, mock_issue_filter_config: IssueFilterConfig
    ) -> None:
        """IssueFilterConfig can be instantiated with valid values."""
        assert mock_issue_filter_config.max_agents == 4
        assert mock_issue_filter_config.max_issues == 10
        assert mock_issue_filter_config.epic_id == "test-epic"
        assert mock_issue_filter_config.only_ids == ["issue-1", "issue-2"]
        assert mock_issue_filter_config.prioritize_wip is True
        assert mock_issue_filter_config.focus is True
        assert mock_issue_filter_config.orphans_only is False
        assert mock_issue_filter_config.epic_override_ids == {"epic-1"}

    def test_immutability_raises_frozen_instance_error(
        self, mock_issue_filter_config: IssueFilterConfig
    ) -> None:
        """Attempting to modify a frozen IssueFilterConfig raises FrozenInstanceError."""
        with pytest.raises(FrozenInstanceError):
            mock_issue_filter_config.max_agents = 8  # type: ignore[misc]

    def test_default_values(self) -> None:
        """IssueFilterConfig has sensible defaults for all optional fields."""
        config = IssueFilterConfig()
        assert config.max_agents is None
        assert config.max_issues is None
        assert config.epic_id is None
        assert config.only_ids is None
        assert config.prioritize_wip is False
        assert config.focus is True
        assert config.orphans_only is False
        assert config.epic_override_ids == set()

    def test_epic_override_ids_default_factory(self) -> None:
        """IssueFilterConfig.epic_override_ids uses default_factory for empty set."""
        config1 = IssueFilterConfig()
        config2 = IssueFilterConfig()
        # Each instance should get its own set instance
        assert config1.epic_override_ids == set()
        assert config2.epic_override_ids == set()
        assert config1.epic_override_ids is not config2.epic_override_ids


@pytest.mark.unit
class TestBuildGateRunner:
    """Tests for build_gate_runner function."""

    def test_returns_gate_runner_tuple(
        self,
        mock_runtime_deps: RuntimeDeps,
        mock_pipeline_config: PipelineConfig,
    ) -> None:
        """build_gate_runner returns (GateRunner, AsyncGateRunner) tuple."""
        gate_runner, async_gate_runner = build_gate_runner(
            mock_runtime_deps, mock_pipeline_config
        )
        assert isinstance(gate_runner, GateRunner)
        assert isinstance(async_gate_runner, AsyncGateRunner)

    def test_gate_runner_uses_pipeline_config(
        self,
        mock_runtime_deps: RuntimeDeps,
        mock_pipeline_config: PipelineConfig,
    ) -> None:
        """GateRunner is configured with PipelineConfig values."""
        gate_runner, _ = build_gate_runner(mock_runtime_deps, mock_pipeline_config)
        assert gate_runner.config.max_gate_retries == 3
        assert gate_runner.config.coverage_threshold == 85.0
        assert gate_runner.config.disable_validations == {"lint"}


@pytest.mark.unit
class TestBuildReviewRunner:
    """Tests for build_review_runner function."""

    def test_returns_review_runner(
        self,
        mock_runtime_deps: RuntimeDeps,
        mock_pipeline_config: PipelineConfig,
    ) -> None:
        """build_review_runner returns a ReviewRunner instance."""
        review_runner = build_review_runner(mock_runtime_deps, mock_pipeline_config)
        assert isinstance(review_runner, ReviewRunner)

    def test_review_runner_uses_config_values(
        self,
        mock_runtime_deps: RuntimeDeps,
        mock_pipeline_config: PipelineConfig,
    ) -> None:
        """ReviewRunner is configured with values from RuntimeDeps and PipelineConfig."""
        review_runner = build_review_runner(mock_runtime_deps, mock_pipeline_config)
        assert review_runner.config.max_review_retries == 3
        assert review_runner.config.review_timeout == 300


@pytest.mark.unit
class TestBuildRunCoordinator:
    """Tests for build_run_coordinator function."""

    def test_returns_run_coordinator(
        self,
        mock_runtime_deps: RuntimeDeps,
        mock_pipeline_config: PipelineConfig,
    ) -> None:
        """build_run_coordinator returns a RunCoordinator instance."""
        mock_sdk_factory = MagicMock()
        run_coordinator = build_run_coordinator(
            mock_runtime_deps, mock_pipeline_config, mock_sdk_factory
        )
        assert isinstance(run_coordinator, RunCoordinator)

    def test_run_coordinator_uses_config_values(
        self,
        mock_runtime_deps: RuntimeDeps,
        mock_pipeline_config: PipelineConfig,
        tmp_path: Path,
    ) -> None:
        """RunCoordinator is configured with values from deps and config."""
        mock_sdk_factory = MagicMock()
        run_coordinator = build_run_coordinator(
            mock_runtime_deps, mock_pipeline_config, mock_sdk_factory
        )
        assert run_coordinator.config.repo_path == tmp_path
        assert run_coordinator.config.timeout_seconds == 3600
        assert run_coordinator.config.max_gate_retries == 3


@pytest.mark.unit
class TestBuildIssueCoordinator:
    """Tests for build_issue_coordinator function."""

    def test_returns_issue_execution_coordinator(
        self,
        mock_issue_filter_config: IssueFilterConfig,
        mock_runtime_deps: RuntimeDeps,
    ) -> None:
        """build_issue_coordinator returns an IssueExecutionCoordinator instance."""
        coordinator = build_issue_coordinator(
            mock_issue_filter_config, mock_runtime_deps
        )
        assert isinstance(coordinator, IssueExecutionCoordinator)

    def test_coordinator_uses_filter_config(
        self,
        mock_issue_filter_config: IssueFilterConfig,
        mock_runtime_deps: RuntimeDeps,
    ) -> None:
        """IssueExecutionCoordinator is configured with IssueFilterConfig values."""
        coordinator = build_issue_coordinator(
            mock_issue_filter_config, mock_runtime_deps
        )
        assert coordinator.config.max_agents == 4
        assert coordinator.config.max_issues == 10
        assert coordinator.config.epic_id == "test-epic"
        assert coordinator.config.only_ids == ["issue-1", "issue-2"]
        assert coordinator.config.prioritize_wip is True
        assert coordinator.config.focus is True
        assert coordinator.config.orphans_only is False


@pytest.mark.unit
class TestBuildSessionConfig:
    """Tests for build_session_config function."""

    def test_returns_agent_session_config(
        self, mock_pipeline_config: PipelineConfig
    ) -> None:
        """build_session_config returns an AgentSessionConfig instance."""
        from src.pipeline.agent_session_runner import AgentSessionConfig

        session_config = build_session_config(mock_pipeline_config, review_enabled=True)
        assert isinstance(session_config, AgentSessionConfig)

    def test_session_config_uses_pipeline_values(
        self, mock_pipeline_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """AgentSessionConfig is configured with PipelineConfig values."""
        session_config = build_session_config(mock_pipeline_config, review_enabled=True)
        assert session_config.repo_path == tmp_path
        assert session_config.timeout_seconds == 3600
        assert session_config.max_gate_retries == 3
        assert session_config.max_review_retries == 3
        assert session_config.review_enabled is True
        assert session_config.context_restart_threshold == 0.90
        assert session_config.context_limit == 200_000

    def test_session_config_review_disabled(
        self, mock_pipeline_config: PipelineConfig
    ) -> None:
        """build_session_config respects review_enabled=False."""
        session_config = build_session_config(
            mock_pipeline_config, review_enabled=False
        )
        assert session_config.review_enabled is False

    def test_session_config_prompts(self, mock_pipeline_config: PipelineConfig) -> None:
        """AgentSessionConfig has correct prompts from PromptProvider."""
        session_config = build_session_config(mock_pipeline_config, review_enabled=True)
        assert session_config.prompts.gate_followup == "gate"
        assert session_config.prompts.review_followup == "review"
        assert session_config.prompts.idle_resume == "idle"
        assert session_config.prompts.checkpoint_request == "checkpoint"
        assert session_config.prompts.continuation == "continuation"
