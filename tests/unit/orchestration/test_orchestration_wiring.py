"""Unit tests for orchestration wiring dataclasses and build functions.

Tests the frozen dataclasses (RuntimeDeps, PipelineConfig, IssueFilterConfig)
and the build functions that construct pipeline components.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.domain.prompts import PromptProvider
from src.domain.validation.config import PromptValidationCommands
from src.infra.io.config import MalaConfig
from src.infra.clients.amp_provider import _create_amp_mcp_server_factory
from src.orchestration.orchestration_wiring import (
    build_gate_runner,
    build_issue_coordinator,
    build_review_runner,
    build_run_coordinator,
    build_session_callback_factory,
    build_session_config,
)
from src.orchestration.types import (
    IssueFilterConfig,
    PipelineConfig,
    RuntimeDeps,
)
from src.pipeline.session_callback_factory import SessionRunContext
from src.pipeline.gate_runner import AsyncGateRunner, GateRunner
from src.pipeline.issue_execution_coordinator import IssueExecutionCoordinator
from src.pipeline.review_runner import ReviewRunner
from src.pipeline.run_coordinator import RunCoordinator
from src.pipeline.session_callback_factory import SessionCallbackFactory
from tests.fakes import (
    FakeCommandRunner,
    FakeEventSink,
    FakeLockManager,
    FakeSDKClientFactory,
)


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
        custom_commands=(),
    )


@pytest.fixture
def mock_runtime_deps() -> RuntimeDeps:
    """Create RuntimeDeps with fake protocol implementations."""
    mock_config = MalaConfig(
        runs_dir=Path("/tmp/runs"),
        lock_dir=Path("/tmp/locks"),
        claude_config_dir=Path("/tmp/claude"),
        review_timeout=300,
    )
    from tests.fakes.agent_provider import FakeAgentProvider

    return RuntimeDeps(
        evidence_check=MagicMock(),
        code_reviewer=MagicMock(),
        beads=MagicMock(),
        event_sink=FakeEventSink(),
        agent_provider=FakeAgentProvider(FakeSDKClientFactory()),  # type: ignore[arg-type]
        command_runner=FakeCommandRunner(allow_unregistered=True),
        env_config=MagicMock(),
        lock_manager=FakeLockManager(),
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
        disabled_validations={"lint"},
        max_idle_retries=2,
        idle_timeout_seconds=None,
        prompts=mock_prompt_provider,
        prompt_validation_commands=mock_prompt_validation_commands,
        validation_config=None,
        validation_config_missing=False,
        deadlock_monitor=None,
        review_timeout_seconds=300,
    )


@pytest.fixture
def mock_issue_filter_config() -> IssueFilterConfig:
    """Create IssueFilterConfig with test values."""
    return IssueFilterConfig(
        max_agents=4,
        max_issues=10,
        epic_id="test-epic",
        only_ids=["issue-1", "issue-2"],
        include_wip=True,
        focus=True,
        orphans_only=False,
        epic_override_ids={"epic-1"},
    )


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
        run_coordinator = build_run_coordinator(mock_runtime_deps, mock_pipeline_config)
        assert isinstance(run_coordinator, RunCoordinator)

    def test_run_coordinator_uses_config_values(
        self,
        mock_runtime_deps: RuntimeDeps,
        mock_pipeline_config: PipelineConfig,
        tmp_path: Path,
    ) -> None:
        """RunCoordinator is configured with values from deps and config."""
        run_coordinator = build_run_coordinator(mock_runtime_deps, mock_pipeline_config)
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
        assert coordinator.config.include_wip is True
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


@pytest.fixture
def mock_session_run_context() -> SessionRunContext:
    """Create a SessionRunContext with lambda stubs."""
    return SessionRunContext(
        evidence_provider_getter=lambda: MagicMock(),
        evidence_check_getter=lambda: MagicMock(),
        on_session_log_path=lambda issue_id, path: None,
        on_review_log_path=lambda issue_id, path: None,
        interrupt_event_getter=lambda: None,
        get_base_sha=lambda issue_id: None,
        get_run_metadata=lambda: None,
        on_abort=lambda reason: None,
        abort_event_getter=lambda: None,
    )


@pytest.mark.unit
class TestBuildSessionCallbackFactory:
    """Tests for build_session_callback_factory function."""

    def test_returns_session_callback_factory(
        self,
        mock_runtime_deps: RuntimeDeps,
        mock_pipeline_config: PipelineConfig,
        mock_session_run_context: SessionRunContext,
    ) -> None:
        """build_session_callback_factory returns a SessionCallbackFactory instance."""
        _, async_gate_runner = build_gate_runner(
            mock_runtime_deps, mock_pipeline_config
        )
        review_runner = build_review_runner(mock_runtime_deps, mock_pipeline_config)

        factory = build_session_callback_factory(
            mock_runtime_deps,
            mock_pipeline_config,
            async_gate_runner,
            review_runner,
            mock_session_run_context,
        )

        assert isinstance(factory, SessionCallbackFactory)

    def test_factory_accepts_optional_cumulative_review_runner(
        self,
        mock_runtime_deps: RuntimeDeps,
        mock_pipeline_config: PipelineConfig,
        mock_session_run_context: SessionRunContext,
    ) -> None:
        """build_session_callback_factory accepts optional cumulative_review_runner."""
        _, async_gate_runner = build_gate_runner(
            mock_runtime_deps, mock_pipeline_config
        )
        review_runner = build_review_runner(mock_runtime_deps, mock_pipeline_config)
        mock_cumulative_runner = MagicMock()

        factory = build_session_callback_factory(
            mock_runtime_deps,
            mock_pipeline_config,
            async_gate_runner,
            review_runner,
            mock_session_run_context,
            cumulative_review_runner=mock_cumulative_runner,
        )

        assert isinstance(factory, SessionCallbackFactory)


def _amp_mcp_spec(repo_path: Path, agent_id: str = "agent-A") -> dict[str, object]:
    """Invoke the factory and return the ``mala-locking`` server spec.

    Centralises the type-narrowing dance so each test can read keys
    without re-asserting ``isinstance`` against ``dict[str, object]``.
    """
    from typing import cast

    factory = _create_amp_mcp_server_factory()
    raw = factory(agent_id, repo_path, None)["mala-locking"]
    assert isinstance(raw, dict)
    return cast("dict[str, object]", raw)


def _amp_mcp_env(spec: dict[str, object]) -> dict[str, object]:
    from typing import cast

    env_raw = spec["env"]
    assert isinstance(env_raw, dict)
    return cast("dict[str, object]", env_raw)


@pytest.mark.unit
class TestCreateAmpMcpServerFactory:
    """Tests for ``_create_amp_mcp_server_factory`` in amp_provider.

    Pins the stdio launch spec the factory emits for Amp's
    ``--mcp-config``. The launcher itself
    (``mala-amp-mcp-locking`` → :mod:`src.infra.tools.locking_mcp_stdio`)
    is exercised in :mod:`tests.unit.infra.tools.test_locking_mcp_stdio`
    and end-to-end through real Amp in
    ``tests/integration/test_lock_mcp_via_amp.py`` (AC#9 closure).
    """

    def test_emits_real_launcher_command(self, tmp_path: Path) -> None:
        """The factory must point at the registered console script, not
        a placeholder. After ``uv sync`` (or any editable install) this
        is the on-PATH ``mala-amp-mcp-locking`` shipped via
        ``[project.scripts]``."""
        spec = _amp_mcp_spec(tmp_path)
        assert spec["command"] == "mala-amp-mcp-locking"

    def test_args_carry_agent_id_and_repo_namespace(self, tmp_path: Path) -> None:
        spec = _amp_mcp_spec(tmp_path)
        assert spec["args"] == [
            "--agent-id",
            "agent-A",
            "--repo-namespace",
            str(tmp_path),
        ]

    def test_env_includes_agent_id_and_repo_namespace(self, tmp_path: Path) -> None:
        """Args + env both carry agent id + namespace so the launcher
        boots whether Amp's MCP transport forwards args, env, or both."""
        env = _amp_mcp_env(_amp_mcp_spec(tmp_path))
        assert env["MALA_AGENT_ID"] == "agent-A"
        assert env["MALA_REPO_NAMESPACE"] == str(tmp_path)

    def test_env_forwards_lock_dir_when_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the orchestrator has chosen a lock dir, the spawned
        launcher must see it — otherwise its lock files would land in a
        different directory than the rest of the run reads, breaking the
        cross-process contract with the safety plugin."""
        monkeypatch.setenv("MALA_LOCK_DIR", "/tmp/some-lock-dir")
        env = _amp_mcp_env(_amp_mcp_spec(tmp_path))
        assert env["MALA_LOCK_DIR"] == "/tmp/some-lock-dir"

    def test_env_omits_lock_dir_when_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When MALA_LOCK_DIR is unset, the launcher must fall through to
        :func:`src.infra.tools.env.get_lock_dir`'s default rather than
        receive an empty value that would shadow it."""
        monkeypatch.delenv("MALA_LOCK_DIR", raising=False)
        env = _amp_mcp_env(_amp_mcp_spec(tmp_path))
        assert "MALA_LOCK_DIR" not in env

    def test_factory_output_is_json_serializable(self, tmp_path: Path) -> None:
        """Amp's --mcp-config consumes JSON; the factory output must
        round-trip through ``json.dumps``."""
        import json as _json

        factory = _create_amp_mcp_server_factory()
        out = factory("agent-A", tmp_path, None)
        encoded = _json.dumps(out)
        assert "mala-locking" in encoded
