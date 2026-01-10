"""Unit tests for RunCoordinator fixer interrupt handling.

Tests the fixer agent interrupt behavior using mock SDK clients,
without subprocess dependencies.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from src.pipeline.run_coordinator import (
    FixerResult,
    RunCoordinator,
    RunCoordinatorConfig,
)
from tests.fakes import FakeEnvConfig
from tests.fakes.command_runner import FakeCommandRunner
from tests.fakes.lock_manager import FakeLockManager


@pytest.fixture
def fake_command_runner() -> FakeCommandRunner:
    """Create a FakeCommandRunner that allows unregistered commands."""
    return FakeCommandRunner(allow_unregistered=True)


@pytest.fixture
def mock_env_config() -> FakeEnvConfig:
    """Create a fake EnvConfigPort."""
    return FakeEnvConfig()


@pytest.fixture
def fake_lock_manager() -> FakeLockManager:
    """Create a FakeLockManager for testing."""
    return FakeLockManager()


@pytest.fixture
def mock_sdk_client_factory() -> MagicMock:
    """Create a mock SDKClientFactoryProtocol."""
    return MagicMock()


class TestFixerInterruptHandling:
    """Test fixer agent interrupt behavior in RunCoordinator."""

    @pytest.fixture
    def coordinator(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> RunCoordinator:
        """Create a RunCoordinator with test dependencies."""
        mock_gate_checker = MagicMock()
        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
        )
        return RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

    @pytest.mark.asyncio
    async def test_fixer_returns_interrupted_when_event_set_before_start(
        self,
        coordinator: RunCoordinator,
    ) -> None:
        """Fixer should return interrupted=True when event is set before starting."""
        interrupt_event = asyncio.Event()
        interrupt_event.set()  # Set before calling

        result = await coordinator._run_fixer_agent(
            failure_output="Test failure",
            attempt=1,
            interrupt_event=interrupt_event,
        )

        assert isinstance(result, FixerResult)
        assert result.interrupted is True
        assert result.success is None

    @pytest.mark.asyncio
    async def test_fixer_checks_interrupt_before_starting(
        self,
        coordinator: RunCoordinator,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Fixer should check interrupt and exit before SDK client is created."""
        interrupt_event = asyncio.Event()
        interrupt_event.set()

        result = await coordinator._run_fixer_agent(
            failure_output="Test failure",
            attempt=1,
            interrupt_event=interrupt_event,
        )

        # SDK client should NOT be created when interrupted before start
        mock_sdk_client_factory.create.assert_not_called()
        assert result.interrupted is True

    @pytest.mark.asyncio
    async def test_fixer_returns_success_when_not_interrupted(
        self,
        coordinator: RunCoordinator,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Fixer should return success=True when completing without interrupt."""
        # Create a mock client that simulates a successful run
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        # Create a mock ResultMessage
        result_message = MagicMock()
        result_message.__class__.__name__ = "ResultMessage"
        type(result_message).__name__ = "ResultMessage"
        result_message.result = "Fixed!"
        result_message.session_id = "test-session-123"

        # Make receive_response return the result message
        async def mock_receive() -> AsyncGenerator[MagicMock, None]:
            yield result_message

        mock_client.receive_response = mock_receive
        mock_sdk_client_factory.create.return_value = mock_client

        # Mock AgentRuntimeBuilder to avoid MCP server factory dependency
        mock_runtime = MagicMock()
        mock_runtime.options = {}
        mock_runtime.lint_cache = MagicMock()

        with patch(
            "src.pipeline.run_coordinator.AgentRuntimeBuilder"
        ) as mock_builder_class:
            mock_builder = MagicMock()
            mock_builder_class.return_value = mock_builder
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = mock_runtime

            result = await coordinator._run_fixer_agent(
                failure_output="Test failure",
                attempt=1,
                interrupt_event=None,  # No interrupt event
            )

        assert isinstance(result, FixerResult)
        assert result.success is True
        assert result.interrupted is False

    @pytest.mark.asyncio
    async def test_fixer_captures_log_path(
        self,
        coordinator: RunCoordinator,
        mock_sdk_client_factory: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Fixer should capture log path using upfront session_id."""
        # Create a mock client
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        result_message = MagicMock()
        result_message.__class__.__name__ = "ResultMessage"
        type(result_message).__name__ = "ResultMessage"
        result_message.result = "Fixed!"

        async def mock_receive() -> AsyncGenerator[MagicMock, None]:
            yield result_message

        mock_client.receive_response = mock_receive
        mock_sdk_client_factory.create.return_value = mock_client

        # Mock AgentRuntimeBuilder to avoid MCP server factory dependency
        mock_runtime = MagicMock()
        mock_runtime.options = {}
        mock_runtime.lint_cache = MagicMock()

        # Create a mock UUID for predictable agent_id (fixer-{uuid.hex[:8]})
        mock_uuid = MagicMock()
        mock_uuid.hex = "abcd1234efgh5678"

        with (
            patch(
                "src.pipeline.run_coordinator.AgentRuntimeBuilder"
            ) as mock_builder_class,
            patch("src.infra.tools.env.get_claude_log_path") as mock_log_path,
            patch("src.pipeline.run_coordinator.uuid.uuid4", return_value=mock_uuid),
        ):
            mock_builder = MagicMock()
            mock_builder_class.return_value = mock_builder
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = mock_runtime

            mock_log_path.return_value = Path("/mock/log/path/session.jsonl")

            result = await coordinator._run_fixer_agent(
                failure_output="Test failure",
                attempt=1,
                interrupt_event=None,
            )

        assert result.success is True
        assert result.log_path == "/mock/log/path/session.jsonl"
        # agent_id is used for log path (fixer-{uuid.hex[:8]})
        mock_log_path.assert_called_once_with(tmp_path, "fixer-abcd1234")
        # Verify agent_id was passed to client.query as session_id
        mock_client.query.assert_called_once()
        call_kwargs = mock_client.query.call_args
        assert call_kwargs[1].get("session_id") == "fixer-abcd1234"

    @pytest.mark.asyncio
    async def test_fixer_returns_interrupted_during_message_loop(
        self,
        coordinator: RunCoordinator,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Fixer should check interrupt between messages and exit early."""
        interrupt_event = asyncio.Event()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        # First message is normal, but we set interrupt before second
        first_message = MagicMock()
        first_message.__class__.__name__ = "AssistantMessage"
        type(first_message).__name__ = "AssistantMessage"
        first_message.content = []

        async def mock_receive() -> AsyncGenerator[MagicMock, None]:
            yield first_message
            # Set interrupt after first message
            interrupt_event.set()
            # This message should not be fully processed
            yield MagicMock()

        mock_client.receive_response = mock_receive
        mock_sdk_client_factory.create.return_value = mock_client

        # Mock AgentRuntimeBuilder to avoid MCP server factory dependency
        mock_runtime = MagicMock()
        mock_runtime.options = {}
        mock_runtime.lint_cache = MagicMock()

        with patch(
            "src.pipeline.run_coordinator.AgentRuntimeBuilder"
        ) as mock_builder_class:
            mock_builder = MagicMock()
            mock_builder_class.return_value = mock_builder
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = mock_runtime

            result = await coordinator._run_fixer_agent(
                failure_output="Test failure",
                attempt=1,
                interrupt_event=interrupt_event,
            )

        assert result.interrupted is True
        assert result.success is None

    @pytest.mark.asyncio
    async def test_fixer_captures_log_path_on_interrupt_during_loop(
        self,
        coordinator: RunCoordinator,
        mock_sdk_client_factory: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Fixer should capture log path even when interrupted during message loop (Finding 2 fix)."""
        interrupt_event = asyncio.Event()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.query = AsyncMock()

        # Receive first message, then set interrupt before second
        assistant_message = MagicMock()
        assistant_message.__class__.__name__ = "AssistantMessage"
        type(assistant_message).__name__ = "AssistantMessage"
        assistant_message.content = []

        async def mock_receive() -> AsyncGenerator[MagicMock, None]:
            yield assistant_message
            # Set interrupt after first message
            interrupt_event.set()
            yield MagicMock()

        mock_client.receive_response = mock_receive
        mock_sdk_client_factory.create.return_value = mock_client

        mock_runtime = MagicMock()
        mock_runtime.options = {}
        mock_runtime.lint_cache = MagicMock()

        # Create a mock UUID for predictable agent_id (fixer-{uuid.hex[:8]})
        mock_uuid = MagicMock()
        mock_uuid.hex = "deadbeef12345678"

        with (
            patch(
                "src.pipeline.run_coordinator.AgentRuntimeBuilder"
            ) as mock_builder_class,
            patch("src.infra.tools.env.get_claude_log_path") as mock_log_path,
            patch("src.pipeline.run_coordinator.uuid.uuid4", return_value=mock_uuid),
        ):
            mock_builder = MagicMock()
            mock_builder_class.return_value = mock_builder
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = mock_runtime

            mock_log_path.return_value = Path("/mock/log/path/interrupted.jsonl")

            result = await coordinator._run_fixer_agent(
                failure_output="Test failure",
                attempt=1,
                interrupt_event=interrupt_event,
            )

        assert result.interrupted is True
        assert result.success is None
        # Key assertion: log_path should be captured even on interrupt during loop
        assert result.log_path == "/mock/log/path/interrupted.jsonl"
        # agent_id is used for log path (fixer-{uuid.hex[:8]})
        mock_log_path.assert_called_once_with(tmp_path, "fixer-deadbeef")


class TestGetTriggerConfig:
    """Test _get_trigger_config method."""

    @pytest.fixture
    def coordinator(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> RunCoordinator:
        """Create a RunCoordinator with test dependencies."""
        mock_gate_checker = MagicMock()
        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
        )
        return RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

    def test_get_trigger_config_run_end(
        self,
        coordinator: RunCoordinator,
    ) -> None:
        """_get_trigger_config returns run_end config for TriggerType.RUN_END."""
        from src.domain.validation.config import (
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerCommandRef,
            TriggerType,
            ValidationTriggersConfig,
        )

        run_end_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(TriggerCommandRef(ref="test"),),
            fire_on=FireOn.SUCCESS,
        )
        triggers_config = ValidationTriggersConfig(run_end=run_end_config)

        result = coordinator._get_trigger_config(triggers_config, TriggerType.RUN_END)

        assert result is not None
        assert result == run_end_config

    def test_get_trigger_config_epic_completion(
        self,
        coordinator: RunCoordinator,
    ) -> None:
        """_get_trigger_config returns epic_completion config."""
        from src.domain.validation.config import (
            EpicCompletionTriggerConfig,
            EpicDepth,
            FailureMode,
            FireOn,
            TriggerCommandRef,
            TriggerType,
            ValidationTriggersConfig,
        )

        epic_config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(TriggerCommandRef(ref="test"),),
            epic_depth=EpicDepth.TOP_LEVEL,
            fire_on=FireOn.SUCCESS,
        )
        triggers_config = ValidationTriggersConfig(epic_completion=epic_config)

        result = coordinator._get_trigger_config(
            triggers_config, TriggerType.EPIC_COMPLETION
        )

        assert result is not None
        assert result == epic_config

    def test_get_trigger_config_session_end(
        self,
        coordinator: RunCoordinator,
    ) -> None:
        """_get_trigger_config returns session_end config."""
        from src.domain.validation.config import (
            FailureMode,
            SessionEndTriggerConfig,
            TriggerCommandRef,
            TriggerType,
            ValidationTriggersConfig,
        )

        session_config = SessionEndTriggerConfig(
            failure_mode=FailureMode.REMEDIATE,
            commands=(TriggerCommandRef(ref="lint"),),
        )
        triggers_config = ValidationTriggersConfig(session_end=session_config)

        result = coordinator._get_trigger_config(
            triggers_config, TriggerType.SESSION_END
        )

        assert result is not None
        assert result == session_config

    def test_get_trigger_config_returns_none_for_unconfigured(
        self,
        coordinator: RunCoordinator,
    ) -> None:
        """_get_trigger_config returns None when trigger type not configured."""
        from src.domain.validation.config import (
            EpicCompletionTriggerConfig,
            EpicDepth,
            FailureMode,
            FireOn,
            TriggerCommandRef,
            TriggerType,
            ValidationTriggersConfig,
        )

        # Create a triggers config with only epic_completion
        epic_config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(TriggerCommandRef(ref="test"),),
            epic_depth=EpicDepth.TOP_LEVEL,
            fire_on=FireOn.SUCCESS,
        )
        triggers_config = ValidationTriggersConfig(epic_completion=epic_config)

        # RUN_END is not configured
        result = coordinator._get_trigger_config(triggers_config, TriggerType.RUN_END)

        assert result is None


class TestRunTriggerCodeReview:
    """Test _run_trigger_code_review method."""

    @pytest.fixture
    def coordinator_with_review_runner(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> tuple[RunCoordinator, MagicMock]:
        """Create a RunCoordinator with a mock CumulativeReviewRunner."""
        mock_gate_checker = MagicMock()
        mock_review_runner = MagicMock()
        mock_run_metadata = MagicMock()

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            cumulative_review_runner=mock_review_runner,
            run_metadata=mock_run_metadata,
        )
        return coordinator, mock_review_runner

    @pytest.mark.asyncio
    async def test_returns_none_when_code_review_disabled(
        self,
        coordinator_with_review_runner: tuple[RunCoordinator, MagicMock],
    ) -> None:
        """_run_trigger_code_review returns None when code_review is disabled."""
        from src.domain.validation.config import (
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerCommandRef,
            TriggerType,
        )

        coordinator, mock_review_runner = coordinator_with_review_runner

        # Create a trigger config without code_review
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(TriggerCommandRef(ref="test"),),
            fire_on=FireOn.SUCCESS,
        )

        interrupt_event = asyncio.Event()
        result = await coordinator._run_trigger_code_review(
            TriggerType.RUN_END,
            trigger_config,
            {},
            interrupt_event,
        )

        assert result is None
        mock_review_runner.run_review.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_review_result_when_enabled(
        self,
        coordinator_with_review_runner: tuple[RunCoordinator, MagicMock],
    ) -> None:
        """_run_trigger_code_review returns result from CumulativeReviewRunner."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerCommandRef,
            TriggerType,
        )
        from src.pipeline.cumulative_review_runner import CumulativeReviewResult

        coordinator, mock_review_runner = coordinator_with_review_runner

        # Create a trigger config with code_review enabled
        code_review_config = CodeReviewConfig(
            enabled=True,
            failure_mode=FailureMode.CONTINUE,
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(TriggerCommandRef(ref="test"),),
            fire_on=FireOn.SUCCESS,
            code_review=code_review_config,
        )

        # Mock the review result
        expected_result = CumulativeReviewResult(
            status="success",
            findings=(),
            new_baseline_commit="abc123",
        )
        mock_review_runner.run_review = AsyncMock(return_value=expected_result)

        interrupt_event = asyncio.Event()
        result = await coordinator._run_trigger_code_review(
            TriggerType.RUN_END,
            trigger_config,
            {"issue_id": "test-issue"},
            interrupt_event,
        )

        assert result == expected_result
        mock_review_runner.run_review.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_runner_not_wired(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """_run_trigger_code_review returns None and logs warning when runner not wired."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerCommandRef,
            TriggerType,
        )

        mock_gate_checker = MagicMock()
        mock_event_sink = MagicMock()

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            event_sink=mock_event_sink,
            # No cumulative_review_runner wired
        )

        code_review_config = CodeReviewConfig(enabled=True)
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(TriggerCommandRef(ref="test"),),
            fire_on=FireOn.SUCCESS,
            code_review=code_review_config,
        )

        interrupt_event = asyncio.Event()
        result = await coordinator._run_trigger_code_review(
            TriggerType.RUN_END,
            trigger_config,
            {},
            interrupt_event,
        )

        assert result is None
        mock_event_sink.on_warning.assert_called_once()
        assert (
            "CumulativeReviewRunner not wired"
            in mock_event_sink.on_warning.call_args[0][0]
        )

    @pytest.mark.asyncio
    async def test_skips_code_review_in_fixer_session(
        self,
        coordinator_with_review_runner: tuple[RunCoordinator, MagicMock],
    ) -> None:
        """_run_trigger_code_review skips review when is_fixer_session is True."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerCommandRef,
            TriggerType,
        )

        coordinator, mock_review_runner = coordinator_with_review_runner

        code_review_config = CodeReviewConfig(
            enabled=True,
            failure_mode=FailureMode.CONTINUE,
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(TriggerCommandRef(ref="test"),),
            fire_on=FireOn.SUCCESS,
            code_review=code_review_config,
        )

        interrupt_event = asyncio.Event()
        # Pass is_fixer_session=True in context
        result = await coordinator._run_trigger_code_review(
            TriggerType.RUN_END,
            trigger_config,
            {"is_fixer_session": True},
            interrupt_event,
        )

        assert result is None
        mock_review_runner.run_review.assert_not_called()


class TestFindingsExceedThreshold:
    """Tests for _findings_exceed_threshold method."""

    @pytest.fixture
    def coordinator(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> RunCoordinator:
        """Create a minimal RunCoordinator for testing."""
        mock_gate_checker = MagicMock()
        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix: {failure_output}",
        )
        return RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

    def test_threshold_none_never_exceeds(self, coordinator: RunCoordinator) -> None:
        """Threshold 'none' never considers findings as exceeding."""
        from src.pipeline.cumulative_review_runner import ReviewFinding

        findings = (
            ReviewFinding(
                file="test.py",
                line_start=1,
                line_end=5,
                priority=0,  # P0
                title="Critical issue",
                body="Details",
                reviewer="test",
            ),
        )
        assert coordinator._findings_exceed_threshold(findings, "none") is False

    def test_threshold_p0_exceeds_with_p0_finding(
        self, coordinator: RunCoordinator
    ) -> None:
        """Threshold 'P0' exceeds when P0 finding exists."""
        from src.pipeline.cumulative_review_runner import ReviewFinding

        findings = (
            ReviewFinding(
                file="test.py",
                line_start=1,
                line_end=5,
                priority=0,  # P0
                title="Critical issue",
                body="Details",
                reviewer="test",
            ),
        )
        assert coordinator._findings_exceed_threshold(findings, "P0") is True

    def test_threshold_p0_not_exceeded_with_p1_finding(
        self, coordinator: RunCoordinator
    ) -> None:
        """Threshold 'P0' not exceeded when only P1+ findings exist."""
        from src.pipeline.cumulative_review_runner import ReviewFinding

        findings = (
            ReviewFinding(
                file="test.py",
                line_start=1,
                line_end=5,
                priority=1,  # P1
                title="High issue",
                body="Details",
                reviewer="test",
            ),
        )
        assert coordinator._findings_exceed_threshold(findings, "P0") is False

    def test_threshold_p1_exceeds_with_p0_or_p1(
        self, coordinator: RunCoordinator
    ) -> None:
        """Threshold 'P1' exceeds when P0 or P1 findings exist."""
        from src.pipeline.cumulative_review_runner import ReviewFinding

        # P0 finding
        findings_p0 = (
            ReviewFinding(
                file="test.py",
                line_start=1,
                line_end=5,
                priority=0,
                title="Critical",
                body="Details",
                reviewer="test",
            ),
        )
        assert coordinator._findings_exceed_threshold(findings_p0, "P1") is True

        # P1 finding
        findings_p1 = (
            ReviewFinding(
                file="test.py",
                line_start=1,
                line_end=5,
                priority=1,
                title="High",
                body="Details",
                reviewer="test",
            ),
        )
        assert coordinator._findings_exceed_threshold(findings_p1, "P1") is True

    def test_threshold_p1_not_exceeded_with_p2_p3(
        self, coordinator: RunCoordinator
    ) -> None:
        """Threshold 'P1' not exceeded when only P2/P3 findings exist."""
        from src.pipeline.cumulative_review_runner import ReviewFinding

        findings = (
            ReviewFinding(
                file="test.py",
                line_start=1,
                line_end=5,
                priority=2,  # P2
                title="Medium issue",
                body="Details",
                reviewer="test",
            ),
            ReviewFinding(
                file="test2.py",
                line_start=10,
                line_end=15,
                priority=3,  # P3
                title="Low issue",
                body="Details",
                reviewer="test",
            ),
        )
        assert coordinator._findings_exceed_threshold(findings, "P1") is False

    def test_empty_findings_never_exceeds(self, coordinator: RunCoordinator) -> None:
        """Empty findings tuple never exceeds any threshold."""
        assert coordinator._findings_exceed_threshold((), "P0") is False
        assert coordinator._findings_exceed_threshold((), "P1") is False
        assert coordinator._findings_exceed_threshold((), "P3") is False


class TestCodeReviewRemediateFailureMode:
    """Tests for failure_mode: remediate handling in code review."""

    @pytest.mark.asyncio
    async def test_remediate_retries_on_execution_error(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """failure_mode: remediate retries review on execution error."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.pipeline.cumulative_review_runner import CumulativeReviewResult

        mock_gate_checker = MagicMock()
        mock_review_runner = MagicMock()
        mock_run_metadata = MagicMock()
        mock_event_sink = MagicMock()

        # Configure code_review with failure_mode=REMEDIATE
        code_review_config = CodeReviewConfig(
            enabled=True,
            failure_mode=FailureMode.REMEDIATE,
            max_retries=2,
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),  # No commands, just code_review
            fire_on=FireOn.SUCCESS,
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            cumulative_review_runner=mock_review_runner,
            run_metadata=mock_run_metadata,
            event_sink=mock_event_sink,
        )

        # First call fails, second succeeds
        mock_review_runner.run_review = AsyncMock(
            side_effect=[
                CumulativeReviewResult(
                    status="failed",
                    findings=(),
                    new_baseline_commit=None,
                    skip_reason="execution_error: timeout",
                ),
                CumulativeReviewResult(
                    status="success",
                    findings=(),
                    new_baseline_commit="abc123",
                ),
            ]
        )

        # Queue the trigger
        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

        result = await coordinator.run_trigger_validation(dry_run=False)

        assert result.status == "passed"
        assert mock_review_runner.run_review.call_count == 2
        mock_event_sink.on_trigger_remediation_started.assert_called_once_with(
            "run_end", 1, 2
        )
        mock_event_sink.on_trigger_remediation_succeeded.assert_called_once_with(
            "run_end", 1
        )

    @pytest.mark.asyncio
    async def test_remediate_exhausted_continues_with_failure(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """failure_mode: remediate exhausted continues and records failure."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.pipeline.cumulative_review_runner import CumulativeReviewResult

        mock_gate_checker = MagicMock()
        mock_review_runner = MagicMock()
        mock_run_metadata = MagicMock()
        mock_event_sink = MagicMock()

        code_review_config = CodeReviewConfig(
            enabled=True,
            failure_mode=FailureMode.REMEDIATE,
            max_retries=2,
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
            fire_on=FireOn.SUCCESS,
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            cumulative_review_runner=mock_review_runner,
            run_metadata=mock_run_metadata,
            event_sink=mock_event_sink,
        )

        # All retries fail
        mock_review_runner.run_review = AsyncMock(
            return_value=CumulativeReviewResult(
                status="failed",
                findings=(),
                new_baseline_commit=None,
                skip_reason="execution_error: timeout",
            )
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

        result = await coordinator.run_trigger_validation(dry_run=False)

        # Should be "failed" not "aborted" per plan behavior matrix
        assert result.status == "failed"
        # Initial call + 2 retries = 3 total
        assert mock_review_runner.run_review.call_count == 3
        mock_event_sink.on_trigger_remediation_exhausted.assert_called_once_with(
            "run_end", 2
        )
        # Must NOT emit validation_passed after emitting validation_failed
        mock_event_sink.on_trigger_validation_passed.assert_not_called()


class TestFindingThresholdEnforcement:
    """Tests for finding_threshold enforcement in code review."""

    @pytest.mark.asyncio
    async def test_findings_below_threshold_pass(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Findings below threshold allow validation to pass."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.pipeline.cumulative_review_runner import (
            CumulativeReviewResult,
            ReviewFinding,
        )

        mock_gate_checker = MagicMock()
        mock_review_runner = MagicMock()
        mock_run_metadata = MagicMock()
        mock_event_sink = MagicMock()

        # P1 threshold, but only P2 findings
        code_review_config = CodeReviewConfig(
            enabled=True,
            finding_threshold="P1",
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
            fire_on=FireOn.SUCCESS,
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            cumulative_review_runner=mock_review_runner,
            run_metadata=mock_run_metadata,
            event_sink=mock_event_sink,
        )

        # Review completes with P2 findings (below P1 threshold)
        mock_review_runner.run_review = AsyncMock(
            return_value=CumulativeReviewResult(
                status="success",
                findings=(
                    ReviewFinding(
                        file="test.py",
                        line_start=1,
                        line_end=5,
                        priority=2,  # P2 - below threshold
                        title="Medium issue",
                        body="Details",
                        reviewer="test",
                    ),
                ),
                new_baseline_commit="abc123",
            )
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

        result = await coordinator.run_trigger_validation(dry_run=False)

        assert result.status == "passed"
        mock_event_sink.on_trigger_validation_passed.assert_called()

    @pytest.mark.asyncio
    async def test_findings_exceed_threshold_aborts_without_retries(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Findings exceeding threshold abort when max_retries=0."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.pipeline.cumulative_review_runner import (
            CumulativeReviewResult,
            ReviewFinding,
        )

        mock_gate_checker = MagicMock()
        mock_review_runner = MagicMock()
        mock_run_metadata = MagicMock()
        mock_event_sink = MagicMock()

        code_review_config = CodeReviewConfig(
            enabled=True,
            finding_threshold="P1",
            failure_mode=FailureMode.ABORT,  # Abort on findings exceeding threshold
            max_retries=0,  # No remediation attempts
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
            fire_on=FireOn.SUCCESS,
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            cumulative_review_runner=mock_review_runner,
            run_metadata=mock_run_metadata,
            event_sink=mock_event_sink,
        )

        # P0 finding exceeds P1 threshold
        mock_review_runner.run_review = AsyncMock(
            return_value=CumulativeReviewResult(
                status="success",
                findings=(
                    ReviewFinding(
                        file="test.py",
                        line_start=1,
                        line_end=5,
                        priority=0,  # P0 - exceeds threshold
                        title="Critical issue",
                        body="Details",
                        reviewer="test",
                    ),
                ),
                new_baseline_commit="abc123",
            )
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

        result = await coordinator.run_trigger_validation(dry_run=False)

        assert result.status == "aborted"
        assert result.details is not None
        assert "findings exceed threshold" in result.details.lower()
        mock_event_sink.on_trigger_validation_failed.assert_called_with(
            "run_end", "code_review_findings", "abort"
        )

    @pytest.mark.asyncio
    async def test_threshold_none_never_fails_on_findings(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """finding_threshold='none' never fails on findings."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.pipeline.cumulative_review_runner import (
            CumulativeReviewResult,
            ReviewFinding,
        )

        mock_gate_checker = MagicMock()
        mock_review_runner = MagicMock()
        mock_run_metadata = MagicMock()
        mock_event_sink = MagicMock()

        code_review_config = CodeReviewConfig(
            enabled=True,
            finding_threshold="none",  # Never fail on findings
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
            fire_on=FireOn.SUCCESS,
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            cumulative_review_runner=mock_review_runner,
            run_metadata=mock_run_metadata,
            event_sink=mock_event_sink,
        )

        # Even P0 findings should pass
        mock_review_runner.run_review = AsyncMock(
            return_value=CumulativeReviewResult(
                status="success",
                findings=(
                    ReviewFinding(
                        file="test.py",
                        line_start=1,
                        line_end=5,
                        priority=0,  # P0 - critical
                        title="Critical issue",
                        body="Details",
                        reviewer="test",
                    ),
                ),
                new_baseline_commit="abc123",
            )
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

        result = await coordinator.run_trigger_validation(dry_run=False)

        assert result.status == "passed"

    @pytest.mark.asyncio
    async def test_findings_exceed_threshold_with_continue_records_failure(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """failure_mode=CONTINUE with findings exceeding threshold records failure."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.pipeline.cumulative_review_runner import (
            CumulativeReviewResult,
            ReviewFinding,
        )

        mock_gate_checker = MagicMock()
        mock_review_runner = MagicMock()
        mock_run_metadata = MagicMock()
        mock_event_sink = MagicMock()

        code_review_config = CodeReviewConfig(
            enabled=True,
            finding_threshold="P1",
            failure_mode=FailureMode.CONTINUE,  # Continue on findings exceeding threshold
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
            fire_on=FireOn.SUCCESS,
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            cumulative_review_runner=mock_review_runner,
            run_metadata=mock_run_metadata,
            event_sink=mock_event_sink,
        )

        # P0 finding exceeds P1 threshold
        mock_review_runner.run_review = AsyncMock(
            return_value=CumulativeReviewResult(
                status="success",
                findings=(
                    ReviewFinding(
                        file="test.py",
                        line_start=1,
                        line_end=5,
                        priority=0,  # P0 - exceeds threshold
                        title="Critical issue",
                        body="Details",
                        reviewer="test",
                    ),
                ),
                new_baseline_commit="abc123",
            )
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

        result = await coordinator.run_trigger_validation(dry_run=False)

        # Should be "failed" not "aborted" - recorded failure and continued
        assert result.status == "failed"
        assert result.details is not None
        assert "code_review_findings" in result.details
        mock_event_sink.on_trigger_validation_failed.assert_called_with(
            "run_end", "code_review_findings", "continue"
        )
        # Must NOT emit validation_passed after emitting validation_failed
        mock_event_sink.on_trigger_validation_passed.assert_not_called()

    @pytest.mark.asyncio
    async def test_findings_remediation_succeeds_on_fixed_findings(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """failure_mode=REMEDIATE with fixer fixing findings passes validation."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.pipeline.cumulative_review_runner import (
            CumulativeReviewResult,
            ReviewFinding,
        )
        from src.pipeline.run_coordinator import FixerResult

        mock_gate_checker = MagicMock()
        mock_review_runner = MagicMock()
        mock_run_metadata = MagicMock()
        mock_event_sink = MagicMock()

        code_review_config = CodeReviewConfig(
            enabled=True,
            finding_threshold="P1",
            failure_mode=FailureMode.REMEDIATE,
            max_retries=2,
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
            fire_on=FireOn.SUCCESS,
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
            validation_config=validation_config,
        )

        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            cumulative_review_runner=mock_review_runner,
            run_metadata=mock_run_metadata,
            event_sink=mock_event_sink,
        )

        # First review: P0 finding exceeds threshold
        # Second review (after fixer): No findings
        mock_review_runner.run_review = AsyncMock(
            side_effect=[
                CumulativeReviewResult(
                    status="success",
                    findings=(
                        ReviewFinding(
                            file="test.py",
                            line_start=1,
                            line_end=5,
                            priority=0,
                            title="Critical issue",
                            body="Details",
                            reviewer="test",
                        ),
                    ),
                    new_baseline_commit="abc123",
                ),
                CumulativeReviewResult(
                    status="success",
                    findings=(),  # Fixer resolved the finding
                    new_baseline_commit="def456",
                ),
            ]
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

        # Mock _run_fixer_agent to avoid MCP setup
        with patch.object(
            coordinator,
            "_run_fixer_agent",
            new=AsyncMock(return_value=FixerResult(success=True, interrupted=False)),
        ):
            result = await coordinator.run_trigger_validation(dry_run=False)

        assert result.status == "passed"
        assert mock_review_runner.run_review.call_count == 2
        mock_event_sink.on_trigger_remediation_succeeded.assert_called()

    @pytest.mark.asyncio
    async def test_findings_remediation_exhausted_records_failure(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """failure_mode=REMEDIATE with fixer failing records failure and continues."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.pipeline.cumulative_review_runner import (
            CumulativeReviewResult,
            ReviewFinding,
        )
        from src.pipeline.run_coordinator import FixerResult

        mock_gate_checker = MagicMock()
        mock_review_runner = MagicMock()
        mock_run_metadata = MagicMock()
        mock_event_sink = MagicMock()

        code_review_config = CodeReviewConfig(
            enabled=True,
            finding_threshold="P1",
            failure_mode=FailureMode.REMEDIATE,
            max_retries=1,
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
            fire_on=FireOn.SUCCESS,
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
            validation_config=validation_config,
        )

        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            cumulative_review_runner=mock_review_runner,
            run_metadata=mock_run_metadata,
            event_sink=mock_event_sink,
        )

        # All reviews return P0 finding - fixer cannot fix it
        mock_review_runner.run_review = AsyncMock(
            return_value=CumulativeReviewResult(
                status="success",
                findings=(
                    ReviewFinding(
                        file="test.py",
                        line_start=1,
                        line_end=5,
                        priority=0,
                        title="Critical issue",
                        body="Details",
                        reviewer="test",
                    ),
                ),
                new_baseline_commit="abc123",
            )
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

        # Mock _run_fixer_agent to avoid MCP setup
        with patch.object(
            coordinator,
            "_run_fixer_agent",
            new=AsyncMock(return_value=FixerResult(success=True, interrupted=False)),
        ):
            result = await coordinator.run_trigger_validation(dry_run=False)

        # Should be "failed" not "aborted" - consistent with execution error remediation
        assert result.status == "failed"
        assert result.details is not None
        assert "code_review_findings" in result.details
        mock_event_sink.on_trigger_remediation_exhausted.assert_called()
        mock_event_sink.on_trigger_validation_failed.assert_called_with(
            "run_end", "code_review_findings", "remediate"
        )
        # Must NOT emit validation_passed after emitting validation_failed
        mock_event_sink.on_trigger_validation_passed.assert_not_called()


class TestR12CodeReviewGating:
    """Tests for R12: run_end failure_mode gating for code review.

    Requirement: When run_end commands fail, whether run-end code review runs
    MUST follow failure_mode: abort skips review; continue and remediate proceed.
    """

    @pytest.mark.asyncio
    async def test_command_failure_abort_skips_code_review(
        self,
        tmp_path: Path,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """failure_mode=abort skips code_review when command fails."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            CommandConfig,
            CommandsConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerCommandRef,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.infra.tools.command_runner import CommandResult

        mock_gate_checker = MagicMock()
        mock_review_runner = MagicMock()
        mock_run_metadata = MagicMock()
        mock_event_sink = MagicMock()

        # Configure a command that will fail
        failing_runner = FakeCommandRunner()
        failing_runner.responses[("lint_cmd",)] = CommandResult(
            command="lint_cmd", returncode=1, stdout="", stderr="Lint failed"
        )

        code_review_config = CodeReviewConfig(enabled=True, finding_threshold="P1")
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,  # ABORT should skip code_review
            commands=(TriggerCommandRef(ref="lint"),),
            fire_on=FireOn.SUCCESS,
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        commands_config = CommandsConfig(lint=CommandConfig(command="lint_cmd"))
        validation_config = ValidationConfig(
            commands=commands_config,
            global_validation_commands=commands_config,
            validation_triggers=triggers_config,
        )

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=failing_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            cumulative_review_runner=mock_review_runner,
            run_metadata=mock_run_metadata,
            event_sink=mock_event_sink,
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})
        result = await coordinator.run_trigger_validation(dry_run=False)

        # Should abort
        assert result.status == "aborted"
        assert "lint" in (result.details or "").lower()
        # Code review should NOT have been called
        mock_review_runner.run_review.assert_not_called()

    @pytest.mark.asyncio
    async def test_command_failure_continue_runs_code_review(
        self,
        tmp_path: Path,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """failure_mode=continue runs code_review even when command fails."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            CommandConfig,
            CommandsConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerCommandRef,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.infra.tools.command_runner import CommandResult
        from src.pipeline.cumulative_review_runner import CumulativeReviewResult

        mock_gate_checker = MagicMock()
        mock_review_runner = MagicMock()
        mock_run_metadata = MagicMock()
        mock_event_sink = MagicMock()

        # Configure a command that will fail
        failing_runner = FakeCommandRunner()
        failing_runner.responses[("lint_cmd",)] = CommandResult(
            command="lint_cmd", returncode=1, stdout="", stderr="Lint failed"
        )

        code_review_config = CodeReviewConfig(enabled=True, finding_threshold="P1")
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,  # CONTINUE should run code_review
            commands=(TriggerCommandRef(ref="lint"),),
            fire_on=FireOn.SUCCESS,
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        commands_config = CommandsConfig(lint=CommandConfig(command="lint_cmd"))
        validation_config = ValidationConfig(
            commands=commands_config,
            global_validation_commands=commands_config,
            validation_triggers=triggers_config,
        )

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=failing_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            cumulative_review_runner=mock_review_runner,
            run_metadata=mock_run_metadata,
            event_sink=mock_event_sink,
        )

        # Mock code_review to pass
        mock_review_runner.run_review = AsyncMock(
            return_value=CumulativeReviewResult(
                status="success", findings=(), new_baseline_commit="abc123"
            )
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})
        result = await coordinator.run_trigger_validation(dry_run=False)

        # Should fail (due to command failure) but code_review should have run
        assert result.status == "failed"
        assert "lint" in (result.details or "").lower()
        # Code review SHOULD have been called
        mock_review_runner.run_review.assert_called_once()

    @pytest.mark.asyncio
    async def test_command_failure_remediate_success_runs_code_review(
        self,
        tmp_path: Path,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """failure_mode=remediate runs code_review after successful remediation."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            CommandConfig,
            CommandsConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerCommandRef,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.infra.tools.command_runner import CommandResult
        from src.pipeline.cumulative_review_runner import CumulativeReviewResult

        mock_gate_checker = MagicMock()
        mock_review_runner = MagicMock()
        mock_run_metadata = MagicMock()
        mock_event_sink = MagicMock()

        # Use a MagicMock for command runner to control call sequence
        smart_runner = MagicMock()
        call_count = {"lint_cmd": 0}

        async def mock_run_async(
            cmd: str | list[str], **kwargs: object
        ) -> CommandResult:
            cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
            if "lint_cmd" in cmd_str:
                call_count["lint_cmd"] += 1
                if call_count["lint_cmd"] == 1:
                    return CommandResult(
                        command=cmd_str,
                        returncode=1,
                        stdout="",
                        stderr="Lint failed",
                    )
                return CommandResult(
                    command=cmd_str, returncode=0, stdout="Lint passed", stderr=""
                )
            return CommandResult(command=cmd_str, returncode=0, stdout="", stderr="")

        smart_runner.run_async = mock_run_async

        code_review_config = CodeReviewConfig(enabled=True, finding_threshold="P1")
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.REMEDIATE,
            max_retries=1,  # Allow one fixer attempt
            commands=(TriggerCommandRef(ref="lint"),),
            fire_on=FireOn.SUCCESS,
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        commands_config = CommandsConfig(lint=CommandConfig(command="lint_cmd"))
        validation_config = ValidationConfig(
            commands=commands_config,
            global_validation_commands=commands_config,
            validation_triggers=triggers_config,
        )

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=smart_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            cumulative_review_runner=mock_review_runner,
            run_metadata=mock_run_metadata,
            event_sink=mock_event_sink,
        )

        # Mock code_review to pass
        mock_review_runner.run_review = AsyncMock(
            return_value=CumulativeReviewResult(
                status="success", findings=(), new_baseline_commit="abc123"
            )
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

        # Mock fixer to succeed
        with patch.object(
            coordinator,
            "_run_fixer_agent",
            new=AsyncMock(return_value=FixerResult(success=True, interrupted=False)),
        ):
            result = await coordinator.run_trigger_validation(dry_run=False)

        # Remediation succeeded, validation should pass, code_review should have run
        assert result.status == "passed"
        mock_review_runner.run_review.assert_called_once()
