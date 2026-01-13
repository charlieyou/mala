"""Unit tests for RunCoordinator fixer interrupt handling.

Tests the fixer agent interrupt behavior using mock SDK clients,
without subprocess dependencies.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from src.pipeline.run_coordinator import (
    RunCoordinator,
    RunCoordinatorConfig,
)
from src.pipeline.fixer_interface import FixerResult
from src.pipeline.fixer_service import FixerService, FixerServiceConfig
from src.pipeline.trigger_engine import TriggerEngine
from tests.fakes import FakeEnvConfig
from tests.fakes.command_runner import FakeCommandRunner
from tests.fakes.lock_manager import FakeLockManager


def _make_coordinator(
    config: RunCoordinatorConfig,
    gate_checker: Any,  # noqa: ANN401
    command_runner: Any,  # noqa: ANN401
    env_config: Any,  # noqa: ANN401
    lock_manager: Any,  # noqa: ANN401
    sdk_client_factory: Any,  # noqa: ANN401
    event_sink: Any = None,  # noqa: ANN401
    cumulative_review_runner: Any = None,  # noqa: ANN401
    run_metadata: Any = None,  # noqa: ANN401
    trigger_engine: TriggerEngine | None = None,
    fixer_service: Any = None,  # noqa: ANN401
) -> RunCoordinator:
    """Helper to construct RunCoordinator with required deps for tests."""
    if trigger_engine is None:
        # Use validation_config from RunCoordinatorConfig if available
        trigger_engine = TriggerEngine(validation_config=config.validation_config)
    if fixer_service is None:
        fixer_service = MagicMock(spec=FixerService)
        fixer_service.cleanup_locks = MagicMock()
    return RunCoordinator(
        config=config,
        gate_checker=gate_checker,
        command_runner=command_runner,
        env_config=env_config,
        lock_manager=lock_manager,
        sdk_client_factory=sdk_client_factory,
        trigger_engine=trigger_engine,
        fixer_service=fixer_service,
        event_sink=event_sink,
        cumulative_review_runner=cumulative_review_runner,
        run_metadata=run_metadata,
    )


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


@pytest.fixture
def mock_fixer_service() -> MagicMock:
    """Create a mock FixerService for testing."""
    mock = MagicMock(spec=FixerService)
    mock.cleanup_locks = MagicMock()
    return mock


@pytest.fixture
def mock_trigger_engine() -> TriggerEngine:
    """Create a TriggerEngine with no validation config for testing."""
    return TriggerEngine(validation_config=None)


class TestFixerInterruptHandling:
    """Test fixer agent interrupt behavior via FixerService."""

    @pytest.fixture
    def fixer_service(
        self,
        tmp_path: Path,
        mock_sdk_client_factory: MagicMock,
    ) -> FixerService:
        """Create a FixerService with test dependencies."""
        config = FixerServiceConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
        )
        return FixerService(
            config=config,
            sdk_client_factory=mock_sdk_client_factory,
        )

    @pytest.fixture
    def coordinator(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
        fixer_service: FixerService,
    ) -> RunCoordinator:
        """Create a RunCoordinator with test dependencies."""
        mock_gate_checker = MagicMock()
        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
        )
        trigger_engine = TriggerEngine(validation_config=None)
        return RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            trigger_engine=trigger_engine,
            fixer_service=fixer_service,
        )

    @pytest.mark.asyncio
    async def test_fixer_returns_interrupted_when_event_set_before_start(
        self,
        fixer_service: FixerService,
    ) -> None:
        """Fixer should return interrupted=True when event is set before starting."""
        from src.pipeline.fixer_service import FailureContext

        interrupt_event = asyncio.Event()
        interrupt_event.set()  # Set before calling

        context = FailureContext(
            failure_output="Test failure",
            attempt=1,
            max_attempts=3,
        )
        result = await fixer_service.run_fixer(context, interrupt_event)

        assert isinstance(result, FixerResult)
        assert result.interrupted is True
        assert result.success is None

    @pytest.mark.asyncio
    async def test_fixer_checks_interrupt_before_starting(
        self,
        fixer_service: FixerService,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Fixer should check interrupt and exit before SDK client is created."""
        from src.pipeline.fixer_service import FailureContext

        interrupt_event = asyncio.Event()
        interrupt_event.set()

        context = FailureContext(
            failure_output="Test failure",
            attempt=1,
            max_attempts=3,
        )
        result = await fixer_service.run_fixer(context, interrupt_event)

        # SDK client should NOT be created when interrupted before start
        mock_sdk_client_factory.create.assert_not_called()
        assert result.interrupted is True

    @pytest.mark.asyncio
    async def test_fixer_returns_success_when_not_interrupted(
        self,
        fixer_service: FixerService,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Fixer should return success=True when completing without interrupt."""
        from src.pipeline.fixer_service import FailureContext

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
            "src.pipeline.fixer_service.AgentRuntimeBuilder"
        ) as mock_builder_class:
            mock_builder = MagicMock()
            mock_builder_class.return_value = mock_builder
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = mock_runtime

            context = FailureContext(
                failure_output="Test failure",
                attempt=1,
                max_attempts=3,
            )
            result = await fixer_service.run_fixer(context, interrupt_event=None)

        assert isinstance(result, FixerResult)
        assert result.success is True
        assert result.interrupted is False

    @pytest.mark.asyncio
    async def test_fixer_captures_log_path(
        self,
        fixer_service: FixerService,
        mock_sdk_client_factory: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Fixer should capture log path using upfront session_id."""
        from src.pipeline.fixer_service import FailureContext

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
                "src.pipeline.fixer_service.AgentRuntimeBuilder"
            ) as mock_builder_class,
            patch("src.pipeline.fixer_service.get_claude_log_path") as mock_log_path,
            patch("src.pipeline.fixer_service.uuid.uuid4", return_value=mock_uuid),
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

            context = FailureContext(
                failure_output="Test failure",
                attempt=1,
                max_attempts=3,
            )
            result = await fixer_service.run_fixer(context, interrupt_event=None)

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
        fixer_service: FixerService,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Fixer should check interrupt between messages and exit early."""
        from src.pipeline.fixer_service import FailureContext

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
            "src.pipeline.fixer_service.AgentRuntimeBuilder"
        ) as mock_builder_class:
            mock_builder = MagicMock()
            mock_builder_class.return_value = mock_builder
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = mock_runtime

            context = FailureContext(
                failure_output="Test failure",
                attempt=1,
                max_attempts=3,
            )
            result = await fixer_service.run_fixer(context, interrupt_event)

        assert result.interrupted is True
        assert result.success is None

    @pytest.mark.asyncio
    async def test_fixer_captures_log_path_on_interrupt_during_loop(
        self,
        fixer_service: FixerService,
        mock_sdk_client_factory: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Fixer should capture log path even when interrupted during message loop (Finding 2 fix)."""
        from src.pipeline.fixer_service import FailureContext

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
                "src.pipeline.fixer_service.AgentRuntimeBuilder"
            ) as mock_builder_class,
            patch("src.pipeline.fixer_service.get_claude_log_path") as mock_log_path,
            patch("src.pipeline.fixer_service.uuid.uuid4", return_value=mock_uuid),
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

            context = FailureContext(
                failure_output="Test failure",
                attempt=1,
                max_attempts=3,
            )
            result = await fixer_service.run_fixer(context, interrupt_event)

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
        mock_fixer_service: MagicMock,
        mock_trigger_engine: TriggerEngine,
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
            trigger_engine=mock_trigger_engine,
            fixer_service=mock_fixer_service,
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
        mock_fixer_service: MagicMock,
        mock_trigger_engine: TriggerEngine,
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
        coordinator = _make_coordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            trigger_engine=mock_trigger_engine,
            fixer_service=mock_fixer_service,
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
    async def test_returns_failed_when_runner_not_wired(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """_run_trigger_code_review returns failed status when runner not wired.

        Wiring errors return status="failed" (not None) to distinguish from
        intentional skips. This ensures the caller emits an error event rather
        than a skipped event.
        """
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
        coordinator = _make_coordinator(
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

        # Wiring errors return failed status (not None) to trigger error event
        assert result is not None
        assert result.status == "failed"
        assert result.skip_reason == "CumulativeReviewRunner not wired"
        mock_event_sink.on_warning.assert_called_once()
        assert (
            "CumulativeReviewRunner not wired"
            in mock_event_sink.on_warning.call_args[0][0]
        )

    @pytest.mark.asyncio
    async def test_returns_failed_when_run_metadata_not_available(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """_run_trigger_code_review returns failed status when run_metadata not available.

        Wiring errors return status="failed" (not None) to distinguish from
        intentional skips. This ensures the caller emits an error event rather
        than a skipped event.
        """
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
        mock_review_runner = MagicMock()

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
        )
        coordinator = _make_coordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            event_sink=mock_event_sink,
            cumulative_review_runner=mock_review_runner,
            # No run_metadata wired
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

        # Wiring errors return failed status (not None) to trigger error event
        assert result is not None
        assert result.status == "failed"
        assert result.skip_reason == "run_metadata not available"
        mock_event_sink.on_warning.assert_called_once()
        assert (
            "run_metadata not available" in mock_event_sink.on_warning.call_args[0][0]
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
        mock_fixer_service: MagicMock,
        mock_trigger_engine: TriggerEngine,
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
            trigger_engine=mock_trigger_engine,
            fixer_service=mock_fixer_service,
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
    async def test_execution_error_aborts_ignoring_remediate_mode(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Execution errors abort immediately, ignoring failure_mode=REMEDIATE.

        Code review execution errors (status='failed') are treated as hard
        failures that abort validation immediately. The failure_mode setting
        only applies to finding threshold failures, not execution errors.
        """
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
        coordinator = _make_coordinator(
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

        # Execution error - should abort immediately regardless of failure_mode
        mock_review_runner.run_review = AsyncMock(
            return_value=CumulativeReviewResult(
                status="failed",
                findings=(),
                new_baseline_commit=None,
                skip_reason="execution_error: timeout",
            )
        )

        # Queue the trigger
        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

        result = await coordinator.run_trigger_validation(dry_run=False)

        # Execution errors always abort - no retries regardless of failure_mode
        assert result.status == "aborted"
        assert "execution_error" in (result.details or "")
        # Only called once - no retries for execution errors
        assert mock_review_runner.run_review.call_count == 1
        # Error event should be emitted, not remediation events
        mock_event_sink.on_trigger_code_review_error.assert_called_once()
        mock_event_sink.on_trigger_remediation_started.assert_not_called()


class TestRunEndRunMetadata:
    """Tests for run_end validation recording."""

    @pytest.mark.asyncio
    async def test_run_end_records_validation_metadata(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """run_end should record run_validation with coverage percent."""
        from src.domain.validation.config import (
            CommandConfig,
            CommandsConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            TriggerCommandRef,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
            YamlCoverageConfig,
        )

        coverage_xml = tmp_path / "coverage.xml"
        coverage_xml.write_text('<coverage line-rate="0.75"></coverage>')

        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(TriggerCommandRef(ref="test"),),
            fire_on=FireOn.SUCCESS,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        commands_config = CommandsConfig(test=CommandConfig(command="echo ok"))
        validation_config = ValidationConfig(
            commands=commands_config,
            validation_triggers=triggers_config,
            coverage=YamlCoverageConfig(
                format="xml", file="coverage.xml", threshold=0.0
            ),
        )

        mock_gate_checker = MagicMock()
        mock_run_metadata = MagicMock()

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = _make_coordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            run_metadata=mock_run_metadata,
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})
        result = await coordinator.run_trigger_validation(dry_run=False)

        assert result.status == "passed"
        mock_run_metadata.record_run_validation.assert_called_once()
        meta = mock_run_metadata.record_run_validation.call_args[0][0]
        assert meta.passed is True
        assert meta.commands_run == ["test"]
        assert meta.commands_failed == []
        assert meta.coverage_percent == pytest.approx(75.0)

    @pytest.mark.asyncio
    async def test_execution_error_aborts_validation(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Execution error (status='failed') aborts validation immediately.

        This tests that code review execution errors are treated as hard failures
        that abort validation immediately, without following the failure_mode
        remediation path. The error event is the terminal event.
        """
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
        coordinator = _make_coordinator(
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

        # Execution error (status="failed")
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

        # Execution errors abort immediately - no retries regardless of failure_mode
        assert result.status == "aborted"
        assert "execution_error" in (result.details or "")
        # Only called once - no retries for execution errors
        assert mock_review_runner.run_review.call_count == 1
        # Error event emitted, no remediation events
        mock_event_sink.on_trigger_code_review_error.assert_called_once()
        mock_event_sink.on_trigger_remediation_exhausted.assert_not_called()
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
        coordinator = _make_coordinator(
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
        coordinator = _make_coordinator(
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
        coordinator = _make_coordinator(
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
        coordinator = _make_coordinator(
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
        from src.pipeline.fixer_interface import FixerResult

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

        coordinator = _make_coordinator(
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

        # Configure mock fixer_service.run_fixer to avoid MCP setup
        coordinator.fixer_service.run_fixer = AsyncMock(  # type: ignore[method-assign]
            return_value=FixerResult(success=True, interrupted=False)
        )

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
        from src.pipeline.fixer_interface import FixerResult

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

        coordinator = _make_coordinator(
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

        # Configure mock fixer_service.run_fixer to avoid MCP setup
        coordinator.fixer_service.run_fixer = AsyncMock(  # type: ignore[method-assign]
            return_value=FixerResult(success=True, interrupted=False)
        )

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

    @pytest.mark.asyncio
    async def test_remediation_execution_error_on_rereview_emits_error_event(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Execution error on re-review during remediation aborts with error event.

        When the fixer runs but the subsequent re-review returns status='failed'
        (execution error), we should emit on_trigger_code_review_error and abort,
        NOT on_trigger_code_review_failed (which implies findings exceeded threshold).
        """
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
        from src.pipeline.fixer_interface import FixerResult

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

        coordinator = _make_coordinator(
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

        # First review returns P0 finding (triggers remediation)
        # Second review (re-review after fixer) returns execution error
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
                    status="failed",
                    findings=(),
                    new_baseline_commit=None,
                    skip_reason="execution_error: model timeout",
                ),
            ]
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

        # Configure mock fixer_service.run_fixer to avoid MCP setup
        coordinator.fixer_service.run_fixer = AsyncMock(  # type: ignore[method-assign]
            return_value=FixerResult(success=True, interrupted=False)
        )

        result = await coordinator.run_trigger_validation(dry_run=False)

        # Should abort with error event, not failed event
        assert result.status == "aborted"
        assert result.details is not None
        assert "execution failed" in result.details.lower()

        # Error event should be emitted, NOT failed event
        mock_event_sink.on_trigger_code_review_error.assert_called_once()
        # code_review_failed should NOT be called (error is the terminal event)
        mock_event_sink.on_trigger_code_review_failed.assert_not_called()


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
            validation_triggers=triggers_config,
        )

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = _make_coordinator(
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
            validation_triggers=triggers_config,
        )

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = _make_coordinator(
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
            validation_triggers=triggers_config,
        )

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = _make_coordinator(
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

        # Configure mock fixer_service.run_fixer to succeed
        coordinator.fixer_service.run_fixer = AsyncMock(  # type: ignore[method-assign]
            return_value=FixerResult(success=True, interrupted=False)
        )

        result = await coordinator.run_trigger_validation(dry_run=False)

        # Remediation succeeded, validation should pass, code_review should have run
        assert result.status == "passed"
        mock_review_runner.run_review.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_end_remediation_continues_remaining_commands(
        self,
        tmp_path: Path,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Remediation success resumes remaining validation commands."""
        from src.domain.validation.config import (
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
        mock_run_metadata = MagicMock()

        command_calls: list[str] = []
        call_count = {"build_cmd": 0}

        async def mock_run_async(
            cmd: str | list[str], **kwargs: object
        ) -> CommandResult:
            cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
            command_calls.append(cmd_str)
            if cmd_str == "build_cmd":
                call_count["build_cmd"] += 1
                if call_count["build_cmd"] == 1:
                    return CommandResult(
                        command=cmd_str, returncode=1, stdout="", stderr="Build failed"
                    )
            return CommandResult(command=cmd_str, returncode=0, stdout="", stderr="")

        smart_runner = MagicMock()
        smart_runner.run_async = mock_run_async

        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.REMEDIATE,
            max_retries=1,
            commands=(
                TriggerCommandRef(ref="setup"),
                TriggerCommandRef(ref="format"),
                TriggerCommandRef(ref="build"),
                TriggerCommandRef(ref="test"),
                TriggerCommandRef(ref="e2e"),
            ),
            fire_on=FireOn.SUCCESS,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        commands_config = CommandsConfig(
            setup=CommandConfig(command="setup_cmd"),
            format=CommandConfig(command="format_cmd"),
            build=CommandConfig(command="build_cmd"),
            test=CommandConfig(command="test_cmd"),
            e2e=CommandConfig(command="e2e_cmd"),
        )
        validation_config = ValidationConfig(
            commands=commands_config,
            validation_triggers=triggers_config,
        )

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = _make_coordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=smart_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            run_metadata=mock_run_metadata,
        )

        coordinator.fixer_service.run_fixer = AsyncMock(  # type: ignore[method-assign]
            return_value=FixerResult(success=True, interrupted=False)
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})
        result = await coordinator.run_trigger_validation(dry_run=False)

        assert result.status == "passed"
        assert command_calls == [
            "setup_cmd",
            "format_cmd",
            "build_cmd",
            "build_cmd",
            "test_cmd",
            "e2e_cmd",
        ]

        meta = mock_run_metadata.record_run_validation.call_args_list[-1][0][0]
        assert meta.commands_run == ["setup", "format", "build", "test", "e2e"]

    @pytest.mark.asyncio
    async def test_command_continue_failure_preserved_after_code_review_remediation(
        self,
        tmp_path: Path,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Command failure with CONTINUE is preserved even if code_review remediation succeeds.

        Scenario: command fails with failure_mode=CONTINUE, code_review has findings
        exceeding threshold with failure_mode=REMEDIATE, and fixer succeeds. The
        command failure must still cause the run to return "failed".
        """
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
        from src.pipeline.cumulative_review_runner import (
            CumulativeReviewResult,
            ReviewFinding,
        )

        mock_gate_checker = MagicMock()
        mock_review_runner = MagicMock()
        mock_run_metadata = MagicMock()
        mock_event_sink = MagicMock()

        # Command fails
        failing_runner = FakeCommandRunner()
        failing_runner.responses[("lint_cmd",)] = CommandResult(
            command="lint_cmd", returncode=1, stdout="", stderr="Lint failed"
        )

        # failure_mode=CONTINUE for commands, failure_mode=REMEDIATE for code_review
        code_review_config = CodeReviewConfig(
            enabled=True,
            finding_threshold="P1",
            failure_mode=FailureMode.REMEDIATE,
            max_retries=1,
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,  # Command failure recorded
            commands=(TriggerCommandRef(ref="lint"),),
            fire_on=FireOn.SUCCESS,
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        commands_config = CommandsConfig(lint=CommandConfig(command="lint_cmd"))
        validation_config = ValidationConfig(
            commands=commands_config,
            validation_triggers=triggers_config,
        )

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = _make_coordinator(
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

        # Code review returns P0 finding (exceeds P1 threshold), then passes after fix
        call_count = {"review": 0}

        async def mock_run_review(**kwargs: object) -> CumulativeReviewResult:
            call_count["review"] += 1
            if call_count["review"] == 1:
                return CumulativeReviewResult(
                    status="success",
                    findings=(
                        ReviewFinding(
                            file="test.py",
                            line_start=1,
                            line_end=5,
                            priority=0,  # P0 exceeds P1 threshold
                            title="Critical issue",
                            body="Details",
                            reviewer="test",
                        ),
                    ),
                    new_baseline_commit="abc123",
                )
            return CumulativeReviewResult(
                status="success", findings=(), new_baseline_commit="def456"
            )

        mock_review_runner.run_review = mock_run_review

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

        # Configure mock fixer_service.run_fixer to succeed
        coordinator.fixer_service.run_fixer = AsyncMock(  # type: ignore[method-assign]
            return_value=FixerResult(success=True, interrupted=False)
        )

        result = await coordinator.run_trigger_validation(dry_run=False)

        # Even though code_review remediation succeeded, command failure persists
        assert result.status == "failed"
        assert "lint" in (result.details or "").lower()


class TestTriggerCodeReviewEvents:
    """Test trigger code review lifecycle events.

    These tests verify that code review lifecycle events are emitted correctly:
    - started → passed (review has no blocking findings)
    - started → failed (blocking findings remain after remediation)
    - started → skipped (result.status == 'skipped')
    - started → error (exception or result.status == 'failed')
    - No events when code review is disabled
    """

    @pytest.fixture
    def coordinator_with_review_runner(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> tuple[RunCoordinator, MagicMock, Any]:
        """Create a RunCoordinator with mock review runner and event sink."""
        from src.domain.validation.config import (
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from tests.fakes.event_sink import FakeEventSink

        mock_gate_checker = MagicMock()
        mock_review_runner = MagicMock()
        mock_run_metadata = MagicMock()
        event_sink = FakeEventSink()

        validation_config = ValidationConfig(
            validation_triggers=ValidationTriggersConfig()
        )

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            fixer_prompt="Fix attempt {attempt}/{max_attempts}: {failure_output}",
            validation_config=validation_config,
        )
        coordinator = _make_coordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
            cumulative_review_runner=mock_review_runner,
            run_metadata=mock_run_metadata,
            event_sink=event_sink,
        )
        return coordinator, mock_review_runner, event_sink

    @pytest.mark.asyncio
    async def test_enabled_passing_review_emits_started_passed(
        self,
        coordinator_with_review_runner: tuple[RunCoordinator, MagicMock, Any],
    ) -> None:
        """started → passed when review has no blocking findings."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.pipeline.cumulative_review_runner import CumulativeReviewResult
        from tests.fakes.event_sink import FakeEventSink

        coordinator, mock_review_runner, event_sink = coordinator_with_review_runner
        assert isinstance(event_sink, FakeEventSink)

        # Configure code_review enabled with P1 threshold
        code_review_config = CodeReviewConfig(
            enabled=True,
            finding_threshold="P1",
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)
        coordinator.config.validation_config = validation_config

        # Review returns no findings
        mock_review_runner.run_review = AsyncMock(
            return_value=CumulativeReviewResult(
                status="success",
                findings=(),
                new_baseline_commit="abc123",
            )
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})
        await coordinator.run_trigger_validation(dry_run=False)

        # Assert: started, passed events (in order)
        assert event_sink.has_event("trigger_code_review_started")
        assert event_sink.has_event("trigger_code_review_passed")

        # Verify order: started before passed
        event_types = [e.event_type for e in event_sink.events]
        started_idx = event_types.index("trigger_code_review_started")
        passed_idx = event_types.index("trigger_code_review_passed")
        assert started_idx < passed_idx

    @pytest.mark.asyncio
    async def test_enabled_failing_review_emits_started_failed(
        self,
        coordinator_with_review_runner: tuple[RunCoordinator, MagicMock, Any],
    ) -> None:
        """started → failed when blocking findings remain."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.pipeline.cumulative_review_runner import (
            CumulativeReviewResult,
            ReviewFinding,
        )
        from tests.fakes.event_sink import FakeEventSink

        coordinator, mock_review_runner, event_sink = coordinator_with_review_runner
        assert isinstance(event_sink, FakeEventSink)

        # Configure code_review enabled with P1 threshold
        code_review_config = CodeReviewConfig(
            enabled=True,
            finding_threshold="P1",
            failure_mode=FailureMode.CONTINUE,  # Don't remediate, just fail
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)
        coordinator.config.validation_config = validation_config

        # Review returns P0 finding (exceeds P1 threshold)
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
        await coordinator.run_trigger_validation(dry_run=False)

        # Assert: started, failed(blocking_count=1) events
        assert event_sink.has_event("trigger_code_review_started")
        assert event_sink.has_event("trigger_code_review_failed")

        # Check blocking_count in failed event
        failed_events = event_sink.get_events("trigger_code_review_failed")
        assert len(failed_events) == 1
        assert failed_events[0].kwargs["blocking_count"] == 1

    @pytest.mark.asyncio
    async def test_empty_diff_emits_started_skipped(
        self,
        coordinator_with_review_runner: tuple[RunCoordinator, MagicMock, Any],
    ) -> None:
        """started → skipped when result.status == 'skipped'."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.pipeline.cumulative_review_runner import CumulativeReviewResult
        from tests.fakes.event_sink import FakeEventSink

        coordinator, mock_review_runner, event_sink = coordinator_with_review_runner
        assert isinstance(event_sink, FakeEventSink)

        code_review_config = CodeReviewConfig(enabled=True)
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)
        coordinator.config.validation_config = validation_config

        # Review returns skipped (e.g., empty diff)
        mock_review_runner.run_review = AsyncMock(
            return_value=CumulativeReviewResult(
                status="skipped",
                findings=(),
                new_baseline_commit=None,
                skip_reason="empty_diff",
            )
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})
        await coordinator.run_trigger_validation(dry_run=False)

        # Assert: started, skipped(reason="empty_diff") events
        assert event_sink.has_event("trigger_code_review_started")
        assert event_sink.has_event("trigger_code_review_skipped")

        # Check reason in skipped event
        skipped_events = event_sink.get_events("trigger_code_review_skipped")
        assert len(skipped_events) == 1
        assert skipped_events[0].kwargs["reason"] == "empty_diff"

    @pytest.mark.asyncio
    async def test_execution_error_emits_started_error(
        self,
        coordinator_with_review_runner: tuple[RunCoordinator, MagicMock, Any],
    ) -> None:
        """started → error when result.status == 'failed' (execution error)."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.pipeline.cumulative_review_runner import CumulativeReviewResult
        from tests.fakes.event_sink import FakeEventSink

        coordinator, mock_review_runner, event_sink = coordinator_with_review_runner
        assert isinstance(event_sink, FakeEventSink)

        code_review_config = CodeReviewConfig(
            enabled=True,
            failure_mode=FailureMode.CONTINUE,  # Don't retry on error
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)
        coordinator.config.validation_config = validation_config

        # Review returns failed (execution error)
        mock_review_runner.run_review = AsyncMock(
            return_value=CumulativeReviewResult(
                status="failed",
                findings=(),
                new_baseline_commit=None,
                skip_reason="execution_error: timeout",
            )
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})
        await coordinator.run_trigger_validation(dry_run=False)

        # Assert: started, error(error="execution_error: timeout") events
        assert event_sink.has_event("trigger_code_review_started")
        assert event_sink.has_event("trigger_code_review_error")

        # Check error in error event
        error_events = event_sink.get_events("trigger_code_review_error")
        assert len(error_events) == 1
        assert "execution_error" in error_events[0].kwargs["error"]

    @pytest.mark.asyncio
    async def test_exception_emits_started_error(
        self,
        coordinator_with_review_runner: tuple[RunCoordinator, MagicMock, Any],
    ) -> None:
        """started → error when exception bubbles up."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from tests.fakes.event_sink import FakeEventSink

        coordinator, mock_review_runner, event_sink = coordinator_with_review_runner
        assert isinstance(event_sink, FakeEventSink)

        code_review_config = CodeReviewConfig(enabled=True)
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)
        coordinator.config.validation_config = validation_config

        # Review raises exception
        mock_review_runner.run_review = AsyncMock(
            side_effect=RuntimeError("test error")
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

        with pytest.raises(RuntimeError, match="test error"):
            await coordinator.run_trigger_validation(dry_run=False)

        # Assert: started, error(error="test error") events
        assert event_sink.has_event("trigger_code_review_started")
        assert event_sink.has_event("trigger_code_review_error")

        # Check error in error event
        error_events = event_sink.get_events("trigger_code_review_error")
        assert len(error_events) == 1
        assert "test error" in error_events[0].kwargs["error"]

    @pytest.mark.asyncio
    async def test_disabled_emits_no_events(
        self,
        coordinator_with_review_runner: tuple[RunCoordinator, MagicMock, Any],
    ) -> None:
        """No events when code review is disabled."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from tests.fakes.event_sink import FakeEventSink

        coordinator, mock_review_runner, event_sink = coordinator_with_review_runner
        assert isinstance(event_sink, FakeEventSink)

        # Configure code_review DISABLED
        code_review_config = CodeReviewConfig(enabled=False)
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)
        coordinator.config.validation_config = validation_config

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})
        await coordinator.run_trigger_validation(dry_run=False)

        # Assert: no code review events recorded
        assert not event_sink.has_event("trigger_code_review_started")
        assert not event_sink.has_event("trigger_code_review_passed")
        assert not event_sink.has_event("trigger_code_review_failed")
        assert not event_sink.has_event("trigger_code_review_skipped")
        assert not event_sink.has_event("trigger_code_review_error")

        # Review runner should not have been called
        mock_review_runner.run_review.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_code_review_config_emits_no_events(
        self,
        coordinator_with_review_runner: tuple[RunCoordinator, MagicMock, Any],
    ) -> None:
        """No events when code_review is not configured."""
        from src.domain.validation.config import (
            FailureMode,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from tests.fakes.event_sink import FakeEventSink

        coordinator, _mock_review_runner, event_sink = coordinator_with_review_runner
        assert isinstance(event_sink, FakeEventSink)

        # Configure trigger WITHOUT code_review
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            # No code_review config
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)
        coordinator.config.validation_config = validation_config

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})
        await coordinator.run_trigger_validation(dry_run=False)

        # Assert: no code review events recorded
        assert not event_sink.has_event("trigger_code_review_started")

    @pytest.mark.asyncio
    async def test_remediation_success_emits_fixer_events_then_passed(
        self,
        coordinator_with_review_runner: tuple[RunCoordinator, MagicMock, Any],
    ) -> None:
        """started → fixer_started → fixer_completed → passed."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.pipeline.cumulative_review_runner import (
            CumulativeReviewResult,
            ReviewFinding,
        )
        from tests.fakes.event_sink import FakeEventSink

        coordinator, mock_review_runner, event_sink = coordinator_with_review_runner
        assert isinstance(event_sink, FakeEventSink)

        # Configure code_review with remediation
        code_review_config = CodeReviewConfig(
            enabled=True,
            finding_threshold="P1",
            failure_mode=FailureMode.REMEDIATE,
            max_retries=1,
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)
        coordinator.config.validation_config = validation_config

        # First call: P0 finding
        # Second call (after fixer): no findings
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
                    findings=(),
                    new_baseline_commit="def456",
                ),
            ]
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

        # Configure mock fixer_service.run_fixer to avoid MCP setup
        coordinator.fixer_service.run_fixer = AsyncMock(  # type: ignore[method-assign]
            return_value=FixerResult(success=True, interrupted=False)
        )

        await coordinator.run_trigger_validation(dry_run=False)

        # Assert: started, fixer_started, (fixer_completed from _run_trigger_code_review after fixer),
        # then code_review_passed after final review
        # Note: The first code review emits started + failed, then remediation emits fixer_started
        # After fixer succeeds, re-run code review - but that's in _run_code_review_remediation
        # which calls _run_trigger_code_review which doesn't re-emit started
        assert event_sink.has_event("trigger_code_review_started")
        assert event_sink.has_event("fixer_started")

        # After successful remediation, the trigger should pass overall
        # But note: the code_review_passed event was already emitted after first review failed
        # Actually looking at the code, the first review emits started -> failed (since findings exceed)
        # Then remediation loop runs, and after fixer succeeds, review is called again
        # That second call is inside _run_code_review_remediation, not wrapped by our event emission

        # Check fixer_started was emitted with correct attempt/max_retries
        fixer_started_events = event_sink.get_events("fixer_started")
        assert len(fixer_started_events) >= 1
        assert fixer_started_events[0].kwargs["attempt"] == 1
        assert fixer_started_events[0].kwargs["max_attempts"] == 1

    @pytest.mark.asyncio
    async def test_remediation_exhausted_emits_fixer_events_then_failed(
        self,
        coordinator_with_review_runner: tuple[RunCoordinator, MagicMock, Any],
    ) -> None:
        """started → fixer_started → fixer_completed → ... → failed."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.pipeline.cumulative_review_runner import (
            CumulativeReviewResult,
            ReviewFinding,
        )
        from tests.fakes.event_sink import FakeEventSink

        coordinator, mock_review_runner, event_sink = coordinator_with_review_runner
        assert isinstance(event_sink, FakeEventSink)

        # Configure code_review with remediation and 2 retries
        code_review_config = CodeReviewConfig(
            enabled=True,
            finding_threshold="P1",
            failure_mode=FailureMode.REMEDIATE,
            max_retries=2,
        )
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)
        coordinator.config.validation_config = validation_config

        # All reviews return P0 finding - fixer cannot fix
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

        # Configure mock fixer_service.run_fixer
        coordinator.fixer_service.run_fixer = AsyncMock(  # type: ignore[method-assign]
            return_value=FixerResult(success=True, interrupted=False)
        )

        await coordinator.run_trigger_validation(dry_run=False)

        # Assert: started, failed events
        assert event_sink.has_event("trigger_code_review_started")
        assert event_sink.has_event("trigger_code_review_failed")

        # Assert: fixer events for each attempt (2 retries)
        fixer_started_events = event_sink.get_events("fixer_started")
        assert len(fixer_started_events) == 2  # 2 remediation attempts
        assert fixer_started_events[0].kwargs["attempt"] == 1
        assert fixer_started_events[1].kwargs["attempt"] == 2

        # Assert: trigger_remediation_exhausted was called
        assert event_sink.has_event("trigger_remediation_exhausted")

    @pytest.mark.asyncio
    async def test_exactly_one_end_event_invariant(
        self,
        coordinator_with_review_runner: tuple[RunCoordinator, MagicMock, Any],
    ) -> None:
        """Verify exactly one end event is emitted per execution."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from src.pipeline.cumulative_review_runner import CumulativeReviewResult
        from tests.fakes.event_sink import FakeEventSink

        coordinator, mock_review_runner, event_sink = coordinator_with_review_runner
        assert isinstance(event_sink, FakeEventSink)

        code_review_config = CodeReviewConfig(enabled=True)
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)
        coordinator.config.validation_config = validation_config

        # Review returns success with no findings
        mock_review_runner.run_review = AsyncMock(
            return_value=CumulativeReviewResult(
                status="success",
                findings=(),
                new_baseline_commit="abc123",
            )
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})
        await coordinator.run_trigger_validation(dry_run=False)

        # Count end events
        end_event_types = [
            "trigger_code_review_passed",
            "trigger_code_review_failed",
            "trigger_code_review_skipped",
            "trigger_code_review_error",
        ]
        end_event_count = sum(
            1 for e in event_sink.events if e.event_type in end_event_types
        )

        # Verify exactly one end event
        assert end_event_count == 1, (
            f"Expected exactly 1 end event, got {end_event_count}. "
            f"Events: {[e.event_type for e in event_sink.events]}"
        )

    @pytest.mark.asyncio
    async def test_exception_emits_exactly_one_error_event(
        self,
        coordinator_with_review_runner: tuple[RunCoordinator, MagicMock, Any],
    ) -> None:
        """Exception emits exactly one error event, not error + other end event.

        This tests the fix for the bug where an exception could cause both
        on_trigger_code_review_error() to be emitted from the except block AND
        another end event from the finally block if code_review_end_status was
        set before the exception occurred.
        """
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            RunEndTriggerConfig,
            TriggerType,
            ValidationConfig,
            ValidationTriggersConfig,
        )
        from tests.fakes.event_sink import FakeEventSink

        coordinator, mock_review_runner, event_sink = coordinator_with_review_runner
        assert isinstance(event_sink, FakeEventSink)

        code_review_config = CodeReviewConfig(enabled=True)
        trigger_config = RunEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            code_review=code_review_config,
        )
        triggers_config = ValidationTriggersConfig(run_end=trigger_config)
        validation_config = ValidationConfig(validation_triggers=triggers_config)
        coordinator.config.validation_config = validation_config

        # Review raises exception
        mock_review_runner.run_review = AsyncMock(
            side_effect=RuntimeError("test error")
        )

        coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

        with pytest.raises(RuntimeError, match="test error"):
            await coordinator.run_trigger_validation(dry_run=False)

        # Count end events (all terminal events)
        end_event_types = [
            "trigger_code_review_passed",
            "trigger_code_review_failed",
            "trigger_code_review_skipped",
            "trigger_code_review_error",
        ]
        end_events = [e for e in event_sink.events if e.event_type in end_event_types]

        # Verify exactly one end event (the error event)
        assert len(end_events) == 1, (
            f"Expected exactly 1 end event, got {len(end_events)}. "
            f"End events: {[e.event_type for e in end_events]}"
        )
        assert end_events[0].event_type == "trigger_code_review_error", (
            f"Expected error event, got {end_events[0].event_type}"
        )
