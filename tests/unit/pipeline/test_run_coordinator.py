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
