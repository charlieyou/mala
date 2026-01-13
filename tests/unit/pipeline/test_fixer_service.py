"""Unit tests for FixerService.

Tests the FixerService class without subprocess or SDK dependencies,
using mock SDK clients and event sinks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.fixer_service import (
    FailureContext,
    FixerService,
    FixerServiceConfig,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Self


class MockSDKClient:
    """Mock SDK client for testing FixerService."""

    def __init__(
        self,
        messages: list[object] | None = None,
        raise_on_receive: Exception | None = None,
    ) -> None:
        self.messages = messages or []
        self.raise_on_receive = raise_on_receive
        self.query_calls: list[tuple[str, str | None]] = []
        self._entered = False

    async def __aenter__(self) -> Self:
        self._entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self._entered = False

    async def query(self, prompt: str, session_id: str | None = None) -> None:
        self.query_calls.append((prompt, session_id))

    async def receive_response(self) -> AsyncIterator[object]:
        if self.raise_on_receive is not None:
            raise self.raise_on_receive
        for msg in self.messages:
            yield msg


def make_mock_sdk_client_factory(
    client: MockSDKClient | None = None,
) -> MagicMock:
    """Create a mock SDKClientFactoryProtocol."""
    factory = MagicMock()
    factory.create.return_value = client or MockSDKClient()
    return factory


def make_mock_runtime() -> MagicMock:
    """Create a mock AgentRuntime."""
    runtime = MagicMock()
    runtime.options = MagicMock()
    runtime.lint_cache = MagicMock()
    runtime.lint_cache.detect_lint_command.return_value = None
    return runtime


def make_config(
    repo_path: Path | None = None,
    timeout_seconds: int = 600,
    fixer_prompt: str = "Fix: {failure_output}",
) -> FixerServiceConfig:
    """Create a FixerServiceConfig for testing."""
    return FixerServiceConfig(
        repo_path=repo_path or Path("/test/repo"),
        timeout_seconds=timeout_seconds,
        fixer_prompt=fixer_prompt,
    )


def make_failure_context(
    failure_output: str = "Test failure",
    attempt: int = 1,
    max_attempts: int = 3,
    failed_command: str = "test",
    validation_commands: str | None = None,
) -> FailureContext:
    """Create a FailureContext for testing."""
    return FailureContext(
        failure_output=failure_output,
        attempt=attempt,
        max_attempts=max_attempts,
        failed_command=failed_command,
        validation_commands=validation_commands,
    )


class TestFailureContext:
    """Test the FailureContext dataclass."""

    def test_required_fields(self) -> None:
        """Verify required fields must be provided."""
        ctx = FailureContext(
            failure_output="Lint failed",
            attempt=1,
            max_attempts=3,
        )
        assert ctx.failure_output == "Lint failed"
        assert ctx.attempt == 1
        assert ctx.max_attempts == 3
        assert ctx.failed_command == "unknown"
        assert ctx.validation_commands is None
        assert ctx.spec is None

    def test_all_fields(self) -> None:
        """Verify all fields can be set."""
        spec = MagicMock()
        ctx = FailureContext(
            failure_output="Tests failed",
            attempt=2,
            max_attempts=5,
            failed_command="pytest",
            validation_commands="   - `pytest`",
            spec=spec,
        )
        assert ctx.failure_output == "Tests failed"
        assert ctx.attempt == 2
        assert ctx.max_attempts == 5
        assert ctx.failed_command == "pytest"
        assert ctx.validation_commands == "   - `pytest`"
        assert ctx.spec is spec

    def test_frozen(self) -> None:
        """Verify FailureContext is immutable."""
        ctx = FailureContext(
            failure_output="Error",
            attempt=1,
            max_attempts=3,
        )
        with pytest.raises(AttributeError):
            ctx.attempt = 2  # type: ignore[misc]


class TestFixerServiceConfig:
    """Test the FixerServiceConfig dataclass."""

    def test_required_fields(self) -> None:
        """Verify required fields must be provided."""
        config = FixerServiceConfig(
            repo_path=Path("/repo"),
            timeout_seconds=300,
            fixer_prompt="Fix it: {failure_output}",
        )
        assert config.repo_path == Path("/repo")
        assert config.timeout_seconds == 300
        assert config.fixer_prompt == "Fix it: {failure_output}"
        assert config.mcp_server_factory is None

    def test_with_mcp_factory(self) -> None:
        """Verify MCP factory can be set."""
        factory = MagicMock()
        config = FixerServiceConfig(
            repo_path=Path("/repo"),
            timeout_seconds=300,
            fixer_prompt="Fix: {failure_output}",
            mcp_server_factory=factory,
        )
        assert config.mcp_server_factory is factory


class TestFixerServiceSuccess:
    """Test successful fixer agent runs."""

    @pytest.mark.asyncio
    async def test_run_fixer_success(self) -> None:
        """Verify run_fixer returns success when agent completes."""
        client = MockSDKClient()
        factory = make_mock_sdk_client_factory(client)
        config = make_config()
        service = FixerService(config, factory)
        ctx = make_failure_context()

        with patch(
            "src.pipeline.fixer_service.AgentRuntimeBuilder"
        ) as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = make_mock_runtime()
            mock_builder_cls.return_value = mock_builder

            with patch(
                "src.pipeline.fixer_service.get_claude_log_path"
            ) as mock_log_path:
                mock_log_path.return_value = Path("/tmp/fixer.log")

                result = await service.run_fixer(ctx)

        assert result.success is True
        assert result.interrupted is False
        assert result.log_path == "/tmp/fixer.log"

    @pytest.mark.asyncio
    async def test_run_fixer_uses_prompt_template(self) -> None:
        """Verify run_fixer formats the prompt correctly."""
        client = MockSDKClient()
        factory = make_mock_sdk_client_factory(client)
        config = make_config(
            fixer_prompt="Attempt {attempt}/{max_attempts}: {failure_output}"
        )
        service = FixerService(config, factory)
        ctx = make_failure_context(
            failure_output="Lint error",
            attempt=2,
            max_attempts=5,
        )

        with patch(
            "src.pipeline.fixer_service.AgentRuntimeBuilder"
        ) as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = make_mock_runtime()
            mock_builder_cls.return_value = mock_builder

            with patch("src.pipeline.fixer_service.get_claude_log_path"):
                await service.run_fixer(ctx)

        assert len(client.query_calls) == 1
        prompt, _ = client.query_calls[0]
        assert "Attempt 2/5" in prompt
        assert "Lint error" in prompt


class TestFixerServiceFailure:
    """Test fixer agent failure scenarios."""

    @pytest.mark.asyncio
    async def test_run_fixer_timeout(self) -> None:
        """Verify run_fixer returns failure on timeout."""

        async def slow_receive() -> AsyncIterator[object]:
            await asyncio.sleep(10)
            yield MagicMock()

        client = MockSDKClient()
        # Override receive_response to be slow
        client.receive_response = slow_receive  # type: ignore[method-assign]
        factory = make_mock_sdk_client_factory(client)
        config = make_config(timeout_seconds=0)  # Immediate timeout
        event_sink = MagicMock()
        service = FixerService(config, factory, event_sink=event_sink)
        ctx = make_failure_context()

        with patch(
            "src.pipeline.fixer_service.AgentRuntimeBuilder"
        ) as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = make_mock_runtime()
            mock_builder_cls.return_value = mock_builder

            with patch(
                "src.pipeline.fixer_service.get_claude_log_path"
            ) as mock_log_path:
                mock_log_path.return_value = Path("/tmp/fixer.log")

                result = await service.run_fixer(ctx)

        assert result.success is False
        assert result.interrupted is False
        assert result.log_path == "/tmp/fixer.log"
        event_sink.on_fixer_failed.assert_called_once_with("timeout")

    @pytest.mark.asyncio
    async def test_run_fixer_exception(self) -> None:
        """Verify run_fixer returns failure on exception."""
        client = MockSDKClient(raise_on_receive=RuntimeError("Connection lost"))
        factory = make_mock_sdk_client_factory(client)
        config = make_config()
        event_sink = MagicMock()
        service = FixerService(config, factory, event_sink=event_sink)
        ctx = make_failure_context()

        with patch(
            "src.pipeline.fixer_service.AgentRuntimeBuilder"
        ) as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = make_mock_runtime()
            mock_builder_cls.return_value = mock_builder

            with patch(
                "src.pipeline.fixer_service.get_claude_log_path"
            ) as mock_log_path:
                mock_log_path.return_value = Path("/tmp/fixer.log")

                result = await service.run_fixer(ctx)

        assert result.success is False
        assert result.log_path == "/tmp/fixer.log"
        event_sink.on_fixer_failed.assert_called_once_with("Connection lost")


class TestFixerServiceInterrupt:
    """Test fixer agent interrupt scenarios."""

    @pytest.mark.asyncio
    async def test_run_fixer_interrupt_before_start(self) -> None:
        """Verify run_fixer returns interrupted when event is set before start."""
        factory = make_mock_sdk_client_factory()
        config = make_config()
        service = FixerService(config, factory)
        ctx = make_failure_context()

        interrupt_event = asyncio.Event()
        interrupt_event.set()  # Already interrupted

        result = await service.run_fixer(ctx, interrupt_event=interrupt_event)

        assert result.success is None
        assert result.interrupted is True
        assert result.log_path is None

    @pytest.mark.asyncio
    async def test_run_fixer_interrupt_during_messages(self) -> None:
        """Verify run_fixer returns interrupted when event is set during processing."""
        interrupt_event = asyncio.Event()

        # Create a message that triggers interrupt check
        mock_msg = MagicMock()
        type(mock_msg).__name__ = "AssistantMessage"
        mock_msg.content = []

        client = MockSDKClient(messages=[mock_msg])
        factory = make_mock_sdk_client_factory(client)
        config = make_config()
        service = FixerService(config, factory)
        ctx = make_failure_context()

        with patch(
            "src.pipeline.fixer_service.AgentRuntimeBuilder"
        ) as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = make_mock_runtime()
            mock_builder_cls.return_value = mock_builder

            with patch(
                "src.pipeline.fixer_service.get_claude_log_path"
            ) as mock_log_path:
                mock_log_path.return_value = Path("/tmp/fixer.log")

                # Patch InterruptGuard to return interrupted after first check
                with patch(
                    "src.pipeline.fixer_service.InterruptGuard"
                ) as mock_guard_cls:
                    mock_guard = MagicMock()
                    # First call returns False (not interrupted), second returns True
                    mock_guard.is_interrupted.side_effect = [False, True]
                    mock_guard_cls.return_value = mock_guard

                    result = await service.run_fixer(
                        ctx, interrupt_event=interrupt_event
                    )

        assert result.success is None
        assert result.interrupted is True
        assert result.log_path == "/tmp/fixer.log"


class TestFixerServiceEvents:
    """Test event sink integration."""

    @pytest.mark.asyncio
    async def test_emits_text_events(self) -> None:
        """Verify run_fixer emits on_fixer_text for text blocks."""
        # Create a text block message
        text_block = MagicMock()
        type(text_block).__name__ = "TextBlock"
        text_block.text = "Fixing the issue..."

        mock_msg = MagicMock()
        type(mock_msg).__name__ = "AssistantMessage"
        mock_msg.content = [text_block]

        client = MockSDKClient(messages=[mock_msg])
        factory = make_mock_sdk_client_factory(client)
        config = make_config()
        event_sink = MagicMock()
        service = FixerService(config, factory, event_sink=event_sink)
        ctx = make_failure_context(attempt=2)

        with patch(
            "src.pipeline.fixer_service.AgentRuntimeBuilder"
        ) as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = make_mock_runtime()
            mock_builder_cls.return_value = mock_builder

            with patch("src.pipeline.fixer_service.get_claude_log_path"):
                await service.run_fixer(ctx)

        event_sink.on_fixer_text.assert_called_once_with(2, "Fixing the issue...")

    @pytest.mark.asyncio
    async def test_emits_tool_use_events(self) -> None:
        """Verify run_fixer emits on_fixer_tool_use for tool use blocks."""
        # Create a tool use block message
        tool_block = MagicMock()
        type(tool_block).__name__ = "ToolUseBlock"
        tool_block.name = "Bash"
        tool_block.input = {"command": "pytest"}

        mock_msg = MagicMock()
        type(mock_msg).__name__ = "AssistantMessage"
        mock_msg.content = [tool_block]

        client = MockSDKClient(messages=[mock_msg])
        factory = make_mock_sdk_client_factory(client)
        config = make_config()
        event_sink = MagicMock()
        service = FixerService(config, factory, event_sink=event_sink)
        ctx = make_failure_context(attempt=1)

        with patch(
            "src.pipeline.fixer_service.AgentRuntimeBuilder"
        ) as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = make_mock_runtime()
            mock_builder_cls.return_value = mock_builder

            with patch("src.pipeline.fixer_service.get_claude_log_path"):
                await service.run_fixer(ctx)

        event_sink.on_fixer_tool_use.assert_called_once_with(
            1, "Bash", {"command": "pytest"}
        )

    @pytest.mark.asyncio
    async def test_emits_completed_event(self) -> None:
        """Verify run_fixer emits on_fixer_completed for result messages."""
        # Create a result message
        result_msg = MagicMock()
        type(result_msg).__name__ = "ResultMessage"
        result_msg.result = "All fixes applied successfully"

        client = MockSDKClient(messages=[result_msg])
        factory = make_mock_sdk_client_factory(client)
        config = make_config()
        event_sink = MagicMock()
        service = FixerService(config, factory, event_sink=event_sink)
        ctx = make_failure_context()

        with patch(
            "src.pipeline.fixer_service.AgentRuntimeBuilder"
        ) as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = make_mock_runtime()
            mock_builder_cls.return_value = mock_builder

            with patch("src.pipeline.fixer_service.get_claude_log_path"):
                await service.run_fixer(ctx)

        event_sink.on_fixer_completed.assert_called_once_with(
            "All fixes applied successfully"
        )


class TestFixerServiceCleanup:
    """Test lock cleanup functionality."""

    def test_cleanup_locks_clears_active_ids(self) -> None:
        """Verify cleanup_locks clears the active fixer IDs list."""
        factory = make_mock_sdk_client_factory()
        config = make_config()
        service = FixerService(config, factory)

        # Simulate active fixer IDs
        service._active_fixer_ids = ["fixer-abc", "fixer-def"]

        with patch("src.pipeline.fixer_service.cleanup_agent_locks") as mock_cleanup:
            service.cleanup_locks()

        # Verify cleanup was called for each ID
        assert mock_cleanup.call_count == 2
        mock_cleanup.assert_any_call("fixer-abc")
        mock_cleanup.assert_any_call("fixer-def")

        # Verify list is cleared
        assert service._active_fixer_ids == []

    def test_cleanup_locks_empty_list(self) -> None:
        """Verify cleanup_locks handles empty list gracefully."""
        factory = make_mock_sdk_client_factory()
        config = make_config()
        service = FixerService(config, factory)

        with patch("src.pipeline.fixer_service.cleanup_agent_locks") as mock_cleanup:
            service.cleanup_locks()

        mock_cleanup.assert_not_called()
        assert service._active_fixer_ids == []


class TestFixerServiceValidationCommands:
    """Test validation commands string building."""

    @pytest.mark.asyncio
    async def test_uses_provided_validation_commands(self) -> None:
        """Verify run_fixer uses provided validation_commands."""
        client = MockSDKClient()
        factory = make_mock_sdk_client_factory(client)
        config = make_config(fixer_prompt="Commands: {validation_commands}")
        service = FixerService(config, factory)
        ctx = make_failure_context(
            validation_commands="   - `ruff check .`\n   - `pytest`"
        )

        with patch(
            "src.pipeline.fixer_service.AgentRuntimeBuilder"
        ) as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = make_mock_runtime()
            mock_builder_cls.return_value = mock_builder

            with patch("src.pipeline.fixer_service.get_claude_log_path"):
                await service.run_fixer(ctx)

        prompt, _ = client.query_calls[0]
        assert "`ruff check .`" in prompt
        assert "`pytest`" in prompt

    @pytest.mark.asyncio
    async def test_builds_commands_from_spec(self) -> None:
        """Verify run_fixer builds commands from spec when not provided."""
        client = MockSDKClient()
        factory = make_mock_sdk_client_factory(client)
        config = make_config(fixer_prompt="Commands: {validation_commands}")
        service = FixerService(config, factory)

        # Create a spec with commands
        spec = MagicMock()
        cmd1 = MagicMock()
        cmd1.command = "ruff check ."
        cmd2 = MagicMock()
        cmd2.command = "pytest"
        spec.commands = [cmd1, cmd2]

        ctx = FailureContext(
            failure_output="Test failure",
            attempt=1,
            max_attempts=3,
            spec=spec,
        )

        with patch(
            "src.pipeline.fixer_service.AgentRuntimeBuilder"
        ) as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = make_mock_runtime()
            mock_builder_cls.return_value = mock_builder

            with patch("src.pipeline.fixer_service.get_claude_log_path"):
                await service.run_fixer(ctx)

        prompt, _ = client.query_calls[0]
        assert "`ruff check .`" in prompt
        assert "`pytest`" in prompt

    @pytest.mark.asyncio
    async def test_uses_placeholder_when_no_spec(self) -> None:
        """Verify run_fixer uses placeholder when no spec provided."""
        client = MockSDKClient()
        factory = make_mock_sdk_client_factory(client)
        config = make_config(fixer_prompt="Commands: {validation_commands}")
        service = FixerService(config, factory)
        ctx = make_failure_context()  # No spec, no validation_commands

        with patch(
            "src.pipeline.fixer_service.AgentRuntimeBuilder"
        ) as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = make_mock_runtime()
            mock_builder_cls.return_value = mock_builder

            with patch("src.pipeline.fixer_service.get_claude_log_path"):
                await service.run_fixer(ctx)

        prompt, _ = client.query_calls[0]
        assert "Run the appropriate validation commands" in prompt


class TestFixerServiceLintCache:
    """Test lint cache integration."""

    @pytest.mark.asyncio
    async def test_tracks_lint_commands(self) -> None:
        """Verify run_fixer tracks lint commands for caching."""
        # Create a bash tool use block
        tool_block = MagicMock()
        type(tool_block).__name__ = "ToolUseBlock"
        tool_block.name = "bash"
        tool_block.id = "tool-123"
        tool_block.input = {"command": "ruff check ."}

        # Create a tool result block
        result_block = MagicMock()
        type(result_block).__name__ = "ToolResultBlock"
        result_block.tool_use_id = "tool-123"
        result_block.is_error = False

        mock_msg1 = MagicMock()
        type(mock_msg1).__name__ = "AssistantMessage"
        mock_msg1.content = [tool_block]

        mock_msg2 = MagicMock()
        type(mock_msg2).__name__ = "AssistantMessage"
        mock_msg2.content = [result_block]

        client = MockSDKClient(messages=[mock_msg1, mock_msg2])
        factory = make_mock_sdk_client_factory(client)
        config = make_config()
        service = FixerService(config, factory)
        ctx = make_failure_context()

        with patch(
            "src.pipeline.fixer_service.AgentRuntimeBuilder"
        ) as mock_builder_cls:
            mock_runtime = make_mock_runtime()
            mock_runtime.lint_cache.detect_lint_command.return_value = "ruff"

            mock_builder = MagicMock()
            mock_builder.with_hooks.return_value = mock_builder
            mock_builder.with_env.return_value = mock_builder
            mock_builder.with_mcp.return_value = mock_builder
            mock_builder.with_disallowed_tools.return_value = mock_builder
            mock_builder.with_lint_tools.return_value = mock_builder
            mock_builder.build.return_value = mock_runtime
            mock_builder_cls.return_value = mock_builder

            with patch("src.pipeline.fixer_service.get_claude_log_path"):
                await service.run_fixer(ctx)

        # Verify lint cache was updated
        mock_runtime.lint_cache.mark_success.assert_called_once_with(
            "ruff", "ruff check ."
        )
