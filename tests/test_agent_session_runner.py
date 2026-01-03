"""Unit tests for AgentSessionRunner pipeline stage.

Tests the extracted SDK session execution logic using fake SDK clients,
without actual SDK/API dependencies.

This module uses the actual SDK types (ResultMessage, etc.) to ensure
isinstance checks work correctly in the runner.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self, cast
from unittest.mock import MagicMock, patch

import pytest

# Import SDK types that the runner uses for isinstance checks
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

from src.pipeline.agent_session_runner import (
    AgentSessionConfig,
    AgentSessionInput,
    AgentSessionRunner,
    SessionCallbacks,
)
from src.domain.quality_gate import GateResult

if TYPE_CHECKING:
    from src.pipeline.agent_session_runner import (
        SDKClientProtocol,
    )
    from collections.abc import AsyncIterator
    from pathlib import Path

    from src.domain.lifecycle import RetryState


def make_result_message(
    session_id: str = "test-session-123",
    result: str | None = "Test completed successfully",
) -> ResultMessage:
    """Create a ResultMessage with the given fields."""
    return ResultMessage(
        subtype="result",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=1,
        session_id=session_id,
        result=result,
    )


class FakeSDKClient:
    """Fake SDK client for testing.

    Allows tests to configure what messages are returned from receive_response.
    Uses actual SDK types so isinstance checks work correctly.
    """

    def __init__(
        self,
        messages: list[Any] | None = None,
        result_message: ResultMessage | None = None,
    ):
        self.messages = messages or []
        self.result_message = result_message or make_result_message()
        self.queries: list[tuple[str, str | None]] = []
        self._response_index = 0
        self.disconnect_called = False
        self.disconnect_delay: float = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        pass

    async def query(self, prompt: str, session_id: str | None = None) -> None:
        self.queries.append((prompt, session_id))

    async def receive_response(self) -> AsyncIterator[Any]:
        """Yield messages then the result message."""
        for msg in self.messages:
            yield msg
        yield self.result_message

    async def disconnect(self) -> None:
        """Disconnect the client."""
        if self.disconnect_delay > 0:
            await asyncio.sleep(self.disconnect_delay)
        self.disconnect_called = True


class HangingSDKClient(FakeSDKClient):
    """Fake SDK client that never yields a response (simulates hung stream)."""

    async def receive_response(self) -> AsyncIterator[Any]:
        while True:
            await asyncio.sleep(3600)
            if False:  # pragma: no cover
                yield None


class HangingAfterMessagesSDKClient(FakeSDKClient):
    """Fake SDK client that yields configured messages then hangs before ResultMessage.

    Use this to test retry behavior where a session_id is obtained before hang.
    """

    async def receive_response(self) -> AsyncIterator[Any]:
        for msg in self.messages:
            yield msg
        # Hang forever instead of yielding result_message
        while True:
            await asyncio.sleep(3600)
            if False:  # pragma: no cover
                yield None


class SlowSDKClient(FakeSDKClient):
    """Fake SDK client that yields messages after a short delay."""

    def __init__(
        self,
        delay: float,
        messages: list[Any] | None = None,
        result_message: ResultMessage | None = None,
    ):
        super().__init__(messages=messages, result_message=result_message)
        self.delay = delay

    async def receive_response(self) -> AsyncIterator[Any]:
        for msg in self.messages:
            await asyncio.sleep(self.delay)
            yield msg
        await asyncio.sleep(self.delay)
        yield self.result_message


class FakeSDKClientFactory:
    """Factory for creating fake SDK clients in tests."""

    def __init__(self, client: FakeSDKClient):
        self.client = client
        self.create_calls: list[Any] = []

    def create(self, options: object) -> SDKClientProtocol:
        self.create_calls.append(options)
        return cast("SDKClientProtocol", self.client)


class SequencedSDKClientFactory:
    """Factory that returns different clients per create() call.

    This factory is designed for testing retry behavior by returning different
    clients on each call to create(). For example, you can return a hanging
    client on the first call and a successful client on the second call.
    """

    def __init__(self, clients: list[FakeSDKClient]):
        self.clients = clients
        self.create_calls: list[Any] = []
        self._index = 0

    def create(self, options: object) -> SDKClientProtocol:
        self.create_calls.append(options)
        client = self.clients[min(self._index, len(self.clients) - 1)]
        self._index += 1
        return cast("SDKClientProtocol", client)


class TestAgentSessionRunnerBasics:
    """Test basic AgentSessionRunner functionality."""

    @pytest.fixture
    def tmp_log_path(self, tmp_path: Path) -> Path:
        """Create a temporary log file path."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")
        return log_path

    @pytest.fixture
    def session_config(self, tmp_path: Path) -> AgentSessionConfig:
        """Create a session config for testing."""
        return AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=3,
            max_review_retries=2,
            morph_enabled=False,
            review_enabled=False,  # Disable review for basic tests
        )

    @pytest.fixture
    def fake_client(self) -> FakeSDKClient:
        """Create a fake SDK client."""
        return FakeSDKClient()

    @pytest.fixture
    def fake_factory(self, fake_client: FakeSDKClient) -> FakeSDKClientFactory:
        """Create a factory for the fake client."""
        return FakeSDKClientFactory(fake_client)

    @pytest.mark.asyncio
    async def test_run_session_returns_output(
        self,
        session_config: AgentSessionConfig,
        fake_factory: FakeSDKClientFactory,
        tmp_log_path: Path,
    ) -> None:
        """Runner should return session output with correct fields."""

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input)

        assert output.session_id == "test-session-123"
        assert output.success is True
        assert output.agent_id.startswith("test-123-")
        assert output.gate_attempts >= 1
        assert output.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_run_session_sends_initial_query(
        self,
        session_config: AgentSessionConfig,
        fake_client: FakeSDKClient,
        fake_factory: FakeSDKClientFactory,
        tmp_log_path: Path,
    ) -> None:
        """Runner should send the initial prompt to the client."""

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="My test prompt",
        )

        await runner.run_session(input)

        # Check the first query was the prompt
        assert len(fake_client.queries) >= 1
        assert fake_client.queries[0] == ("My test prompt", None)

    @pytest.mark.asyncio
    async def test_idle_timeout_aborts_session(
        self,
        tmp_path: Path,
    ) -> None:
        """Runner should fail fast when SDK stream is idle."""
        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            idle_timeout_seconds=0.1,
            review_enabled=False,
        )
        fake_client = HangingSDKClient()
        fake_factory = FakeSDKClientFactory(fake_client)

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=SessionCallbacks(),
            sdk_client_factory=fake_factory,
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input)

        assert output.success is False
        assert "idle" in output.summary.lower()

    @pytest.mark.asyncio
    async def test_idle_timeout_allows_slow_stream(
        self,
        tmp_path: Path,
        tmp_log_path: Path,
    ) -> None:
        """Idle watchdog should not trip when messages arrive in time."""
        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=5,
            idle_timeout_seconds=0.2,
            review_enabled=False,
        )
        slow_client = SlowSDKClient(delay=0.05)
        fake_factory = FakeSDKClientFactory(slow_client)

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input)

        assert output.success is True

    @pytest.mark.asyncio
    async def test_idle_timeout_disabled(
        self,
        tmp_path: Path,
        tmp_log_path: Path,
    ) -> None:
        """Idle watchdog can be disabled with idle_timeout_seconds=0."""
        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=5,
            idle_timeout_seconds=0,
            review_enabled=False,
        )
        slow_client = SlowSDKClient(delay=0.2)
        fake_factory = FakeSDKClientFactory(slow_client)

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input)

        assert output.success is True


class TestAgentSessionRunnerGateHandling:
    """Test gate checking behavior."""

    @pytest.fixture
    def tmp_log_path(self, tmp_path: Path) -> Path:
        """Create a temporary log file path."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")
        return log_path

    @pytest.fixture
    def session_config(self, tmp_path: Path) -> AgentSessionConfig:
        """Create a session config for testing."""
        return AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=3,
            max_review_retries=2,
            morph_enabled=False,
            review_enabled=False,
        )

    @pytest.mark.asyncio
    async def test_gate_passed_completes_successfully(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """Runner should complete successfully when gate passes."""
        fake_client = FakeSDKClient()
        fake_factory = FakeSDKClientFactory(fake_client)

        gate_check_calls: list[str] = []

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            gate_check_calls.append(issue_id)
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input)

        assert output.success is True
        assert len(gate_check_calls) == 1
        assert gate_check_calls[0] == "test-123"

    @pytest.mark.asyncio
    async def test_gate_failed_no_retries_fails(
        self,
        tmp_path: Path,
        tmp_log_path: Path,
    ) -> None:
        """Runner should fail when gate fails and max retries exhausted."""
        # Set max_gate_retries to 1 so we fail immediately
        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=1,
            morph_enabled=False,
            review_enabled=False,
        )

        fake_client = FakeSDKClient()
        fake_factory = FakeSDKClientFactory(fake_client)

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(
                    passed=False,
                    failure_reasons=["Missing commit"],
                    commit_hash=None,
                ),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input)

        assert output.success is False
        assert "Quality gate failed" in output.summary


class TestAgentSessionRunnerCallbacks:
    """Test callback invocation."""

    @pytest.fixture
    def tmp_log_path(self, tmp_path: Path) -> Path:
        """Create a temporary log file path."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")
        return log_path

    @pytest.fixture
    def session_config(self, tmp_path: Path) -> AgentSessionConfig:
        """Create a session config for testing."""
        return AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=3,
            max_review_retries=2,
            morph_enabled=False,
            review_enabled=False,
        )

    @pytest.mark.asyncio
    async def test_get_log_path_callback_invoked(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """Runner should call get_log_path callback with session ID."""
        fake_client = FakeSDKClient()
        fake_factory = FakeSDKClientFactory(fake_client)

        log_path_calls: list[str] = []

        def get_log_path(session_id: str) -> Path:
            log_path_calls.append(session_id)
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        await runner.run_session(input)

        assert len(log_path_calls) == 1
        assert log_path_calls[0] == "test-session-123"

    @pytest.mark.asyncio
    async def test_raises_when_callback_missing(
        self,
        session_config: AgentSessionConfig,
    ) -> None:
        """Runner should raise when required callback is missing."""
        fake_client = FakeSDKClient()
        fake_factory = FakeSDKClientFactory(fake_client)

        # No callbacks configured
        callbacks = SessionCallbacks()

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        # Should fail because get_log_path is not set
        output = await runner.run_session(input)
        # The error should be caught and reported in summary
        assert output.success is False
        assert "get_log_path" in output.summary


class TestAgentSessionRunnerConfig:
    """Test configuration handling."""

    def test_config_with_custom_values(self, tmp_path: Path) -> None:
        """Config should accept custom values via flags."""
        config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=120,
            max_gate_retries=5,
            max_review_retries=4,
            morph_enabled=True,
            morph_api_key="test-key",
            review_enabled=False,
        )

        assert config.timeout_seconds == 120
        assert config.max_gate_retries == 5
        assert config.max_review_retries == 4
        assert config.morph_enabled is True
        assert config.morph_api_key == "test-key"
        assert config.review_enabled is False


class TestAgentSessionInput:
    """Test input data handling."""

    def test_input_required_fields(self) -> None:
        """Input should require issue_id and prompt."""
        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        assert input.issue_id == "test-123"
        assert input.prompt == "Test prompt"
        assert input.baseline_commit is None
        assert input.issue_description is None

    def test_input_with_optional_fields(self) -> None:
        """Input should accept optional fields."""
        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
            baseline_commit="abc123",
            issue_description="Fix the bug",
        )

        assert input.baseline_commit == "abc123"
        assert input.issue_description == "Fix the bug"


class TestAgentSessionOutput:
    """Test output data handling."""

    def test_output_required_fields(self) -> None:
        """Output should have required fields with defaults."""
        from src.pipeline.agent_session_runner import AgentSessionOutput

        output = AgentSessionOutput(
            success=True,
            summary="Done",
        )

        assert output.success is True
        assert output.summary == "Done"
        assert output.session_id is None
        assert output.log_path is None
        assert output.gate_attempts == 1
        assert output.review_attempts == 0
        assert output.resolution is None
        assert output.duration_seconds == 0.0
        assert output.agent_id == ""


class TestAgentSessionRunnerStreamingCallbacks:
    """Test streaming event callbacks (on_tool_use, on_agent_text)."""

    @pytest.fixture
    def tmp_log_path(self, tmp_path: Path) -> Path:
        """Create a temporary log file path."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")
        return log_path

    @pytest.fixture
    def session_config(self, tmp_path: Path) -> AgentSessionConfig:
        """Create a session config for testing."""
        return AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=3,
            max_review_retries=2,
            morph_enabled=False,
            review_enabled=False,
        )

    @pytest.mark.asyncio
    async def test_on_tool_use_callback_invoked(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """Runner should invoke on_tool_use callback for ToolUseBlock messages."""
        # Create messages with a ToolUseBlock
        tool_block = ToolUseBlock(id="tool-1", name="Read", input={"path": "test.py"})
        assistant_msg = AssistantMessage(content=[tool_block], model="test-model")

        fake_client = FakeSDKClient(messages=[assistant_msg])
        fake_factory = FakeSDKClientFactory(fake_client)

        tool_use_calls: list[tuple[str, str, dict[str, Any] | None]] = []

        def on_tool_use(
            agent_id: str, tool_name: str, arguments: dict[str, Any] | None
        ) -> None:
            tool_use_calls.append((agent_id, tool_name, arguments))

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
            on_tool_use=on_tool_use,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        await runner.run_session(input)

        assert len(tool_use_calls) == 1
        assert tool_use_calls[0] == ("test-123", "Read", {"path": "test.py"})

    @pytest.mark.asyncio
    async def test_on_agent_text_callback_invoked(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """Runner should invoke on_agent_text callback for TextBlock messages."""
        # Create messages with a TextBlock
        text_block = TextBlock(text="Processing the request...")
        assistant_msg = AssistantMessage(content=[text_block], model="test-model")

        fake_client = FakeSDKClient(messages=[assistant_msg])
        fake_factory = FakeSDKClientFactory(fake_client)

        agent_text_calls: list[tuple[str, str]] = []

        def on_agent_text(agent_id: str, text: str) -> None:
            agent_text_calls.append((agent_id, text))

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
            on_agent_text=on_agent_text,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        await runner.run_session(input)

        assert len(agent_text_calls) == 1
        assert agent_text_calls[0] == ("test-123", "Processing the request...")

    @pytest.mark.asyncio
    async def test_streaming_callbacks_optional(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """Runner should work without streaming callbacks (optional)."""
        # Create messages with both TextBlock and ToolUseBlock
        text_block = TextBlock(text="Working...")
        tool_block = ToolUseBlock(id="tool-1", name="Read", input={"path": "test.py"})
        assistant_msg = AssistantMessage(
            content=[text_block, tool_block], model="test-model"
        )

        fake_client = FakeSDKClient(messages=[assistant_msg])
        fake_factory = FakeSDKClientFactory(fake_client)

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        # No on_tool_use or on_agent_text callbacks
        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        # Should succeed without raising errors
        output = await runner.run_session(input)
        assert output.success is True


class FakeEventSink:
    """Fake event sink for testing that records all event calls."""

    def __init__(self) -> None:
        self.events: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, *args: object, **kwargs: object) -> None:
        self.events.append((name, args, dict(kwargs)))

    def on_validation_started(self, agent_id: str, issue_id: str | None = None) -> None:
        self._record("on_validation_started", agent_id, issue_id=issue_id)

    def on_validation_result(
        self,
        agent_id: str,
        passed: bool,
        issue_id: str | None = None,
    ) -> None:
        self._record("on_validation_result", agent_id, passed=passed, issue_id=issue_id)

    def on_gate_started(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "on_gate_started", agent_id, attempt, max_attempts, issue_id=issue_id
        )

    def on_gate_passed(self, agent_id: str | None, issue_id: str | None = None) -> None:
        self._record("on_gate_passed", agent_id, issue_id=issue_id)

    def on_gate_failed(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "on_gate_failed", agent_id, attempt, max_attempts, issue_id=issue_id
        )

    def on_gate_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "on_gate_retry", agent_id, attempt, max_attempts, issue_id=issue_id
        )

    def on_gate_result(
        self,
        agent_id: str | None,
        passed: bool,
        failure_reasons: list[str] | None = None,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "on_gate_result",
            agent_id,
            passed=passed,
            failure_reasons=failure_reasons,
            issue_id=issue_id,
        )

    def on_review_started(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "on_review_started", agent_id, attempt, max_attempts, issue_id=issue_id
        )

    def on_review_passed(self, agent_id: str, issue_id: str | None = None) -> None:
        self._record("on_review_passed", agent_id, issue_id=issue_id)

    def on_review_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        error_count: int | None = None,
        parse_error: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "on_review_retry",
            agent_id,
            attempt,
            max_attempts,
            error_count=error_count,
            parse_error=parse_error,
            issue_id=issue_id,
        )

    # Pipeline module events
    def on_lifecycle_state(self, agent_id: str, state: str) -> None:
        self._record("on_lifecycle_state", agent_id, state)

    def on_log_waiting(self, agent_id: str) -> None:
        self._record("on_log_waiting", agent_id)

    def on_log_ready(self, agent_id: str) -> None:
        self._record("on_log_ready", agent_id)

    def on_log_timeout(self, agent_id: str, log_path: str) -> None:
        self._record("on_log_timeout", agent_id, log_path)

    def on_review_skipped_no_progress(self, agent_id: str) -> None:
        self._record("on_review_skipped_no_progress", agent_id)

    def on_warning(self, message: str, agent_id: str | None = None) -> None:
        self._record("on_warning", message, agent_id=agent_id)

    def on_fixer_text(self, attempt: int, text: str) -> None:
        self._record("on_fixer_text", attempt, text)

    def on_fixer_tool_use(
        self,
        attempt: int,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        self._record("on_fixer_tool_use", attempt, tool_name, arguments=arguments)


class TestAgentSessionRunnerEventSink:
    """Test event sink integration for gate/review events."""

    @pytest.fixture
    def tmp_log_path(self, tmp_path: Path) -> Path:
        """Create a temporary log file path."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")
        return log_path

    @pytest.fixture
    def session_config(self, tmp_path: Path) -> AgentSessionConfig:
        """Create a session config for testing."""
        return AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=3,
            max_review_retries=2,
            morph_enabled=False,
            review_enabled=False,
        )

    @pytest.mark.asyncio
    async def test_gate_passed_emits_sink_events(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """Runner should emit on_gate_started and on_gate_passed when gate passes."""
        fake_client = FakeSDKClient()
        fake_factory = FakeSDKClientFactory(fake_client)
        fake_sink = FakeEventSink()

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input)

        assert output.success is True

        # Check that gate events were emitted
        event_names = [e[0] for e in fake_sink.events]
        assert "on_validation_started" in event_names
        assert "on_gate_started" in event_names
        assert "on_gate_passed" in event_names
        assert "on_validation_result" in event_names

        # Verify on_validation_started is called before on_gate_started
        validation_started_idx = event_names.index("on_validation_started")
        gate_started_idx = event_names.index("on_gate_started")
        assert validation_started_idx < gate_started_idx, (
            "on_validation_started should be emitted before on_gate_started"
        )

        # Verify on_gate_started was called with correct args
        gate_started = next(e for e in fake_sink.events if e[0] == "on_gate_started")
        assert gate_started[1][0] == "test-123"  # agent_id
        assert gate_started[1][1] == 1  # attempt
        assert gate_started[1][2] == 3  # max_attempts

        # Verify on_gate_passed was called with correct agent_id
        gate_passed = next(e for e in fake_sink.events if e[0] == "on_gate_passed")
        assert gate_passed[1][0] == "test-123"

        # Verify on_validation_result was called with passed=True
        validation_result = next(
            e for e in fake_sink.events if e[0] == "on_validation_result"
        )
        assert validation_result[1][0] == "test-123"  # agent_id
        assert validation_result[2]["passed"] is True

    @pytest.mark.asyncio
    async def test_gate_failed_emits_sink_events(
        self,
        tmp_path: Path,
        tmp_log_path: Path,
    ) -> None:
        """Runner should emit on_gate_started, on_gate_failed, and on_gate_result when gate fails."""
        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=1,  # Fail immediately
            morph_enabled=False,
            review_enabled=False,
        )

        fake_client = FakeSDKClient()
        fake_factory = FakeSDKClientFactory(fake_client)
        fake_sink = FakeEventSink()

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(
                    passed=False,
                    failure_reasons=["Missing commit", "Tests failed"],
                    commit_hash=None,
                ),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input)

        assert output.success is False

        # Check that gate events were emitted
        event_names = [e[0] for e in fake_sink.events]
        assert "on_validation_started" in event_names
        assert "on_gate_started" in event_names
        assert "on_gate_failed" in event_names
        assert "on_gate_result" in event_names
        assert "on_validation_result" in event_names

        # Verify on_gate_failed was called with correct args
        gate_failed = next(e for e in fake_sink.events if e[0] == "on_gate_failed")
        assert gate_failed[1][0] == "test-123"  # agent_id
        assert gate_failed[1][1] == 1  # attempt
        assert gate_failed[1][2] == 1  # max_attempts

        # Verify on_gate_result was called with failure reasons
        gate_result = next(e for e in fake_sink.events if e[0] == "on_gate_result")
        assert gate_result[1][0] == "test-123"  # agent_id
        assert gate_result[2]["passed"] is False
        assert "Missing commit" in gate_result[2]["failure_reasons"]
        assert "Tests failed" in gate_result[2]["failure_reasons"]

        # Verify on_validation_result was called with passed=False
        validation_result = next(
            e for e in fake_sink.events if e[0] == "on_validation_result"
        )
        assert validation_result[1][0] == "test-123"  # agent_id
        assert validation_result[2]["passed"] is False

    @pytest.mark.asyncio
    async def test_gate_retry_emits_sink_events(
        self,
        tmp_path: Path,
        tmp_log_path: Path,
    ) -> None:
        """Runner should emit on_gate_retry when gate fails and retries are available."""
        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=2,  # Allow 1 retry
            morph_enabled=False,
            review_enabled=False,
        )

        fake_client = FakeSDKClient()
        fake_factory = FakeSDKClientFactory(fake_client)
        fake_sink = FakeEventSink()

        gate_check_count = 0

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            nonlocal gate_check_count
            gate_check_count += 1
            if gate_check_count == 1:
                # First check fails, triggers retry
                return (
                    GateResult(
                        passed=False,
                        failure_reasons=["Tests failed"],
                        commit_hash=None,
                    ),
                    1000,
                )
            else:
                # Second check passes
                return (
                    GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                    2000,
                )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input)

        assert output.success is True

        # Check that gate retry event was emitted
        event_names = [e[0] for e in fake_sink.events]
        assert "on_gate_started" in event_names
        assert "on_gate_retry" in event_names
        assert "on_gate_result" in event_names  # For the failed attempt
        assert "on_gate_passed" in event_names

        # Verify on_gate_retry was called
        gate_retry = next(e for e in fake_sink.events if e[0] == "on_gate_retry")
        assert gate_retry[1][0] == "test-123"  # agent_id

    @pytest.mark.asyncio
    async def test_no_sink_events_when_sink_is_none(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """Runner should work without event sink (sink is None)."""
        fake_client = FakeSDKClient()
        fake_factory = FakeSDKClientFactory(fake_client)

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
            event_sink=None,  # No event sink
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        # Should succeed without errors
        output = await runner.run_session(input)
        assert output.success is True

    @pytest.mark.asyncio
    async def test_review_passed_emits_sink_events(
        self,
        tmp_path: Path,
        tmp_log_path: Path,
    ) -> None:
        """Runner should emit on_review_started and on_review_passed when review passes."""
        from src.infra.clients.cerberus_review import ReviewResult

        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=3,
            max_review_retries=2,
            morph_enabled=False,
            review_enabled=True,  # Enable review
        )

        fake_client = FakeSDKClient()
        fake_factory = FakeSDKClientFactory(fake_client)
        fake_sink = FakeEventSink()

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        captured_session_id: str | None = None

        async def on_review_check(
            issue_id: str,
            description: str | None,
            baseline: str | None,
            session_id: str | None,
            _retry_state: RetryState,
        ) -> ReviewResult:
            nonlocal captured_session_id
            captured_session_id = session_id
            return ReviewResult(passed=True, issues=[], parse_error=None)

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
            on_review_check=on_review_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input)

        assert output.success is True
        assert captured_session_id == "test-session-123"

        # Check that review events were emitted
        event_names = [e[0] for e in fake_sink.events]
        assert "on_review_started" in event_names
        assert "on_review_passed" in event_names

        # Verify gate passed events are emitted before review when review is enabled
        # This ensures every on_validation_started has a corresponding on_validation_result
        assert "on_gate_passed" in event_names
        assert "on_validation_result" in event_names

        # Verify ordering: gate_passed and validation_result before review_started
        gate_passed_idx = event_names.index("on_gate_passed")
        validation_result_idx = event_names.index("on_validation_result")
        review_started_idx = event_names.index("on_review_started")
        assert gate_passed_idx < review_started_idx, (
            "on_gate_passed should be emitted before on_review_started"
        )
        assert validation_result_idx < review_started_idx, (
            "on_validation_result should be emitted before on_review_started"
        )

        # Verify on_review_started was called with correct args
        review_started = next(
            e for e in fake_sink.events if e[0] == "on_review_started"
        )
        assert review_started[1][0] == "test-123"  # agent_id
        assert review_started[1][1] == 1  # attempt
        assert review_started[1][2] == 2  # max_attempts

    @pytest.mark.asyncio
    async def test_review_retry_emits_sink_events(
        self,
        tmp_path: Path,
        tmp_log_path: Path,
    ) -> None:
        """Runner should emit on_review_retry when review fails and retries are available."""
        from src.infra.clients.cerberus_review import ReviewIssue, ReviewResult

        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=3,
            max_review_retries=2,  # Allow 1 retry
            morph_enabled=False,
            review_enabled=True,  # Enable review
        )

        fake_client = FakeSDKClient()
        fake_factory = FakeSDKClientFactory(fake_client)
        fake_sink = FakeEventSink()

        review_check_count = 0

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        async def on_review_check(
            issue_id: str,
            description: str | None,
            baseline: str | None,
            session_id: str | None,
            _retry_state: RetryState,
        ) -> ReviewResult:
            nonlocal review_check_count
            review_check_count += 1
            if review_check_count == 1:
                # First check fails with errors
                return ReviewResult(
                    passed=False,
                    issues=[
                        ReviewIssue(
                            title="Bug found",
                            body="A bug was found in the code",
                            priority=1,
                            file="test.py",
                            line_start=10,
                            line_end=15,
                            reviewer="cerberus",
                        )
                    ],
                    parse_error=None,
                )
            else:
                # Second check passes
                return ReviewResult(passed=True, issues=[], parse_error=None)

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
            on_review_check=on_review_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        input = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input)

        assert output.success is True

        # Check that review retry event was emitted
        event_names = [e[0] for e in fake_sink.events]
        assert "on_review_started" in event_names
        assert "on_review_retry" in event_names
        assert "on_review_passed" in event_names

        # Verify on_review_retry was called with error_count
        review_retry = next(e for e in fake_sink.events if e[0] == "on_review_retry")
        assert review_retry[1][0] == "test-123"  # agent_id
        assert review_retry[2]["error_count"] == 1  # one P1 error


class TestIdleTimeoutRetry:
    """Test idle timeout retry behavior."""

    @pytest.fixture
    def tmp_log_path(self, tmp_path: Path) -> Path:
        """Create a temporary log file path."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")
        return log_path

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_idle_timeout_retries_and_recovers(
        self,
        tmp_path: Path,
        tmp_log_path: Path,
    ) -> None:
        """Hang once then succeed on retry; verify disconnect() and resume."""
        # First client: yields session_id via ResultMessage partial then hangs
        # We need a ResultMessage to provide session_id for resume
        hanging_result = make_result_message(session_id="hang-session-123")
        hanging_client = HangingAfterMessagesSDKClient(
            messages=[hanging_result],  # Yield result before hanging
        )

        # Second client: succeeds immediately
        success_client = FakeSDKClient()

        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            idle_timeout_seconds=0.01,
            max_idle_retries=2,
            idle_retry_backoff=(0.0, 0.0, 0.0),
            review_enabled=False,
        )

        factory = SequencedSDKClientFactory([hanging_client, success_client])

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=factory,
        )

        input_data = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input_data)

        # Verify success after retry
        assert output.success is True
        assert len(factory.create_calls) == 2

        # Verify disconnect was called on the hanging client
        assert hanging_client.disconnect_called is True

        # Verify second client used resume prompt with session_id
        assert len(success_client.queries) == 1
        assert "Continue on issue test-123" in success_client.queries[0][0]
        assert success_client.queries[0][1] == "hang-session-123"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_idle_timeout_gives_up_after_max_retries(
        self,
        tmp_path: Path,
    ) -> None:
        """Disconnect on all clients; fail after max retries."""
        # Create 3 hanging clients (initial + 2 retries)
        clients = [HangingSDKClient() for _ in range(3)]

        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            idle_timeout_seconds=0.01,
            max_idle_retries=2,
            idle_retry_backoff=(0.0, 0.0, 0.0),
            review_enabled=False,
        )

        factory = SequencedSDKClientFactory(clients)

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=SessionCallbacks(),
            sdk_client_factory=factory,
        )

        input_data = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input_data)

        # Verify failure after max retries
        assert output.success is False
        assert "idle" in output.summary.lower()

        # Verify disconnect was called on all clients
        for client in clients:
            assert client.disconnect_called is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_idle_timeout_always_disconnects_even_on_final_failure(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify disconnect() called even when max retries exceeded."""
        # Only 1 client since max_idle_retries=0 means no retries
        client = HangingSDKClient()

        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            idle_timeout_seconds=0.01,
            max_idle_retries=0,  # No retries allowed
            review_enabled=False,
        )

        factory = FakeSDKClientFactory(client)

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=SessionCallbacks(),
            sdk_client_factory=factory,
        )

        input_data = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input_data)

        # Verify failure
        assert output.success is False

        # Verify disconnect was still called
        assert client.disconnect_called is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_idle_timeout_first_turn_no_side_effects_starts_fresh(
        self,
        tmp_path: Path,
        tmp_log_path: Path,
    ) -> None:
        """Hang before any messages (no tool calls, no session_id); retry with original prompt."""
        # First client: hangs immediately (no messages, no session_id)
        hanging_client = HangingSDKClient()

        # Second client: succeeds
        success_client = FakeSDKClient()

        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            idle_timeout_seconds=0.01,
            max_idle_retries=2,
            idle_retry_backoff=(0.0, 0.0, 0.0),
            review_enabled=False,
        )

        factory = SequencedSDKClientFactory([hanging_client, success_client])

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=factory,
        )

        input_data = AgentSessionInput(
            issue_id="test-123",
            prompt="Original test prompt",
        )

        output = await runner.run_session(input_data)

        # Verify success
        assert output.success is True

        # Verify retry used the ORIGINAL prompt (fresh session), not resume prompt
        assert len(success_client.queries) == 1
        assert success_client.queries[0][0] == "Original test prompt"
        # No session_id for fresh session
        assert success_client.queries[0][1] is None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_idle_timeout_disabled_during_tool_execution(
        self,
        tmp_path: Path,
    ) -> None:
        """Hang after tool calls - no idle timeout during tool execution.

        When a ToolUseBlock is pending (waiting for result), idle timeout is
        disabled to allow long-running tool executions. The session timeout
        is the only backstop in this case.
        """
        # Create a client that yields tool calls then hangs
        tool_block = ToolUseBlock(id="tool-1", name="Bash", input={"command": "ls"})
        assistant_msg = AssistantMessage(content=[tool_block], model="test-model")

        hanging_client = HangingAfterMessagesSDKClient(messages=[assistant_msg])

        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=1,  # Short session timeout for test
            idle_timeout_seconds=0.01,  # Would fire instantly, but disabled during tool exec
            max_idle_retries=2,
            idle_retry_backoff=(0.0, 0.0, 0.0),
            review_enabled=False,
        )

        factory = FakeSDKClientFactory(hanging_client)

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=SessionCallbacks(),
            sdk_client_factory=factory,
        )

        input_data = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input_data)

        # Verify failure - session timeout, not idle timeout
        assert output.success is False
        # With idle timeout disabled during tool execution, session timeout fires
        assert "timeout" in output.summary.lower()

        # Verify only 1 client was created (no retry during tool execution)
        assert len(factory.create_calls) == 1

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_idle_timeout_non_query_phases_skip_query_block(
        self,
        tmp_path: Path,
        tmp_log_path: Path,
    ) -> None:
        """Verify non-query phases (like WAIT_FOR_LOG) don't create clients.

        This test verifies the loop structure: when pending_query is None,
        no client is created and no query is sent.
        """
        # Setup a successful client
        fake_client = FakeSDKClient()
        factory = FakeSDKClientFactory(fake_client)

        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            idle_timeout_seconds=0.01,
            review_enabled=False,
        )

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=factory,
        )

        input_data = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        output = await runner.run_session(input_data)

        # Verify success
        assert output.success is True

        # Verify only 1 client was created for the initial query
        # (no extra clients for WAIT_FOR_LOG or RUN_GATE phases)
        assert len(factory.create_calls) == 1

        # Verify only 1 query was sent (the initial one)
        assert len(fake_client.queries) == 1
        assert fake_client.queries[0][0] == "Test prompt"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_idle_timeout_backoff_delays(
        self,
        tmp_path: Path,
        tmp_log_path: Path,
    ) -> None:
        """Verify backoff delays are applied: 0s (retry 1), 5s (retry 2)."""
        # First two clients hang, third succeeds
        result_msg = make_result_message(session_id="session-123")
        hanging1 = HangingAfterMessagesSDKClient(messages=[result_msg])
        hanging2 = HangingAfterMessagesSDKClient(messages=[result_msg])
        success_client = FakeSDKClient()

        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            idle_timeout_seconds=0.01,
            max_idle_retries=2,
            idle_retry_backoff=(0.0, 5.0, 15.0),
            review_enabled=False,
        )

        factory = SequencedSDKClientFactory([hanging1, hanging2, success_client])

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=factory,
        )

        input_data = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        sleep_calls: list[float] = []
        original_sleep = asyncio.sleep

        async def mock_sleep(delay: float) -> None:
            sleep_calls.append(delay)
            # Use a very short real sleep to avoid infinite loops
            # but allow the hanging clients to timeout quickly
            if delay > 1:
                # This is a backoff delay - just record it
                await original_sleep(0.001)
            else:
                # This is likely part of the test infrastructure
                await original_sleep(delay)

        # Patch asyncio.sleep in the runner module specifically
        with patch(
            "src.pipeline.agent_session_runner.asyncio.sleep", side_effect=mock_sleep
        ):
            output = await runner.run_session(input_data)

        # Verify success after 2 retries
        assert output.success is True

        # Check backoff delays were used:
        # - Retry 1: backoff_idx = 0 -> 0.0s (but we don't sleep for 0)
        # - Retry 2: backoff_idx = 1 -> 5.0s
        assert 5.0 in sleep_calls, f"Expected 5.0s backoff, got: {sleep_calls}"

    @pytest.mark.asyncio
    async def test_gate_retry_resets_log_offset_when_session_changes(
        self,
        tmp_path: Path,
    ) -> None:
        """Regression test: log_offset must reset when session file changes.

        When Claude SDK creates a new session file for a gate retry (instead of
        appending to the original), the stale log_offset from the old session
        must be reset to 0. Otherwise, seeking to an offset larger than the new
        file's size yields no entries → "Missing validation evidence".

        This was a real bug: mala-iy6l.6 failed because log_offset was ~4MB
        (from old session) but new session file was only 34KB.
        """
        # Create two separate log files to simulate session change
        log_path_1 = tmp_path / "session1.jsonl"
        log_path_2 = tmp_path / "session2.jsonl"
        log_path_1.write_text("")
        log_path_2.write_text("")

        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=2,
            morph_enabled=False,
            review_enabled=False,
        )

        # Track which session IDs we return
        session_ids = ["session1", "session2"]

        def make_client_for_session(session_id: str) -> FakeSDKClient:
            return FakeSDKClient(
                result_message=make_result_message(session_id=session_id)
            )

        # Factory that returns clients with different session IDs
        class SessionChangingFactory:
            def __init__(self) -> None:
                self.create_calls: list[Any] = []
                self.idx = 0

            def create(self, options: object) -> SDKClientProtocol:
                self.create_calls.append(options)
                sid = session_ids[min(self.idx, len(session_ids) - 1)]
                self.idx += 1
                return cast("SDKClientProtocol", make_client_for_session(sid))

        fake_factory = SessionChangingFactory()

        # Track log_offset values passed to gate check
        gate_log_offsets: list[int] = []

        def get_log_path(session_id: str) -> Path:
            if session_id == "session1":
                return log_path_1
            return log_path_2

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            gate_log_offsets.append(retry_state.log_offset)

            if len(gate_log_offsets) == 1:
                # First check: fail and return large offset (simulating big log)
                return (
                    GateResult(
                        passed=False,
                        failure_reasons=["Tests failed"],
                        commit_hash=None,
                    ),
                    4_000_000,  # 4MB offset - larger than any real new session file
                )
            else:
                # Second check: verify offset was reset, then pass
                return (
                    GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                    5000,
                )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,  # type: ignore[arg-type]
        )

        input_data = AgentSessionInput(
            issue_id="test-offset-reset",
            prompt="Test prompt",
        )

        output = await runner.run_session(input_data)

        assert output.success is True
        assert len(gate_log_offsets) == 2

        # First gate check should have offset 0 (initial)
        assert gate_log_offsets[0] == 0

        # Second gate check MUST have offset 0 (reset due to session change)
        # Before the fix, this would be 4_000_000 (stale from first session)
        assert gate_log_offsets[1] == 0, (
            f"log_offset should be reset to 0 when session changes, "
            f"but got {gate_log_offsets[1]}"
        )


class TestInitializeSession:
    """Unit tests for _initialize_session helper.

    Tests the session initialization logic that creates lifecycle,
    hooks, SDK options, and mutable state for a session.
    """

    @pytest.fixture
    def session_config(self, tmp_path: Path) -> AgentSessionConfig:
        """Create a session config for testing."""
        return AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=600,
            max_gate_retries=3,
            max_review_retries=2,
            morph_enabled=False,
            review_enabled=True,
        )

    @pytest.fixture
    def runner(self, session_config: AgentSessionConfig) -> AgentSessionRunner:
        """Create a runner for testing initialization."""
        return AgentSessionRunner(
            config=session_config,
            callbacks=SessionCallbacks(),
        )

    @pytest.mark.unit
    def test_initialize_session_creates_agent_id(
        self,
        runner: AgentSessionRunner,
    ) -> None:
        """_initialize_session should create agent_id from issue_id and uuid."""
        input_data = AgentSessionInput(
            issue_id="test-issue-123",
            prompt="Test prompt",
        )

        session_cfg, _ = runner._initialize_session(input_data)

        # Agent ID should start with issue_id and have a UUID suffix
        assert session_cfg.agent_id.startswith("test-issue-123-")
        # UUID suffix is 8 hex chars
        suffix = session_cfg.agent_id.split("-")[-1]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)

    @pytest.mark.unit
    def test_initialize_session_creates_lifecycle(
        self,
        runner: AgentSessionRunner,
    ) -> None:
        """_initialize_session should create lifecycle with correct config."""
        input_data = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        _, state = runner._initialize_session(input_data)

        # Lifecycle should be in INITIAL state
        from src.domain.lifecycle import LifecycleState

        assert state.lifecycle.state == LifecycleState.INITIAL
        assert state.lifecycle.config.max_gate_retries == 3
        assert state.lifecycle.config.max_review_retries == 2
        assert state.lifecycle.config.review_enabled is True

    @pytest.mark.unit
    def test_initialize_session_creates_lifecycle_context(
        self,
        runner: AgentSessionRunner,
    ) -> None:
        """_initialize_session should create lifecycle context with baseline timestamp."""
        input_data = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        _, state = runner._initialize_session(input_data)

        # Context should have baseline_timestamp set
        assert state.lifecycle_ctx.retry_state.baseline_timestamp > 0
        # Other retry state fields should be at defaults
        assert state.lifecycle_ctx.retry_state.gate_attempt == 1
        assert state.lifecycle_ctx.retry_state.review_attempt == 0
        assert state.lifecycle_ctx.retry_state.log_offset == 0

    @pytest.mark.unit
    def test_initialize_session_creates_lint_cache(
        self,
        runner: AgentSessionRunner,
    ) -> None:
        """_initialize_session should create lint cache with repo path."""
        input_data = AgentSessionInput(
            issue_id="test-123",
            prompt="Test prompt",
        )

        session_cfg, _ = runner._initialize_session(input_data)

        # Lint cache should be created
        assert session_cfg.lint_cache is not None
        # LintCache stores repo_path as _repo_path (private)
        assert session_cfg.lint_cache._repo_path == runner.config.repo_path

    @pytest.mark.unit
    def test_initialize_session_computes_idle_timeout(
        self,
        tmp_path: Path,
    ) -> None:
        """_initialize_session should compute idle timeout from session timeout."""
        # Test default idle timeout calculation: min(900, max(300, timeout * 0.2))
        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=3000,  # 50 min -> derived = 600
            idle_timeout_seconds=None,  # Let it be computed
        )
        runner = AgentSessionRunner(config=session_config, callbacks=SessionCallbacks())

        input_data = AgentSessionInput(issue_id="test-123", prompt="Test")
        session_cfg, _ = runner._initialize_session(input_data)

        # 3000 * 0.2 = 600, clamped to [300, 900] = 600
        assert session_cfg.idle_timeout_seconds == 600.0

    @pytest.mark.unit
    def test_initialize_session_respects_idle_timeout_minimum(
        self,
        tmp_path: Path,
    ) -> None:
        """_initialize_session should clamp idle timeout to minimum 300s."""
        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=600,  # 10 min -> derived = 120 < 300
            idle_timeout_seconds=None,
        )
        runner = AgentSessionRunner(config=session_config, callbacks=SessionCallbacks())

        input_data = AgentSessionInput(issue_id="test-123", prompt="Test")
        session_cfg, _ = runner._initialize_session(input_data)

        # 600 * 0.2 = 120, clamped to min 300
        assert session_cfg.idle_timeout_seconds == 300.0

    @pytest.mark.unit
    def test_initialize_session_respects_idle_timeout_maximum(
        self,
        tmp_path: Path,
    ) -> None:
        """_initialize_session should clamp idle timeout to maximum 900s."""
        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=6000,  # 100 min -> derived = 1200 > 900
            idle_timeout_seconds=None,
        )
        runner = AgentSessionRunner(config=session_config, callbacks=SessionCallbacks())

        input_data = AgentSessionInput(issue_id="test-123", prompt="Test")
        session_cfg, _ = runner._initialize_session(input_data)

        # 6000 * 0.2 = 1200, clamped to max 900
        assert session_cfg.idle_timeout_seconds == 900.0

    @pytest.mark.unit
    def test_initialize_session_disables_idle_timeout_with_zero(
        self,
        tmp_path: Path,
    ) -> None:
        """_initialize_session should disable idle timeout when set to 0."""
        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=600,
            idle_timeout_seconds=0,  # Explicitly disabled
        )
        runner = AgentSessionRunner(config=session_config, callbacks=SessionCallbacks())

        input_data = AgentSessionInput(issue_id="test-123", prompt="Test")
        session_cfg, _ = runner._initialize_session(input_data)

        # 0 means disabled -> None
        assert session_cfg.idle_timeout_seconds is None

    @pytest.mark.unit
    def test_initialize_session_uses_explicit_idle_timeout(
        self,
        tmp_path: Path,
    ) -> None:
        """_initialize_session should use explicit idle timeout if provided."""
        session_config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=600,
            idle_timeout_seconds=42.5,  # Explicit value
        )
        runner = AgentSessionRunner(config=session_config, callbacks=SessionCallbacks())

        input_data = AgentSessionInput(issue_id="test-123", prompt="Test")
        session_cfg, _ = runner._initialize_session(input_data)

        assert session_cfg.idle_timeout_seconds == 42.5


class TestBuildSessionOutput:
    """Unit tests for _build_session_output helper.

    Tests the output building logic that constructs AgentSessionOutput
    from session configuration and execution state.
    """

    @pytest.fixture
    def session_config(self, tmp_path: Path) -> AgentSessionConfig:
        """Create a session config for testing."""
        return AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=600,
            max_gate_retries=3,
            max_review_retries=2,
        )

    @pytest.fixture
    def runner(self, session_config: AgentSessionConfig) -> AgentSessionRunner:
        """Create a runner for testing output building."""
        return AgentSessionRunner(
            config=session_config,
            callbacks=SessionCallbacks(),
        )

    @pytest.mark.unit
    def test_build_session_output_success(
        self,
        runner: AgentSessionRunner,
        tmp_path: Path,
    ) -> None:
        """_build_session_output should build output for successful session."""
        from src.pipeline.agent_session_runner import (
            SessionConfig,
            SessionExecutionState,
        )
        from src.infra.hooks import LintCache
        from src.domain.lifecycle import (
            ImplementerLifecycle,
            LifecycleConfig,
            LifecycleContext,
        )

        session_cfg = SessionConfig(
            agent_id="test-123-abc12345",
            options={},
            lint_cache=LintCache(repo_path=tmp_path),
            log_file_wait_timeout=60.0,
            log_file_poll_interval=0.5,
            idle_timeout_seconds=300.0,
        )

        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle_ctx = LifecycleContext()
        lifecycle_ctx.success = True
        lifecycle_ctx.retry_state.gate_attempt = 1
        lifecycle_ctx.retry_state.review_attempt = 1

        log_path = tmp_path / "session.jsonl"
        state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
            session_id="session-xyz",
            log_path=log_path,
            final_result="Done successfully",
            cerberus_review_log_path="/path/to/review.log",
        )

        output = runner._build_session_output(session_cfg, state, duration=123.45)

        assert output.success is True
        assert output.summary == "Done successfully"
        assert output.session_id == "session-xyz"
        assert output.log_path == log_path
        assert output.gate_attempts == 1
        assert output.review_attempts == 1
        assert output.duration_seconds == 123.45
        assert output.agent_id == "test-123-abc12345"
        assert output.review_log_path == "/path/to/review.log"

    @pytest.mark.unit
    def test_build_session_output_failure(
        self,
        runner: AgentSessionRunner,
        tmp_path: Path,
    ) -> None:
        """_build_session_output should build output for failed session."""
        from src.pipeline.agent_session_runner import (
            SessionConfig,
            SessionExecutionState,
        )
        from src.infra.hooks import LintCache
        from src.domain.lifecycle import (
            ImplementerLifecycle,
            LifecycleConfig,
            LifecycleContext,
        )

        session_cfg = SessionConfig(
            agent_id="test-fail-abc",
            options={},
            lint_cache=LintCache(repo_path=tmp_path),
            log_file_wait_timeout=60.0,
            log_file_poll_interval=0.5,
            idle_timeout_seconds=None,
        )

        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle_ctx = LifecycleContext()
        lifecycle_ctx.success = False
        lifecycle_ctx.retry_state.gate_attempt = 3
        lifecycle_ctx.retry_state.review_attempt = 0

        state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
            session_id=None,  # No session ID received
            log_path=None,
            final_result="Quality gate failed: Missing commit",
        )

        output = runner._build_session_output(session_cfg, state, duration=45.0)

        assert output.success is False
        assert output.summary == "Quality gate failed: Missing commit"
        assert output.session_id is None
        assert output.log_path is None
        assert output.gate_attempts == 3
        assert output.review_attempts == 0
        assert output.duration_seconds == 45.0
        assert output.agent_id == "test-fail-abc"

    @pytest.mark.unit
    def test_build_session_output_with_resolution(
        self,
        runner: AgentSessionRunner,
        tmp_path: Path,
    ) -> None:
        """_build_session_output should include resolution if present."""
        from src.pipeline.agent_session_runner import (
            SessionConfig,
            SessionExecutionState,
        )
        from src.infra.hooks import LintCache
        from src.domain.lifecycle import (
            ImplementerLifecycle,
            LifecycleConfig,
            LifecycleContext,
        )
        from src.domain.validation.spec import IssueResolution, ResolutionOutcome

        session_cfg = SessionConfig(
            agent_id="test-res-abc",
            options={},
            lint_cache=LintCache(repo_path=tmp_path),
            log_file_wait_timeout=60.0,
            log_file_poll_interval=0.5,
            idle_timeout_seconds=None,
        )

        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle_ctx = LifecycleContext()
        lifecycle_ctx.success = True
        lifecycle_ctx.resolution = IssueResolution(
            outcome=ResolutionOutcome.NO_CHANGE,
            rationale="Already fixed",
        )

        state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
            session_id="session-res",
            final_result="ISSUE_NO_CHANGE: Already fixed",
        )

        output = runner._build_session_output(session_cfg, state, duration=10.0)

        assert output.success is True
        assert output.resolution is not None
        assert output.resolution.outcome == ResolutionOutcome.NO_CHANGE
        assert output.resolution.rationale == "Already fixed"

    @pytest.mark.unit
    def test_build_session_output_with_low_priority_issues(
        self,
        runner: AgentSessionRunner,
        tmp_path: Path,
    ) -> None:
        """_build_session_output should include low priority review issues."""
        from src.pipeline.agent_session_runner import (
            SessionConfig,
            SessionExecutionState,
        )
        from src.infra.hooks import LintCache
        from src.domain.lifecycle import (
            ImplementerLifecycle,
            LifecycleConfig,
            LifecycleContext,
        )
        from src.infra.clients.cerberus_review import ReviewIssue

        session_cfg = SessionConfig(
            agent_id="test-low-abc",
            options={},
            lint_cache=LintCache(repo_path=tmp_path),
            log_file_wait_timeout=60.0,
            log_file_poll_interval=0.5,
            idle_timeout_seconds=None,
        )

        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle_ctx = LifecycleContext()
        lifecycle_ctx.success = True
        lifecycle_ctx.low_priority_review_issues = [
            ReviewIssue(
                title="Minor style issue",
                body="Consider improving style",
                priority=2,
                file="test.py",
                line_start=10,
                line_end=15,
                reviewer="gemini",
            ),
        ]

        state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
            session_id="session-low",
            final_result="Review passed",
        )

        output = runner._build_session_output(session_cfg, state, duration=20.0)

        assert output.success is True
        assert output.low_priority_review_issues is not None
        assert len(output.low_priority_review_issues) == 1
        assert output.low_priority_review_issues[0].title == "Minor style issue"


class TestEmitReviewResultEvents:
    """Unit tests for _emit_review_result_events helper.

    Tests the isolated event emission logic extracted from _handle_review_effect.
    """

    @pytest.fixture
    def fake_sink(self) -> FakeEventSink:
        """Create a fake event sink for testing."""
        return FakeEventSink()

    @pytest.fixture
    def input_data(self) -> AgentSessionInput:
        """Create test input."""
        return AgentSessionInput(issue_id="test-123", prompt="Test")

    @pytest.mark.unit
    def test_complete_success_emits_review_passed(
        self, fake_sink: FakeEventSink, input_data: AgentSessionInput
    ) -> None:
        """COMPLETE_SUCCESS should emit on_review_passed event."""
        from src.domain.lifecycle import Effect, LifecycleState, TransitionResult
        from src.infra.clients.cerberus_review import ReviewResult
        from src.pipeline.agent_session_runner import _emit_review_result_events

        result = TransitionResult(
            state=LifecycleState.RUNNING_REVIEW, effect=Effect.COMPLETE_SUCCESS
        )
        review_result = ReviewResult(passed=True, issues=[], parse_error=None)
        lifecycle_ctx = MagicMock()
        lifecycle_ctx.retry_state.review_attempt = 1

        _emit_review_result_events(
            fake_sink,  # type: ignore[arg-type]
            input_data,
            result,
            review_result,
            lifecycle_ctx,
            max_review_retries=3,
            blocking_count=0,
        )

        event_names = [e[0] for e in fake_sink.events]
        assert "on_review_passed" in event_names
        review_passed = next(e for e in fake_sink.events if e[0] == "on_review_passed")
        assert review_passed[1][0] == "test-123"

    @pytest.mark.unit
    def test_run_review_emits_warning(
        self, fake_sink: FakeEventSink, input_data: AgentSessionInput
    ) -> None:
        """RUN_REVIEW (parse error) should emit on_warning event."""
        from src.domain.lifecycle import Effect, LifecycleState, TransitionResult
        from src.infra.clients.cerberus_review import ReviewResult
        from src.pipeline.agent_session_runner import _emit_review_result_events

        result = TransitionResult(
            state=LifecycleState.RUNNING_REVIEW, effect=Effect.RUN_REVIEW
        )
        review_result = ReviewResult(
            passed=False, issues=[], parse_error="JSON decode error"
        )
        lifecycle_ctx = MagicMock()
        lifecycle_ctx.retry_state.review_attempt = 1

        _emit_review_result_events(
            fake_sink,  # type: ignore[arg-type]
            input_data,
            result,
            review_result,
            lifecycle_ctx,
            max_review_retries=3,
            blocking_count=0,
        )

        event_names = [e[0] for e in fake_sink.events]
        assert "on_warning" in event_names
        warning = next(e for e in fake_sink.events if e[0] == "on_warning")
        assert "JSON decode error" in warning[1][0]

    @pytest.mark.unit
    def test_send_review_retry_emits_retry_event(
        self, fake_sink: FakeEventSink, input_data: AgentSessionInput
    ) -> None:
        """SEND_REVIEW_RETRY should emit on_review_retry with blocking count."""
        from src.domain.lifecycle import Effect, LifecycleState, TransitionResult
        from src.infra.clients.cerberus_review import ReviewIssue, ReviewResult
        from src.pipeline.agent_session_runner import _emit_review_result_events

        result = TransitionResult(
            state=LifecycleState.RUNNING_REVIEW, effect=Effect.SEND_REVIEW_RETRY
        )
        # 2 blocking issues (priority 0 and 1), 1 non-blocking (priority 2)
        review_result = ReviewResult(
            passed=False,
            issues=[
                ReviewIssue(
                    title="Critical",
                    body="desc",
                    reviewer="test",
                    file="f.py",
                    line_start=1,
                    line_end=1,
                    priority=0,
                ),
                ReviewIssue(
                    title="Blocking",
                    body="desc",
                    reviewer="test",
                    file="f.py",
                    line_start=2,
                    line_end=2,
                    priority=1,
                ),
                ReviewIssue(
                    title="Minor",
                    body="desc",
                    reviewer="test",
                    file="f.py",
                    line_start=3,
                    line_end=3,
                    priority=2,
                ),
            ],
            parse_error=None,
        )
        lifecycle_ctx = MagicMock()
        lifecycle_ctx.retry_state.review_attempt = 2

        _emit_review_result_events(
            fake_sink,  # type: ignore[arg-type]
            input_data,
            result,
            review_result,
            lifecycle_ctx,
            max_review_retries=3,
            blocking_count=2,
        )

        event_names = [e[0] for e in fake_sink.events]
        assert "on_review_retry" in event_names
        retry_event = next(e for e in fake_sink.events if e[0] == "on_review_retry")
        assert retry_event[1][0] == "test-123"  # agent_id
        assert retry_event[1][1] == 2  # attempt
        assert retry_event[1][2] == 3  # max_attempts
        assert retry_event[2]["error_count"] == 2  # 2 blocking issues

    @pytest.mark.unit
    def test_no_events_when_sink_is_none(self, input_data: AgentSessionInput) -> None:
        """Should not raise when event_sink is None."""
        from src.domain.lifecycle import Effect, LifecycleState, TransitionResult
        from src.infra.clients.cerberus_review import ReviewResult
        from src.pipeline.agent_session_runner import _emit_review_result_events

        result = TransitionResult(
            state=LifecycleState.RUNNING_REVIEW, effect=Effect.COMPLETE_SUCCESS
        )
        review_result = ReviewResult(passed=True, issues=[], parse_error=None)
        lifecycle_ctx = MagicMock()

        # Should not raise
        _emit_review_result_events(
            None, input_data, result, review_result, lifecycle_ctx, 3, 0
        )

    @pytest.mark.unit
    def test_complete_failure_emits_no_events(
        self, fake_sink: FakeEventSink, input_data: AgentSessionInput
    ) -> None:
        """COMPLETE_FAILURE should not emit any events."""
        from src.domain.lifecycle import Effect, LifecycleState, TransitionResult
        from src.infra.clients.cerberus_review import ReviewResult
        from src.pipeline.agent_session_runner import _emit_review_result_events

        result = TransitionResult(
            state=LifecycleState.RUNNING_REVIEW, effect=Effect.COMPLETE_FAILURE
        )
        review_result = ReviewResult(passed=False, issues=[], parse_error=None)
        lifecycle_ctx = MagicMock()

        _emit_review_result_events(
            fake_sink,  # type: ignore[arg-type]
            input_data,
            result,
            review_result,
            lifecycle_ctx,
            max_review_retries=3,
            blocking_count=0,
        )

        assert len(fake_sink.events) == 0


class TestHandleGateCheck:
    """Unit tests for _handle_gate_check helper.

    Tests the gate check handling logic including event emission
    and retry prompt generation.
    """

    @pytest.fixture
    def tmp_log_path(self, tmp_path: Path) -> Path:
        """Create a temporary log file path."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")
        return log_path

    @pytest.fixture
    def session_config(self, tmp_path: Path) -> AgentSessionConfig:
        """Create a session config for testing."""
        return AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=600,
            max_gate_retries=3,
            max_review_retries=2,
            review_enabled=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_gate_check_passed(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """_handle_gate_check should emit events and return no retry on pass."""
        from src.pipeline.agent_session_runner import (
            SessionExecutionState,
        )
        from src.domain.lifecycle import (
            ImplementerLifecycle,
            LifecycleConfig,
            LifecycleContext,
            LifecycleState,
        )

        fake_sink = FakeEventSink()

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(on_gate_check=on_gate_check)
        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        lifecycle = ImplementerLifecycle(LifecycleConfig(review_enabled=False))
        lifecycle.start()
        lifecycle._state = LifecycleState.RUNNING_GATE
        lifecycle_ctx = LifecycleContext()

        state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
            session_id="session-abc",
            log_path=tmp_log_path,
        )

        input_data = AgentSessionInput(issue_id="test-gate", prompt="Test")

        pending_query, _ = await runner._handle_gate_check(
            input_data, state, lifecycle, lifecycle_ctx
        )

        # No retry needed
        assert pending_query is None
        # Gate passed events emitted
        event_names = [e[0] for e in fake_sink.events]
        assert "on_validation_started" in event_names
        assert "on_gate_started" in event_names
        assert "on_gate_passed" in event_names
        assert "on_validation_result" in event_names

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_gate_check_failed_with_retry(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """_handle_gate_check should return retry query on failure with retries left."""
        from src.pipeline.agent_session_runner import (
            SessionExecutionState,
        )
        from src.domain.lifecycle import (
            ImplementerLifecycle,
            LifecycleConfig,
            LifecycleContext,
            LifecycleState,
        )

        fake_sink = FakeEventSink()

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(
                    passed=False,
                    failure_reasons=["Tests failed"],
                    commit_hash=None,
                ),
                1000,
            )

        callbacks = SessionCallbacks(on_gate_check=on_gate_check)
        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        lifecycle = ImplementerLifecycle(LifecycleConfig(max_gate_retries=3))
        lifecycle.start()
        lifecycle._state = LifecycleState.RUNNING_GATE
        lifecycle_ctx = LifecycleContext()

        state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
            session_id="session-retry",
            log_path=tmp_log_path,
        )

        input_data = AgentSessionInput(issue_id="test-retry", prompt="Test")

        pending_query, _ = await runner._handle_gate_check(
            input_data, state, lifecycle, lifecycle_ctx
        )

        # Retry query should be generated
        assert pending_query is not None
        assert "Tests failed" in pending_query
        # Retry event emitted
        event_names = [e[0] for e in fake_sink.events]
        assert "on_gate_retry" in event_names
        assert "on_validation_result" in event_names

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_gate_check_raises_without_callback(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """_handle_gate_check should raise ValueError if callback not set."""
        from src.pipeline.agent_session_runner import (
            SessionExecutionState,
        )
        from src.domain.lifecycle import (
            ImplementerLifecycle,
            LifecycleConfig,
            LifecycleContext,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=SessionCallbacks(),  # No on_gate_check
        )

        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle_ctx = LifecycleContext()

        state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
            log_path=tmp_log_path,
        )

        input_data = AgentSessionInput(issue_id="test-err", prompt="Test")

        with pytest.raises(ValueError, match="on_gate_check callback must be set"):
            await runner._handle_gate_check(input_data, state, lifecycle, lifecycle_ctx)


class TestHandleReviewCheck:
    """Unit tests for _handle_review_check helper.

    Tests the review check handling logic including event emission
    and retry prompt generation.
    """

    @pytest.fixture
    def tmp_log_path(self, tmp_path: Path) -> Path:
        """Create a temporary log file path."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")
        return log_path

    @pytest.fixture
    def session_config(self, tmp_path: Path) -> AgentSessionConfig:
        """Create a session config for testing."""
        return AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=600,
            max_gate_retries=3,
            max_review_retries=3,
            review_enabled=True,
        )

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_review_check_passed(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """_handle_review_check should emit events and return no retry on pass."""
        from src.pipeline.agent_session_runner import (
            SessionExecutionState,
        )
        from src.domain.lifecycle import (
            ImplementerLifecycle,
            LifecycleConfig,
            LifecycleContext,
            LifecycleState,
        )
        from src.infra.clients.cerberus_review import ReviewResult

        fake_sink = FakeEventSink()

        async def on_review_check(
            issue_id: str,
            description: str | None,
            baseline: str | None,
            session_id: str | None,
            retry_state: RetryState,
        ) -> ReviewResult:
            return ReviewResult(passed=True, issues=[], parse_error=None)

        callbacks = SessionCallbacks(on_review_check=on_review_check)
        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        lifecycle = ImplementerLifecycle(
            LifecycleConfig(review_enabled=True, max_review_retries=3)
        )
        lifecycle.start()
        lifecycle._state = LifecycleState.RUNNING_REVIEW
        lifecycle_ctx = LifecycleContext()
        lifecycle_ctx.retry_state.review_attempt = 1

        state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
            session_id="session-review",
            log_path=tmp_log_path,
        )

        input_data = AgentSessionInput(issue_id="test-review", prompt="Test")

        pending_query, _ = await runner._handle_review_check(
            input_data, state, lifecycle, lifecycle_ctx
        )

        # No retry needed
        assert pending_query is None
        # Review passed events emitted
        event_names = [e[0] for e in fake_sink.events]
        # First review attempt emits gate_passed
        assert "on_gate_passed" in event_names
        assert "on_review_started" in event_names
        assert "on_review_passed" in event_names

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_review_check_failed_with_retry(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """_handle_review_check should return retry query on failure with retries left."""
        from src.pipeline.agent_session_runner import (
            SessionExecutionState,
        )
        from src.domain.lifecycle import (
            ImplementerLifecycle,
            LifecycleConfig,
            LifecycleContext,
            LifecycleState,
        )
        from src.infra.clients.cerberus_review import ReviewIssue, ReviewResult

        fake_sink = FakeEventSink()

        async def on_review_check(
            issue_id: str,
            description: str | None,
            baseline: str | None,
            session_id: str | None,
            retry_state: RetryState,
        ) -> ReviewResult:
            return ReviewResult(
                passed=False,
                issues=[
                    ReviewIssue(
                        title="Bug found",
                        body="A bug was found",
                        priority=1,  # P1 = blocking
                        file="test.py",
                        line_start=10,
                        line_end=15,
                        reviewer="gemini",
                    ),
                ],
                parse_error=None,
            )

        callbacks = SessionCallbacks(on_review_check=on_review_check)
        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        lifecycle = ImplementerLifecycle(
            LifecycleConfig(review_enabled=True, max_review_retries=3)
        )
        lifecycle.start()
        lifecycle._state = LifecycleState.RUNNING_REVIEW
        lifecycle_ctx = LifecycleContext()
        lifecycle_ctx.retry_state.review_attempt = 1

        state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
            session_id="session-rev-retry",
            log_path=tmp_log_path,
        )

        input_data = AgentSessionInput(issue_id="test-rev-retry", prompt="Test")

        pending_query, _ = await runner._handle_review_check(
            input_data, state, lifecycle, lifecycle_ctx
        )

        # Retry query should be generated
        assert pending_query is not None
        assert "Bug found" in pending_query or "test.py" in pending_query
        # Retry event emitted
        event_names = [e[0] for e in fake_sink.events]
        assert "on_review_retry" in event_names

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_review_check_skips_on_no_progress(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """_handle_review_check should skip review on no-progress detection."""
        from src.pipeline.agent_session_runner import (
            SessionExecutionState,
        )
        from src.domain.lifecycle import (
            ImplementerLifecycle,
            LifecycleConfig,
            LifecycleContext,
            LifecycleState,
        )
        from src.domain.quality_gate import GateResult

        fake_sink = FakeEventSink()

        # on_review_check should NOT be called due to no-progress
        review_check_called = False

        async def on_review_check(
            issue_id: str,
            description: str | None,
            baseline: str | None,
            session_id: str | None,
            retry_state: RetryState,
        ) -> None:
            nonlocal review_check_called
            review_check_called = True
            raise AssertionError("Should not be called")

        def on_review_no_progress(
            log_path: Path,
            log_offset: int,
            prev_commit: str | None,
            curr_commit: str | None,
        ) -> bool:
            return True  # No progress detected

        callbacks = SessionCallbacks(
            on_review_check=on_review_check,  # type: ignore[arg-type]
            on_review_no_progress=on_review_no_progress,
        )
        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        lifecycle = ImplementerLifecycle(
            LifecycleConfig(review_enabled=True, max_review_retries=3)
        )
        lifecycle.start()
        lifecycle._state = LifecycleState.RUNNING_REVIEW
        lifecycle_ctx = LifecycleContext()
        lifecycle_ctx.retry_state.review_attempt = 2  # Not first attempt
        lifecycle_ctx.last_gate_result = GateResult(
            passed=True, failure_reasons=[], commit_hash="same-commit"
        )

        state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
            session_id="session-no-progress",
            log_path=tmp_log_path,
        )

        input_data = AgentSessionInput(issue_id="test-no-progress", prompt="Test")

        pending_query, _ = await runner._handle_review_check(
            input_data, state, lifecycle, lifecycle_ctx
        )

        # on_review_check should not have been called
        assert not review_check_called
        # No retry
        assert pending_query is None
        # Skip event emitted
        event_names = [e[0] for e in fake_sink.events]
        assert "on_review_skipped_no_progress" in event_names

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_review_check_raises_without_callback(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """_handle_review_check should raise ValueError if callback not set."""
        from src.pipeline.agent_session_runner import (
            SessionExecutionState,
        )
        from src.domain.lifecycle import (
            ImplementerLifecycle,
            LifecycleConfig,
            LifecycleContext,
            LifecycleState,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=SessionCallbacks(),  # No on_review_check
        )

        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle.start()
        lifecycle._state = LifecycleState.RUNNING_REVIEW
        lifecycle_ctx = LifecycleContext()
        lifecycle_ctx.retry_state.review_attempt = 1

        state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
            log_path=tmp_log_path,
        )

        input_data = AgentSessionInput(issue_id="test-err", prompt="Test")

        with pytest.raises(ValueError, match="on_review_check callback must be set"):
            await runner._handle_review_check(
                input_data, state, lifecycle, lifecycle_ctx
            )


class TestRunLifecycleLoop:
    """Unit tests for _run_lifecycle_loop helper.

    Tests the main lifecycle loop that drives message iteration,
    gate checks, and review checks.
    """

    @pytest.fixture
    def tmp_log_path(self, tmp_path: Path) -> Path:
        """Create a temporary log file path."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")
        return log_path

    @pytest.fixture
    def session_config(self, tmp_path: Path) -> AgentSessionConfig:
        """Create a session config for testing."""
        return AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=3,
            max_review_retries=2,
            review_enabled=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_lifecycle_loop_completes_on_gate_pass(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """_run_lifecycle_loop should complete when gate passes."""
        from src.pipeline.agent_session_runner import (
            SessionConfig,
            SessionExecutionState,
        )
        from src.infra.hooks import LintCache
        from src.domain.lifecycle import (
            ImplementerLifecycle,
            LifecycleConfig,
            LifecycleContext,
            LifecycleState,
        )

        fake_client = FakeSDKClient()
        fake_factory = FakeSDKClientFactory(fake_client)
        fake_sink = FakeEventSink()

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        session_cfg = SessionConfig(
            agent_id="test-loop-abc",
            options=runner._build_sdk_options([], [], {}),
            lint_cache=LintCache(repo_path=session_config.repo_path),
            log_file_wait_timeout=60.0,
            log_file_poll_interval=0.5,
            idle_timeout_seconds=None,
        )

        lifecycle = ImplementerLifecycle(LifecycleConfig(review_enabled=False))
        lifecycle_ctx = LifecycleContext()

        state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
        )

        input_data = AgentSessionInput(issue_id="test-loop", prompt="Test prompt")

        await runner._run_lifecycle_loop(input_data, session_cfg, state)

        # Should complete successfully
        assert lifecycle.state == LifecycleState.SUCCESS
        assert lifecycle_ctx.success is True
        assert state.session_id == "test-session-123"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_lifecycle_loop_fails_without_session_id(
        self,
        session_config: AgentSessionConfig,
    ) -> None:
        """_run_lifecycle_loop should fail if no session_id received."""
        from src.pipeline.agent_session_runner import (
            SessionConfig,
            SessionExecutionState,
        )
        from src.infra.hooks import LintCache
        from src.domain.lifecycle import (
            ImplementerLifecycle,
            LifecycleConfig,
            LifecycleContext,
            LifecycleState,
        )

        # Client that returns no session_id
        no_session_result = ResultMessage(
            subtype="result",
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            num_turns=1,
            session_id=None,  # type: ignore[arg-type]  # No session ID!
            result="Done",
        )
        fake_client = FakeSDKClient(result_message=no_session_result)
        fake_factory = FakeSDKClientFactory(fake_client)

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=SessionCallbacks(),
            sdk_client_factory=fake_factory,
        )

        session_cfg = SessionConfig(
            agent_id="test-no-session",
            options=runner._build_sdk_options([], [], {}),
            lint_cache=LintCache(repo_path=session_config.repo_path),
            log_file_wait_timeout=60.0,
            log_file_poll_interval=0.5,
            idle_timeout_seconds=None,
        )

        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle_ctx = LifecycleContext()

        state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
        )

        input_data = AgentSessionInput(issue_id="test-no-session", prompt="Test")

        await runner._run_lifecycle_loop(input_data, session_cfg, state)

        # Should fail due to missing session ID
        assert lifecycle.state == LifecycleState.FAILED
        assert lifecycle_ctx.success is False
        assert "No session ID" in state.final_result

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_run_lifecycle_loop_emits_lifecycle_state_events(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """_run_lifecycle_loop should emit lifecycle state change events."""
        from src.pipeline.agent_session_runner import (
            SessionConfig,
            SessionExecutionState,
        )
        from src.infra.hooks import LintCache
        from src.domain.lifecycle import (
            ImplementerLifecycle,
            LifecycleConfig,
            LifecycleContext,
        )

        fake_client = FakeSDKClient()
        fake_factory = FakeSDKClientFactory(fake_client)
        fake_sink = FakeEventSink()

        def get_log_path(session_id: str) -> Path:
            return tmp_log_path

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult, int]:
            return (
                GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
                1000,
            )

        callbacks = SessionCallbacks(
            get_log_path=get_log_path,
            on_gate_check=on_gate_check,
        )

        runner = AgentSessionRunner(
            config=session_config,
            callbacks=callbacks,
            sdk_client_factory=fake_factory,
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        session_cfg = SessionConfig(
            agent_id="test-events-abc",
            options=runner._build_sdk_options([], [], {}),
            lint_cache=LintCache(repo_path=session_config.repo_path),
            log_file_wait_timeout=60.0,
            log_file_poll_interval=0.5,
            idle_timeout_seconds=None,
        )

        lifecycle = ImplementerLifecycle(LifecycleConfig(review_enabled=False))
        lifecycle_ctx = LifecycleContext()

        state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
        )

        input_data = AgentSessionInput(issue_id="test-events", prompt="Test")

        await runner._run_lifecycle_loop(input_data, session_cfg, state)

        # Verify lifecycle state events were emitted
        lifecycle_events = [e for e in fake_sink.events if e[0] == "on_lifecycle_state"]
        # Should have at least: PROCESSING, AWAITING_LOG
        states_emitted = [e[1][1] for e in lifecycle_events]
        assert "PROCESSING" in states_emitted
        assert "AWAITING_LOG" in states_emitted
