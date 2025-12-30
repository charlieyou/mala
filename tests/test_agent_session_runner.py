"""Unit tests for AgentSessionRunner pipeline stage.

Tests the extracted SDK session execution logic using fake SDK clients,
without actual SDK/API dependencies.

This module uses the actual SDK types (ResultMessage, etc.) to ensure
isinstance checks work correctly in the runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import pytest

# Import SDK types that the runner uses for isinstance checks
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

from src.pipeline.agent_session_runner import (
    AgentSessionConfig,
    AgentSessionInput,
    AgentSessionRunner,
    SessionCallbacks,
)
from src.quality_gate import GateResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from src.lifecycle import RetryState
    from src.pipeline.agent_session_runner import SDKClientProtocol


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


class FakeSDKClientFactory:
    """Factory for creating fake SDK clients in tests."""

    def __init__(self, client: FakeSDKClient):
        self.client = client
        self.create_calls: list[Any] = []

    def create(self, options: object) -> SDKClientProtocol:
        self.create_calls.append(options)
        return self.client


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
            codex_review_enabled=False,  # Disable review for basic tests
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
            codex_review_enabled=False,
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
            codex_review_enabled=False,
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
            codex_review_enabled=False,
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

    def test_config_default_values(self, tmp_path: Path) -> None:
        """Config should have sensible defaults."""
        config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
        )

        assert config.max_gate_retries == 3
        assert config.max_review_retries == 2
        assert config.morph_enabled is False
        assert config.codex_review_enabled is True

    def test_config_with_custom_values(self, tmp_path: Path) -> None:
        """Config should accept custom values."""
        config = AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=120,
            max_gate_retries=5,
            max_review_retries=4,
            morph_enabled=True,
            morph_api_key="test-key",
            codex_review_enabled=False,
        )

        assert config.timeout_seconds == 120
        assert config.max_gate_retries == 5
        assert config.max_review_retries == 4
        assert config.morph_enabled is True
        assert config.morph_api_key == "test-key"
        assert config.codex_review_enabled is False


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
            codex_review_enabled=False,
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
