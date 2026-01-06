"""Unit tests for IdleTimeoutRetryPolicy.

Tests the idle timeout retry logic in isolation with mock SDK client,
stream processor, and lifecycle context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest

from src.domain.lifecycle import LifecycleContext
from src.pipeline.idle_retry_policy import (
    IdleTimeoutRetryPolicy,
    RetryConfig,
)
from src.pipeline.message_stream_processor import (
    IdleTimeoutError,
    MessageIterationResult,
    MessageIterationState,
)
from tests.fakes import FakeLintCache, FakeSDKClientFactory, FakeStreamProcessor

if TYPE_CHECKING:
    from src.infra.telemetry import TelemetrySpan
    from src.pipeline.message_stream_processor import (
        IdleTimeoutStream,
        LintCacheProtocol,
    )


# --- Fake/Mock helpers ---


@dataclass
class StateCaptureTracker:
    """Captures state values when process_stream is called.

    Wraps as a processor with a process_stream method.
    """

    captured_values: dict[str, object] = field(default_factory=dict)

    async def process_stream(
        self,
        stream: IdleTimeoutStream,
        issue_id: str,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
        lint_cache: LintCacheProtocol,
        query_start: float,
        tracer: TelemetrySpan | None,
    ) -> MessageIterationResult:
        self.captured_values["tool_calls_this_turn"] = state.tool_calls_this_turn
        self.captured_values["first_message_received"] = state.first_message_received
        self.captured_values["pending_tool_ids"] = state.pending_tool_ids.copy()
        self.captured_values["pending_lint_commands"] = (
            state.pending_lint_commands.copy()
        )
        return MessageIterationResult(success=True, session_id="sess")


@pytest.fixture
def sdk_factory() -> FakeSDKClientFactory:
    return FakeSDKClientFactory()


@pytest.fixture
def lint_cache() -> FakeLintCache:
    return FakeLintCache()


@pytest.fixture
def lifecycle_ctx() -> LifecycleContext:
    return LifecycleContext()


# --- RetryConfig validation tests ---


@pytest.mark.unit
class TestRetryConfigValidation:
    """Tests for RetryConfig __post_init__ validation."""

    def test_default_config_valid(self) -> None:
        """Default RetryConfig with zero retries should pass validation."""
        config = RetryConfig(max_idle_retries=0)
        assert config.max_idle_retries == 0
        assert config.idle_retry_backoff == (0.0, 5.0, 15.0)

    def test_valid_config_with_retries(self) -> None:
        """Valid config with retries requires idle_resume_prompt."""
        config = RetryConfig(
            max_idle_retries=2,
            idle_resume_prompt="Continue working on {issue_id}",
        )
        assert config.max_idle_retries == 2

    def test_negative_max_retries_raises(self) -> None:
        """Negative max_idle_retries should raise ValueError."""
        with pytest.raises(ValueError, match="max_idle_retries must be non-negative"):
            RetryConfig(max_idle_retries=-1)

    def test_empty_resume_prompt_with_retries_raises(self) -> None:
        """Empty idle_resume_prompt with positive max_idle_retries should raise."""
        with pytest.raises(
            ValueError,
            match=r"idle_resume_prompt must be non-empty when max_idle_retries > 0",
        ):
            RetryConfig(max_idle_retries=3, idle_resume_prompt="")

    def test_backoff_exact_length_valid(self) -> None:
        """Backoff with exactly max_idle_retries + 1 entries should be valid."""
        config = RetryConfig(
            max_idle_retries=2,
            idle_retry_backoff=(0.0, 5.0, 10.0),
            idle_resume_prompt="Continue {issue_id}",
        )
        assert config.max_idle_retries == 2
        assert len(config.idle_retry_backoff) == 3

    def test_backoff_longer_than_needed_valid(self) -> None:
        """Backoff with more entries than needed should be valid."""
        config = RetryConfig(
            max_idle_retries=1,
            idle_retry_backoff=(0.0, 5.0, 10.0, 20.0),
            idle_resume_prompt="Continue {issue_id}",
        )
        assert config.max_idle_retries == 1
        assert len(config.idle_retry_backoff) == 4

    def test_backoff_shorter_than_retries_valid(self) -> None:
        """Backoff shorter than max_idle_retries is valid (reuses last entry)."""
        config = RetryConfig(
            max_idle_retries=3,
            idle_retry_backoff=(0.0, 5.0),
            idle_resume_prompt="Continue {issue_id}",
        )
        assert config.max_idle_retries == 3
        assert len(config.idle_retry_backoff) == 2

    def test_zero_retries_with_single_backoff_valid(self) -> None:
        """Zero retries with a single backoff value should be valid."""
        config = RetryConfig(max_idle_retries=0, idle_retry_backoff=(0.0,))
        assert config.max_idle_retries == 0
        assert config.idle_retry_backoff == (0.0,)

    def test_zero_retries_with_empty_backoff_valid(self) -> None:
        """Zero retries with empty backoff should be valid (no retries needed)."""
        config = RetryConfig(max_idle_retries=0, idle_retry_backoff=())
        assert config.max_idle_retries == 0
        assert config.idle_retry_backoff == ()


# --- IdleTimeoutRetryPolicy tests ---


@pytest.mark.unit
class TestExecuteIterationSuccess:
    """Tests for successful iteration execution."""

    @pytest.mark.asyncio
    async def test_execute_iteration_success(
        self,
        sdk_factory: FakeSDKClientFactory,
        lifecycle_ctx: LifecycleContext,
        lint_cache: FakeLintCache,
    ) -> None:
        """Normal execution returns success with session ID."""
        sdk_factory.configure_next_client(responses=[])

        processor = FakeStreamProcessor(
            result=MessageIterationResult(
                success=True,
                session_id="test-session-123",
            )
        )

        config = RetryConfig(max_idle_retries=0)
        policy = IdleTimeoutRetryPolicy(
            sdk_client_factory=sdk_factory,
            stream_processor_factory=lambda: processor,
            config=config,
        )

        state = MessageIterationState()
        result = await policy.execute_iteration(
            query="Test query",
            issue_id="TEST-1",
            options={},
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=lint_cache,
            idle_timeout_seconds=300.0,
        )

        assert result.success is True
        assert result.session_id == "test-session-123"
        assert len(sdk_factory.clients) == 1
        assert sdk_factory.clients[0]._query_calls == [("Test query", None)]


@pytest.mark.unit
class TestExecuteIterationTimeout:
    """Tests for idle timeout handling."""

    @pytest.mark.asyncio
    async def test_execute_iteration_timeout_triggers_retry(
        self,
        sdk_factory: FakeSDKClientFactory,
        lifecycle_ctx: LifecycleContext,
        lint_cache: FakeLintCache,
    ) -> None:
        """Timeout triggers backoff and retry with resume prompt."""
        sdk_factory.configure_next_client(responses=[])
        sdk_factory.configure_next_client(responses=[])

        processor = FakeStreamProcessor(
            timeout_until=1,
            result=MessageIterationResult(success=True, session_id="resumed-session"),
        )

        config = RetryConfig(
            max_idle_retries=2,
            idle_retry_backoff=(0.0, 0.0),
            idle_resume_prompt="Continue {issue_id}",
        )
        policy = IdleTimeoutRetryPolicy(
            sdk_client_factory=sdk_factory,
            stream_processor_factory=lambda: processor,
            config=config,
        )

        state = MessageIterationState()
        lifecycle_ctx.session_id = "existing-session"

        result = await policy.execute_iteration(
            query="Initial query",
            issue_id="TEST-2",
            options={},
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=lint_cache,
            idle_timeout_seconds=300.0,
        )

        assert result.success is True
        assert result.session_id == "resumed-session"
        assert len(sdk_factory.clients) == 2
        assert sdk_factory.clients[0]._query_calls[0] == ("Initial query", None)
        assert sdk_factory.clients[1]._query_calls[0] == (
            "Continue TEST-2",
            None,
        )
        assert sdk_factory.clients[0]._disconnect_called is True
        assert state.idle_retry_count == 1
        assert len(sdk_factory.with_resume_calls) == 1
        assert sdk_factory.with_resume_calls[0][1] == "existing-session"
        options = cast("dict[str, Any]", sdk_factory.create_calls[1])
        assert options.get("resume") == "existing-session"

    @pytest.mark.asyncio
    async def test_execute_iteration_max_retries_exceeded(
        self,
        sdk_factory: FakeSDKClientFactory,
        lifecycle_ctx: LifecycleContext,
        lint_cache: FakeLintCache,
    ) -> None:
        """Max retries exceeded raises IdleTimeoutError."""
        for _ in range(3):
            sdk_factory.configure_next_client(responses=[])

        processor = FakeStreamProcessor(
            side_effect=IdleTimeoutError("idle for 300 seconds")
        )

        config = RetryConfig(
            max_idle_retries=2,
            idle_retry_backoff=(0.0, 0.0, 0.0),
            idle_resume_prompt="Continue {issue_id}",
        )
        policy = IdleTimeoutRetryPolicy(
            sdk_client_factory=sdk_factory,
            stream_processor_factory=lambda: processor,
            config=config,
        )

        state = MessageIterationState()
        lifecycle_ctx.session_id = "test-session"

        with pytest.raises(IdleTimeoutError, match="Max idle retries \\(2\\) exceeded"):
            await policy.execute_iteration(
                query="Test query",
                issue_id="TEST-3",
                options={},
                state=state,
                lifecycle_ctx=lifecycle_ctx,
                lint_cache=lint_cache,
                idle_timeout_seconds=300.0,
            )

        assert len(sdk_factory.clients) == 3
        assert state.idle_retry_count == 2


@pytest.mark.unit
class TestRetryConfigZeroMaxRetries:
    """Tests for RetryConfig with max_idle_retries=0."""

    @pytest.mark.asyncio
    async def test_retry_config_zero_max_retries_fails_immediately(
        self,
        sdk_factory: FakeSDKClientFactory,
        lifecycle_ctx: LifecycleContext,
        lint_cache: FakeLintCache,
    ) -> None:
        """max_idle_retries=0 causes immediate failure on first timeout."""
        sdk_factory.configure_next_client(responses=[])

        processor = FakeStreamProcessor(
            side_effect=IdleTimeoutError("idle for 300 seconds")
        )

        config = RetryConfig(max_idle_retries=0)
        policy = IdleTimeoutRetryPolicy(
            sdk_client_factory=sdk_factory,
            stream_processor_factory=lambda: processor,
            config=config,
        )

        state = MessageIterationState()

        with pytest.raises(IdleTimeoutError, match="Max idle retries \\(0\\) exceeded"):
            await policy.execute_iteration(
                query="Test query",
                issue_id="TEST-4",
                options={},
                state=state,
                lifecycle_ctx=lifecycle_ctx,
                lint_cache=lint_cache,
                idle_timeout_seconds=300.0,
            )

        assert len(sdk_factory.clients) == 1
        assert state.idle_retry_count == 0


@pytest.mark.unit
class TestBackoffTiming:
    """Tests for backoff timing behavior."""

    @pytest.mark.asyncio
    async def test_backoff_timing_uses_config(
        self,
        sdk_factory: FakeSDKClientFactory,
        lifecycle_ctx: LifecycleContext,
        lint_cache: FakeLintCache,
    ) -> None:
        """Backoff uses configured delays from idle_retry_backoff."""
        for _ in range(3):
            sdk_factory.configure_next_client(responses=[])

        sleep_calls: list[float] = []

        async def mock_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        processor = FakeStreamProcessor(
            timeout_until=2,
            result=MessageIterationResult(success=True, session_id="final-session"),
        )

        config = RetryConfig(
            max_idle_retries=2,
            idle_retry_backoff=(0.0, 5.0, 15.0),
            idle_resume_prompt="Continue {issue_id}",
        )
        policy = IdleTimeoutRetryPolicy(
            sdk_client_factory=sdk_factory,
            stream_processor_factory=lambda: processor,
            config=config,
        )

        state = MessageIterationState()
        lifecycle_ctx.session_id = "test-session"

        with patch(
            "src.pipeline.idle_retry_policy.asyncio.sleep", side_effect=mock_sleep
        ):
            result = await policy.execute_iteration(
                query="Test query",
                issue_id="TEST-5",
                options={},
                state=state,
                lifecycle_ctx=lifecycle_ctx,
                lint_cache=lint_cache,
                idle_timeout_seconds=300.0,
            )

        assert result.success is True
        # Retry 1 uses backoff[0] = 0.0 (no sleep since 0)
        # Retry 2 uses backoff[1] = 5.0 (only non-zero backoff triggers sleep)
        assert sleep_calls == [5.0]

    @pytest.mark.asyncio
    async def test_backoff_reuses_last_entry(
        self,
        sdk_factory: FakeSDKClientFactory,
        lifecycle_ctx: LifecycleContext,
        lint_cache: FakeLintCache,
    ) -> None:
        """Backoff reuses last entry when retry count exceeds backoff length."""
        for _ in range(4):
            sdk_factory.configure_next_client(responses=[])

        sleep_calls: list[float] = []

        async def mock_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        processor = FakeStreamProcessor(
            timeout_until=3,
            result=MessageIterationResult(success=True, session_id="final-session"),
        )

        config = RetryConfig(
            max_idle_retries=3,
            idle_retry_backoff=(1.0, 2.0),
            idle_resume_prompt="Continue {issue_id}",
        )
        policy = IdleTimeoutRetryPolicy(
            sdk_client_factory=sdk_factory,
            stream_processor_factory=lambda: processor,
            config=config,
        )

        state = MessageIterationState()
        lifecycle_ctx.session_id = "test-session"

        with patch(
            "src.pipeline.idle_retry_policy.asyncio.sleep", side_effect=mock_sleep
        ):
            result = await policy.execute_iteration(
                query="Test query",
                issue_id="TEST-6",
                options={},
                state=state,
                lifecycle_ctx=lifecycle_ctx,
                lint_cache=lint_cache,
                idle_timeout_seconds=300.0,
            )

        assert result.success is True
        # Retry 1: backoff[0] = 1.0
        # Retry 2: backoff[1] = 2.0
        # Retry 3: backoff[1] = 2.0 (reused)
        assert sleep_calls == [1.0, 2.0, 2.0]


@pytest.mark.unit
class TestRetryWithoutSessionId:
    """Tests for retry behavior without existing session ID."""

    @pytest.mark.asyncio
    async def test_retry_without_session_id_keeps_original_query(
        self,
        sdk_factory: FakeSDKClientFactory,
        lifecycle_ctx: LifecycleContext,
        lint_cache: FakeLintCache,
    ) -> None:
        """Retry without session_id and no tool calls keeps original query."""
        sdk_factory.configure_next_client(responses=[])
        sdk_factory.configure_next_client(responses=[])

        processor = FakeStreamProcessor(
            timeout_until=1,
            result=MessageIterationResult(success=True, session_id="new-session"),
        )

        config = RetryConfig(
            max_idle_retries=1,
            idle_retry_backoff=(0.0,),
            idle_resume_prompt="Continue {issue_id}",
        )
        policy = IdleTimeoutRetryPolicy(
            sdk_client_factory=sdk_factory,
            stream_processor_factory=lambda: processor,
            config=config,
        )

        state = MessageIterationState()

        result = await policy.execute_iteration(
            query="Original query",
            issue_id="TEST-7",
            options={},
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=lint_cache,
            idle_timeout_seconds=300.0,
        )

        assert result.success is True
        assert sdk_factory.clients[0]._query_calls[0] == ("Original query", None)
        assert sdk_factory.clients[1]._query_calls[0] == ("Original query", None)

    @pytest.mark.asyncio
    async def test_retry_with_tool_calls_no_session_raises(
        self,
        sdk_factory: FakeSDKClientFactory,
        lifecycle_ctx: LifecycleContext,
        lint_cache: FakeLintCache,
    ) -> None:
        """Retry with tool calls but no session_id raises IdleTimeoutError."""
        sdk_factory.configure_next_client(responses=[])

        class ToolCallModifyingProcessor:
            """Processor that modifies tool_calls_this_turn before raising timeout."""

            async def process_stream(
                self,
                stream: IdleTimeoutStream,
                issue_id: str,
                state: MessageIterationState,
                lifecycle_ctx: LifecycleContext,
                lint_cache: LintCacheProtocol,
                query_start: float,
                tracer: TelemetrySpan | None,
            ) -> MessageIterationResult:
                state.tool_calls_this_turn = 3
                raise IdleTimeoutError("idle timeout")

        processor = ToolCallModifyingProcessor()

        config = RetryConfig(
            max_idle_retries=2,
            idle_retry_backoff=(0.0,),
            idle_resume_prompt="Continue {issue_id}",
        )
        policy = IdleTimeoutRetryPolicy(
            sdk_client_factory=sdk_factory,
            stream_processor_factory=lambda: processor,
            config=config,
        )

        state = MessageIterationState()

        with pytest.raises(
            IdleTimeoutError, match="Cannot retry: 3 tool calls occurred"
        ):
            await policy.execute_iteration(
                query="Test query",
                issue_id="TEST-8",
                options={},
                state=state,
                lifecycle_ctx=lifecycle_ctx,
                lint_cache=lint_cache,
                idle_timeout_seconds=300.0,
            )


@pytest.mark.unit
class TestDisconnectBehavior:
    """Tests for client disconnect behavior on timeout."""

    @pytest.mark.asyncio
    async def test_disconnect_called_on_timeout(
        self,
        sdk_factory: FakeSDKClientFactory,
        lifecycle_ctx: LifecycleContext,
        lint_cache: FakeLintCache,
    ) -> None:
        """Client disconnect is called when timeout occurs."""
        sdk_factory.configure_next_client(responses=[])

        processor = FakeStreamProcessor(side_effect=IdleTimeoutError("idle timeout"))

        config = RetryConfig(max_idle_retries=0)
        policy = IdleTimeoutRetryPolicy(
            sdk_client_factory=sdk_factory,
            stream_processor_factory=lambda: processor,
            config=config,
        )

        state = MessageIterationState()

        with pytest.raises(IdleTimeoutError):
            await policy.execute_iteration(
                query="Test query",
                issue_id="TEST-9",
                options={},
                state=state,
                lifecycle_ctx=lifecycle_ctx,
                lint_cache=lint_cache,
                idle_timeout_seconds=300.0,
            )

        assert sdk_factory.clients[0]._disconnect_called is True


@pytest.mark.unit
class TestStateManagement:
    """Tests for state management during iteration."""

    @pytest.mark.asyncio
    async def test_state_reset_at_iteration_start(
        self,
        sdk_factory: FakeSDKClientFactory,
        lifecycle_ctx: LifecycleContext,
        lint_cache: FakeLintCache,
    ) -> None:
        """State fields are reset at the start of execute_iteration."""
        sdk_factory.configure_next_client(responses=[])

        tracker = StateCaptureTracker()

        config = RetryConfig(max_idle_retries=0)
        policy = IdleTimeoutRetryPolicy(
            sdk_client_factory=sdk_factory,
            stream_processor_factory=lambda: tracker,
            config=config,
        )

        state = MessageIterationState()
        state.tool_calls_this_turn = 5
        state.first_message_received = True
        state.pending_tool_ids = {"stale-tool-1", "stale-tool-2"}
        state.pending_lint_commands = {"stale-lint-1": ("ruff", "ruff check foo.py")}

        await policy.execute_iteration(
            query="Test query",
            issue_id="TEST-10",
            options={},
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=lint_cache,
            idle_timeout_seconds=300.0,
        )

        assert tracker.captured_values["tool_calls_this_turn"] == 0
        assert tracker.captured_values["first_message_received"] is False
        assert tracker.captured_values["pending_tool_ids"] == set()
        assert tracker.captured_values["pending_lint_commands"] == {}

    @pytest.mark.asyncio
    async def test_pending_lint_commands_cleared_on_idle_retry(
        self,
        sdk_factory: FakeSDKClientFactory,
        lifecycle_ctx: LifecycleContext,
        lint_cache: FakeLintCache,
    ) -> None:
        """pending_lint_commands are cleared when retrying after idle timeout."""
        # First attempt: times out; second: succeeds
        for _ in range(2):
            sdk_factory.configure_next_client(responses=[])

        class RetryStateCaptureProcessor:
            """Processor that captures state on retry."""

            def __init__(self) -> None:
                self.call_count = 0
                self.captured_on_retry: dict[str, object] = {}

            async def process_stream(
                self,
                stream: IdleTimeoutStream,
                issue_id: str,
                state: MessageIterationState,
                lifecycle_ctx: LifecycleContext,
                lint_cache: LintCacheProtocol,
                query_start: float,
                tracer: TelemetrySpan | None,
            ) -> MessageIterationResult:
                self.call_count += 1
                if self.call_count == 1:
                    # First attempt: populate stale lint commands, then timeout
                    state.pending_lint_commands["tool-abc"] = ("ruff", "ruff check .")
                    raise IdleTimeoutError("idle timeout")
                # Second attempt: capture state after retry preparation
                self.captured_on_retry["pending_lint_commands"] = (
                    state.pending_lint_commands.copy()
                )
                self.captured_on_retry["pending_tool_ids"] = (
                    state.pending_tool_ids.copy()
                )
                return MessageIterationResult(success=True, session_id="sess")

        processor = RetryStateCaptureProcessor()

        config = RetryConfig(
            max_idle_retries=1,
            idle_retry_backoff=(0.0,),
            idle_resume_prompt="Continue {issue_id}",
        )
        policy = IdleTimeoutRetryPolicy(
            sdk_client_factory=sdk_factory,
            stream_processor_factory=lambda: processor,
            config=config,
        )

        state = MessageIterationState()
        lifecycle_ctx.session_id = "test-session"

        await policy.execute_iteration(
            query="Test query",
            issue_id="TEST-11",
            options={},
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=lint_cache,
            idle_timeout_seconds=300.0,
        )

        assert processor.call_count == 2
        assert processor.captured_on_retry["pending_lint_commands"] == {}
        assert processor.captured_on_retry["pending_tool_ids"] == set()
