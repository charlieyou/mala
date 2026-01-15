"""Unit tests for MessageStreamProcessor.

Tests the extracted stream processing logic using fake SDK messages,
without actual SDK/API dependencies.

This module uses the actual SDK types (ResultMessage, etc.) to ensure
type name checks work correctly in the processor.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self

import pytest

# Import SDK types for realistic test messages
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from src.pipeline.message_stream_processor import (
    ContextPressureError,
    IdleTimeoutError,
    IdleTimeoutStream,
    MessageIterationState,
    MessageStreamProcessor,
    StreamProcessorCallbacks,
    StreamProcessorConfig,
)
from src.domain.lifecycle import LifecycleContext
from tests.fakes.lint_cache import FakeLintCache

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# --- Fixtures and helpers ---


def make_result_message(
    session_id: str = "test-session-123",
    result: str | None = "Test completed successfully",
    usage: dict[str, int] | None = None,
) -> ResultMessage:
    """Create a ResultMessage with the given fields."""
    msg = ResultMessage(
        subtype="result",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=1,
        session_id=session_id,
        result=result,
    )
    if usage is not None:
        # Set usage via attribute since it's optional in constructor
        object.__setattr__(msg, "usage", usage)
    return msg


def make_assistant_message(content: list[Any]) -> AssistantMessage:
    """Create an AssistantMessage with the given content blocks."""
    return AssistantMessage(content=content, model="test-model")


async def messages_to_stream(
    messages: list[Any], result: ResultMessage
) -> AsyncIterator[Any]:
    """Convert a list of messages to an async iterator."""
    for msg in messages:
        yield msg
    yield result


class FakeTracer:
    """Fake tracer for testing, satisfies TelemetrySpan protocol."""

    def __init__(self) -> None:
        self.logged_messages: list[Any] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        pass

    def log_input(self, prompt: str) -> None:
        pass

    def log_message(self, message: object) -> None:
        self.logged_messages.append(message)

    def set_success(self, success: bool) -> None:
        pass

    def set_error(self, error: str) -> None:
        pass


@pytest.fixture
def processor() -> MessageStreamProcessor:
    """Create a basic MessageStreamProcessor for testing."""
    return MessageStreamProcessor()


@pytest.fixture
def lint_cache() -> FakeLintCache:
    """Create a fake lint cache for testing."""
    return FakeLintCache()


@pytest.fixture
def lifecycle_ctx() -> LifecycleContext:
    """Create a real LifecycleContext for testing."""
    return LifecycleContext()


# --- IdleTimeoutStream tests ---


class TestIdleTimeoutStream:
    """Tests for IdleTimeoutStream wrapper."""

    @pytest.mark.asyncio
    async def test_stream_yields_messages_without_timeout(self) -> None:
        """Stream passes through messages when no timeout occurs."""

        async def gen() -> AsyncIterator[str]:
            yield "msg1"
            yield "msg2"

        stream = IdleTimeoutStream(gen(), timeout_seconds=10.0, pending_tool_ids=set())
        results = [msg async for msg in stream]
        assert results == ["msg1", "msg2"]

    @pytest.mark.asyncio
    async def test_stream_raises_on_timeout(self) -> None:
        """Stream raises IdleTimeoutError when timeout exceeded."""

        async def slow_gen() -> AsyncIterator[str]:
            await asyncio.sleep(0.5)
            yield "never"

        stream = IdleTimeoutStream(
            slow_gen(), timeout_seconds=0.01, pending_tool_ids=set()
        )
        with pytest.raises(IdleTimeoutError, match="idle for 0 seconds"):
            async for _ in stream:
                pass

    @pytest.mark.asyncio
    async def test_stream_disables_timeout_with_pending_tools(self) -> None:
        """Stream disables timeout when pending_tool_ids is non-empty."""

        async def slow_gen() -> AsyncIterator[str]:
            await asyncio.sleep(0.05)
            yield "msg"

        pending = {"tool-1"}
        stream = IdleTimeoutStream(
            slow_gen(), timeout_seconds=0.01, pending_tool_ids=pending
        )
        # Should not timeout because pending tools exist
        results = [msg async for msg in stream]
        assert results == ["msg"]

    @pytest.mark.asyncio
    async def test_stream_none_timeout_never_times_out(self) -> None:
        """Stream with None timeout never times out."""

        async def gen() -> AsyncIterator[str]:
            yield "fast"

        stream = IdleTimeoutStream(gen(), timeout_seconds=None, pending_tool_ids=set())
        results = [msg async for msg in stream]
        assert results == ["fast"]


# --- MessageStreamProcessor tests ---


class TestMessageStreamProcessorBasic:
    """Basic stream processing tests."""

    @pytest.mark.asyncio
    async def test_process_empty_stream_with_result(
        self, processor: MessageStreamProcessor, lifecycle_ctx: LifecycleContext
    ) -> None:
        """Processing stream with just ResultMessage succeeds."""
        result_msg = make_result_message(session_id="sess-abc")
        state = MessageIterationState()
        lint_cache = FakeLintCache()

        stream = messages_to_stream([], result_msg)
        result = await processor.process_stream(
            stream,
            issue_id="test-issue",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=lint_cache,
            query_start=0.0,
            tracer=None,
        )

        assert result.success is True
        assert result.session_id == "sess-abc"
        assert state.session_id == "sess-abc"
        assert lifecycle_ctx.session_id == "sess-abc"

    @pytest.mark.asyncio
    async def test_process_text_block_invokes_callback(
        self, lifecycle_ctx: LifecycleContext
    ) -> None:
        """TextBlock triggers on_agent_text callback."""
        text_calls: list[tuple[str, str]] = []

        def on_text(issue_id: str, text: str) -> None:
            text_calls.append((issue_id, text))

        callbacks = StreamProcessorCallbacks(on_agent_text=on_text)
        processor = MessageStreamProcessor(callbacks=callbacks)

        text_block = TextBlock(text="Hello agent")
        assistant_msg = make_assistant_message([text_block])
        result_msg = make_result_message()
        state = MessageIterationState()

        stream = messages_to_stream([assistant_msg], result_msg)
        await processor.process_stream(
            stream,
            issue_id="issue-1",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=FakeLintCache(),
            query_start=0.0,
            tracer=None,
        )

        assert text_calls == [("issue-1", "Hello agent")]

    @pytest.mark.asyncio
    async def test_process_tool_use_block_invokes_callback(
        self, lifecycle_ctx: LifecycleContext
    ) -> None:
        """ToolUseBlock triggers on_tool_use callback and tracks pending."""
        tool_calls: list[tuple[str, str, dict[str, Any] | None]] = []

        def on_tool(issue_id: str, name: str, arguments: dict[str, Any] | None) -> None:
            tool_calls.append((issue_id, name, arguments))

        callbacks = StreamProcessorCallbacks(on_tool_use=on_tool)
        processor = MessageStreamProcessor(callbacks=callbacks)

        tool_block = ToolUseBlock(
            id="tool-abc",
            name="Read",
            input={"file_path": "/test.py"},
        )
        assistant_msg = make_assistant_message([tool_block])
        result_msg = make_result_message()
        state = MessageIterationState()

        stream = messages_to_stream([assistant_msg], result_msg)
        await processor.process_stream(
            stream,
            issue_id="issue-2",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=FakeLintCache(),
            query_start=0.0,
            tracer=None,
        )

        assert tool_calls == [("issue-2", "Read", {"file_path": "/test.py"})]
        assert state.tool_calls_this_turn == 1
        # Tool ID should be in pending until result received
        assert "tool-abc" in state.pending_tool_ids


class TestMessageStreamProcessorLintCache:
    """Tests for lint cache integration."""

    @pytest.mark.asyncio
    async def test_bash_lint_command_detected_and_cached(
        self, lifecycle_ctx: LifecycleContext
    ) -> None:
        """Bash lint commands are detected and cached on success."""
        lint_cache = FakeLintCache()
        lint_cache.configure_detect("ruff check .", "ruff")

        processor = MessageStreamProcessor()

        # ToolUseBlock for bash with lint command
        tool_block = ToolUseBlock(
            id="tool-lint",
            name="Bash",
            input={"command": "ruff check ."},
        )
        assistant_msg = make_assistant_message([tool_block])

        # Use real ToolResultBlock for accurate type name checking
        result_block = ToolResultBlock(
            tool_use_id="tool-lint",
            content="All checks passed",
            is_error=False,
        )

        # Create assistant message containing the result
        assistant_with_result = make_assistant_message([result_block])
        result_msg = make_result_message()
        state = MessageIterationState()

        stream = messages_to_stream([assistant_msg, assistant_with_result], result_msg)
        await processor.process_stream(
            stream,
            issue_id="issue-3",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=lint_cache,
            query_start=0.0,
            tracer=None,
        )

        # Lint command was detected
        assert ("detect", "ruff check .") in lint_cache.detected_commands
        # And marked as successful
        assert ("ruff", "ruff check .") in lint_cache.marked_successes

    @pytest.mark.asyncio
    async def test_bash_lint_error_not_cached(
        self, lifecycle_ctx: LifecycleContext
    ) -> None:
        """Bash lint commands with errors are not cached as success."""
        lint_cache = FakeLintCache()
        lint_cache.configure_detect("ruff check .", "ruff")

        processor = MessageStreamProcessor()

        tool_block = ToolUseBlock(
            id="tool-lint",
            name="Bash",
            input={"command": "ruff check ."},
        )
        assistant_msg = make_assistant_message([tool_block])

        # Use real ToolResultBlock with is_error=True
        result_block = ToolResultBlock(
            tool_use_id="tool-lint",
            content="Error: linting failed",
            is_error=True,
        )
        assistant_with_result = make_assistant_message([result_block])
        result_msg = make_result_message()
        state = MessageIterationState()

        stream = messages_to_stream([assistant_msg, assistant_with_result], result_msg)
        await processor.process_stream(
            stream,
            issue_id="issue-4",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=lint_cache,
            query_start=0.0,
            tracer=None,
        )

        # Lint command was detected
        assert ("detect", "ruff check .") in lint_cache.detected_commands
        # But NOT marked as successful due to error
        assert lint_cache.marked_successes == []


class TestMessageStreamProcessorContextPressure:
    """Tests for context pressure detection.

    Context pressure is calculated per-request (not accumulated) using:
    pressure = (input + cache_read + cache_creation) / limit

    Output tokens are excluded since prior outputs are reflected in future prompts.
    """

    @pytest.mark.asyncio
    async def test_context_pressure_raises_error(
        self, lifecycle_ctx: LifecycleContext
    ) -> None:
        """High context pressure raises ContextPressureError."""
        config = StreamProcessorConfig(
            context_limit=200_000,
            context_restart_threshold=0.90,
        )
        processor = MessageStreamProcessor(config=config)

        # Prompt tokens = 2 + 12K + 185K = 197K (98.5% of 200K limit)
        usage = {
            "input_tokens": 2,
            "output_tokens": 7_000,
            "cache_read_input_tokens": 12_000,
            "cache_creation_input_tokens": 185_000,
        }
        result_msg = make_result_message(session_id="sess-pressure", usage=usage)
        state = MessageIterationState()

        stream = messages_to_stream([], result_msg)
        with pytest.raises(ContextPressureError) as exc_info:
            await processor.process_stream(
                stream,
                issue_id="issue-pressure",
                state=state,
                lifecycle_ctx=lifecycle_ctx,
                lint_cache=FakeLintCache(),
                query_start=0.0,
                tracer=None,
            )

        err = exc_info.value
        assert err.session_id == "sess-pressure"
        assert err.input_tokens == 2
        assert err.pressure_ratio >= 0.90

    @pytest.mark.asyncio
    async def test_low_context_pressure_succeeds(
        self, lifecycle_ctx: LifecycleContext
    ) -> None:
        """Low context pressure completes successfully."""
        config = StreamProcessorConfig(
            context_limit=200_000,
            context_restart_threshold=0.90,
        )
        processor = MessageStreamProcessor(config=config)

        # Prompt tokens = 1K + 50K + 40K = 91K (45.5% of limit)
        usage = {
            "input_tokens": 1_000,
            "output_tokens": 10_000,
            "cache_read_input_tokens": 50_000,
            "cache_creation_input_tokens": 40_000,
        }
        result_msg = make_result_message(session_id="sess-ok", usage=usage)
        state = MessageIterationState()

        stream = messages_to_stream([], result_msg)
        result = await processor.process_stream(
            stream,
            issue_id="issue-ok",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=FakeLintCache(),
            query_start=0.0,
            tracer=None,
        )

        assert result.success is True
        assert lifecycle_ctx.context_usage.input_tokens == 1_000

    @pytest.mark.asyncio
    async def test_replaces_usage_per_request(
        self, lifecycle_ctx: LifecycleContext
    ) -> None:
        """Multiple ResultMessages replace (not accumulate) token values."""
        config = StreamProcessorConfig(
            context_limit=200_000,
            context_restart_threshold=0.90,
        )
        processor = MessageStreamProcessor(config=config)

        # First message
        usage1 = {
            "input_tokens": 10_000,
            "output_tokens": 5_000,
            "cache_read_input_tokens": 1_000,
            "cache_creation_input_tokens": 50_000,
        }
        result1 = make_result_message(session_id="sess-replace", usage=usage1)
        state = MessageIterationState()

        stream1 = messages_to_stream([], result1)
        await processor.process_stream(
            stream1,
            issue_id="issue-replace",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=FakeLintCache(),
            query_start=0.0,
            tracer=None,
        )

        assert lifecycle_ctx.context_usage.input_tokens == 10_000
        assert lifecycle_ctx.context_usage.output_tokens == 5_000
        assert lifecycle_ctx.context_usage.cache_read_tokens == 1_000
        assert lifecycle_ctx.context_usage.cache_creation_tokens == 50_000

        # Second message - replaces previous values
        usage2 = {
            "input_tokens": 2,
            "output_tokens": 8_000,
            "cache_read_input_tokens": 60_000,
            "cache_creation_input_tokens": 0,
        }
        result2 = make_result_message(session_id="sess-replace", usage=usage2)

        stream2 = messages_to_stream([], result2)
        await processor.process_stream(
            stream2,
            issue_id="issue-replace",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=FakeLintCache(),
            query_start=0.0,
            tracer=None,
        )

        # Values should be replaced, not accumulated
        assert lifecycle_ctx.context_usage.input_tokens == 2
        assert lifecycle_ctx.context_usage.output_tokens == 8_000
        assert lifecycle_ctx.context_usage.cache_read_tokens == 60_000
        assert lifecycle_ctx.context_usage.cache_creation_tokens == 0

    @pytest.mark.asyncio
    async def test_pressure_calculation_includes_cache_tokens(
        self, lifecycle_ctx: LifecycleContext
    ) -> None:
        """Pressure includes input + cache_read + cache_creation."""
        config = StreamProcessorConfig(
            context_limit=200_000,
            context_restart_threshold=0.90,
        )
        processor = MessageStreamProcessor(config=config)

        # Prompt = 5K + 80K + 100K = 185K (92.5% - triggers)
        usage = {
            "input_tokens": 5_000,
            "output_tokens": 50_000,  # Should NOT affect pressure
            "cache_read_input_tokens": 80_000,
            "cache_creation_input_tokens": 100_000,
        }
        result_msg = make_result_message(session_id="sess-cache", usage=usage)
        state = MessageIterationState()

        stream = messages_to_stream([], result_msg)
        with pytest.raises(ContextPressureError) as exc_info:
            await processor.process_stream(
                stream,
                issue_id="issue-cache",
                state=state,
                lifecycle_ctx=lifecycle_ctx,
                lint_cache=FakeLintCache(),
                query_start=0.0,
                tracer=None,
            )

        err = exc_info.value
        # 185K / 200K = 0.925
        assert err.pressure_ratio >= 0.90

    @pytest.mark.asyncio
    async def test_pressure_excludes_output_tokens(
        self, lifecycle_ctx: LifecycleContext
    ) -> None:
        """Output tokens should not affect pressure calculation."""
        config = StreamProcessorConfig(
            context_limit=200_000,
            context_restart_threshold=0.90,
        )
        processor = MessageStreamProcessor(config=config)

        # Prompt = 10K + 40K + 50K = 100K (50% - OK)
        # Even though output is 100K (would be 100% if counted)
        usage = {
            "input_tokens": 10_000,
            "output_tokens": 100_000,
            "cache_read_input_tokens": 40_000,
            "cache_creation_input_tokens": 50_000,
        }
        result_msg = make_result_message(session_id="sess-output", usage=usage)
        state = MessageIterationState()

        stream = messages_to_stream([], result_msg)
        result = await processor.process_stream(
            stream,
            issue_id="issue-output",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=FakeLintCache(),
            query_start=0.0,
            tracer=None,
        )

        # Should succeed - output tokens not counted in pressure
        assert result.success is True

    @pytest.mark.asyncio
    async def test_missing_usage_disables_tracking(
        self, lifecycle_ctx: LifecycleContext
    ) -> None:
        """ResultMessage without usage field disables tracking."""
        config = StreamProcessorConfig(
            context_limit=100_000,
            context_restart_threshold=0.90,
        )
        processor = MessageStreamProcessor(config=config)

        # Result with no usage field
        result_msg = make_result_message(session_id="sess-no-usage", usage=None)
        state = MessageIterationState()

        stream = messages_to_stream([], result_msg)
        result = await processor.process_stream(
            stream,
            issue_id="issue-no-usage",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=FakeLintCache(),
            query_start=0.0,
            tracer=None,
        )

        assert result.success is True
        assert lifecycle_ctx.context_usage.tracking_disabled is True


class TestMessageStreamProcessorTracer:
    """Tests for tracer integration."""

    @pytest.mark.asyncio
    async def test_tracer_logs_all_messages(
        self, lifecycle_ctx: LifecycleContext
    ) -> None:
        """Tracer receives all messages from stream."""
        processor = MessageStreamProcessor()
        tracer = FakeTracer()

        text_block = TextBlock(text="hello")
        assistant_msg = make_assistant_message([text_block])
        result_msg = make_result_message()
        state = MessageIterationState()

        stream = messages_to_stream([assistant_msg], result_msg)
        await processor.process_stream(
            stream,
            issue_id="issue-trace",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=FakeLintCache(),
            query_start=0.0,
            tracer=tracer,
        )

        assert len(tracer.logged_messages) == 2
        assert tracer.logged_messages[0] == assistant_msg
        assert tracer.logged_messages[1] == result_msg


class TestMessageIterationState:
    """Tests for MessageIterationState tracking."""

    @pytest.mark.asyncio
    async def test_first_message_received_flag(
        self, processor: MessageStreamProcessor, lifecycle_ctx: LifecycleContext
    ) -> None:
        """first_message_received is set on first message."""
        result_msg = make_result_message()
        state = MessageIterationState()

        assert state.first_message_received is False

        stream = messages_to_stream([], result_msg)
        await processor.process_stream(
            stream,
            issue_id="issue-first",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=FakeLintCache(),
            query_start=0.0,
            tracer=None,
        )

        assert state.first_message_received is True

    @pytest.mark.asyncio
    async def test_tool_calls_counter(
        self, processor: MessageStreamProcessor, lifecycle_ctx: LifecycleContext
    ) -> None:
        """tool_calls_this_turn counts ToolUseBlocks."""
        tool1 = ToolUseBlock(id="t1", name="Read", input={})
        tool2 = ToolUseBlock(id="t2", name="Write", input={})
        assistant_msg = make_assistant_message([tool1, tool2])
        result_msg = make_result_message()
        state = MessageIterationState()

        stream = messages_to_stream([assistant_msg], result_msg)
        await processor.process_stream(
            stream,
            issue_id="issue-count",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=FakeLintCache(),
            query_start=0.0,
            tracer=None,
        )

        assert state.tool_calls_this_turn == 2
        assert "t1" in state.pending_tool_ids
        assert "t2" in state.pending_tool_ids
