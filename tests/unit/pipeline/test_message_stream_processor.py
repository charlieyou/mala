"""Unit tests for MessageStreamProcessor.

Tests the extracted stream processing logic using ``AgentEvent``s,
without SDK/API dependencies. The processor consumes the coder-agnostic
event protocol; tests feed the corresponding dataclasses directly.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self

import pytest

from src.core.protocols.agent_event import (
    AgentResultEvent,
    AgentTextEvent,
    AgentToolResultEvent,
    AgentToolUseEvent,
)
from src.pipeline.message_stream_processor import (
    IdleTimeoutError,
    IdleTimeoutStream,
    MessageIterationState,
    MessageStreamProcessor,
    StreamProcessorCallbacks,
)
from src.domain.lifecycle import LifecycleContext
from tests.fakes.lint_cache import FakeLintCache

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# --- Fixtures and helpers ---


def make_result_event(
    session_id: str = "test-session-123",
    result: object = "Test completed successfully",
    subtype: str = "result",
    is_error: bool = False,
) -> AgentResultEvent:
    """Create a result AgentEvent for testing."""
    return AgentResultEvent(
        session_id=session_id,
        result=result,
        subtype=subtype,
        is_error=is_error,
    )


async def events_to_stream(events: list[Any]) -> AsyncIterator[Any]:
    """Convert a flat list of events to an async iterator."""
    for ev in events:
        yield ev


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
        """Stream with just a result event succeeds."""
        result_event = make_result_event(session_id="sess-abc")
        state = MessageIterationState()
        lint_cache = FakeLintCache()

        stream = events_to_stream([result_event])
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
    async def test_error_result_event_is_not_success(
        self, processor: MessageStreamProcessor, lifecycle_ctx: LifecycleContext
    ) -> None:
        """Amp error result events should not be treated as completed turns."""
        result_event = make_result_event(
            session_id="sess-error",
            result="error_during_execution",
        )
        state = MessageIterationState()

        result = await processor.process_stream(
            events_to_stream([result_event]),
            issue_id="test-issue",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=FakeLintCache(),
            query_start=0.0,
            tracer=None,
        )

        assert result.success is False
        assert result.session_id == "sess-error"
        assert result.error == "error_during_execution"
        assert lifecycle_ctx.session_id == "sess-error"

    @pytest.mark.asyncio
    async def test_error_subtype_with_human_result_is_not_success(
        self, processor: MessageStreamProcessor, lifecycle_ctx: LifecycleContext
    ) -> None:
        """Detect Amp shape where subtype classifies a human error result."""
        result_event = make_result_event(
            session_id="sess-error",
            result="Response incomplete: stream ended unexpectedly",
            subtype="error_during_execution",
            is_error=True,
        )
        state = MessageIterationState()

        result = await processor.process_stream(
            events_to_stream([result_event]),
            issue_id="test-issue",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=FakeLintCache(),
            query_start=0.0,
            tracer=None,
        )

        assert result.success is False
        assert result.session_id == "sess-error"
        assert result.error == "Response incomplete: stream ended unexpectedly"
        assert lifecycle_ctx.session_id == "sess-error"

    @pytest.mark.asyncio
    async def test_process_text_event_invokes_callback(
        self, lifecycle_ctx: LifecycleContext
    ) -> None:
        """AgentTextEvent triggers on_agent_text callback."""
        text_calls: list[tuple[str, str]] = []

        def on_text(issue_id: str, text: str) -> None:
            text_calls.append((issue_id, text))

        callbacks = StreamProcessorCallbacks(on_agent_text=on_text)
        processor = MessageStreamProcessor(callbacks=callbacks)

        text_event = AgentTextEvent(text="Hello agent")
        result_event = make_result_event()
        state = MessageIterationState()

        stream = events_to_stream([text_event, result_event])
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
    async def test_process_tool_use_event_invokes_callback(
        self, lifecycle_ctx: LifecycleContext
    ) -> None:
        """AgentToolUseEvent triggers on_tool_use callback and tracks pending."""
        tool_calls: list[tuple[str, str, dict[str, Any] | None]] = []

        def on_tool(issue_id: str, name: str, arguments: dict[str, Any] | None) -> None:
            tool_calls.append((issue_id, name, arguments))

        callbacks = StreamProcessorCallbacks(on_tool_use=on_tool)
        processor = MessageStreamProcessor(callbacks=callbacks)

        tool_event = AgentToolUseEvent(
            id="tool-abc",
            name="Read",
            input={"file_path": "/test.py"},
        )
        result_event = make_result_event()
        state = MessageIterationState()

        stream = events_to_stream([tool_event, result_event])
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

        tool_event = AgentToolUseEvent(
            id="tool-lint",
            name="Bash",
            input={"command": "ruff check ."},
        )
        result_block = AgentToolResultEvent(
            tool_use_id="tool-lint",
            content="All checks passed",
            is_error=False,
        )
        result_event = make_result_event()
        state = MessageIterationState()

        stream = events_to_stream([tool_event, result_block, result_event])
        await processor.process_stream(
            stream,
            issue_id="issue-3",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=lint_cache,
            query_start=0.0,
            tracer=None,
        )

        assert ("detect", "ruff check .") in lint_cache.detected_commands
        assert ("ruff", "ruff check .") in lint_cache.marked_successes

    @pytest.mark.asyncio
    async def test_bash_lint_error_not_cached(
        self, lifecycle_ctx: LifecycleContext
    ) -> None:
        """Bash lint commands with errors are not cached as success."""
        lint_cache = FakeLintCache()
        lint_cache.configure_detect("ruff check .", "ruff")

        processor = MessageStreamProcessor()

        tool_event = AgentToolUseEvent(
            id="tool-lint",
            name="Bash",
            input={"command": "ruff check ."},
        )
        result_block = AgentToolResultEvent(
            tool_use_id="tool-lint",
            content="Error: linting failed",
            is_error=True,
        )
        result_event = make_result_event()
        state = MessageIterationState()

        stream = events_to_stream([tool_event, result_block, result_event])
        await processor.process_stream(
            stream,
            issue_id="issue-4",
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=lint_cache,
            query_start=0.0,
            tracer=None,
        )

        assert ("detect", "ruff check .") in lint_cache.detected_commands
        assert lint_cache.marked_successes == []


class TestMessageStreamProcessorTracer:
    """Tests for tracer integration."""

    @pytest.mark.asyncio
    async def test_tracer_logs_all_events(
        self, lifecycle_ctx: LifecycleContext
    ) -> None:
        """Tracer receives all events from stream."""
        processor = MessageStreamProcessor()
        tracer = FakeTracer()

        text_event = AgentTextEvent(text="hello")
        result_event = make_result_event()
        state = MessageIterationState()

        stream = events_to_stream([text_event, result_event])
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
        assert tracer.logged_messages[0] == text_event
        assert tracer.logged_messages[1] == result_event


class TestMessageIterationState:
    """Tests for MessageIterationState tracking."""

    @pytest.mark.asyncio
    async def test_first_message_received_flag(
        self, processor: MessageStreamProcessor, lifecycle_ctx: LifecycleContext
    ) -> None:
        """first_message_received is set on first event."""
        result_event = make_result_event()
        state = MessageIterationState()

        assert state.first_message_received is False

        stream = events_to_stream([result_event])
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
        """tool_calls_this_turn counts AgentToolUseEvents."""
        tool1 = AgentToolUseEvent(id="t1", name="Read", input={})
        tool2 = AgentToolUseEvent(id="t2", name="Write", input={})
        result_event = make_result_event()
        state = MessageIterationState()

        stream = events_to_stream([tool1, tool2, result_event])
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
