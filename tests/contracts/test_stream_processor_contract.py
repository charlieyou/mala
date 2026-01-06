"""Contract tests for FakeStreamProcessor.

Ensures FakeStreamProcessor matches the MessageStreamProcessor.process_stream
interface and exhibits correct behavioral parity.
"""

import inspect

import pytest

from src.pipeline.message_stream_processor import (
    IdleTimeoutError,
    IdleTimeoutStream,
    MessageIterationResult,
    MessageIterationState,
    MessageStreamProcessor,
)
from tests.fakes.stream_processor import FakeStreamProcessor


@pytest.mark.unit
def test_fake_stream_processor_signature_matches_real() -> None:
    """FakeStreamProcessor.process_stream has same parameters as real."""
    real_sig = inspect.signature(MessageStreamProcessor.process_stream)
    fake_sig = inspect.signature(FakeStreamProcessor.process_stream)

    real_params = list(real_sig.parameters.keys())
    fake_params = list(fake_sig.parameters.keys())

    # Both should have same parameter names in same order
    assert real_params == fake_params, (
        f"Parameter mismatch: real={real_params}, fake={fake_params}"
    )


class TestFakeStreamProcessorBehavior:
    """Behavioral tests for FakeStreamProcessor."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_configured_result(self) -> None:
        """Returns the configured result by default."""
        expected = MessageIterationResult(success=True, session_id="test-123")
        processor = FakeStreamProcessor(result=expected)

        result = await processor.process_stream(
            stream=_empty_stream(),
            issue_id="issue-1",
            state=MessageIterationState(),
            lifecycle_ctx=None,  # type: ignore[arg-type]
            lint_cache=None,  # type: ignore[arg-type]
            query_start=0.0,
            tracer=None,
        )

        assert result == expected

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_increments_call_count(self) -> None:
        """call_count increments on each call."""
        processor = FakeStreamProcessor()

        assert processor.call_count == 0
        await processor.process_stream(
            stream=_empty_stream(),
            issue_id="issue-1",
            state=MessageIterationState(),
            lifecycle_ctx=None,  # type: ignore[arg-type]
            lint_cache=None,  # type: ignore[arg-type]
            query_start=0.0,
            tracer=None,
        )
        assert processor.call_count == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_records_calls(self) -> None:
        """Calls are recorded with arguments."""
        processor = FakeStreamProcessor()
        state = MessageIterationState()

        await processor.process_stream(
            stream=_empty_stream(),
            issue_id="test-issue",
            state=state,
            lifecycle_ctx=None,  # type: ignore[arg-type]
            lint_cache=None,  # type: ignore[arg-type]
            query_start=123.0,
            tracer=None,
        )

        assert len(processor.calls) == 1
        assert processor.calls[0]["issue_id"] == "test-issue"
        assert processor.calls[0]["state"] is state
        assert processor.calls[0]["query_start"] == 123.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_until_raises_then_succeeds(self) -> None:
        """timeout_until causes N timeouts before success."""
        processor = FakeStreamProcessor(
            timeout_until=2,
            result=MessageIterationResult(success=True, session_id="final"),
        )

        # First two calls should timeout
        with pytest.raises(IdleTimeoutError):
            await processor.process_stream(
                stream=_empty_stream(),
                issue_id="issue-1",
                state=MessageIterationState(),
                lifecycle_ctx=None,  # type: ignore[arg-type]
                lint_cache=None,  # type: ignore[arg-type]
                query_start=0.0,
                tracer=None,
            )

        with pytest.raises(IdleTimeoutError):
            await processor.process_stream(
                stream=_empty_stream(),
                issue_id="issue-1",
                state=MessageIterationState(),
                lifecycle_ctx=None,  # type: ignore[arg-type]
                lint_cache=None,  # type: ignore[arg-type]
                query_start=0.0,
                tracer=None,
            )

        # Third call should succeed
        result = await processor.process_stream(
            stream=_empty_stream(),
            issue_id="issue-1",
            state=MessageIterationState(),
            lifecycle_ctx=None,  # type: ignore[arg-type]
            lint_cache=None,  # type: ignore[arg-type]
            query_start=0.0,
            tracer=None,
        )
        assert result.session_id == "final"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_side_effect_raises_exception(self) -> None:
        """side_effect causes configured exception to be raised."""
        processor = FakeStreamProcessor(side_effect=IdleTimeoutError("custom error"))

        with pytest.raises(IdleTimeoutError, match="custom error"):
            await processor.process_stream(
                stream=_empty_stream(),
                issue_id="issue-1",
                state=MessageIterationState(),
                lifecycle_ctx=None,  # type: ignore[arg-type]
                lint_cache=None,  # type: ignore[arg-type]
                query_start=0.0,
                tracer=None,
            )


def _empty_stream() -> IdleTimeoutStream:
    """Create an empty IdleTimeoutStream for testing."""
    from collections.abc import AsyncIterator
    from typing import Never

    async def _gen() -> AsyncIterator[Never]:
        return
        yield  # Makes this an async generator  # type: ignore[misc]

    return IdleTimeoutStream(
        stream=_gen(), timeout_seconds=None, pending_tool_ids=set()
    )
