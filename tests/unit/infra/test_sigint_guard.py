"""Tests for sigint_guard module."""

from __future__ import annotations

import asyncio

import pytest

from src.infra.sigint_guard import (
    FlowInterruptedError,
    InterruptGuard,
    await_interruptible,
    run_with_interrupt_checks,
    run_with_timeout_and_interrupt,
)


class TestFlowInterruptedError:
    """Tests for FlowInterruptedError exception."""

    def test_is_exception(self) -> None:
        """FlowInterruptedError is an Exception."""
        assert issubclass(FlowInterruptedError, Exception)

    def test_raise_with_message(self) -> None:
        """Can raise with a message."""
        with pytest.raises(FlowInterruptedError, match="test message"):
            raise FlowInterruptedError("test message")

    def test_not_interrupted_error(self) -> None:
        """FlowInterruptedError is NOT Python's InterruptedError (OSError)."""
        assert not issubclass(FlowInterruptedError, InterruptedError)
        assert not issubclass(FlowInterruptedError, OSError)


class TestInterruptGuard:
    """Tests for InterruptGuard class."""

    def test_is_interrupted_returns_false_when_event_none(self) -> None:
        """is_interrupted returns False when event is None."""
        guard = InterruptGuard(None)
        assert guard.is_interrupted() is False

    def test_is_interrupted_returns_false_when_event_not_set(self) -> None:
        """is_interrupted returns False when event is not set."""
        event = asyncio.Event()
        guard = InterruptGuard(event)
        assert guard.is_interrupted() is False

    def test_is_interrupted_returns_true_when_event_set(self) -> None:
        """is_interrupted returns True when event is set."""
        event = asyncio.Event()
        event.set()
        guard = InterruptGuard(event)
        assert guard.is_interrupted() is True

    def test_raise_if_interrupted_does_nothing_when_event_none(self) -> None:
        """raise_if_interrupted does nothing when event is None."""
        guard = InterruptGuard(None)
        guard.raise_if_interrupted()  # Should not raise

    def test_raise_if_interrupted_does_nothing_when_event_not_set(self) -> None:
        """raise_if_interrupted does nothing when event is not set."""
        event = asyncio.Event()
        guard = InterruptGuard(event)
        guard.raise_if_interrupted()  # Should not raise

    def test_raise_if_interrupted_raises_when_event_set(self) -> None:
        """raise_if_interrupted raises FlowInterruptedError when event is set."""
        event = asyncio.Event()
        event.set()
        guard = InterruptGuard(event)
        with pytest.raises(FlowInterruptedError, match="interrupted"):
            guard.raise_if_interrupted()


class TestAwaitInterruptible:
    """Tests for await_interruptible function."""

    @pytest.mark.asyncio
    async def test_waits_full_duration_when_no_event(self) -> None:
        """Waits full duration when interrupt_event is None."""
        interrupted = await await_interruptible(0.01, None)
        assert interrupted is False

    @pytest.mark.asyncio
    async def test_waits_full_duration_when_event_not_set(self) -> None:
        """Waits full duration when event is not set."""
        event = asyncio.Event()
        interrupted = await await_interruptible(0.01, event)
        assert interrupted is False

    @pytest.mark.asyncio
    async def test_returns_immediately_when_event_already_set(self) -> None:
        """Returns immediately when event is already set."""
        event = asyncio.Event()
        event.set()
        interrupted = await await_interruptible(10.0, event)
        assert interrupted is True

    @pytest.mark.asyncio
    async def test_returns_early_when_event_set_during_wait(self) -> None:
        """Returns early when event is set during the wait."""
        event = asyncio.Event()

        async def set_event_later() -> None:
            await asyncio.sleep(0.01)
            event.set()

        task = asyncio.create_task(set_event_later())
        interrupted = await await_interruptible(10.0, event)
        await task
        assert interrupted is True


class TestRunWithInterruptChecks:
    """Tests for run_with_interrupt_checks function."""

    @pytest.mark.asyncio
    async def test_returns_result_on_first_success(self) -> None:
        """Returns result when attempt_fn succeeds on first try."""

        async def success() -> str:
            return "success"

        result = await run_with_interrupt_checks(success, None)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retries_on_failure(self) -> None:
        """Retries when attempt_fn raises an exception."""
        attempts = 0

        async def fail_then_succeed() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("not yet")
            return "success"

        result = await run_with_interrupt_checks(
            fail_then_succeed, None, max_retries=3, retry_delay=0.001
        )
        assert result == "success"
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_raises_last_exception_when_all_retries_fail(self) -> None:
        """Raises the last exception when all retries are exhausted."""

        async def always_fail() -> str:
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            await run_with_interrupt_checks(
                always_fail, None, max_retries=3, retry_delay=0.001
            )

    @pytest.mark.asyncio
    async def test_aborts_between_retries_when_interrupted(self) -> None:
        """Aborts between retries when interrupt event is set."""
        event = asyncio.Event()
        attempts = 0

        async def fail_and_interrupt() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                event.set()  # Set interrupt after first failure
            raise ValueError("fail")

        with pytest.raises(FlowInterruptedError):
            await run_with_interrupt_checks(
                fail_and_interrupt, event, max_retries=3, retry_delay=0.001
            )
        assert attempts == 1  # Only one attempt before interrupt

    @pytest.mark.asyncio
    async def test_checks_interrupt_before_retry(self) -> None:
        """Checks interrupt before waiting for retry delay."""
        event = asyncio.Event()
        event.set()  # Pre-set interrupt
        attempts = 0

        async def fail() -> str:
            nonlocal attempts
            attempts += 1
            raise ValueError("fail")

        with pytest.raises(FlowInterruptedError):
            await run_with_interrupt_checks(
                fail, event, max_retries=3, retry_delay=10.0
            )
        assert attempts == 0  # Aborted before first attempt (interrupt checked first)

    @pytest.mark.asyncio
    async def test_raises_value_error_when_max_retries_zero(self) -> None:
        """Raises ValueError when max_retries is 0."""

        async def success() -> str:
            return "success"

        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            await run_with_interrupt_checks(success, None, max_retries=0)

    @pytest.mark.asyncio
    async def test_raises_value_error_when_max_retries_negative(self) -> None:
        """Raises ValueError when max_retries is negative."""

        async def success() -> str:
            return "success"

        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            await run_with_interrupt_checks(success, None, max_retries=-1)

    @pytest.mark.asyncio
    async def test_aborts_before_first_attempt_when_already_interrupted(self) -> None:
        """Aborts before first attempt when interrupt is already set."""
        event = asyncio.Event()
        event.set()
        attempts = 0

        async def count_attempts() -> str:
            nonlocal attempts
            attempts += 1
            return "success"

        with pytest.raises(FlowInterruptedError):
            await run_with_interrupt_checks(
                count_attempts, event, max_retries=3, retry_delay=0.001
            )
        assert attempts == 0  # Never attempted


class TestRunWithTimeoutAndInterrupt:
    """Tests for run_with_timeout_and_interrupt function."""

    @pytest.mark.asyncio
    async def test_returns_result_on_success(self) -> None:
        """Returns result when coroutine completes successfully."""

        async def quick_task() -> str:
            return "done"

        result, timed_out, interrupted = await run_with_timeout_and_interrupt(
            quick_task(), timeout=10.0, interrupt_event=None
        )
        assert result == "done"
        assert timed_out is False
        assert interrupted is False

    @pytest.mark.asyncio
    async def test_returns_timeout_when_task_too_slow(self) -> None:
        """Returns (None, True, False) when timeout is reached."""

        async def slow_task() -> str:
            await asyncio.sleep(10.0)
            return "done"

        result, timed_out, interrupted = await run_with_timeout_and_interrupt(
            slow_task(), timeout=0.01, interrupt_event=None
        )
        assert result is None
        assert timed_out is True
        assert interrupted is False

    @pytest.mark.asyncio
    async def test_returns_interrupted_when_event_set(self) -> None:
        """Returns (None, False, True) when interrupt event is set."""
        event = asyncio.Event()

        async def slow_task() -> str:
            await asyncio.sleep(10.0)
            return "done"

        async def set_event_soon() -> None:
            await asyncio.sleep(0.01)
            event.set()

        task = asyncio.create_task(set_event_soon())
        result, timed_out, interrupted = await run_with_timeout_and_interrupt(
            slow_task(), timeout=10.0, interrupt_event=event
        )
        assert result is None
        assert timed_out is False
        assert interrupted is True
        await task

    @pytest.mark.asyncio
    async def test_returns_immediately_when_event_already_set(self) -> None:
        """Returns immediately when event is already set."""
        event = asyncio.Event()
        event.set()

        async def slow_task() -> str:
            await asyncio.sleep(10.0)
            return "done"

        result, timed_out, interrupted = await run_with_timeout_and_interrupt(
            slow_task(), timeout=10.0, interrupt_event=event
        )
        assert result is None
        assert timed_out is False
        assert interrupted is True

    @pytest.mark.asyncio
    async def test_success_with_interrupt_event_provided(self) -> None:
        """Returns result when coroutine completes before timeout/interrupt."""
        event = asyncio.Event()

        async def quick_task() -> str:
            return "done"

        result, timed_out, interrupted = await run_with_timeout_and_interrupt(
            quick_task(), timeout=10.0, interrupt_event=event
        )
        assert result == "done"
        assert timed_out is False
        assert interrupted is False

    @pytest.mark.asyncio
    async def test_timeout_with_interrupt_event_provided(self) -> None:
        """Returns timeout when task is slow and interrupt_event is provided but not set."""
        event = asyncio.Event()

        async def slow_task() -> str:
            await asyncio.sleep(10.0)
            return "done"

        result, timed_out, interrupted = await run_with_timeout_and_interrupt(
            slow_task(), timeout=0.01, interrupt_event=event
        )
        assert result is None
        assert timed_out is True
        assert interrupted is False
