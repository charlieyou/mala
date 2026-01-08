"""Shared interrupt handling helpers for SDK flows.

This module provides consistent interrupt handling across all SDK flows,
enabling graceful cancellation of long-running operations.

Key components:
- FlowInterruptedError: Exception raised when a flow is interrupted
- InterruptGuard: Helper class to check/raise on interrupt events
- await_interruptible(): Interruptible sleep function
- run_with_interrupt_checks(): Retry loop with interrupt awareness
- run_with_timeout_and_interrupt(): Timeout + interrupt handling
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine  # noqa: TC003 - runtime for TypeVar
from typing import TypeVar

__all__ = [
    "FlowInterruptedError",
    "InterruptGuard",
    "await_interruptible",
    "run_with_interrupt_checks",
    "run_with_timeout_and_interrupt",
]


class FlowInterruptedError(Exception):
    """Raised when a flow is interrupted by SIGINT.

    Named to avoid shadowing Python's built-in InterruptedError
    (which is an OSError with errno.EINTR).
    """


class InterruptGuard:
    """Helper class to check and raise on interrupt events.

    Wraps an asyncio.Event to provide convenient methods for
    checking interrupt state and raising FlowInterruptedError.
    """

    def __init__(self, event: asyncio.Event | None) -> None:
        """Initialize the interrupt guard.

        Args:
            event: The interrupt event to monitor. If None, interrupt
                   checking is disabled (is_interrupted always returns False).
        """
        self._event = event

    def is_interrupted(self) -> bool:
        """Check if the interrupt event is set.

        Returns:
            True if event is set, False if event is None or not set.
        """
        if self._event is None:
            return False
        return self._event.is_set()

    def raise_if_interrupted(self) -> None:
        """Raise FlowInterruptedError if the interrupt event is set.

        Raises:
            FlowInterruptedError: If the interrupt event is set.
        """
        if self.is_interrupted():
            raise FlowInterruptedError("Flow interrupted by SIGINT")


async def await_interruptible(
    delay: float, interrupt_event: asyncio.Event | None
) -> bool:
    """Wait for delay seconds, but return early if interrupted.

    Args:
        delay: Number of seconds to wait.
        interrupt_event: Event to monitor for interruption. If None,
                        waits the full duration.

    Returns:
        True if interrupted before delay elapsed, False if waited full duration.
    """
    if interrupt_event is None:
        await asyncio.sleep(delay)
        return False

    if interrupt_event.is_set():
        return True

    try:
        await asyncio.wait_for(interrupt_event.wait(), timeout=delay)
        # Event was set
        return True
    except TimeoutError:
        # Waited full duration
        return False


T = TypeVar("T")


async def run_with_interrupt_checks(
    attempt_fn: Callable[[], Awaitable[T]],
    interrupt_event: asyncio.Event | None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> T:
    """Run an async function with retries, checking for interrupts between attempts.

    Args:
        attempt_fn: Async function to call on each attempt. Should raise
                   an exception to trigger a retry.
        interrupt_event: Event to check between retries. If None, no
                        interrupt checking is performed.
        max_retries: Maximum number of retry attempts (default 3, must be >= 1).
        retry_delay: Seconds to wait between retries (default 1.0).

    Returns:
        The result of a successful attempt_fn call.

    Raises:
        FlowInterruptedError: If interrupted before or between retries.
        ValueError: If max_retries < 1.
        Exception: The last exception from attempt_fn if all retries fail.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    guard = InterruptGuard(interrupt_event)
    last_exception: Exception | None = None

    for attempt in range(max_retries):
        # Check for interrupt before each attempt
        guard.raise_if_interrupted()

        # Wait before retry (not before first attempt)
        if attempt > 0:
            interrupted = await await_interruptible(retry_delay, interrupt_event)
            if interrupted:
                raise FlowInterruptedError("Flow interrupted during retry delay")

        try:
            return await attempt_fn()
        except Exception as e:
            last_exception = e
            # Continue to next retry

    # All retries exhausted
    assert last_exception is not None
    raise last_exception


async def run_with_timeout_and_interrupt(
    coro: Coroutine[object, object, T],
    timeout: float,
    interrupt_event: asyncio.Event | None,
) -> tuple[T | None, bool, bool]:
    """Run a coroutine with timeout and interrupt monitoring.

    Args:
        coro: The coroutine to run.
        timeout: Maximum seconds to wait for completion.
        interrupt_event: Event to monitor for interruption. If None,
                        only timeout is monitored.

    Returns:
        A tuple of (result, timed_out, interrupted) where:
        - result: The coroutine result if successful, None otherwise.
        - timed_out: True if the timeout was reached.
        - interrupted: True if interrupted by the event.

    Note:
        The coroutine is cancelled if timeout or interrupt occurs.
        At most one of timed_out and interrupted will be True.
    """
    # Check for early interrupt before starting task
    if interrupt_event is not None and interrupt_event.is_set():
        # Close the coroutine to avoid "coroutine was never awaited" warning
        coro.close()
        return (None, False, True)

    task = asyncio.create_task(coro)

    if interrupt_event is None:
        # No interrupt monitoring, just timeout
        try:
            result = await asyncio.wait_for(task, timeout=timeout)
            return (result, False, False)
        except TimeoutError:
            # Ensure task is cancelled and awaited
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return (None, True, False)
        except BaseException:
            # Cleanup on outer cancellation or other errors
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            raise

    # Create waiter for interrupt event
    interrupt_task = asyncio.create_task(interrupt_event.wait())

    async def cleanup_tasks(
        cancel_main: bool = True, cancel_interrupt: bool = True
    ) -> None:
        """Cancel and await tasks to ensure proper cleanup."""
        if cancel_main:
            task.cancel()
        if cancel_interrupt:
            interrupt_task.cancel()
        # Always await both to ensure cleanup
        for t in [task, interrupt_task]:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    try:
        done, _ = await asyncio.wait(
            {task, interrupt_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if task in done:
            # Main task completed - cleanup interrupt_task
            interrupt_task.cancel()
            try:
                await interrupt_task
            except asyncio.CancelledError:
                pass
            # Return result (may raise if coro raised)
            return (task.result(), False, False)

        if interrupt_task in done:
            # Interrupt event was set - cleanup task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return (None, False, True)

        # Timeout reached (neither task completed)
        await cleanup_tasks()
        return (None, True, False)

    except BaseException:
        # Cleanup on any error (including CancelledError)
        await cleanup_tasks()
        raise
