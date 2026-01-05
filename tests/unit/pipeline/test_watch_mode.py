"""Unit tests for watch mode behavior in IssueExecutionCoordinator.

These tests verify watch mode loop behavior including:
- Sleep when no ready issues
- Exit on interrupt
- Validation triggers at threshold
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.orchestration.types import WatchConfig
from src.pipeline.issue_execution_coordinator import (
    CoordinatorConfig,
    IssueExecutionCoordinator,
)
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider


class TestWatchModeSleeps:
    """Tests for watch mode sleep behavior when no issues ready."""

    @pytest.fixture
    def event_sink(self) -> FakeEventSink:
        return FakeEventSink()

    @pytest.fixture
    def sleep_calls(self) -> list[float]:
        """Track calls to the injected sleep function."""
        return []

    @pytest.fixture
    def sleep_fn(self, sleep_calls: list[float]) -> AsyncMock:
        """Create an injectable sleep function that records calls."""

        async def _sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        return AsyncMock(side_effect=_sleep)

    @pytest.mark.asyncio
    async def test_watch_mode_sleeps_when_no_ready_issues(
        self,
        event_sink: FakeEventSink,
        sleep_calls: list[float],
        sleep_fn: AsyncMock,
    ) -> None:
        """When watch mode is enabled and no issues ready, coordinator should sleep.

        This test expects to FAIL until watch mode sleep is implemented.
        Current behavior: exits immediately with success when no work.
        Expected behavior: sleeps and re-polls when watch mode enabled.
        """
        provider = FakeIssueProvider()  # No issues
        watch_config = WatchConfig(enabled=True, poll_interval_seconds=30.0)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(),
        )

        # Run with timeout to avoid hanging if test is broken
        await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=AsyncMock(return_value=None),
                finalize_callback=AsyncMock(),
                abort_callback=AsyncMock(),
                watch_config=watch_config,
                sleep_fn=sleep_fn,
            ),
            timeout=1.0,
        )

        # Expect: coordinator should have called sleep with poll_interval_seconds
        # Current stub behavior: exits immediately, so sleep is never called
        assert len(sleep_calls) > 0, "Expected sleep to be called in watch mode"
        assert sleep_calls[0] == 30.0, "Expected sleep duration to match poll_interval"


class TestWatchModeInterrupt:
    """Tests for watch mode interrupt handling."""

    @pytest.fixture
    def event_sink(self) -> FakeEventSink:
        return FakeEventSink()

    @pytest.mark.asyncio
    async def test_watch_mode_exits_on_interrupt(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """When interrupt_event is set, watch mode should exit gracefully.

        This test expects to FAIL until interrupt handling is implemented.
        """
        provider = FakeIssueProvider()  # No issues
        watch_config = WatchConfig(enabled=True, poll_interval_seconds=60.0)
        interrupt_event = asyncio.Event()
        interrupt_event.set()  # Set immediately

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(),
        )

        result = await coord.run_loop(
            spawn_callback=AsyncMock(return_value=None),
            finalize_callback=AsyncMock(),
            abort_callback=AsyncMock(),
            watch_config=watch_config,
            interrupt_event=interrupt_event,
        )

        # Expect: should exit with interrupted exit code (130)
        assert result.exit_code == 130, "Expected exit_code 130 for interrupt"
        assert result.exit_reason == "interrupted"


class TestWatchModeValidation:
    """Tests for periodic validation triggers in watch mode."""

    @pytest.fixture
    def event_sink(self) -> FakeEventSink:
        return FakeEventSink()

    @pytest.mark.asyncio
    async def test_validation_triggers_at_threshold(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """Validation callback should be called when completed count reaches threshold.

        This test expects to FAIL until validation triggering is implemented.
        """
        # Set up issues that will complete
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(5)
            }
        )
        watch_config = WatchConfig(enabled=True, validate_every=3)
        validation_callback = AsyncMock(return_value=True)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_issues=5, max_agents=1),
        )

        # Track spawned tasks so we can ensure they complete
        spawned_tasks: list[asyncio.Task[None]] = []

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            task = asyncio.create_task(complete_immediately())
            spawned_tasks.append(task)
            return task

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            # Mark as completed so it's excluded from next poll
            coord.mark_completed(issue_id)

        await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(),
                watch_config=watch_config,
                validation_callback=validation_callback,
            ),
            timeout=5.0,
        )

        # Expect: validation should have been called at least once (after 3 issues)
        assert validation_callback.call_count >= 1, (
            "Expected validation to be called after threshold reached"
        )
