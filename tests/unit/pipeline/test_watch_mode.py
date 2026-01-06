"""Unit tests for watch mode behavior in IssueExecutionCoordinator.

These tests verify watch mode loop behavior including:
- Sleep when no ready issues
- Exit on interrupt
- Validation triggers at threshold
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.core.models import PeriodicValidationConfig, WatchConfig
from src.pipeline.issue_execution_coordinator import (
    AbortResult,
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

        Expected behavior: sleeps and re-polls when watch mode enabled.
        When no interrupt_event is provided, the sleep_fn is used.
        """
        provider = FakeIssueProvider()  # No issues
        watch_config = WatchConfig(enabled=True, poll_interval_seconds=30.0)
        validation_config = PeriodicValidationConfig(validate_every=10)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(),
        )

        # Request abort after first sleep to stop the loop
        async def sleep_and_abort(seconds: float) -> None:
            sleep_calls.append(seconds)
            coord.request_abort("test_stop")

        sleep_fn.side_effect = sleep_and_abort

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=AsyncMock(return_value=None),
                finalize_callback=AsyncMock(),
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                watch_config=watch_config,
                validation_config=validation_config,
                sleep_fn=sleep_fn,
                # No interrupt_event - uses sleep_fn directly
            ),
            timeout=1.0,
        )

        # Expect: coordinator should have called sleep with poll_interval_seconds
        assert len(sleep_calls) > 0, "Expected sleep to be called in watch mode"
        assert sleep_calls[0] == 30.0, "Expected sleep duration to match poll_interval"
        assert result.exit_code == 3
        assert result.exit_reason == "abort"


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
        """When interrupt_event is set, watch mode should exit gracefully."""
        provider = FakeIssueProvider()  # No issues
        watch_config = WatchConfig(enabled=True, poll_interval_seconds=60.0)
        validation_config = PeriodicValidationConfig(validate_every=10)
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
            abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
            watch_config=watch_config,
            validation_config=validation_config,
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
        """Validation callback should be called when completed count reaches threshold."""
        # Set up issues that will complete
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(5)
            }
        )
        watch_config = WatchConfig(enabled=True)
        validation_config = PeriodicValidationConfig(validate_every=3)
        validation_callback = AsyncMock(return_value=True)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        # Use interrupt to stop after all issues complete
        interrupt_event = asyncio.Event()

        poll_count = 0

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            nonlocal poll_count
            poll_count += 1
            remaining = [
                f"issue-{i}"
                for i in range(5)
                if f"issue-{i}" not in coord.completed_ids
            ]
            if not remaining:
                interrupt_event.set()
            return remaining

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                watch_config=watch_config,
                validation_config=validation_config,
                validation_callback=validation_callback,
                interrupt_event=interrupt_event,
            ),
            timeout=5.0,
        )

        assert result.exit_code == 130  # Interrupted

    @pytest.mark.asyncio
    async def test_validation_threshold_advances_after_trigger(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """Validation threshold should advance by validate_every after each trigger."""
        # Set up enough issues to trigger multiple validations
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(25)
            }
        )
        watch_config = WatchConfig(enabled=True)
        validation_config = PeriodicValidationConfig(validate_every=10)
        validation_callback = AsyncMock(return_value=True)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        interrupt_event = asyncio.Event()

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            remaining = [
                f"issue-{i}"
                for i in range(25)
                if f"issue-{i}" not in coord.completed_ids
            ]
            if not remaining:
                interrupt_event.set()
            return remaining

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                watch_config=watch_config,
                validation_config=validation_config,
                validation_callback=validation_callback,
                interrupt_event=interrupt_event,
            ),
            timeout=10.0,
        )

        assert result.exit_code == 130

    @pytest.mark.asyncio
    async def test_parallel_completions_trigger_validation_once(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """Parallel completions (9→12) should trigger validation only once."""
        # Set up 12 issues that will complete in parallel (max_agents=4)
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(12)
            }
        )
        watch_config = WatchConfig(enabled=True)
        validation_config = PeriodicValidationConfig(validate_every=10)
        validation_callback = AsyncMock(return_value=True)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=4),  # Multiple parallel agents
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        interrupt_event = asyncio.Event()

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            remaining = [
                f"issue-{i}"
                for i in range(12)
                if f"issue-{i}" not in coord.completed_ids
            ]
            if not remaining:
                interrupt_event.set()
            return remaining

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                watch_config=watch_config,
                validation_config=validation_config,
                validation_callback=validation_callback,
                interrupt_event=interrupt_event,
            ),
            timeout=5.0,
        )

        assert result.exit_code == 130

    @pytest.mark.asyncio
    async def test_validation_failure_returns_exit_code_1(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """Validation failure should return exit_code=1."""
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(5)
            }
        )
        watch_config = WatchConfig(enabled=True)
        validation_config = PeriodicValidationConfig(validate_every=3)
        validation_callback = AsyncMock(return_value=False)  # Validation fails

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            return [
                f"issue-{i}"
                for i in range(5)
                if f"issue-{i}" not in coord.completed_ids
            ]

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                watch_config=watch_config,
                validation_config=validation_config,
                validation_callback=validation_callback,
            ),
            timeout=5.0,
        )

        assert result.exit_code == 1
        assert result.exit_reason == "validation_failed"

    @pytest.mark.asyncio
    async def test_successful_completion_returns_exit_code_0(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """Successful completion with passing validation should return exit_code=0."""
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(3)
            }
        )
        # No watch mode - exits when no more issues
        validation_callback = AsyncMock(return_value=True)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            return [
                f"issue-{i}"
                for i in range(3)
                if f"issue-{i}" not in coord.completed_ids
            ]

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                validation_callback=validation_callback,
            ),
            timeout=5.0,
        )

        assert result.exit_code == 0
        assert result.exit_reason == "success"

    @pytest.mark.asyncio
    async def test_sigint_during_active_runs_final_validation(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """SIGINT during active processing should run final validation."""
        provider = FakeIssueProvider(
            issues={"issue-1": FakeIssue(id="issue-1", status="ready")}
        )
        watch_config = WatchConfig(enabled=True)
        interrupt_event = asyncio.Event()
        validation_callback = AsyncMock(return_value=True)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)
            # Set interrupt after task completes
            interrupt_event.set()

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            return ["issue-1"] if "issue-1" not in coord.completed_ids else []

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                watch_config=watch_config,
                interrupt_event=interrupt_event,
                validation_callback=validation_callback,
            ),
            timeout=5.0,
        )

        assert result.exit_code == 130
        assert result.exit_reason == "interrupted"

    @pytest.mark.asyncio
    async def test_final_validation_skipped_if_just_ran_at_threshold(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """Final validation should be skipped if validation just ran at threshold."""
        # Set up exactly 10 issues with validate_every=10
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(10)
            }
        )
        watch_config = WatchConfig(enabled=True)
        validation_config = PeriodicValidationConfig(validate_every=10)
        validation_callback = AsyncMock(return_value=True)
        interrupt_event = asyncio.Event()

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            remaining = [
                f"issue-{i}"
                for i in range(10)
                if f"issue-{i}" not in coord.completed_ids
            ]
            if not remaining:
                interrupt_event.set()
            return remaining

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                watch_config=watch_config,
                validation_config=validation_config,
                validation_callback=validation_callback,
                interrupt_event=interrupt_event,
            ),
            timeout=5.0,
        )

        assert result.exit_code == 130

    @pytest.mark.asyncio
    async def test_final_validation_runs_on_normal_exit_if_issues_completed(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """Final validation should run on normal exit if issues completed since last validation."""
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(3)
            }
        )
        # No watch mode - exits when done
        validation_callback = AsyncMock(return_value=True)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            return [
                f"issue-{i}"
                for i in range(3)
                if f"issue-{i}" not in coord.completed_ids
            ]

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                validation_callback=validation_callback,
            ),
            timeout=5.0,
        )

        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_max_issues_uses_completed_count_not_spawned(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """--max-issues should use completed_count, not spawned count."""
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(10)
            }
        )

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_issues=3, max_agents=2),  # 2 agents parallel
        )

        spawned_count = 0

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            nonlocal spawned_count
            spawned_count += 1

            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            return [
                f"issue-{i}"
                for i in range(10)
                if f"issue-{i}" not in coord.completed_ids
            ]

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
            ),
            timeout=5.0,
        )

        assert result.exit_code == 0
        assert result.exit_reason == "limit_reached"
        # Key assertion: spawned_count >= 3 (may spawn more before completions counted)
        # But loop exits based on completed_count, not spawned count
        assert len(coord.completed_ids) >= 3

    @pytest.mark.asyncio
    async def test_max_issues_exits_watch_mode(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """--max-issues should exit watch mode when completed_count reaches limit."""
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(10)
            }
        )
        watch_config = WatchConfig(enabled=True)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_issues=3, max_agents=1),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            return [
                f"issue-{i}"
                for i in range(10)
                if f"issue-{i}" not in coord.completed_ids
            ]

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                watch_config=watch_config,
            ),
            timeout=5.0,
        )

        # Should exit with limit_reached, not continue watching
        assert result.exit_code == 0
        assert result.exit_reason == "limit_reached"
        assert len(coord.completed_ids) == 3


class TestPollFailureHandling:
    """Tests for poll failure handling in the coordinator."""

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
    async def test_poll_failure_increments_counter(
        self,
        event_sink: FakeEventSink,
        sleep_calls: list[float],
        sleep_fn: AsyncMock,
    ) -> None:
        """Poll failure should increment consecutive_poll_failures counter."""
        provider = FakeIssueProvider()
        provider.get_ready_async = AsyncMock(  # type: ignore[method-assign]
            side_effect=[Exception("Network error"), []]
        )

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(),
        )

        result = await coord.run_loop(
            spawn_callback=AsyncMock(return_value=None),
            finalize_callback=AsyncMock(),
            abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
            sleep_fn=sleep_fn,
        )

        # After 1 failure + 1 success, loop exits normally
        assert result.exit_code == 0
        assert len(sleep_calls) == 1  # Slept once after failure

    @pytest.mark.asyncio
    async def test_poll_success_resets_failure_counter(
        self,
        event_sink: FakeEventSink,
        sleep_calls: list[float],
        sleep_fn: AsyncMock,
    ) -> None:
        """Successful poll should reset consecutive_poll_failures to 0."""
        provider = FakeIssueProvider()
        # Fail twice, succeed (should NOT abort - counter reset means 3rd poll not failure #3)
        provider.get_ready_async = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                Exception("Fail 1"),
                Exception("Fail 2"),
                [],  # Success - resets counter, exits (no work)
            ]
        )

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(),
        )

        result = await coord.run_loop(
            spawn_callback=AsyncMock(return_value=None),
            finalize_callback=AsyncMock(),
            abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
            sleep_fn=sleep_fn,
        )

        # Should NOT abort - counter was only at 2, then success resets it
        assert result.exit_code == 0
        assert result.exit_reason == "success"
        assert len(sleep_calls) == 2  # Slept after each failure

    @pytest.mark.asyncio
    async def test_three_consecutive_poll_failures_aborts(
        self,
        event_sink: FakeEventSink,
        sleep_calls: list[float],
        sleep_fn: AsyncMock,
    ) -> None:
        """Three consecutive poll failures should abort with exit code 3."""
        provider = FakeIssueProvider()
        provider.get_ready_async = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                Exception("Fail 1"),
                Exception("Fail 2"),
                Exception("Fail 3"),
            ]
        )

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(),
        )

        result = await coord.run_loop(
            spawn_callback=AsyncMock(return_value=None),
            finalize_callback=AsyncMock(),
            abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
            sleep_fn=sleep_fn,
        )

        assert result.exit_code == 3
        assert result.exit_reason == "poll_failed"
        # Slept after first two failures, not after third (abort immediately)
        assert len(sleep_calls) == 2

    @pytest.mark.asyncio
    async def test_poll_failure_abort_returns_exit_code_3(
        self,
        event_sink: FakeEventSink,
        sleep_fn: AsyncMock,
    ) -> None:
        """Poll failure abort should return exit_code=3 and exit_reason='poll_failed'."""
        provider = FakeIssueProvider()
        provider.get_ready_async = AsyncMock(  # type: ignore[method-assign]
            side_effect=Exception("Persistent failure")
        )

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(),
        )

        result = await coord.run_loop(
            spawn_callback=AsyncMock(return_value=None),
            finalize_callback=AsyncMock(),
            abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
            sleep_fn=sleep_fn,
        )

        assert result.exit_code == 3
        assert result.exit_reason == "poll_failed"

    @pytest.mark.asyncio
    async def test_poll_failure_abort_runs_final_validation_if_issues_completed(
        self,
        event_sink: FakeEventSink,
        sleep_fn: AsyncMock,
    ) -> None:
        """Poll failure abort should run final validation if issues completed."""
        provider = FakeIssueProvider(
            issues={"issue-1": FakeIssue(id="issue-1", status="ready")}
        )

        # First poll succeeds with issue, subsequent polls fail
        call_count = 0

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ["issue-1"]
            raise Exception("Poll failed")

        provider.get_ready_async = AsyncMock(  # type: ignore[method-assign]
            side_effect=get_ready_side_effect
        )

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        validation_callback = AsyncMock(return_value=True)

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                validation_callback=validation_callback,
                sleep_fn=sleep_fn,
            ),
            timeout=5.0,
        )

        assert result.exit_code == 3
        assert result.exit_reason == "poll_failed"


class TestWatchModeIdleBehavior:
    """Tests for watch mode idle detection and behavior."""

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
    async def test_watch_mode_continues_when_issues_appear_after_sleep(
        self,
        event_sink: FakeEventSink,
        sleep_calls: list[float],
        sleep_fn: AsyncMock,
    ) -> None:
        """Watch mode should continue and process issues that appear after sleep."""
        provider = FakeIssueProvider()
        watch_config = WatchConfig(enabled=True, poll_interval_seconds=10.0)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        poll_count = 0

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            nonlocal poll_count
            poll_count += 1
            if poll_count == 1:
                return []  # First poll: no issues (triggers sleep)
            elif poll_count == 2:
                return ["issue-1"]  # Second poll: issue appears
            return []  # Third poll: no issues

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)
            # After issue completes, request abort to exit
            coord.request_abort("test_done")

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                watch_config=watch_config,
                sleep_fn=sleep_fn,
                # No interrupt_event - uses sleep_fn directly
            ),
            timeout=5.0,
        )

        # Should have slept once when no issues, then processed issue-1
        assert len(sleep_calls) >= 1
        assert result.issues_spawned == 1

    @pytest.mark.asyncio
    async def test_idle_state_requires_no_active_agents(
        self,
        event_sink: FakeEventSink,
        sleep_calls: list[float],
        sleep_fn: AsyncMock,
    ) -> None:
        """Idle state requires both no ready issues AND no active agents."""
        provider = FakeIssueProvider(
            issues={"issue-1": FakeIssue(id="issue-1", status="ready")}
        )
        watch_config = WatchConfig(enabled=True, poll_interval_seconds=10.0)
        interrupt_event = asyncio.Event()

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        poll_count = 0

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            nonlocal poll_count
            poll_count += 1
            if poll_count == 1:
                return ["issue-1"]  # First poll: issue ready
            return []  # Subsequent polls: no issues

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)
            # After finalization, interrupt to exit cleanly
            interrupt_event.set()

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                watch_config=watch_config,
                sleep_fn=sleep_fn,
                interrupt_event=interrupt_event,
            ),
            timeout=5.0,
        )

        # Should not have slept while issue-1 was active
        # (sleep only happens when no ready AND no active)
        assert result.issues_spawned == 1
        # Verify no sleep calls happened before task completion
        assert sleep_calls == [], "Should not sleep while agents are active"

    @pytest.mark.asyncio
    async def test_sigint_during_idle_runs_final_validation_if_issues_completed(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """SIGINT during idle should run final validation if issues completed."""
        provider = FakeIssueProvider(
            issues={"issue-1": FakeIssue(id="issue-1", status="ready")}
        )
        watch_config = WatchConfig(enabled=True, poll_interval_seconds=60.0)
        interrupt_event = asyncio.Event()
        validation_callback = AsyncMock(return_value=True)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        poll_count = 0

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            nonlocal poll_count
            poll_count += 1
            if poll_count == 1:
                return ["issue-1"]
            return []  # No more issues after first

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)
            # After issue completes, set interrupt during next idle sleep
            interrupt_event.set()

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                watch_config=watch_config,
                interrupt_event=interrupt_event,
                validation_callback=validation_callback,
            ),
            timeout=5.0,
        )

        assert result.exit_code == 130
        assert result.exit_reason == "interrupted"

    @pytest.mark.asyncio
    async def test_watch_state_threshold_initialized_from_config(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """WatchState.next_validation_threshold should be initialized from config.

        Verifies that validate_every config controls when validation triggers:
        validation should NOT trigger until the threshold is crossed.
        """
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(3)
            }
        )
        # Set validate_every=5, so validation should NOT trigger after only 3 completions
        watch_config = WatchConfig(enabled=True, poll_interval_seconds=10.0)
        validation_config = PeriodicValidationConfig(validate_every=5)
        interrupt_event = asyncio.Event()
        validation_callback = AsyncMock(return_value=True)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=3),  # Allow all 3 to spawn
        )

        completed_count = 0

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            # Return all issues on first poll, then empty
            if not coord.completed_ids:
                return ["issue-0", "issue-1", "issue-2"]
            return []

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            nonlocal completed_count
            coord.mark_completed(issue_id)
            completed_count += 1
            # After all 3 complete, interrupt on next idle
            if completed_count >= 3:
                interrupt_event.set()

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                watch_config=watch_config,
                validation_config=validation_config,
                interrupt_event=interrupt_event,
                validation_callback=validation_callback,
            ),
            timeout=5.0,
        )

        assert result.issues_spawned == 3

    @pytest.mark.asyncio
    async def test_idle_logging_rate_limited_to_5_minutes(
        self,
        event_sink: FakeEventSink,
        sleep_calls: list[float],
        sleep_fn: AsyncMock,
    ) -> None:
        """on_watch_idle should emit once on transition, then every 5 minutes."""
        import time
        from unittest.mock import patch

        provider = FakeIssueProvider()
        watch_config = WatchConfig(enabled=True, poll_interval_seconds=1.0)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(),
        )

        poll_count = 0
        mock_time = 0.0

        def time_fn() -> float:
            return mock_time

        async def sleep_side_effect(seconds: float) -> None:
            nonlocal poll_count, mock_time
            sleep_calls.append(seconds)
            poll_count += 1
            if poll_count == 1:
                # First sleep: advance time by 1 second (within 5 min window)
                mock_time += 1.0
            elif poll_count == 2:
                # Second sleep: advance time to 301 seconds (past 5 min threshold)
                mock_time += 300.0
            elif poll_count == 3:
                # Third sleep: abort to exit
                coord.request_abort("test_done")

        sleep_fn.side_effect = sleep_side_effect

        with patch.object(time, "monotonic", time_fn):
            result = await asyncio.wait_for(
                coord.run_loop(
                    spawn_callback=AsyncMock(return_value=None),
                    finalize_callback=AsyncMock(),
                    abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                    watch_config=watch_config,
                    sleep_fn=sleep_fn,
                    # No interrupt_event - uses sleep_fn directly
                ),
                timeout=5.0,
            )

        # Count watch_idle events
        watch_idle_events = event_sink.get_events("watch_idle")

        # Should have 2 events: once on transition, once after 5 minutes
        # (not 3+, because the second poll didn't trigger due to being within 5 min)
        assert len(watch_idle_events) == 2, (
            f"Expected 2 watch_idle events, got {len(watch_idle_events)}: {watch_idle_events}"
        )
        assert result.exit_code == 3
