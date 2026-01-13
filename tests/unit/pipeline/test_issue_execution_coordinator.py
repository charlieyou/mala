"""Unit tests for IssueExecutionCoordinator.

These tests verify the coordinator's loop logic without SDK dependencies.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.models import OrderPreference, PeriodicValidationConfig
from src.pipeline.issue_execution_coordinator import (
    AbortResult,
    CoordinatorConfig,
    IssueExecutionCoordinator,
)


class MockIssueProvider:
    """Mock IssueProvider for testing."""

    def __init__(self, ready_issues: list[list[str]] | None = None) -> None:
        """Initialize with optional ready issue sequences.

        Args:
            ready_issues: List of lists, each returned by successive get_ready_async calls.
        """
        self._ready_sequences = ready_issues or []
        self._call_count = 0

    async def get_ready_async(
        self,
        exclude_ids: set[str] | None = None,
        epic_id: str | None = None,
        only_ids: list[str] | None = None,
        suppress_warn_ids: set[str] | None = None,
        include_wip: bool = False,
        focus: bool = True,
        orphans_only: bool = False,
        order_preference: OrderPreference = OrderPreference.FOCUS,
    ) -> list[str]:
        """Return next sequence of ready issues."""
        exclude = exclude_ids or set()
        if self._call_count < len(self._ready_sequences):
            result = [
                issue_id
                for issue_id in self._ready_sequences[self._call_count]
                if issue_id not in exclude
            ]
            self._call_count += 1
            return result
        return []


class MockEventSink:
    """Mock event sink for testing."""

    def __init__(self) -> None:
        self.events: list[tuple[str, tuple]] = []  # type: ignore[type-arg]

    def on_ready_issues(self, issues: list[str]) -> None:
        self.events.append(("ready_issues", (issues,)))

    def on_waiting_for_agents(self, count: int) -> None:
        self.events.append(("waiting_for_agents", (count,)))

    def on_no_more_issues(self, reason: str) -> None:
        self.events.append(("no_more_issues", (reason,)))

    def on_abort_requested(self, reason: str) -> None:
        self.events.append(("abort_requested", (reason,)))

    def on_tasks_aborting(self, count: int, reason: str) -> None:
        self.events.append(("tasks_aborting", (count, reason)))


class TestIssueExecutionCoordinator:
    """Tests for IssueExecutionCoordinator."""

    @pytest.fixture
    def event_sink(self) -> MockEventSink:
        return MockEventSink()

    @pytest.fixture
    def coordinator(self, event_sink: MockEventSink) -> IssueExecutionCoordinator:
        """Create coordinator with default config."""
        beads = MockIssueProvider()
        return IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(),
        )

    def test_request_abort(self, coordinator: IssueExecutionCoordinator) -> None:
        """request_abort sets abort state."""
        assert coordinator.abort_run is False
        coordinator.request_abort("test error")
        assert coordinator.abort_run is True
        assert coordinator.abort_reason == "test error"

    def test_request_abort_idempotent(
        self, coordinator: IssueExecutionCoordinator
    ) -> None:
        """Second request_abort is ignored."""
        coordinator.request_abort("first")
        coordinator.request_abort("second")
        assert coordinator.abort_reason == "first"

    def test_mark_failed(self, coordinator: IssueExecutionCoordinator) -> None:
        """mark_failed adds to failed set."""
        coordinator.mark_failed("issue-1")
        assert "issue-1" in coordinator.failed_issues

    def test_mark_completed(self, coordinator: IssueExecutionCoordinator) -> None:
        """mark_completed updates completed_ids and removes from active."""
        task = MagicMock()
        coordinator.active_tasks["issue-1"] = task
        coordinator.mark_completed("issue-1")
        assert "issue-1" in coordinator.completed_ids
        assert "issue-1" not in coordinator.active_tasks

    def test_register_task(self, coordinator: IssueExecutionCoordinator) -> None:
        """register_task adds to active_tasks."""
        task = MagicMock()
        coordinator.register_task("issue-1", task)
        assert coordinator.active_tasks["issue-1"] is task


class TestRunLoop:
    """Tests for run_loop execution flow."""

    @pytest.fixture
    def event_sink(self) -> MockEventSink:
        return MockEventSink()

    @pytest.mark.asyncio
    async def test_exits_when_no_ready_issues(self, event_sink: MockEventSink) -> None:
        """Loop exits immediately when no issues are ready."""
        beads = MockIssueProvider(ready_issues=[[]])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(),
        )

        spawn_callback = AsyncMock(return_value=None)
        finalize_callback = AsyncMock()
        abort_callback = AsyncMock()

        result = await coord.run_loop(spawn_callback, finalize_callback, abort_callback)

        assert result.issues_spawned == 0
        # Behavioral assertion: no issues spawned means spawn_callback was never invoked
        # (verified by issues_spawned == 0 above and no completed_ids)
        assert coord.completed_ids == set()
        assert ("no_more_issues", ("none_ready",)) in event_sink.events

    @pytest.mark.asyncio
    async def test_spawns_single_issue(self, event_sink: MockEventSink) -> None:
        """Loop spawns and completes a single issue."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(),
        )

        spawned_tasks: dict[str, asyncio.Task] = {}  # type: ignore[type-arg]

        async def spawn_callback(issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
            async def work() -> None:
                pass

            task = asyncio.create_task(work())
            spawned_tasks[issue_id] = task
            return task

        async def finalize_callback(issue_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
            coord.mark_completed(issue_id)

        result = await coord.run_loop(spawn_callback, finalize_callback, AsyncMock())

        assert result.issues_spawned == 1
        assert "issue-1" in coord.completed_ids

    @pytest.mark.asyncio
    async def test_respects_max_agents(self, event_sink: MockEventSink) -> None:
        """Loop only spawns up to max_agents concurrently."""
        # Provide issue-3 in subsequent calls so it's available after issue-1 completes
        beads = MockIssueProvider(
            ready_issues=[
                ["issue-1", "issue-2", "issue-3"],
                ["issue-3"],  # Still available after issue-1 completes
                [],
            ]
        )
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(max_agents=2),
        )

        spawned_order: list[str] = []
        pending_events: dict[str, asyncio.Event] = {}

        async def spawn_callback(issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
            spawned_order.append(issue_id)
            event = asyncio.Event()
            pending_events[issue_id] = event

            async def work() -> None:
                await event.wait()

            task = asyncio.create_task(work())
            return task

        async def finalize_callback(issue_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
            coord.mark_completed(issue_id)

        # Start the loop in a separate task
        loop_task = asyncio.create_task(
            coord.run_loop(spawn_callback, finalize_callback, AsyncMock())
        )

        # Wait for initial spawns
        await asyncio.sleep(0.01)

        # Should have spawned 2 (not 3) due to max_agents
        assert len(spawned_order) == 2
        assert "issue-1" in spawned_order
        assert "issue-2" in spawned_order
        assert len(coord.active_tasks) == 2

        # Complete one task to allow third to spawn
        pending_events["issue-1"].set()
        await asyncio.sleep(0.01)

        # Now issue-3 should spawn
        assert "issue-3" in spawned_order
        assert len(spawned_order) == 3

        # Complete remaining
        pending_events["issue-2"].set()
        pending_events["issue-3"].set()

        await loop_task

    @pytest.mark.asyncio
    async def test_respects_max_issues(self, event_sink: MockEventSink) -> None:
        """Loop exits with limit_reached after max_issues completions."""
        # max_issues counts terminal states (completions), not spawn attempts
        # Use max_agents=1 to ensure sequential execution so limit_reached is checked
        # between completions, preventing spawning the 3rd issue.
        # Each poll returns remaining issues so they're available for subsequent spawns.
        beads = MockIssueProvider(
            ready_issues=[
                ["issue-1", "issue-2", "issue-3"],  # First poll
                ["issue-2", "issue-3"],  # After issue-1 excluded
                ["issue-3"],  # After issue-2 excluded
            ]
        )
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(max_issues=2, max_agents=1),
        )

        completed_count = 0

        async def spawn_callback(issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
            async def work() -> None:
                pass

            task = asyncio.create_task(work())
            return task

        async def finalize_callback(issue_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
            nonlocal completed_count
            coord.mark_completed(issue_id)
            completed_count += 1

        result = await coord.run_loop(spawn_callback, finalize_callback, AsyncMock())

        # With max_agents=1, issues run sequentially. After 2 complete, limit_reached
        # prevents spawning the 3rd issue.
        assert completed_count == 2
        assert result.exit_reason == "limit_reached"
        assert ("no_more_issues", ("limit_reached (2)",)) in event_sink.events

    @pytest.mark.asyncio
    async def test_spawn_failure_marks_failed(self, event_sink: MockEventSink) -> None:
        """Failed spawn marks issue as failed."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
            coord.mark_failed(issue_id)
            return None

        result = await coord.run_loop(spawn_callback, AsyncMock(), AsyncMock())

        assert result.issues_spawned == 0
        assert "issue-1" in coord.failed_issues

    @pytest.mark.asyncio
    async def test_abort_triggers_callback(self, event_sink: MockEventSink) -> None:
        """Abort during loop triggers abort callback."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(),
        )

        abort_called = False
        tasks_to_cancel: list[asyncio.Task] = []  # type: ignore[type-arg]

        async def spawn_callback(issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
            async def work() -> None:
                # Short-lived work that will complete quickly
                await asyncio.sleep(0.001)

            task = asyncio.create_task(work())
            tasks_to_cancel.append(task)
            return task

        async def finalize_callback(issue_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
            coord.mark_completed(issue_id)
            # Trigger abort after first issue completes
            coord.request_abort("test abort")

        async def abort_callback(*, is_interrupt: bool = False) -> AbortResult:
            nonlocal abort_called
            abort_called = True
            # Cancel any remaining tasks
            count = len(coord.active_tasks)
            for task in coord.active_tasks.values():
                task.cancel()
            coord.active_tasks.clear()
            return AbortResult(aborted_count=count, has_unresponsive_tasks=False)

        await coord.run_loop(spawn_callback, finalize_callback, abort_callback)

        assert abort_called
        assert ("abort_requested", ("test abort",)) in event_sink.events


class CapturingIssueProvider:
    """Issue provider that captures get_ready_async call arguments."""

    def __init__(self) -> None:
        self.captured_kwargs: dict[str, object] = {}

    async def get_ready_async(
        self,
        exclude_ids: set[str] | None = None,
        epic_id: str | None = None,
        only_ids: list[str] | None = None,
        suppress_warn_ids: set[str] | None = None,
        include_wip: bool = False,
        focus: bool = True,
        orphans_only: bool = False,
        order_preference: OrderPreference = OrderPreference.FOCUS,
    ) -> list[str]:
        """Capture kwargs and return empty list."""
        self.captured_kwargs = {
            "exclude_ids": exclude_ids,
            "epic_id": epic_id,
            "only_ids": only_ids,
            "suppress_warn_ids": suppress_warn_ids,
            "include_wip": include_wip,
            "focus": focus,
            "orphans_only": orphans_only,
            "order_preference": order_preference,
        }
        return []


class TestOnlyIdsFiltering:
    """Tests for only_ids filtering behavior."""

    @pytest.fixture
    def event_sink(self) -> MockEventSink:
        return MockEventSink()

    @pytest.mark.asyncio
    async def test_only_ids_passed_to_provider(self, event_sink: MockEventSink) -> None:
        """only_ids config is passed to issue provider."""
        beads = CapturingIssueProvider()

        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(only_ids=["issue-1", "issue-2"]),
        )

        await coord.run_loop(AsyncMock(), AsyncMock(), AsyncMock())

        assert beads.captured_kwargs["only_ids"] == ["issue-1", "issue-2"]


class TestEpicIdFiltering:
    """Tests for epic_id filtering behavior."""

    @pytest.fixture
    def event_sink(self) -> MockEventSink:
        return MockEventSink()

    @pytest.mark.asyncio
    async def test_epic_id_passed_to_provider(self, event_sink: MockEventSink) -> None:
        """epic_id config is passed to issue provider."""
        beads = CapturingIssueProvider()

        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(epic_id="epic-123"),
        )

        await coord.run_loop(AsyncMock(), AsyncMock(), AsyncMock())

        assert beads.captured_kwargs["epic_id"] == "epic-123"


class TestDrainMode:
    """Tests for drain_event handling (Stage 1 SIGINT)."""

    @pytest.fixture
    def event_sink(self) -> MockEventSink:
        return MockEventSink()

    @pytest.mark.asyncio
    async def test_drain_with_no_active_tasks_returns_immediately(
        self, event_sink: MockEventSink
    ) -> None:
        """Drain mode with no active tasks returns immediately with drained reason."""
        beads = MockIssueProvider(ready_issues=[["issue-1"]])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(),
        )

        drain_event = asyncio.Event()
        drain_event.set()  # Already draining

        result = await coord.run_loop(
            AsyncMock(return_value=None),  # spawn fails, no active tasks
            AsyncMock(),
            AsyncMock(),
            drain_event=drain_event,
        )

        assert result.exit_reason == "drained"
        assert result.exit_code == 0
        assert result.issues_spawned == 0

    @pytest.mark.asyncio
    async def test_drain_stops_spawning_new_issues(
        self, event_sink: MockEventSink
    ) -> None:
        """Drain mode stops spawning new issues but lets active tasks complete."""
        beads = MockIssueProvider(
            ready_issues=[["issue-1", "issue-2"], ["issue-2"], []]
        )
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(max_agents=1),  # Limit to 1 concurrent
        )

        spawned_ids: list[str] = []
        pending_events: dict[str, asyncio.Event] = {}
        drain_event = asyncio.Event()

        async def spawn_callback(issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
            spawned_ids.append(issue_id)
            event = asyncio.Event()
            pending_events[issue_id] = event

            async def work() -> None:
                await event.wait()

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
            coord.mark_completed(issue_id)

        loop_task = asyncio.create_task(
            coord.run_loop(
                spawn_callback,
                finalize_callback,
                AsyncMock(),
                drain_event=drain_event,
            )
        )

        # Wait for first issue to spawn
        await asyncio.sleep(0.01)
        assert "issue-1" in spawned_ids

        # Set drain mode before issue-1 completes
        drain_event.set()

        # Complete issue-1
        pending_events["issue-1"].set()

        # Wait for loop to complete
        result = await loop_task

        # Only issue-1 should have been spawned (drain prevented issue-2)
        assert spawned_ids == ["issue-1"]
        assert result.exit_reason == "drained"
        assert result.issues_spawned == 1

    @pytest.mark.asyncio
    async def test_drain_waits_for_active_tasks_to_complete(
        self, event_sink: MockEventSink
    ) -> None:
        """Drain mode waits for active tasks to complete (no cancellation)."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(),
        )

        task_completed = False
        pending_event = asyncio.Event()
        drain_event = asyncio.Event()

        async def spawn_callback(issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
            async def work() -> None:
                nonlocal task_completed
                await pending_event.wait()
                task_completed = True

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
            coord.mark_completed(issue_id)

        loop_task = asyncio.create_task(
            coord.run_loop(
                spawn_callback,
                finalize_callback,
                AsyncMock(),
                drain_event=drain_event,
            )
        )

        await asyncio.sleep(0.01)
        drain_event.set()  # Enter drain mode

        # Task should still be running (not cancelled)
        assert not task_completed

        # Complete the task
        pending_event.set()
        await loop_task

        # Task completed normally (not cancelled)
        assert task_completed

    @pytest.mark.asyncio
    async def test_drain_triggers_validation_callback(
        self, event_sink: MockEventSink
    ) -> None:
        """Drain completion triggers validation callback when provided."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(),
        )

        validation_called = False
        pending_event = asyncio.Event()
        drain_event = asyncio.Event()

        async def spawn_callback(issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
            async def work() -> None:
                await pending_event.wait()

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
            coord.mark_completed(issue_id)

        async def validation_callback() -> bool:
            nonlocal validation_called
            validation_called = True
            return True

        # Create a mock WatchConfig that acts like enabled=True
        # The coordinator checks watch_enabled = bool(watch_config and watch_config.enabled)
        mock_watch_config = MagicMock()
        mock_watch_config.enabled = True

        loop_task = asyncio.create_task(
            coord.run_loop(
                spawn_callback,
                finalize_callback,
                AsyncMock(),
                watch_config=mock_watch_config,
                validation_config=PeriodicValidationConfig(validate_every=10),
                validation_callback=validation_callback,
                drain_event=drain_event,
            )
        )

        await asyncio.sleep(0.01)
        drain_event.set()
        pending_event.set()

        result = await loop_task

        assert validation_called
        assert result.exit_reason == "drained"
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_drain_validation_failure_returns_exit_code_1(
        self, event_sink: MockEventSink
    ) -> None:
        """Drain validation failure returns exit_code=1."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(),
        )

        pending_event = asyncio.Event()
        drain_event = asyncio.Event()

        async def spawn_callback(issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
            async def work() -> None:
                await pending_event.wait()

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
            coord.mark_completed(issue_id)

        async def validation_callback() -> bool:
            return False  # Validation fails

        # Create a mock WatchConfig that acts like enabled=True
        mock_watch_config = MagicMock()
        mock_watch_config.enabled = True

        loop_task = asyncio.create_task(
            coord.run_loop(
                spawn_callback,
                finalize_callback,
                AsyncMock(),
                watch_config=mock_watch_config,
                validation_config=PeriodicValidationConfig(validate_every=10),
                validation_callback=validation_callback,
                drain_event=drain_event,
            )
        )

        await asyncio.sleep(0.01)
        drain_event.set()
        pending_event.set()

        result = await loop_task

        assert result.exit_reason == "validation_failed"
        assert result.exit_code == 1

    @pytest.mark.asyncio
    async def test_interrupt_takes_precedence_over_drain(
        self, event_sink: MockEventSink
    ) -> None:
        """interrupt_event takes precedence over drain_event."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(),
        )

        pending_event = asyncio.Event()
        drain_event = asyncio.Event()
        interrupt_event = asyncio.Event()
        abort_called = False

        async def spawn_callback(issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
            async def work() -> None:
                await pending_event.wait()

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
            coord.mark_completed(issue_id)

        async def abort_callback(*, is_interrupt: bool = False) -> AbortResult:
            nonlocal abort_called
            abort_called = True
            tasks = list(coord.active_tasks.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            coord.active_tasks.clear()
            return AbortResult(aborted_count=1, has_unresponsive_tasks=False)

        loop_task = asyncio.create_task(
            coord.run_loop(
                spawn_callback,
                finalize_callback,
                abort_callback,
                interrupt_event=interrupt_event,
                drain_event=drain_event,
            )
        )

        await asyncio.sleep(0.01)

        # Set both events - interrupt should take precedence
        drain_event.set()
        interrupt_event.set()

        result = await loop_task

        assert result.exit_reason == "interrupted"
        assert abort_called

    @pytest.mark.asyncio
    async def test_existing_behavior_unchanged_when_drain_none(
        self, event_sink: MockEventSink
    ) -> None:
        """Existing behavior unchanged when drain_event is None."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
            async def work() -> None:
                pass

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
            coord.mark_completed(issue_id)

        result = await coord.run_loop(
            spawn_callback,
            finalize_callback,
            AsyncMock(),
            drain_event=None,  # Explicitly None
        )

        assert result.exit_reason == "success"
        assert result.issues_spawned == 1
        assert "issue-1" in coord.completed_ids

    @pytest.mark.asyncio
    async def test_on_validation_failed_callback_invoked(
        self, event_sink: MockEventSink
    ) -> None:
        """on_validation_failed callback is invoked when validation fails."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(),
        )

        validation_failed_called = False

        async def spawn_callback(issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
            async def work() -> None:
                pass

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
            coord.mark_completed(issue_id)

        async def validation_callback() -> bool:
            return False  # Validation fails

        def on_validation_failed() -> None:
            nonlocal validation_failed_called
            validation_failed_called = True

        result = await coord.run_loop(
            spawn_callback,
            finalize_callback,
            AsyncMock(),
            validation_config=PeriodicValidationConfig(validate_every=10),
            validation_callback=validation_callback,
            on_validation_failed=on_validation_failed,
        )

        assert result.exit_reason == "validation_failed"
        assert result.exit_code == 1
        assert validation_failed_called


class TestInterruptWithUnresponsiveTasks:
    """Tests for interrupt handling when tasks are unresponsive."""

    @pytest.fixture
    def event_sink(self) -> MockEventSink:
        return MockEventSink()

    @pytest.mark.asyncio
    async def test_skips_validation_when_tasks_unresponsive(
        self, event_sink: MockEventSink
    ) -> None:
        """Validation is skipped when abort reports unresponsive tasks."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(),
        )

        pending_event = asyncio.Event()
        interrupt_event = asyncio.Event()
        validation_called = False

        async def spawn_callback(issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
            async def work() -> None:
                await pending_event.wait()

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
            coord.mark_completed(issue_id)

        async def abort_callback(*, is_interrupt: bool = False) -> AbortResult:
            # Simulate unresponsive tasks scenario
            tasks = list(coord.active_tasks.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            coord.active_tasks.clear()
            return AbortResult(aborted_count=1, has_unresponsive_tasks=True)

        async def validation_callback() -> bool:
            nonlocal validation_called
            validation_called = True
            return False  # Would fail validation if called

        loop_task = asyncio.create_task(
            coord.run_loop(
                spawn_callback,
                finalize_callback,
                abort_callback,
                interrupt_event=interrupt_event,
                validation_callback=validation_callback,
            )
        )

        await asyncio.sleep(0.01)
        interrupt_event.set()

        result = await loop_task

        # Validation should be skipped due to unresponsive tasks
        assert validation_called is False
        # Exit code should be 130 (not 1 from failed validation)
        assert result.exit_code == 130
        assert result.exit_reason == "interrupted"

    @pytest.mark.asyncio
    async def test_no_validation_when_tasks_responsive(
        self, event_sink: MockEventSink
    ) -> None:
        """Stage 2 interrupt does NOT run validation - exit code determined by orchestrator.

        Per spec, Stage 2 (graceful abort) exit code is determined by the orchestrator's
        _abort_exit_code snapshot at Stage 2 entry, not by validation during abort.
        The coordinator always returns exit_code=130 for interrupts.
        """
        beads = MockIssueProvider(ready_issues=[["issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]
            event_sink=event_sink,  # type: ignore[arg-type]
            config=CoordinatorConfig(),
        )

        pending_event = asyncio.Event()
        interrupt_event = asyncio.Event()
        validation_called = False

        async def spawn_callback(issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
            async def work() -> None:
                await pending_event.wait()

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
            coord.mark_completed(issue_id)

        async def abort_callback(*, is_interrupt: bool = False) -> AbortResult:
            # All tasks respond normally
            tasks = list(coord.active_tasks.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            coord.active_tasks.clear()
            return AbortResult(aborted_count=1, has_unresponsive_tasks=False)

        async def validation_callback() -> bool:
            nonlocal validation_called
            validation_called = True
            return False  # Would fail if called

        loop_task = asyncio.create_task(
            coord.run_loop(
                spawn_callback,
                finalize_callback,
                abort_callback,
                interrupt_event=interrupt_event,
                validation_callback=validation_callback,
            )
        )

        await asyncio.sleep(0.01)
        interrupt_event.set()

        result = await loop_task

        # Validation should NOT run during Stage 2 interrupt
        assert validation_called is False
        # Coordinator always returns 130 for interrupt; orchestrator overrides with _abort_exit_code
        assert result.exit_code == 130
        assert result.exit_reason == "interrupted"
