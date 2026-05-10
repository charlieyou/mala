"""Unit tests for IssueExecutionCoordinator.

These tests verify the coordinator's loop logic without SDK dependencies.
"""

import asyncio
import ast
import inspect
import textwrap
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.models import OrderPreference, PeriodicValidationConfig
import src.pipeline.issue_execution_coordinator as coordinator_module
from src.pipeline.issue_execution_coordinator import (
    AbortResult,
    CoordinatorConfig,
    IssueExecutionCoordinator,
)
from src.pipeline.work_queue import (
    PollDecision,
    PollResult,
    WorkQueueConfig,
    WorkQueueSnapshot,
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

    def on_watch_idle(self, wait_seconds: float, issues_blocked: int | None) -> None:
        self.events.append(("watch_idle", (wait_seconds, issues_blocked)))


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
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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

    def test_release_task(self, coordinator: IssueExecutionCoordinator) -> None:
        """release_task removes active work without marking the issue completed."""
        task = MagicMock()
        coordinator.active_tasks["issue-1"] = task

        coordinator.release_task("issue-1")

        assert "issue-1" not in coordinator.active_tasks
        assert "issue-1" not in coordinator.completed_ids

    def test_register_task(self, coordinator: IssueExecutionCoordinator) -> None:
        """register_task adds to active_tasks."""
        task = MagicMock()
        coordinator.register_task("issue-1", task)
        assert coordinator.active_tasks["issue-1"] is task

    @pytest.mark.asyncio
    async def test_run_loop_port_uses_passed_interrupt_event(
        self, coordinator: IssueExecutionCoordinator
    ) -> None:
        """run_loop wires the passed lifecycle interrupt event into the port."""
        interrupt_event = asyncio.Event()
        interrupt_event.set()
        abort_callback = AsyncMock(return_value=AbortResult(aborted_count=0))

        result = await coordinator.run_loop(
            AsyncMock(),
            AsyncMock(),
            abort_callback,
            interrupt_event=interrupt_event,
        )

        assert result.exit_reason == "interrupted"
        assert coordinator.interrupt_event is interrupt_event
        assert coordinator.current_state.interrupt_requested is True


class TestRunLoop:
    """Tests for run_loop execution flow."""

    @pytest.fixture
    def event_sink(self) -> MockEventSink:
        return MockEventSink()

    def test_run_loop_body_does_not_sleep_directly(self) -> None:
        """Polling waits live behind WorkQueue/PollStrategy effects."""
        source = textwrap.dedent(inspect.getsource(IssueExecutionCoordinator.run_loop))
        tree = ast.parse(source)
        calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]

        for func in calls:
            if not isinstance(func, ast.Attribute):
                continue
            assert func.attr != "sleep"

    @pytest.mark.asyncio
    async def test_run_loop_polls_through_work_queue(
        self, event_sink: MockEventSink, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_loop gets ready work from WorkQueue instead of provider polling."""

        class FakeWorkQueue:
            polls = 0

            def __init__(
                self, _beads: object, config: WorkQueueConfig, _strategy: object
            ) -> None:
                self.config = config

            def snapshot(self, **kwargs: object) -> WorkQueueSnapshot:
                return WorkQueueSnapshot(
                    active_issue_ids=frozenset(
                        cast("set[str]", kwargs["active_issue_ids"])
                    ),
                    completed_issue_ids=frozenset(
                        cast("set[str]", kwargs["completed_issue_ids"])
                    ),
                    failed_issue_ids=frozenset(
                        cast("set[str]", kwargs["failed_issue_ids"])
                    ),
                    completed_count=cast("int", kwargs["completed_count"]),
                    last_validation_at=cast("int", kwargs["last_validation_at"]),
                    next_validation_threshold=cast(
                        "int | None", kwargs["next_validation_threshold"]
                    ),
                    consecutive_poll_failures=cast(
                        "int", kwargs["consecutive_poll_failures"]
                    ),
                    watch_enabled=cast("bool", kwargs["watch_enabled"]),
                    startup_no_ready_check_pending=cast(
                        "bool", kwargs["startup_no_ready_check_pending"]
                    ),
                    abort_requested=cast("bool", kwargs["abort_requested"]),
                    interrupt_requested=cast("bool", kwargs["interrupt_requested"]),
                    is_draining=cast("bool", kwargs["is_draining"]),
                    max_agents=self.config.max_agents,
                    max_issues=self.config.max_issues,
                ).with_ready(cast("list[str]", kwargs["ready_issue_ids"]))

            async def poll(self, snapshot: WorkQueueSnapshot) -> PollResult:
                type(self).polls += 1
                return PollResult(
                    snapshot=snapshot.with_ready(["issue-from-queue"]),
                    poll_attempted=True,
                    decision=PollDecision("ready"),
                )

        beads = MockIssueProvider(ready_issues=[["issue-from-provider"]])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            config=CoordinatorConfig(max_issues=1),
        )
        monkeypatch.setattr(coordinator_module, "WorkQueue", FakeWorkQueue)

        async def spawn_callback(issue_id: str) -> asyncio.Task[None] | None:
            async def work() -> None:
                pass

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        result = await coord.run_loop(
            spawn_callback,
            finalize_callback,
            AsyncMock(),
        )

        assert FakeWorkQueue.polls == 1
        assert result.issues_spawned == 1
        assert coord.completed_ids == {"issue-from-queue"}

    @pytest.mark.asyncio
    async def test_exits_when_no_ready_issues(self, event_sink: MockEventSink) -> None:
        """Loop exits immediately when no issues are ready."""
        beads = MockIssueProvider(ready_issues=[[]])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
    async def test_repolls_after_startup_no_ready_work(
        self, event_sink: MockEventSink
    ) -> None:
        """Startup no-ready work can create issues for the loop to process."""
        beads = MockIssueProvider(ready_issues=[[], ["remediation-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            config=CoordinatorConfig(),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None] | None:
            async def work() -> None:
                pass

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        startup_checks = 0

        async def on_startup_no_ready() -> bool:
            nonlocal startup_checks
            startup_checks += 1
            return True

        result = await coord.run_loop(
            spawn_callback,
            finalize_callback,
            AsyncMock(),
            on_startup_no_ready=on_startup_no_ready,
        )

        assert result.issues_spawned == 1
        assert "remediation-1" in coord.completed_ids
        assert startup_checks == 1

    @pytest.mark.asyncio
    async def test_exits_when_startup_no_ready_work_has_no_changes(
        self, event_sink: MockEventSink
    ) -> None:
        """Startup no-ready work can decline a repoll and exit normally."""
        beads = MockIssueProvider(ready_issues=[[]])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            config=CoordinatorConfig(),
        )

        startup_checks = 0

        async def on_startup_no_ready() -> bool:
            nonlocal startup_checks
            startup_checks += 1
            return False

        result = await coord.run_loop(
            AsyncMock(return_value=None),
            AsyncMock(),
            AsyncMock(),
            on_startup_no_ready=on_startup_no_ready,
        )

        assert result.issues_spawned == 0
        assert coord.completed_ids == set()
        assert startup_checks == 1
        assert ("no_more_issues", ("none_ready",)) in event_sink.events

    @pytest.mark.asyncio
    async def test_startup_no_ready_work_skipped_when_first_poll_has_ready_issue(
        self, event_sink: MockEventSink
    ) -> None:
        """Startup no-ready work only runs when the first poll is empty."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            config=CoordinatorConfig(),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None] | None:
            async def work() -> None:
                pass

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        startup_checks = 0

        async def on_startup_no_ready() -> bool:
            nonlocal startup_checks
            startup_checks += 1
            return True

        result = await coord.run_loop(
            spawn_callback,
            finalize_callback,
            AsyncMock(),
            on_startup_no_ready=on_startup_no_ready,
        )

        assert result.issues_spawned == 1
        assert "issue-1" in coord.completed_ids
        assert startup_checks == 0

    @pytest.mark.asyncio
    async def test_spawns_single_issue(self, event_sink: MockEventSink) -> None:
        """Loop spawns and completes a single issue."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
    async def test_spawn_capacity_counts_successful_spawns(
        self, event_sink: MockEventSink
    ) -> None:
        """Spawn failures do not consume available agent capacity."""
        beads = MockIssueProvider(ready_issues=[["locked", "issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            config=CoordinatorConfig(max_agents=1),
        )
        attempted: list[str] = []

        async def spawn_callback(issue_id: str) -> asyncio.Task[None] | None:
            attempted.append(issue_id)
            if issue_id == "locked":
                return None

            async def work() -> None:
                pass

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        result = await coord.run_loop(
            spawn_callback,
            finalize_callback,
            AsyncMock(),
        )

        assert attempted == ["locked", "issue-1"]
        assert result.issues_spawned == 1
        assert coord.completed_ids == {"issue-1"}

    @pytest.mark.asyncio
    async def test_spawn_capacity_skips_active_ready_ids(
        self, event_sink: MockEventSink
    ) -> None:
        """Already-active ready IDs do not consume available agent capacity."""
        coordinator = IssueExecutionCoordinator(
            beads=MockIssueProvider(),  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            config=CoordinatorConfig(),
        )
        active_task = asyncio.create_task(asyncio.Event().wait())
        spawned_task: asyncio.Task[None] | None = None
        coordinator.register_task("active", active_task)
        attempted: list[str] = []

        async def spawn_callback(issue_id: str) -> asyncio.Task[None] | None:
            nonlocal spawned_task
            attempted.append(issue_id)

            async def work() -> None:
                pass

            spawned_task = asyncio.create_task(work())
            return spawned_task

        try:
            spawned, remaining = await coordinator._spawn_ready_issues(
                ["active", "issue-1", "issue-2"],
                capacity=1,
                spawn_callback=spawn_callback,
            )
        finally:
            active_task.cancel()
            await asyncio.gather(active_task, return_exceptions=True)
            if spawned_task is not None:
                await spawned_task

        assert attempted == ["issue-1"]
        assert spawned == 1
        assert remaining == ["issue-2"]
        assert "issue-1" in coordinator.active_tasks

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
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
    async def test_runtime_max_issues_update_stops_next_poll(
        self, event_sink: MockEventSink
    ) -> None:
        """Lifecycle max_issues updates are visible to subsequent queue snapshots."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], ["issue-2"]])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            config=CoordinatorConfig(max_issues=None, max_agents=1),
        )
        spawned_ids: list[str] = []

        async def spawn_callback(issue_id: str) -> asyncio.Task[None] | None:
            spawned_ids.append(issue_id)

            async def work() -> None:
                pass

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)
            coord.max_issues = 1

        result = await coord.run_loop(spawn_callback, finalize_callback, AsyncMock())

        assert spawned_ids == ["issue-1"]
        assert result.exit_reason == "limit_reached"
        assert ("no_more_issues", ("limit_reached (1)",)) in event_sink.events

    @pytest.mark.asyncio
    async def test_released_attempt_does_not_count_toward_max_issues(
        self, event_sink: MockEventSink
    ) -> None:
        """A released attempt stays retryable and does not consume max_issues."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], ["issue-1"]])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            config=CoordinatorConfig(max_issues=1, max_agents=1),
        )

        attempts = 0

        async def spawn_callback(issue_id: str) -> asyncio.Task[None] | None:
            async def work() -> None:
                pass

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                coord.release_task(issue_id)
            else:
                coord.mark_completed(issue_id)

        result = await coord.run_loop(spawn_callback, finalize_callback, AsyncMock())

        assert attempts == 2
        assert result.issues_spawned == 2
        assert result.exit_reason == "limit_reached"
        assert coord.completed_ids == {"issue-1"}

    @pytest.mark.asyncio
    async def test_spawn_failure_marks_failed(self, event_sink: MockEventSink) -> None:
        """Failed spawn marks issue as failed."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
    async def test_interrupt_after_completion_preempts_periodic_validation(
        self, event_sink: MockEventSink
    ) -> None:
        """Interrupts observed after a task wait skip periodic validation."""
        beads = MockIssueProvider(ready_issues=[["issue-1", "issue-2"]])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            config=CoordinatorConfig(max_agents=2),
        )
        pending_event = asyncio.Event()
        interrupt_event = asyncio.Event()
        validation_called = False

        async def spawn_callback(issue_id: str) -> asyncio.Task[None] | None:
            async def work() -> None:
                if issue_id == "issue-2":
                    await pending_event.wait()

            return asyncio.create_task(work())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)
            if issue_id == "issue-1":
                interrupt_event.set()

        async def abort_callback(*, is_interrupt: bool = False) -> AbortResult:
            tasks = list(coord.active_tasks.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            coord.active_tasks.clear()
            return AbortResult(
                aborted_count=len(tasks),
                has_unresponsive_tasks=False,
            )

        async def validation_callback() -> bool:
            nonlocal validation_called
            validation_called = True
            return True

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback,
                finalize_callback,
                abort_callback,
                validation_config=PeriodicValidationConfig(validate_every=1),
                interrupt_event=interrupt_event,
                validation_callback=validation_callback,
            ),
            timeout=1,
        )

        assert result.exit_reason == "interrupted"
        assert validation_called is False

    @pytest.mark.asyncio
    async def test_existing_behavior_unchanged_when_drain_none(
        self, event_sink: MockEventSink
    ) -> None:
        """Existing behavior unchanged when drain_event is None."""
        beads = MockIssueProvider(ready_issues=[["issue-1"], []])
        coord = IssueExecutionCoordinator(
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
            beads=beads,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            event_sink=event_sink,  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
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
