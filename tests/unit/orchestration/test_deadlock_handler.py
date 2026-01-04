"""Unit tests for DeadlockHandler service."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.deadlock import DeadlockInfo
from src.orchestration.deadlock_handler import DeadlockHandler, DeadlockHandlerCallbacks
from src.orchestration.orchestrator_state import OrchestratorState
from src.pipeline.issue_result import IssueResult

if TYPE_CHECKING:
    from src.infra.io.log_output.run_metadata import RunMetadata


class MockCallbacks:
    """Mock callbacks wrapper with proper typing for test assertions."""

    def __init__(self) -> None:
        self.add_dependency: Any = AsyncMock(return_value=True)
        self.mark_needs_followup: Any = AsyncMock(return_value=True)
        self.on_deadlock_detected: Any = MagicMock()
        self.on_locks_cleaned: Any = MagicMock()
        self.on_tasks_aborting: Any = MagicMock()
        self.do_cleanup_agent_locks: Any = MagicMock(
            return_value=(1, ["/path/to/lock"])
        )
        self.unregister_agent: Any = MagicMock()
        self.finalize_issue_result: Any = AsyncMock()
        self.mark_completed: Any = MagicMock()

    def as_callbacks(self) -> DeadlockHandlerCallbacks:
        """Convert to DeadlockHandlerCallbacks."""
        return DeadlockHandlerCallbacks(
            add_dependency=self.add_dependency,
            mark_needs_followup=self.mark_needs_followup,
            on_deadlock_detected=self.on_deadlock_detected,
            on_locks_cleaned=self.on_locks_cleaned,
            on_tasks_aborting=self.on_tasks_aborting,
            do_cleanup_agent_locks=self.do_cleanup_agent_locks,
            unregister_agent=self.unregister_agent,
            finalize_issue_result=self.finalize_issue_result,
            mark_completed=self.mark_completed,
        )


@pytest.fixture
def mock_callbacks() -> MockCallbacks:
    """Create mock callbacks for DeadlockHandler."""
    return MockCallbacks()


@pytest.fixture
def handler(mock_callbacks: MockCallbacks) -> DeadlockHandler:
    """Create DeadlockHandler with mock callbacks."""
    return DeadlockHandler(callbacks=mock_callbacks.as_callbacks())


@pytest.fixture
def state() -> OrchestratorState:
    """Create OrchestratorState for tests."""
    return OrchestratorState()


@pytest.fixture
def deadlock_info() -> DeadlockInfo:
    """Create sample DeadlockInfo for tests."""
    return DeadlockInfo(
        cycle=["agent-a", "agent-b"],
        victim_id="agent-b",
        victim_issue_id="issue-b",
        blocked_on="/path/lock.py",
        blocker_id="agent-a",
        blocker_issue_id="issue-a",
    )


@pytest.fixture
def mock_run_metadata() -> RunMetadata:
    """Create mock RunMetadata for tests."""
    return MagicMock()


class TestHandleDeadlock:
    """Tests for handle_deadlock method."""

    @pytest.mark.asyncio
    async def test_acquires_lock_and_calls_callbacks_in_order(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
        state: OrchestratorState,
        deadlock_info: DeadlockInfo,
    ) -> None:
        """handle_deadlock acquires lock and calls callbacks in correct order."""
        # Use a shared call_order list to track callback sequence
        call_order: list[str] = []
        mock_callbacks.on_deadlock_detected.side_effect = lambda _: call_order.append(
            "on_deadlock_detected"
        )
        mock_callbacks.do_cleanup_agent_locks.side_effect = lambda _: (
            call_order.append("do_cleanup_agent_locks"),
            (1, ["/path/to/lock"]),
        )[1]
        mock_callbacks.on_locks_cleaned.side_effect = lambda *_: call_order.append(
            "on_locks_cleaned"
        )

        async def track_add_dependency(*args: object, **kwargs: object) -> bool:
            call_order.append("add_dependency")
            return True

        async def track_mark_needs_followup(*args: object, **kwargs: object) -> bool:
            call_order.append("mark_needs_followup")
            return True

        mock_callbacks.add_dependency.side_effect = track_add_dependency
        mock_callbacks.mark_needs_followup.side_effect = track_mark_needs_followup
        # Recreate handler with updated callbacks
        handler = DeadlockHandler(callbacks=mock_callbacks.as_callbacks())

        active_tasks: dict[str, asyncio.Task[IssueResult]] = {}

        await handler.handle_deadlock(deadlock_info, state, active_tasks)

        # Verify callback order: on_deadlock_detected -> cleanup -> resolve
        assert call_order == [
            "on_deadlock_detected",
            "do_cleanup_agent_locks",
            "on_locks_cleaned",
            "add_dependency",
            "mark_needs_followup",
        ]

        # Verify key callbacks were called with correct arguments
        # Use .args/.kwargs to be robust against positional vs keyword argument calls
        mock_callbacks.add_dependency.assert_awaited_once()
        add_dep_args = mock_callbacks.add_dependency.await_args
        assert add_dep_args is not None
        # add_dependency signature: (dependent_id, dependency_id)
        victim_arg = (
            add_dep_args.kwargs["dependent_id"]
            if "dependent_id" in add_dep_args.kwargs
            else add_dep_args.args[0]
        )
        blocker_arg = (
            add_dep_args.kwargs["dependency_id"]
            if "dependency_id" in add_dep_args.kwargs
            else add_dep_args.args[1]
        )
        assert victim_arg == deadlock_info.victim_issue_id
        assert blocker_arg == deadlock_info.blocker_issue_id

        mock_callbacks.mark_needs_followup.assert_awaited_once()
        # Check mark_needs_followup was called with victim issue_id
        # mark_needs_followup signature: (issue_id, summary, log_path)
        followup_args = mock_callbacks.mark_needs_followup.await_args
        assert followup_args is not None
        issue_id_arg = (
            followup_args.kwargs["issue_id"]
            if "issue_id" in followup_args.kwargs
            else followup_args.args[0]
        )
        assert issue_id_arg == deadlock_info.victim_issue_id

    @pytest.mark.asyncio
    async def test_cleans_up_locks_and_tracks_in_state(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
        state: OrchestratorState,
        deadlock_info: DeadlockInfo,
    ) -> None:
        """handle_deadlock cleans up locks and tracks in state."""
        active_tasks: dict[str, asyncio.Task[IssueResult]] = {}

        await handler.handle_deadlock(deadlock_info, state, active_tasks)

        # Verify agent tracked as cleaned
        assert "agent-b" in state.deadlock_cleaned_agents
        mock_callbacks.do_cleanup_agent_locks.assert_called_once_with("agent-b")
        mock_callbacks.unregister_agent.assert_called_once_with("agent-b")

    @pytest.mark.asyncio
    async def test_cancels_victim_task_non_self(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
        state: OrchestratorState,
        deadlock_info: DeadlockInfo,
    ) -> None:
        """handle_deadlock cancels victim task when not self-cancellation."""
        # Create a task that will be cancelled
        cancelled = asyncio.Event()

        async def victim_work() -> IssueResult:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return IssueResult(
                issue_id="issue-b", agent_id="agent-b", success=True, summary="done"
            )

        victim_task = asyncio.create_task(victim_work())
        active_tasks = {"issue-b": victim_task}

        await handler.handle_deadlock(deadlock_info, state, active_tasks)

        # Wait deterministically for cancellation to be processed
        await asyncio.wait_for(cancelled.wait(), timeout=1.0)
        # Ensure task reaches terminal state
        await asyncio.gather(victim_task, return_exceptions=True)
        assert victim_task.cancelled()

    @pytest.mark.asyncio
    async def test_defers_self_cancellation(
        self,
        mock_callbacks: MockCallbacks,
        state: OrchestratorState,
    ) -> None:
        """handle_deadlock defers self-cancellation correctly."""
        handler = DeadlockHandler(callbacks=mock_callbacks.as_callbacks())

        info = DeadlockInfo(
            cycle=["agent-a", "agent-b"],
            victim_id="agent-b",
            victim_issue_id="issue-b",
            blocked_on="/path/lock.py",
            blocker_id="agent-a",
            blocker_issue_id="issue-a",
        )

        self_cancelled = False
        resolution_completed = False

        async def self_cancel_task() -> IssueResult:
            nonlocal self_cancelled, resolution_completed
            try:
                # This task calls handle_deadlock on itself
                active_tasks: dict[str, asyncio.Task[IssueResult]] = {}
                current = asyncio.current_task()
                assert current is not None
                # Use info.victim_issue_id as key to ensure consistency
                active_tasks[info.victim_issue_id] = current  # type: ignore[assignment]

                await handler.handle_deadlock(info, state, active_tasks)
                resolution_completed = True
                # After handle_deadlock returns, the deferred cancellation should fire
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                self_cancelled = True
                raise
            return IssueResult(
                issue_id="issue-b", agent_id="agent-b", success=True, summary="done"
            )

        task = asyncio.create_task(self_cancel_task())

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

        assert resolution_completed, "Resolution should complete before cancellation"
        assert self_cancelled, "Task should be cancelled after resolution"

    @pytest.mark.asyncio
    async def test_shields_resolution_from_cancellation(
        self,
        mock_callbacks: MockCallbacks,
        state: OrchestratorState,
        deadlock_info: DeadlockInfo,
    ) -> None:
        """handle_deadlock shields resolution from cancellation."""
        # Track when shielded work completes
        dependency_called = asyncio.Event()
        resolution_finished = asyncio.Event()

        async def slow_add_dependency(dependent: str, dependency: str) -> bool:
            dependency_called.set()
            await asyncio.sleep(0.05)
            return True

        async def mark_followup_and_signal(
            issue_id: str, reason: str, log_path: Path | None
        ) -> bool:
            resolution_finished.set()
            return True

        mock_callbacks.add_dependency = AsyncMock(side_effect=slow_add_dependency)
        mock_callbacks.mark_needs_followup = AsyncMock(
            side_effect=mark_followup_and_signal
        )
        # Create handler AFTER modifying callbacks
        handler = DeadlockHandler(callbacks=mock_callbacks.as_callbacks())

        active_tasks: dict[str, asyncio.Task[IssueResult]] = {}

        async def run_handler() -> None:
            await handler.handle_deadlock(deadlock_info, state, active_tasks)

        task = asyncio.create_task(run_handler())

        # Wait for resolution to start then cancel
        await dependency_called.wait()
        task.cancel()

        # The handler should complete despite cancellation (due to shield)
        # but will re-raise CancelledError after
        with pytest.raises(asyncio.CancelledError):
            await task

        # Wait for shielded work to complete to avoid cross-test timing issues
        await asyncio.wait_for(resolution_finished.wait(), timeout=1.0)

        # Resolution should have completed (add_dependency was called)
        mock_callbacks.add_dependency.assert_awaited()

    @pytest.mark.asyncio
    async def test_uses_log_path_from_state(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
        state: OrchestratorState,
        deadlock_info: DeadlockInfo,
    ) -> None:
        """handle_deadlock uses session log path from state."""
        log_path = Path("/logs/session.log")
        state.active_session_log_paths["issue-b"] = log_path
        active_tasks: dict[str, asyncio.Task[IssueResult]] = {}

        await handler.handle_deadlock(deadlock_info, state, active_tasks)

        # Verify log path passed to mark_needs_followup
        # mark_needs_followup signature: (issue_id, summary, log_path)
        mock_callbacks.mark_needs_followup.assert_awaited_once()
        followup_args = mock_callbacks.mark_needs_followup.await_args
        assert followup_args is not None
        log_path_arg = (
            followup_args.kwargs["log_path"]
            if "log_path" in followup_args.kwargs
            else followup_args.args[2]
        )
        assert log_path_arg == log_path

    @pytest.mark.asyncio
    async def test_handles_none_issue_ids(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
        state: OrchestratorState,
    ) -> None:
        """handle_deadlock handles None issue IDs gracefully."""
        info = DeadlockInfo(
            cycle=["agent-a", "agent-b"],
            victim_id="agent-b",
            victim_issue_id=None,
            blocked_on="/path/lock.py",
            blocker_id="agent-a",
            blocker_issue_id=None,
        )
        active_tasks: dict[str, asyncio.Task[IssueResult]] = {}

        await handler.handle_deadlock(info, state, active_tasks)

        # add_dependency should not be called with None issue IDs
        mock_callbacks.add_dependency.assert_not_awaited()
        # mark_needs_followup should not be called with None victim_issue_id
        mock_callbacks.mark_needs_followup.assert_not_awaited()


class TestAbortActiveTasks:
    """Tests for abort_active_tasks method."""

    @pytest.mark.asyncio
    async def test_cancels_running_tasks(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
        state: OrchestratorState,
        mock_run_metadata: RunMetadata,
    ) -> None:
        """abort_active_tasks cancels running tasks."""
        task_started = asyncio.Event()

        async def running_task() -> IssueResult:
            task_started.set()
            await asyncio.sleep(10)
            return IssueResult(
                issue_id="issue-1", agent_id="agent-1", success=True, summary="done"
            )

        task = asyncio.create_task(running_task())
        await task_started.wait()
        active_tasks = {"issue-1": task}
        state.agent_ids["issue-1"] = "agent-1"

        await handler.abort_active_tasks(
            active_tasks, "Test abort", state, mock_run_metadata
        )

        # Await task with timeout to avoid hanging if cancellation fails
        await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True), timeout=2.0
        )
        assert task.cancelled() or task.done()
        mock_callbacks.on_tasks_aborting.assert_called_once_with(1, "Test abort")
        mock_callbacks.finalize_issue_result.assert_awaited_once()
        mock_callbacks.mark_completed.assert_called_once_with("issue-1")

    @pytest.mark.asyncio
    async def test_uses_real_results_for_completed_tasks(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
        state: OrchestratorState,
        mock_run_metadata: RunMetadata,
    ) -> None:
        """abort_active_tasks uses real results for already completed tasks."""

        async def completed_task() -> IssueResult:
            return IssueResult(
                issue_id="issue-1",
                agent_id="agent-1",
                success=True,
                summary="Completed successfully",
            )

        task = asyncio.create_task(completed_task())
        await task  # Let it complete
        active_tasks = {"issue-1": task}
        state.agent_ids["issue-1"] = "agent-1"

        await handler.abort_active_tasks(
            active_tasks, "Test abort", state, mock_run_metadata
        )

        # Should use the real result, not an aborted result
        # finalize_issue_result signature: (issue_id, result, run_metadata)
        finalize_args = mock_callbacks.finalize_issue_result.call_args
        assert finalize_args is not None
        result = (
            finalize_args.kwargs["result"]
            if "result" in finalize_args.kwargs
            else finalize_args.args[1]
        )
        assert result.success is True
        assert result.summary == "Completed successfully"

    @pytest.mark.asyncio
    async def test_handles_task_exception(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
        state: OrchestratorState,
        mock_run_metadata: RunMetadata,
    ) -> None:
        """abort_active_tasks handles tasks that raised exceptions."""

        async def failing_task() -> IssueResult:
            raise ValueError("Task failed")

        task = asyncio.create_task(failing_task())
        # Ensure task completes with exception before calling abort_active_tasks
        with pytest.raises(ValueError, match="Task failed"):
            await task
        active_tasks = {"issue-1": task}
        state.agent_ids["issue-1"] = "agent-1"

        await handler.abort_active_tasks(
            active_tasks, "Test abort", state, mock_run_metadata
        )

        # finalize_issue_result signature: (issue_id, result, run_metadata)
        finalize_args = mock_callbacks.finalize_issue_result.call_args
        assert finalize_args is not None
        result = (
            finalize_args.kwargs["result"]
            if "result" in finalize_args.kwargs
            else finalize_args.args[1]
        )
        assert result.success is False
        assert "Task failed" in result.summary

    @pytest.mark.asyncio
    async def test_uses_default_reason(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
        state: OrchestratorState,
        mock_run_metadata: RunMetadata,
    ) -> None:
        """abort_active_tasks uses default reason when None provided."""

        async def running_task() -> IssueResult:
            await asyncio.sleep(10)
            return IssueResult(
                issue_id="issue-1", agent_id="agent-1", success=True, summary="done"
            )

        task = asyncio.create_task(running_task())
        active_tasks = {"issue-1": task}
        state.agent_ids["issue-1"] = "agent-1"

        await handler.abort_active_tasks(active_tasks, None, state, mock_run_metadata)

        mock_callbacks.on_tasks_aborting.assert_called_once_with(
            1, "Unrecoverable error"
        )

        # Ensure task reaches terminal state to avoid "Task was destroyed" warnings
        await asyncio.gather(task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_handles_empty_active_tasks(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
        state: OrchestratorState,
        mock_run_metadata: RunMetadata,
    ) -> None:
        """abort_active_tasks returns early for empty tasks dict."""
        active_tasks: dict[str, asyncio.Task[IssueResult]] = {}

        await handler.abort_active_tasks(
            active_tasks, "Test abort", state, mock_run_metadata
        )

        mock_callbacks.on_tasks_aborting.assert_not_called()
        mock_callbacks.finalize_issue_result.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_includes_session_log_path(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
        state: OrchestratorState,
        mock_run_metadata: RunMetadata,
    ) -> None:
        """abort_active_tasks includes session log path in result."""
        log_path = Path("/logs/session.log")
        state.active_session_log_paths["issue-1"] = log_path

        async def running_task() -> IssueResult:
            await asyncio.sleep(10)
            return IssueResult(
                issue_id="issue-1", agent_id="agent-1", success=True, summary="done"
            )

        task = asyncio.create_task(running_task())
        active_tasks = {"issue-1": task}
        state.agent_ids["issue-1"] = "agent-1"

        await handler.abort_active_tasks(
            active_tasks, "Test abort", state, mock_run_metadata
        )

        # finalize_issue_result signature: (issue_id, result, run_metadata)
        finalize_args = mock_callbacks.finalize_issue_result.call_args
        assert finalize_args is not None
        result = (
            finalize_args.kwargs["result"]
            if "result" in finalize_args.kwargs
            else finalize_args.args[1]
        )
        assert result.session_log_path == log_path

        # Ensure task reaches terminal state to avoid "Task was destroyed" warnings
        await asyncio.gather(task, return_exceptions=True)


class TestCleanupAgentLocks:
    """Tests for cleanup_agent_locks method."""

    def test_calls_do_cleanup_callback(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
    ) -> None:
        """cleanup_agent_locks calls do_cleanup_agent_locks callback."""
        handler.cleanup_agent_locks("agent-1")

        mock_callbacks.do_cleanup_agent_locks.assert_called_once_with("agent-1")

    def test_emits_on_locks_cleaned_when_locks_found(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
    ) -> None:
        """cleanup_agent_locks emits on_locks_cleaned when locks cleaned."""
        mock_callbacks.do_cleanup_agent_locks.return_value = (2, ["/a.py", "/b.py"])

        handler.cleanup_agent_locks("agent-1")

        mock_callbacks.on_locks_cleaned.assert_called_once_with("agent-1", 2)

    def test_no_event_when_no_locks_cleaned(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
    ) -> None:
        """cleanup_agent_locks does not emit event when no locks cleaned."""
        mock_callbacks.do_cleanup_agent_locks.return_value = (0, [])

        handler.cleanup_agent_locks("agent-1")

        mock_callbacks.on_locks_cleaned.assert_not_called()

    def test_unregisters_agent_from_monitor(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
    ) -> None:
        """cleanup_agent_locks unregisters agent from deadlock monitor."""
        handler.cleanup_agent_locks("agent-1")

        mock_callbacks.unregister_agent.assert_called_once_with("agent-1")

    def test_handles_none_unregister_callback(
        self,
        mock_callbacks: MockCallbacks,
    ) -> None:
        """cleanup_agent_locks handles None unregister_agent callback."""
        mock_callbacks.unregister_agent = None
        handler = DeadlockHandler(callbacks=mock_callbacks.as_callbacks())

        # Should not raise
        handler.cleanup_agent_locks("agent-1")

        mock_callbacks.do_cleanup_agent_locks.assert_called_once()

    def test_emits_event_only_when_locks_released(
        self,
        handler: DeadlockHandler,
        mock_callbacks: MockCallbacks,
    ) -> None:
        """cleanup_agent_locks only emits on_locks_cleaned when locks are released."""
        # First cleanup releases locks
        mock_callbacks.do_cleanup_agent_locks.return_value = (2, ["/a.py", "/b.py"])
        handler.cleanup_agent_locks("agent-1")
        mock_callbacks.on_locks_cleaned.assert_called_once_with("agent-1", 2)

        # Reset mocks for second call
        mock_callbacks.do_cleanup_agent_locks.reset_mock()
        mock_callbacks.on_locks_cleaned.reset_mock()

        # Second cleanup returns 0 (no locks to release - idempotency in lock server)
        mock_callbacks.do_cleanup_agent_locks.return_value = (0, [])
        handler.cleanup_agent_locks("agent-1")

        # Callback is always invoked (delegates to lock server)
        mock_callbacks.do_cleanup_agent_locks.assert_called_once_with("agent-1")
        # No event emitted since count is 0
        mock_callbacks.on_locks_cleaned.assert_not_called()


class TestResolutionLockSerialization:
    """Tests for resolution lock behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_deadlocks_serialized(
        self,
        mock_callbacks: MockCallbacks,
        state: OrchestratorState,
    ) -> None:
        """Concurrent handle_deadlock calls are serialized by lock."""
        # Track call order
        call_order: list[str] = []

        async def slow_add_dependency(dependent: str, dependency: str) -> bool:
            call_order.append(f"start_{dependent}")
            await asyncio.sleep(0.05)
            call_order.append(f"end_{dependent}")
            return True

        mock_callbacks.add_dependency = AsyncMock(side_effect=slow_add_dependency)
        # Create handler AFTER modifying callbacks
        handler = DeadlockHandler(callbacks=mock_callbacks.as_callbacks())

        info1 = DeadlockInfo(
            cycle=["a", "b"],
            victim_id="b",
            victim_issue_id="issue-1",
            blocked_on="/lock1",
            blocker_id="a",
            blocker_issue_id="issue-a",
        )
        info2 = DeadlockInfo(
            cycle=["c", "d"],
            victim_id="d",
            victim_issue_id="issue-2",
            blocked_on="/lock2",
            blocker_id="c",
            blocker_issue_id="issue-c",
        )

        active_tasks: dict[str, asyncio.Task[IssueResult]] = {}

        # Start both concurrently
        task1 = asyncio.create_task(handler.handle_deadlock(info1, state, active_tasks))
        task2 = asyncio.create_task(handler.handle_deadlock(info2, state, active_tasks))

        await asyncio.gather(task1, task2)

        # Due to serialization, one should complete before the other starts
        # Either [start_1, end_1, start_2, end_2] or [start_2, end_2, start_1, end_1]
        assert len(call_order) == 4
        # Check that both pairs don't interleave (each start followed by its end)
        assert call_order[:2] == [
            call_order[0],
            call_order[0].replace("start_", "end_"),
        ]
        assert call_order[2:] == [
            call_order[2],
            call_order[2].replace("start_", "end_"),
        ]
