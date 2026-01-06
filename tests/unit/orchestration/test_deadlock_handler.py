"""Unit tests for DeadlockHandler service."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from src.domain.deadlock import DeadlockInfo
from src.orchestration.deadlock_handler import DeadlockHandler, DeadlockHandlerCallbacks
from src.pipeline.issue_execution_coordinator import AbortResult
from src.orchestration.orchestrator_state import OrchestratorState
from src.pipeline.issue_result import IssueResult


@dataclass
class FakeCallbacks:
    """Fake callbacks with observable state for behavioral testing.

    Observable state:
    - add_dependency_calls: List of (dependent_id, dependency_id) tuples
    - mark_needs_followup_calls: List of (issue_id, reason, log_path) tuples
    - reopen_issue_calls: List of issue_ids
    - on_deadlock_detected_calls: List of DeadlockInfo objects
    - on_locks_cleaned_calls: List of (agent_id, count) tuples
    - on_tasks_aborting_calls: List of (count, reason) tuples
    - do_cleanup_agent_locks_calls: List of agent_ids
    - unregister_agent_calls: List of agent_ids
    - finalize_issue_result_calls: List of (issue_id, result, run_metadata) tuples
    - mark_completed_calls: List of issue_ids
    """

    # Call tracking
    add_dependency_calls: list[tuple[str, str]] = field(default_factory=list)
    mark_needs_followup_calls: list[tuple[str, str, Path | None]] = field(
        default_factory=list
    )
    reopen_issue_calls: list[str] = field(default_factory=list)
    on_deadlock_detected_calls: list[DeadlockInfo] = field(default_factory=list)
    on_locks_cleaned_calls: list[tuple[str, int]] = field(default_factory=list)
    on_tasks_aborting_calls: list[tuple[int, str]] = field(default_factory=list)
    do_cleanup_agent_locks_calls: list[str] = field(default_factory=list)
    unregister_agent_calls: list[str] = field(default_factory=list)
    finalize_issue_result_calls: list[tuple[str, IssueResult, object]] = field(
        default_factory=list
    )
    mark_completed_calls: list[str] = field(default_factory=list)

    # Configurable return values
    add_dependency_return: bool = True
    mark_needs_followup_return: bool = True
    reopen_issue_return: bool = True
    cleanup_locks_return: tuple[int, list[str]] = field(
        default_factory=lambda: (1, ["/path/to/lock"])
    )

    # Custom side effects for advanced tests
    add_dependency_side_effect: Any = None
    mark_needs_followup_side_effect: Any = None

    async def add_dependency(self, dependent_id: str, dependency_id: str) -> bool:
        """Track add_dependency calls."""
        self.add_dependency_calls.append((dependent_id, dependency_id))
        if self.add_dependency_side_effect:
            return await self.add_dependency_side_effect(dependent_id, dependency_id)
        return self.add_dependency_return

    async def mark_needs_followup(
        self, issue_id: str, reason: str, log_path: Path | None
    ) -> bool:
        """Track mark_needs_followup calls."""
        self.mark_needs_followup_calls.append((issue_id, reason, log_path))
        if self.mark_needs_followup_side_effect:
            return await self.mark_needs_followup_side_effect(
                issue_id, reason, log_path
            )
        return self.mark_needs_followup_return

    async def reopen_issue(self, issue_id: str) -> bool:
        """Track reopen_issue calls."""
        self.reopen_issue_calls.append(issue_id)
        return self.reopen_issue_return

    def on_deadlock_detected(self, info: DeadlockInfo) -> None:
        """Track on_deadlock_detected calls."""
        self.on_deadlock_detected_calls.append(info)

    def on_locks_cleaned(self, agent_id: str, count: int) -> None:
        """Track on_locks_cleaned calls."""
        self.on_locks_cleaned_calls.append((agent_id, count))

    def on_tasks_aborting(self, count: int, reason: str) -> None:
        """Track on_tasks_aborting calls."""
        self.on_tasks_aborting_calls.append((count, reason))

    def do_cleanup_agent_locks(self, agent_id: str) -> tuple[int, list[str]]:
        """Track do_cleanup_agent_locks calls and return configured value."""
        self.do_cleanup_agent_locks_calls.append(agent_id)
        return self.cleanup_locks_return

    def unregister_agent(self, agent_id: str) -> None:
        """Track unregister_agent calls."""
        self.unregister_agent_calls.append(agent_id)

    async def finalize_issue_result(
        self, issue_id: str, result: IssueResult, run_metadata: object
    ) -> None:
        """Track finalize_issue_result calls."""
        self.finalize_issue_result_calls.append((issue_id, result, run_metadata))

    def mark_completed(self, issue_id: str) -> None:
        """Track mark_completed calls."""
        self.mark_completed_calls.append(issue_id)

    def as_callbacks(self) -> DeadlockHandlerCallbacks:
        """Convert to DeadlockHandlerCallbacks."""
        return DeadlockHandlerCallbacks(
            add_dependency=self.add_dependency,
            mark_needs_followup=self.mark_needs_followup,
            reopen_issue=self.reopen_issue,
            on_deadlock_detected=self.on_deadlock_detected,
            on_locks_cleaned=self.on_locks_cleaned,
            on_tasks_aborting=self.on_tasks_aborting,
            do_cleanup_agent_locks=self.do_cleanup_agent_locks,
            unregister_agent=self.unregister_agent,
            finalize_issue_result=self.finalize_issue_result,
            mark_completed=self.mark_completed,
        )


@pytest.fixture
def fake_callbacks() -> FakeCallbacks:
    """Create fake callbacks for DeadlockHandler."""
    return FakeCallbacks()


@pytest.fixture
def handler(fake_callbacks: FakeCallbacks) -> DeadlockHandler:
    """Create DeadlockHandler with fake callbacks."""
    return DeadlockHandler(callbacks=fake_callbacks.as_callbacks())


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
def fake_run_metadata() -> object:
    """Create fake RunMetadata for tests."""
    # RunMetadata is a simple data class; we only need an object to pass through
    return object()


class TestHandleDeadlock:
    """Tests for handle_deadlock method."""

    @pytest.mark.asyncio
    async def test_calls_callbacks_with_correct_arguments(
        self,
        state: OrchestratorState,
        deadlock_info: DeadlockInfo,
    ) -> None:
        """handle_deadlock calls callbacks with correct arguments."""
        fake_cbs = FakeCallbacks()
        handler = DeadlockHandler(callbacks=fake_cbs.as_callbacks())
        active_tasks: dict[str, asyncio.Task[IssueResult]] = {}

        await handler.handle_deadlock(deadlock_info, state, active_tasks)

        # Verify add_dependency was called with correct arguments
        assert len(fake_cbs.add_dependency_calls) == 1
        dependent_id, dependency_id = fake_cbs.add_dependency_calls[0]
        assert dependent_id == deadlock_info.victim_issue_id
        assert dependency_id == deadlock_info.blocker_issue_id

        # Verify mark_needs_followup was called with victim issue_id
        assert len(fake_cbs.mark_needs_followup_calls) == 1
        issue_id, _reason, _log_path = fake_cbs.mark_needs_followup_calls[0]
        assert issue_id == deadlock_info.victim_issue_id

        # Verify on_deadlock_detected was called
        assert len(fake_cbs.on_deadlock_detected_calls) == 1

        # Verify cleanup happened
        assert fake_cbs.do_cleanup_agent_locks_calls == ["agent-b"]
        assert fake_cbs.unregister_agent_calls == ["agent-b"]

    @pytest.mark.asyncio
    async def test_cleans_up_locks_and_tracks_in_state(
        self,
        handler: DeadlockHandler,
        fake_callbacks: FakeCallbacks,
        state: OrchestratorState,
        deadlock_info: DeadlockInfo,
    ) -> None:
        """handle_deadlock cleans up locks and tracks in state."""
        active_tasks: dict[str, asyncio.Task[IssueResult]] = {}

        await handler.handle_deadlock(deadlock_info, state, active_tasks)

        # Verify agent tracked as cleaned (state change)
        assert "agent-b" in state.deadlock_cleaned_agents
        # Verify issue tracked as deadlock victim
        assert "issue-b" in state.deadlock_victim_issues
        # Verify cleanup callbacks were invoked (via observable state)
        assert fake_callbacks.do_cleanup_agent_locks_calls == ["agent-b"]
        assert fake_callbacks.unregister_agent_calls == ["agent-b"]

    @pytest.mark.asyncio
    async def test_tracks_deadlock_victim_issue_in_state(
        self,
        handler: DeadlockHandler,
        state: OrchestratorState,
        deadlock_info: DeadlockInfo,
    ) -> None:
        """handle_deadlock tracks victim issue_id in state.deadlock_victim_issues."""
        active_tasks: dict[str, asyncio.Task[IssueResult]] = {}

        await handler.handle_deadlock(deadlock_info, state, active_tasks)

        # Victim issue should be tracked for special error messaging
        assert deadlock_info.victim_issue_id in state.deadlock_victim_issues
        # Agent should also be tracked in cleaned set
        assert deadlock_info.victim_id in state.deadlock_cleaned_agents

    @pytest.mark.asyncio
    async def test_cancels_victim_task_non_self(
        self,
        handler: DeadlockHandler,
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
        # Behavioral assertion: task was cancelled
        assert victim_task.cancelled()

    @pytest.mark.asyncio
    async def test_defers_self_cancellation(
        self,
        state: OrchestratorState,
    ) -> None:
        """handle_deadlock defers self-cancellation correctly."""
        fake_cbs = FakeCallbacks()
        handler = DeadlockHandler(callbacks=fake_cbs.as_callbacks())

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

        fake_cbs = FakeCallbacks(
            add_dependency_side_effect=slow_add_dependency,
            mark_needs_followup_side_effect=mark_followup_and_signal,
        )
        handler = DeadlockHandler(callbacks=fake_cbs.as_callbacks())

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
        assert len(fake_cbs.add_dependency_calls) == 1

    @pytest.mark.asyncio
    async def test_uses_log_path_from_state(
        self,
        handler: DeadlockHandler,
        fake_callbacks: FakeCallbacks,
        state: OrchestratorState,
        deadlock_info: DeadlockInfo,
    ) -> None:
        """handle_deadlock uses session log path from state."""
        log_path = Path("/logs/session.log")
        state.active_session_log_paths["issue-b"] = log_path
        active_tasks: dict[str, asyncio.Task[IssueResult]] = {}

        await handler.handle_deadlock(deadlock_info, state, active_tasks)

        # Verify log path passed to mark_needs_followup
        assert len(fake_callbacks.mark_needs_followup_calls) == 1
        _issue_id, _reason, actual_log_path = fake_callbacks.mark_needs_followup_calls[
            0
        ]
        assert actual_log_path == log_path

    @pytest.mark.asyncio
    async def test_handles_none_issue_ids(
        self,
        handler: DeadlockHandler,
        fake_callbacks: FakeCallbacks,
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

        # With None issue IDs, resolution callbacks should not be called
        assert len(fake_callbacks.add_dependency_calls) == 0
        assert len(fake_callbacks.mark_needs_followup_calls) == 0
        assert len(fake_callbacks.reopen_issue_calls) == 0


class TestAbortActiveTasks:
    """Tests for abort_active_tasks method."""

    @pytest.mark.asyncio
    async def test_cancels_running_tasks(
        self,
        handler: DeadlockHandler,
        fake_callbacks: FakeCallbacks,
        state: OrchestratorState,
        fake_run_metadata: object,
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

        result = await handler.abort_active_tasks(
            active_tasks,
            "Test abort",
            state,
            fake_run_metadata,  # type: ignore[arg-type]
        )

        # Await task with timeout to avoid hanging if cancellation fails
        await asyncio.wait_for(
            asyncio.gather(task, return_exceptions=True), timeout=2.0
        )
        # Behavioral assertions on task state and callback tracking
        assert task.cancelled() or task.done()
        assert fake_callbacks.on_tasks_aborting_calls == [(1, "Test abort")]
        assert len(fake_callbacks.finalize_issue_result_calls) == 1
        assert fake_callbacks.mark_completed_calls == ["issue-1"]
        # Return value assertions
        assert isinstance(result, AbortResult)
        assert result.aborted_count == 1
        assert result.has_unresponsive_tasks is False

    @pytest.mark.asyncio
    async def test_uses_real_results_for_completed_tasks(
        self,
        handler: DeadlockHandler,
        fake_callbacks: FakeCallbacks,
        state: OrchestratorState,
        fake_run_metadata: object,
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
            active_tasks,
            "Test abort",
            state,
            fake_run_metadata,  # type: ignore[arg-type]
        )

        # Should use the real result, not an aborted result
        assert len(fake_callbacks.finalize_issue_result_calls) == 1
        _issue_id, result, _run_metadata = fake_callbacks.finalize_issue_result_calls[0]
        assert result.success is True
        assert result.summary == "Completed successfully"

    @pytest.mark.asyncio
    async def test_handles_task_exception(
        self,
        handler: DeadlockHandler,
        fake_callbacks: FakeCallbacks,
        state: OrchestratorState,
        fake_run_metadata: object,
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
            active_tasks,
            "Test abort",
            state,
            fake_run_metadata,  # type: ignore[arg-type]
        )

        # Result should indicate failure with exception message
        assert len(fake_callbacks.finalize_issue_result_calls) == 1
        _issue_id, result, _run_metadata = fake_callbacks.finalize_issue_result_calls[0]
        assert result.success is False
        assert "Task failed" in result.summary

    @pytest.mark.asyncio
    async def test_uses_default_reason(
        self,
        handler: DeadlockHandler,
        fake_callbacks: FakeCallbacks,
        state: OrchestratorState,
        fake_run_metadata: object,
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

        await handler.abort_active_tasks(
            active_tasks,
            None,
            state,
            fake_run_metadata,  # type: ignore[arg-type]
        )

        # Default reason should be "Unrecoverable error"
        assert fake_callbacks.on_tasks_aborting_calls == [(1, "Unrecoverable error")]

        # Ensure task reaches terminal state to avoid "Task was destroyed" warnings
        await asyncio.gather(task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_handles_empty_active_tasks(
        self,
        handler: DeadlockHandler,
        fake_callbacks: FakeCallbacks,
        state: OrchestratorState,
        fake_run_metadata: object,
    ) -> None:
        """abort_active_tasks returns early for empty tasks dict."""
        active_tasks: dict[str, asyncio.Task[IssueResult]] = {}

        result = await handler.abort_active_tasks(
            active_tasks,
            "Test abort",
            state,
            fake_run_metadata,  # type: ignore[arg-type]
        )

        # No callbacks should be invoked for empty active_tasks
        assert len(fake_callbacks.on_tasks_aborting_calls) == 0
        assert len(fake_callbacks.finalize_issue_result_calls) == 0
        # Empty result
        assert result == AbortResult(aborted_count=0, has_unresponsive_tasks=False)

    @pytest.mark.asyncio
    async def test_includes_session_log_path(
        self,
        handler: DeadlockHandler,
        fake_callbacks: FakeCallbacks,
        state: OrchestratorState,
        fake_run_metadata: object,
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
            active_tasks,
            "Test abort",
            state,
            fake_run_metadata,  # type: ignore[arg-type]
        )

        # Result should include the session log path
        assert len(fake_callbacks.finalize_issue_result_calls) == 1
        _issue_id, result, _run_metadata = fake_callbacks.finalize_issue_result_calls[0]
        assert result.session_log_path == log_path

        # Ensure task reaches terminal state to avoid "Task was destroyed" warnings
        await asyncio.gather(task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_unresponsive_tasks_flagged(
        self,
        handler: DeadlockHandler,
        fake_callbacks: FakeCallbacks,
        state: OrchestratorState,
        fake_run_metadata: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """abort_active_tasks sets has_unresponsive_tasks when tasks don't respond to cancel."""
        import src.orchestration.deadlock_handler as dh_module

        # Use very short grace period for test
        monkeypatch.setattr(dh_module, "ABORT_GRACE_SECONDS", 0.05)

        task_started = asyncio.Event()
        allow_finish = asyncio.Event()

        async def unresponsive_task() -> IssueResult:
            task_started.set()
            # Use a loop that catches and ignores CancelledError
            while not allow_finish.is_set():
                try:
                    await asyncio.sleep(10)  # Long sleep, will be cancelled
                except asyncio.CancelledError:
                    # Ignore cancellation and continue - simulates unresponsive task
                    pass
            return IssueResult(
                issue_id="issue-1", agent_id="agent-1", success=True, summary="done"
            )

        task = asyncio.create_task(unresponsive_task())
        await task_started.wait()
        active_tasks = {"issue-1": task}
        state.agent_ids["issue-1"] = "agent-1"

        result = await handler.abort_active_tasks(
            active_tasks,
            "Test abort",
            state,
            fake_run_metadata,  # type: ignore[arg-type]
        )

        # Task should still be running (unresponsive)
        assert not task.done()
        assert result.has_unresponsive_tasks is True
        assert result.aborted_count == 1

        # Verify task was finalized with unresponsive summary
        assert len(fake_callbacks.finalize_issue_result_calls) == 1
        _issue_id, finalized_result, _ = fake_callbacks.finalize_issue_result_calls[0]
        assert "unresponsive" in finalized_result.summary

        # Clean up - allow task to finish and cancel again (it will finish this time)
        allow_finish.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class TestCleanupAgentLocks:
    """Tests for cleanup_agent_locks method."""

    def test_calls_do_cleanup_callback(
        self,
        handler: DeadlockHandler,
        fake_callbacks: FakeCallbacks,
    ) -> None:
        """cleanup_agent_locks calls do_cleanup_agent_locks callback."""
        handler.cleanup_agent_locks("agent-1")

        assert fake_callbacks.do_cleanup_agent_locks_calls == ["agent-1"]

    def test_emits_on_locks_cleaned_when_locks_found(self) -> None:
        """cleanup_agent_locks emits on_locks_cleaned when locks cleaned."""
        fake_cbs = FakeCallbacks(cleanup_locks_return=(2, ["/a.py", "/b.py"]))
        handler = DeadlockHandler(callbacks=fake_cbs.as_callbacks())

        handler.cleanup_agent_locks("agent-1")

        assert fake_cbs.on_locks_cleaned_calls == [("agent-1", 2)]

    def test_no_event_when_no_locks_cleaned(self) -> None:
        """cleanup_agent_locks does not emit event when no locks cleaned."""
        fake_cbs = FakeCallbacks(cleanup_locks_return=(0, []))
        handler = DeadlockHandler(callbacks=fake_cbs.as_callbacks())

        handler.cleanup_agent_locks("agent-1")

        # No event emitted since count is 0
        assert len(fake_cbs.on_locks_cleaned_calls) == 0

    def test_unregisters_agent_from_monitor(
        self,
        handler: DeadlockHandler,
        fake_callbacks: FakeCallbacks,
    ) -> None:
        """cleanup_agent_locks unregisters agent from deadlock monitor."""
        handler.cleanup_agent_locks("agent-1")

        assert fake_callbacks.unregister_agent_calls == ["agent-1"]

    def test_emits_event_only_when_locks_released(self) -> None:
        """cleanup_agent_locks only emits on_locks_cleaned when locks are released."""
        # First cleanup releases locks
        fake_cbs = FakeCallbacks(cleanup_locks_return=(2, ["/a.py", "/b.py"]))
        handler = DeadlockHandler(callbacks=fake_cbs.as_callbacks())
        handler.cleanup_agent_locks("agent-1")
        assert fake_cbs.on_locks_cleaned_calls == [("agent-1", 2)]

        # Second cleanup returns 0 (no locks to release - idempotency in lock server)
        # We need a new handler since FakeCallbacks doesn't support changing return value
        fake_cbs2 = FakeCallbacks(cleanup_locks_return=(0, []))
        handler2 = DeadlockHandler(callbacks=fake_cbs2.as_callbacks())
        handler2.cleanup_agent_locks("agent-1")

        # Callback is always invoked (delegates to lock server)
        assert fake_cbs2.do_cleanup_agent_locks_calls == ["agent-1"]
        # No event emitted since count is 0
        assert len(fake_cbs2.on_locks_cleaned_calls) == 0


class TestResolutionLockSerialization:
    """Tests for resolution lock behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_deadlocks_serialized(
        self,
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

        fake_cbs = FakeCallbacks(add_dependency_side_effect=slow_add_dependency)
        handler = DeadlockHandler(callbacks=fake_cbs.as_callbacks())

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
