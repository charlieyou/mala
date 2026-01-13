"""Integration tests for deadlock detection and resolution.

Tests the complete flow:
- Lock events -> cycle detection -> victim selection -> resolution -> cleanup.

Uses mock BeadsClient and simulates agent sessions to verify the deadlock
handling end-to-end without needing actual Claude SDK sessions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from src.core.models import LockEvent, LockEventType
from src.domain.deadlock import DeadlockMonitor
from src.infra.io.base_sink import NullEventSink

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols.lifecycle import DeadlockInfoProtocol
    from src.domain.deadlock import DeadlockInfo

pytestmark = pytest.mark.integration


@dataclass
class MockBeadsClient:
    """Mock BeadsClient for testing deadlock resolution callbacks."""

    dependencies_added: list[tuple[str, str]] = field(default_factory=list)
    needs_followup_marked: list[tuple[str, str, Path | None]] = field(
        default_factory=list
    )

    async def add_dependency_async(self, issue_id: str, depends_on_id: str) -> bool:
        """Record dependency addition."""
        self.dependencies_added.append((issue_id, depends_on_id))
        return True

    async def mark_needs_followup_async(
        self, issue_id: str, reason: str, log_path: Path | None = None
    ) -> bool:
        """Record needs-followup marking."""
        self.needs_followup_marked.append((issue_id, reason, log_path))
        return True


@dataclass
class MockAgentSession:
    """Mock agent session for testing deadlock scenarios."""

    agent_id: str
    issue_id: str
    start_time: float
    task: asyncio.Task[None] | None = None
    cancelled: bool = False

    async def run(self) -> None:
        """Simulated agent work that can be cancelled."""
        try:
            # Simulate ongoing work
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class DeadlockEventSink(NullEventSink):
    """Event sink that records deadlock events for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.deadlock_events: list[DeadlockInfoProtocol] = []

    def on_deadlock_detected(self, info: DeadlockInfoProtocol) -> None:
        """Record deadlock detection event."""
        self.deadlock_events.append(info)


class TestTwoAgentDeadlock:
    """Test 2-agent deadlock detection and resolution."""

    @pytest.mark.asyncio
    async def test_two_agent_deadlock_detected_and_resolved(self) -> None:
        """A holds L1, B holds L2; A waits L2, B waits L1 -> deadlock detected.

        Verifies:
        - Deadlock is detected when cycle completes
        - Youngest agent (B) selected as victim
        - Victim issue_id and blocker issue_id are correct
        """
        monitor = DeadlockMonitor()
        event_sink = DeadlockEventSink()
        beads = MockBeadsClient()

        # Register agents (agent-b is younger)
        monitor.register_agent("agent-a", "issue-a", 1000.0)
        monitor.register_agent("agent-b", "issue-b", 2000.0)

        # Track deadlock info
        detected_deadlock: DeadlockInfo | None = None
        cleaned_agents: list[str] = []

        async def on_deadlock(info: DeadlockInfo) -> None:
            nonlocal detected_deadlock
            detected_deadlock = info
            event_sink.on_deadlock_detected(info)
            # Simulate cleanup_agent_locks
            cleaned_agents.append(info.victim_id)
            # Simulate add_dependency and mark_needs_followup
            if info.victim_issue_id and info.blocker_issue_id:
                await beads.add_dependency_async(
                    info.victim_issue_id, info.blocker_issue_id
                )
            if info.victim_issue_id:
                await beads.mark_needs_followup_async(
                    info.victim_issue_id,
                    f"Deadlock victim: blocked on {info.blocked_on}",
                )

        monitor.on_deadlock = on_deadlock

        # A holds L1, B holds L2
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-a", "/path/l1.py", 1001.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-b", "/path/l2.py", 2001.0)
        )

        # A waits for L2 (held by B) - no deadlock yet
        result = await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-a", "/path/l2.py", 1002.0)
        )
        assert result is None
        assert detected_deadlock is None

        # B waits for L1 (held by A) - deadlock!
        result = await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-b", "/path/l1.py", 2002.0)
        )

        # Verify deadlock detected
        assert result is not None
        assert detected_deadlock is not None
        assert set(detected_deadlock.cycle) == {"agent-a", "agent-b"}

        # Verify youngest agent (B) is victim
        assert detected_deadlock.victim_id == "agent-b"
        assert detected_deadlock.victim_issue_id == "issue-b"

        # Verify blocker info
        assert detected_deadlock.blocked_on == "/path/l1.py"
        assert detected_deadlock.blocker_id == "agent-a"
        assert detected_deadlock.blocker_issue_id == "issue-a"

        # Verify cleanup_agent_locks was called for victim
        assert "agent-b" in cleaned_agents

        # Verify add_dependency was called correctly
        assert len(beads.dependencies_added) == 1
        assert beads.dependencies_added[0] == ("issue-b", "issue-a")

        # Verify mark_needs_followup was called
        assert len(beads.needs_followup_marked) == 1
        assert beads.needs_followup_marked[0][0] == "issue-b"
        assert "blocked on /path/l1.py" in beads.needs_followup_marked[0][1]

        # Verify event was emitted
        assert len(event_sink.deadlock_events) == 1
        assert event_sink.deadlock_events[0].victim_id == "agent-b"

    @pytest.mark.asyncio
    async def test_victim_task_cancelled(self) -> None:
        """Verify victim's task is properly cancelled during deadlock resolution."""
        monitor = DeadlockMonitor()

        # Create mock sessions
        session_a = MockAgentSession("agent-a", "issue-a", 1000.0)
        session_b = MockAgentSession("agent-b", "issue-b", 2000.0)

        # Register agents
        monitor.register_agent(
            session_a.agent_id, session_a.issue_id, session_a.start_time
        )
        monitor.register_agent(
            session_b.agent_id, session_b.issue_id, session_b.start_time
        )

        # Track active tasks
        active_tasks: dict[str, asyncio.Task[None]] = {}

        async def on_deadlock(info: DeadlockInfo) -> None:
            # Cancel victim task
            if info.victim_issue_id and info.victim_issue_id in active_tasks:
                task = active_tasks[info.victim_issue_id]
                task.cancel()

        monitor.on_deadlock = on_deadlock

        # Start mock agent tasks
        session_b.task = asyncio.create_task(session_b.run())
        active_tasks["issue-b"] = session_b.task

        # Give the task a chance to start running
        await asyncio.sleep(0)

        # Setup deadlock
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-a", "/path/l1.py", 1001.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-b", "/path/l2.py", 2001.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-a", "/path/l2.py", 1002.0)
        )

        # Trigger deadlock
        await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-b", "/path/l1.py", 2002.0)
        )

        # Wait for cancellation to propagate
        try:
            await asyncio.wait_for(session_b.task, timeout=1.0)
        except asyncio.CancelledError:
            pass

        # Verify victim was cancelled (task.cancelled() or our flag)
        assert session_b.task.cancelled() or session_b.cancelled


class TestThreeAgentDeadlock:
    """Test 3-agent cycle detection (A->B->C->A)."""

    @pytest.mark.asyncio
    async def test_three_agent_cycle_detected(self) -> None:
        """A->B->C->A cycle: A waits L2 (B), B waits L3 (C), C waits L1 (A).

        Verifies:
        - 3-agent cycle is detected
        - Youngest agent (C) is selected as victim
        """
        monitor = DeadlockMonitor()
        event_sink = DeadlockEventSink()

        # Register agents with ascending start times (C is youngest)
        monitor.register_agent("agent-a", "issue-a", 1000.0)
        monitor.register_agent("agent-b", "issue-b", 2000.0)
        monitor.register_agent("agent-c", "issue-c", 3000.0)

        detected_deadlock: DeadlockInfo | None = None

        async def on_deadlock(info: DeadlockInfo) -> None:
            nonlocal detected_deadlock
            detected_deadlock = info
            event_sink.on_deadlock_detected(info)

        monitor.on_deadlock = on_deadlock

        # Each agent holds one lock
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-a", "/path/l1.py", 1001.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-b", "/path/l2.py", 2001.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-c", "/path/l3.py", 3001.0)
        )

        # A waits for L2 (held by B)
        result = await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-a", "/path/l2.py", 1002.0)
        )
        assert result is None

        # B waits for L3 (held by C)
        result = await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-b", "/path/l3.py", 2002.0)
        )
        assert result is None

        # C waits for L1 (held by A) - cycle completes!
        result = await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-c", "/path/l1.py", 3002.0)
        )

        # Verify cycle detected
        assert result is not None
        assert detected_deadlock is not None
        assert set(detected_deadlock.cycle) == {"agent-a", "agent-b", "agent-c"}
        assert len(detected_deadlock.cycle) == 3

        # Verify youngest (C) is victim
        assert detected_deadlock.victim_id == "agent-c"
        assert detected_deadlock.victim_issue_id == "issue-c"

        # Verify blocker (A holds L1 that C needs)
        assert detected_deadlock.blocked_on == "/path/l1.py"
        assert detected_deadlock.blocker_id == "agent-a"

        # Verify event emitted
        assert len(event_sink.deadlock_events) == 1


class TestSingleAgentNoDeadlock:
    """Test that single agent never triggers false positive."""

    @pytest.mark.asyncio
    async def test_single_agent_waiting_no_deadlock(self) -> None:
        """Single agent waiting for an unheld lock is not a deadlock."""
        monitor = DeadlockMonitor()
        monitor.register_agent("agent-a", "issue-a", 1000.0)

        callback_invoked = False

        async def on_deadlock(info: DeadlockInfo) -> None:
            nonlocal callback_invoked
            callback_invoked = True

        monitor.on_deadlock = on_deadlock

        # Agent acquires one lock
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-a", "/path/l1.py", 1001.0)
        )

        # Agent waits for another lock (no holder)
        result = await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-a", "/path/l2.py", 1002.0)
        )

        assert result is None
        assert not callback_invoked


class TestUnregisterPreventsDeadlock:
    """Test that unregistering an agent prevents deadlock detection."""

    @pytest.mark.asyncio
    async def test_unregister_clears_graph_state(self) -> None:
        """Unregistering an agent clears its state from the graph."""
        monitor = DeadlockMonitor()
        monitor.register_agent("agent-a", "issue-a", 1000.0)
        monitor.register_agent("agent-b", "issue-b", 2000.0)

        callback_invoked = False

        async def on_deadlock(info: DeadlockInfo) -> None:
            nonlocal callback_invoked
            callback_invoked = True

        monitor.on_deadlock = on_deadlock

        # Setup potential deadlock
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-a", "/path/l1.py", 1001.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-b", "/path/l2.py", 2001.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-a", "/path/l2.py", 1002.0)
        )

        # Unregister A before B waits
        monitor.unregister_agent("agent-a")

        # B waits for L1 - but A is gone, no deadlock
        result = await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-b", "/path/l1.py", 2002.0)
        )

        assert result is None
        assert not callback_invoked


class TestNoneIssueIdHandling:
    """Test deadlock handling with agents that have no issue_id."""

    @pytest.mark.asyncio
    async def test_agents_with_none_issue_id(self) -> None:
        """Agents with None issue_id are handled correctly in deadlock."""
        monitor = DeadlockMonitor()
        beads = MockBeadsClient()

        # Register agents without issue IDs
        monitor.register_agent("agent-a", None, 1000.0)
        monitor.register_agent("agent-b", None, 2000.0)

        detected_deadlock: DeadlockInfo | None = None

        async def on_deadlock(info: DeadlockInfo) -> None:
            nonlocal detected_deadlock
            detected_deadlock = info
            # Should not call add_dependency when issue_ids are None
            if info.victim_issue_id and info.blocker_issue_id:
                await beads.add_dependency_async(
                    info.victim_issue_id, info.blocker_issue_id
                )

        monitor.on_deadlock = on_deadlock

        # Create deadlock
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-a", "/path/l1.py", 1001.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-b", "/path/l2.py", 2001.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-a", "/path/l2.py", 1002.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-b", "/path/l1.py", 2002.0)
        )

        # Verify deadlock detected
        assert detected_deadlock is not None
        assert detected_deadlock.victim_issue_id is None
        assert detected_deadlock.blocker_issue_id is None

        # Verify add_dependency was NOT called (no issue IDs)
        assert len(beads.dependencies_added) == 0


class TestReleaseBreaksPotentialDeadlock:
    """Test that releasing a lock prevents deadlock formation."""

    @pytest.mark.asyncio
    async def test_release_before_wait_no_deadlock(self) -> None:
        """Releasing a lock before waiting breaks potential cycle."""
        monitor = DeadlockMonitor()
        monitor.register_agent("agent-a", "issue-a", 1000.0)
        monitor.register_agent("agent-b", "issue-b", 2000.0)

        callback_invoked = False

        async def on_deadlock(info: DeadlockInfo) -> None:
            nonlocal callback_invoked
            callback_invoked = True

        monitor.on_deadlock = on_deadlock

        # A holds L1, B holds L2
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-a", "/path/l1.py", 1001.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-b", "/path/l2.py", 2001.0)
        )

        # A waits for L2
        await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-a", "/path/l2.py", 1002.0)
        )

        # B releases L2 before waiting for L1
        await monitor.handle_event(
            LockEvent(LockEventType.RELEASED, "agent-b", "/path/l2.py", 2002.0)
        )

        # B waits for L1 - no deadlock since B no longer holds L2
        result = await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-b", "/path/l1.py", 2003.0)
        )

        assert result is None
        assert not callback_invoked


class TestAcquireClearsWait:
    """Test that acquiring a lock clears the waiting state."""

    @pytest.mark.asyncio
    async def test_acquire_after_wait_clears_wait(self) -> None:
        """Acquiring a lock clears the agent's wait state."""
        monitor = DeadlockMonitor()
        monitor.register_agent("agent-a", "issue-a", 1000.0)
        monitor.register_agent("agent-b", "issue-b", 2000.0)

        callback_invoked = False

        async def on_deadlock(info: DeadlockInfo) -> None:
            nonlocal callback_invoked
            callback_invoked = True

        monitor.on_deadlock = on_deadlock

        # A holds L1
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-a", "/path/l1.py", 1001.0)
        )

        # B waits for L1
        await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-b", "/path/l1.py", 2001.0)
        )

        # A releases L1
        await monitor.handle_event(
            LockEvent(LockEventType.RELEASED, "agent-a", "/path/l1.py", 1002.0)
        )

        # B acquires L1 (clears wait)
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-b", "/path/l1.py", 2002.0)
        )

        # A waits for L1 (now held by B) - but B is no longer waiting, so no cycle
        result = await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-a", "/path/l1.py", 1003.0)
        )

        assert result is None
        assert not callback_invoked
