"""Unit tests for deadlock detection domain model."""

from src.core.models import LockEvent, LockEventType
from src.domain.deadlock import (
    DeadlockInfo,
    DeadlockMonitor,
    WaitForGraph,
)


class TestWaitForGraph:
    """Tests for WaitForGraph cycle detection."""

    def test_no_cycle_when_empty(self) -> None:
        """Empty graph has no cycles."""
        graph = WaitForGraph()
        assert graph.detect_cycle() is None

    def test_no_cycle_single_hold(self) -> None:
        """Single agent holding lock has no cycle."""
        graph = WaitForGraph()
        graph.add_hold("agent-1", "/path/file.py")
        assert graph.detect_cycle() is None

    def test_no_cycle_single_wait(self) -> None:
        """Single agent waiting (no holder) has no cycle."""
        graph = WaitForGraph()
        graph.add_wait("agent-1", "/path/file.py")
        assert graph.detect_cycle() is None

    def test_no_cycle_wait_for_unheld_lock(self) -> None:
        """Waiting for a lock no one holds is not a cycle."""
        graph = WaitForGraph()
        graph.add_hold("agent-1", "/path/a.py")
        graph.add_wait("agent-2", "/path/b.py")  # Different lock
        assert graph.detect_cycle() is None

    def test_two_party_deadlock(self) -> None:
        """A<->B deadlock: A holds L1, waits L2; B holds L2, waits L1."""
        graph = WaitForGraph()
        # A holds L1, B holds L2
        graph.add_hold("agent-a", "/path/l1.py")
        graph.add_hold("agent-b", "/path/l2.py")
        # A waits for L2, B waits for L1
        graph.add_wait("agent-a", "/path/l2.py")
        graph.add_wait("agent-b", "/path/l1.py")

        cycle = graph.detect_cycle()
        assert cycle is not None
        assert set(cycle) == {"agent-a", "agent-b"}
        assert len(cycle) == 2

    def test_three_party_deadlock(self) -> None:
        """A->B->C->A deadlock cycle."""
        graph = WaitForGraph()
        # A holds L1, B holds L2, C holds L3
        graph.add_hold("agent-a", "/path/l1.py")
        graph.add_hold("agent-b", "/path/l2.py")
        graph.add_hold("agent-c", "/path/l3.py")
        # A waits L2 (held by B), B waits L3 (held by C), C waits L1 (held by A)
        graph.add_wait("agent-a", "/path/l2.py")
        graph.add_wait("agent-b", "/path/l3.py")
        graph.add_wait("agent-c", "/path/l1.py")

        cycle = graph.detect_cycle()
        assert cycle is not None
        assert set(cycle) == {"agent-a", "agent-b", "agent-c"}
        assert len(cycle) == 3

    def test_acquire_clears_wait(self) -> None:
        """Acquiring a lock clears the wait state."""
        graph = WaitForGraph()
        graph.add_wait("agent-1", "/path/file.py")
        graph.add_hold("agent-1", "/path/file.py")
        # Agent is no longer waiting
        assert graph.detect_cycle() is None

    def test_remove_agent_clears_all_state(self) -> None:
        """remove_agent clears holds and waits for that agent."""
        graph = WaitForGraph()
        graph.add_hold("agent-1", "/path/a.py")
        graph.add_hold("agent-1", "/path/b.py")
        graph.add_wait("agent-1", "/path/c.py")

        graph.remove_agent("agent-1")

        assert graph.get_holder("/path/a.py") is None
        assert graph.get_holder("/path/b.py") is None
        assert graph.detect_cycle() is None

    def test_remove_hold(self) -> None:
        """remove_hold only removes the specific lock."""
        graph = WaitForGraph()
        graph.add_hold("agent-1", "/path/a.py")
        graph.add_hold("agent-1", "/path/b.py")

        graph.remove_hold("agent-1", "/path/a.py")

        assert graph.get_holder("/path/a.py") is None
        assert graph.get_holder("/path/b.py") == "agent-1"

    def test_get_holder(self) -> None:
        """get_holder returns correct holder or None."""
        graph = WaitForGraph()
        assert graph.get_holder("/path/file.py") is None

        graph.add_hold("agent-1", "/path/file.py")
        assert graph.get_holder("/path/file.py") == "agent-1"


class TestDeadlockMonitor:
    """Tests for DeadlockMonitor event handling and victim selection."""

    def test_register_and_unregister_agent(self) -> None:
        """Agents can be registered and unregistered."""
        monitor = DeadlockMonitor()
        monitor.register_agent("agent-1", "issue-123", 1000.0)
        monitor.unregister_agent("agent-1")
        # Should not raise

    async def test_handle_acquired_event(self) -> None:
        """ACQUIRED event updates graph but doesn't detect deadlock."""
        monitor = DeadlockMonitor()
        monitor.register_agent("agent-1", "issue-123", 1000.0)

        event = LockEvent(
            event_type=LockEventType.ACQUIRED,
            agent_id="agent-1",
            lock_path="/path/file.py",
            timestamp=1001.0,
        )
        result = await monitor.handle_event(event)

        assert result is None

    async def test_handle_released_event(self) -> None:
        """RELEASED event clears hold."""
        monitor = DeadlockMonitor()
        monitor.register_agent("agent-1", "issue-123", 1000.0)

        # Acquire then release
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-1", "/path/file.py", 1001.0)
        )
        result = await monitor.handle_event(
            LockEvent(LockEventType.RELEASED, "agent-1", "/path/file.py", 1002.0)
        )

        assert result is None

    async def test_no_deadlock_single_wait(self) -> None:
        """Single agent waiting doesn't trigger deadlock."""
        monitor = DeadlockMonitor()
        monitor.register_agent("agent-1", "issue-123", 1000.0)

        event = LockEvent(
            event_type=LockEventType.WAITING,
            agent_id="agent-1",
            lock_path="/path/file.py",
            timestamp=1001.0,
        )
        result = await monitor.handle_event(event)

        assert result is None

    async def test_two_party_deadlock_detected(self) -> None:
        """Two-party deadlock is detected with correct victim."""
        monitor = DeadlockMonitor()
        # agent-a started at 1000, agent-b at 2000 (younger)
        monitor.register_agent("agent-a", "issue-a", 1000.0)
        monitor.register_agent("agent-b", "issue-b", 2000.0)

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

        # B waits for L1 (held by A) - deadlock!
        result = await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-b", "/path/l1.py", 2002.0)
        )

        assert result is not None
        assert isinstance(result, DeadlockInfo)
        assert set(result.cycle) == {"agent-a", "agent-b"}
        # Victim is agent-b (younger, start_time=2000)
        assert result.victim_id == "agent-b"
        assert result.victim_issue_id == "issue-b"
        assert result.blocked_on == "/path/l1.py"
        assert result.blocker_id == "agent-a"
        assert result.blocker_issue_id == "issue-a"

    async def test_three_party_deadlock_victim_is_youngest(self) -> None:
        """Three-party deadlock selects youngest agent as victim."""
        monitor = DeadlockMonitor()
        # Register with different start times
        monitor.register_agent("agent-a", "issue-a", 1000.0)
        monitor.register_agent("agent-b", "issue-b", 2000.0)
        monitor.register_agent("agent-c", "issue-c", 3000.0)  # Youngest

        # A holds L1, B holds L2, C holds L3
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-a", "/path/l1.py", 1001.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-b", "/path/l2.py", 2001.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-c", "/path/l3.py", 3001.0)
        )

        # A waits L2, B waits L3
        await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-a", "/path/l2.py", 1002.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-b", "/path/l3.py", 2002.0)
        )

        # C waits L1 - completes the cycle
        result = await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-c", "/path/l1.py", 3002.0)
        )

        assert result is not None
        assert set(result.cycle) == {"agent-a", "agent-b", "agent-c"}
        # Victim is agent-c (youngest, start_time=3000)
        assert result.victim_id == "agent-c"

    async def test_unregister_clears_graph_state(self) -> None:
        """Unregistering an agent prevents deadlock detection involving it."""
        monitor = DeadlockMonitor()
        monitor.register_agent("agent-a", "issue-a", 1000.0)
        monitor.register_agent("agent-b", "issue-b", 2000.0)

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

        # Unregister A before B tries to wait
        monitor.unregister_agent("agent-a")

        # B waits for L1 - but A is gone, no deadlock
        result = await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-b", "/path/l1.py", 2002.0)
        )
        assert result is None

    async def test_handle_event_with_none_issue_id(self) -> None:
        """Agents with None issue_id are handled correctly."""
        monitor = DeadlockMonitor()
        monitor.register_agent("agent-a", None, 1000.0)
        monitor.register_agent("agent-b", None, 2000.0)

        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-a", "/path/l1.py", 1001.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.ACQUIRED, "agent-b", "/path/l2.py", 2001.0)
        )
        await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-a", "/path/l2.py", 1002.0)
        )

        result = await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-b", "/path/l1.py", 2002.0)
        )

        assert result is not None
        assert result.victim_issue_id is None
        assert result.blocker_issue_id is None

    async def test_on_deadlock_callback_invoked(self) -> None:
        """on_deadlock callback is invoked when deadlock detected."""
        monitor = DeadlockMonitor()
        monitor.register_agent("agent-a", "issue-a", 1000.0)
        monitor.register_agent("agent-b", "issue-b", 2000.0)

        # Track callback invocations
        callback_calls: list[DeadlockInfo] = []

        async def on_deadlock(info: DeadlockInfo) -> None:
            callback_calls.append(info)

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
        assert len(callback_calls) == 0  # No deadlock yet

        # B waits for L1 - deadlock, callback should fire
        result = await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-b", "/path/l1.py", 2002.0)
        )

        assert result is not None
        assert len(callback_calls) == 1
        assert callback_calls[0] is result
        assert callback_calls[0].victim_id == "agent-b"

    async def test_on_deadlock_sync_callback_invoked(self) -> None:
        """Sync on_deadlock callback is also invoked."""
        monitor = DeadlockMonitor()
        monitor.register_agent("agent-a", "issue-a", 1000.0)
        monitor.register_agent("agent-b", "issue-b", 2000.0)

        callback_calls: list[DeadlockInfo] = []

        def on_deadlock(info: DeadlockInfo) -> None:
            callback_calls.append(info)

        monitor.on_deadlock = on_deadlock

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
        await monitor.handle_event(
            LockEvent(LockEventType.WAITING, "agent-b", "/path/l1.py", 2002.0)
        )

        assert len(callback_calls) == 1
        assert callback_calls[0].victim_id == "agent-b"
