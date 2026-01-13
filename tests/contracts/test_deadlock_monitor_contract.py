"""Contract tests for DeadlockMonitorProtocol implementations.

Ensures FakeDeadlockMonitor implements all methods of DeadlockMonitorProtocol.
"""

from dataclasses import dataclass

import pytest

from src.core.protocols.lifecycle import DeadlockMonitorProtocol
from tests.contracts import get_protocol_members
from tests.fakes.deadlock_monitor import FakeDeadlockMonitor


@pytest.mark.unit
def test_fake_deadlock_monitor_implements_all_protocol_methods() -> None:
    """FakeDeadlockMonitor must implement all public methods of DeadlockMonitorProtocol."""
    protocol_methods = get_protocol_members(DeadlockMonitorProtocol)
    fake_methods = {
        name for name in dir(FakeDeadlockMonitor) if not name.startswith("_")
    }

    missing = protocol_methods - fake_methods
    assert not missing, (
        f"FakeDeadlockMonitor missing protocol methods: {sorted(missing)}"
    )


@pytest.mark.unit
def test_fake_deadlock_monitor_protocol_compliance() -> None:
    """FakeDeadlockMonitor passes runtime isinstance check for DeadlockMonitorProtocol."""
    monitor = FakeDeadlockMonitor()
    assert isinstance(monitor, DeadlockMonitorProtocol)


class TestFakeDeadlockMonitorBehavior:
    """Behavioral tests for FakeDeadlockMonitor."""

    @pytest.mark.unit
    async def test_handle_event_records_events(self) -> None:
        """handle_event() records all events passed to it."""
        monitor = FakeDeadlockMonitor()
        event1 = {"type": "lock_acquired", "agent": "agent-1"}
        event2 = {"type": "lock_released", "agent": "agent-2"}

        await monitor.handle_event(event1)
        await monitor.handle_event(event2)

        assert len(monitor.events) == 2
        assert monitor.events[0] == event1
        assert monitor.events[1] == event2

    @pytest.mark.unit
    async def test_handle_event_returns_none_by_default(self) -> None:
        """handle_event() returns None when no responses configured."""
        monitor = FakeDeadlockMonitor()
        result = await monitor.handle_event({"type": "lock_acquired"})
        assert result is None

    @pytest.mark.unit
    async def test_handle_event_returns_configured_responses(self) -> None:
        """handle_event() returns responses in order when configured."""

        @dataclass
        class FakeDeadlockInfo:
            cycle: list[str]
            victim_id: str

        info1 = FakeDeadlockInfo(cycle=["a", "b"], victim_id="a")
        info2 = FakeDeadlockInfo(cycle=["x", "y", "z"], victim_id="y")

        monitor = FakeDeadlockMonitor(responses=[info1, info2])

        result1 = await monitor.handle_event({})
        result2 = await monitor.handle_event({})
        result3 = await monitor.handle_event({})

        assert result1 == info1
        assert result2 == info2
        assert result3 is None  # Exhausted responses

    @pytest.mark.unit
    async def test_events_and_responses_independent(self) -> None:
        """Events are recorded regardless of response configuration."""
        monitor = FakeDeadlockMonitor(responses=["deadlock"])
        await monitor.handle_event("event1")
        await monitor.handle_event("event2")

        assert monitor.events == ["event1", "event2"]
