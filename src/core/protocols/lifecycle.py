"""Lifecycle protocols for deadlock detection and lock events.

This module defines protocols for deadlock monitoring and lock event handling,
enabling the orchestrator to detect and resolve resource contention issues.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DeadlockInfoProtocol(Protocol):
    """Protocol for deadlock detection information.

    Matches the shape of domain.deadlock.DeadlockInfo for structural typing.
    """

    cycle: list[str]
    """List of agent IDs forming the deadlock cycle."""

    victim_id: str
    """Agent ID selected to be killed (youngest in cycle)."""

    victim_issue_id: str | None
    """Issue ID the victim was working on."""

    blocked_on: str
    """Lock path the victim was waiting for."""

    blocker_id: str
    """Agent ID holding the lock the victim needs."""

    blocker_issue_id: str | None
    """Issue ID the blocker was working on."""


@runtime_checkable
class LockEventProtocol(Protocol):
    """Protocol for lock events.

    Matches the shape of core.models.LockEvent for structural typing.
    """

    event_type: Any
    """Type of lock event (LockEventType enum value)."""

    agent_id: str
    """ID of the agent that emitted this event."""

    lock_path: str
    """Path to the lock file."""

    timestamp: float
    """Unix timestamp when the event occurred."""


@runtime_checkable
class DeadlockMonitorProtocol(Protocol):
    """Protocol for deadlock monitor.

    Matches the interface of domain.deadlock.DeadlockMonitor for structural typing.
    Only includes the handle_event method used by hooks.
    """

    async def handle_event(self, event: Any) -> Any:  # noqa: ANN401
        """Process a lock event and check for deadlocks.

        Args:
            event: The lock event to process (LockEvent).

        Returns:
            DeadlockInfo if a deadlock is detected, None otherwise.
        """
        ...
