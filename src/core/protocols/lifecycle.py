"""Lifecycle protocols for deadlock detection and lock events.

This module defines protocols for deadlock monitoring and lock event handling,
enabling the orchestrator to detect and resolve resource contention issues.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import asyncio
    from pathlib import Path


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


@runtime_checkable
class ISessionLifecycle(Protocol):
    """Protocol for session lifecycle operations.

    This protocol defines methods for session logging, abort handling,
    and event callbacks. It replaces the get_log_path, get_log_offset,
    on_abort, get_abort_event, on_tool_use, and on_agent_text callbacks
    from SessionCallbacks.

    The canonical implementation is SessionCallbackFactory in
    src/pipeline/session_callback_factory.py.
    """

    def get_log_path(self, session_id: str) -> Path:
        """Get the log file path for a session.

        Args:
            session_id: The session ID.

        Returns:
            Path to the JSONL log file for this session.
        """
        ...

    def get_log_offset(self, log_path: Path, start_offset: int) -> int:
        """Get the current byte offset at the end of a log file.

        Args:
            log_path: Path to the JSONL log file.
            start_offset: Byte offset to start from.

        Returns:
            The byte offset at the end of the file.
        """
        ...

    def on_abort(self, reason: str) -> None:
        """Handle session abort.

        Args:
            reason: The reason for aborting the session.
        """
        ...

    def get_abort_event(self) -> asyncio.Event | None:
        """Get the abort event for run abort detection.

        Returns:
            An asyncio.Event that is set when abort is requested,
            or None if abort detection is not available.
        """
        ...

    def on_tool_use(
        self, agent_id: str, tool_name: str, args: dict[str, Any] | None
    ) -> None:
        """Handle tool use event from SDK.

        Args:
            agent_id: The agent ID that used the tool.
            tool_name: The name of the tool used.
            args: The tool arguments, or None if no arguments.
        """
        ...

    def on_agent_text(self, agent_id: str, text: str) -> None:
        """Handle text output event from SDK.

        Args:
            agent_id: The agent ID that produced the text.
            text: The text output.
        """
        ...
