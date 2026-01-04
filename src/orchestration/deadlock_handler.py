"""DeadlockHandler service for managing deadlock resolution.

This module provides the DeadlockHandler class which owns the asyncio.Lock
for serializing deadlock resolution and coordinates cleanup via callbacks.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from src.domain.deadlock import DeadlockInfo
    from src.orchestration.orchestrator_state import OrchestratorState  # noqa: F401


@dataclass
class DeadlockHandlerCallbacks:
    """Callback references for DeadlockHandler operations.

    These are callable getters that allow late binding to orchestrator state
    and external services (beads, event_sink, lock server).

    Attributes:
        add_dependency: Add dependency between issues. Args: (dependent_id, dependency_id).
        mark_needs_followup: Mark issue as needing followup. Args: (issue_id, summary, log_path).
        on_deadlock_detected: Event callback when deadlock is detected.
        on_locks_cleaned: Event callback when locks are cleaned. Args: (agent_id, count).
        on_tasks_aborting: Event callback when tasks are being aborted. Args: (count, reason).
        do_cleanup_agent_locks: Clean up locks held by agent. Args: (agent_id).
            Returns: (count, paths).
        unregister_agent: Unregister agent from deadlock monitor. Args: (agent_id).
    """

    add_dependency: Callable[[str, str], Awaitable[bool]]
    mark_needs_followup: Callable[[str, str, Path | None], Awaitable[None]]
    on_deadlock_detected: Callable[[DeadlockInfo], None]
    on_locks_cleaned: Callable[[str, int], None]
    on_tasks_aborting: Callable[[int, str], None]
    do_cleanup_agent_locks: Callable[[str], tuple[int, list[str]]]
    unregister_agent: Callable[[str], None]


class DeadlockHandler:
    """Service for handling deadlock detection and resolution.

    Owns the asyncio.Lock for serializing deadlock resolution and coordinates
    cleanup via callbacks to external services.
    """

    def __init__(self, callbacks: DeadlockHandlerCallbacks) -> None:
        """Initialize DeadlockHandler with callback references.

        Args:
            callbacks: Callback references for external operations.
        """
        self._callbacks = callbacks
        self._resolution_lock = asyncio.Lock()

    async def handle_deadlock(self, info: DeadlockInfo) -> None:
        """Handle a detected deadlock by cancelling victim and recording dependency.

        Called by DeadlockMonitor when a cycle is detected. Uses an asyncio.Lock
        to prevent concurrent resolution of multiple deadlocks.

        Args:
            info: DeadlockInfo with cycle, victim, and blocker details.
        """
        raise NotImplementedError("Method implementation in T004")

    async def abort_active_tasks(
        self,
        active_tasks: dict[str, asyncio.Task[object]],
        agent_ids: dict[str, str],
        abort_reason: str | None,
    ) -> None:
        """Cancel active tasks and prepare them for finalization.

        Tasks that have already completed are left for the caller to finalize
        with their real results rather than being marked as aborted.

        Args:
            active_tasks: Mapping of issue_id to asyncio.Task.
            agent_ids: Mapping of issue_id to agent_id.
            abort_reason: Reason for aborting tasks.
        """
        raise NotImplementedError("Method implementation in T004")

    def cleanup_agent_locks(self, agent_id: str) -> None:
        """Remove locks held by a specific agent (crash/timeout cleanup).

        Args:
            agent_id: The agent whose locks should be cleaned up.
        """
        raise NotImplementedError("Method implementation in T004")
