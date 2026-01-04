"""Deadlock detection domain model.

Provides WaitForGraph and DeadlockMonitor for detecting cycles in lock
acquisition patterns among parallel agents.

The WaitForGraph tracks:
- Which agents hold which locks (holds: dict[lock_path, agent_id])
- Which agents are waiting for which locks (waits: dict[agent_id, lock_path])

Cycle detection uses DFS from waiting agents to find circular dependencies.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.models import LockEventType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.models import LockEvent

logger = logging.getLogger(__name__)

__all__ = [
    "AgentInfo",
    "DeadlockCallback",
    "DeadlockInfo",
    "DeadlockMonitor",
    "WaitForGraph",
]

# Type alias for the deadlock callback
DeadlockCallback = Callable[["DeadlockInfo"], Awaitable[None] | None]


@dataclass
class DeadlockInfo:
    """Information about a detected deadlock.

    Attributes:
        cycle: List of agent IDs forming the deadlock cycle.
        victim_id: Agent ID selected to be killed (youngest in cycle).
        victim_issue_id: Issue ID the victim was working on.
        blocked_on: Lock path the victim was waiting for.
        blocker_id: Agent ID holding the lock the victim needs.
        blocker_issue_id: Issue ID the blocker was working on.
    """

    cycle: list[str]
    victim_id: str
    victim_issue_id: str | None
    blocked_on: str
    blocker_id: str
    blocker_issue_id: str | None


@dataclass
class AgentInfo:
    """Metadata about a registered agent.

    Attributes:
        agent_id: Unique identifier for the agent.
        issue_id: Issue ID the agent is working on.
        start_time: Unix timestamp when the agent was registered.
    """

    agent_id: str
    issue_id: str | None
    start_time: float


class WaitForGraph:
    """Graph tracking lock holds and waits for cycle detection.

    The graph maintains two mappings:
    - holds: lock_path -> agent_id (who holds each lock)
    - waits: agent_id -> lock_path (what each agent is waiting for)

    Cycle detection walks from a waiting agent through the hold/wait
    edges to find circular dependencies.
    """

    def __init__(self) -> None:
        """Initialize empty graph."""
        self._holds: dict[str, str] = {}  # lock_path -> agent_id
        self._waits: dict[str, str] = {}  # agent_id -> lock_path

    def add_hold(self, agent_id: str, lock_path: str) -> None:
        """Record that an agent holds a lock.

        Args:
            agent_id: The agent that acquired the lock.
            lock_path: Path to the lock.
        """
        existing_holder = self._holds.get(lock_path)
        if existing_holder is not None and existing_holder != agent_id:
            logger.warning(
                "Invariant: ACQUIRED for lock held by other agent: "
                "lock=%s holder=%s new_agent=%s",
                lock_path,
                existing_holder,
                agent_id,
            )
        self._holds[lock_path] = agent_id
        logger.debug("Lock acquired: agent_id=%s lock_path=%s", agent_id, lock_path)
        # Clear wait if this agent was waiting for this lock
        if self._waits.get(agent_id) == lock_path:
            del self._waits[agent_id]

    def add_wait(self, agent_id: str, lock_path: str) -> None:
        """Record that an agent is waiting for a lock.

        Args:
            agent_id: The agent that is waiting.
            lock_path: Path to the lock being waited on.
        """
        # Check for invariant violations
        if self._holds.get(lock_path) == agent_id:
            logger.warning(
                "Invariant: WAITING on lock already held by same agent: "
                "agent=%s lock=%s",
                agent_id,
                lock_path,
            )
        old_wait = self._waits.get(agent_id)
        if old_wait is not None and old_wait != lock_path:
            logger.warning(
                "Wait edge overwritten: agent_id=%s old_lock=%s new_lock=%s",
                agent_id,
                old_wait,
                lock_path,
            )
        self._waits[agent_id] = lock_path
        logger.debug("Wait added: agent_id=%s lock_path=%s", agent_id, lock_path)

    def remove_hold(self, agent_id: str, lock_path: str) -> None:
        """Remove a hold record when a lock is released.

        Args:
            agent_id: The agent releasing the lock.
            lock_path: Path to the lock being released.
        """
        current_holder = self._holds.get(lock_path)
        if current_holder != agent_id:
            logger.warning(
                "Invariant: RELEASED for lock not held by agent: "
                "lock=%s holder=%s agent=%s",
                lock_path,
                current_holder,
                agent_id,
            )
        if current_holder == agent_id:
            del self._holds[lock_path]
            logger.debug("Lock released: agent_id=%s lock_path=%s", agent_id, lock_path)

    def remove_agent(self, agent_id: str) -> None:
        """Remove all state for an agent.

        Called when an agent exits (success or failure).

        Args:
            agent_id: The agent to remove.
        """
        # Remove wait entry
        if agent_id in self._waits:
            del self._waits[agent_id]
        # Remove all holds by this agent
        locks_to_remove = [
            lock for lock, holder in self._holds.items() if holder == agent_id
        ]
        for lock in locks_to_remove:
            del self._holds[lock]

    def get_holder(self, lock_path: str) -> str | None:
        """Get the agent holding a lock.

        Args:
            lock_path: Path to the lock.

        Returns:
            Agent ID if the lock is held, None otherwise.
        """
        return self._holds.get(lock_path)

    def get_waited_lock(self, agent_id: str) -> str | None:
        """Get the lock an agent is waiting for.

        Args:
            agent_id: The agent ID.

        Returns:
            Lock path if the agent is waiting, None otherwise.
        """
        return self._waits.get(agent_id)

    def detect_cycle(self) -> list[str] | None:
        """Detect a deadlock cycle in the wait-for graph.

        Uses single-pass DFS with three-color marking to achieve O(n) time
        complexity where n is the number of waiting agents. Each agent is
        fully processed at most once across all DFS starts.

        Colors:
        - WHITE (not in any set): unvisited
        - GRAY (in path): currently being explored in this DFS path
        - BLACK (in safe): fully explored, proven not to lead to a cycle

        Returns:
            List of agent IDs in the cycle if found, None otherwise.
            The cycle is returned in order of discovery (first agent
            is where the cycle was detected).
        """
        safe: set[str] = set()  # BLACK: agents proven not in any cycle

        for start_agent in self._waits:
            if start_agent in safe:
                continue

            cycle = self._find_cycle_from(start_agent, safe)
            if cycle:
                return cycle
        return None

    def _find_cycle_from(self, start_agent: str, safe: set[str]) -> list[str] | None:
        """DFS from a single agent to find a cycle.

        Updates the safe set with agents proven not to lead to a cycle.

        Args:
            start_agent: Agent to start searching from.
            safe: Set of agents already proven not to lead to a cycle.

        Returns:
            Cycle path if found, None otherwise.
        """
        path: list[str] = []
        path_set: set[str] = set()  # GRAY: agents in current path

        current = start_agent
        while True:
            if current in safe:
                # Reached a node proven safe, entire path is safe
                safe.update(path_set)
                return None

            if current in path_set:
                # Found a cycle - extract it from the path
                cycle_start_idx = path.index(current)
                return path[cycle_start_idx:]

            # What lock is this agent waiting for?
            lock_waiting = self._waits.get(current)
            if lock_waiting is None:
                # Agent not waiting for anything, path is safe
                safe.update(path_set)
                return None

            # Who holds that lock?
            holder = self._holds.get(lock_waiting)
            if holder is None:
                # Lock not held, path is safe
                safe.update(path_set)
                return None

            path.append(current)
            path_set.add(current)
            current = holder


class DeadlockMonitor:
    """Orchestrates deadlock detection and victim selection.

    Maintains a registry of active agents and their metadata, handles
    lock events to update the wait-for graph, and selects victims
    when deadlocks are detected.

    Victim selection picks the youngest agent (highest start_time) in
    the cycle to minimize wasted work.

    The on_deadlock callback is invoked when a deadlock is detected.
    If set, handle_event will call it with the DeadlockInfo. The
    callback may be sync or async.
    """

    def __init__(self) -> None:
        """Initialize the monitor with empty state."""
        self._graph = WaitForGraph()
        self._agents: dict[str, AgentInfo] = {}
        self.on_deadlock: DeadlockCallback | None = None

    def register_agent(
        self, agent_id: str, issue_id: str | None, start_time: float
    ) -> None:
        """Register an agent with the monitor.

        Args:
            agent_id: Unique identifier for the agent.
            issue_id: Issue the agent is working on (may be None).
            start_time: Unix timestamp when the agent started.
        """
        self._agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            issue_id=issue_id,
            start_time=start_time,
        )
        logger.info("Agent registered: agent_id=%s issue_id=%s", agent_id, issue_id)

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent and clear its state.

        Args:
            agent_id: Agent to unregister.
        """
        self._graph.remove_agent(agent_id)
        if agent_id in self._agents:
            del self._agents[agent_id]
        logger.info("Agent unregistered: agent_id=%s", agent_id)

    async def handle_event(self, event: LockEvent) -> DeadlockInfo | None:
        """Process a lock event and check for deadlocks.

        Updates the wait-for graph based on the event type, then checks
        for cycles if the event indicates waiting. If a deadlock is detected
        and on_deadlock is set, invokes the callback.

        Args:
            event: The lock event to process.

        Returns:
            DeadlockInfo if a deadlock is detected, None otherwise.
        """
        # Check for events from unregistered agents
        if event.agent_id not in self._agents:
            logger.warning("Event for unregistered agent: agent_id=%s", event.agent_id)

        logger.debug(
            "Event received: type=%s agent_id=%s lock_path=%s",
            event.event_type.value,
            event.agent_id,
            event.lock_path,
        )

        if event.event_type == LockEventType.ACQUIRED:
            self._graph.add_hold(event.agent_id, event.lock_path)
        elif event.event_type == LockEventType.WAITING:
            self._graph.add_wait(event.agent_id, event.lock_path)
            # Check for deadlock after adding wait
            deadlock_info = self._check_for_deadlock(event.agent_id, event.lock_path)
            if deadlock_info is not None and self.on_deadlock is not None:
                result = self.on_deadlock(deadlock_info)
                if asyncio.iscoroutine(result):
                    await result
            logger.debug(
                "Graph updated: holds=%d waits=%d",
                len(self._graph._holds),
                len(self._graph._waits),
            )
            return deadlock_info
        elif event.event_type == LockEventType.RELEASED:
            self._graph.remove_hold(event.agent_id, event.lock_path)

        logger.debug(
            "Graph updated: holds=%d waits=%d",
            len(self._graph._holds),
            len(self._graph._waits),
        )
        return None

    def _check_for_deadlock(
        self, waiting_agent: str, lock_path: str
    ) -> DeadlockInfo | None:
        """Check for deadlock and select victim if found.

        Args:
            waiting_agent: Agent that just started waiting.
            lock_path: Lock the agent is waiting for.

        Returns:
            DeadlockInfo with victim selection if deadlock detected.
        """
        cycle = self._graph.detect_cycle()
        logger.debug("Cycle check: found=%s", cycle is not None)
        if not cycle:
            return None

        logger.warning("Cycle detected: agents=%s", cycle)

        # Select victim: youngest agent (max start_time) in cycle
        victim = self._select_victim(cycle)
        if victim is None:
            # No registered agents in cycle (shouldn't happen)
            return None

        # Find what the victim is blocked on (use victim's wait, not triggering lock)
        victim_info = self._agents.get(victim.agent_id)
        victim_waited_lock = self._graph.get_waited_lock(victim.agent_id)
        blocked_on = victim_waited_lock or lock_path
        blocker_id = self._graph.get_holder(blocked_on)
        blocker_info = self._agents.get(blocker_id) if blocker_id else None

        return DeadlockInfo(
            cycle=cycle,
            victim_id=victim.agent_id,
            victim_issue_id=victim_info.issue_id if victim_info else None,
            blocked_on=blocked_on,
            blocker_id=blocker_id or "",
            blocker_issue_id=blocker_info.issue_id if blocker_info else None,
        )

    def _select_victim(self, cycle: Sequence[str]) -> AgentInfo | None:
        """Select the victim from a deadlock cycle.

        Picks the youngest agent (highest start_time) to minimize wasted work.

        Args:
            cycle: List of agent IDs in the deadlock cycle.

        Returns:
            AgentInfo for the selected victim, or None if no registered agents.
        """
        candidates = [self._agents[a] for a in cycle if a in self._agents]
        if not candidates:
            return None
        victim = max(candidates, key=lambda a: a.start_time)
        logger.info(
            "Victim selected: agent_id=%s start_time=%f (youngest in cycle)",
            victim.agent_id,
            victim.start_time,
        )
        return victim
