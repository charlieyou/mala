"""Deadlock detection domain model.

Provides WaitForGraph and DeadlockMonitor for detecting cycles in lock
acquisition patterns among parallel agents.

The WaitForGraph tracks:
- Which agents hold which locks (holds: dict[lock_path, agent_id])
- Which agents are waiting for which locks (waits: dict[agent_id, lock_path])

Cycle detection uses DFS from waiting agents to find circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class LockEventType(Enum):
    """Types of lock events emitted by agents."""

    ACQUIRED = "acquired"
    WAITING = "waiting"
    RELEASED = "released"


@dataclass
class LockEvent:
    """A lock event from an agent.

    Attributes:
        event_type: Type of lock event.
        agent_id: ID of the agent that emitted this event.
        lock_path: Path to the lock file.
        timestamp: Unix timestamp when the event occurred.
    """

    event_type: LockEventType
    agent_id: str
    lock_path: str
    timestamp: float


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
        self._holds[lock_path] = agent_id
        # Clear wait if this agent was waiting for this lock
        if self._waits.get(agent_id) == lock_path:
            del self._waits[agent_id]

    def add_wait(self, agent_id: str, lock_path: str) -> None:
        """Record that an agent is waiting for a lock.

        Args:
            agent_id: The agent that is waiting.
            lock_path: Path to the lock being waited on.
        """
        self._waits[agent_id] = lock_path

    def remove_hold(self, agent_id: str, lock_path: str) -> None:
        """Remove a hold record when a lock is released.

        Args:
            agent_id: The agent releasing the lock.
            lock_path: Path to the lock being released.
        """
        if self._holds.get(lock_path) == agent_id:
            del self._holds[lock_path]

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

        Uses DFS from each waiting agent to find circular dependencies.
        A cycle exists when following waits -> holds edges leads back to
        a previously visited agent.

        Returns:
            List of agent IDs in the cycle if found, None otherwise.
            The cycle is returned in order of discovery (first agent
            is where the cycle was detected).
        """
        for start_agent in self._waits:
            cycle = self._find_cycle_from(start_agent)
            if cycle:
                return cycle
        return None

    def _find_cycle_from(self, start_agent: str) -> list[str] | None:
        """DFS from a single agent to find a cycle.

        Args:
            start_agent: Agent to start searching from.

        Returns:
            Cycle path if found, None otherwise.
        """
        visited: set[str] = set()
        path: list[str] = []

        current = start_agent
        while current not in visited:
            visited.add(current)
            path.append(current)

            # What lock is this agent waiting for?
            lock_waiting = self._waits.get(current)
            if lock_waiting is None:
                # Agent not waiting for anything, no cycle through here
                return None

            # Who holds that lock?
            holder = self._holds.get(lock_waiting)
            if holder is None:
                # Lock not held, no cycle
                return None

            current = holder

        # Found a cycle - extract it from the path
        cycle_start_idx = path.index(current)
        return path[cycle_start_idx:]


class DeadlockMonitor:
    """Orchestrates deadlock detection and victim selection.

    Maintains a registry of active agents and their metadata, handles
    lock events to update the wait-for graph, and selects victims
    when deadlocks are detected.

    Victim selection picks the youngest agent (highest start_time) in
    the cycle to minimize wasted work.
    """

    def __init__(self) -> None:
        """Initialize the monitor with empty state."""
        self._graph = WaitForGraph()
        self._agents: dict[str, AgentInfo] = {}

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

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent and clear its state.

        Args:
            agent_id: Agent to unregister.
        """
        self._graph.remove_agent(agent_id)
        if agent_id in self._agents:
            del self._agents[agent_id]

    def handle_event(self, event: LockEvent) -> DeadlockInfo | None:
        """Process a lock event and check for deadlocks.

        Updates the wait-for graph based on the event type, then checks
        for cycles if the event indicates waiting.

        Args:
            event: The lock event to process.

        Returns:
            DeadlockInfo if a deadlock is detected, None otherwise.
        """
        if event.event_type == LockEventType.ACQUIRED:
            self._graph.add_hold(event.agent_id, event.lock_path)
        elif event.event_type == LockEventType.WAITING:
            self._graph.add_wait(event.agent_id, event.lock_path)
            # Check for deadlock after adding wait
            return self._check_for_deadlock(event.agent_id, event.lock_path)
        elif event.event_type == LockEventType.RELEASED:
            self._graph.remove_hold(event.agent_id, event.lock_path)

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
        if not cycle:
            return None

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
        return max(candidates, key=lambda a: a.start_time)
