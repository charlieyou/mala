"""Deadlock detection domain model.

Provides WaitForGraph and DeadlockMonitor for detecting cycles in lock
acquisition patterns among parallel agents.

The WaitForGraph tracks:
- Which agents hold which locks (holds: dict[lock_path, agent_id])
- Which agents are waiting for which locks (waits: dict[agent_id, set[lock_path]])

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
    "WaitEdge",
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


@dataclass(frozen=True)
class WaitEdge:
    """One blocking wait-for edge: waiting agent --lock--> holder agent."""

    waiting_agent: str
    lock_path: str
    holder_agent: str


class WaitForGraph:
    """Graph tracking lock holds and waits for cycle detection.

    The graph maintains two mappings:
    - holds: lock_path -> agent_id (who holds each lock)
    - waits: agent_id -> set[lock_path] (what each agent is waiting for)

    Cycle detection walks from a waiting agent through the hold/wait
    edges to find circular dependencies.
    """

    def __init__(self) -> None:
        """Initialize empty graph."""
        self._holds: dict[str, str] = {}  # lock_path -> agent_id
        self._waits: dict[str, set[str]] = {}  # agent_id -> lock paths

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
        # Clear only this wait edge if the agent was waiting for this lock.
        waited_locks = self._waits.get(agent_id)
        if waited_locks is not None:
            waited_locks.discard(lock_path)
            if not waited_locks:
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
        self._waits.setdefault(agent_id, set()).add(lock_path)
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
        # Remove all wait entries
        self._waits.pop(agent_id, None)
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

    def get_waited_locks(self, agent_id: str) -> set[str]:
        """Get the locks an agent is waiting for.

        Args:
            agent_id: The agent ID.

        Returns:
            A copy of the lock paths the agent is waiting for.
        """
        return set(self._waits.get(agent_id, set()))

    def held_lock_count(self) -> int:
        """Return the number of currently held locks."""
        return len(self._holds)

    def waiting_agent_count(self) -> int:
        """Return the number of agents with at least one wait edge."""
        return len(self._waits)

    def waited_lock_count(self) -> int:
        """Return the total number of wait edges."""
        return sum(len(locks) for locks in self._waits.values())

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
        cycle_edges = self.detect_cycle_edges()
        if cycle_edges is None:
            return None
        return [edge.waiting_agent for edge in cycle_edges]

    def detect_cycle_edges(self) -> list[WaitEdge] | None:
        """Detect a deadlock cycle and return the lock edge for each hop.

        Returns:
            List of wait edges in cycle order if found, None otherwise.
            Each edge is ``waiting_agent --lock_path--> holder_agent``.
        """
        safe: set[str] = set()  # BLACK: agents proven not in any cycle

        for start_agent in list(self._waits):
            if start_agent in safe:
                continue

            cycle = self._find_cycle_edges_from(start_agent, safe)
            if cycle:
                return cycle
        return None

    def _find_cycle_edges_from(
        self, start_agent: str, safe: set[str]
    ) -> list[WaitEdge] | None:
        """DFS from a single agent to find a cycle.

        Updates the safe set with agents proven not to lead to a cycle.

        Args:
            start_agent: Agent to start searching from.
            safe: Set of agents already proven not to lead to a cycle.

        Returns:
            Cycle path if found, None otherwise.
        """
        visiting: dict[str, int] = {}  # GRAY: agent -> index in path_edges
        path_edges: list[WaitEdge] = []

        def dfs(agent_id: str) -> list[WaitEdge] | None:
            if agent_id in safe:
                return None

            visiting[agent_id] = len(path_edges)
            for edge in self._outgoing_edges(agent_id):
                if edge.holder_agent in visiting:
                    cycle_start_idx = visiting[edge.holder_agent]
                    return [*path_edges[cycle_start_idx:], edge]
                path_edges.append(edge)
                cycle = dfs(edge.holder_agent)
                if cycle is not None:
                    return cycle
                path_edges.pop()

            visiting.pop(agent_id, None)
            safe.add(agent_id)
            return None

        return dfs(start_agent)

    def _outgoing_edges(self, agent_id: str) -> list[WaitEdge]:
        """Return blocking wait edges for an agent in deterministic order."""
        edges: list[WaitEdge] = []
        for lock_path in sorted(self._waits.get(agent_id, set())):
            holder = self._holds.get(lock_path)
            if holder is None:
                continue
            if holder == agent_id:
                logger.warning(
                    "Invariant: wait edge points to same agent: agent=%s lock=%s",
                    agent_id,
                    lock_path,
                )
                continue
            edges.append(
                WaitEdge(
                    waiting_agent=agent_id,
                    lock_path=lock_path,
                    holder_agent=holder,
                )
            )
        return edges


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
            self._log_graph_state()
            return deadlock_info
        elif event.event_type == LockEventType.RELEASED:
            self._graph.remove_hold(event.agent_id, event.lock_path)

        self._log_graph_state()
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
        cycle_edges = self._graph.detect_cycle_edges()
        cycle = [edge.waiting_agent for edge in cycle_edges] if cycle_edges else None
        logger.debug("Cycle check: found=%s", cycle is not None)
        if not cycle:
            return None

        logger.warning(
            "Cycle detected: path=%s",
            self._format_cycle_edges(cycle_edges or []),
        )

        # Select victim: youngest agent (max start_time) in cycle
        victim = self._select_victim(cycle)
        if victim is None:
            # No registered agents in cycle (shouldn't happen)
            return None

        # Find what the victim is blocked on. Prefer the exact cycle edge
        # from victim -> next cycle agent so multi-wait victims report the
        # blocker that actually closes the detected cycle.
        victim_info = self._agents.get(victim.agent_id)
        victim_edge = next(
            (
                edge
                for edge in cycle_edges or []
                if edge.waiting_agent == victim.agent_id
            ),
            None,
        )
        if victim_edge is None:
            blocked_on, blocker_id = self._find_victim_blocker(
                victim.agent_id, set(cycle), fallback_lock=lock_path
            )
        else:
            blocked_on = victim_edge.lock_path
            blocker_id = victim_edge.holder_agent
        blocker_info = self._agents.get(blocker_id) if blocker_id else None

        return DeadlockInfo(
            cycle=cycle,
            victim_id=victim.agent_id,
            victim_issue_id=victim_info.issue_id if victim_info else None,
            blocked_on=blocked_on,
            blocker_id=blocker_id or "",
            blocker_issue_id=blocker_info.issue_id if blocker_info else None,
        )

    def _find_victim_blocker(
        self, victim_id: str, cycle_agents: set[str], *, fallback_lock: str
    ) -> tuple[str, str | None]:
        """Choose a deterministic blocker for a victim when edge metadata is absent."""
        for waited_lock in sorted(self._graph.get_waited_locks(victim_id)):
            holder = self._graph.get_holder(waited_lock)
            if holder in cycle_agents:
                return waited_lock, holder
        for waited_lock in sorted(self._graph.get_waited_locks(victim_id)):
            holder = self._graph.get_holder(waited_lock)
            if holder is not None:
                return waited_lock, holder
        return fallback_lock, self._graph.get_holder(fallback_lock)

    def _log_graph_state(self) -> None:
        """Log compact wait-for graph diagnostics for incident debugging."""
        logger.debug(
            "Graph updated: held_locks=%d waiting_agents=%d waited_locks=%d",
            self._graph.held_lock_count(),
            self._graph.waiting_agent_count(),
            self._graph.waited_lock_count(),
        )

    def diagnostics(self) -> dict[str, int]:
        """Return current monitor graph counts for status/debug callers."""
        return {
            "registered_agents": len(self._agents),
            "held_locks": self._graph.held_lock_count(),
            "waiting_agents": self._graph.waiting_agent_count(),
            "waited_locks": self._graph.waited_lock_count(),
        }

    def _format_cycle_edges(self, edges: Sequence[WaitEdge]) -> str:
        """Format cycle edges as ``agent --lock--> holder`` for logs."""
        return " | ".join(
            f"{edge.waiting_agent} --{edge.lock_path}--> {edge.holder_agent}"
            for edge in edges
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
