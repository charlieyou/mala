"""Fake locking backend for testing MCP tool handlers.

Provides an in-memory implementation of LockingBackend with observable state,
enabling tests to verify tool handler behavior without patching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.infra.tools.locking_mcp import LockingBackend


@dataclass
class FakeLockingBackend:
    """In-memory locking backend implementing LockingBackend protocol.

    Observable state:
        locks: dict mapping canonical path to agent_id that holds the lock
        try_lock_calls: list of (filepath, agent_id, ns) tuples
        release_lock_calls: list of (filepath, agent_id, ns) tuples
        wait_for_lock_calls: list of call records with parameters

    Configurable behavior:
        try_lock_results: dict of filepath -> bool (default: True)
        wait_for_lock_results: dict of filepath -> bool (default: True)
        holders: dict of filepath -> holder agent_id (for get_lock_holder)
        cleanup_result: (count, paths) to return from cleanup_agent_locks

    Example:
        >>> backend = FakeLockingBackend()
        >>> backend.try_lock("/a.py", "agent-1", None)
        True
        >>> backend.locks
        {'/a.py': 'agent-1'}
        >>> backend.try_lock_results["/b.py"] = False
        >>> backend.try_lock("/b.py", "agent-1", None)
        False
    """

    # Observable state
    locks: dict[str, str] = field(default_factory=dict)
    try_lock_calls: list[tuple[str, str, str | None]] = field(default_factory=list)
    release_lock_calls: list[tuple[str, str, str | None]] = field(default_factory=list)
    wait_for_lock_calls: list[dict] = field(default_factory=list)
    canonicalize_calls: list[tuple[str, str | None]] = field(default_factory=list)
    cleanup_calls: list[str] = field(default_factory=list)

    # Configurable behavior
    try_lock_results: dict[str, bool] = field(default_factory=dict)
    wait_for_lock_results: dict[str, bool] = field(default_factory=dict)
    holders: dict[str, str | None] = field(default_factory=dict)
    cleanup_result: tuple[int, list[str]] = field(default_factory=lambda: (0, []))
    canonicalize_fn: Callable[[str, str | None], str] | None = (
        None  # Override canonicalize_path
    )

    def canonicalize_path(self, path: str, namespace: str | None) -> str:
        """Canonicalize path. By default returns the path unchanged."""
        self.canonicalize_calls.append((path, namespace))
        if self.canonicalize_fn is not None:
            return self.canonicalize_fn(path, namespace)
        return path

    def try_lock(self, filepath: str, agent_id: str, ns: str | None) -> bool:
        """Try to acquire a lock. Uses try_lock_results or defaults to True."""
        self.try_lock_calls.append((filepath, agent_id, ns))
        if filepath in self.try_lock_results:
            result = self.try_lock_results[filepath]
        else:
            # Default: acquire if not held or held by same agent
            current = self.locks.get(filepath)
            result = current is None or current == agent_id
        if result:
            self.locks[filepath] = agent_id
        return result

    def release_lock(self, filepath: str, agent_id: str, ns: str | None) -> bool:
        """Release a lock. Returns True if released."""
        self.release_lock_calls.append((filepath, agent_id, ns))
        current = self.locks.get(filepath)
        if current == agent_id:
            del self.locks[filepath]
            return True
        return False

    def get_lock_holder(self, filepath: str, ns: str | None) -> str | None:
        """Get lock holder. Uses holders dict or falls back to locks."""
        if filepath in self.holders:
            return self.holders[filepath]
        return self.locks.get(filepath)

    async def wait_for_lock_async(
        self,
        filepath: str,
        agent_id: str,
        ns: str | None,
        timeout_seconds: float,
        poll_interval_ms: int,
    ) -> bool:
        """Wait for lock. Uses wait_for_lock_results or defaults to try_lock."""
        self.wait_for_lock_calls.append(
            {
                "filepath": filepath,
                "agent_id": agent_id,
                "ns": ns,
                "timeout_seconds": timeout_seconds,
                "poll_interval_ms": poll_interval_ms,
            }
        )
        if filepath in self.wait_for_lock_results:
            result = self.wait_for_lock_results[filepath]
            if result:
                self.locks[filepath] = agent_id
            return result
        # Default: behave like try_lock
        return self.try_lock(filepath, agent_id, ns)

    def cleanup_agent_locks(self, agent_id: str) -> tuple[int, list[str]]:
        """Cleanup all locks for an agent. Returns cleanup_result."""
        self.cleanup_calls.append(agent_id)
        return self.cleanup_result


# Protocol compliance assertion
def _assert_protocol_compliance() -> None:
    """Static assertion that FakeLockingBackend implements LockingBackend."""
    backend: LockingBackend = FakeLockingBackend()
    _ = backend  # Suppress unused variable warning
