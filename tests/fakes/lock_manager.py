"""Fake lock manager for testing.

Provides an in-memory implementation of LockManagerPort with observable state
for testing lock acquisition, release, and contention scenarios.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.protocols.infra import LockManagerPort


@dataclass
class LockAcquireCall:
    """Record of a lock acquisition attempt."""

    filepath: str
    agent_id: str
    repo_namespace: str | None
    acquired: bool


@dataclass
class WaitForLockCall:
    """Record of a wait_for_lock call with its specific parameters."""

    filepath: str
    agent_id: str
    repo_namespace: str | None
    timeout_seconds: float
    poll_interval_ms: int
    acquired: bool


@dataclass
class FakeLockManager:
    """In-memory lock manager implementing LockManagerPort.

    Observable state:
        locks: dict mapping filepath to agent_id that holds the lock
        acquire_calls: list of all lock acquisition attempts

    Example:
        >>> manager = FakeLockManager()
        >>> manager.try_lock("file.py", "agent-1")
        True
        >>> manager.locks
        {'file.py': 'agent-1'}
        >>> manager.try_lock("file.py", "agent-2")
        False
        >>> manager.acquire_calls[-1].acquired
        False
    """

    locks: dict[str, str] = field(default_factory=dict)
    acquire_calls: list[LockAcquireCall] = field(default_factory=list)
    wait_for_lock_calls: list[WaitForLockCall] = field(default_factory=list)

    def lock_path(self, filepath: str, repo_namespace: str | None = None) -> Path:
        """Get the lock file path for a given filepath.

        Returns a synthetic path based on the filepath. In the fake,
        this is just used for consistency with the protocol.
        """
        ns = repo_namespace or "default"
        return Path(f"/fake-locks/{ns}/{filepath}.lock")

    def try_lock(
        self, filepath: str, agent_id: str, repo_namespace: str | None = None
    ) -> bool:
        """Try to acquire a lock without blocking.

        Returns True if lock was acquired, False if already held by another agent.
        Re-acquiring a lock already held by the same agent succeeds (idempotent).
        """
        current_holder = self.locks.get(filepath)
        if current_holder is None or current_holder == agent_id:
            self.locks[filepath] = agent_id
            self.acquire_calls.append(
                LockAcquireCall(filepath, agent_id, repo_namespace, acquired=True)
            )
            return True
        else:
            self.acquire_calls.append(
                LockAcquireCall(filepath, agent_id, repo_namespace, acquired=False)
            )
            return False

    def wait_for_lock(
        self,
        filepath: str,
        agent_id: str,
        repo_namespace: str | None = None,
        timeout_seconds: float = 30.0,
        poll_interval_ms: int = 100,
    ) -> bool:
        """Wait for and acquire a lock on a file.

        In the fake, this behaves identically to try_lock since there's no
        actual waiting. Tests that need to simulate timeout behavior should
        pre-populate the locks dict. Records call parameters for test assertions.
        """
        acquired = self.try_lock(filepath, agent_id, repo_namespace)
        self.wait_for_lock_calls.append(
            WaitForLockCall(
                filepath=filepath,
                agent_id=agent_id,
                repo_namespace=repo_namespace,
                timeout_seconds=timeout_seconds,
                poll_interval_ms=poll_interval_ms,
                acquired=acquired,
            )
        )
        return acquired

    def release_lock(
        self, filepath: str, agent_id: str, repo_namespace: str | None = None
    ) -> bool:
        """Release a lock on a file.

        Only releases if the lock is held by the specified agent_id.
        Returns True if released, False if not held by agent_id.
        """
        current_holder = self.locks.get(filepath)
        if current_holder == agent_id:
            del self.locks[filepath]
            return True
        return False


# Protocol compliance assertion
def _assert_protocol_compliance() -> None:
    """Static assertion that FakeLockManager implements LockManagerPort."""
    manager: LockManagerPort = FakeLockManager()
    _ = manager  # Suppress unused variable warning
