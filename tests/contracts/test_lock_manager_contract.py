"""Contract tests for LockManagerPort implementations.

Ensures FakeLockManager implements all methods of LockManagerPort protocol
and exhibits correct behavioral parity with the real LockManager.
"""

from pathlib import Path

import pytest

from src.core.protocols.infra import LockManagerPort
from tests.contracts import get_protocol_members
from tests.fakes.lock_manager import FakeLockManager


@pytest.mark.unit
def test_fake_lock_manager_implements_all_protocol_methods() -> None:
    """FakeLockManager must implement all public methods of LockManagerPort."""
    protocol_methods = get_protocol_members(LockManagerPort)
    fake_methods = {name for name in dir(FakeLockManager) if not name.startswith("_")}

    missing = protocol_methods - fake_methods
    assert not missing, f"FakeLockManager missing protocol methods: {sorted(missing)}"


@pytest.mark.unit
def test_fake_lock_manager_protocol_compliance() -> None:
    """FakeLockManager passes runtime isinstance check for LockManagerPort."""
    manager = FakeLockManager()
    assert isinstance(manager, LockManagerPort)


class TestFakeLockManagerBehavior:
    """Behavioral tests for FakeLockManager."""

    @pytest.mark.unit
    def test_lock_path_returns_path(self) -> None:
        """lock_path() returns a Path object."""
        manager = FakeLockManager()
        path = manager.lock_path("file.py", repo_namespace="test-repo")
        assert isinstance(path, Path)
        assert "file.py" in str(path)

    @pytest.mark.unit
    def test_try_lock_acquires_lock(self) -> None:
        """try_lock() acquires lock and updates locks dict."""
        manager = FakeLockManager()
        result = manager.try_lock("file.py", "agent-1")
        assert result is True
        assert manager.locks["file.py"] == "agent-1"

    @pytest.mark.unit
    def test_try_lock_rejects_if_held_by_other(self) -> None:
        """try_lock() returns False if lock held by another agent."""
        manager = FakeLockManager()
        manager.try_lock("file.py", "agent-1")
        result = manager.try_lock("file.py", "agent-2")
        assert result is False
        assert manager.locks["file.py"] == "agent-1"  # Still held by agent-1

    @pytest.mark.unit
    def test_try_lock_idempotent_for_same_agent(self) -> None:
        """try_lock() succeeds if same agent already holds lock."""
        manager = FakeLockManager()
        manager.try_lock("file.py", "agent-1")
        result = manager.try_lock("file.py", "agent-1")
        assert result is True

    @pytest.mark.unit
    def test_try_lock_records_attempts(self) -> None:
        """try_lock() records all acquisition attempts."""
        manager = FakeLockManager()
        manager.try_lock("file.py", "agent-1", repo_namespace="repo")
        manager.try_lock("file.py", "agent-2", repo_namespace="repo")
        assert len(manager.acquire_calls) == 2
        assert manager.acquire_calls[0].acquired is True
        assert manager.acquire_calls[1].acquired is False

    @pytest.mark.unit
    def test_wait_for_lock_behaves_like_try_lock(self) -> None:
        """wait_for_lock() in fake behaves same as try_lock (no actual waiting)."""
        manager = FakeLockManager()
        result = manager.wait_for_lock("file.py", "agent-1", timeout_seconds=60.0)
        assert result is True
        assert manager.locks["file.py"] == "agent-1"

    @pytest.mark.unit
    def test_release_lock_releases_if_held(self) -> None:
        """release_lock() releases lock if held by requesting agent."""
        manager = FakeLockManager()
        manager.try_lock("file.py", "agent-1")
        result = manager.release_lock("file.py", "agent-1")
        assert result is True
        assert "file.py" not in manager.locks

    @pytest.mark.unit
    def test_release_lock_fails_if_held_by_other(self) -> None:
        """release_lock() returns False if lock held by different agent."""
        manager = FakeLockManager()
        manager.try_lock("file.py", "agent-1")
        result = manager.release_lock("file.py", "agent-2")
        assert result is False
        assert manager.locks["file.py"] == "agent-1"  # Still held

    @pytest.mark.unit
    def test_release_lock_fails_if_not_held(self) -> None:
        """release_lock() returns False if lock not held."""
        manager = FakeLockManager()
        result = manager.release_lock("file.py", "agent-1")
        assert result is False
