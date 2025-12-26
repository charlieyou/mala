"""Centralized file locking for multi-agent coordination.

Consolidates locking behavior from shell scripts.
"""

import hashlib
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .env import LOCK_DIR, SCRIPTS_DIR

if TYPE_CHECKING:
    from claude_agent_sdk.types import (
        HookContext,
        StopHookInput,
        SyncHookJSONOutput,
    )


def _canonicalize_path(filepath: str, repo_namespace: str | None = None) -> str:
    """Canonicalize a file path for consistent lock key generation.

    Normalizes paths by:
    - Resolving symlinks (if the path exists)
    - Making paths absolute
    - Making paths repo-relative when a namespace is provided
    - Normalizing . and .. segments

    Args:
        filepath: The file path to canonicalize.
        repo_namespace: Optional repo root path for making paths repo-relative.

    Returns:
        A canonicalized, repo-relative path string.
    """
    path = Path(filepath)

    # When we have a namespace and a relative path, resolve relative to the namespace
    if repo_namespace and not path.is_absolute():
        namespace_path = Path(repo_namespace).resolve()
        candidate = namespace_path / path

        if candidate.exists():
            # Path exists relative to namespace - resolve symlinks
            resolved = candidate.resolve()
        else:
            # Normalize . and .. segments relative to namespace
            resolved = Path(os.path.normpath(candidate))

        # Return repo-relative path
        try:
            relative = resolved.relative_to(namespace_path)
            return str(relative)
        except ValueError:
            return str(resolved)

    # Absolute path or no namespace
    if path.exists():
        resolved = path.resolve()  # Resolves symlinks
    else:
        if path.is_absolute():
            resolved = path
        else:
            resolved = Path.cwd() / path
        # Normalize . and .. segments
        resolved = Path(os.path.normpath(resolved))

    # Make repo-relative if namespace is provided
    if repo_namespace:
        namespace_path = Path(repo_namespace).resolve()
        try:
            relative = resolved.relative_to(namespace_path)
            return str(relative)
        except ValueError:
            # Path is not under the repo namespace, return as-is
            return str(resolved)

    return str(resolved)


def _lock_key(filepath: str, repo_namespace: str | None = None) -> str:
    """Build a canonical key for the lock.

    Args:
        filepath: The file path to lock.
        repo_namespace: Optional repo namespace for cross-repo disambiguation.

    Returns:
        The canonical key string.
    """
    # Treat empty namespace as None
    if repo_namespace == "":
        repo_namespace = None

    canonical_path = _canonicalize_path(filepath, repo_namespace)

    if repo_namespace:
        # Normalize namespace too for consistency
        namespace_key = str(Path(repo_namespace).resolve())
        return f"{namespace_key}:{canonical_path}"
    return canonical_path


def _lock_path(filepath: str, repo_namespace: str | None = None) -> Path:
    """Convert a file path to its lock file path.

    Uses SHA-256 hash of the canonical key to avoid collisions
    (e.g., 'a/b' vs 'a_b' which would both become 'a_b.lock' with simple replacement).

    Args:
        filepath: The file path to lock.
        repo_namespace: Optional repo namespace for cross-repo disambiguation.

    Returns:
        Path to the lock file.
    """
    key = _lock_key(filepath, repo_namespace)
    key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
    return LOCK_DIR / f"{key_hash}.lock"


def release_all_locks() -> None:
    """Release all locks in the lock directory."""
    if LOCK_DIR.exists():
        for lock in LOCK_DIR.glob("*.lock"):
            lock.unlink(missing_ok=True)


def try_lock(filepath: str, agent_id: str, repo_namespace: str | None = None) -> bool:
    """Try to acquire a lock on a file.

    Args:
        filepath: The file path to lock.
        agent_id: The agent ID to record in the lock.
        repo_namespace: Optional repo namespace for cross-repo disambiguation.

    Returns:
        True if lock was acquired, False if already locked.
    """
    lock_path = _lock_path(filepath, repo_namespace)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)

    # Fast-path if already locked
    if lock_path.exists():
        return False

    # Atomic lock creation using temp file + rename
    import tempfile

    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".locktmp.{agent_id}.", dir=LOCK_DIR, text=True
        )
        os.write(fd, f"{agent_id}\n".encode())
        os.close(fd)

        # Atomic hardlink attempt
        try:
            os.link(tmp_path, lock_path)
            os.unlink(tmp_path)
            return True
        except OSError:
            os.unlink(tmp_path)
            return False
    except OSError:
        return False


def wait_for_lock(
    filepath: str,
    agent_id: str,
    repo_namespace: str | None = None,
    timeout_seconds: float = 30.0,
    poll_interval_ms: int = 100,
) -> bool:
    """Wait for and acquire a lock on a file.

    Polls until the lock becomes available or timeout is reached.

    Args:
        filepath: The file path to lock.
        agent_id: The agent ID to record in the lock.
        repo_namespace: Optional repo namespace for cross-repo disambiguation.
        timeout_seconds: Maximum time to wait for the lock (default 30).
        poll_interval_ms: Polling interval in milliseconds (default 100).

    Returns:
        True if lock was acquired, False if timeout.
    """
    import time

    deadline = time.monotonic() + timeout_seconds
    poll_interval_sec = poll_interval_ms / 1000.0

    while True:
        if try_lock(filepath, agent_id, repo_namespace):
            return True

        if time.monotonic() >= deadline:
            return False

        time.sleep(poll_interval_sec)


def is_locked(filepath: str, repo_namespace: str | None = None) -> bool:
    """Check if a file is currently locked.

    Args:
        filepath: The file path to check.
        repo_namespace: Optional repo namespace for cross-repo disambiguation.

    Returns:
        True if the file is locked, False otherwise.
    """
    return _lock_path(filepath, repo_namespace).exists()


def get_lock_holder(filepath: str, repo_namespace: str | None = None) -> str | None:
    """Get the agent ID holding a lock, or None if not locked.

    Args:
        filepath: The file path to check.
        repo_namespace: Optional repo namespace for cross-repo disambiguation.

    Returns:
        The agent ID of the lock holder, or None if not locked.
    """
    lock_path = _lock_path(filepath, repo_namespace)
    if lock_path.exists():
        try:
            return lock_path.read_text().strip()
        except OSError:
            return None
    return None


def cleanup_agent_locks(agent_id: str) -> int:
    """Remove locks held by a specific agent (crash/timeout cleanup).

    Args:
        agent_id: The agent ID whose locks should be cleaned up.

    Returns:
        Number of locks cleaned up.
    """
    if not LOCK_DIR.exists():
        return 0

    cleaned = 0
    for lock in LOCK_DIR.glob("*.lock"):
        try:
            if lock.is_file() and lock.read_text().strip() == agent_id:
                lock.unlink()
                cleaned += 1
        except OSError:
            pass

    return cleaned


def make_stop_hook(agent_id: str):
    """Create a Stop hook that cleans up locks for the given agent.

    Args:
        agent_id: The agent ID to clean up locks for when the agent stops.

    Returns:
        An async hook function suitable for use with ClaudeAgentOptions.hooks["Stop"].
    """

    async def cleanup_locks_on_stop(
        hook_input: "StopHookInput",
        stderr: str | None,
        context: "HookContext",
    ) -> "SyncHookJSONOutput":
        """Stop hook to release all locks held by this agent."""
        script = SCRIPTS_DIR / "lock-release-all.sh"
        try:
            subprocess.run(
                [str(script)],
                env={
                    **os.environ,
                    "LOCK_DIR": str(LOCK_DIR),
                    "AGENT_ID": agent_id,
                },
                check=False,
                capture_output=True,
            )
        except Exception:
            pass  # Best effort cleanup, orchestrator has fallback
        return {}

    return cleanup_locks_on_stop
