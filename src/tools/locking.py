"""Centralized file locking for multi-agent coordination.

Consolidates locking behavior from shell scripts.
"""

import hashlib
import os
import sys
from pathlib import Path

from .env import get_lock_dir

__all__ = ["get_lock_dir"]


def _get_lock_dir() -> Path:
    """Get the lock directory using the accessor from env module.

    This allows tests to either:
    1. Patch os.environ["MALA_LOCK_DIR"] before calling
    2. Patch src.tools.env.get_lock_dir if more control needed
    """
    return get_lock_dir()


def _canonicalize_path(filepath: str, repo_namespace: str | None = None) -> str:
    """Canonicalize a file path for consistent lock key generation.

    Normalizes paths by:
    - Resolving symlinks (if the path exists)
    - Making paths absolute
    - Normalizing . and .. segments

    This matches the shell script behavior (realpath -m), which always produces
    absolute paths. The repo_namespace is used by _lock_key to build the final
    key as "namespace:absolute_path".

    Args:
        filepath: The file path to canonicalize.
        repo_namespace: Optional repo root path for resolving relative paths.
            When provided and filepath is relative, the path is resolved
            relative to the namespace directory (mimicking cwd=repo behavior).

    Returns:
        A canonicalized absolute path string.
    """
    path = Path(filepath)

    # When we have a namespace and a relative path, resolve relative to the namespace
    # This mimics shell script behavior when cwd is the repo directory
    if repo_namespace and not path.is_absolute():
        namespace_path = Path(repo_namespace).resolve()
        candidate = namespace_path / path

        if candidate.exists():
            # Path exists - resolve symlinks
            return str(candidate.resolve())
        else:
            # Normalize . and .. segments
            return str(Path(os.path.normpath(candidate)))

    # Absolute path or no namespace - resolve to absolute
    if path.exists():
        return str(path.resolve())  # Resolves symlinks
    else:
        if path.is_absolute():
            resolved = path
        else:
            resolved = Path.cwd() / path
        # Normalize . and .. segments
        return str(Path(os.path.normpath(resolved)))


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
        # Use namespace as-is to match shell script behavior
        # Shell scripts pass REPO_NAMESPACE directly without normalizing
        return f"{repo_namespace}:{canonical_path}"
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
    return _get_lock_dir() / f"{key_hash}.lock"


def release_all_locks() -> None:
    """Release all locks in the lock directory."""
    lock_dir = _get_lock_dir()
    if lock_dir.exists():
        for lock in lock_dir.glob("*.lock"):
            lock.unlink(missing_ok=True)


def release_run_locks(agent_ids: list[str]) -> int:
    """Release locks owned by the specified agent IDs.

    Used by orchestrator shutdown to only clean up locks from this run,
    leaving locks from other concurrent runs intact.

    Args:
        agent_ids: List of agent IDs whose locks should be released.

    Returns:
        Number of locks released.
    """
    lock_dir = _get_lock_dir()
    if not lock_dir.exists() or not agent_ids:
        return 0

    agent_set = set(agent_ids)
    released = 0
    for lock in lock_dir.glob("*.lock"):
        try:
            if lock.is_file() and lock.read_text().strip() in agent_set:
                lock.unlink()
                released += 1
        except OSError:
            pass

    return released


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
    lock_dir = _get_lock_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)

    # Fast-path if already locked
    if lock_path.exists():
        return False

    # Atomic lock creation using temp file + rename
    import tempfile

    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".locktmp.{agent_id}.", dir=lock_dir, text=True
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
    if not _get_lock_dir().exists():
        return 0

    cleaned = 0
    for lock in _get_lock_dir().glob("*.lock"):
        try:
            if lock.is_file() and lock.read_text().strip() == agent_id:
                lock.unlink()
                cleaned += 1
        except OSError:
            pass

    return cleaned


def _cli_main() -> int:
    """CLI entry point for shell script delegation.

    Usage:
        python -m src.tools.locking try <filepath>
        python -m src.tools.locking wait <filepath> [timeout_seconds] [poll_interval_ms]
        python -m src.tools.locking check <filepath>
        python -m src.tools.locking holder <filepath>
        python -m src.tools.locking release <filepath>
        python -m src.tools.locking release-all

    Environment variables:
        LOCK_DIR: Directory for lock files (required for most commands)
        AGENT_ID: Agent identifier (required for most commands)
        REPO_NAMESPACE: Optional repo namespace for cross-repo disambiguation

    Exit codes:
        0: Success (lock acquired, held, released, etc.)
        1: Failure (lock blocked, timeout, not held, etc.)
        2: Usage error (missing env vars, invalid arguments)
    """
    if len(sys.argv) < 2:
        print("Usage: python -m src.tools.locking <command> [args...]", file=sys.stderr)
        print(
            "Commands: try, wait, check, holder, release, release-all", file=sys.stderr
        )
        return 2

    command = sys.argv[1]
    lock_dir = os.environ.get("LOCK_DIR")
    agent_id = os.environ.get("AGENT_ID")
    repo_namespace = os.environ.get("REPO_NAMESPACE") or None

    # Commands that require LOCK_DIR
    if command in ("try", "wait", "check", "holder", "release", "release-all"):
        if not lock_dir:
            print("Error: LOCK_DIR must be set", file=sys.stderr)
            return 2

    # Commands that require AGENT_ID
    if command in ("try", "wait", "check", "release", "release-all"):
        if not agent_id:
            print("Error: AGENT_ID must be set", file=sys.stderr)
            return 2

    # Set MALA_LOCK_DIR so our functions use the shell script's LOCK_DIR
    os.environ["MALA_LOCK_DIR"] = lock_dir or ""

    if command == "try":
        if len(sys.argv) != 3:
            print("Usage: python -m src.tools.locking try <filepath>", file=sys.stderr)
            return 2
        filepath = sys.argv[2]
        if try_lock(filepath, agent_id, repo_namespace):  # type: ignore[arg-type]
            return 0
        return 1

    elif command == "wait":
        if len(sys.argv) < 3:
            print(
                "Usage: python -m src.tools.locking wait <filepath> [timeout_seconds] [poll_interval_ms]",
                file=sys.stderr,
            )
            return 2
        filepath = sys.argv[2]
        timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
        poll_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 100
        if wait_for_lock(filepath, agent_id, repo_namespace, timeout, poll_ms):  # type: ignore[arg-type]
            return 0
        return 1

    elif command == "check":
        if len(sys.argv) != 3:
            print(
                "Usage: python -m src.tools.locking check <filepath>", file=sys.stderr
            )
            return 2
        filepath = sys.argv[2]
        holder = get_lock_holder(filepath, repo_namespace)
        if holder == agent_id:
            return 0
        return 1

    elif command == "holder":
        if len(sys.argv) != 3:
            print(
                "Usage: python -m src.tools.locking holder <filepath>", file=sys.stderr
            )
            return 2
        filepath = sys.argv[2]
        holder = get_lock_holder(filepath, repo_namespace)
        if holder:
            print(holder)
        return 0

    elif command == "release":
        if len(sys.argv) != 3:
            print(
                "Usage: python -m src.tools.locking release <filepath>", file=sys.stderr
            )
            return 2
        filepath = sys.argv[2]
        # Only release if we hold the lock
        holder = get_lock_holder(filepath, repo_namespace)
        if holder == agent_id:
            lock_path = _lock_path(filepath, repo_namespace)
            lock_path.unlink(missing_ok=True)
        return 0

    elif command == "release-all":
        if len(sys.argv) != 2:
            print("Usage: python -m src.tools.locking release-all", file=sys.stderr)
            return 2
        cleanup_agent_locks(agent_id)  # type: ignore[arg-type]
        return 0

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print(
            "Commands: try, wait, check, holder, release, release-all", file=sys.stderr
        )
        return 2


if __name__ == "__main__":
    sys.exit(_cli_main())
