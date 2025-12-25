"""Centralized file locking for multi-agent coordination.

Consolidates locking behavior from filelock.py and shell scripts.
"""

import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_agent_sdk.types import (
        HookContext,
        StopHookInput,
        SyncHookJSONOutput,
    )

LOCK_DIR = Path("/tmp/mala-locks")

# Lock scripts directory (relative to src/tools/locking.py -> ../.. -> scripts/)
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


def _lock_path(filepath: str) -> Path:
    """Convert a file path to its lock file path."""
    # Replace / to create a flat lock filename (match lock scripts)
    safe_name = filepath.replace("/", "_") + ".lock"
    return LOCK_DIR / safe_name


@contextmanager
def file_lock(filepath: str, timeout: int = 60):
    """
    Acquire exclusive lock on a file. Use before editing.

    Args:
        filepath: Path to the file to lock (relative or absolute)
        timeout: Max seconds to wait for lock (default 60)

    Raises:
        TimeoutError: If lock cannot be acquired within timeout

    Usage:
        with file_lock("src/foo.py"):
            # edit the file safely
    """
    LOCK_DIR.mkdir(exist_ok=True)
    lock_path = _lock_path(filepath)

    start = time.time()
    while True:
        try:
            # O_CREAT | O_EXCL ensures atomic creation - fails if exists
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}".encode())
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Could not acquire lock for {filepath}")
            time.sleep(1)

    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def release_all_locks() -> None:
    """Release all locks in the lock directory."""
    if LOCK_DIR.exists():
        for lock in LOCK_DIR.glob("*.lock"):
            lock.unlink(missing_ok=True)


def is_locked(filepath: str) -> bool:
    """Check if a file is currently locked."""
    return _lock_path(filepath).exists()


def get_lock_holder(filepath: str) -> int | None:
    """Get the PID of the process holding a lock, or None if not locked."""
    lock_path = _lock_path(filepath)
    if lock_path.exists():
        try:
            return int(lock_path.read_text().strip())
        except (ValueError, OSError):
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
