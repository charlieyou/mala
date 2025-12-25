"""Filesystem-based file locking for multi-agent coordination."""

import os
import time
from pathlib import Path
from contextlib import contextmanager

LOCK_DIR = Path("/tmp/mala-locks")


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


def release_all_locks():
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
