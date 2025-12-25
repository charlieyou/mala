"""Backward compatibility shim for file locking.

This module re-exports locking functionality from the new location
at src/tools/locking.py. New code should import from there directly.
"""

from .tools.locking import (
    LOCK_DIR,
    file_lock,
    get_lock_holder,
    is_locked,
    release_all_locks,
)

__all__ = [
    "LOCK_DIR",
    "file_lock",
    "get_lock_holder",
    "is_locked",
    "release_all_locks",
]
