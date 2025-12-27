"""Dependency sync optimization for mala validation.

Tracks pyproject.toml and uv.lock file hashes to skip redundant `uv sync`
operations when dependency files haven't changed.

Usage:
    state = DepsSyncState(cwd)
    if not state.needs_sync():
        print("Skipping uv sync - dependencies unchanged")
    else:
        run_uv_sync()
        state.mark_synced()  # Update state after successful sync
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime for .exists() and file I/O

# Files that indicate dependencies may have changed
DEPS_FILES = ("pyproject.toml", "uv.lock")

# State file name (stored in .uv directory to avoid polluting project root)
STATE_FILE = ".uv/sync-state"


def _compute_file_hash(path: Path) -> str | None:
    """Compute SHA-256 hash of a file, or None if file doesn't exist."""
    if not path.exists():
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _compute_deps_hash(cwd: Path) -> str:
    """Compute combined hash of all dependency files.

    If any dependency file is missing or unreadable, returns a hash that
    won't match any stored state, forcing a sync.
    """
    hashes: list[str] = []
    for filename in DEPS_FILES:
        file_hash = _compute_file_hash(cwd / filename)
        if file_hash is None:
            # Missing file - return a unique hash that won't match stored state
            return "missing-deps-file"
        hashes.append(file_hash)
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


@dataclass
class DepsSyncState:
    """Tracks dependency sync state for a project directory.

    Attributes:
        cwd: The working directory to check.
        force: If True, always requires sync regardless of state.
    """

    cwd: Path
    force: bool = False

    def _state_path(self) -> Path:
        """Get path to state file."""
        return self.cwd / STATE_FILE

    def _read_stored_hash(self) -> str | None:
        """Read previously stored hash, or None if no state exists."""
        state_path = self._state_path()
        if not state_path.exists():
            return None
        try:
            return state_path.read_text().strip()
        except OSError:
            return None

    def _write_stored_hash(self, hash_value: str) -> bool:
        """Write hash to state file. Returns True on success."""
        state_path = self._state_path()
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(hash_value + "\n")
            return True
        except OSError:
            return False

    def needs_sync(self) -> bool:
        """Check if uv sync needs to run.

        Returns True if:
        - Force sync is requested
        - No prior sync state exists (first run)
        - Dependency files have changed since last sync
        """
        if self.force:
            return True

        stored_hash = self._read_stored_hash()
        if stored_hash is None:
            return True

        current_hash = _compute_deps_hash(self.cwd)
        return current_hash != stored_hash

    def mark_synced(self) -> bool:
        """Record current dependency state after successful sync.

        Call this after `uv sync` completes successfully.

        Returns True if state was saved successfully.
        """
        current_hash = _compute_deps_hash(self.cwd)
        return self._write_stored_hash(current_hash)

    def clear(self) -> bool:
        """Clear stored sync state, forcing next sync.

        Returns True if state was cleared successfully (or didn't exist).
        """
        state_path = self._state_path()
        if not state_path.exists():
            return True
        try:
            state_path.unlink()
            return True
        except OSError:
            return False
