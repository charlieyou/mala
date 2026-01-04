"""Centralized file locking for multi-agent coordination.

Consolidates locking behavior from shell scripts.
"""

import hashlib
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .env import get_lock_dir

logger = logging.getLogger(__name__)

__all__ = [
    "LockManager",
    "canonicalize_path",
    "get_all_locks",
    "get_lock_dir",
    "get_lock_holder",
    "lock_path",
    "parse_lock_file",
    "wait_for_lock_async",
]


def canonicalize_path(filepath: str, repo_namespace: str | None = None) -> str:
    """Canonicalize a file path for consistent lock key generation.

    Public wrapper for _canonicalize_path. See _canonicalize_path for details.

    Args:
        filepath: The file path to canonicalize.
        repo_namespace: Optional repo root path for resolving relative paths.

    Returns:
        A canonicalized absolute path string, or the literal key as-is.
    """
    return _canonicalize_path(filepath, repo_namespace)


def _get_lock_dir() -> Path:
    """Get the lock directory using the accessor from env module.

    This allows tests to either:
    1. Patch os.environ["MALA_LOCK_DIR"] before calling
    2. Patch src.infra.tools.env.get_lock_dir if more control needed
    """
    return get_lock_dir()


def _is_literal_key(filepath: str) -> bool:
    """Check if a filepath is a literal key (not a real path).

    Literal keys are special identifiers like __test_mutex__ that should
    not be normalized as file paths. They are used for global locks that
    are not tied to specific files.
    """
    return filepath.startswith("__") and filepath.endswith("__")


def _resolve_with_parents(path: Path) -> Path:
    """Resolve a path by resolving existing parent directories.

    For non-existent paths, walks up to find the first existing ancestor,
    resolves its symlinks, then appends the remaining path components.
    This ensures consistent lock keys for paths through symlinked directories.

    Args:
        path: The path to resolve (should be absolute).

    Returns:
        The resolved path with parent symlinks resolved.
    """
    if path.exists():
        return path.resolve()

    # Walk up to find an existing ancestor
    # Collect the parts that don't exist yet
    missing_parts: list[str] = []
    current = path

    max_iterations = 100
    iterations = 0
    while not current.exists() and iterations < max_iterations:
        iterations += 1
        missing_parts.append(current.name)
        parent = current.parent
        if parent == current:
            # Reached root without finding existing path
            break
        current = parent

    if iterations >= max_iterations:
        import logging

        logging.warning(
            f"_resolve_with_parents: max iterations reached for path {path}, "
            "using unresolved path which may cause inconsistent lock keys"
        )
        return path

    # Resolve the existing ancestor (resolves symlinks)
    resolved_base = current.resolve()

    # Append the missing parts back
    for part in reversed(missing_parts):
        resolved_base = resolved_base / part

    return resolved_base


def _canonicalize_path(filepath: str, repo_namespace: str | None = None) -> str:
    """Canonicalize a file path for consistent lock key generation.

    Normalizes paths by:
    - Resolving symlinks (including parent directory symlinks for non-existent paths)
    - Making paths absolute
    - Normalizing . and .. segments

    Literal keys (like __test_mutex__) are returned as-is without normalization.

    This matches the shell script behavior (realpath -m), which always produces
    absolute paths. The repo_namespace is used by _lock_key to build the final
    key as "namespace:absolute_path".

    Args:
        filepath: The file path to canonicalize.
        repo_namespace: Optional repo root path for resolving relative paths.
            When provided and filepath is relative, the path is resolved
            relative to the namespace directory (mimicking cwd=repo behavior).

    Returns:
        A canonicalized absolute path string, or the literal key as-is.
    """
    # Skip normalization for literal keys (non-path identifiers like __test_mutex__)
    if _is_literal_key(filepath):
        return filepath

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
            # Normalize and resolve parent symlinks for non-existent paths
            normalized = Path(os.path.normpath(candidate))
            return str(_resolve_with_parents(normalized))

    # Absolute path or no namespace - resolve to absolute
    if path.exists():
        return str(path.resolve())  # Resolves symlinks
    else:
        if path.is_absolute():
            resolved = path
        else:
            resolved = Path.cwd() / path
        # Normalize . and .. segments, then resolve parent symlinks
        normalized = Path(os.path.normpath(resolved))
        return str(_resolve_with_parents(normalized))


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


def lock_path(filepath: str, repo_namespace: str | None = None) -> Path:
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
            # Also remove companion .meta file
            lock.with_suffix(".meta").unlink(missing_ok=True)
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
                # Also remove companion .meta file
                lock.with_suffix(".meta").unlink(missing_ok=True)
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
    lp = lock_path(filepath, repo_namespace)
    lock_dir = _get_lock_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)

    # Fast-path if already locked
    if lp.exists():
        # Get holder to check for re-entrant acquire
        try:
            holder = lp.read_text().strip()
            if holder == agent_id:
                # Idempotent: same agent already holds the lock
                logger.debug(
                    "Lock re-acquired (idempotent): path=%s agent_id=%s",
                    filepath,
                    agent_id,
                )
                return True
            logger.debug(
                "Lock contention: path=%s holder=%s requester=%s",
                filepath,
                holder,
                agent_id,
            )
        except OSError:
            pass
        return False

    # Atomic lock creation using temp file + rename
    import tempfile

    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".locktmp.{agent_id}.", dir=lock_dir, text=True
        )
        # Lock file contains only agent_id (simple, atomic reads)
        # Filepath is stored in companion .meta file for diagnostics
        canonical = _canonicalize_path(filepath, repo_namespace)
        os.write(fd, f"{agent_id}\n".encode())
        os.close(fd)

        # Atomic hardlink attempt
        try:
            os.link(tmp_path, lp)
            os.unlink(tmp_path)
            # Write meta file after successful lock acquisition
            # If meta write fails, we still own the lock - proceed anyway
            try:
                meta_path = lp.with_suffix(".meta")
                meta_path.write_text(f"{canonical}\n")
            except OSError:
                pass  # Lock acquired; meta is optional
            logger.debug("Lock acquired: path=%s agent_id=%s", filepath, agent_id)
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
            logger.warning(
                "Lock timeout: path=%s agent_id=%s after=%.1fs",
                filepath,
                agent_id,
                timeout_seconds,
            )
            return False

        time.sleep(poll_interval_sec)


async def wait_for_lock_async(
    filepath: str,
    agent_id: str,
    repo_namespace: str | None = None,
    timeout_seconds: float = 30.0,
    poll_interval_ms: int = 100,
) -> bool:
    """Wait for and acquire a lock on a file asynchronously.

    Non-blocking async version of wait_for_lock. Uses asyncio.sleep
    instead of time.sleep to avoid blocking the event loop.

    Args:
        filepath: The file path to lock.
        agent_id: The agent ID to record in the lock.
        repo_namespace: Optional repo namespace for cross-repo disambiguation.
        timeout_seconds: Maximum time to wait for the lock (default 30).
        poll_interval_ms: Polling interval in milliseconds (default 100).

    Returns:
        True if lock was acquired, False if timeout.
    """
    import asyncio
    import time

    deadline = time.monotonic() + timeout_seconds
    poll_interval_sec = poll_interval_ms / 1000.0

    while True:
        if try_lock(filepath, agent_id, repo_namespace):
            return True

        if time.monotonic() >= deadline:
            logger.warning(
                "Lock timeout: path=%s agent_id=%s after=%.1fs",
                filepath,
                agent_id,
                timeout_seconds,
            )
            return False

        await asyncio.sleep(poll_interval_sec)


def is_locked(filepath: str, repo_namespace: str | None = None) -> bool:
    """Check if a file is currently locked.

    Args:
        filepath: The file path to check.
        repo_namespace: Optional repo namespace for cross-repo disambiguation.

    Returns:
        True if the file is locked, False otherwise.
    """
    return lock_path(filepath, repo_namespace).exists()


def release_lock(
    filepath: str, agent_id: str, repo_namespace: str | None = None
) -> bool:
    """Release a lock on a file.

    Only releases the lock if it is held by the specified agent_id.
    This prevents accidental or malicious release of locks held by
    other agents.

    Args:
        filepath: Path to the file to unlock.
        agent_id: Identifier of the agent releasing the lock.
        repo_namespace: Optional repo namespace for cross-repo disambiguation.

    Returns:
        True if lock was released, False if lock was not held by agent_id.
    """
    holder = get_lock_holder(filepath, repo_namespace)
    if holder != agent_id:
        return False
    lp = lock_path(filepath, repo_namespace)
    # Also remove companion .meta file
    lp.with_suffix(".meta").unlink(missing_ok=True)
    lp.unlink(missing_ok=True)
    logger.debug("Lock released: path=%s agent_id=%s", filepath, agent_id)
    return True


def get_lock_holder(filepath: str, repo_namespace: str | None = None) -> str | None:
    """Get the agent ID holding a lock, or None if not locked.

    Args:
        filepath: The file path to check.
        repo_namespace: Optional repo namespace for cross-repo disambiguation.

    Returns:
        The agent ID of the lock holder, or None if not locked.
    """
    lp = lock_path(filepath, repo_namespace)
    if lp.exists():
        try:
            return lp.read_text().strip()
        except OSError:
            return None
    return None


def parse_lock_file(lock_file: Path) -> tuple[str, str | None] | None:
    """Parse a lock file to get agent_id and original filepath.

    Args:
        lock_file: Path to the lock file (.lock file).

    Returns:
        Tuple of (agent_id, filepath) or None if file cannot be read.
        filepath may be None for legacy lock files without a .meta file.
    """
    try:
        agent_id = lock_file.read_text().strip()
        if not agent_id:
            return None
        # Read filepath from companion .meta file
        meta_file = lock_file.with_suffix(".meta")
        filepath = meta_file.read_text().strip() if meta_file.exists() else None
        return (agent_id, filepath)
    except OSError:
        return None


def get_all_locks() -> dict[str, list[str]]:
    """Get all active locks grouped by agent ID.

    Returns:
        Dictionary mapping agent_id -> list of locked filepaths.
        Filepaths may be the hash stem for legacy locks without filepath info.
    """
    lock_dir = _get_lock_dir()
    if not lock_dir.exists():
        return {}

    locks_by_agent: dict[str, list[str]] = {}
    for lock in lock_dir.glob("*.lock"):
        parsed = parse_lock_file(lock)
        if parsed:
            agent_id, filepath = parsed
            if agent_id not in locks_by_agent:
                locks_by_agent[agent_id] = []
            # Use filepath if available, else fall back to hash stem
            locks_by_agent[agent_id].append(filepath or lock.stem)

    # Clean up orphaned .meta files (whose .lock was deleted externally)
    for meta in lock_dir.glob("*.meta"):
        if not meta.with_suffix(".lock").exists():
            try:
                meta.unlink()
            except OSError:
                pass

    return locks_by_agent


def cleanup_agent_locks(agent_id: str) -> tuple[int, list[str]]:
    """Remove locks held by a specific agent (crash/timeout cleanup).

    Args:
        agent_id: The agent ID whose locks should be cleaned up.

    Returns:
        Tuple of (count cleaned, list of released file paths).
    """
    if not _get_lock_dir().exists():
        return 0, []

    cleaned = 0
    released_paths: list[str] = []
    for lock in _get_lock_dir().glob("*.lock"):
        try:
            if lock.is_file() and lock.read_text().strip() == agent_id:
                # Extract original filepath from .meta file before deleting
                meta_path = lock.with_suffix(".meta")
                original_path: str | None = None
                if meta_path.is_file():
                    try:
                        original_path = meta_path.read_text().strip()
                    except OSError:
                        pass
                    meta_path.unlink(missing_ok=True)
                lock.unlink()
                cleaned += 1
                if original_path:
                    released_paths.append(original_path)
        except OSError:
            pass

    logger.info("Agent locks cleaned: agent_id=%s count=%d", agent_id, cleaned)
    return cleaned, released_paths


class LockManager:
    """Implementation of LockManagerPort using standalone locking functions.

    This class provides an object-oriented wrapper around the standalone locking
    functions, enabling dependency injection into domain modules.
    """

    def lock_path(self, filepath: str, repo_namespace: str | None = None) -> Path:
        """Get the lock file path for a given filepath."""
        return lock_path(filepath, repo_namespace)

    def try_lock(
        self, filepath: str, agent_id: str, repo_namespace: str | None = None
    ) -> bool:
        """Try to acquire a lock without blocking."""
        return try_lock(filepath, agent_id, repo_namespace)

    def wait_for_lock(
        self,
        filepath: str,
        agent_id: str,
        repo_namespace: str | None = None,
        timeout_seconds: float = 30.0,
        poll_interval_ms: int = 100,
    ) -> bool:
        """Wait for and acquire a lock on a file."""
        return wait_for_lock(
            filepath, agent_id, repo_namespace, timeout_seconds, poll_interval_ms
        )

    def release_lock(
        self, filepath: str, agent_id: str, repo_namespace: str | None = None
    ) -> bool:
        """Release a lock on a file."""
        return release_lock(filepath, agent_id, repo_namespace)


# ---------------------------------------------------------------------------
# CLI Command Dispatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CliContext:
    """Parsed CLI context for command dispatch."""

    command: str
    lock_dir: str
    agent_id: str
    repo_namespace: str | None
    filepath: str | None
    timeout: float
    poll_ms: int


def _cmd_try(ctx: CliContext) -> int:
    """Handle 'try' command: attempt to acquire lock."""
    if try_lock(ctx.filepath, ctx.agent_id, ctx.repo_namespace):  # type: ignore[arg-type]
        return 0
    return 1


def _cmd_wait(ctx: CliContext) -> int:
    """Handle 'wait' command: wait for lock with timeout."""
    if wait_for_lock(
        ctx.filepath,  # type: ignore[arg-type]
        ctx.agent_id,
        ctx.repo_namespace,
        ctx.timeout,
        ctx.poll_ms,
    ):
        return 0
    return 1


def _cmd_check(ctx: CliContext) -> int:
    """Handle 'check' command: check if we hold the lock."""
    holder = get_lock_holder(ctx.filepath, ctx.repo_namespace)  # type: ignore[arg-type]
    if holder == ctx.agent_id:
        return 0
    return 1


def _cmd_holder(ctx: CliContext) -> int:
    """Handle 'holder' command: print lock holder."""
    holder = get_lock_holder(ctx.filepath, ctx.repo_namespace)  # type: ignore[arg-type]
    if holder:
        print(holder)
    return 0


def _cmd_release(ctx: CliContext) -> int:
    """Handle 'release' command: release lock if we hold it."""
    holder = get_lock_holder(ctx.filepath, ctx.repo_namespace)  # type: ignore[arg-type]
    if holder == ctx.agent_id:
        lp = lock_path(ctx.filepath, ctx.repo_namespace)  # type: ignore[arg-type]
        lp.with_suffix(".meta").unlink(missing_ok=True)
        lp.unlink(missing_ok=True)
    return 0


def _cmd_release_all(ctx: CliContext) -> int:
    """Handle 'release-all' command: release all locks for agent."""
    cleanup_agent_locks(ctx.agent_id)
    return 0


COMMANDS: dict[str, tuple[Callable[[CliContext], int], bool, bool]] = {
    # command: (handler, requires_filepath, requires_agent_id)
    "try": (_cmd_try, True, True),
    "wait": (_cmd_wait, True, True),
    "check": (_cmd_check, True, True),
    "holder": (_cmd_holder, True, False),
    "release": (_cmd_release, True, True),
    "release-all": (_cmd_release_all, False, True),
}


def _cli_main() -> int:
    """CLI entry point for shell script delegation.

    Usage:
        python -m src.infra.tools.locking try <filepath>
        python -m src.infra.tools.locking wait <filepath> [timeout_seconds] [poll_interval_ms]
        python -m src.infra.tools.locking check <filepath>
        python -m src.infra.tools.locking holder <filepath>
        python -m src.infra.tools.locking release <filepath>
        python -m src.infra.tools.locking release-all

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
        print(
            "Usage: python -m src.infra.tools.locking <command> [args...]",
            file=sys.stderr,
        )
        print(
            "Commands: try, wait, check, holder, release, release-all", file=sys.stderr
        )
        return 2

    command = sys.argv[1]

    # Validate command exists
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print(
            "Commands: try, wait, check, holder, release, release-all", file=sys.stderr
        )
        return 2

    handler, requires_filepath, requires_agent_id = COMMANDS[command]

    # Parse environment
    lock_dir = os.environ.get("LOCK_DIR")
    agent_id = os.environ.get("AGENT_ID")
    repo_namespace = os.environ.get("REPO_NAMESPACE") or None

    # Validate LOCK_DIR (required for all commands)
    if not lock_dir:
        print("Error: LOCK_DIR must be set", file=sys.stderr)
        return 2

    # Validate AGENT_ID (required for most commands)
    if requires_agent_id and not agent_id:
        print("Error: AGENT_ID must be set", file=sys.stderr)
        return 2

    # Parse filepath argument
    filepath: str | None = None
    if requires_filepath:
        if len(sys.argv) < 3:
            print(
                f"Usage: python -m src.infra.tools.locking {command} <filepath>",
                file=sys.stderr,
            )
            return 2
        filepath = sys.argv[2]
        # Enforce exact arg count for commands without optional arguments
        if command != "wait" and len(sys.argv) > 3:
            print(
                f"Usage: python -m src.infra.tools.locking {command} <filepath>",
                file=sys.stderr,
            )
            return 2
    elif command == "release-all" and len(sys.argv) != 2:
        print("Usage: python -m src.infra.tools.locking release-all", file=sys.stderr)
        return 2

    # Parse wait-specific arguments
    timeout = 30.0
    poll_ms = 100
    if command == "wait":
        timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
        poll_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 100

    # Set MALA_LOCK_DIR so our functions use the shell script's LOCK_DIR
    os.environ["MALA_LOCK_DIR"] = lock_dir

    # Build context and dispatch
    ctx = CliContext(
        command=command,
        lock_dir=lock_dir,
        agent_id=agent_id or "",
        repo_namespace=repo_namespace,
        filepath=filepath,
        timeout=timeout,
        poll_ms=poll_ms,
    )
    return handler(ctx)


if __name__ == "__main__":
    sys.exit(_cli_main())
