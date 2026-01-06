"""File read caching hooks for reducing redundant file reads.

Contains the FileReadCache class and hook factory for blocking redundant
file reads when the file hasn't changed since the last read.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .dangerous_commands import deny_pretool_use

if TYPE_CHECKING:
    from .dangerous_commands import PreToolUseHook

# Tools that write to files and require lock ownership
FILE_WRITE_TOOLS: frozenset[str] = frozenset(
    [
        "Write",  # Claude Code Write tool: file_path
        "Edit",  # Claude Code Edit tool: file_path
        "NotebookEdit",  # Claude Code NotebookEdit: notebook_path
    ]
)

# Map of tool name to the key in tool_input that contains the file path
FILE_PATH_KEYS: dict[str, str] = {
    "Write": "file_path",
    "Edit": "file_path",
    "NotebookEdit": "notebook_path",
}


@dataclass
class CachedFileInfo:
    """Cached information about a previously-read file.

    Attributes:
        mtime_ns: File modification time in nanoseconds at time of read.
        size: File size in bytes at time of read.
        content_hash: SHA-256 hash of the file content. None if hash computation
            was deferred (when mtime/size change detected file modification).
        read_count: Number of times this file was read.
    """

    mtime_ns: int
    size: int
    content_hash: str | None
    read_count: int = 1


class FileReadCache:
    """Cache for tracking file reads and detecting redundant re-reads.

    This cache tracks files that have been read during an agent session.
    When a file is re-read without modification, the cache blocks the read
    and informs the agent that the file hasn't changed, saving tokens.

    The cache uses file mtime and size as fast change detection, falling back
    to content hash comparison only when mtime/size match.

    Attributes:
        _cache: Mapping of absolute file paths to cached file info.
        _blocked_count: Count of reads that were blocked due to cache hits.
    """

    def __init__(self) -> None:
        """Initialize an empty file read cache."""
        # Cache key is (resolved_path, offset, limit) to support range-specific caching
        self._cache: dict[tuple[str, int | None, int | None], CachedFileInfo] = {}
        self._blocked_count: int = 0

    def check_and_update(
        self,
        file_path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> tuple[bool, str]:
        """Check if a file read is redundant and update the cache.

        Args:
            file_path: Path to the file being read.
            offset: Line offset for partial reads. If provided with different
                value than cached read, allows the read.
            limit: Line limit for partial reads. If provided with different
                value than cached read, allows the read.

        Returns:
            Tuple of (is_redundant, message). If is_redundant is True,
            the message explains why the read is blocked.
        """
        try:
            path = Path(file_path).resolve()
            if not path.is_file():
                # File doesn't exist or is not a file, allow the read
                return (False, "")

            stat = path.stat()
            mtime_ns = stat.st_mtime_ns
            size = stat.st_size

            # Create cache key that includes offset/limit for range-specific caching
            # Use (None, None) as default to represent full file reads
            cache_key = (str(path), offset, limit)

            # Check if we have a cached entry for this exact file + range
            cached = self._cache.get(cache_key)
            if cached is None:
                # First read of this file/range combination - cache it
                content_hash = self._compute_hash(path)
                self._cache[cache_key] = CachedFileInfo(
                    mtime_ns=mtime_ns,
                    size=size,
                    content_hash=content_hash,
                    read_count=1,
                )
                return (False, "")

            # Check if file has changed based on mtime/size
            if mtime_ns != cached.mtime_ns or size != cached.size:
                # File modified - update cache with new mtime/size, defer hash
                # computation since we already know the file changed
                self._cache[cache_key] = CachedFileInfo(
                    mtime_ns=mtime_ns,
                    size=size,
                    content_hash=None,  # Defer hash computation
                    read_count=cached.read_count + 1,
                )
                return (False, "")

            # mtime/size match - verify with content hash
            # Always recompute hash to detect rare cases where content changes
            # without affecting mtime/size (e.g., coarse timestamp resolution)
            content_hash = self._compute_hash(path)
            if cached.content_hash is None or content_hash != cached.content_hash:
                # Content changed despite same mtime/size (rare but possible)
                # Or no cached hash yet - update cache
                self._cache[cache_key] = CachedFileInfo(
                    mtime_ns=mtime_ns,
                    size=size,
                    content_hash=content_hash,
                    read_count=cached.read_count + 1,
                )
                return (False, "")

            # Hash matches - file truly unchanged, block the redundant read
            cached.read_count += 1
            self._blocked_count += 1
            return (
                True,
                f"File unchanged since last read (read {cached.read_count}x). "
                "Content already in context - use what you have.",
            )

        except OSError:
            # File access error - allow the read (tool will report the error)
            return (False, "")

    def _compute_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of file content.

        Args:
            path: Path to the file.

        Returns:
            Hex-encoded SHA-256 hash of the file content.
        """
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            # Read in 64KB chunks for memory efficiency with large files
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def invalidate(self, file_path: str) -> None:
        """Invalidate all cache entries for a file (all offset/limit combinations).

        Call this when a file is modified (e.g., after a Write or edit).

        Args:
            file_path: Path to the file to invalidate.
        """
        try:
            path = str(Path(file_path).resolve())
            # Remove all entries for this file path (any offset/limit)
            keys_to_remove = [key for key in self._cache if key[0] == path]
            for key in keys_to_remove:
                del self._cache[key]
        except OSError:
            pass

    @property
    def blocked_count(self) -> int:
        """Return the number of reads blocked due to cache hits."""
        return self._blocked_count

    @property
    def cache_size(self) -> int:
        """Return the number of files currently cached."""
        return len(self._cache)


def make_file_read_cache_hook(cache: FileReadCache) -> PreToolUseHook:
    """Create a PreToolUse hook that blocks redundant file reads.

    This hook checks Read tool invocations against the cache. If the file
    hasn't changed since the last read, the hook blocks the read and
    informs the agent to use the content already in context.

    The hook also invalidates cache entries when files are written to,
    ensuring subsequent reads see the updated content.

    Args:
        cache: The FileReadCache instance to use for tracking reads.

    Returns:
        An async hook function that can be passed to ClaudeAgentOptions.hooks["PreToolUse"].
    """

    async def file_read_cache_hook(
        hook_input: Any,  # noqa: ANN401 - SDK type, avoid import
        stderr: str | None,
        context: Any,  # noqa: ANN401 - SDK type, avoid import
    ) -> dict[str, Any]:
        """PreToolUse hook to block redundant file reads."""
        tool_name = hook_input["tool_name"]
        tool_input = hook_input["tool_input"]

        # Check for Read tool
        if tool_name == "Read":
            file_path = tool_input.get("file_path")
            if file_path:
                # Extract offset/limit for range-specific caching
                offset = tool_input.get("offset")
                limit = tool_input.get("limit")
                is_redundant, message = cache.check_and_update(
                    file_path, offset=offset, limit=limit
                )
                if is_redundant:
                    return deny_pretool_use(message)

        # Invalidate cache on file writes
        if tool_name in FILE_WRITE_TOOLS:
            path_key = FILE_PATH_KEYS.get(tool_name)
            if path_key:
                file_path = tool_input.get(path_key)
                if file_path:
                    cache.invalidate(file_path)

        return {}

    return file_read_cache_hook
