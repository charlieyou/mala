"""
Hook logic for Claude Agent SDK.

Contains PreToolUse hooks and related constants for blocking dangerous commands
and managing tool restrictions.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_agent_sdk.types import (
        HookContext,
        PreToolUseHookInput,
        StopHookInput,
        SyncHookJSONOutput,
    )

from .tools.env import SCRIPTS_DIR, get_lock_dir
from .tools.locking import get_lock_holder

# Type alias for PreToolUse hooks (using string annotations to avoid import)
PreToolUseHook = Callable[
    ["PreToolUseHookInput", str | None, "HookContext"],
    Awaitable["SyncHookJSONOutput"],
]

# Type alias for Stop hooks (using string annotations to avoid import)
StopHook = Callable[
    ["StopHookInput", str | None, "HookContext"],
    Awaitable["SyncHookJSONOutput"],
]

# Dangerous bash command patterns to block
DANGEROUS_PATTERNS = [
    "rm -rf /",
    "rm -rf ~",
    "rm -rf $HOME",
    ":(){:|:&};:",  # fork bomb
    "mkfs.",
    "dd if=",
    "> /dev/sd",
    "chmod -R 777 /",
    "curl | bash",
    "wget | bash",
    "curl | sh",
    "wget | sh",
]

# Destructive git command patterns to block in multi-agent contexts.
# These operations modify working tree or history in ways that can conflict
# between concurrent agents.
DESTRUCTIVE_GIT_PATTERNS = [
    # Hard reset - discards uncommitted changes silently
    "git reset --hard",
    # Clean - removes untracked files
    "git clean -fd",
    "git clean -f",
    "git clean -df",
    "git clean -d -f",
    # Force checkout - discards local changes
    "git checkout -- .",
    "git checkout -f",
    "git checkout --force",
    # Restore - discards uncommitted changes without confirmation
    "git restore",
    # Rebase - can rewrite history and requires conflict resolution
    "git rebase",
    # Force delete branches
    "git branch -D",
    "git branch -d -f",
    # Stash - hides changes that other agents cannot see
    "git stash",
    # Abort operations - may discard other agents' work in progress
    "git merge --abort",
    "git rebase --abort",
    "git cherry-pick --abort",
]

# Safe alternatives to blocked git operations (for error messages)
SAFE_GIT_ALTERNATIVES: dict[str, str] = {
    "git stash": "commit changes instead: git add . && git commit -m 'WIP: ...'",
    "git reset --hard": "use git checkout <file> to revert specific files, or commit first",
    "git rebase": "use git merge instead, or coordinate with other agents",
    "git checkout -f": "commit or stash changes first (in non-agent context)",
    "git checkout --force": "commit or stash changes first (in non-agent context)",
    "git restore": "commit changes first, or use git diff to review before discarding",
    "git clean -f": "manually remove specific untracked files with rm",
    "git merge --abort": "resolve merge conflicts instead of aborting",
    "git rebase --abort": "resolve rebase conflicts instead of aborting",
    "git cherry-pick --abort": "resolve cherry-pick conflicts instead of aborting",
}

# Tool names that should be treated as bash (case-insensitive matching)
BASH_TOOL_NAMES = frozenset(["bash"])

# Tools replaced by MorphLLM MCP (use disallowed_tools parameter, not hooks)
MORPH_DISALLOWED_TOOLS = ["Edit", "Grep"]

# Tools that write to files and require lock ownership
FILE_WRITE_TOOLS: frozenset[str] = frozenset(
    [
        "Write",  # Claude Code Write tool: file_path
        "NotebookEdit",  # Claude Code NotebookEdit: notebook_path
        "mcp__morphllm__edit_file",  # MorphLLM MCP: path
    ]
)

# Map of tool name to the key in tool_input that contains the file path
FILE_PATH_KEYS: dict[str, str] = {
    "Write": "file_path",
    "NotebookEdit": "notebook_path",
    "mcp__morphllm__edit_file": "path",
}


async def block_dangerous_commands(
    hook_input: PreToolUseHookInput,
    stderr: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    """PreToolUse hook to block dangerous bash commands.

    In multi-agent contexts, certain git operations are blocked because they
    can cause conflicts between concurrent agents. Blocked operations include:
    - git stash (all subcommands) - hides changes other agents cannot see
    - git reset --hard - discards uncommitted changes silently
    - git rebase - requires human input and can rewrite history
    - git checkout -f/--force - discards local changes
    - git clean -f - removes untracked files without warning
    - git merge/rebase/cherry-pick --abort - may discard other agents' work

    When a blocked operation is detected, the error message includes a safe
    alternative when available.
    """
    tool_name = hook_input["tool_name"].lower()
    if tool_name not in BASH_TOOL_NAMES:
        return {}  # Allow non-Bash tools

    command = hook_input["tool_input"].get("command", "")

    # Block dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if pattern in command:
            return {
                "decision": "block",
                "reason": f"Blocked dangerous command pattern: {pattern}",
            }

    # Block destructive git patterns with safe alternatives
    for pattern in DESTRUCTIVE_GIT_PATTERNS:
        if pattern in command:
            alternative = SAFE_GIT_ALTERNATIVES.get(pattern, "")
            reason = f"Blocked destructive git command: {pattern}"
            if alternative:
                reason = f"{reason}. Safe alternative: {alternative}"
            return {
                "decision": "block",
                "reason": reason,
            }

    # Block force push to ALL branches (--force-with-lease is allowed as safer alternative)
    if "git push" in command:
        # Allow --force-with-lease (safer alternative)
        if "--force-with-lease" in command:
            pass  # Allow
        elif "--force" in command or "-f " in command:
            return {
                "decision": "block",
                "reason": "Blocked force push (use --force-with-lease if needed)",
            }

    return {}  # Allow the command


async def block_morph_replaced_tools(
    hook_input: PreToolUseHookInput,
    stderr: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    """PreToolUse hook to block tools replaced by MorphLLM MCP.

    Note: We use a hook instead of the SDK's `disallowed_tools` parameter because
    it has a known bug where it's sometimes ignored.
    See: https://github.com/anthropics/claude-agent-sdk-python/issues/361
    """
    tool_name = hook_input["tool_name"]
    if tool_name in MORPH_DISALLOWED_TOOLS:
        return {
            "decision": "block",
            "reason": f"Use MorphLLM MCP tool instead of {tool_name}. "
            "Available: edit_file, warpgrep_codebase_search",
        }
    return {}


def make_lock_enforcement_hook(
    agent_id: str, repo_path: str | None = None
) -> PreToolUseHook:
    """Create a PreToolUse hook that enforces lock ownership for file writes.

    Args:
        agent_id: The agent ID to check lock ownership against.
        repo_path: The repository root path, used as REPO_NAMESPACE for lock
            key computation. Must match the REPO_NAMESPACE environment variable
            set for the agent's shell scripts.

    Returns:
        An async hook function that blocks file writes unless the agent holds the lock.
    """

    async def enforce_lock_ownership(
        hook_input: PreToolUseHookInput,
        stderr: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        """PreToolUse hook to block file writes unless this agent holds the lock."""
        tool_name = hook_input["tool_name"]

        # Only check file-write tools
        if tool_name not in FILE_WRITE_TOOLS:
            return {}

        # Get the file path from the tool input
        path_key = FILE_PATH_KEYS.get(tool_name)
        if not path_key:
            return {}

        file_path = hook_input["tool_input"].get(path_key)
        if not file_path:
            # No path provided, can't check lock - allow (tool will fail anyway)
            return {}

        # Check if this agent holds the lock
        # Pass repo_path as repo_namespace to match shell script key computation
        lock_holder = get_lock_holder(file_path, repo_namespace=repo_path)

        if lock_holder is None:
            return {
                "decision": "block",
                "reason": f"File {file_path} is not locked. Acquire lock with: lock-try.sh {file_path}",
            }

        if lock_holder != agent_id:
            return {
                "decision": "block",
                "reason": f"File {file_path} is locked by {lock_holder}. Wait or coordinate to acquire the lock.",
            }

        # Agent holds the lock, allow the write
        return {}

    return enforce_lock_ownership


def make_stop_hook(agent_id: str) -> StopHook:
    """Create a Stop hook that cleans up locks for the given agent.

    Args:
        agent_id: The agent ID to clean up locks for when the agent stops.

    Returns:
        An async hook function suitable for use with ClaudeAgentOptions.hooks["Stop"].
    """

    async def cleanup_locks_on_stop(
        hook_input: StopHookInput,
        stderr: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        """Stop hook to release all locks held by this agent."""
        script = SCRIPTS_DIR / "lock-release-all.sh"
        try:
            subprocess.run(
                [str(script)],
                env={
                    **os.environ,
                    "LOCK_DIR": str(get_lock_dir()),
                    "AGENT_ID": agent_id,
                },
                check=False,
                capture_output=True,
            )
        except Exception:
            pass  # Best effort cleanup, orchestrator has fallback
        return {}

    return cleanup_locks_on_stop


@dataclass
class CachedFileInfo:
    """Cached information about a previously-read file.

    Attributes:
        mtime_ns: File modification time in nanoseconds at time of read.
        size: File size in bytes at time of read.
        content_hash: SHA-256 hash of the file content.
        read_count: Number of times this file was read.
    """

    mtime_ns: int
    size: int
    content_hash: str
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
        self._cache: dict[str, CachedFileInfo] = {}
        self._blocked_count: int = 0

    def check_and_update(self, file_path: str) -> tuple[bool, str]:
        """Check if a file read is redundant and update the cache.

        Args:
            file_path: Path to the file being read.

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

            # Check if we have a cached entry for this file
            cached = self._cache.get(str(path))
            if cached is None:
                # First read - cache it
                content_hash = self._compute_hash(path)
                self._cache[str(path)] = CachedFileInfo(
                    mtime_ns=mtime_ns,
                    size=size,
                    content_hash=content_hash,
                    read_count=1,
                )
                return (False, "")

            # Check if file has changed
            if mtime_ns != cached.mtime_ns or size != cached.size:
                # File modified - update cache and allow read
                content_hash = self._compute_hash(path)
                self._cache[str(path)] = CachedFileInfo(
                    mtime_ns=mtime_ns,
                    size=size,
                    content_hash=content_hash,
                    read_count=cached.read_count + 1,
                )
                return (False, "")

            # mtime/size match - verify with content hash
            content_hash = self._compute_hash(path)
            if content_hash != cached.content_hash:
                # Content changed despite same mtime/size (rare but possible)
                self._cache[str(path)] = CachedFileInfo(
                    mtime_ns=mtime_ns,
                    size=size,
                    content_hash=content_hash,
                    read_count=cached.read_count + 1,
                )
                return (False, "")

            # File unchanged - block the redundant read
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
        """Invalidate the cache entry for a file.

        Call this when a file is modified (e.g., after a Write or edit).

        Args:
            file_path: Path to the file to invalidate.
        """
        try:
            path = str(Path(file_path).resolve())
            self._cache.pop(path, None)
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
        hook_input: PreToolUseHookInput,
        stderr: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        """PreToolUse hook to block redundant file reads."""
        tool_name = hook_input["tool_name"]
        tool_input = hook_input["tool_input"]

        # Check for Read tool
        if tool_name == "Read":
            file_path = tool_input.get("file_path")
            if file_path:
                is_redundant, message = cache.check_and_update(file_path)
                if is_redundant:
                    return {
                        "decision": "block",
                        "reason": message,
                    }

        # Invalidate cache on file writes
        if tool_name in FILE_WRITE_TOOLS:
            path_key = FILE_PATH_KEYS.get(tool_name)
            if path_key:
                file_path = tool_input.get(path_key)
                if file_path:
                    cache.invalidate(file_path)

        return {}

    return file_read_cache_hook
