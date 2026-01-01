"""Lint command caching hooks for reducing redundant lint runs.

Contains the LintCache class and hook factory for blocking redundant
lint commands when the git state hasn't changed since the last successful run.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_agent_sdk.types import (
        HookContext,
        PreToolUseHookInput,
        SyncHookJSONOutput,
    )

    from .dangerous_commands import PreToolUseHook

from ..tools.command_runner import run_command
from .dangerous_commands import BASH_TOOL_NAMES
from .file_cache import FILE_WRITE_TOOLS

# Lint command patterns for detecting lint/format/typecheck commands
# Using regex patterns for precise matching to avoid false positives
LINT_COMMAND_PATTERNS: dict[str, re.Pattern[str]] = {
    # Match "ruff check" but not "ruff format"
    "ruff_check": re.compile(r"\bruff\s+check\b", re.IGNORECASE),
    # Match "ruff format"
    "ruff_format": re.compile(r"\bruff\s+format\b", re.IGNORECASE),
    # Match "ty check" as a standalone command (ty must be a word boundary)
    # This avoids matching "pytest --check" or similar
    "ty_check": re.compile(r"\bty\s+check\b", re.IGNORECASE),
}


def _get_git_state(repo_path: Path | None = None) -> str | None:
    """Get a hash representing the current git state including commit SHA.

    This captures:
    - Current HEAD commit SHA
    - Staged and unstaged changes to tracked files
    - Untracked files list

    Returns None if git command fails or not in a git repo.

    Args:
        repo_path: Path to the repository. If None, uses current directory.

    Returns:
        A hash string representing the complete git state, or None on failure.
    """
    try:
        cwd = repo_path or Path.cwd()

        # Get current HEAD commit SHA
        head_result = run_command(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
        )
        if not head_result.ok:
            return None
        head_sha = head_result.stdout.strip()

        # Get hash of working tree state (staged + unstaged changes)
        diff_result = run_command(
            ["git", "diff", "HEAD"],
            cwd=cwd,
        )
        if not diff_result.ok:
            return None

        # Also include untracked files in the hash
        untracked = run_command(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=cwd,
        )

        # Combine HEAD SHA, diff, and untracked files for complete state
        combined = (
            head_sha
            + "\n"
            + diff_result.stdout
            + "\n"
            + (untracked.stdout if untracked.ok else "")
        )
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    except Exception:
        return None


def _detect_lint_command(command: str) -> str | None:
    """Detect which lint command type is being run.

    Uses regex patterns with word boundaries to avoid false positives.

    Args:
        command: The bash command string.

    Returns:
        The lint command key (e.g., "ruff_check") or None if not a lint command.
    """
    for lint_key, pattern in LINT_COMMAND_PATTERNS.items():
        if pattern.search(command):
            return lint_key
    return None


@dataclass
class LintCacheEntry:
    """Cached information about a successful lint run.

    Attributes:
        git_state: Git state hash when lint successfully completed.
        timestamp: Unix timestamp when lint passed.
        skipped_count: Number of times this lint was skipped due to cache hit.
    """

    git_state: str
    timestamp: float
    skipped_count: int = 0


class LintCache:
    """In-memory cache for tracking successful lint runs during agent sessions.

    This cache tracks the git state when lint commands pass. A lint is only
    cached after an explicit success via `mark_success()`. The cache uses
    commit SHA + working tree diff hash to detect any file changes.

    Note:
        This is an IN-MEMORY cache designed for use with Claude agent hooks
        (see make_lint_cache_hook). For a disk-persisted cache used in batch
        validation, see src/validation/lint_cache.py which has a different API
        (should_skip/mark_passed) suited for SpecValidationRunner.

    Attributes:
        _cache: Mapping of lint command type to cached entry.
        _skipped_count: Total count of lints skipped due to cache hits.
        _repo_path: Path to the repository for git operations.
    """

    def __init__(self, repo_path: Path | None = None) -> None:
        """Initialize an empty lint cache.

        Args:
            repo_path: Path to the repository. If None, uses current directory.
        """
        self._cache: dict[str, LintCacheEntry] = {}
        self._skipped_count: int = 0
        self._repo_path = repo_path

    def _make_cache_key(self, lint_type: str, command: str) -> str:
        """Create a cache key combining lint type and command.

        This ensures commands with different arguments (e.g., 'ruff check src/'
        vs 'ruff check .') are cached separately.

        Args:
            lint_type: Type of lint command (e.g., "ruff_check").
            command: Full command string.

        Returns:
            A cache key string combining lint type and command hash.
        """
        # Use SHA-256 hash of command to create a stable key
        command_hash = hashlib.sha256(command.encode()).hexdigest()[:12]
        return f"{lint_type}:{command_hash}"

    def check_and_update(self, lint_type: str, command: str = "") -> tuple[bool, str]:
        """Check if a lint run is redundant based on cached success.

        Only skips if there is a confirmed successful lint at the current git
        state (via prior `mark_success()` call).

        Args:
            lint_type: Type of lint command (e.g., "ruff_check").
            command: Full command string for cache key differentiation.

        Returns:
            Tuple of (is_redundant, message). If is_redundant is True,
            the message explains why the lint is skipped.
        """
        current_state = _get_git_state(self._repo_path)
        if current_state is None:
            # Can't determine git state, allow the lint
            return (False, "")

        # Create cache key combining lint type and command
        cache_key = self._make_cache_key(lint_type, command)

        # Check if we have a confirmed successful lint at this state
        cached = self._cache.get(cache_key)
        if cached is not None and cached.git_state == current_state:
            # State unchanged since last confirmed success - skip
            cached.skipped_count += 1
            self._skipped_count += 1
            lint_name = lint_type.replace("_", " ")
            return (
                True,
                f"No changes since last {lint_name} (skipped {cached.skipped_count}x). "
                "Git state unchanged - lint would produce same results.",
            )

        # No cached success at current state - allow lint to run
        return (False, "")

    def mark_success(self, lint_type: str, command: str = "") -> None:
        """Explicitly mark a lint as successful at current state.

        Call this after a lint command completes successfully to cache
        the result.

        Args:
            lint_type: Type of lint command that succeeded.
            command: Full command string for cache key differentiation.
        """
        current_state = _get_git_state(self._repo_path)
        if current_state is not None:
            cache_key = self._make_cache_key(lint_type, command)
            self._cache[cache_key] = LintCacheEntry(
                git_state=current_state,
                timestamp=time.time(),
                skipped_count=0,
            )

    def invalidate(self, lint_type: str | None = None) -> None:
        """Invalidate cache entries.

        Call this when files are modified to ensure lint runs again.

        Args:
            lint_type: Specific lint type to invalidate. If None, clears all.
                When provided, invalidates all commands for that lint type.
        """
        if lint_type is None:
            self._cache.clear()
        else:
            # Cache keys are in format "lint_type:command_hash"
            # Remove all entries matching the lint type prefix
            prefix = f"{lint_type}:"
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_remove:
                del self._cache[key]

    @property
    def skipped_count(self) -> int:
        """Return the total number of lints skipped due to cache hits."""
        return self._skipped_count

    @property
    def cache_size(self) -> int:
        """Return the number of lint types currently cached."""
        return len(self._cache)


def make_lint_cache_hook(
    cache: LintCache,
) -> PreToolUseHook:
    """Create a PreToolUse hook that blocks redundant lint commands.

    This hook checks Bash tool invocations for lint commands (ruff check,
    ruff format, ty check). If the working tree hasn't changed since the
    last run of that lint type, the hook blocks the command.

    The hook also invalidates cache entries when files are written to,
    ensuring subsequent lints see the updated state.

    Args:
        cache: The LintCache instance to use for tracking lint runs.

    Returns:
        An async hook function for ClaudeAgentOptions.hooks["PreToolUse"].
    """

    async def lint_cache_hook(
        hook_input: PreToolUseHookInput,
        stderr: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        """PreToolUse hook to block redundant lint commands."""
        tool_name = hook_input["tool_name"]
        tool_input = hook_input["tool_input"]

        # Check for Bash tool with lint command
        if tool_name.lower() in BASH_TOOL_NAMES:
            command = tool_input.get("command", "")
            lint_type = _detect_lint_command(command)
            if lint_type:
                # Don't block compound commands - only block simple lint commands
                # This ensures "ruff check . && pytest" runs the test portion
                if any(sep in command for sep in ["&&", "||", ";"]):
                    return {}  # Allow compound commands to run

                is_redundant, message = cache.check_and_update(lint_type, command)
                if is_redundant:
                    return {
                        "decision": "block",
                        "reason": message,
                    }

        # Invalidate cache on file writes (lint results may change)
        if tool_name in FILE_WRITE_TOOLS:
            cache.invalidate()

        return {}

    return lint_cache_hook
