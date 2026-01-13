"""Disk-persisted lint cache for SpecValidationRunner.

This module provides LintCache which tracks the last successful run for each
lint command and skips the command if no files have changed since. The cache
is persisted to disk (lint_cache.json) so it survives across validation runs.

The cache is based on:
1. The current git HEAD commit SHA
2. Whether there are uncommitted changes (git status)
3. Hash of uncommitted changes (if any)

If all match the cached state, the lint command can be skipped.

Note:
    This module is distinct from the in-memory LintCache in src/hooks.py,
    which is designed for Claude agent hooks (make_lint_cache_hook). The two
    have different APIs and persistence models suited to their use cases.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols.infra import CommandRunnerPort


def _run_git_command(
    args: list[str], cwd: Path, command_runner: CommandRunnerPort
) -> str | None:
    """Run a git command and return stdout, or None on failure.

    This is a separate function to make it easy to mock in tests without
    affecting other subprocess usage.

    Args:
        args: Git command arguments (without 'git' prefix).
        cwd: Working directory.
        command_runner: Command runner for executing the git command.

    Returns:
        stdout as a string, or None if the command failed.
    """
    result = command_runner.run(["git", *args], cwd=cwd, timeout=5.0)
    if result.ok:
        return result.stdout.strip()
    return None


@dataclass(frozen=True)
class LintCacheKey:
    """Key identifying a lint command in the cache.

    Attributes:
        command_name: Name of the command (e.g., "ruff check", "ty check").
        working_dir: Path where the command runs (normalized to string).
    """

    command_name: str
    working_dir: str

    def to_dict(self) -> dict[str, str]:
        """Convert to a JSON-serializable dict."""
        return {"command_name": self.command_name, "working_dir": self.working_dir}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> LintCacheKey:
        """Create from a dict."""
        return cls(
            command_name=data["command_name"],
            working_dir=data["working_dir"],
        )


@dataclass(frozen=True)
class LintCacheEntry:
    """Cache entry for a lint command.

    Attributes:
        head_sha: Git HEAD commit SHA when the command was run.
        has_uncommitted: Whether there were uncommitted changes.
        files_hash: Hash of uncommitted file contents (None if clean).
    """

    head_sha: str
    has_uncommitted: bool
    files_hash: str | None

    def to_dict(self) -> dict[str, str | bool | None]:
        """Convert to a JSON-serializable dict."""
        return {
            "head_sha": self.head_sha,
            "has_uncommitted": self.has_uncommitted,
            "files_hash": self.files_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | bool | None]) -> LintCacheEntry:
        """Create from a dict."""
        return cls(
            head_sha=str(data["head_sha"]),
            has_uncommitted=bool(data["has_uncommitted"]),
            files_hash=data.get("files_hash"),  # type: ignore[arg-type]
        )


class LintCache:
    """Disk-persisted cache for SpecValidationRunner to skip redundant lint runs.

    The cache stores the git state when each lint command last passed:
    - HEAD commit SHA
    - Whether there were uncommitted changes
    - Hash of uncommitted changes (if any)

    If the current state matches the cached state, the lint command can be
    skipped since it would produce the same result.

    Note:
        This is a DISK-PERSISTED cache designed for batch validation runs
        in SpecValidationRunner. For an in-memory cache used with Claude
        agent hooks, see src/hooks.py LintCache which has a different API
        (check_and_update with two-phase pending/confirmed) suited for
        make_lint_cache_hook.

    Example:
        cache = LintCache(Path("/tmp/lint-cache"), repo_path)
        if cache.should_skip("ruff check"):
            print("Skipping ruff check - no changes since last run")
        else:
            # run ruff check...
            if success:
                cache.mark_passed("ruff check")
    """

    def __init__(
        self,
        cache_dir: Path,
        repo_path: Path,
        command_runner: CommandRunnerPort,
        git_cwd: Path | None = None,
    ) -> None:
        """Initialize the lint cache.

        Args:
            cache_dir: Directory to store the cache file.
            repo_path: Path to the git repository (used for cache key stability).
            command_runner: Command runner for executing git commands.
            git_cwd: Working directory for git commands. If None, uses repo_path.
                This allows using a stable repo_path for cache keys while running
                git commands in a per-run worktree.
        """
        self.cache_dir = cache_dir
        self.repo_path = repo_path
        self._command_runner = command_runner
        self._git_cwd = git_cwd if git_cwd is not None else repo_path
        self._cache_file = cache_dir / "lint_cache.json"
        self._entries: dict[str, dict[str, str | bool | None]] = {}
        self._load()

    def _load(self) -> None:
        """Load the cache from disk."""
        if self._cache_file.exists():
            try:
                self._entries = json.loads(self._cache_file.read_text())
            except (json.JSONDecodeError, OSError):
                self._entries = {}

    def _save(self) -> None:
        """Save the cache to disk with immediate flush.

        Uses explicit flush() and fsync() to ensure data is written to disk
        before returning. This prevents data loss if mala is interrupted.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self._cache_file, "w") as f:
            json.dump(self._entries, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def _get_key_str(self, command_name: str) -> str:
        """Get the cache key string for a command."""
        key = LintCacheKey(
            command_name=command_name,
            working_dir=str(self.repo_path),
        )
        return json.dumps(key.to_dict(), sort_keys=True)

    def _get_current_state(self) -> LintCacheEntry | None:
        """Get the current git state.

        Returns:
            LintCacheEntry with current HEAD SHA and uncommitted status,
            or None if git commands fail (not a git repo or mocked environment).
        """
        # Get HEAD SHA
        try:
            head_sha = _run_git_command(
                ["rev-parse", "HEAD"], self._git_cwd, self._command_runner
            )
            if head_sha is None:
                # Not a git repo, no commits, or subprocess mocked - can't cache
                return None
        except Exception:
            # Not a git repo, no commits, or subprocess mocked - can't cache
            return None

        # Check for uncommitted changes
        try:
            status_output = _run_git_command(
                ["status", "--porcelain"], self._git_cwd, self._command_runner
            )
            if status_output is None:
                # Can't determine state - don't cache
                return None
            has_uncommitted = bool(status_output)
        except Exception:
            # Can't determine state - don't cache
            return None

        # If there are uncommitted changes, hash them
        files_hash = None
        if has_uncommitted:
            files_hash = self._hash_uncommitted()

        return LintCacheEntry(
            head_sha=head_sha,
            has_uncommitted=has_uncommitted,
            files_hash=files_hash,
        )

    def _hash_uncommitted(self) -> str:
        """Hash the uncommitted changes including untracked files.

        Returns:
            SHA256 hash of the uncommitted diff plus untracked file contents.
        """
        hasher = hashlib.sha256()

        # Get staged + unstaged diff (captures tracked file changes)
        try:
            diff = _run_git_command(
                ["diff", "HEAD"], self._git_cwd, self._command_runner
            )
            if diff is None:
                diff = ""
        except Exception:
            diff = ""
        hasher.update(diff.encode())

        # Get untracked files and include their contents in the hash
        try:
            untracked_output = _run_git_command(
                ["ls-files", "--others", "--exclude-standard"],
                self._git_cwd,
                self._command_runner,
            )
            if untracked_output:
                untracked_files = untracked_output.split("\n")
                # Sort for deterministic ordering
                for filepath in sorted(untracked_files):
                    if filepath:
                        # Include the path in the hash
                        hasher.update(f"\n--- untracked: {filepath}\n".encode())
                        # Include the file content using chunked reading
                        # to avoid OOM on large files
                        full_path = self._git_cwd / filepath
                        try:
                            with open(full_path, "rb") as f:
                                for chunk in iter(lambda: f.read(65536), b""):
                                    hasher.update(chunk)
                        except OSError:
                            # File may have been deleted or be unreadable
                            hasher.update(b"<unreadable>")
        except Exception:
            pass

        return hasher.hexdigest()[:16]

    def should_skip(self, command_name: str) -> bool:
        """Check if a lint command can be skipped.

        Args:
            command_name: Name of the command (e.g., "ruff check").

        Returns:
            True if the command can be skipped (no changes since last run).
        """
        key_str = self._get_key_str(command_name)
        if key_str not in self._entries:
            return False

        current = self._get_current_state()
        if current is None:
            # Can't determine state - don't skip
            return False

        cached = LintCacheEntry.from_dict(self._entries[key_str])

        # Must match on all dimensions
        return (
            cached.head_sha == current.head_sha
            and cached.has_uncommitted == current.has_uncommitted
            and cached.files_hash == current.files_hash
        )

    def mark_passed(self, command_name: str) -> None:
        """Mark a lint command as having passed.

        Args:
            command_name: Name of the command that passed.
        """
        current = self._get_current_state()
        if current is None:
            # Can't determine state - don't cache
            return

        key_str = self._get_key_str(command_name)
        self._entries[key_str] = current.to_dict()
        self._save()

    def invalidate(self, command_name: str) -> None:
        """Invalidate the cache for a command.

        Args:
            command_name: Name of the command to invalidate.
        """
        key_str = self._get_key_str(command_name)
        if key_str in self._entries:
            del self._entries[key_str]
            self._save()

    def invalidate_all(self) -> int:
        """Invalidate all cache entries.

        Unlike clear(), this removes entries but keeps the cache file,
        ensuring consistent behavior with per-command invalidate().

        Returns:
            Number of entries invalidated.
        """
        count = len(self._entries)
        if count > 0:
            self._entries = {}
            self._save()
        return count

    def clear(self) -> None:
        """Clear the entire cache."""
        self._entries = {}
        if self._cache_file.exists():
            self._cache_file.unlink()
