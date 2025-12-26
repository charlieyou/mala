"""Git utility functions for mala.

Provides helpers for getting git repository information.
"""

import asyncio
import subprocess
from pathlib import Path

# Default timeout for git commands (seconds)
DEFAULT_GIT_TIMEOUT = 5.0


def get_git_commit(cwd: Path) -> str:
    """Get the current git commit hash (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def get_git_branch(cwd: Path) -> str:
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


# --- Async versions (non-blocking) ---


async def get_git_commit_async(cwd: Path, timeout: float = DEFAULT_GIT_TIMEOUT) -> str:
    """Get the current git commit hash (short) - async version."""

    def run_blocking() -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(run_blocking),
            timeout=timeout,
        )
    except TimeoutError:
        return ""


async def get_git_branch_async(cwd: Path, timeout: float = DEFAULT_GIT_TIMEOUT) -> str:
    """Get the current git branch name - async version."""

    def run_blocking() -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(run_blocking),
            timeout=timeout,
        )
    except TimeoutError:
        return ""
