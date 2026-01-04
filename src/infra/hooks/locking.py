"""Lock enforcement hooks for multi-agent file coordination.

Contains hooks for enforcing file locks when writing and for cleanup
on agent stop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dangerous_commands import PreToolUseHook

from ..tools.command_runner import run_command
from ..tools.env import SCRIPTS_DIR, get_lock_dir
from ..tools.locking import get_lock_holder
from .file_cache import FILE_PATH_KEYS, FILE_WRITE_TOOLS

# Type alias for Stop hooks (using Any to avoid SDK import)
StopHook = Callable[
    [Any, str | None, Any],
    Awaitable[dict[str, Any]],
]


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
        hook_input: Any,  # noqa: ANN401 - SDK type, avoid import
        stderr: str | None,
        context: Any,  # noqa: ANN401 - SDK type, avoid import
    ) -> dict[str, Any]:
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
                "reason": f'File {file_path} is not locked. Use lock_acquire tool with filepaths: ["{file_path}"]',
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
        hook_input: Any,  # noqa: ANN401 - SDK type, avoid import
        stderr: str | None,
        context: Any,  # noqa: ANN401 - SDK type, avoid import
    ) -> dict[str, Any]:
        """Stop hook to release all locks held by this agent."""
        script = SCRIPTS_DIR / "lock-release-all.sh"
        try:
            run_command(
                [str(script)],
                cwd=Path.cwd(),
                env={
                    "LOCK_DIR": str(get_lock_dir()),
                    "AGENT_ID": agent_id,
                },
            )
        except Exception:
            pass  # Best effort cleanup, orchestrator has fallback
        return {}

    return cleanup_locks_on_stop
