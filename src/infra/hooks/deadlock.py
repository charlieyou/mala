"""PostToolUse hook for emitting lock events to the deadlock monitor.

Captures lock command outcomes (lock-try.sh, lock-wait.sh, lock-release.sh)
and emits LockEvents for deadlock detection.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from claude_agent_sdk.types import (
        HookContext,
        PostToolUseHookInput,
        SyncHookJSONOutput,
    )

    from .dangerous_commands import PostToolUseHook

from src.domain.deadlock import LockEvent, LockEventType
from src.infra.tools.locking import canonicalize_path

logger = logging.getLogger(__name__)

# Patterns for lock commands
# Capture quoted paths (double or single) OR unquoted paths (stop at shell operators)
# Unquoted: [^\s;&|]+ matches non-whitespace excluding shell operators ; & |
_PATH_PATTERN = r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^\s;&|]+'
_LOCK_TRY_PATTERN = re.compile(rf"lock-try\.sh\s+({_PATH_PATTERN})")
_LOCK_WAIT_PATTERN = re.compile(rf"lock-wait\.sh\s+({_PATH_PATTERN})")
_LOCK_RELEASE_PATTERN = re.compile(rf"lock-release\.sh\s+({_PATH_PATTERN})")


def _strip_quotes(path: str) -> str:
    """Remove surrounding shell quotes from a path."""
    path = path.strip()
    if len(path) >= 2:
        if (path.startswith('"') and path.endswith('"')) or (
            path.startswith("'") and path.endswith("'")
        ):
            return path[1:-1]
    return path


def _extract_all_lock_paths(command: str) -> list[tuple[str, str]]:
    """Extract all lock commands from a bash command string.

    Args:
        command: The bash command string (may contain multiple commands).

    Returns:
        List of (command_type, file_path) tuples for each lock command found.
        command_type is one of "try", "wait", "release".
    """
    results: list[tuple[str, str]] = []

    for match in _LOCK_TRY_PATTERN.finditer(command):
        results.append(("try", _strip_quotes(match.group(1))))
    for match in _LOCK_WAIT_PATTERN.finditer(command):
        results.append(("wait", _strip_quotes(match.group(1))))
    for match in _LOCK_RELEASE_PATTERN.finditer(command):
        results.append(("release", _strip_quotes(match.group(1))))

    return results


def _extract_lock_path(command: str) -> tuple[str, str] | None:
    """Extract first lock command type and file path from a bash command.

    Args:
        command: The bash command string.

    Returns:
        Tuple of (command_type, file_path) if a lock command is found,
        None otherwise. command_type is one of "try", "wait", "release".
    """
    results = _extract_all_lock_paths(command)
    return results[0] if results else None


def _get_exit_code(tool_result: str) -> int | None:
    """Extract exit code from tool result.

    The SDK returns exit code in the tool result when a bash command completes.
    This function parses it from various possible formats.

    Args:
        tool_result: The result string from the bash tool.

    Returns:
        The exit code as an integer, or None if not found.
    """
    # Check for explicit exit code patterns in tool result
    # Common format: "exit code: N" or "(exit N)" or just the exit code
    if "exit code: " in tool_result.lower():
        match = re.search(r"exit code:\s*(\d+)", tool_result, re.IGNORECASE)
        if match:
            return int(match.group(1))

    # For successful commands, tool_result often doesn't include exit code
    # We need to check the exit_code field from the hook input directly
    # This function may need adjustment based on actual SDK behavior
    return None


def make_lock_event_hook(
    agent_id: str,
    emit_event: Callable[[LockEvent], Awaitable[None] | None],
    repo_namespace: str | None = None,
) -> PostToolUseHook:
    """Create a PostToolUse hook that emits lock events.

    Args:
        agent_id: The agent ID emitting events.
        emit_event: Callback to emit lock events. Can be sync or async.
        repo_namespace: Optional repo root for path canonicalization.

    Returns:
        An async hook function for PostToolUse events.
    """

    async def lock_event_hook(
        hook_input: PostToolUseHookInput,
        stderr: str | None,
        context: HookContext,
    ) -> SyncHookJSONOutput:
        """PostToolUse hook to capture lock command outcomes."""
        tool_name = hook_input["tool_name"]

        # Only process bash tool calls
        if tool_name not in ("Bash", "bash"):
            return {}

        # Get the command from tool input
        tool_input = hook_input.get("tool_input", {})
        command = tool_input.get("command", "")
        if not command:
            return {}

        # Extract all lock commands from the bash call
        lock_infos = _extract_all_lock_paths(command)
        if not lock_infos:
            return {}

        # Get exit code from tool result
        tool_result = hook_input.get("tool_result", "")
        exit_code = hook_input.get("exit_code")

        # If exit_code not in hook_input, try parsing from result
        if exit_code is None:
            exit_code = _get_exit_code(str(tool_result))

        # Handle error exit codes (2 = script error)
        if exit_code == 2:
            logger.warning(
                "Lock command error (exit code 2), command=%s",
                command,
            )
            return {}

        # Process each lock command found
        # For batched commands with exit 0, all succeeded
        # For single command, exit code determines event type
        is_single_command = len(lock_infos) == 1

        for cmd_type, raw_path in lock_infos:
            # Canonicalize the path
            try:
                lock_path = canonicalize_path(raw_path, repo_namespace)
            except Exception:
                logger.warning("Failed to canonicalize lock path: %s", raw_path)
                continue

            # Determine event type based on command and exit code
            event_type: LockEventType | None = None

            if cmd_type == "try":
                if exit_code == 0:
                    event_type = LockEventType.ACQUIRED
                elif exit_code == 1 and is_single_command:
                    # Only emit WAITING for single-command case
                    # (for batched, we can't tell which command had contention)
                    event_type = LockEventType.WAITING
            elif cmd_type == "wait":
                if exit_code == 0:
                    event_type = LockEventType.ACQUIRED
                # exit_code 1 means timeout - no event (agent will retry or abort)
            elif cmd_type == "release":
                if exit_code == 0:
                    event_type = LockEventType.RELEASED

            if event_type is None:
                continue

            # Create and emit the event
            event = LockEvent(
                event_type=event_type,
                agent_id=agent_id,
                lock_path=lock_path,
                timestamp=time.time(),
            )

            # Call emit_event (may be sync or async)
            result = emit_event(event)
            if result is not None:
                # It's a coroutine, await it
                await result

        return {}

    return lock_event_hook
