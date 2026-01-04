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


def _is_safe_batch_command(command: str) -> bool:
    """Check if a batched command is safe for emitting events for all matches.

    A command is "safe" if it only uses && to chain commands, meaning
    success (exit 0) implies all commands ran successfully. Commands using
    ;, ||, |, or & operators are unsafe because:
    - ; runs commands regardless of previous exit codes
    - || short-circuits on success
    - | creates pipelines where exit code reflects last command
    - & runs commands in background

    Args:
        command: The bash command string.

    Returns:
        True if the command is safe for batch emission, False otherwise.
    """
    # Check for unsafe operators outside of quoted strings
    # Simple heuristic: if any of ; || | & appear outside quotes, it's unsafe
    # Note: && is safe, so we need to distinguish || from &&
    in_single_quote = False
    in_double_quote = False
    i = 0
    while i < len(command):
        char = command[i]

        # Handle quote state changes
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif not in_single_quote and not in_double_quote:
            # Check for unsafe operators
            if char == ";":
                return False
            if char == "|":
                # Check if it's || (unsafe) or just | (unsafe)
                # Note: && is handled separately - it's safe
                if i + 1 < len(command) and command[i + 1] == "|":
                    return False  # ||
                return False  # single |
            if char == "&":
                # Check if it's && (safe), redirection (safe), or background (unsafe)
                if i + 1 < len(command) and command[i + 1] == "&":
                    i += 1  # Skip the second &, && is safe
                elif i > 0 and command[i - 1] in ">0123456789":
                    # Part of redirection: 2>&1, >&, &> patterns
                    # The & here is part of file descriptor redirection, not backgrounding
                    pass
                elif i + 1 < len(command) and command[i + 1] == ">":
                    # &> redirection (stdout+stderr to file)
                    pass
                else:
                    return False  # single & (background)

        i += 1

    return True


def _extract_all_lock_paths(command: str) -> list[tuple[str, str]]:
    """Extract all lock commands from a bash command string.

    Args:
        command: The bash command string (may contain multiple commands).

    Returns:
        List of (command_type, file_path) tuples for each lock command found,
        sorted by position in the command string (preserving execution order).
        command_type is one of "try", "wait", "release".
    """
    # Collect (position, command_type, file_path) tuples
    matches: list[tuple[int, str, str]] = []

    for match in _LOCK_TRY_PATTERN.finditer(command):
        matches.append((match.start(), "try", _strip_quotes(match.group(1))))
    for match in _LOCK_WAIT_PATTERN.finditer(command):
        matches.append((match.start(), "wait", _strip_quotes(match.group(1))))
    for match in _LOCK_RELEASE_PATTERN.finditer(command):
        matches.append((match.start(), "release", _strip_quotes(match.group(1))))

    # Sort by position to preserve execution order
    matches.sort(key=lambda x: x[0])

    return [(cmd_type, path) for _, cmd_type, path in matches]


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
    # Guard uses "exit code:" (no space after colon) to match regex which uses \s*
    if "exit code:" in tool_result.lower():
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

        # Process lock commands found
        # For batched commands, we can only safely emit events for all if:
        # 1. Single command, or
        # 2. Commands are chained with && only (safe batch)
        # For unsafe batches (;, ||, |, &), only emit for the last command
        is_single_command = len(lock_infos) == 1
        is_safe_batch = is_single_command or _is_safe_batch_command(command)

        # If unsafe batch, only process the last command (whose exit code we have)
        commands_to_process = lock_infos if is_safe_batch else lock_infos[-1:]

        for cmd_type, raw_path in commands_to_process:
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
