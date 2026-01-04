"""Hooks for emitting lock events to the deadlock monitor.

Provides PostToolUse hook that captures ACQUIRED/RELEASED events from MCP
locking tools. WAITING events are emitted by the MCP tool handlers directly.

Note: LockEvent and LockEventType are injected via parameters to avoid importing
from src.core.models, which would violate the "Hooks isolated" contract.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .dangerous_commands import PostToolUseHook

from src.infra.tools.locking import canonicalize_path

logger = logging.getLogger(__name__)

# MCP tool names for locking operations
_MCP_LOCK_ACQUIRE = "mcp__mala-locking__lock_acquire"
_MCP_LOCK_RELEASE = "mcp__mala-locking__lock_release"


def make_lock_event_hook(
    agent_id: str,
    emit_event: Callable[[Any], Awaitable[object] | None],
    repo_namespace: str | None = None,
    *,
    lock_event_class: type[Any],
    lock_event_type_enum: type[Any],
) -> PostToolUseHook:
    """Create a PostToolUse hook that emits lock events from MCP tools.

    Args:
        agent_id: The agent ID emitting events.
        emit_event: Callback to emit lock events. Can be sync or async.
            Return value is awaited if async, but discarded.
        repo_namespace: Optional repo root for path canonicalization.
        lock_event_class: The LockEvent class to instantiate.
        lock_event_type_enum: The LockEventType enum.

    Returns:
        An async hook function for PostToolUse events.
    """
    LockEvent = lock_event_class
    LockEventType = lock_event_type_enum

    async def lock_event_hook(
        hook_input: Any,  # noqa: ANN401 - SDK type, avoid import
        stderr: str | None,
        context: Any,  # noqa: ANN401 - SDK type, avoid import
    ) -> dict[str, Any]:
        """PostToolUse hook to capture lock events from MCP tools."""
        tool_name = hook_input.get("tool_name", "")

        # Only process MCP locking tool calls
        if tool_name not in (_MCP_LOCK_ACQUIRE, _MCP_LOCK_RELEASE):
            return {}

        # Extract JSON from tool response (tool_response may be a list or dict)
        tool_response = hook_input.get("tool_response", {})
        if isinstance(tool_response, list):
            content_list = tool_response
        elif isinstance(tool_response, dict):
            content_list = tool_response.get("content", [])
        else:
            content_list = []
        if not content_list:
            return {}

        # Get text content from first content block
        first_content = content_list[0]
        if first_content.get("type") != "text":
            return {}

        text = first_content.get("text", "")
        if not text:
            return {}

        # Parse JSON response
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse MCP tool response as JSON: %s", text[:100])
            return {}

        # Check for error responses
        if "error" in data:
            logger.debug("MCP tool returned error: %s", data["error"])
            return {}

        # Process based on tool type
        if tool_name == _MCP_LOCK_ACQUIRE:
            await _process_acquire_response(
                data, agent_id, repo_namespace, emit_event, LockEvent, LockEventType
            )
        elif tool_name == _MCP_LOCK_RELEASE:
            await _process_release_response(
                data, agent_id, repo_namespace, emit_event, LockEvent, LockEventType
            )

        return {}

    return lock_event_hook


async def _process_acquire_response(
    data: dict[str, Any],
    agent_id: str,
    repo_namespace: str | None,
    emit_event: Callable[[Any], Awaitable[object] | None],
    LockEvent: type[Any],
    LockEventType: type[Any],
) -> None:
    """Process lock_acquire tool response and emit ACQUIRED events.

    Args:
        data: Parsed JSON response from lock_acquire tool.
        agent_id: The agent ID emitting events.
        repo_namespace: Optional repo root for path canonicalization.
        emit_event: Callback to emit lock events.
        LockEvent: The LockEvent class.
        LockEventType: The LockEventType enum.
    """
    results = data.get("results", [])

    for result in results:
        if not result.get("acquired", False):
            # Skip non-acquired files (WAITING already emitted by tool handler)
            continue

        filepath = result.get("filepath", "")
        if not filepath:
            continue

        # Canonicalize the path
        try:
            lock_path = canonicalize_path(filepath, repo_namespace)
        except Exception:
            logger.warning("Failed to canonicalize lock path: %s", filepath)
            continue

        # Emit ACQUIRED event
        event = LockEvent(
            event_type=LockEventType.ACQUIRED,
            agent_id=agent_id,
            lock_path=lock_path,
            timestamp=time.time(),
        )

        result_coro = emit_event(event)
        if result_coro is not None:
            await result_coro

        logger.debug(
            "Lock event emitted: type=ACQUIRED agent_id=%s lock_path=%s",
            agent_id,
            lock_path,
        )


async def _process_release_response(
    data: dict[str, Any],
    agent_id: str,
    repo_namespace: str | None,
    emit_event: Callable[[Any], Awaitable[object] | None],
    LockEvent: type[Any],
    LockEventType: type[Any],
) -> None:
    """Process lock_release tool response and emit RELEASED events.

    Args:
        data: Parsed JSON response from lock_release tool.
        agent_id: The agent ID emitting events.
        repo_namespace: Optional repo root for path canonicalization.
        emit_event: Callback to emit lock events.
        LockEvent: The LockEvent class.
        LockEventType: The LockEventType enum.
    """
    released = data.get("released", [])

    for filepath in released:
        if not filepath:
            continue

        # Canonicalize the path
        try:
            lock_path = canonicalize_path(filepath, repo_namespace)
        except Exception:
            logger.warning("Failed to canonicalize lock path: %s", filepath)
            continue

        # Emit RELEASED event
        event = LockEvent(
            event_type=LockEventType.RELEASED,
            agent_id=agent_id,
            lock_path=lock_path,
            timestamp=time.time(),
        )

        result_coro = emit_event(event)
        if result_coro is not None:
            await result_coro

        logger.debug(
            "Lock event emitted: type=RELEASED agent_id=%s lock_path=%s",
            agent_id,
            lock_path,
        )
