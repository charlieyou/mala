"""MCP server with locking tools for multi-agent file coordination.

Provides named MCP tools for file locking operations:
- lock_acquire: Try to acquire locks, wait if blocked until ANY progress
- lock_release: Release specific files or all held locks

Tools emit WAITING events via closure-captured callback when blocked,
enabling real-time deadlock detection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from claude_agent_sdk import SdkMcpTool
    from claude_agent_sdk.types import McpSdkServerConfig

    from src.core.models import LockEvent

    # Type alias for emit_lock_event callback (sync or async)
    LockEventCallback = Callable[[LockEvent], None | Awaitable[Any]]

from .locking import (
    canonicalize_path,
    cleanup_agent_locks,
    get_lock_holder,
    release_lock,
    try_lock,
    wait_for_lock_async,
)

logger = logging.getLogger(__name__)

# Tool names for hook matching
LOCK_ACQUIRE_TOOL = "lock_acquire"
LOCK_RELEASE_TOOL = "lock_release"


class LockingToolHandlers:
    """Container for lock tool handlers, exposed for testing.

    Use create_locking_mcp_server() to get the MCP server config.
    Access .lock_acquire and .lock_release handlers for direct testing.
    """

    def __init__(
        self,
        lock_acquire: SdkMcpTool[Any],
        lock_release: SdkMcpTool[Any],
    ) -> None:
        self.lock_acquire = lock_acquire
        self.lock_release = lock_release


def create_locking_mcp_server(
    agent_id: str,
    repo_namespace: str | None,
    emit_lock_event: Callable[[LockEvent], object],
    *,
    _return_handlers: bool = False,
) -> McpSdkServerConfig | tuple[McpSdkServerConfig, LockingToolHandlers]:
    """Create MCP server config with locking tools bound to agent context.

    The emit_lock_event callback is captured by tool handler closures,
    allowing WAITING events to be emitted during lock acquisition.

    Args:
        agent_id: The agent ID for lock ownership.
        repo_namespace: Optional repo namespace for cross-repo disambiguation.
        emit_lock_event: Callback to emit lock events (captured in closures).
        _return_handlers: If True, also return handler objects for testing.

    Returns:
        MCP server configuration dict for Claude Agent SDK.
        If _return_handlers=True, returns (config, handlers) tuple.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    # Import LockEvent and LockEventType here to avoid circular imports
    from src.core.models import LockEvent, LockEventType

    def _canonical(filepath: str) -> str:
        """Canonicalize path for consistent deadlock graph nodes."""
        return canonicalize_path(filepath, repo_namespace)

    def _emit_waiting(canonical_path: str) -> None:
        """Emit WAITING event for a blocked file.

        Args:
            canonical_path: Already-canonicalized path (from _canonical()).
        """
        event = LockEvent(
            event_type=LockEventType.WAITING,
            agent_id=agent_id,
            lock_path=canonical_path,
            timestamp=time.time(),
        )
        emit_lock_event(event)
        logger.debug(
            "WAITING emitted: agent=%s path=%s",
            agent_id,
            canonical_path,
        )

    def _strip_internal_fields(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip internal fields from results before returning to caller."""
        return [{k: v for k, v in r.items() if k != "canonical"} for r in results]

    @tool(
        name=LOCK_ACQUIRE_TOOL,
        description=(
            "Try to acquire locks on multiple files. If some are blocked, "
            "waits until ANY becomes available (or timeout). Returns per-file "
            "results. Use timeout_seconds=0 for non-blocking try."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "filepaths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to lock",
                    "minItems": 1,
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Max seconds to wait (default: 30, use 0 for non-blocking)",
                },
            },
            "required": ["filepaths"],
        },
    )
    async def lock_acquire(args: dict) -> dict:
        """Try to acquire locks, wait if blocked until ANY progress."""
        filepaths = args.get("filepaths", [])
        timeout_seconds = args.get("timeout_seconds", 30.0)

        # Validate input
        if not filepaths:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"error": "filepaths must be a non-empty array"}
                        ),
                    }
                ]
            }

        # Deduplicate and sort by canonical path to reduce deadlock risk
        # Use dict to preserve first occurrence while deduping by canonical path
        seen_canonical: dict[str, str] = {}  # canonical -> original
        for fp in filepaths:
            canon = _canonical(fp)
            if canon not in seen_canonical:
                seen_canonical[canon] = fp
        sorted_canonical = sorted(seen_canonical.keys())

        # First pass: try all locks (using canonical paths)
        results: list[dict[str, Any]] = []
        blocked_paths: list[str] = []

        for canon in sorted_canonical:
            original = seen_canonical[canon]
            acquired = try_lock(canon, agent_id, repo_namespace)
            holder = None if acquired else get_lock_holder(canon, repo_namespace)
            results.append(
                {
                    "filepath": original,
                    "canonical": canon,
                    "acquired": acquired,
                    "holder": holder,
                }
            )
            if not acquired:
                blocked_paths.append(canon)

        # If all acquired or non-blocking mode, return immediately
        if not blocked_paths or timeout_seconds == 0:
            all_acquired = len(blocked_paths) == 0
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "results": _strip_internal_fields(results),
                                "all_acquired": all_acquired,
                            }
                        ),
                    }
                ]
            }

        # Emit WAITING events for blocked files (once per file per call)
        for canon in blocked_paths:
            _emit_waiting(canon)

        # Wait until ANY blocked file becomes available
        # Spawn wait tasks for each blocked file (using canonical paths)
        wait_tasks: dict[asyncio.Task[bool], str] = {}  # task -> canonical path

        for canon in blocked_paths:
            task = asyncio.create_task(
                wait_for_lock_async(
                    canon,
                    agent_id,
                    repo_namespace,
                    timeout_seconds=timeout_seconds,
                    poll_interval_ms=100,
                )
            )
            wait_tasks[task] = canon

        try:
            # Wait for FIRST_COMPLETED - returns when ANY task finishes
            done, pending = await asyncio.wait(
                wait_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel remaining wait tasks to prevent unwanted acquisitions
            for task in pending:
                task.cancel()
            # Await cancelled tasks to ensure clean shutdown
            for task in pending:
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Process completed tasks
            for task in done:
                canon = wait_tasks[task]
                try:
                    acquired = task.result()
                except Exception:
                    acquired = False

                # Update result for this canonical path
                for r in results:
                    if r["canonical"] == canon:
                        r["acquired"] = acquired
                        r["holder"] = (
                            None if acquired else get_lock_holder(canon, repo_namespace)
                        )
                        break

        except BaseException:
            # On any exception (including CancelledError), cancel all wait tasks
            for task in wait_tasks:
                if not task.done():
                    task.cancel()
            # Await all tasks to ensure clean shutdown
            for task in wait_tasks:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            raise

        all_acquired = all(r["acquired"] for r in results)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "results": _strip_internal_fields(results),
                            "all_acquired": all_acquired,
                        }
                    ),
                }
            ]
        }

    @tool(
        name=LOCK_RELEASE_TOOL,
        description=(
            "Release locks on files. Use filepaths to release specific files, "
            "or all=true to release all locks held by this agent. Idempotent "
            "(succeeds even if locks not held)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "filepaths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to unlock",
                },
                "all": {
                    "type": "boolean",
                    "description": "Release ALL locks held by this agent",
                },
            },
            "oneOf": [
                {"required": ["filepaths"]},
                {"required": ["all"]},
            ],
        },
    )
    async def lock_release(args: dict) -> dict:
        """Release locks on specific files or all held locks."""
        filepaths = args.get("filepaths")
        release_all = args.get("all", False)

        # Validate mutually exclusive params
        if filepaths is not None and release_all:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"error": "Cannot specify both 'filepaths' and 'all'"}
                        ),
                    }
                ]
            }

        if filepaths is None and not release_all:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"error": "Must specify either 'filepaths' or 'all=true'"}
                        ),
                    }
                ]
            }

        if release_all:
            # Release all locks held by this agent
            count, released_paths = cleanup_agent_locks(agent_id)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "released": released_paths,
                                "count": count,
                            }
                        ),
                    }
                ]
            }

        # Release specific files
        if not filepaths:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"error": "filepaths must be a non-empty array"}
                        ),
                    }
                ]
            }

        # Deduplicate by canonical path
        seen_canonical: set[str] = set()
        released: list[str] = []
        for fp in filepaths:
            canon = _canonical(fp)
            if canon in seen_canonical:
                continue
            seen_canonical.add(canon)
            # release_lock returns True if released, False if not held
            # We track the path regardless (idempotent behavior)
            release_lock(canon, agent_id, repo_namespace)
            released.append(fp)

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "released": released,
                            "count": len(released),
                        }
                    ),
                }
            ]
        }

    config = create_sdk_mcp_server(
        name="mala-locking",
        version="1.0.0",
        tools=[lock_acquire, lock_release],
    )

    if _return_handlers:
        handlers = LockingToolHandlers(
            lock_acquire=lock_acquire,
            lock_release=lock_release,
        )
        return config, handlers

    return config
