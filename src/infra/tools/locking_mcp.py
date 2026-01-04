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
    from collections.abc import Callable

    from claude_agent_sdk.types import McpSdkServerConfig

    from src.core.models import LockEvent

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


def create_locking_mcp_server(
    agent_id: str,
    repo_namespace: str | None,
    emit_lock_event: Callable[[LockEvent], None],
) -> McpSdkServerConfig:
    """Create MCP server config with locking tools bound to agent context.

    The emit_lock_event callback is captured by tool handler closures,
    allowing WAITING events to be emitted during lock acquisition.

    Args:
        agent_id: The agent ID for lock ownership.
        repo_namespace: Optional repo namespace for cross-repo disambiguation.
        emit_lock_event: Callback to emit lock events (captured in closures).

    Returns:
        MCP server configuration dict for Claude Agent SDK.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    # Import LockEvent and LockEventType here to avoid circular imports
    from src.core.models import LockEvent, LockEventType

    def _canonical(filepath: str) -> str:
        """Canonicalize path for consistent deadlock graph nodes."""
        return canonicalize_path(filepath, repo_namespace)

    def _emit_waiting(filepath: str) -> None:
        """Emit WAITING event for a blocked file."""
        canonical = _canonical(filepath)
        event = LockEvent(
            event_type=LockEventType.WAITING,
            agent_id=agent_id,
            lock_path=canonical,
            timestamp=time.time(),
        )
        emit_lock_event(event)
        logger.debug(
            "WAITING emitted: agent=%s path=%s",
            agent_id,
            canonical,
        )

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

        # Sort by canonical path to reduce deadlock risk
        sorted_paths = sorted(filepaths, key=_canonical)

        # First pass: try all locks
        results: list[dict[str, Any]] = []
        blocked_paths: list[str] = []

        for fp in sorted_paths:
            acquired = try_lock(fp, agent_id, repo_namespace)
            holder = None if acquired else get_lock_holder(fp, repo_namespace)
            results.append(
                {
                    "filepath": fp,
                    "acquired": acquired,
                    "holder": holder,
                }
            )
            if not acquired:
                blocked_paths.append(fp)

        # If all acquired or non-blocking mode, return immediately
        if not blocked_paths or timeout_seconds == 0:
            all_acquired = len(blocked_paths) == 0
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "results": results,
                                "all_acquired": all_acquired,
                            }
                        ),
                    }
                ]
            }

        # Emit WAITING events for blocked files (once per file per call)
        for fp in blocked_paths:
            _emit_waiting(fp)

        # Wait until ANY blocked file becomes available
        # Spawn wait tasks for each blocked file
        wait_tasks: dict[asyncio.Task, str] = {}  # task -> filepath

        for fp in blocked_paths:
            task = asyncio.create_task(
                wait_for_lock_async(
                    fp,
                    agent_id,
                    repo_namespace,
                    timeout_seconds=timeout_seconds,
                    poll_interval_ms=100,
                )
            )
            wait_tasks[task] = fp

        try:
            # Wait for FIRST_COMPLETED - returns when ANY task finishes
            done, pending = await asyncio.wait(
                wait_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel remaining wait tasks to prevent unwanted acquisitions
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Process completed tasks
            for task in done:
                fp = wait_tasks[task]
                try:
                    acquired = task.result()
                except Exception:
                    acquired = False

                # Update result for this filepath
                for r in results:
                    if r["filepath"] == fp:
                        r["acquired"] = acquired
                        r["holder"] = (
                            None if acquired else get_lock_holder(fp, repo_namespace)
                        )
                        break

        except Exception as e:
            logger.exception("Error waiting for locks: %s", e)
            # On error, update all blocked as still blocked
            pass

        all_acquired = all(r["acquired"] for r in results)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "results": results,
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
            count = cleanup_agent_locks(agent_id)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "released": [],
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

        released: list[str] = []
        for fp in filepaths:
            # release_lock returns True if released, False if not held
            # We track the path regardless (idempotent behavior)
            release_lock(fp, agent_id, repo_namespace)
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

    return create_sdk_mcp_server(
        name="mala-locking",
        version="1.0.0",
        tools=[lock_acquire, lock_release],
    )
