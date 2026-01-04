"""MCP server configuration for Claude Agent SDK.

This module provides the single source of truth for MCP server configuration.
All MCP-related configuration should be imported from this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.core.models import LockEvent

# Tools disabled for mala agents to reduce token waste
MALA_DISALLOWED_TOOLS: list[str] = ["TodoWrite"]


def get_mcp_servers(
    repo_path: Path,
    *,
    agent_id: str | None = None,
    emit_lock_event: Callable[[LockEvent], object] | None = None,
) -> dict[str, Any]:
    """Get MCP servers configuration for agents.

    Args:
        repo_path: Path to the repository.
        agent_id: Optional agent ID for lock ownership. Required for locking server.
        emit_lock_event: Optional callback to emit lock events. Required for locking server.
            May be sync or async; the MCP tool handlers will schedule async callbacks.

    Returns:
        Dictionary of MCP server configurations. Includes 'mala-locking' server
        when both agent_id and emit_lock_event are provided.
    """
    servers: dict[str, Any] = {}

    # Include locking MCP server when both agent_id and emit_lock_event are provided
    if agent_id is not None and emit_lock_event is not None:
        from src.infra.tools.locking_mcp import create_locking_mcp_server

        servers["mala-locking"] = create_locking_mcp_server(
            agent_id=agent_id,
            repo_namespace=str(repo_path),
            emit_lock_event=emit_lock_event,
        )

    return servers


def get_disallowed_tools() -> list[str]:
    """Get list of disallowed tools.

    Always includes MALA_DISALLOWED_TOOLS (tools disabled for mala agents).

    Returns:
        List of tool names that should be disallowed.
    """
    return list(MALA_DISALLOWED_TOOLS)
