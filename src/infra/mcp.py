"""MCP server configuration for Claude Agent SDK.

This module provides the single source of truth for MCP server configuration.
All MCP-related configuration should be imported from this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

# Tools disabled for mala agents to reduce token waste
MALA_DISALLOWED_TOOLS: list[str] = ["TodoWrite"]


def get_mcp_servers(
    repo_path: Path,
) -> dict[str, Any]:
    """Get MCP servers configuration for agents.

    Args:
        repo_path: Path to the repository.

    Returns:
        Empty dictionary (no MCP servers configured).
    """
    return {}


def get_disallowed_tools() -> list[str]:
    """Get list of disallowed tools.

    Always includes MALA_DISALLOWED_TOOLS (tools disabled for mala agents).

    Returns:
        List of tool names that should be disallowed.
    """
    return list(MALA_DISALLOWED_TOOLS)
