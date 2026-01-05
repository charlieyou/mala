"""MCP tool configuration for mala agents.

This module provides tool configuration functions that don't depend on
the Claude Agent SDK. SDK-dependent MCP server configuration is handled
directly in agent_runtime.py.
"""

from __future__ import annotations


def get_disallowed_tools() -> list[str]:
    """Get list of disallowed tools.

    Always includes MALA_DISALLOWED_TOOLS (tools disabled for mala agents).

    Returns:
        List of tool names that should be disallowed.
    """
    from src.infra.tool_config import MALA_DISALLOWED_TOOLS

    return list(MALA_DISALLOWED_TOOLS)
