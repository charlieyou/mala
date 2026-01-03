"""MCP server configuration for Claude Agent SDK.

This module provides the single source of truth for MCP server configuration,
including Morph integration settings. All MCP-related configuration should be
imported from this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

# Tools replaced by MorphLLM MCP (use disallowed_tools parameter, not hooks)
MORPH_DISALLOWED_TOOLS: list[str] = ["Edit", "Grep"]

# Tools disabled for mala agents to reduce token waste
MALA_DISALLOWED_TOOLS: list[str] = ["TodoWrite"]


def get_mcp_servers(
    repo_path: Path, morph_api_key: str | None = None, morph_enabled: bool = True
) -> dict[str, Any]:
    """Get MCP servers configuration for agents.

    Args:
        repo_path: Path to the repository.
        morph_api_key: Morph API key. Required when morph_enabled=True.
        morph_enabled: Whether to enable Morph MCP server. Defaults to True.

    Returns:
        Dictionary of MCP server configurations. Empty if morph_enabled=False.

    Raises:
        ValueError: If morph_enabled=True but morph_api_key is not provided.
    """
    if not morph_enabled:
        return {}
    if not morph_api_key:
        raise ValueError(
            "morph_api_key is required when morph_enabled=True. "
            "Pass morph_api_key from MalaConfig or set morph_enabled=False."
        )
    return {
        "morphllm": {
            "command": "npx",
            "args": ["-y", "@morphllm/morphmcp"],
            "cwd": str(repo_path),
            "env": {
                "MORPH_API_KEY": morph_api_key,
                "ENABLED_TOOLS": "all",
                "WORKSPACE_MODE": "true",
                "WORKSPACE_PATH": str(repo_path),
            },
        }
    }


def get_disallowed_tools(morph_enabled: bool) -> list[str]:
    """Get list of disallowed tools based on Morph enablement.

    Always includes MALA_DISALLOWED_TOOLS (tools disabled for mala agents).
    Additionally includes MORPH_DISALLOWED_TOOLS when Morph MCP is enabled.

    Args:
        morph_enabled: Whether Morph MCP is enabled.

    Returns:
        List of tool names that should be disallowed. Always includes
        mala-specific tools; adds Morph-replaced tools when Morph is enabled.
    """
    tools = list(MALA_DISALLOWED_TOOLS)  # Always disable mala-specific tools
    if morph_enabled:
        tools.extend(MORPH_DISALLOWED_TOOLS)
    return tools
