"""Backward-compatibility shim for src.mcp.

This module re-exports all public symbols from src.infra.mcp.
New code should import directly from src.infra.mcp.
"""

from src.infra.mcp import (
    MORPH_DISALLOWED_TOOLS,
    get_disallowed_tools,
    get_mcp_servers,
)

__all__ = ["MORPH_DISALLOWED_TOOLS", "get_disallowed_tools", "get_mcp_servers"]
