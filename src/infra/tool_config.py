"""Tool configuration constants for mala agents.

This module provides constants for tool configuration that are shared between
the MCP server configuration and hook implementations. Keeping these in a
separate module avoids circular dependencies between mcp.py and hooks.
"""

from __future__ import annotations

# Tools disabled for mala agents to reduce token waste
MALA_DISALLOWED_TOOLS: list[str] = ["TodoWrite"]
