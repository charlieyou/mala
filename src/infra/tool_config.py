"""Tool configuration constants for mala agents.

This module provides constants for tool configuration that are shared between
the MCP server configuration and hook implementations.
"""

from __future__ import annotations

# Tools disabled for mala agents to reduce token waste
MALA_DISALLOWED_TOOLS: list[str] = ["TodoWrite"]
