"""Tool configuration constants for mala agents.

This module provides constants for tool configuration that are shared between
the MCP server configuration and hook implementations.
"""

from __future__ import annotations

# Tools disabled for mala agents to reduce token waste.
#
# "Monitor" is disabled so long-running work is steered to
# ``Bash(run_in_background=true)``, whose completion mala can observe via the
# SDK's structured task-notification stream (it cannot observe Monitor).
MALA_DISALLOWED_TOOLS: list[str] = ["TodoWrite", "Monitor"]
