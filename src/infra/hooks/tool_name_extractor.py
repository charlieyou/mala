"""Extract human-readable tool names from shell commands.

The implementation has been moved to src/core/tool_name_extractor.py.
Import extract_tool_name from there.

This file re-exports for compatibility with existing imports.
"""

# Re-export from canonical location
from src.core.tool_name_extractor import extract_tool_name

__all__ = ["extract_tool_name"]
