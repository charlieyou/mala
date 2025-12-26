"""Console logging helpers for mala.

Claude Code style colored logging with agent identification support.
"""

import json
from datetime import datetime
from typing import Any


# Global truncation setting (can be modified at runtime)
_truncate_enabled: bool = True


def set_truncation(enabled: bool) -> None:
    """Enable or disable output truncation globally."""
    global _truncate_enabled
    _truncate_enabled = enabled


def is_truncation_enabled() -> bool:
    """Check if truncation is currently enabled."""
    return _truncate_enabled


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length, adding ellipsis if truncated.

    Respects global truncation setting. If truncation is disabled,
    returns the original text unchanged.
    """
    if not _truncate_enabled:
        return text
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


class Colors:
    """ANSI color codes for terminal output (bright variants)."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    # Bright colors for better terminal visibility
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    GRAY = "\033[37m"  # Brighter than 90
    WHITE = "\033[97m"


# Agent color palette for distinguishing concurrent agents
AGENT_COLORS = [
    "\033[96m",  # Bright Cyan
    "\033[93m",  # Bright Yellow
    "\033[95m",  # Bright Magenta
    "\033[92m",  # Bright Green
    "\033[94m",  # Bright Blue
    "\033[97m",  # Bright White
]

# Maps agent/issue IDs to their assigned colors
_agent_color_map: dict[str, str] = {}
_agent_color_index = 0


def get_agent_color(agent_id: str) -> str:
    """Get a consistent color for an agent based on its ID."""
    global _agent_color_index
    if agent_id not in _agent_color_map:
        _agent_color_map[agent_id] = AGENT_COLORS[
            _agent_color_index % len(AGENT_COLORS)
        ]
        _agent_color_index += 1
    return _agent_color_map[agent_id]


def log(
    icon: str,
    message: str,
    color: str = Colors.RESET,
    dim: bool = False,
    agent_id: str | None = None,
):
    """Claude Code style logging with optional agent color coding."""
    style = Colors.DIM if dim else ""
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Use agent color if provided, otherwise use specified color
    if agent_id:
        agent_color = get_agent_color(agent_id)
        prefix = f"{agent_color}[{agent_id}]{Colors.RESET} "
    else:
        prefix = ""

    print(
        f"{Colors.GRAY}{timestamp}{Colors.RESET} {prefix}{style}{color}{icon} {message}{Colors.RESET}"
    )


def _format_arguments(arguments: dict[str, Any] | None, truncate: bool) -> str:
    """Format tool arguments for display.

    Args:
        arguments: Tool arguments as a dictionary.
        truncate: Whether to truncate/abbreviate the output.

    Returns:
        Formatted string representation of arguments.
    """
    if not arguments:
        return ""

    if truncate:
        # Abbreviated format: key=value pairs, truncated
        parts = []
        for key, value in arguments.items():
            if isinstance(value, str):
                # Truncate long string values
                if len(value) > 30:
                    value = value[:30] + "..."
                parts.append(f"{key}={value!r}")
            elif isinstance(value, dict):
                parts.append(f"{key}={{...}}")
            elif isinstance(value, list):
                parts.append(f"{key}=[...]")
            else:
                parts.append(f"{key}={value!r}")
        result = ", ".join(parts)
        # Truncate overall length
        if len(result) > 80:
            result = result[:80] + "..."
        return result
    else:
        # Full format: pretty-printed JSON for complex args
        try:
            formatted = json.dumps(arguments, indent=2, ensure_ascii=False)
            return formatted
        except (TypeError, ValueError):
            # Fallback for non-serializable objects
            return str(arguments)


def log_tool(
    tool_name: str,
    description: str = "",
    agent_id: str | None = None,
    arguments: dict[str, Any] | None = None,
):
    """Log tool usage in Claude Code style.

    Args:
        tool_name: Name of the tool being called.
        description: Brief description of the tool action.
        agent_id: Optional agent ID for color coding.
        arguments: Optional tool arguments to display.
    """
    icon = "\u2699"
    # Apply truncation to description
    desc_text = truncate_text(description, 50) if description else ""
    desc = f" {Colors.DIM}{desc_text}{Colors.RESET}" if desc_text else ""

    if agent_id:
        agent_color = get_agent_color(agent_id)
        prefix = f"{agent_color}[{agent_id}]{Colors.RESET} "
    else:
        prefix = ""

    # Format arguments if provided
    args_output = ""
    if arguments:
        truncate = is_truncation_enabled()
        formatted_args = _format_arguments(arguments, truncate)
        if formatted_args:
            if truncate:
                # Single line for truncated mode
                args_output = f"\n    {Colors.DIM}args: {formatted_args}{Colors.RESET}"
            else:
                # Multi-line for full mode
                lines = formatted_args.split("\n")
                indented = "\n    ".join(lines)
                args_output = f"\n    {Colors.DIM}args:\n    {indented}{Colors.RESET}"

    print(f"  {prefix}{Colors.CYAN}{icon} {tool_name}{Colors.RESET}{desc}{args_output}")


def log_agent_text(text: str, agent_id: str) -> None:
    """Log agent text output with consistent formatting.

    Uses 2-space indent like other log functions for consistency.
    Applies truncation if enabled.
    """
    truncated = truncate_text(text, 100)
    agent_color = get_agent_color(agent_id)
    print(
        f"  {agent_color}[{agent_id}]{Colors.RESET} {Colors.DIM}{truncated}{Colors.RESET}"
    )


def log_result(success: bool, message: str, agent_id: str | None = None):
    """Log result in Claude Code style."""
    if agent_id:
        agent_color = get_agent_color(agent_id)
        prefix = f"{agent_color}[{agent_id}]{Colors.RESET} "
    else:
        prefix = ""

    if success:
        print(f"  {prefix}{Colors.GREEN}\u2713 {message}{Colors.RESET}")
    else:
        print(f"  {prefix}{Colors.RED}\u2717 {message}{Colors.RESET}")
