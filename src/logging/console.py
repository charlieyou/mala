"""Console logging helpers for mala.

Claude Code style colored logging with agent identification support.
"""

import json
from datetime import datetime
from typing import Any


# Global verbose setting (can be modified at runtime)
_verbose_enabled: bool = False


def set_verbose(enabled: bool) -> None:
    """Enable or disable verbose output globally."""
    global _verbose_enabled
    _verbose_enabled = enabled


def is_verbose_enabled() -> bool:
    """Check if verbose output is currently enabled."""
    return _verbose_enabled


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length, adding ellipsis if truncated.

    Respects global verbose setting. If verbose is enabled,
    returns the original text unchanged.
    """
    if _verbose_enabled:
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

# Tools that have code fields which should be pretty-printed
EDIT_TOOLS = frozenset(
    {
        "mcp__morphllm__edit_file",
        "Edit",
        "Write",
        "NotebookEdit",
    }
)

# Fields in edit tools that contain code (should be pretty-printed with newlines)
CODE_FIELDS = frozenset(
    {
        "code_edit",
        "content",
        "new_source",
        "old_string",
        "new_string",
    }
)

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


def _format_arguments(
    arguments: dict[str, Any] | None, verbose: bool, tool_name: str = ""
) -> str:
    """Format tool arguments for display.

    Args:
        arguments: Tool arguments as a dictionary.
        verbose: Whether to show full output (vs abbreviated).
        tool_name: Name of the tool (used to detect edit tools for code formatting).

    Returns:
        Formatted string representation of arguments as key: value lines.
    """
    if not arguments:
        return ""

    is_edit_tool = tool_name in EDIT_TOOLS
    lines = []

    for key, value in arguments.items():
        is_code_field = key in CODE_FIELDS

        if isinstance(value, str):
            if is_code_field and is_edit_tool:
                # Code field: show with actual newlines
                if verbose:
                    # Full code display with distinct coloring
                    code_lines = value.split("\n")
                    lines.append(f"{Colors.CYAN}{key}:{Colors.RESET}")
                    for code_line in code_lines:
                        lines.append(f"  {Colors.DIM}{code_line}{Colors.RESET}")
                else:
                    # Truncated code preview
                    preview = value[:60].replace("\n", "↵")
                    if len(value) > 60:
                        preview += "..."
                    lines.append(
                        f"{Colors.CYAN}{key}:{Colors.RESET} {Colors.DIM}{preview}{Colors.RESET}"
                    )
            else:
                # Regular string field
                if verbose or len(value) <= 80:
                    lines.append(
                        f"{Colors.CYAN}{key}:{Colors.RESET} {Colors.WHITE}{value}{Colors.RESET}"
                    )
                else:
                    truncated = value[:80] + "..."
                    lines.append(
                        f"{Colors.CYAN}{key}:{Colors.RESET} {Colors.WHITE}{truncated}{Colors.RESET}"
                    )
        elif isinstance(value, bool):
            lines.append(
                f"{Colors.CYAN}{key}:{Colors.RESET} {Colors.WHITE}{str(value).lower()}{Colors.RESET}"
            )
        elif isinstance(value, (int, float)):
            lines.append(
                f"{Colors.CYAN}{key}:{Colors.RESET} {Colors.WHITE}{value}{Colors.RESET}"
            )
        elif isinstance(value, dict):
            if verbose:
                formatted = json.dumps(value, indent=2, ensure_ascii=False)
                lines.append(f"{Colors.CYAN}{key}:{Colors.RESET}")
                for dict_line in formatted.split("\n"):
                    lines.append(f"  {Colors.DIM}{dict_line}{Colors.RESET}")
            else:
                lines.append(
                    f"{Colors.CYAN}{key}:{Colors.RESET} {Colors.DIM}{{...}}{Colors.RESET}"
                )
        elif isinstance(value, list):
            if verbose:
                formatted = json.dumps(value, indent=2, ensure_ascii=False)
                lines.append(f"{Colors.CYAN}{key}:{Colors.RESET}")
                for list_line in formatted.split("\n"):
                    lines.append(f"  {Colors.DIM}{list_line}{Colors.RESET}")
            else:
                lines.append(
                    f"{Colors.CYAN}{key}:{Colors.RESET} {Colors.DIM}[...]{Colors.RESET}"
                )
        else:
            # Fallback for other types
            lines.append(
                f"{Colors.CYAN}{key}:{Colors.RESET} {Colors.WHITE}{value!r}{Colors.RESET}"
            )

    return "\n    ".join(lines)


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
        verbose = is_verbose_enabled()
        formatted_args = _format_arguments(arguments, verbose, tool_name)
        if formatted_args:
            # Multi-line key:value format (no "args:" prefix)
            args_output = f"\n    {formatted_args}"

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
