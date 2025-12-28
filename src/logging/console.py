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
    # Bright colors for better terminal visibility
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    GRAY = "\033[90m"  # Bright black - more visible on dark terminals
    WHITE = "\033[97m"
    # Subdued style for secondary info (uses gray instead of dim for visibility)
    MUTED = "\033[90m"


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

# Tools that show file path in quiet mode
FILE_TOOLS = frozenset(
    {
        "mcp__morphllm__edit_file",
        "Edit",
        "Write",
        "NotebookEdit",
        "Read",
        "Glob",
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
) -> None:
    """Claude Code style logging with optional agent color coding."""
    style = Colors.MUTED if dim else ""
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
    arguments: dict[str, Any] | None,
    verbose: bool,
    tool_name: str = "",
    key_color: str = Colors.CYAN,
) -> str:
    """Format tool arguments for display.

    Args:
        arguments: Tool arguments as a dictionary.
        verbose: Whether to show full output (vs abbreviated).
        tool_name: Name of the tool (used to detect edit tools for code formatting).
        key_color: Color to use for argument keys (defaults to CYAN).

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
                    lines.append(f"{key_color}{key}:{Colors.RESET}")
                    for code_line in code_lines:
                        lines.append(f"  {Colors.MUTED}{code_line}{Colors.RESET}")
                else:
                    # Truncated code preview
                    preview = value[:60].replace("\n", "↵")
                    if len(value) > 60:
                        preview += "..."
                    lines.append(
                        f"{key_color}{key}:{Colors.RESET} {Colors.MUTED}{preview}{Colors.RESET}"
                    )
            else:
                # Regular string field
                if verbose or len(value) <= 80:
                    lines.append(
                        f"{key_color}{key}:{Colors.RESET} {Colors.WHITE}{value}{Colors.RESET}"
                    )
                else:
                    truncated = value[:80] + "..."
                    lines.append(
                        f"{key_color}{key}:{Colors.RESET} {Colors.WHITE}{truncated}{Colors.RESET}"
                    )
        elif isinstance(value, bool):
            lines.append(
                f"{key_color}{key}:{Colors.RESET} {Colors.WHITE}{str(value).lower()}{Colors.RESET}"
            )
        elif isinstance(value, (int, float)):
            lines.append(
                f"{key_color}{key}:{Colors.RESET} {Colors.WHITE}{value}{Colors.RESET}"
            )
        elif isinstance(value, dict):
            if verbose:
                formatted = json.dumps(value, indent=2, ensure_ascii=False)
                lines.append(f"{key_color}{key}:{Colors.RESET}")
                for dict_line in formatted.split("\n"):
                    lines.append(f"  {Colors.MUTED}{dict_line}{Colors.RESET}")
            else:
                lines.append(
                    f"{key_color}{key}:{Colors.RESET} {Colors.MUTED}{{...}}{Colors.RESET}"
                )
        elif isinstance(value, list):
            if verbose:
                formatted = json.dumps(value, indent=2, ensure_ascii=False)
                lines.append(f"{key_color}{key}:{Colors.RESET}")
                for list_line in formatted.split("\n"):
                    lines.append(f"  {Colors.MUTED}{list_line}{Colors.RESET}")
            else:
                lines.append(
                    f"{key_color}{key}:{Colors.RESET} {Colors.MUTED}[...]{Colors.RESET}"
                )
        else:
            # Fallback for other types
            lines.append(
                f"{key_color}{key}:{Colors.RESET} {Colors.WHITE}{value!r}{Colors.RESET}"
            )

    return "\n    ".join(lines)


def log_tool(
    tool_name: str,
    description: str = "",
    agent_id: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> None:
    """Log tool usage in Claude Code style.

    Args:
        tool_name: Name of the tool being called.
        description: Brief description of the tool action.
        agent_id: Optional agent ID for color coding.
        arguments: Optional tool arguments to display.

    In quiet mode (non-verbose), shows single line per tool call:
    - File tools (Read/Edit/Write/etc): show file_path or path
    - Bash: show description field
    - Other tools: show truncated args dict
    """
    icon = "\u2699"
    verbose = is_verbose_enabled()

    if agent_id:
        agent_color = get_agent_color(agent_id)
        prefix = f"{agent_color}[{agent_id}]{Colors.RESET} "
    else:
        agent_color = Colors.CYAN
        prefix = ""

    if not verbose:
        # Quiet mode: single line output
        summary = _get_quiet_summary(tool_name, description, arguments)
        if summary:
            print(
                f"  {prefix}{Colors.CYAN}{icon} {tool_name}{Colors.RESET} "
                f"{Colors.MUTED}{summary}{Colors.RESET}"
            )
        else:
            print(f"  {prefix}{Colors.CYAN}{icon} {tool_name}{Colors.RESET}")
        return

    # Verbose mode: full output with arguments
    desc_text = truncate_text(description, 50) if description else ""
    desc = f" {Colors.MUTED}{desc_text}{Colors.RESET}" if desc_text else ""

    # Format arguments if provided
    args_output = ""
    if arguments:
        formatted_args = _format_arguments(arguments, verbose, tool_name, agent_color)
        if formatted_args:
            # Multi-line key:value format (no "args:" prefix)
            args_output = f"\n    {formatted_args}"

    print(f"  {prefix}{Colors.CYAN}{icon} {tool_name}{Colors.RESET}{desc}{args_output}")


def _get_quiet_summary(
    tool_name: str, description: str, arguments: dict[str, Any] | None
) -> str:
    """Get single-line summary for quiet mode output.

    Args:
        tool_name: Name of the tool being called.
        description: Brief description of the tool action.
        arguments: Tool arguments.

    Returns:
        Single-line summary string.
    """
    # File tools: show file path
    if tool_name in FILE_TOOLS and arguments:
        path = (
            arguments.get("file_path")
            or arguments.get("path")
            or arguments.get("pattern")
        )
        if path:
            return str(path)

    # Bash: show description field
    if tool_name == "Bash":
        if description:
            return description
        if arguments and arguments.get("description"):
            return str(arguments["description"])

    # Other tools: truncated args dict
    if arguments:
        keys = list(arguments.keys())[:3]
        if keys:
            preview = ", ".join(f"{k}=..." for k in keys)
            if len(arguments) > 3:
                preview += f", +{len(arguments) - 3} more"
            return f"{{{preview}}}"

    return ""


def log_agent_text(text: str, agent_id: str) -> None:
    """Log agent text output with consistent formatting.

    Uses 2-space indent like other log functions for consistency.
    Applies truncation if enabled.
    """
    truncated = truncate_text(text, 100)
    agent_color = get_agent_color(agent_id)
    print(
        f"  {agent_color}[{agent_id}]{Colors.RESET} {Colors.MUTED}{truncated}{Colors.RESET}"
    )
