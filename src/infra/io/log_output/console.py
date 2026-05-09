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
    GRAY = "\033[37;1m"  # Bold light gray - brighter for dark terminal visibility
    WHITE = "\033[97m"
    # Subdued style for secondary info (brighter than ANSI 90 for visibility)
    MUTED = "\033[37m"  # Standard gray - visible but subdued on dark terminals


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
        "Edit",
        "Write",
        "NotebookEdit",
    }
)

# Tools that show file path in quiet mode
FILE_TOOLS = frozenset(
    {
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

QUIET_SUMMARY_MAX_LENGTH = 80

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
    issue_id: str | None = None,
) -> None:
    """Claude Code style logging with optional agent color coding.

    Note: The dim parameter is accepted for API compatibility but no longer
    applies the DIM ANSI attribute, as it reduces visibility on dark terminals.
    Callers should use Colors.GRAY or Colors.MUTED directly for subdued output.

    Args:
        icon: Icon to display (e.g. "→", "◦").
        message: Message to log.
        color: Color for the message.
        dim: Kept for API compatibility (no longer applies DIM).
        agent_id: Optional agent ID for color mapping.
        issue_id: Optional issue ID for display. When provided, shows [issue_id]
            instead of [agent_id], but still uses agent_id for color mapping.
    """
    # Use caller's color directly - dim parameter kept for API compatibility
    # but no longer reduces visibility (improves dark terminal readability)
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Build prefix: prefer issue_id for display, agent_id for color
    if issue_id:
        # Use issue_id as display, agent_id for color mapping
        agent_color = get_agent_color(agent_id) if agent_id else Colors.CYAN
        prefix = f"{agent_color}[{issue_id}]{Colors.RESET} "
    elif agent_id:
        agent_color = get_agent_color(agent_id)
        prefix = f"{agent_color}[{agent_id}]{Colors.RESET} "
    else:
        prefix = ""

    print(
        f"{Colors.GRAY}{timestamp}{Colors.RESET} {prefix}{color}{icon} {message}{Colors.RESET}"
    )


def log_verbose(
    icon: str,
    message: str,
    color: str = Colors.MUTED,
    agent_id: str | None = None,
    issue_id: str | None = None,
) -> None:
    """Log a message only when verbose mode is enabled.

    Use this for detailed state transitions and debug info that users
    only need when troubleshooting with the -v flag.

    Args:
        icon: Icon to display (e.g. "→", "◦").
        message: Message to log.
        color: Color for the message (defaults to MUTED).
        agent_id: Optional agent ID for color coding.
        issue_id: Optional issue ID for display. When provided, shows [issue_id]
            instead of [agent_id], but still uses agent_id for color mapping.
    """
    if _verbose_enabled:
        log(icon, message, color, agent_id=agent_id, issue_id=issue_id)


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
    - Shell tools (Bash/shell_command): show command or cmd
    - Other tools: show truncated args dict
    """
    icon = "\u2699"
    verbose = is_verbose_enabled()
    display_tool_name = _display_tool_name(tool_name)

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
                f"  {prefix}{Colors.CYAN}{icon} {display_tool_name}{Colors.RESET} "
                f"{Colors.MUTED}{summary}{Colors.RESET}"
            )
        else:
            print(f"  {prefix}{Colors.CYAN}{icon} {display_tool_name}{Colors.RESET}")
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

    print(
        f"  {prefix}{Colors.CYAN}{icon} {display_tool_name}{Colors.RESET}"
        f"{desc}{args_output}"
    )


def _display_tool_name(tool_name: str) -> str:
    """Return the user-facing tool name for console output."""
    if _is_shell_tool(tool_name):
        return "Bash"
    if _is_lock_acquire_tool(tool_name):
        return "Lock"
    if _is_lock_release_tool(tool_name):
        return "Release"
    if _is_edit_tool(tool_name):
        return "Edit"
    return tool_name


def _is_shell_tool(tool_name: str) -> bool:
    return tool_name.lower() in {"bash", "shell_command"}


def _is_lock_acquire_tool(tool_name: str) -> bool:
    normalized = tool_name.replace("-", "_").replace(".", "__")
    return normalized in {
        "lock_acquire",
        "mcp__mala_locking__lock_acquire",
        "mala_locking__lock_acquire",
    }


def _is_lock_release_tool(tool_name: str) -> bool:
    normalized = tool_name.replace("-", "_").replace(".", "__")
    return normalized in {
        "lock_release",
        "mcp__mala_locking__lock_release",
        "mala_locking__lock_release",
    }


def _is_edit_tool(tool_name: str) -> bool:
    return tool_name in {"apply_patch", "edit_file", "file_change"}


def _get_shell_command(arguments: dict[str, Any] | None) -> str | None:
    if not arguments:
        return None
    command = arguments.get("command")
    if command is None:
        command = arguments.get("cmd")
    if command is None:
        return None
    return str(command)


def _format_quiet_summary_text(text: str) -> str:
    summary = text.replace("\n", "↵")
    if len(summary) > QUIET_SUMMARY_MAX_LENGTH:
        return summary[:QUIET_SUMMARY_MAX_LENGTH] + "..."
    return summary


def _format_lock_filepaths(arguments: dict[str, Any] | None) -> str | None:
    if not arguments:
        return None

    filepaths = arguments.get("filepaths")
    if isinstance(filepaths, list):
        paths = [str(path) for path in filepaths if path]
        if paths:
            return ", ".join(paths)
    if filepaths:
        return str(filepaths)
    if arguments.get("all") is True:
        return "all"
    return None


def _format_edit_paths(arguments: dict[str, Any] | None) -> str | None:
    if not arguments:
        return None

    paths = arguments.get("paths")
    if isinstance(paths, list):
        path_strings = [str(path) for path in paths if path]
        if path_strings:
            return ", ".join(path_strings)
    if paths:
        return str(paths)

    path = arguments.get("path")
    if path:
        return str(path)

    changes = arguments.get("changes")
    if isinstance(changes, list):
        path_strings = [
            str(change.get("path")) for change in changes if change.get("path")
        ]
        if path_strings:
            return ", ".join(path_strings)
    return None


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
            return _format_quiet_summary_text(str(path))

    if _is_lock_acquire_tool(tool_name):
        filepaths = _format_lock_filepaths(arguments)
        if filepaths is not None:
            return _format_quiet_summary_text(filepaths)

    if _is_lock_release_tool(tool_name):
        filepaths = _format_lock_filepaths(arguments)
        if filepaths is not None:
            return _format_quiet_summary_text(filepaths)

    if _is_edit_tool(tool_name):
        paths = _format_edit_paths(arguments)
        if paths is not None:
            return _format_quiet_summary_text(paths)

    # Shell tools: show the command itself. Amp deep mode reports
    # `command`; Amp smart mode reports `cmd`.
    if _is_shell_tool(tool_name):
        command = _get_shell_command(arguments)
        if command is not None:
            return _format_quiet_summary_text(command)
        if description:
            return _format_quiet_summary_text(description)
        if arguments and arguments.get("description"):
            return _format_quiet_summary_text(str(arguments["description"]))

    # Other tools: truncated args dict
    if arguments:
        keys = list(arguments.keys())[:3]
        if keys:
            preview = ", ".join(f"{k}=..." for k in keys)
            if len(arguments) > 3:
                preview += f", +{len(arguments) - 3} more"
            return _format_quiet_summary_text(f"{{{preview}}}")

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


# Color name to ANSI code mapping
_COLOR_MAP: dict[str, str] = {
    "cyan": Colors.CYAN,
    "green": Colors.GREEN,
    "red": Colors.RED,
    "yellow": Colors.YELLOW,
    "blue": Colors.BLUE,
    "magenta": Colors.MAGENTA,
    "gray": Colors.GRAY,
    "white": Colors.WHITE,
}


# Known icon characters that can prefix messages
# Only these characters will be parsed as icon prefixes in LoggerPort.log()
KNOWN_ICONS: frozenset[str] = frozenset(
    {
        "▸",  # Running/in-progress
        "→",  # Default/transition
        "◦",  # Verbose/secondary
        "●",  # Primary marker
        "○",  # Empty/skipped marker
        "✓",  # Success
        "✗",  # Failure
        "⚠",  # Warning
        "!",  # Warning (alternate)
        "⚙",  # Tool/processing
        "◐",  # Config/settings
        "◌",  # Ready/waiting
        "▶",  # Agent started
        "🧹",  # Cleanup
        "🔍",  # Verification
    }
)


class ConsoleLoggerAdapter:
    """Adapter that implements LoggerPort for console output.

    Maps color names to ANSI codes and delegates to the log() function.
    """

    def log(
        self,
        message: str,
        *,
        level: str = "info",
        color: str | None = None,
    ) -> None:
        """Log a message to the console.

        Args:
            message: The message to log.
            level: Log level (unused, kept for interface compatibility).
            color: Optional color name (e.g., "cyan", "green", "red").
        """
        # Map color name to ANSI code, default to reset if unknown
        ansi_color = _COLOR_MAP.get(color, Colors.RESET) if color else Colors.RESET
        # Extract icon prefix if present and is a known icon
        # (e.g., "▸ Running..." -> icon="▸", rest="Running...")
        # Only split on known icons to avoid misparsing messages like "A thing happened"
        if (
            message
            and len(message) >= 2
            and message[1] == " "
            and message[0] in KNOWN_ICONS
        ):
            icon = message[0]
            rest = message[2:]
        else:
            icon = "→"
            rest = message
        log(icon, rest, color=ansi_color)
