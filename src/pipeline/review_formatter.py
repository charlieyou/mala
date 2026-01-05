"""Review issue formatting for follow-up prompts.

This module provides formatting utilities for review issues that are used
within the pipeline layer. This keeps the pipeline decoupled from infra
client implementations.

The formatter works with any object that satisfies the ReviewIssue protocol
from src.domain.lifecycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.domain.lifecycle import ReviewIssue


def _to_relative_path(file_path: str, base_path: Path | None = None) -> str:
    """Convert an absolute file path to a relative path for display.

    Strips the base path prefix to show paths relative to the repository root.

    Args:
        file_path: Absolute or relative file path.
        base_path: Base path (typically repository root) for relativization.
            If None, uses Path.cwd() as fallback.

    Returns:
        Relative path suitable for display. If relativization fails,
        returns the original path to preserve directory context.
    """
    # If already relative, return as-is
    if not file_path.startswith("/"):
        return file_path

    # Use provided base_path or fall back to cwd
    base = base_path.resolve() if base_path else Path.cwd()
    try:
        abs_path = Path(file_path)
        if abs_path.is_relative_to(base):
            return str(abs_path.relative_to(base))
    except (ValueError, OSError):
        pass

    # Preserve original path if relativization fails (don't strip to filename)
    return file_path


def format_review_issues(
    issues: Sequence[ReviewIssue], base_path: Path | None = None
) -> str:
    """Format review issues as a human-readable string for follow-up prompts.

    Args:
        issues: Sequence of objects satisfying the ReviewIssue protocol.
        base_path: Base path (typically repository root) for path relativization.
            If None, uses Path.cwd() as fallback.

    Returns:
        Formatted string with issues grouped by file.
    """
    if not issues:
        return "No specific issues found."

    lines: list[str] = []
    current_file: str | None = None

    # Sort by file, then by line, then by priority (lower = more important)
    sorted_issues = sorted(
        issues,
        key=lambda x: (
            x.file,
            x.line_start,
            x.priority if x.priority is not None else 4,
        ),
    )

    for issue in sorted_issues:
        # Convert absolute paths to relative for cleaner display
        display_file = _to_relative_path(issue.file, base_path)
        if display_file != current_file:
            if current_file is not None:
                lines.append("")  # Blank line between files
            current_file = display_file
            lines.append(f"File: {display_file}")

        loc = (
            f"L{issue.line_start}-{issue.line_end}"
            if issue.line_start != issue.line_end
            else f"L{issue.line_start}"
        )
        # Include reviewer attribution
        reviewer_tag = f"[{issue.reviewer}]" if issue.reviewer else ""
        lines.append(f"  {loc}: {reviewer_tag} {issue.title}")
        if issue.body:
            lines.append(f"    {issue.body}")

    return "\n".join(lines)
