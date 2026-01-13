"""Review issue formatting for follow-up prompts.

This module provides formatting utilities for review issues that are used
within the pipeline layer. This keeps the pipeline decoupled from infra
client implementations.

The formatter works with any object that satisfies ReviewIssueProtocol
from src.core.protocols.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.protocols.review import ReviewIssueProtocol


def _to_relative_path(file_path: str, resolved_base: Path) -> str:
    """Convert an absolute file path to a relative path for display.

    Strips the base path prefix to show paths relative to the repository root.

    Args:
        file_path: Absolute or relative file path.
        resolved_base: Pre-resolved base path for relativization.

    Returns:
        Relative path suitable for display. If relativization fails,
        returns the original path to preserve directory context.
    """
    path = Path(file_path)
    # If already relative, return as-is
    if not path.is_absolute():
        return file_path

    try:
        if path.is_relative_to(resolved_base):
            return str(path.relative_to(resolved_base))
    except (ValueError, OSError):
        pass

    # Preserve original path if relativization fails (don't strip to filename)
    return file_path


def format_review_issues(
    issues: Sequence[ReviewIssueProtocol], base_path: Path | None = None
) -> str:
    """Format review issues as a human-readable string for follow-up prompts.

    Args:
        issues: Sequence of objects satisfying ReviewIssueProtocol.
        base_path: Base path (typically repository root) for path relativization.
            If None, uses Path.cwd() as fallback.

    Returns:
        Formatted string with issues grouped by file.
    """
    if not issues:
        return "No specific issues found."

    # Resolve base_path once outside the loop (Finding 6)
    resolved_base = base_path.resolve() if base_path else Path.cwd()

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
        display_file = _to_relative_path(issue.file, resolved_base)
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
        # Include reviewer attribution (Finding 7: no extra space when empty)
        reviewer_tag = f"[{issue.reviewer}] " if issue.reviewer else ""
        lines.append(f"  {loc}: {reviewer_tag}{issue.title}")
        if issue.body:
            # Finding 4: Indent multi-line bodies properly
            indented_body = issue.body.replace("\n", "\n    ")
            lines.append(f"    {indented_body}")

    return "\n".join(lines)
