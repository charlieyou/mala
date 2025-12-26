"""Handoff writer for failed agent issues.

Implements Track A6 from coordination-plan-codex.md:
Write .mala/handoff/<issue_id>.md with last commands and error summary
when an agent fails.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


def sanitize_issue_id(issue_id: str) -> str:
    """Sanitize issue ID for use as a filename.

    Removes path separators and special characters that could cause
    security issues or filesystem problems.

    Args:
        issue_id: The raw issue ID.

    Returns:
        A safe string for use in filenames.
    """
    # Remove path separators and common special chars
    sanitized = re.sub(r'[/\\<>:"|?*\.]', "", issue_id)

    # If nothing left, use placeholder
    if not sanitized:
        return "unknown"

    return sanitized


class HandoffWriter:
    """Writes handoff files for failed issues."""

    # Default limit for command history
    MAX_COMMANDS = 10

    def __init__(self, repo_path: Path):
        """Initialize handoff writer.

        Args:
            repo_path: Path to the repository root.
        """
        self.repo_path = repo_path
        self.handoff_dir = repo_path / ".mala" / "handoff"

    def _extract_commands_from_log(self, log_path: Path) -> list[str]:
        """Extract Bash commands from a JSONL log file.

        Args:
            log_path: Path to the JSONL log file.

        Returns:
            List of command strings from Bash tool calls.
        """
        commands: list[str] = []

        if not log_path.exists():
            return commands

        try:
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Look for Bash tool_use entries in assistant messages
                    message = entry.get("message", {})
                    content = message.get("content", [])

                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_use":
                            continue
                        if block.get("name") != "Bash":
                            continue

                        input_data = block.get("input", {})
                        command = input_data.get("command", "")
                        if command:
                            commands.append(command)

        except OSError:
            pass

        return commands

    def _extract_last_error(self, log_path: Path) -> str | None:
        """Extract the last error from tool results in a JSONL log file.

        Args:
            log_path: Path to the JSONL log file.

        Returns:
            Error content string if found, None otherwise.
        """
        if not log_path.exists():
            return None

        last_error: str | None = None

        try:
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Look for tool_result entries with is_error=True
                    message = entry.get("message", {})
                    content = message.get("content", [])

                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_result":
                            continue
                        if block.get("is_error"):
                            last_error = block.get("content", "")

        except OSError:
            pass

        return last_error

    def write_handoff(
        self,
        issue_id: str,
        log_path: Path | None = None,
        error_summary: str = "",
    ) -> Path:
        """Write a handoff file for a failed issue.

        Args:
            issue_id: The issue ID that failed.
            log_path: Optional path to the JSONL log file.
            error_summary: Summary of the error/failure reason.

        Returns:
            Path to the created handoff file.
        """
        # Ensure handoff directory exists
        self.handoff_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize issue ID for filename
        safe_id = sanitize_issue_id(issue_id)
        handoff_file = self.handoff_dir / f"{safe_id}.md"

        # Extract commands from log
        commands: list[str] = []
        last_error: str | None = None
        if log_path:
            commands = self._extract_commands_from_log(log_path)
            last_error = self._extract_last_error(log_path)

        # Limit to last N commands
        if len(commands) > self.MAX_COMMANDS:
            commands = commands[-self.MAX_COMMANDS :]

        # Build handoff content
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines = [
            f"# Handoff: {issue_id}",
            "",
            f"**Created:** {timestamp}",
            "",
            "## Error Summary",
            "",
            error_summary if error_summary else "No error summary provided.",
            "",
        ]

        # Add last tool error if present
        if last_error:
            lines.extend(
                [
                    "## Last Tool Error",
                    "",
                    "```",
                    last_error,
                    "```",
                    "",
                ]
            )

        # Add command history
        lines.extend(
            [
                "## Last Commands",
                "",
            ]
        )

        if commands:
            for cmd in commands:
                lines.append("```bash")
                lines.append(cmd)
                lines.append("```")
                lines.append("")
        else:
            lines.append("No command history available.")
            lines.append("")

        # Write file
        handoff_file.write_text("\n".join(lines))

        return handoff_file
