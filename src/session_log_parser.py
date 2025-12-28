"""Session log parser for JSONL log files from Claude Agent SDK.

This module provides SessionLogParser, a reusable class for parsing JSONL log
files produced by Claude Agent SDK. It extracts structured information from
logs that can be used for:
- Validation evidence detection (did pytest/ruff/ty run?)
- Issue resolution marker detection (ISSUE_NO_CHANGE, ISSUE_OBSOLETE, etc.)
- Debugging and analytics

The parser uses typed log events from log_events.py to ensure type safety
and contract adherence with the Claude Agent SDK schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.log_events import (
    AssistantLogEntry,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserLogEntry,
    parse_log_entry,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from src.log_events import LogEntry


@dataclass
class JsonlEntry:
    """A parsed JSONL log entry with byte offset tracking.

    Attributes:
        data: The parsed JSON object from this line.
        entry: The typed LogEntry if successfully parsed, None otherwise.
        line_len: Length of the raw line in bytes (for offset tracking).
        offset: Byte offset where this line started in the file.
    """

    data: dict[str, Any]
    entry: LogEntry | None
    line_len: int
    offset: int


class SessionLogParser:
    """Parser for JSONL session logs from Claude Agent SDK.

    This class provides methods to iterate over and extract structured
    information from JSONL log files. It handles:
    - Byte-offset-aware iteration for incremental parsing
    - Extraction of Bash commands from tool_use blocks
    - Extraction of tool results with error status
    - Extraction of text blocks from assistant messages

    Example:
        >>> parser = SessionLogParser()
        >>> for entry in parser.iter_jsonl_entries(log_path):
        ...     commands = parser.extract_bash_commands(entry)
        ...     for tool_id, command in commands:
        ...         print(f"Command: {command}")
    """

    def iter_jsonl_entries(
        self, log_path: Path, offset: int = 0
    ) -> Iterator[JsonlEntry]:
        """Iterate over parsed JSONL entries from a log file.

        Reads the file in binary mode for accurate byte offset tracking,
        decodes each line as UTF-8, parses JSON, and yields structured entries.

        Args:
            log_path: Path to the JSONL log file.
            offset: Byte offset to start reading from (default 0).

        Yields:
            JsonlEntry objects for each successfully parsed JSON line.
            The entry field contains the typed LogEntry if parsing succeeded.

        Raises:
            OSError: If file cannot be read. Callers should handle this.

        Note:
            - Lines that fail UTF-8 decoding are silently skipped
            - Empty lines are silently skipped
            - Lines that fail JSON parsing are silently skipped
        """
        if not log_path.exists():
            return

        with open(log_path, "rb") as f:
            f.seek(offset)
            current_offset = offset

            for line_bytes in f:
                line_len = len(line_bytes)
                line_offset = current_offset
                current_offset += line_len

                try:
                    line = line_bytes.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue

                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Parse into typed LogEntry (returns None for unrecognized entries)
                entry = parse_log_entry(data)

                yield JsonlEntry(
                    data=data, entry=entry, line_len=line_len, offset=line_offset
                )

    def get_log_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        """Get the byte offset at the end of a log file.

        This is a lightweight method for getting the current file position.
        Use this when you only need the offset for retry scoping, not the
        parsed entries themselves.

        Args:
            log_path: Path to the JSONL log file.
            start_offset: Byte offset to start from (default 0).

        Returns:
            The byte offset at the end of the file, or start_offset if file
            doesn't exist or can't be read.
        """
        if not log_path.exists():
            return start_offset

        try:
            with open(log_path, "rb") as f:
                f.seek(0, 2)  # Seek to end
                return f.tell()
        except OSError:
            return start_offset

    def extract_bash_commands(self, entry: JsonlEntry) -> list[tuple[str, str]]:
        """Extract Bash tool_use commands from an entry.

        Args:
            entry: A JsonlEntry from iter_jsonl_entries.

        Returns:
            List of (tool_id, command) tuples for Bash tool_use blocks.
            Returns empty list if entry is not an assistant message.
        """
        # Use typed entry if available
        if entry.entry is not None:
            if not isinstance(entry.entry, AssistantLogEntry):
                return []
            commands = []
            for block in entry.entry.message.content:
                if isinstance(block, ToolUseBlock) and block.name.lower() == "bash":
                    command = block.input.get("command", "")
                    commands.append((block.id, command))
            return commands

        # Fallback to raw data parsing for backward compatibility
        return self._extract_bash_commands_from_data(entry.data)

    def _extract_bash_commands_from_data(
        self, data: dict[str, Any]
    ) -> list[tuple[str, str]]:
        """Extract Bash commands from raw entry data (fallback method).

        Args:
            data: Parsed JSONL entry data.

        Returns:
            List of (tool_id, command) tuples for Bash tool_use blocks.
        """
        if data.get("type") != "assistant":
            return []

        commands = []
        message = data.get("message", {})
        for block in message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_name = block.get("name", "")
                if tool_name.lower() == "bash":
                    tool_id = block.get("id", "")
                    command = block.get("input", {}).get("command", "")
                    commands.append((tool_id, command))
        return commands

    def extract_tool_results(self, entry: JsonlEntry) -> list[tuple[str, bool]]:
        """Extract tool_result entries from an entry.

        Args:
            entry: A JsonlEntry from iter_jsonl_entries.

        Returns:
            List of (tool_use_id, is_error) tuples for tool_result blocks.
            Returns empty list if entry is not a user message.
        """
        # Use typed entry if available
        if entry.entry is not None:
            if not isinstance(entry.entry, UserLogEntry):
                return []
            results = []
            for block in entry.entry.message.content:
                if isinstance(block, ToolResultBlock):
                    results.append((block.tool_use_id, block.is_error))
            return results

        # Fallback to raw data parsing for backward compatibility
        return self._extract_tool_results_from_data(entry.data)

    def _extract_tool_results_from_data(
        self, data: dict[str, Any]
    ) -> list[tuple[str, bool]]:
        """Extract tool results from raw entry data (fallback method).

        Args:
            data: Parsed JSONL entry data.

        Returns:
            List of (tool_use_id, is_error) tuples for tool_result blocks.
        """
        if data.get("type") != "user":
            return []

        results = []
        message = data.get("message", {})
        for block in message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                is_error = block.get("is_error", False)
                results.append((tool_use_id, is_error))
        return results

    def extract_assistant_text_blocks(self, entry: JsonlEntry) -> list[str]:
        """Extract text content from assistant message blocks.

        Args:
            entry: A JsonlEntry from iter_jsonl_entries.

        Returns:
            List of text strings from text blocks in assistant messages.
            Returns empty list if entry is not an assistant message.
        """
        # Use typed entry if available
        if entry.entry is not None:
            if not isinstance(entry.entry, AssistantLogEntry):
                return []
            texts = []
            for block in entry.entry.message.content:
                if isinstance(block, TextBlock):
                    texts.append(block.text)
            return texts

        # Fallback to raw data parsing for backward compatibility
        return self._extract_assistant_text_blocks_from_data(entry.data)

    def _extract_assistant_text_blocks_from_data(
        self, data: dict[str, Any]
    ) -> list[str]:
        """Extract text blocks from raw entry data (fallback method).

        Args:
            data: Parsed JSONL entry data.

        Returns:
            List of text strings from text blocks in assistant messages.
        """
        entry_type = data.get("type", "")
        entry_role = data.get("message", {}).get("role", "")
        if entry_type != "assistant" and entry_role != "assistant":
            return []

        texts = []
        message = data.get("message", {})
        for block in message.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return texts
