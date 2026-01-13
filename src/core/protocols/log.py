"""Log provider protocol for SDK log abstraction.

This module defines protocols for abstracting SDK log storage and schema,
enabling testing with mock log providers and isolation from SDK log format changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@runtime_checkable
class JsonlEntryProtocol(Protocol):
    """Protocol for parsed JSONL log entries with byte offset tracking.

    Matches the shape of session_log_parser.JsonlEntry for structural typing.
    """

    data: dict[str, Any]
    """The parsed JSON object from this line."""

    entry: object | None
    """The typed LogEntry if successfully parsed, None otherwise."""

    line_len: int
    """Length of the raw line in bytes (for offset tracking)."""

    offset: int
    """Byte offset where this line started in the file."""


@runtime_checkable
class LogProvider(Protocol):
    """Protocol for abstracting SDK log storage and schema.

    Provides methods for accessing session logs without hardcoding filesystem
    paths or Claude SDK's internal log format. This enables:
    - Testing with mock log providers that return synthetic events
    - Future support for remote log storage or SDK API access
    - Isolation from SDK log format changes

    The canonical implementation is FileSystemLogProvider, which reads JSONL
    logs from the Claude SDK's ~/.claude/projects/{encoded-path}/ directory.
    Test implementations can return in-memory events for isolation.

    Methods:
        get_log_path: Get the filesystem path for a session log.
        iter_events: Iterate over typed log entries from a session.
        get_end_offset: Get the byte offset at the end of a log file.
    """

    def get_log_path(self, repo_path: Path, session_id: str) -> Path:
        """Get the filesystem path for a session's log file.

        This method computes the expected log file path based on the repo
        and session. The path may or may not exist yet.

        Args:
            repo_path: Path to the repository the session was run in.
            session_id: Claude SDK session ID (UUID from ResultMessage).

        Returns:
            Path to the JSONL log file.
        """
        ...

    def iter_events(
        self, log_path: Path, offset: int = 0
    ) -> Iterator[JsonlEntryProtocol]:
        """Iterate over parsed JSONL entries from a log file.

        Reads the file starting from the given byte offset and yields
        structured entries. This enables incremental parsing across
        retry attempts.

        Args:
            log_path: Path to the JSONL log file.
            offset: Byte offset to start reading from (default 0).

        Yields:
            JsonlEntryProtocol objects for each successfully parsed JSON line.
            The entry field contains the typed LogEntry if parsing succeeded.

        Note:
            - Lines that fail UTF-8 decoding are silently skipped
            - Empty lines are silently skipped
            - Lines that fail JSON parsing are silently skipped
            - If file doesn't exist, yields nothing
        """
        ...

    def get_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
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
        ...

    def extract_bash_commands(self, entry: JsonlEntryProtocol) -> list[tuple[str, str]]:
        """Extract Bash tool_use commands from an entry.

        Args:
            entry: A JsonlEntryProtocol from iter_events.

        Returns:
            List of (tool_id, command) tuples for Bash tool_use blocks.
            Returns empty list if entry is not an assistant message.
        """
        ...

    def extract_tool_results(self, entry: JsonlEntryProtocol) -> list[tuple[str, bool]]:
        """Extract tool_result entries from an entry.

        Args:
            entry: A JsonlEntryProtocol from iter_events.

        Returns:
            List of (tool_use_id, is_error) tuples for tool_result blocks.
            Returns empty list if entry is not a user message.
        """
        ...

    def extract_assistant_text_blocks(self, entry: JsonlEntryProtocol) -> list[str]:
        """Extract text content from assistant message blocks.

        Args:
            entry: A JsonlEntryProtocol from iter_events.

        Returns:
            List of text strings from text blocks in assistant messages.
            Returns empty list if entry is not an assistant message.
        """
        ...

    def extract_tool_result_content(
        self, entry: JsonlEntryProtocol
    ) -> list[tuple[str, str]]:
        """Extract textual content from all tool_result blocks.

        Used for marker detection in custom validation commands. Returns content
        from all tool results (not just Bash) since custom command markers have
        a specific format `[custom:<name>:<status>]` that avoids false positives.

        Args:
            entry: A JsonlEntryProtocol from iter_events.

        Returns:
            List of (tool_use_id, content) tuples for all tool_result blocks.
            Content is normalized to string with text extracted from content blocks.
            Returns empty list if entry is not a user message.
        """
        ...
