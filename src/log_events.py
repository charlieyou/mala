"""JSONL log event types for Claude Agent SDK schema contract.

This module defines explicit types for the JSONL log format produced by
Claude Agent SDK. These types serve as a contract between mala and the SDK,
enabling validation and clearer parsing.

Schema Overview:
    Log entries have a top-level "type" field that determines message direction:
    - "assistant": Messages from the assistant (tool_use, text blocks)
    - "user": Messages to the assistant (tool_result blocks)

    Message content is a list of blocks, each with a "type" field:
    - "tool_use": Tool invocation with name, id, input
    - "tool_result": Tool output with tool_use_id, content, is_error
    - "text": Plain text content

Example JSONL entries:

    Assistant message with tool_use:
    {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "toolu_123", "name": "Bash", "input": {"command": "ls"}}
    ]}}

    User message with tool_result:
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "toolu_123", "content": "file.txt", "is_error": false}
    ]}}

    Assistant message with text:
    {"type": "assistant", "message": {"content": [
        {"type": "text", "text": "Here are the files..."}
    ]}}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextBlock:
    """A text content block in a message.

    Attributes:
        text: The text content.
    """

    text: str


@dataclass(frozen=True)
class ToolUseBlock:
    """A tool_use block representing a tool invocation.

    Attributes:
        id: Unique identifier for this tool use (used to correlate with tool_result).
        name: Name of the tool being invoked (e.g., "Bash", "Read", "Write").
        input: Tool-specific input parameters (e.g., {"command": "ls"} for Bash).
    """

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResultBlock:
    """A tool_result block representing tool output.

    Attributes:
        tool_use_id: ID of the tool_use this is a response to.
        content: The tool output content (usually a string, but can be structured).
        is_error: Whether the tool execution resulted in an error.
    """

    tool_use_id: str
    content: Any
    is_error: bool


# Type alias for content blocks
ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


@dataclass(frozen=True)
class AssistantMessage:
    """An assistant message containing content blocks.

    Attributes:
        content: List of content blocks (text, tool_use).
    """

    content: list[ContentBlock]


@dataclass(frozen=True)
class UserMessage:
    """A user message containing content blocks.

    Attributes:
        content: List of content blocks (typically tool_result).
    """

    content: list[ContentBlock]


@dataclass(frozen=True)
class AssistantLogEntry:
    """A log entry from the assistant.

    Attributes:
        message: The assistant message.
    """

    message: AssistantMessage


@dataclass(frozen=True)
class UserLogEntry:
    """A log entry from the user (typically tool results).

    Attributes:
        message: The user message.
    """

    message: UserMessage


# Type alias for all log entry types
LogEntry = AssistantLogEntry | UserLogEntry


class LogParseError(Exception):
    """Error raised when log parsing fails with schema validation error.

    Attributes:
        reason: Human-readable explanation of what was expected.
        data: The raw data that failed to parse.
    """

    def __init__(self, reason: str, data: dict[str, Any] | None = None):
        self.reason = reason
        self.data = data
        super().__init__(f"Log parse error: {reason}")


def _parse_content_block(block: dict[str, Any]) -> ContentBlock | None:
    """Parse a content block from raw dict data.

    Args:
        block: Raw dict data for a content block.

    Returns:
        Parsed ContentBlock or None if the block type is unrecognized.
        Unknown block types are silently ignored for forward compatibility.
    """
    if not isinstance(block, dict):
        return None

    block_type = block.get("type")

    if block_type == "text":
        text = block.get("text", "")
        if not isinstance(text, str):
            return None
        return TextBlock(text=text)

    if block_type == "tool_use":
        tool_id = block.get("id", "")
        name = block.get("name", "")
        tool_input = block.get("input", {})
        if not isinstance(tool_id, str) or not isinstance(name, str):
            return None
        if not isinstance(tool_input, dict):
            tool_input = {}
        return ToolUseBlock(id=tool_id, name=name, input=tool_input)

    if block_type == "tool_result":
        tool_use_id = block.get("tool_use_id", "")
        content = block.get("content", "")
        is_error = block.get("is_error", False)
        if not isinstance(tool_use_id, str):
            return None
        if not isinstance(is_error, bool):
            is_error = bool(is_error)
        return ToolResultBlock(
            tool_use_id=tool_use_id, content=content, is_error=is_error
        )

    # Unknown block type - ignore for forward compatibility
    return None


def parse_log_entry(data: dict[str, Any]) -> LogEntry | None:
    """Parse a raw JSONL entry dict into a typed LogEntry.

    This function validates the structure of JSONL log entries from Claude
    Agent SDK and returns typed objects. Unknown entry types or malformed
    entries return None (not an error) to support forward compatibility.

    Args:
        data: Parsed JSON object from a JSONL line.

    Returns:
        LogEntry (AssistantLogEntry or UserLogEntry) if the entry matches
        expected schema, None if the entry type is unrecognized or the
        structure is invalid.

    Note:
        - Unknown fields are ignored (forward compatibility)
        - Unknown block types within content are skipped
        - Empty content arrays are valid

    Example:
        >>> data = {"type": "assistant", "message": {"content": [
        ...     {"type": "text", "text": "Hello"}
        ... ]}}
        >>> entry = parse_log_entry(data)
        >>> isinstance(entry, AssistantLogEntry)
        True
    """
    if not isinstance(data, dict):
        return None

    entry_type = data.get("type")
    message_data = data.get("message")

    # Also check for role-based messages (alternative format)
    # Some entries use message.role instead of top-level type
    if entry_type is None and isinstance(message_data, dict):
        entry_type = message_data.get("role")

    if entry_type not in ("assistant", "user"):
        return None

    if not isinstance(message_data, dict):
        return None

    content_data = message_data.get("content", [])
    if not isinstance(content_data, list):
        return None

    # Parse content blocks, filtering out unrecognized ones
    content_blocks: list[ContentBlock] = []
    for block_data in content_data:
        block = _parse_content_block(block_data)
        if block is not None:
            content_blocks.append(block)

    if entry_type == "assistant":
        return AssistantLogEntry(message=AssistantMessage(content=content_blocks))
    else:
        return UserLogEntry(message=UserMessage(content=content_blocks))
