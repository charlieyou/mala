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

Parsing Modes:
    - parse_log_entry(): Lenient mode for production use. Returns None for
      unrecognized entries (forward compatibility).
    - parse_log_entry_strict(): Strict mode for testing/debugging. Raises
      LogParseError with detailed schema information on parse failures.
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


# Expected schema description for error messages
_SCHEMA_DESCRIPTION = """
Expected Claude Agent SDK JSONL schema:
  {"type": "assistant"|"user", "message": {"content": [<blocks>]}}

Content block types:
  - {"type": "text", "text": "<string>"}
  - {"type": "tool_use", "id": "<string>", "name": "<string>", "input": {...}}
  - {"type": "tool_result", "tool_use_id": "<string>", "content": ..., "is_error": <bool>}
""".strip()


class LogParseError(Exception):
    """Error raised when log parsing fails with schema validation error.

    Attributes:
        reason: Human-readable explanation of what was expected.
        data: The raw data that failed to parse.
        schema_hint: Reference to the expected schema format.
    """

    def __init__(self, reason: str, data: dict[str, Any] | None = None):
        self.reason = reason
        self.data = data
        self.schema_hint = _SCHEMA_DESCRIPTION
        super().__init__(f"Log parse error: {reason}\n\n{_SCHEMA_DESCRIPTION}")


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
        # Reject non-bool is_error to avoid "false" -> True misclassification
        if not isinstance(is_error, bool):
            return None
        return ToolResultBlock(
            tool_use_id=tool_use_id, content=content, is_error=is_error
        )

    # Unknown block type - ignore for forward compatibility
    return None


def _parse_content_block_strict(block: dict[str, Any], index: int) -> ContentBlock:
    """Parse a content block in strict mode, raising on errors.

    Args:
        block: Raw dict data for a content block.
        index: Index of this block in the content array (for error messages).

    Returns:
        Parsed ContentBlock.

    Raises:
        LogParseError: If the block cannot be parsed.
    """
    if not isinstance(block, dict):
        raise LogParseError(
            f"Content block at index {index} must be a dict, got {type(block).__name__}",
            data={"block": block, "index": index},
        )

    block_type = block.get("type")
    if block_type is None:
        raise LogParseError(
            f"Content block at index {index} missing required 'type' field",
            data={"block": block, "index": index},
        )

    if block_type == "text":
        text = block.get("text")
        if text is None:
            raise LogParseError(
                f"Text block at index {index} missing required 'text' field",
                data={"block": block, "index": index},
            )
        if not isinstance(text, str):
            raise LogParseError(
                f"Text block at index {index} has invalid 'text' type: "
                f"expected str, got {type(text).__name__}",
                data={"block": block, "index": index},
            )
        return TextBlock(text=text)

    if block_type == "tool_use":
        tool_id = block.get("id")
        name = block.get("name")
        tool_input = block.get("input", {})
        # Require id and name fields in strict mode
        if tool_id is None:
            raise LogParseError(
                f"tool_use block at index {index} missing required 'id' field",
                data={"block": block, "index": index},
            )
        if not isinstance(tool_id, str):
            raise LogParseError(
                f"tool_use block at index {index} has invalid 'id' type: "
                f"expected str, got {type(tool_id).__name__}",
                data={"block": block, "index": index},
            )
        if name is None:
            raise LogParseError(
                f"tool_use block at index {index} missing required 'name' field",
                data={"block": block, "index": index},
            )
        if not isinstance(name, str):
            raise LogParseError(
                f"tool_use block at index {index} has invalid 'name' type: "
                f"expected str, got {type(name).__name__}",
                data={"block": block, "index": index},
            )
        if not isinstance(tool_input, dict):
            raise LogParseError(
                f"tool_use block at index {index} has invalid 'input' type: "
                f"expected dict, got {type(tool_input).__name__}",
                data={"block": block, "index": index},
            )
        return ToolUseBlock(id=tool_id, name=name, input=tool_input)

    if block_type == "tool_result":
        tool_use_id = block.get("tool_use_id")
        content = block.get("content", "")
        is_error = block.get("is_error", False)
        # Require tool_use_id field in strict mode
        if tool_use_id is None:
            raise LogParseError(
                f"tool_result block at index {index} missing required 'tool_use_id' field",
                data={"block": block, "index": index},
            )
        if not isinstance(tool_use_id, str):
            raise LogParseError(
                f"tool_result block at index {index} has invalid 'tool_use_id' type: "
                f"expected str, got {type(tool_use_id).__name__}",
                data={"block": block, "index": index},
            )
        # Require is_error to be a proper boolean in strict mode
        if not isinstance(is_error, bool):
            raise LogParseError(
                f"tool_result block at index {index} has invalid 'is_error' type: "
                f"expected bool, got {type(is_error).__name__}",
                data={"block": block, "index": index},
            )
        return ToolResultBlock(
            tool_use_id=tool_use_id, content=content, is_error=is_error
        )

    # Unknown block type - raise in strict mode
    raise LogParseError(
        f"Unknown content block type '{block_type}' at index {index}. "
        f"Expected: text, tool_use, or tool_result",
        data={"block": block, "index": index},
    )


def parse_log_entry(data: dict[str, Any]) -> LogEntry | None:
    """Parse a raw JSONL entry dict into a typed LogEntry (lenient mode).

    This function validates the structure of JSONL log entries from Claude
    Agent SDK and returns typed objects. Unknown entry types or malformed
    entries return None (not an error) to support forward compatibility.

    For strict parsing with detailed error messages, use parse_log_entry_strict().

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

    # Return None if content field is missing (required field)
    content_data = message_data.get("content")
    if content_data is None:
        return None
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


def parse_log_entry_strict(data: dict[str, Any]) -> LogEntry:
    """Parse a raw JSONL entry dict into a typed LogEntry (strict mode).

    Unlike parse_log_entry(), this function raises LogParseError with detailed
    schema information when parsing fails. Use this for testing, debugging,
    or when you need clear error messages about schema violations.

    Args:
        data: Parsed JSON object from a JSONL line.

    Returns:
        LogEntry (AssistantLogEntry or UserLogEntry).

    Raises:
        LogParseError: If the entry doesn't match the expected schema.
            The error includes:
            - A specific reason explaining what was wrong
            - The problematic data
            - A reference to the expected schema format

    Example:
        >>> data = {"type": "invalid"}
        >>> parse_log_entry_strict(data)
        Traceback (most recent call last):
            ...
        LogParseError: Log parse error: Entry type must be 'assistant' or 'user', got 'invalid'
    """
    if not isinstance(data, dict):
        raise LogParseError(
            f"Entry must be a dict, got {type(data).__name__}",
            data=None,
        )

    entry_type = data.get("type")
    message_data = data.get("message")

    # Also check for role-based messages (alternative format)
    if entry_type is None and isinstance(message_data, dict):
        entry_type = message_data.get("role")

    if entry_type is None:
        raise LogParseError(
            "Entry missing required 'type' field. "
            "Expected top-level 'type' or 'message.role'",
            data=data,
        )

    if entry_type not in ("assistant", "user"):
        raise LogParseError(
            f"Entry type must be 'assistant' or 'user', got '{entry_type}'",
            data=data,
        )

    if message_data is None:
        raise LogParseError(
            "Entry missing required 'message' field",
            data=data,
        )

    if not isinstance(message_data, dict):
        raise LogParseError(
            f"Entry 'message' must be a dict, got {type(message_data).__name__}",
            data=data,
        )

    content_data = message_data.get("content")
    if content_data is None:
        raise LogParseError(
            "Entry 'message' missing required 'content' field",
            data=data,
        )

    if not isinstance(content_data, list):
        raise LogParseError(
            f"Entry 'message.content' must be a list, got {type(content_data).__name__}",
            data=data,
        )

    # Parse content blocks in strict mode
    content_blocks: list[ContentBlock] = []
    for i, block_data in enumerate(content_data):
        block = _parse_content_block_strict(block_data, i)
        content_blocks.append(block)

    if entry_type == "assistant":
        return AssistantLogEntry(message=AssistantMessage(content=content_blocks))
    else:
        return UserLogEntry(message=UserMessage(content=content_blocks))
