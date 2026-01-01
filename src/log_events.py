"""Backward-compatibility shim for src.log_events.

This module re-exports all public types from src.core.log_events.
New code should import directly from src.core.log_events.
"""

from src.core.log_events import (
    AssistantLogEntry,
    AssistantMessage,
    ContentBlock,
    LogEntry,
    LogParseError,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserLogEntry,
    UserMessage,
    parse_log_entry,
    parse_log_entry_strict,
)

__all__ = [
    "AssistantLogEntry",
    "AssistantMessage",
    "ContentBlock",
    "LogEntry",
    "LogParseError",
    "TextBlock",
    "ToolResultBlock",
    "ToolUseBlock",
    "UserLogEntry",
    "UserMessage",
    "parse_log_entry",
    "parse_log_entry_strict",
]
