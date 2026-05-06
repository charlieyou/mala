"""Synthetic Anthropic-shaped message dataclasses for the Amp adapter.

`MessageStreamProcessor` (`src/pipeline/message_stream_processor.py`) keys off
`type(message).__name__` and reads fields via `getattr`. These dataclasses use
the same class names and field names as the Claude SDK message types, so the
processor can consume Amp output unchanged. The Amp adapter constructs these
objects from parsed stream-json events; the processor never imports them
directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResultBlock:
    tool_use_id: str
    is_error: bool = False
    content: object = None


@dataclass(frozen=True)
class AssistantMessage:
    content: list[object] = field(default_factory=list)


@dataclass(frozen=True)
class ResultMessage:
    session_id: str
    result: object = None
    subtype: str = ""
    is_error: bool = False
