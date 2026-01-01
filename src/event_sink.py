"""Backward-compatibility shim for src.event_sink.

This module re-exports all public symbols from src.infra.io.event_sink.
New code should import directly from src.infra.io.event_sink.
"""

from src.infra.io.event_sink import (
    ConsoleEventSink,
    EventRunConfig,
    MalaEventSink,
    NullEventSink,
)
from src.infra.io.log_output.console import (
    Colors,
    log,
    log_agent_text,
    log_tool,
    log_verbose,
    truncate_text,
)

__all__ = [
    "Colors",
    "ConsoleEventSink",
    "EventRunConfig",
    "MalaEventSink",
    "NullEventSink",
    "log",
    "log_agent_text",
    "log_tool",
    "log_verbose",
    "truncate_text",
]
