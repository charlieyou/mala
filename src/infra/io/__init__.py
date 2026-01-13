"""I/O utilities for mala.

This package contains:
- config: MalaConfig dataclass for configuration management
- event_sink: MalaEventSink protocol and implementations
- session_log_parser: JSONL log file parsing
- log_output/: Console logging and run metadata
"""

from src.core.protocols.events import EventRunConfig, MalaEventSink
from src.infra.io.base_sink import BaseEventSink, NullEventSink
from src.infra.io.config import ConfigurationError, MalaConfig
from src.infra.io.console_sink import ConsoleEventSink
from src.infra.io.session_log_parser import (
    FileSystemLogProvider,
    JsonlEntry,
    SessionLogParser,
)

__all__ = [
    "BaseEventSink",
    "ConfigurationError",
    "ConsoleEventSink",
    "EventRunConfig",
    "FileSystemLogProvider",
    "JsonlEntry",
    "MalaConfig",
    "MalaEventSink",
    "NullEventSink",
    "SessionLogParser",
]
