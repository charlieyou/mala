"""I/O utilities for mala.

This package contains:
- config: MalaConfig dataclass for configuration management
- event_sink: MalaEventSink protocol and implementations
- session_log_parser: JSONL log file parsing
- log_output/: Console logging and run metadata
"""

from src.infra.io.config import ConfigurationError, MalaConfig
from src.infra.io.event_protocol import EventRunConfig, MalaEventSink
from src.infra.io.event_sink import (
    ConsoleEventSink,
    NullEventSink,
)
from src.infra.io.session_log_parser import (
    FileSystemLogProvider,
    JsonlEntry,
    SessionLogParser,
)

__all__ = [
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
