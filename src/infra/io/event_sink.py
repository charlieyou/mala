"""Event sink implementations for MalaOrchestrator.

Provides concrete implementations of the MalaEventSink protocol:
- BaseEventSink: Base class with no-op implementations
- NullEventSink: Silent sink for testing
- ConsoleEventSink: Full console output implementation
"""

from .base_sink import BaseEventSink, NullEventSink
from .console_sink import ConsoleEventSink

__all__ = [
    "BaseEventSink",
    "ConsoleEventSink",
    "NullEventSink",
]
