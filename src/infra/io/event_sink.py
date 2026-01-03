"""Event sink implementations for MalaOrchestrator.

Provides concrete implementations of the MalaEventSink protocol:
- ConsoleEventSink: Full console output implementation

For base classes, import directly from base_sink:
- BaseEventSink: Base class with no-op implementations
- NullEventSink: Silent sink for testing
"""

from .console_sink import ConsoleEventSink

__all__ = [
    "ConsoleEventSink",
]
