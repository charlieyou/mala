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

__all__ = ["ConsoleEventSink", "EventRunConfig", "MalaEventSink", "NullEventSink"]
