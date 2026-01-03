"""Event sink protocol for MalaOrchestrator.

The protocol definition has been moved to src/core/protocols.py.
Import EventRunConfig and MalaEventSink from there.

This file is kept for module documentation. Implementations are in:
- base_sink.py: BaseEventSink, NullEventSink
- console_sink.py: ConsoleEventSink
"""

# Re-export from canonical location (this is not a compatibility shim,
# it's the public interface for infra.io module)
from src.core.protocols import EventRunConfig, MalaEventSink

__all__ = ["EventRunConfig", "MalaEventSink"]
