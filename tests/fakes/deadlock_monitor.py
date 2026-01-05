"""Fake deadlock monitor for testing."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeDeadlockMonitor:
    """In-memory deadlock monitor implementing DeadlockMonitorProtocol.

    Records events and returns configurable responses.
    """

    events: list[Any] = field(default_factory=list)
    responses: list[Any] = field(default_factory=list)
    _response_index: int = field(default=0, init=False)

    async def handle_event(self, event: Any) -> Any:  # noqa: ANN401
        """Record event and return next configured response (or None)."""
        self.events.append(event)
        if self._response_index < len(self.responses):
            response = self.responses[self._response_index]
            self._response_index += 1
            return response
        return None
