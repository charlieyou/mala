"""
Telemetry abstraction for agent tracing.

Provides a pluggable telemetry system with:
- TelemetryProvider protocol for abstraction
- TelemetrySpan protocol for span context managers
- NullTelemetryProvider for testing and opt-out

Usage:
    # For tests or opt-out:
    provider = NullTelemetryProvider()

    # Use via protocol:
    if provider.is_enabled():
        with provider.create_span("task-123", {"agent_id": "agent-1"}):
            # Work happens here
            pass
        provider.flush()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Self

if TYPE_CHECKING:
    from types import TracebackType


class TelemetrySpan(Protocol):
    """Protocol for a telemetry span context manager."""

    def __enter__(self) -> Self:
        """Enter the span context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the span context."""
        ...

    def log_input(self, prompt: str) -> None:
        """Log the initial user prompt."""
        ...

    def log_message(self, message: object) -> None:
        """Log a message from the SDK."""
        ...

    def set_success(self, success: bool) -> None:
        """Mark the execution as successful or failed."""
        ...

    def set_error(self, error: str) -> None:
        """Record an error message."""
        ...


class TelemetryProvider(Protocol):
    """Protocol for telemetry providers.

    Telemetry providers abstract the underlying tracing system,
    allowing tests to use a null implementation.
    """

    def is_enabled(self) -> bool:
        """Check if telemetry is active and configured."""
        ...

    def create_span(
        self, name: str, metadata: dict[str, Any] | None = None
    ) -> TelemetrySpan:
        """Create a span context manager for tracing an operation.

        Args:
            name: Span name (typically issue_id for agent executions)
            metadata: Optional metadata dict (e.g., agent_id, custom fields)

        Returns:
            A TelemetrySpan context manager for the operation
        """
        ...

    def flush(self) -> None:
        """Flush pending telemetry data."""
        ...


class NullSpan:
    """No-op span implementation for testing and opt-out.

    All methods are no-ops that return immediately.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    def log_input(self, prompt: str) -> None:
        pass

    def log_message(self, message: object) -> None:
        pass

    def set_success(self, success: bool) -> None:
        pass

    def set_error(self, error: str) -> None:
        pass


class NullTelemetryProvider:
    """No-op telemetry provider for testing and opt-out.

    This provider is completely stateless and has no side effects.
    All operations return immediately without doing anything.

    Usage:
        provider = NullTelemetryProvider()
        with provider.create_span("task-123"):
            # Work happens here - nothing is traced
            pass
    """

    def is_enabled(self) -> bool:
        """Always returns False - null provider is never 'enabled'."""
        return False

    def create_span(
        self, name: str, metadata: dict[str, Any] | None = None
    ) -> NullSpan:
        """Return a no-op span that ignores all operations."""
        return NullSpan()

    def flush(self) -> None:
        """No-op flush."""
        pass
