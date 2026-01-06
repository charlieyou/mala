"""Tests for telemetry provider abstraction."""

from src.infra.telemetry import (
    NullSpan,
    NullTelemetryProvider,
)


class TestNullSpan:
    """Tests for NullSpan no-op implementation."""

    def test_enter_returns_self(self) -> None:
        span = NullSpan()
        with span as entered:
            assert entered is span

    def test_log_input_is_noop(self) -> None:
        span = NullSpan()
        # Should not raise
        span.log_input("test prompt")

    def test_log_message_is_noop(self) -> None:
        span = NullSpan()
        # Should not raise
        span.log_message({"some": "message"})

    def test_set_success_is_noop(self) -> None:
        span = NullSpan()
        # Should not raise
        span.set_success(True)
        span.set_success(False)

    def test_set_error_is_noop(self) -> None:
        span = NullSpan()
        # Should not raise
        span.set_error("some error")


class TestNullTelemetryProvider:
    """Tests for NullTelemetryProvider."""

    def test_is_enabled_returns_false(self) -> None:
        provider = NullTelemetryProvider()
        assert provider.is_enabled() is False

    def test_create_span_returns_null_span(self) -> None:
        provider = NullTelemetryProvider()
        span = provider.create_span("test-task")
        assert isinstance(span, NullSpan)

    def test_create_span_with_metadata(self) -> None:
        provider = NullTelemetryProvider()
        span = provider.create_span(
            "test-task", {"agent_id": "agent-1", "custom": "value"}
        )
        assert isinstance(span, NullSpan)

    def test_flush_is_noop(self) -> None:
        provider = NullTelemetryProvider()
        # Should not raise
        provider.flush()

    def test_full_workflow_no_side_effects(self) -> None:
        """Test complete workflow produces no side effects."""
        provider = NullTelemetryProvider()

        assert provider.is_enabled() is False

        with provider.create_span("task-123", {"agent_id": "agent-1"}) as span:
            span.log_input("test prompt")
            span.log_message({"content": "test"})
            span.set_success(True)

        provider.flush()
        # If we get here without error, no side effects occurred
