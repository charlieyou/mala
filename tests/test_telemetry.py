"""Tests for telemetry provider abstraction."""

from types import TracebackType
from typing import Any, Self

import pytest

import src.infra.clients.braintrust_integration as infra_braintrust
from src.infra.clients.braintrust_integration import (
    BraintrustProvider,
    BraintrustSpan,
)
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


class DummySpan:
    """Test double for Braintrust span."""

    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.logged: dict[str, Any] | None = None

    def __enter__(self) -> Self:
        self.entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.exited = True

    def log(self, **kwargs: object) -> None:
        self.logged = dict(kwargs)


class TestBraintrustProvider:
    """Tests for BraintrustProvider."""

    def test_is_enabled_delegates_to_braintrust_integration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(infra_braintrust, "BRAINTRUST_AVAILABLE", True)
        monkeypatch.setenv("BRAINTRUST_API_KEY", "test-key")

        provider = BraintrustProvider()
        assert provider.is_enabled() is True

    def test_is_enabled_false_without_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(infra_braintrust, "BRAINTRUST_AVAILABLE", True)
        monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)

        provider = BraintrustProvider()
        assert provider.is_enabled() is False

    def test_is_enabled_false_without_braintrust(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(infra_braintrust, "BRAINTRUST_AVAILABLE", False)

        provider = BraintrustProvider()
        assert provider.is_enabled() is False

    def test_create_span_returns_braintrust_span(self) -> None:
        provider = BraintrustProvider()
        span = provider.create_span("task-123", {"agent_id": "agent-1"})
        assert isinstance(span, BraintrustSpan)

    def test_create_span_extracts_agent_id_from_metadata(self) -> None:
        provider = BraintrustProvider()
        span = provider.create_span(
            "task-123", {"agent_id": "my-agent", "custom": "value"}
        )
        # Verify the span was created with correct agent_id
        assert span._tracer.agent_id == "my-agent"
        assert span._tracer.issue_id == "task-123"
        # Custom metadata should be passed through
        assert span._tracer.metadata == {"custom": "value"}

    def test_create_span_uses_default_agent_id(self) -> None:
        provider = BraintrustProvider()
        span = provider.create_span("task-123")
        assert span._tracer.agent_id == "unknown"

    def test_flush_delegates_to_braintrust(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flush_calls = {"count": 0}

        def _flush() -> None:
            flush_calls["count"] += 1

        monkeypatch.setattr(infra_braintrust, "BRAINTRUST_AVAILABLE", True)
        monkeypatch.setattr(infra_braintrust, "flush", _flush)

        provider = BraintrustProvider()
        provider.flush()

        assert flush_calls["count"] == 1


class TestBraintrustSpan:
    """Tests for BraintrustSpan adapter."""

    def test_enter_exit_delegates_to_tracer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dummy_span = DummySpan()

        def _start_span(**_kwargs: object) -> DummySpan:
            return dummy_span

        def _flush() -> None:
            pass

        monkeypatch.setattr(infra_braintrust, "BRAINTRUST_AVAILABLE", True)
        monkeypatch.setattr(infra_braintrust, "start_span", _start_span)
        monkeypatch.setattr(infra_braintrust, "flush", _flush)
        monkeypatch.setenv("BRAINTRUST_API_KEY", "test-key")

        span = BraintrustSpan("issue-1", "agent-1")
        with span:
            assert dummy_span.entered is True

        assert dummy_span.exited is True

    def test_log_input_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(infra_braintrust, "BRAINTRUST_AVAILABLE", False)

        span = BraintrustSpan("issue-1", "agent-1")
        span.log_input("test prompt")
        assert span._tracer.input_prompt == "test prompt"

    def test_set_success_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(infra_braintrust, "BRAINTRUST_AVAILABLE", False)

        span = BraintrustSpan("issue-1", "agent-1")
        span.set_success(True)
        assert span._tracer.success is True

    def test_set_error_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(infra_braintrust, "BRAINTRUST_AVAILABLE", False)

        span = BraintrustSpan("issue-1", "agent-1")
        span.set_error("boom")
        assert span._tracer.error == "boom"
        assert span._tracer.success is False


class TestProtocolCompliance:
    """Verify implementations match the protocol contracts."""

    def test_null_provider_matches_protocol(self) -> None:
        """NullTelemetryProvider should match TelemetryProvider protocol."""
        provider = NullTelemetryProvider()

        # Should have all required methods
        assert hasattr(provider, "is_enabled")
        assert hasattr(provider, "create_span")
        assert hasattr(provider, "flush")

        # Methods should be callable
        assert callable(provider.is_enabled)
        assert callable(provider.create_span)
        assert callable(provider.flush)

    def test_braintrust_provider_matches_protocol(self) -> None:
        """BraintrustProvider should match TelemetryProvider protocol."""
        provider = BraintrustProvider()

        # Should have all required methods
        assert hasattr(provider, "is_enabled")
        assert hasattr(provider, "create_span")
        assert hasattr(provider, "flush")

        # Methods should be callable
        assert callable(provider.is_enabled)
        assert callable(provider.create_span)
        assert callable(provider.flush)

    def test_null_span_matches_protocol(self) -> None:
        """NullSpan should match TelemetrySpan protocol."""
        span = NullSpan()

        # Should have all required methods
        assert hasattr(span, "__enter__")
        assert hasattr(span, "__exit__")
        assert hasattr(span, "log_input")
        assert hasattr(span, "log_message")
        assert hasattr(span, "set_success")
        assert hasattr(span, "set_error")

    def test_braintrust_span_matches_protocol(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BraintrustSpan should match TelemetrySpan protocol."""
        monkeypatch.setattr(infra_braintrust, "BRAINTRUST_AVAILABLE", False)

        span = BraintrustSpan("issue-1", "agent-1")

        # Should have all required methods
        assert hasattr(span, "__enter__")
        assert hasattr(span, "__exit__")
        assert hasattr(span, "log_input")
        assert hasattr(span, "log_message")
        assert hasattr(span, "set_success")
        assert hasattr(span, "set_error")
