"""Contract test for FakeEventSink protocol completeness.

Ensures FakeEventSink implements all methods of MalaEventSink protocol.
"""

import inspect

import pytest

from src.core.protocols import MalaEventSink
from tests.fakes.event_sink import FakeEventSink


@pytest.mark.unit
def test_fake_event_sink_implements_all_protocol_methods() -> None:
    """FakeEventSink must implement all public methods of MalaEventSink."""
    protocol_methods = {
        name
        for name, _ in inspect.getmembers(MalaEventSink, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    fake_methods = {
        name
        for name, _ in inspect.getmembers(FakeEventSink, predicate=inspect.isfunction)
        if not name.startswith("_")
    }

    missing = protocol_methods - fake_methods
    assert not missing, f"FakeEventSink missing protocol methods: {sorted(missing)}"


@pytest.mark.unit
def test_fake_event_sink_has_event_helper() -> None:
    """has_event() correctly identifies recorded events."""
    sink = FakeEventSink()

    assert not sink.has_event("run_started")

    sink.on_run_started(config=None)  # type: ignore[arg-type]
    assert sink.has_event("run_started")
    assert not sink.has_event("run_completed")


@pytest.mark.unit
def test_fake_event_sink_get_events_helper() -> None:
    """get_events() returns all events of a given type."""
    sink = FakeEventSink()

    sink.on_agent_started(agent_id="a1", issue_id="i1")
    sink.on_agent_started(agent_id="a2", issue_id="i2")
    sink.on_agent_completed(
        agent_id="a1", issue_id="i1", success=True, duration_seconds=1.0, summary="ok"
    )

    events = sink.get_events("agent_started")
    assert len(events) == 2
    assert events[0].kwargs["agent_id"] == "a1"
    assert events[1].kwargs["agent_id"] == "a2"


@pytest.mark.unit
def test_fake_event_sink_clear() -> None:
    """clear() removes all recorded events."""
    sink = FakeEventSink()

    sink.on_run_started(config=None)  # type: ignore[arg-type]
    sink.on_warning(message="test")
    assert len(sink.events) == 2

    sink.clear()
    assert len(sink.events) == 0
    assert not sink.has_event("run_started")
