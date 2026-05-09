"""Unit tests for the cross-coder ``AgentEvent`` protocol.

Pin the discriminator values, dataclass field shapes, and
``runtime_checkable`` protocol conformance so that adapter implementations
(Claude via :mod:`src.infra.sdk_adapter`, Amp via
:mod:`src.infra.clients.amp_client`) cannot drift from the contract
``MessageStreamProcessor`` consumes.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.core.protocols.agent_event import (
    AgentEvent,
    AgentResultEvent,
    AgentTextEvent,
    AgentToolResultEvent,
    AgentToolUseEvent,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cls", "expected_kind"),
    [
        (AgentTextEvent, "text"),
        (AgentToolUseEvent, "tool_use"),
        (AgentToolResultEvent, "tool_result"),
        (AgentResultEvent, "result"),
    ],
)
def test_kind_discriminator_default(cls: type, expected_kind: str) -> None:
    """Each event type pins ``kind`` to its discriminator string."""
    instance = cls()
    assert getattr(instance, "kind") == expected_kind


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cls", "expected_fields"),
    [
        (AgentTextEvent, {"kind", "text"}),
        (AgentToolUseEvent, {"kind", "id", "name", "input"}),
        (AgentToolResultEvent, {"kind", "tool_use_id", "is_error", "content"}),
        (AgentResultEvent, {"kind", "session_id", "is_error", "subtype", "result"}),
    ],
)
def test_dataclass_fields_match_protocol(cls: type, expected_fields: set[str]) -> None:
    """Fields read by the processor / fixer must exist by exact name."""
    actual = {f.name for f in dataclasses.fields(cls)}
    assert actual == expected_fields


@pytest.mark.unit
def test_events_are_frozen() -> None:
    """Events are frozen so consumers can rely on immutability."""
    event = AgentTextEvent(text="hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.text = "bye"  # type: ignore[misc]  # ty:ignore[invalid-assignment]


@pytest.mark.unit
@pytest.mark.parametrize(
    "instance",
    [
        AgentTextEvent(text="hi"),
        AgentToolUseEvent(id="t1", name="Bash", input={"command": "ls"}),
        AgentToolResultEvent(tool_use_id="t1", is_error=False, content="ok"),
        AgentResultEvent(
            session_id="sess-1", is_error=False, subtype="result", result="done"
        ),
    ],
)
def test_runtime_checkable_protocol(instance: object) -> None:
    """Each concrete event satisfies the ``AgentEvent`` runtime protocol."""
    assert isinstance(instance, AgentEvent)


@pytest.mark.unit
def test_agent_tool_use_event_default_input_is_independent() -> None:
    """``input`` defaults are per-instance — no shared mutable state."""
    a = AgentToolUseEvent()
    b = AgentToolUseEvent()
    # Frozen dataclass; we can't mutate the field, but we should still get
    # distinct dict instances per default invocation.
    assert a.input is not b.input
