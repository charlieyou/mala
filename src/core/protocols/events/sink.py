"""Combined MalaEventSink protocol.

This module defines the main MalaEventSink protocol that combines all
event category protocols for orchestrator event handling.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .agent_lifecycle import AgentLifecycleEvents
from .diagnostics import DiagnosticsEvents
from .gate_events import GateEvents
from .issue_lifecycle import IssueLifecycleEvents
from .run_lifecycle import RunLifecycleEvents
from .trigger_events import TriggerEvents


@runtime_checkable
class MalaEventSink(
    RunLifecycleEvents,
    AgentLifecycleEvents,
    GateEvents,
    IssueLifecycleEvents,
    DiagnosticsEvents,
    TriggerEvents,
    Protocol,
):
    """Protocol for receiving orchestrator events.

    Implementations handle presentation (console, logging, metrics) while
    the orchestrator focuses on coordination logic. Each method corresponds
    to a semantic event in the orchestration flow.

    All methods are synchronous and should be non-blocking. Implementations
    that need async behavior should queue events internally.

    This protocol combines all event categories:
    - RunLifecycleEvents: Run start/end, issues ready
    - AgentLifecycleEvents: Agent spawn/complete, SDK messaging
    - GateEvents: Quality gate and code review events
    - IssueLifecycleEvents: Issue completion, validation, fixer events
    - DiagnosticsEvents: Warnings, abort, SIGINT, epic verification
    - TriggerEvents: Trigger validation, code review, session end
    """

    pass
