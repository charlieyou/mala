"""Event sink protocols for orchestrator event handling.

This module defines protocols and dataclasses for the orchestrator event system,
enabling decoupled presentation (console, logging, metrics) from coordination logic.

The MalaEventSink protocol combines all event protocol categories:
- RunLifecycleEvents: Run start/end, issue fetching
- AgentLifecycleEvents: Agent spawn/complete, SDK messaging
- GateEvents: Quality gate and code review events
- IssueLifecycleEvents: Issue completion, validation, fixer events
- DiagnosticsEvents: Warnings, abort, SIGINT, epic verification, pipeline state
- TriggerEvents: Trigger validation, code review, session end
"""

from __future__ import annotations

from .agent_lifecycle import AgentLifecycleEvents
from .dataclasses import EventRunConfig, TriggerSummary, ValidationTriggersSummary
from .diagnostics import DiagnosticsEvents
from .gate_events import GateEvents
from .issue_lifecycle import IssueLifecycleEvents
from .run_lifecycle import RunLifecycleEvents
from .sink import MalaEventSink
from .trigger_events import TriggerEvents

__all__ = [
    "AgentLifecycleEvents",
    "DiagnosticsEvents",
    "EventRunConfig",
    "GateEvents",
    "IssueLifecycleEvents",
    "MalaEventSink",
    "RunLifecycleEvents",
    "TriggerEvents",
    "TriggerSummary",
    "ValidationTriggersSummary",
]
