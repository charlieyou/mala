"""Issue lifecycle state port.

This module defines the narrow state/effect interface used to decouple
orchestration lifecycle delegation from concrete coordinator classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import asyncio


IssueLifecycleEffectKind = Literal[
    "mark_failed",
    "mark_completed",
    "release_task",
    "reserve_issue",
    "release_issue_reservation",
    "request_abort",
    "set_max_issues",
    "set_interrupt_event",
]


@dataclass(frozen=True)
class IssueLifecycleState:
    """Dataclass snapshot of lifecycle state exposed by IssueLifecyclePort."""

    active_issue_ids: frozenset[str]
    reserved_issue_ids: frozenset[str]
    failed_issues: frozenset[str]
    abort_requested: bool
    abort_reason: str | None
    max_issues: int | None
    interrupt_requested: bool


@dataclass(frozen=True)
class IssueLifecycleEffect:
    """Dataclass mutation request accepted by IssueLifecyclePort.apply_effect."""

    kind: IssueLifecycleEffectKind
    issue_id: str | None = None
    reason: str | None = None
    max_issues: int | None = None
    interrupt_event: asyncio.Event | None = None


@runtime_checkable
class IssueLifecyclePort(Protocol):
    """Port for issue execution lifecycle state.

    The surface mirrors the orchestrator's delegated lifecycle properties:
    active tasks, failed issue IDs, abort state, interrupt event, and max issue
    limit. Mutations flow through explicit effects so callers do not need to
    know which concrete coordinator owns the state.
    """

    @property
    def current_state(self) -> IssueLifecycleState:
        """Return an immutable snapshot of lifecycle state."""
        ...

    def apply_effect(self, effect: IssueLifecycleEffect) -> None:
        """Apply a lifecycle state mutation."""
        ...

    @property
    def active_tasks(self) -> dict[str, asyncio.Task[Any]]:
        """Active tasks keyed by issue ID."""
        ...

    @property
    def failed_issues(self) -> set[str]:
        """Issue IDs that failed during the current run."""
        ...

    @property
    def abort_requested(self) -> bool:
        """Whether the current run has been asked to abort."""
        ...

    @property
    def interrupt_event(self) -> asyncio.Event | None:
        """Event set when the run should stop spawning or abort active work."""
        ...

    @interrupt_event.setter
    def interrupt_event(self, value: asyncio.Event | None) -> None:
        """Replace the interrupt event when a lifecycle owner is reset."""
        ...

    @property
    def max_issues(self) -> int | None:
        """Maximum number of issues to process, or None for unlimited."""
        ...

    @max_issues.setter
    def max_issues(self, value: int | None) -> None:
        """Update the maximum number of issues to process."""
        ...
