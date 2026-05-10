"""Fake IssueLifecyclePort for tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.core.protocols.issue_lifecycle_port import IssueLifecycleState

if TYPE_CHECKING:
    from src.core.protocols.issue_lifecycle_port import IssueLifecycleEffect


@dataclass
class FakeIssueLifecyclePort:
    """In-memory implementation of IssueLifecyclePort."""

    active_tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)
    failed_issues: set[str] = field(default_factory=set)
    abort_reason: str | None = None
    _max_issues: int | None = None
    _interrupt_event: asyncio.Event | None = field(default_factory=asyncio.Event)

    @property
    def current_state(self) -> IssueLifecycleState:
        """Return a snapshot of issue lifecycle state."""
        interrupt_requested = (
            self._interrupt_event.is_set()
            if self._interrupt_event is not None
            else False
        )
        return IssueLifecycleState(
            active_issue_ids=frozenset(self.active_tasks),
            failed_issues=frozenset(self.failed_issues),
            abort_requested=self.abort_requested,
            abort_reason=self.abort_reason,
            max_issues=self.max_issues,
            interrupt_requested=interrupt_requested,
        )

    def apply_effect(self, effect: IssueLifecycleEffect) -> None:
        """Apply an explicit lifecycle state mutation."""
        if effect.kind == "mark_failed":
            if effect.issue_id is None:
                msg = "mark_failed effect requires issue_id"
                raise ValueError(msg)
            self.failed_issues.add(effect.issue_id)
            return

        if effect.kind == "mark_completed":
            if effect.issue_id is None:
                msg = "mark_completed effect requires issue_id"
                raise ValueError(msg)
            self.active_tasks.pop(effect.issue_id, None)
            return

        if effect.kind == "release_task":
            if effect.issue_id is None:
                msg = "release_task effect requires issue_id"
                raise ValueError(msg)
            self.active_tasks.pop(effect.issue_id, None)
            return

        if effect.kind == "request_abort":
            if effect.reason is None:
                msg = "request_abort effect requires reason"
                raise ValueError(msg)
            if self.abort_requested:
                return
            self.abort_reason = effect.reason
            return

        if effect.kind == "set_max_issues":
            self.max_issues = effect.max_issues
            return

        if effect.kind == "set_interrupt_event":
            self.interrupt_event = effect.interrupt_event
            return

    @property
    def abort_requested(self) -> bool:
        """Whether abort has been requested."""
        return self.abort_reason is not None

    @property
    def interrupt_event(self) -> asyncio.Event | None:
        """Event set when abort is requested."""
        return self._interrupt_event

    @interrupt_event.setter
    def interrupt_event(self, value: asyncio.Event | None) -> None:
        if value is not None:
            self._interrupt_event = value

    @property
    def max_issues(self) -> int | None:
        """Maximum issues to process."""
        return self._max_issues

    @max_issues.setter
    def max_issues(self, value: int | None) -> None:
        self._max_issues = value
