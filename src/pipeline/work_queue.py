"""Typed work queue surface for issue execution polling.

This module intentionally does not own coordinator control flow. It packages
the mutable run-loop inputs into snapshot/result objects so scheduling logic can
be tested without sleeping, clocks, or issue-provider I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

from src.core.models import OrderPreference


@dataclass(frozen=True)
class WorkQueueConfig:
    """Issue selection and execution limits for polling ready work."""

    max_agents: int | None = None
    max_issues: int | None = None
    epic_id: str | None = None
    only_ids: list[str] | None = None
    include_wip: bool = False
    focus: bool = True
    orphans_only: bool = False
    order_preference: OrderPreference = OrderPreference.EPIC_PRIORITY


@dataclass(frozen=True)
class WorkQueueSnapshot:
    """Immutable scheduler state used by pure decision functions."""

    ready_issue_ids: tuple[str, ...] = ()
    active_issue_ids: frozenset[str] = field(default_factory=frozenset)
    completed_issue_ids: frozenset[str] = field(default_factory=frozenset)
    failed_issue_ids: frozenset[str] = field(default_factory=frozenset)
    completed_count: int = 0
    last_validation_at: int = 0
    next_validation_threshold: int = 10
    consecutive_poll_failures: int = 0
    max_agents: int | None = None
    max_issues: int | None = None
    watch_enabled: bool = False
    startup_no_ready_check_pending: bool = False
    abort_requested: bool = False
    interrupt_requested: bool = False
    is_draining: bool = False

    @property
    def active_count(self) -> int:
        return len(self.active_issue_ids)

    @property
    def ready_count(self) -> int:
        return len(self.ready_issue_ids)

    @property
    def has_active_work(self) -> bool:
        return self.active_count > 0

    @property
    def issue_limit_reached(self) -> bool:
        return self.max_issues is not None and self.completed_count >= self.max_issues

    @property
    def spawn_limit_reached(self) -> bool:
        return (
            self.max_issues is not None
            and self.completed_count + self.active_count >= self.max_issues
        )

    def with_ready(
        self,
        ready_issue_ids: list[str] | tuple[str, ...],
        *,
        consecutive_poll_failures: int | None = None,
    ) -> WorkQueueSnapshot:
        """Return a copy with updated ready IDs and optional failure count."""

        return replace(
            self,
            ready_issue_ids=tuple(ready_issue_ids),
            consecutive_poll_failures=(
                self.consecutive_poll_failures
                if consecutive_poll_failures is None
                else consecutive_poll_failures
            ),
        )


@dataclass(frozen=True)
class PollResult:
    """Outcome of a work queue poll attempt."""

    snapshot: WorkQueueSnapshot
    poll_attempted: bool
    error: Exception | None = None

    @property
    def ready_issue_ids(self) -> tuple[str, ...]:
        return self.snapshot.ready_issue_ids

    @property
    def consecutive_poll_failures(self) -> int:
        return self.snapshot.consecutive_poll_failures

    @property
    def succeeded(self) -> bool:
        return self.poll_attempted and self.error is None


class PollStrategy(Protocol):
    """Owns poll waits and rate limits outside pure decision functions."""

    async def wait_after_poll_failure(
        self, snapshot: WorkQueueSnapshot, error: Exception
    ) -> None:
        """Wait before the next poll retry after a provider failure."""
        ...

    async def wait_until_next_poll(self, snapshot: WorkQueueSnapshot) -> None:
        """Wait before the next watch-mode poll when no work is ready."""
        ...


class ReadyIssueProvider(Protocol):
    """Issue-provider surface needed for work queue polling."""

    async def get_ready_async(
        self,
        exclude_ids: set[str] | None = None,
        epic_id: str | None = None,
        only_ids: list[str] | None = None,
        suppress_warn_ids: set[str] | None = None,
        include_wip: bool = False,
        focus: bool = True,
        orphans_only: bool = False,
        order_preference: OrderPreference = OrderPreference.EPIC_PRIORITY,
    ) -> list[str]:
        """Get list of ready issue IDs."""
        ...


class WorkQueue:
    """Poll ready issues and produce immutable queue snapshots."""

    def __init__(
        self,
        issue_provider: ReadyIssueProvider,
        config: WorkQueueConfig,
        poll_strategy: PollStrategy,
    ) -> None:
        self._issue_provider = issue_provider
        self._config = config
        self._poll_strategy = poll_strategy

    def snapshot(
        self,
        *,
        active_issue_ids: set[str] | frozenset[str] = frozenset(),
        completed_issue_ids: set[str] | frozenset[str] = frozenset(),
        failed_issue_ids: set[str] | frozenset[str] = frozenset(),
        ready_issue_ids: list[str] | tuple[str, ...] = (),
        completed_count: int | None = None,
        last_validation_at: int = 0,
        next_validation_threshold: int = 10,
        consecutive_poll_failures: int = 0,
        watch_enabled: bool = False,
        startup_no_ready_check_pending: bool = False,
        abort_requested: bool = False,
        interrupt_requested: bool = False,
        is_draining: bool = False,
    ) -> WorkQueueSnapshot:
        """Build a queue snapshot from coordinator state."""

        completed = frozenset(completed_issue_ids)
        return WorkQueueSnapshot(
            ready_issue_ids=tuple(ready_issue_ids),
            active_issue_ids=frozenset(active_issue_ids),
            completed_issue_ids=completed,
            failed_issue_ids=frozenset(failed_issue_ids),
            completed_count=(
                len(completed) if completed_count is None else completed_count
            ),
            last_validation_at=last_validation_at,
            next_validation_threshold=next_validation_threshold,
            consecutive_poll_failures=consecutive_poll_failures,
            max_agents=self._config.max_agents,
            max_issues=self._config.max_issues,
            watch_enabled=watch_enabled,
            startup_no_ready_check_pending=startup_no_ready_check_pending,
            abort_requested=abort_requested,
            interrupt_requested=interrupt_requested,
            is_draining=is_draining,
        )

    async def poll(self, snapshot: WorkQueueSnapshot) -> PollResult:
        """Fetch ready work unless limits or drain mode suppress polling."""

        if snapshot.issue_limit_reached or snapshot.is_draining:
            return PollResult(
                snapshot=snapshot.with_ready((), consecutive_poll_failures=0),
                poll_attempted=False,
            )

        try:
            ready_issue_ids = await self._issue_provider.get_ready_async(
                set(snapshot.failed_issue_ids | snapshot.completed_issue_ids),
                epic_id=self._config.epic_id,
                only_ids=self._config.only_ids,
                suppress_warn_ids=self._suppress_warn_ids(snapshot),
                include_wip=self._config.include_wip,
                focus=self._config.focus,
                orphans_only=self._config.orphans_only,
                order_preference=self._config.order_preference,
            )
        except Exception as error:
            failed = snapshot.with_ready(
                (), consecutive_poll_failures=snapshot.consecutive_poll_failures + 1
            )
            await self._poll_strategy.wait_after_poll_failure(failed, error)
            return PollResult(snapshot=failed, poll_attempted=True, error=error)

        polled = snapshot.with_ready(ready_issue_ids, consecutive_poll_failures=0)
        if (
            not ready_issue_ids
            and snapshot.watch_enabled
            and not snapshot.has_active_work
            and not snapshot.startup_no_ready_check_pending
        ):
            await self._poll_strategy.wait_until_next_poll(polled)
        return PollResult(snapshot=polled, poll_attempted=True)

    def _suppress_warn_ids(self, snapshot: WorkQueueSnapshot) -> set[str] | None:
        if not self._config.only_ids:
            return None
        return (
            set(snapshot.failed_issue_ids)
            | set(snapshot.active_issue_ids)
            | set(snapshot.completed_issue_ids)
        )
