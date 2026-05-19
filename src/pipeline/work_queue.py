"""Typed work queue surface for issue execution polling and scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

from src.core.models import OrderPreference

ExitReason = Literal[
    "abort",
    "interrupted",
    "drained",
    "limit_reached",
    "success",
    "poll_failed",
]

PollDecisionKind = Literal[
    "ready",
    "empty",
    "terminal_drain",
    "transient_error",
]


@dataclass(frozen=True)
class PollDecision:
    """Typed poll scheduling outcome."""

    kind: PollDecisionKind


@dataclass(frozen=True)
class CapacityDecision:
    """Typed spawn-capacity scheduling outcome."""

    kind: Literal["capacity"]
    capacity: int


@dataclass(frozen=True)
class ExitDecision:
    """Run-loop exit decision without side effects."""

    should_exit: bool
    reason: ExitReason | None = None
    exit_code: int | None = None
    run_final_validation: bool = False
    wait_for_active_work: bool = False


@dataclass(frozen=True)
class PeriodicValidationDecision:
    """Validation effect requested by scheduler state."""

    should_validate: bool
    block_for_active_work: bool = False
    threshold: int | None = None


@dataclass(frozen=True)
class InterruptDrainDecision:
    """Drain-mode effect requested by scheduler state."""

    should_drain: bool
    run_final_validation: bool = False
    exit_when_validation_passes: bool = False


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
    next_validation_threshold: int | None = None
    consecutive_poll_failures: int = 0
    max_agents: int | None = None
    max_issues: int | None = None
    watch_enabled: bool = False
    startup_no_ready_check_pending: bool = False
    follow_up_repoll_pending: bool = False
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
    decision: PollDecision
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
        next_validation_threshold: int | None = None,
        consecutive_poll_failures: int = 0,
        watch_enabled: bool = False,
        startup_no_ready_check_pending: bool = False,
        follow_up_repoll_pending: bool = False,
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
            follow_up_repoll_pending=follow_up_repoll_pending,
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
                decision=PollDecision("terminal_drain"),
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
            if failed.consecutive_poll_failures < 3:
                await self._poll_strategy.wait_after_poll_failure(failed, error)
            return PollResult(
                snapshot=failed,
                poll_attempted=True,
                decision=PollDecision("transient_error"),
                error=error,
            )

        polled = snapshot.with_ready(ready_issue_ids, consecutive_poll_failures=0)
        if (
            not ready_issue_ids
            and snapshot.watch_enabled
            and not snapshot.has_active_work
            and not snapshot.startup_no_ready_check_pending
        ):
            await self._poll_strategy.wait_until_next_poll(polled)
        return PollResult(
            snapshot=polled,
            poll_attempted=True,
            decision=PollDecision("ready" if ready_issue_ids else "empty"),
        )

    def _suppress_warn_ids(self, snapshot: WorkQueueSnapshot) -> set[str] | None:
        if not self._config.only_ids:
            return None
        return (
            set(snapshot.failed_issue_ids)
            | set(snapshot.active_issue_ids)
            | set(snapshot.completed_issue_ids)
        )


def capacity_decision(snapshot: WorkQueueSnapshot) -> CapacityDecision:
    """Return how many ready issues may be spawned from this snapshot."""

    if (
        snapshot.abort_requested
        or snapshot.interrupt_requested
        or snapshot.is_draining
        or snapshot.issue_limit_reached
        or snapshot.spawn_limit_reached
    ):
        return CapacityDecision("capacity", 0)

    spawnable_ready_count = len(
        [
            issue_id
            for issue_id in snapshot.ready_issue_ids
            if issue_id not in snapshot.active_issue_ids
        ]
    )
    agent_capacity = spawnable_ready_count
    if snapshot.max_agents is not None:
        agent_capacity = min(
            agent_capacity, snapshot.max_agents - snapshot.active_count
        )
    if snapshot.max_issues is not None:
        remaining_issues = (
            snapshot.max_issues - snapshot.completed_count - snapshot.active_count
        )
        agent_capacity = min(agent_capacity, remaining_issues)
    return CapacityDecision("capacity", max(0, agent_capacity))


def exit_reason(snapshot: WorkQueueSnapshot) -> ExitDecision:
    """Return the terminal loop decision for the current snapshot."""

    if snapshot.interrupt_requested:
        return ExitDecision(True, "interrupted", 130)
    if snapshot.abort_requested:
        return ExitDecision(True, "abort", 3)
    if snapshot.is_draining and not snapshot.has_active_work:
        return ExitDecision(
            True,
            "drained",
            0,
            run_final_validation=_has_unvalidated_completions(snapshot),
        )
    if snapshot.consecutive_poll_failures >= 3:
        return ExitDecision(
            True,
            "poll_failed",
            3,
            run_final_validation=_has_unvalidated_completions(snapshot),
            wait_for_active_work=snapshot.has_active_work,
        )
    if snapshot.has_active_work:
        return ExitDecision(False)
    if snapshot.issue_limit_reached:
        return ExitDecision(
            True,
            "limit_reached",
            0,
            run_final_validation=_has_unvalidated_completions(snapshot),
        )
    if snapshot.consecutive_poll_failures > 0:
        return ExitDecision(False)
    if snapshot.ready_issue_ids:
        return ExitDecision(False)
    if snapshot.watch_enabled or snapshot.startup_no_ready_check_pending:
        return ExitDecision(False)
    if snapshot.follow_up_repoll_pending:
        return ExitDecision(False)
    return ExitDecision(
        True,
        "success",
        0,
        run_final_validation=_has_unvalidated_completions(snapshot),
    )


def periodic_validation(
    snapshot: WorkQueueSnapshot, validate_every: int | None
) -> PeriodicValidationDecision:
    """Return whether periodic validation should run after completions."""

    if validate_every is None:
        return PeriodicValidationDecision(False)
    threshold = snapshot.next_validation_threshold or validate_every
    if snapshot.completed_count < threshold:
        return PeriodicValidationDecision(False)
    return PeriodicValidationDecision(
        True,
        block_for_active_work=snapshot.has_active_work,
        threshold=threshold,
    )


def interrupt_drain(snapshot: WorkQueueSnapshot) -> InterruptDrainDecision:
    """Return the drain-mode effect for Stage 1 interrupt handling."""

    if not snapshot.is_draining:
        return InterruptDrainDecision(False)
    if snapshot.has_active_work:
        return InterruptDrainDecision(True)
    return InterruptDrainDecision(
        True,
        run_final_validation=_has_unvalidated_completions(snapshot),
        exit_when_validation_passes=True,
    )


def next_validation_threshold(
    completed_count: int, threshold: int, interval: int
) -> int:
    """Advance a validation threshold beyond the completed count."""

    while threshold <= completed_count:
        threshold += interval
    return threshold


def _has_unvalidated_completions(snapshot: WorkQueueSnapshot) -> bool:
    return snapshot.completed_count > snapshot.last_validation_at
