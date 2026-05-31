"""Issue execution coordination for MalaOrchestrator.

This module contains the IssueExecutionCoordinator which manages the main agent
spawning loop and issue lifecycle during a run.

Design principles:
- Protocol-based dependencies for testability without SDK
- Callback-based agent spawning (SDK logic stays in orchestrator)
- Clear separation: coordinator handles scheduling, orchestrator handles SDK
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, assert_never

from src.core.models import OrderPreference, RunResult, WatchState
from src.core.protocols.issue_lifecycle_port import IssueLifecycleState
from src.pipeline.work_queue import (
    WorkQueue,
    WorkQueueConfig,
    capacity_decision,
    exit_reason,
    next_validation_threshold,
    periodic_validation,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.core.models import PeriodicValidationConfig, WatchConfig
    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.issue import IssueProvider
    from src.core.protocols.issue_lifecycle_port import IssueLifecycleEffect
    from src.pipeline.work_queue import (
        ExitDecision,
        PeriodicValidationDecision,
        WorkQueueSnapshot,
    )

logger = logging.getLogger(__name__)


class _InterruptRequested(Exception):
    """Raised by poll effects when an interrupt arrives during a wait."""


class _CoordinatorPollStrategy:
    """Coordinator poll effects: retry waits and watch-idle reporting."""

    def __init__(
        self,
        *,
        beads: IssueProvider,
        event_sink: MalaEventSink,
        watch_state: WatchState,
        watch_config: WatchConfig | None,
        interrupt_event: asyncio.Event | None,
        sleep_fn: Callable[[float], Awaitable[None]],
    ) -> None:
        self._beads = beads
        self._event_sink = event_sink
        self._watch_state = watch_state
        self._watch_config = watch_config
        self._interrupt_event = interrupt_event
        self._sleep_fn = sleep_fn
        self._idle_waited = False

    async def wait_after_poll_failure(
        self, snapshot: WorkQueueSnapshot, error: Exception
    ) -> None:
        """Wait before retrying a failed poll."""

        await self._interruptible_wait(self._poll_interval())

    async def wait_until_next_poll(self, snapshot: WorkQueueSnapshot) -> None:
        """Emit watch-idle events and wait until the next poll."""

        await self._emit_idle_event_if_needed()
        await self._interruptible_wait(self._poll_interval())
        self._idle_waited = True

    def clear_idle(self) -> None:
        """Record that the coordinator is no longer idle."""

        self._watch_state.was_idle = False

    def consume_idle_waited(self) -> bool:
        """Return whether the current poll already performed the idle wait."""

        idle_waited = self._idle_waited
        self._idle_waited = False
        return idle_waited

    async def _emit_idle_event_if_needed(self) -> None:
        now = time.monotonic()
        should_emit = not self._watch_state.was_idle
        if self._watch_state.was_idle:
            should_emit = now - self._watch_state.last_idle_log_time >= 300
        if not should_emit:
            return

        self._watch_state.was_idle = True
        self._watch_state.last_idle_log_time = now
        blocked_count = await self._beads.get_blocked_count_async()
        self._event_sink.on_watch_idle(self._poll_interval(), blocked_count)

    async def _interruptible_wait(self, seconds: float) -> None:
        if self._interrupt_event is None:
            await self._sleep_fn(seconds)
            return
        try:
            await asyncio.wait_for(self._interrupt_event.wait(), timeout=seconds)
        except TimeoutError:
            return
        raise _InterruptRequested

    def _poll_interval(self) -> float:
        if self._watch_config is None:
            return 60.0
        return self._watch_config.poll_interval_seconds


class SpawnCallback(Protocol):
    """Callback for spawning an agent for an issue.

    Returns the spawned Task on success, or None if spawn failed.
    The coordinator automatically registers the returned task.
    """

    async def __call__(self, issue_id: str) -> asyncio.Task[Any] | None:
        """Spawn an agent for the given issue."""
        ...


class FinalizeCallback(Protocol):
    """Callback for finalizing an issue result.

    Takes the issue_id and the task that completed.
    """

    async def __call__(self, issue_id: str, task: asyncio.Task[Any]) -> bool | None:
        """Finalize the result of a completed task."""
        ...


@dataclass
class AbortResult:
    """Result of aborting active tasks.

    Attributes:
        aborted_count: Number of tasks that were aborted.
        has_unresponsive_tasks: True if any tasks didn't respond to cancellation
            within the grace period and may still be running.
    """

    aborted_count: int
    has_unresponsive_tasks: bool = False


class AbortCallback(Protocol):
    """Callback for aborting active tasks."""

    async def __call__(self, *, is_interrupt: bool = False) -> AbortResult:
        """Abort all active tasks.

        Args:
            is_interrupt: If True, use "Interrupted" summary instead of "Aborted".

        Returns:
            AbortResult with aborted count and unresponsive task flag.
        """
        ...


@dataclass
class CoordinatorConfig:
    """Configuration for IssueExecutionCoordinator.

    Attributes:
        max_agents: Maximum concurrent agents (None = unlimited).
        max_issues: Maximum issues to process (None = unlimited).
        epic_id: Only process tasks under this epic.
        only_ids: List of issue IDs to process exclusively.
        include_wip: Include in_progress issues in scope (no ordering changes).
        focus: Legacy flag for epic grouping (use order_preference instead).
        orphans_only: Only process issues with no parent epic.
        order_preference: Issue ordering (focus, epic-priority, issue-priority, or input).
        prioritize_wip: Process in_progress issues before ready issues.
    """

    max_agents: int | None = None
    max_issues: int | None = None
    epic_id: str | None = None
    only_ids: list[str] | None = None
    include_wip: bool = False
    focus: bool = True
    orphans_only: bool = False
    order_preference: OrderPreference = OrderPreference.EPIC_PRIORITY
    prioritize_wip: bool = False


class IssueExecutionCoordinator:
    """Coordinates issue execution without SDK dependencies.

    This class manages the main agent spawning loop, handling:
    - Issue fetching from IssueProvider
    - Concurrent agent limiting (max_agents)
    - Issue count limiting (max_issues)
    - Epic and only_ids filtering
    - WIP inclusion and focus grouping

    The coordinator uses callbacks for agent spawning and finalization,
    keeping SDK-specific logic in the orchestrator.

    Example:
        coordinator = IssueExecutionCoordinator(
            beads=beads_client,
            event_sink=console_sink,
            config=CoordinatorConfig(max_agents=2),
        )
        issues_spawned = await coordinator.run_loop(
            spawn_callback=orchestrator.spawn_agent,
            finalize_callback=orchestrator._finalize_issue_result,
            abort_callback=orchestrator._abort_active_tasks,
        )
    """

    def __init__(
        self,
        beads: IssueProvider,
        event_sink: MalaEventSink,
        config: CoordinatorConfig,
    ) -> None:
        """Initialize the coordinator.

        Args:
            beads: Issue provider for fetching ready issues.
            event_sink: Event sink for logging lifecycle events.
            config: Coordinator configuration.
        """
        self.beads = beads
        self.event_sink = event_sink
        self.config = config

        # Runtime state
        self.active_tasks: dict[str, asyncio.Task[Any]] = {}
        self.completed_ids: set[str] = set()
        self.failed_issues: set[str] = set()
        self.abort_run: bool = False
        self.abort_reason: str | None = None
        self.abort_event: asyncio.Event = asyncio.Event()
        self._interrupt_event: asyncio.Event = asyncio.Event()

    @property
    def current_state(self) -> IssueLifecycleState:
        """Return a snapshot of issue lifecycle state."""
        return IssueLifecycleState(
            active_issue_ids=frozenset(self.active_tasks),
            failed_issues=frozenset(self.failed_issues),
            abort_requested=self.abort_run,
            abort_reason=self.abort_reason,
            max_issues=self.config.max_issues,
            interrupt_requested=self._interrupt_event.is_set(),
        )

    def apply_effect(self, effect: IssueLifecycleEffect) -> None:
        """Apply an explicit lifecycle state mutation."""
        if effect.kind == "mark_failed":
            if effect.issue_id is None:
                msg = "mark_failed effect requires issue_id"
                raise ValueError(msg)
            self.mark_failed(effect.issue_id)
            return

        if effect.kind == "mark_completed":
            if effect.issue_id is None:
                msg = "mark_completed effect requires issue_id"
                raise ValueError(msg)
            self.mark_completed(effect.issue_id)
            return

        if effect.kind == "release_task":
            if effect.issue_id is None:
                msg = "release_task effect requires issue_id"
                raise ValueError(msg)
            self.release_task(effect.issue_id)
            return

        if effect.kind == "request_abort":
            if effect.reason is None:
                msg = "request_abort effect requires reason"
                raise ValueError(msg)
            self.request_abort(effect.reason)
            return

        if effect.kind == "set_max_issues":
            self.max_issues = effect.max_issues
            return

        if effect.kind == "set_interrupt_event":
            self.interrupt_event = effect.interrupt_event
            return

    @property
    def abort_requested(self) -> bool:
        """Whether the current run has been asked to abort."""
        return self.abort_run

    @property
    def interrupt_event(self) -> asyncio.Event | None:
        """Event used by lifecycle consumers to detect run interruption."""
        return self._interrupt_event

    @interrupt_event.setter
    def interrupt_event(self, value: asyncio.Event | None) -> None:
        if value is not None:
            self._interrupt_event = value

    @property
    def max_issues(self) -> int | None:
        """Maximum issues to process."""
        return self.config.max_issues

    @max_issues.setter
    def max_issues(self, value: int | None) -> None:
        self.config.max_issues = value

    def request_abort(self, reason: str) -> None:
        """Signal that the current run should stop due to a fatal error.

        Sets both abort_run flag and abort_event for sync and async checking.

        Args:
            reason: Description of why the run should abort.
        """
        if self.abort_run:
            return
        self.abort_run = True
        self.abort_reason = reason
        self.abort_event.set()
        logger.warning("Abort requested: reason=%s", reason)
        self.event_sink.on_abort_requested(reason)

    async def _handle_interrupt(
        self,
        abort_callback: AbortCallback,
        watch_state: WatchState,
        issues_spawned: int,
    ) -> RunResult:
        """Handle Stage 2 SIGINT by aborting active tasks and returning exit_code=130.

        Stage 2 (graceful abort) does NOT run validation. The exit code is determined
        by the orchestrator's _abort_exit_code which was snapshotted at Stage 2 entry
        based on whether validation had already failed. This method always returns 130;
        the orchestrator overrides with _abort_exit_code when _abort_mode_active.

        Note: abort_callback waits up to ABORT_GRACE_SECONDS for tasks to finish.
        Tasks that don't respond to cancellation within the grace period are finalized
        as "unresponsive" but may still be running.
        """
        abort_result = await abort_callback(is_interrupt=True)
        aborted_count = abort_result.aborted_count
        # Recompute completed_count from completed_ids to avoid polluted state
        watch_state.completed_count = len(self.completed_ids) + aborted_count

        if abort_result.has_unresponsive_tasks:
            logger.warning("Unresponsive tasks may still be mutating repo")

        return RunResult(issues_spawned, exit_code=130, exit_reason="interrupted")

    async def run_loop(
        self,
        spawn_callback: SpawnCallback,
        finalize_callback: FinalizeCallback,
        abort_callback: AbortCallback,
        watch_config: WatchConfig | None = None,
        validation_config: PeriodicValidationConfig | None = None,
        interrupt_event: asyncio.Event | None = None,
        validation_callback: Callable[[], Awaitable[bool]] | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
        drain_event: asyncio.Event | None = None,
        on_validation_failed: Callable[[], None] | None = None,
        on_no_ready_epic_sweep: Callable[[], Awaitable[bool]] | None = None,
    ) -> RunResult:
        """Run the main agent spawning and completion loop.

        Args:
            spawn_callback: Called to spawn an agent for an issue.
                Returns the spawned Task on success, or None if spawn failed.
                The coordinator automatically registers the returned task.
            finalize_callback: Called when a task completes.
                Receives issue_id and the completed task.
            abort_callback: Called when abort is triggered.
                Should cancel and finalize all active tasks.
            watch_config: Watch mode configuration. If None or not enabled,
                loop exits when no work remains.
            validation_config: Periodic validation configuration. Controls when
                validation_callback is triggered based on completed issue count.
            interrupt_event: Event set to signal graceful shutdown (e.g., SIGINT).
            validation_callback: Called periodically to run validation.
                Returns True if validation passed, False otherwise.
            sleep_fn: Async sleep function, injectable for testing.
            drain_event: Event set to enter drain mode. In drain mode, no new issues
                are spawned, but active tasks complete normally. When all active tasks
                finish, validation is triggered (if validation_callback present) and
                the loop exits. If interrupt_event is also set, interrupt takes precedence.
            on_validation_failed: Called when validation fails, before returning with
                exit_code=1. Used to propagate validation failure state to orchestrator
                (e.g., for correct SIGINT exit code handling).
            on_no_ready_epic_sweep: Called whenever the task queue drains to empty with
                no active work (at startup or after issues complete) to verify and close
                eligible epics. Return True to re-poll before entering idle/exit handling
                (e.g. it created remediation work or closed an epic).

        Returns:
            RunResult with issues_spawned, exit_code, and exit_reason.
        """
        issues_spawned = 0
        interrupt_task: asyncio.Task[object] | None = None
        epic_sweep_pending = True
        follow_up_repoll_pending = False
        validate_every = self._validate_every(validation_config)
        watch_state = WatchState(next_validation_threshold=validate_every or 10)
        poll_strategy = _CoordinatorPollStrategy(
            beads=self.beads,
            event_sink=self.event_sink,
            watch_state=watch_state,
            watch_config=watch_config,
            interrupt_event=interrupt_event,
            sleep_fn=sleep_fn,
        )
        work_queue = WorkQueue(
            self.beads,
            self._work_queue_config(),
            poll_strategy,
        )
        if interrupt_event is not None:
            self.interrupt_event = interrupt_event

        async def finalize_task(issue_id: str, task: asyncio.Task[Any]) -> bool:
            nonlocal follow_up_repoll_pending, epic_sweep_pending
            already_completed = issue_id in self.completed_ids
            created_follow_up_work = await finalize_callback(issue_id, task)
            if created_follow_up_work is True:
                follow_up_repoll_pending = True
            newly_completed = issue_id in self.completed_ids and not already_completed
            if newly_completed:
                # A closed issue may have made its parent epic eligible; re-arm the
                # sweep so it fires once the queue drains.
                epic_sweep_pending = True
            return newly_completed

        try:
            while True:
                logger.debug(
                    "Loop iteration: active=%d completed=%d",
                    len(self.active_tasks),
                    len(self.completed_ids),
                )
                snapshot = self._queue_snapshot(
                    work_queue,
                    watch_state,
                    ready_issue_ids=(),
                    watch_config=watch_config,
                    epic_sweep_pending=epic_sweep_pending,
                    follow_up_repoll_pending=follow_up_repoll_pending,
                    drain_event=drain_event,
                )
                decision = exit_reason(snapshot)
                if decision.should_exit and decision.reason != "success":
                    return await self._interpret_exit(
                        decision,
                        abort_callback,
                        validation_callback,
                        on_validation_failed,
                        watch_state,
                        issues_spawned,
                        finalize_task,
                    )

                try:
                    poll_result = await work_queue.poll(snapshot)
                except _InterruptRequested:
                    return await self._handle_interrupt(
                        abort_callback,
                        watch_state,
                        issues_spawned,
                    )
                snapshot = poll_result.snapshot
                watch_state.consecutive_poll_failures = (
                    poll_result.consecutive_poll_failures
                )
                if poll_result.succeeded:
                    follow_up_repoll_pending = False

                poll_decision = poll_result.decision
                if poll_decision.kind == "transient_error":
                    if poll_result.error is not None:
                        logger.error("Poll failed", exc_info=poll_result.error)
                    decision = exit_reason(snapshot)
                    if decision.should_exit:
                        return await self._interpret_exit(
                            decision,
                            abort_callback,
                            validation_callback,
                            on_validation_failed,
                            watch_state,
                            issues_spawned,
                            finalize_task,
                        )
                    continue

                if poll_decision.kind == "ready":
                    ready = list(poll_result.ready_issue_ids)
                    self.event_sink.on_ready_issues(ready)
                    poll_strategy.clear_idle()
                elif poll_decision.kind == "empty":
                    ready = []
                    # Sweep eligible epics only once the queue is fully drained: a
                    # re-armed flag must wait for active agents to finish first.
                    if epic_sweep_pending and not self.active_tasks:
                        epic_sweep_pending = False
                        repoll = await self._handle_no_ready_epic_sweep(
                            on_no_ready_epic_sweep,
                        )
                        if repoll:
                            continue
                elif poll_decision.kind == "terminal_drain":
                    ready = []
                else:
                    assert_never(poll_decision.kind)

                capacity = capacity_decision(snapshot).capacity
                spawned_now, ready = await self._spawn_ready_issues(
                    ready,
                    capacity,
                    spawn_callback,
                )
                issues_spawned += spawned_now
                snapshot = self._queue_snapshot(
                    work_queue,
                    watch_state,
                    ready_issue_ids=ready,
                    watch_config=watch_config,
                    epic_sweep_pending=epic_sweep_pending,
                    follow_up_repoll_pending=follow_up_repoll_pending,
                    drain_event=drain_event,
                )

                decision = exit_reason(snapshot)
                if decision.should_exit:
                    return await self._interpret_exit(
                        decision,
                        abort_callback,
                        validation_callback,
                        on_validation_failed,
                        watch_state,
                        issues_spawned,
                        finalize_task,
                    )

                validation_decision = periodic_validation(snapshot, validate_every)
                validation_result = await self._interpret_periodic_validation(
                    validation_decision,
                    validation_callback,
                    on_validation_failed,
                    watch_state,
                    issues_spawned,
                    finalize_task,
                    validate_every,
                )
                if validation_result is not None:
                    return validation_result

                if self.active_tasks:
                    poll_strategy.clear_idle()
                    interrupt_task = await self._wait_for_agent_completion(
                        finalize_task,
                        watch_state,
                        interrupt_event,
                        interrupt_task,
                    )
                    snapshot = self._queue_snapshot(
                        work_queue,
                        watch_state,
                        ready_issue_ids=ready,
                        watch_config=watch_config,
                        epic_sweep_pending=epic_sweep_pending,
                        follow_up_repoll_pending=follow_up_repoll_pending,
                        drain_event=drain_event,
                    )
                    if snapshot.interrupt_requested or snapshot.abort_requested:
                        decision = exit_reason(snapshot)
                        return await self._interpret_exit(
                            decision,
                            abort_callback,
                            validation_callback,
                            on_validation_failed,
                            watch_state,
                            issues_spawned,
                            finalize_task,
                        )
                    validation_decision = periodic_validation(snapshot, validate_every)
                    validation_result = await self._interpret_periodic_validation(
                        validation_decision,
                        validation_callback,
                        on_validation_failed,
                        watch_state,
                        issues_spawned,
                        finalize_task,
                        validate_every,
                    )
                    if validation_result is not None:
                        return validation_result
                else:
                    result = await self._wait_for_idle_poll(
                        poll_strategy,
                        snapshot,
                        abort_callback,
                        watch_state,
                        issues_spawned,
                    )
                    if result is not None:
                        return result
        finally:
            await self._cleanup_interrupt_task(interrupt_task)

        return RunResult(issues_spawned, exit_code=0, exit_reason="success")

    def _work_queue_config(self) -> WorkQueueConfig:
        return WorkQueueConfig(
            max_agents=self.config.max_agents,
            max_issues=self.config.max_issues,
            epic_id=self.config.epic_id,
            only_ids=self.config.only_ids,
            include_wip=self.config.include_wip,
            focus=self.config.focus,
            orphans_only=self.config.orphans_only,
            order_preference=self.config.order_preference,
            prioritize_wip=self.config.prioritize_wip,
        )

    def _queue_snapshot(
        self,
        work_queue: WorkQueue,
        watch_state: WatchState,
        *,
        ready_issue_ids: list[str] | tuple[str, ...],
        watch_config: WatchConfig | None,
        epic_sweep_pending: bool,
        follow_up_repoll_pending: bool,
        drain_event: asyncio.Event | None,
    ) -> WorkQueueSnapshot:
        snapshot = work_queue.snapshot(
            active_issue_ids=set(self.active_tasks),
            completed_issue_ids=self.completed_ids,
            failed_issue_ids=self.failed_issues,
            ready_issue_ids=ready_issue_ids,
            completed_count=watch_state.completed_count,
            last_validation_at=watch_state.last_validation_at,
            next_validation_threshold=watch_state.next_validation_threshold,
            consecutive_poll_failures=watch_state.consecutive_poll_failures,
            watch_enabled=self._watch_enabled(watch_config),
            epic_sweep_pending=epic_sweep_pending,
            follow_up_repoll_pending=follow_up_repoll_pending,
            abort_requested=self.abort_run,
            interrupt_requested=self._interrupt_event.is_set(),
            is_draining=self._event_is_set(drain_event),
        )
        return replace(snapshot, max_issues=self.config.max_issues)

    async def _interpret_exit(
        self,
        decision: ExitDecision,
        abort_callback: AbortCallback,
        validation_callback: Callable[[], Awaitable[bool]] | None,
        on_validation_failed: Callable[[], None] | None,
        watch_state: WatchState,
        issues_spawned: int,
        finalize_task: Callable[[str, asyncio.Task[Any]], Awaitable[bool]],
    ) -> RunResult:
        if decision.reason == "interrupted":
            return await self._handle_interrupt(
                abort_callback,
                watch_state,
                issues_spawned,
            )
        if decision.reason == "abort":
            await abort_callback()
            return RunResult(issues_spawned, exit_code=3, exit_reason="abort")
        if decision.wait_for_active_work:
            await self._wait_for_all_active(finalize_task, watch_state)
            await abort_callback()
        if decision.reason == "limit_reached":
            self.event_sink.on_no_more_issues(
                f"limit_reached ({self.config.max_issues})"
            )
        if decision.reason == "success":
            self.event_sink.on_no_more_issues("none_ready")
        if decision.run_final_validation:
            result = await self._run_validation(
                validation_callback,
                on_validation_failed,
                watch_state,
                issues_spawned,
            )
            if result is not None:
                return result
        return RunResult(
            issues_spawned,
            exit_code=decision.exit_code or 0,
            exit_reason=decision.reason or "success",
        )

    async def _interpret_periodic_validation(
        self,
        decision: PeriodicValidationDecision,
        validation_callback: Callable[[], Awaitable[bool]] | None,
        on_validation_failed: Callable[[], None] | None,
        watch_state: WatchState,
        issues_spawned: int,
        finalize_task: Callable[[str, asyncio.Task[Any]], Awaitable[bool]],
        validate_every: int | None,
    ) -> RunResult | None:
        if not decision.should_validate:
            return None
        if validation_callback is None:
            return None
        if validate_every is None:
            return None
        if decision.block_for_active_work:
            await self._wait_for_all_active(finalize_task, watch_state)
        result = await self._run_validation(
            validation_callback,
            on_validation_failed,
            watch_state,
            issues_spawned,
        )
        if result is not None:
            return result
        watch_state.next_validation_threshold = next_validation_threshold(
            watch_state.completed_count,
            decision.threshold or watch_state.next_validation_threshold,
            validate_every,
        )
        return None

    async def _run_validation(
        self,
        validation_callback: Callable[[], Awaitable[bool]] | None,
        on_validation_failed: Callable[[], None] | None,
        watch_state: WatchState,
        issues_spawned: int,
    ) -> RunResult | None:
        if validation_callback is None:
            return None
        validation_passed = await validation_callback()
        watch_state.last_validation_at = watch_state.completed_count
        if validation_passed:
            return None
        if on_validation_failed is not None:
            on_validation_failed()
        return RunResult(
            issues_spawned,
            exit_code=1,
            exit_reason="validation_failed",
        )

    async def _spawn_ready_issues(
        self,
        ready: list[str],
        capacity: int,
        spawn_callback: SpawnCallback,
    ) -> tuple[int, list[str]]:
        spawned = 0
        remaining = list(ready)
        while remaining and spawned < capacity:
            issue_id = remaining.pop(0)
            if issue_id in self.active_tasks:
                continue
            task = await spawn_callback(issue_id)
            if task is None:
                continue
            self.register_task(issue_id, task)
            spawned += 1
        return spawned, remaining

    async def _wait_for_agent_completion(
        self,
        finalize_task: Callable[[str, asyncio.Task[Any]], Awaitable[bool]],
        watch_state: WatchState,
        interrupt_event: asyncio.Event | None,
        interrupt_task: asyncio.Task[object] | None,
    ) -> asyncio.Task[object] | None:
        self.event_sink.on_waiting_for_agents(len(self.active_tasks))
        wait_tasks: set[asyncio.Task[object]] = set(self.active_tasks.values())
        if interrupt_event is not None:
            if interrupt_task is None:
                interrupt_task = asyncio.create_task(interrupt_event.wait())
            if interrupt_task.done():
                interrupt_task = asyncio.create_task(interrupt_event.wait())
            wait_tasks.add(interrupt_task)

        done, _ = await asyncio.wait(
            wait_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        done_agent_tasks = done
        if interrupt_task is not None:
            done_agent_tasks = done - {interrupt_task}
        await self._finalize_done_tasks(done_agent_tasks, finalize_task, watch_state)
        return interrupt_task

    async def _wait_for_all_active(
        self,
        finalize_task: Callable[[str, asyncio.Task[Any]], Awaitable[bool]],
        watch_state: WatchState,
    ) -> None:
        if not self.active_tasks:
            return
        await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        await self._finalize_done_tasks(
            set(self.active_tasks.values()),
            finalize_task,
            watch_state,
        )

    async def _finalize_done_tasks(
        self,
        done_agent_tasks: set[asyncio.Task[object]],
        finalize_task: Callable[[str, asyncio.Task[Any]], Awaitable[bool]],
        watch_state: WatchState,
    ) -> None:
        for task in done_agent_tasks:
            for issue_id, active_task in list(self.active_tasks.items()):
                if active_task is not task:
                    continue
                if await finalize_task(issue_id, active_task):
                    watch_state.completed_count += 1
                break

    async def _wait_for_idle_poll(
        self,
        poll_strategy: _CoordinatorPollStrategy,
        snapshot: WorkQueueSnapshot,
        abort_callback: AbortCallback,
        watch_state: WatchState,
        issues_spawned: int,
    ) -> RunResult | None:
        if not snapshot.watch_enabled:
            return None
        if poll_strategy.consume_idle_waited():
            return None
        try:
            await poll_strategy.wait_until_next_poll(snapshot)
        except _InterruptRequested:
            return await self._handle_interrupt(
                abort_callback,
                watch_state,
                issues_spawned,
            )
        return None

    async def _handle_no_ready_epic_sweep(
        self,
        on_no_ready_epic_sweep: Callable[[], Awaitable[bool]] | None,
    ) -> bool:
        if on_no_ready_epic_sweep is None:
            return False
        return await on_no_ready_epic_sweep()

    async def _cleanup_interrupt_task(
        self, interrupt_task: asyncio.Task[object] | None
    ) -> None:
        if interrupt_task is None:
            return
        if interrupt_task.done():
            return
        interrupt_task.cancel()
        try:
            await interrupt_task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None:
                if current.cancelling() > 0:
                    raise

    @staticmethod
    def _watch_enabled(watch_config: WatchConfig | None) -> bool:
        if watch_config is None:
            return False
        return bool(watch_config.enabled)

    @staticmethod
    def _event_is_set(event: asyncio.Event | None) -> bool:
        if event is None:
            return False
        return event.is_set()

    @staticmethod
    def _validate_every(
        validation_config: PeriodicValidationConfig | None,
    ) -> int | None:
        if validation_config is None:
            return None
        return validation_config.validate_every

    def register_task(self, issue_id: str, task: asyncio.Task[Any]) -> None:
        """Register an active task for an issue.

        Called by spawn_callback after successfully creating a task.

        Args:
            issue_id: The issue ID.
            task: The asyncio task running the agent.
        """
        self.active_tasks[issue_id] = task
        logger.debug("Task registered: issue_id=%s", issue_id)

    def mark_failed(self, issue_id: str) -> None:
        """Mark an issue as failed (e.g., claim failed).

        Args:
            issue_id: The issue ID that failed.
        """
        self.failed_issues.add(issue_id)
        logger.info("Issue marked failed: issue_id=%s", issue_id)

    def mark_completed(self, issue_id: str) -> None:
        """Mark an issue as completed and remove from active.

        Args:
            issue_id: The issue ID that completed.
        """
        self.completed_ids.add(issue_id)
        self.active_tasks.pop(issue_id, None)
        logger.debug("Issue marked completed: issue_id=%s", issue_id)

    def release_task(self, issue_id: str) -> None:
        """Remove an issue from active tracking without marking it terminal.

        Use this when an in-flight attempt was cancelled but the issue should remain
        eligible for a later retry in the same run.

        Args:
            issue_id: The issue ID whose active task should be released.
        """
        self.active_tasks.pop(issue_id, None)
        logger.debug("Issue released for retry: issue_id=%s", issue_id)
