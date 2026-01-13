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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from src.core.models import OrderPreference, RunResult, WatchState

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.core.models import PeriodicValidationConfig, WatchConfig
    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.issue import IssueProvider

logger = logging.getLogger(__name__)


class SpawnCallback(Protocol):
    """Callback for spawning an agent for an issue.

    Returns the spawned Task on success, or None if spawn failed.
    The coordinator automatically registers the returned task.
    """

    async def __call__(self, issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
        """Spawn an agent for the given issue."""
        ...


class FinalizeCallback(Protocol):
    """Callback for finalizing an issue result.

    Takes the issue_id and the task that completed.
    """

    async def __call__(self, issue_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
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
    """

    max_agents: int | None = None
    max_issues: int | None = None
    epic_id: str | None = None
    only_ids: list[str] | None = None
    include_wip: bool = False
    focus: bool = True
    orphans_only: bool = False
    order_preference: OrderPreference = OrderPreference.EPIC_PRIORITY


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
        self.active_tasks: dict[str, asyncio.Task] = {}  # type: ignore[type-arg]
        self.completed_ids: set[str] = set()
        self.failed_issues: set[str] = set()
        self.abort_run: bool = False
        self.abort_reason: str | None = None
        self.abort_event: asyncio.Event = asyncio.Event()

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

        Returns:
            RunResult with issues_spawned, exit_code, and exit_reason.
        """
        issues_spawned = 0

        # Initialize watch state from config (used in T004-T006)
        watch_state = WatchState(
            next_validation_threshold=(
                validation_config.validate_every
                if validation_config and validation_config.validate_every is not None
                else 10
            )
        )
        # Track interrupt_task outside the loop to ensure cleanup on any exit path
        interrupt_task: asyncio.Task[object] | None = None

        async def _cleanup_interrupt_task() -> None:
            """Cancel interrupt_task if it exists and is pending."""
            nonlocal interrupt_task
            if interrupt_task is not None and not interrupt_task.done():
                interrupt_task.cancel()
                try:
                    await interrupt_task
                except asyncio.CancelledError:
                    # Only suppress if this is from our cancelled interrupt_task,
                    # not from the current task being cancelled externally
                    current = asyncio.current_task()
                    if current is not None and current.cancelling() > 0:
                        raise
                    # Otherwise suppress - expected when we cancel interrupt_task

        try:
            while True:
                logger.debug(
                    "Loop iteration: active=%d completed=%d",
                    len(self.active_tasks),
                    len(self.completed_ids),
                )
                # Check for interrupt (SIGINT)
                if interrupt_event and interrupt_event.is_set():
                    return await self._handle_interrupt(
                        abort_callback,
                        watch_state,
                        issues_spawned,
                    )

                # Check for abort
                if self.abort_run:
                    await abort_callback()
                    return RunResult(
                        issues_spawned=issues_spawned,
                        exit_code=3,
                        exit_reason="abort",
                    )

                # Check for drain mode (Stage 1 SIGINT - stop spawning, let active complete)
                is_draining = drain_event is not None and drain_event.is_set()
                if is_draining and not self.active_tasks:
                    # All drained - trigger validation if callback present
                    if validation_callback:
                        if watch_state.completed_count > watch_state.last_validation_at:
                            validation_passed = await validation_callback()
                            watch_state.last_validation_at = watch_state.completed_count
                            if not validation_passed:
                                if on_validation_failed:
                                    on_validation_failed()
                                return RunResult(
                                    issues_spawned=issues_spawned,
                                    exit_code=1,
                                    exit_reason="validation_failed",
                                )
                    return RunResult(
                        issues_spawned=issues_spawned,
                        exit_code=0,
                        exit_reason="drained",
                    )

                # Check if we've hit the issue limit (using completed_count, not spawned)
                limit_reached = (
                    self.config.max_issues is not None
                    and watch_state.completed_count >= self.config.max_issues
                )

                # Build suppress_warn_ids for only_ids mode
                suppress_warn_ids = None
                if self.config.only_ids:
                    suppress_warn_ids = (
                        self.failed_issues
                        | set(self.active_tasks.keys())
                        | self.completed_ids
                    )

                # Fetch ready issues (unless we've hit the limit or draining)
                if limit_reached or is_draining:
                    ready: list[str] = []
                else:
                    try:
                        ready = await self.beads.get_ready_async(
                            self.failed_issues,
                            epic_id=self.config.epic_id,
                            only_ids=self.config.only_ids,
                            suppress_warn_ids=suppress_warn_ids,
                            include_wip=self.config.include_wip,
                            focus=self.config.focus,
                            orphans_only=self.config.orphans_only,
                            order_preference=self.config.order_preference,
                        )
                        watch_state.consecutive_poll_failures = 0  # Reset on success
                    except Exception:
                        logger.exception("Poll failed")
                        watch_state.consecutive_poll_failures += 1

                        if watch_state.consecutive_poll_failures >= 3:
                            # Wait for remaining active agents before aborting
                            if self.active_tasks:
                                await asyncio.gather(
                                    *self.active_tasks.values(), return_exceptions=True
                                )
                                # Finalize any tasks that completed during gather
                                for issue_id, t in list(self.active_tasks.items()):
                                    if t.done():
                                        await finalize_callback(issue_id, t)
                                        watch_state.completed_count += 1
                            # Abort any active tasks before returning
                            await abort_callback()
                            # Run final validation if any issues completed
                            if (
                                watch_state.completed_count
                                > watch_state.last_validation_at
                                and validation_callback
                            ):
                                valid = await validation_callback()
                                watch_state.last_validation_at = (
                                    watch_state.completed_count
                                )
                                if not valid:
                                    if on_validation_failed:
                                        on_validation_failed()
                                    return RunResult(
                                        issues_spawned,
                                        exit_code=1,
                                        exit_reason="validation_failed",
                                    )
                            return RunResult(
                                issues_spawned,
                                exit_code=3,
                                exit_reason="poll_failed",
                            )

                        # Interruptible wait before retry
                        poll_interval = (
                            watch_config.poll_interval_seconds if watch_config else 60.0
                        )
                        if interrupt_event:
                            try:
                                await asyncio.wait_for(
                                    interrupt_event.wait(),
                                    timeout=poll_interval,
                                )
                                # Interrupted during poll retry wait
                                return await self._handle_interrupt(
                                    abort_callback,
                                    watch_state,
                                    issues_spawned,
                                )
                            except TimeoutError:
                                pass  # Normal timeout, retry poll
                        else:
                            await sleep_fn(poll_interval)

                        # Retry poll immediately after sleep (don't block on task completion)
                        continue

                if ready:
                    self.event_sink.on_ready_issues(list(ready))

                # Spawn agents while we have capacity, ready issues, and haven't hit limit
                # Note: max_issues counts terminal states (completed_count), not spawn attempts
                # Once limit_reached, stop spawning but let active tasks complete
                # Also limit spawns to max_issues to prevent over-spawning on first iteration
                spawn_limit_reached = (
                    self.config.max_issues is not None
                    and issues_spawned >= self.config.max_issues
                )
                while (
                    not limit_reached
                    and not spawn_limit_reached
                    and (
                        self.config.max_agents is None
                        or len(self.active_tasks) < self.config.max_agents
                    )
                    and ready
                ):
                    issue_id = ready.pop(0)
                    if issue_id not in self.active_tasks:
                        task = await spawn_callback(issue_id)
                        if task is not None:
                            self.register_task(issue_id, task)
                            issues_spawned += 1
                            # Update spawn limit check for next iteration
                            spawn_limit_reached = (
                                self.config.max_issues is not None
                                and issues_spawned >= self.config.max_issues
                            )

                # Exit if no active work
                if not self.active_tasks:
                    if limit_reached:
                        self.event_sink.on_no_more_issues(
                            f"limit_reached ({self.config.max_issues})"
                        )
                        # Run final validation if needed
                        if watch_state.completed_count > watch_state.last_validation_at:
                            if validation_callback:
                                validation_passed = await validation_callback()
                                watch_state.last_validation_at = (
                                    watch_state.completed_count
                                )
                                if not validation_passed:
                                    if on_validation_failed:
                                        on_validation_failed()
                                    return RunResult(
                                        issues_spawned=issues_spawned,
                                        exit_code=1,
                                        exit_reason="validation_failed",
                                    )
                        return RunResult(
                            issues_spawned=issues_spawned,
                            exit_code=0,
                            exit_reason="limit_reached",
                        )
                    elif not ready:
                        # Idle: no ready issues AND no active agents
                        if watch_config and watch_config.enabled:
                            # Detect transition to idle and rate-limit logging
                            # Use monotonic clock to avoid NTP/clock change issues
                            now = time.monotonic()
                            if not watch_state.was_idle:
                                watch_state.was_idle = True
                                watch_state.last_idle_log_time = now
                                blocked_count = (
                                    await self.beads.get_blocked_count_async()
                                )
                                self.event_sink.on_watch_idle(
                                    watch_config.poll_interval_seconds, blocked_count
                                )
                            elif now - watch_state.last_idle_log_time >= 300:
                                watch_state.last_idle_log_time = now
                                blocked_count = (
                                    await self.beads.get_blocked_count_async()
                                )
                                self.event_sink.on_watch_idle(
                                    watch_config.poll_interval_seconds, blocked_count
                                )

                            # Interruptible sleep - respond to SIGINT immediately
                            if interrupt_event:
                                try:
                                    await asyncio.wait_for(
                                        interrupt_event.wait(),
                                        timeout=watch_config.poll_interval_seconds,
                                    )
                                    # Event was set - SIGINT received during idle
                                    return await self._handle_interrupt(
                                        abort_callback,
                                        watch_state,
                                        issues_spawned,
                                    )
                                except TimeoutError:
                                    pass  # Normal timeout, continue loop
                            else:
                                await sleep_fn(watch_config.poll_interval_seconds)
                            continue  # Re-poll
                        else:
                            # Watch mode not enabled: exit immediately
                            self.event_sink.on_no_more_issues("none_ready")
                            # Run final validation if any issues completed
                            if (
                                watch_state.completed_count
                                > watch_state.last_validation_at
                            ):
                                if validation_callback:
                                    validation_passed = await validation_callback()
                                    watch_state.last_validation_at = (
                                        watch_state.completed_count
                                    )
                                    if not validation_passed:
                                        if on_validation_failed:
                                            on_validation_failed()
                                        return RunResult(
                                            issues_spawned=issues_spawned,
                                            exit_code=1,
                                            exit_reason="validation_failed",
                                        )
                            return RunResult(
                                issues_spawned=issues_spawned,
                                exit_code=0,
                                exit_reason="success",
                            )
                    else:
                        # Ready issues exist but spawn returned None for all
                        # (e.g., already in-progress). Clear idle state and re-poll.
                        watch_state.was_idle = False
                        continue

                # Clear idle state when there's active work
                if ready or self.active_tasks:
                    watch_state.was_idle = False

                # Wait for at least one task to complete, interruptible by SIGINT
                self.event_sink.on_waiting_for_agents(len(self.active_tasks))
                wait_tasks: set[asyncio.Task[object]] = set(self.active_tasks.values())
                # Reuse or create interrupt_task (created once, reused across iterations)
                if interrupt_event:
                    if interrupt_task is None or interrupt_task.done():
                        interrupt_task = asyncio.create_task(interrupt_event.wait())
                    wait_tasks.add(interrupt_task)

                done, _ = await asyncio.wait(
                    wait_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Note: interrupt_task is NOT cancelled here - it's reused across
                # iterations. Cleanup happens in the finally block when the loop exits.

                # Remove interrupt task from done set for processing
                done_agent_tasks = done - {interrupt_task} if interrupt_task else done

                # Finalize completed tasks
                for task in done_agent_tasks:
                    for issue_id, t in list(self.active_tasks.items()):
                        if t is task:
                            await finalize_callback(issue_id, task)
                            watch_state.completed_count += 1
                            break

                # Check for abort after processing completions
                if self.abort_run:
                    await abort_callback()
                    return RunResult(
                        issues_spawned=issues_spawned,
                        exit_code=3,
                        exit_reason="abort",
                    )

                # Check for interrupt after processing completions
                if interrupt_event and interrupt_event.is_set():
                    return await self._handle_interrupt(
                        abort_callback,
                        watch_state,
                        issues_spawned,
                    )

                # Check if validation threshold crossed
                # Only enter this block if validation is actually configured
                validate_every = (
                    validation_config.validate_every
                    if validation_config is not None
                    else None
                )
                if (
                    validate_every is not None
                    and validation_callback
                    and watch_state.completed_count
                    >= watch_state.next_validation_threshold
                ):
                    # Wait for all active agents to finish (blocking validation)
                    if self.active_tasks:
                        await asyncio.gather(
                            *self.active_tasks.values(), return_exceptions=True
                        )
                        # Finalize any additional completed tasks
                        for issue_id, t in list(self.active_tasks.items()):
                            if t.done():
                                await finalize_callback(issue_id, t)
                                watch_state.completed_count += 1

                    # Run validation
                    if validation_callback:
                        validation_passed = await validation_callback()
                        if not validation_passed:
                            if on_validation_failed:
                                on_validation_failed()
                            return RunResult(
                                issues_spawned,
                                exit_code=1,
                                exit_reason="validation_failed",
                            )

                    # Advance threshold to next future threshold beyond completed_count
                    # This handles cases where completed_count jumps past multiple thresholds
                    watch_state.last_validation_at = watch_state.completed_count
                    while (
                        watch_state.next_validation_threshold
                        <= watch_state.completed_count
                    ):
                        watch_state.next_validation_threshold += validate_every

        finally:
            # Clean up interrupt_task when exiting the loop (any return path)
            await _cleanup_interrupt_task()

        # This should not be reached; included for type safety
        return RunResult(
            issues_spawned=issues_spawned, exit_code=0, exit_reason="success"
        )

    def register_task(self, issue_id: str, task: asyncio.Task) -> None:  # type: ignore[type-arg]
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
