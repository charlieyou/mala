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

from src.orchestration.types import RunResult, WatchState

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.core.protocols import IssueProvider, MalaEventSink
    from src.orchestration.types import WatchConfig

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


class AbortCallback(Protocol):
    """Callback for aborting active tasks."""

    async def __call__(self) -> None:
        """Abort all active tasks."""
        ...


@dataclass
class CoordinatorConfig:
    """Configuration for IssueExecutionCoordinator.

    Attributes:
        max_agents: Maximum concurrent agents (None = unlimited).
        max_issues: Maximum issues to process (None = unlimited).
        epic_id: Only process tasks under this epic.
        only_ids: Set of issue IDs to process exclusively.
        prioritize_wip: Prioritize in_progress issues before open issues.
        focus: Group tasks by epic for focused work.
        orphans_only: Only process issues with no parent epic.
    """

    max_agents: int | None = None
    max_issues: int | None = None
    epic_id: str | None = None
    only_ids: set[str] | None = None
    prioritize_wip: bool = False
    focus: bool = True
    orphans_only: bool = False


class IssueExecutionCoordinator:
    """Coordinates issue execution without SDK dependencies.

    This class manages the main agent spawning loop, handling:
    - Issue fetching from IssueProvider
    - Concurrent agent limiting (max_agents)
    - Issue count limiting (max_issues)
    - Epic and only_ids filtering
    - WIP prioritization and focus grouping

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

    def request_abort(self, reason: str) -> None:
        """Signal that the current run should stop due to a fatal error.

        Args:
            reason: Description of why the run should abort.
        """
        if self.abort_run:
            return
        self.abort_run = True
        self.abort_reason = reason
        logger.warning("Abort requested: reason=%s", reason)
        self.event_sink.on_abort_requested(reason)

    async def run_loop(
        self,
        spawn_callback: SpawnCallback,
        finalize_callback: FinalizeCallback,
        abort_callback: AbortCallback,
        watch_config: WatchConfig | None = None,
        interrupt_event: asyncio.Event | None = None,
        validation_callback: Callable[[], Awaitable[bool]] | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
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
            interrupt_event: Event set to signal graceful shutdown (e.g., SIGINT).
            validation_callback: Called periodically in watch mode to run validation.
                Returns True if validation passed, False otherwise.
            sleep_fn: Async sleep function, injectable for testing.

        Returns:
            RunResult with issues_spawned, exit_code, and exit_reason.
        """
        issues_spawned = 0

        # Initialize watch state from config (used in T004-T006)
        watch_state = WatchState(
            next_validation_threshold=(
                watch_config.validate_every if watch_config else 10
            )
        )

        while True:
            logger.debug(
                "Loop iteration: active=%d completed=%d",
                len(self.active_tasks),
                len(self.completed_ids),
            )
            # Check for abort
            if self.abort_run:
                await abort_callback()
                return RunResult(
                    issues_spawned=issues_spawned,
                    exit_code=3,
                    exit_reason="abort",
                )

            # Check if we've hit the issue limit
            limit_reached = (
                self.config.max_issues is not None
                and issues_spawned >= self.config.max_issues
            )

            # Build suppress_warn_ids for only_ids mode
            suppress_warn_ids = None
            if self.config.only_ids:
                suppress_warn_ids = (
                    self.failed_issues
                    | set(self.active_tasks.keys())
                    | self.completed_ids
                )

            # Fetch ready issues (unless we've hit the limit)
            if limit_reached:
                ready: list[str] = []
            else:
                try:
                    ready = await self.beads.get_ready_async(
                        self.failed_issues,
                        epic_id=self.config.epic_id,
                        only_ids=self.config.only_ids,
                        suppress_warn_ids=suppress_warn_ids,
                        prioritize_wip=self.config.prioritize_wip,
                        focus=self.config.focus,
                        orphans_only=self.config.orphans_only,
                    )
                    watch_state.consecutive_poll_failures = 0  # Reset on success
                except Exception as e:
                    logger.error("Poll failed: %s", e)
                    watch_state.consecutive_poll_failures += 1

                    if watch_state.consecutive_poll_failures >= 3:
                        # Run final validation if any issues completed
                        if watch_state.completed_count > watch_state.last_validation_at:
                            if validation_callback:
                                await validation_callback()
                        return RunResult(
                            issues_spawned,
                            exit_code=3,
                            exit_reason="poll_failed",
                        )

                    # Wait and retry
                    poll_interval = (
                        watch_config.poll_interval_seconds if watch_config else 60.0
                    )
                    await sleep_fn(poll_interval)
                    continue

            if ready:
                self.event_sink.on_ready_issues(list(ready))

            # Spawn agents while we have capacity and ready issues
            while (
                self.config.max_agents is None
                or len(self.active_tasks) < self.config.max_agents
            ) and ready:
                issue_id = ready.pop(0)
                if issue_id not in self.active_tasks:
                    task = await spawn_callback(issue_id)
                    if task is not None:
                        self.register_task(issue_id, task)
                        issues_spawned += 1
                        if (
                            self.config.max_issues is not None
                            and issues_spawned >= self.config.max_issues
                        ):
                            break

            # Exit if no active work
            if not self.active_tasks:
                if limit_reached:
                    self.event_sink.on_no_more_issues(
                        f"limit_reached ({self.config.max_issues})"
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
                        now = time.time()
                        if not watch_state.was_idle:
                            watch_state.was_idle = True
                            watch_state.last_idle_log_time = now
                            blocked_count = await self.beads.get_blocked_count_async()
                            self.event_sink.on_watch_idle(
                                watch_config.poll_interval_seconds, blocked_count
                            )
                        elif now - watch_state.last_idle_log_time >= 300:
                            watch_state.last_idle_log_time = now
                            blocked_count = await self.beads.get_blocked_count_async()
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
                                # Run final validation if any issues completed
                                if (
                                    watch_state.completed_count
                                    > watch_state.last_validation_at
                                ):
                                    if validation_callback:
                                        await validation_callback()
                                return RunResult(
                                    issues_spawned,
                                    exit_code=130,
                                    exit_reason="interrupted",
                                )
                            except TimeoutError:
                                pass  # Normal timeout, continue loop
                        else:
                            await sleep_fn(watch_config.poll_interval_seconds)
                        continue  # Re-poll
                    else:
                        # Watch mode not enabled: exit immediately
                        self.event_sink.on_no_more_issues("none_ready")
                        return RunResult(
                            issues_spawned=issues_spawned,
                            exit_code=0,
                            exit_reason="success",
                        )
                else:
                    # Ready issues exist but no active tasks - clear idle state
                    watch_state.was_idle = False

            # Wait for at least one task to complete
            self.event_sink.on_waiting_for_agents(len(self.active_tasks))
            done, _ = await asyncio.wait(
                self.active_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Finalize completed tasks
            for task in done:
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
