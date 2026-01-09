"""DeadlockHandler service for managing deadlock resolution.

This module provides the DeadlockHandler class which owns the asyncio.Lock
for serializing deadlock resolution and coordinates cleanup via callbacks.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.pipeline.issue_execution_coordinator import AbortResult
from src.pipeline.issue_result import IssueResult

# Stage 2 abort grace period (seconds) - how long to wait for tasks to finish
# before allowing Stage 3 (force kill). Defined here to avoid circular import
# with orchestrator.py.
ABORT_GRACE_SECONDS = 10.0

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from src.domain.deadlock import DeadlockInfo
    from src.orchestration.orchestrator_state import OrchestratorState
    from src.infra.io.log_output.run_metadata import RunMetadata

logger = logging.getLogger(__name__)


@dataclass
class DeadlockHandlerCallbacks:
    """Callback references for DeadlockHandler operations.

    These are callable getters that allow late binding to orchestrator state
    and external services (beads, event_sink, lock server).

    Attributes:
        add_dependency: Add dependency between issues. Args: (dependent_id, dependency_id).
        mark_needs_followup: Mark issue as needing followup. Args: (issue_id, summary, log_path).
        reopen_issue: Reopen issue by setting status to ready. Args: (issue_id).
        on_deadlock_detected: Event callback when deadlock is detected. Args: (info).
        on_locks_cleaned: Event callback when locks are cleaned. Args: (agent_id, count).
        on_tasks_aborting: Event callback when tasks are being aborted. Args: (count, reason).
        do_cleanup_agent_locks: Clean up locks held by agent. Args: (agent_id).
            Returns: (count, paths).
        unregister_agent: Unregister agent from deadlock monitor. Args: (agent_id).
        finalize_issue_result: Finalize an issue result. Args: (issue_id, result, run_metadata).
        mark_completed: Mark issue as completed in coordinator. Args: (issue_id).
    """

    add_dependency: Callable[[str, str], Awaitable[bool]]
    mark_needs_followup: Callable[[str, str, Path | None], Awaitable[bool]]
    reopen_issue: Callable[[str], Awaitable[bool]]
    on_deadlock_detected: Callable[[DeadlockInfo], None]
    on_locks_cleaned: Callable[[str, int], None]
    on_tasks_aborting: Callable[[int, str], None]
    do_cleanup_agent_locks: Callable[[str], tuple[int, list[str]]]
    unregister_agent: Callable[[str], None]
    finalize_issue_result: Callable[[str, IssueResult, RunMetadata], Awaitable[None]]
    mark_completed: Callable[[str], None]


class DeadlockHandler:
    """Service for handling deadlock detection and resolution.

    Owns the asyncio.Lock for serializing deadlock resolution and coordinates
    cleanup via callbacks to external services.
    """

    def __init__(self, callbacks: DeadlockHandlerCallbacks) -> None:
        """Initialize DeadlockHandler with callback references.

        Args:
            callbacks: Callback references for external operations.
        """
        self._callbacks = callbacks
        self._resolution_lock = asyncio.Lock()

    async def handle_deadlock(
        self,
        info: DeadlockInfo,
        state: OrchestratorState,
        active_tasks: dict[str, asyncio.Task[IssueResult]],
    ) -> None:
        """Handle a detected deadlock by cancelling victim and recording dependency.

        Called by DeadlockMonitor when a cycle is detected. Uses an asyncio.Lock
        to prevent concurrent resolution of multiple deadlocks.

        Args:
            info: DeadlockInfo with cycle, victim, and blocker details.
            state: OrchestratorState for tracking cleaned agents and log paths.
            active_tasks: Mapping of issue_id to asyncio.Task for cancellation.
        """
        logger.debug("Acquiring deadlock resolution lock for victim %s", info.victim_id)
        async with self._resolution_lock:
            logger.info(
                "Deadlock resolution started: victim_id=%s issue_id=%s blocked_on=%s",
                info.victim_id,
                info.victim_issue_id,
                info.blocked_on,
            )
            self._callbacks.on_deadlock_detected(info)

            victim_issue_id = info.victim_issue_id
            task_to_cancel: asyncio.Task[object] | None = None
            is_self_cancel = False

            # Identify victim task for cancellation
            if victim_issue_id and victim_issue_id in active_tasks:
                task = active_tasks[victim_issue_id]
                if not task.done():
                    task_to_cancel = task
                    is_self_cancel = task is asyncio.current_task()

            # Clean up victim's locks first (before any await that could raise)
            # Track that we cleaned this agent to avoid double cleanup in run_implementer
            self.cleanup_agent_locks(info.victim_id)
            state.deadlock_cleaned_agents.add(info.victim_id)
            if victim_issue_id:
                state.deadlock_victim_issues.add(victim_issue_id)

            # Use shield to protect resolution from cancellation
            # Track whether cancellation occurred during shielded section
            cancelled_during_shield = False
            try:
                await asyncio.shield(self._resolve_deadlock(info, state))
            except asyncio.CancelledError:
                cancelled_during_shield = True
                # In self-cancel case, we'll schedule deferred cancellation below,
                # so don't re-raise yet. For external cancellation, re-raise.
                if not is_self_cancel:
                    raise

            # Cancel victim task after resolution is complete
            if task_to_cancel is not None:
                if is_self_cancel:
                    # Defer self-cancellation to avoid interrupting this handler
                    loop = asyncio.get_running_loop()
                    loop.call_soon(task_to_cancel.cancel)
                    logger.info("Victim killed: agent_id=%s", info.victim_id)
                else:
                    task_to_cancel.cancel()
                    logger.info("Victim killed: agent_id=%s", info.victim_id)

            # If we caught CancelledError in self-cancel case but it arrived before
            # we scheduled our deferred cancellation, it was from an external source.
            # Re-raise after scheduling our own cancellation to not mask it.
            if cancelled_during_shield and is_self_cancel:
                raise asyncio.CancelledError()

    async def _resolve_deadlock(
        self, info: DeadlockInfo, state: OrchestratorState
    ) -> None:
        """Perform dependency and needs-followup updates for deadlock resolution.

        Separated from handle_deadlock to allow shielding from cancellation.

        Args:
            info: DeadlockInfo with cycle, victim, and blocker details.
            state: OrchestratorState for accessing session log paths.
        """
        victim_issue_id = info.victim_issue_id

        # Add dependency: victim issue depends on blocker issue
        if victim_issue_id and info.blocker_issue_id:
            success = await self._callbacks.add_dependency(
                victim_issue_id, info.blocker_issue_id
            )
            if success:
                logger.info(
                    "Added dependency: %s depends on %s",
                    victim_issue_id,
                    info.blocker_issue_id,
                )
            else:
                logger.warning(
                    "Failed to add dependency: %s depends on %s",
                    victim_issue_id,
                    info.blocker_issue_id,
                )

        # Mark victim issue as needs-followup
        if victim_issue_id:
            reason = (
                f"Deadlock victim: blocked on {info.blocked_on} "
                f"held by {info.blocker_id}"
            )
            log_path = state.active_session_log_paths.get(victim_issue_id)
            await self._callbacks.mark_needs_followup(victim_issue_id, reason, log_path)
            logger.info("Marked issue %s as needs-followup", victim_issue_id)

            # Reset status to open so issue can be picked up again after blocker finishes
            if await self._callbacks.reopen_issue(victim_issue_id):
                logger.info(
                    "Reopened issue %s for retry after blocker completes",
                    victim_issue_id,
                )
            else:
                logger.warning("Failed to reopen issue %s for retry", victim_issue_id)

    async def abort_active_tasks(
        self,
        active_tasks: dict[str, asyncio.Task[IssueResult]],
        abort_reason: str | None,
        state: OrchestratorState,
        run_metadata: RunMetadata,
        *,
        is_interrupt: bool = False,
    ) -> AbortResult:
        """Cancel active tasks, wait up to ABORT_GRACE_SECONDS, and finalize them.

        Tasks that have already completed are finalized with their real results
        rather than being marked as aborted. Tasks that don't respond to
        cancellation within the grace period are finalized as "unresponsive"
        but may still be running (Stage 3 force kill handles these).

        Args:
            active_tasks: Mapping of issue_id to asyncio.Task.
            abort_reason: Reason for aborting tasks.
            state: Orchestrator state for agent_ids and session log paths.
            run_metadata: Run metadata for issue finalization.
            is_interrupt: If True, use "Interrupted" summary instead of "Aborted".

        Returns:
            AbortResult with aborted count and flag indicating unresponsive tasks.
        """
        if not active_tasks:
            return AbortResult(aborted_count=0, has_unresponsive_tasks=False)
        reason = abort_reason or (
            "Interrupted by user" if is_interrupt else "Unrecoverable error"
        )
        self._callbacks.on_tasks_aborting(len(active_tasks), reason)

        # Snapshot tasks before cancellation (abort_callback may clear active_tasks)
        tasks_snapshot = list(active_tasks.items())

        # Cancel tasks that are still running
        for _issue_id, task in tasks_snapshot:
            if not task.done():
                task.cancel()

        # Await with bounded timeout (Stage 2 grace period)
        # This prevents Stage 2 abort from waiting indefinitely for tasks to finish.
        # If tasks don't complete within the grace period, Stage 3 (force kill)
        # is still available via another Ctrl-C.
        #
        # Note: We use asyncio.wait() instead of asyncio.wait_for(gather(...))
        # because wait_for+gather blocks indefinitely when tasks suppress CancelledError.
        # asyncio.wait() returns immediately after the timeout with pending tasks.
        tasks_set = {task for _, task in tasks_snapshot}
        _done, pending = await asyncio.wait(tasks_set, timeout=ABORT_GRACE_SECONDS)
        has_unresponsive_tasks = len(pending) > 0

        # Finalize each issue - use real result if completed, abort summary if cancelled
        aborted_count = 0
        abort_summary = (
            f"Interrupted: {reason}" if is_interrupt else f"Aborted: {reason}"
        )
        for issue_id, task in tasks_snapshot:
            # Handle tasks still pending after timeout (didn't respond to cancel)
            if not task.done():
                aborted_count += 1
                result = IssueResult(
                    issue_id=issue_id,
                    agent_id=state.agent_ids.get(issue_id, "unknown"),
                    success=False,
                    summary=f"{abort_summary} (task unresponsive)",
                    session_log_path=state.active_session_log_paths.get(issue_id),
                )
            else:
                try:
                    result = task.result()
                except asyncio.CancelledError:
                    aborted_count += 1
                    result = IssueResult(
                        issue_id=issue_id,
                        agent_id=state.agent_ids.get(issue_id, "unknown"),
                        success=False,
                        summary=abort_summary,
                        session_log_path=state.active_session_log_paths.get(issue_id),
                    )
                except Exception as e:
                    result = IssueResult(
                        issue_id=issue_id,
                        agent_id=state.agent_ids.get(issue_id, "unknown"),
                        success=False,
                        summary=str(e),
                        session_log_path=state.active_session_log_paths.get(issue_id),
                    )
            await self._callbacks.finalize_issue_result(issue_id, result, run_metadata)
            # Mark completed in coordinator to keep state consistent
            self._callbacks.mark_completed(issue_id)
        return AbortResult(
            aborted_count=aborted_count, has_unresponsive_tasks=has_unresponsive_tasks
        )

    def cleanup_agent_locks(self, agent_id: str) -> None:
        """Remove locks held by a specific agent (crash/timeout cleanup).

        Args:
            agent_id: The agent whose locks should be cleaned up.
        """
        count, _ = self._callbacks.do_cleanup_agent_locks(agent_id)
        if count:
            logger.info("Agent locks cleaned: agent_id=%s count=%d", agent_id, count)
            self._callbacks.on_locks_cleaned(agent_id, count)
        # Unregister agent from deadlock monitor
        self._callbacks.unregister_agent(agent_id)
