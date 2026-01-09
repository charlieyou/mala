"""EpicVerificationCoordinator: Epic closure verification pipeline stage.

Extracted from MalaOrchestrator to separate epic verification logic from orchestration.
This module handles:
- Checking if closing an issue should close its parent epic
- Retry loop for epic verification with interrupt support
- Executing remediation issues when verification fails
- Graceful cancellation of remediation tasks on interrupt

Design principles:
- Protocol-based callbacks for orchestrator-owned operations
- State management: verified_epics, epics_being_verified sets
- Explicit config for retry behavior
- InterruptGuard for consistent interrupt checking
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.core.models import EpicVerificationResult
    from src.domain.validation.config import EpicCompletionTriggerConfig, TriggerType
    from src.infra.io.log_output.run_metadata import RunMetadata
    from src.pipeline.issue_result import IssueResult

logger = logging.getLogger(__name__)


@dataclass
class EpicVerificationConfig:
    """Configuration for EpicVerificationCoordinator.

    Attributes:
        max_retries: Maximum number of retry attempts after initial verification.
    """

    max_retries: int = 0


@dataclass
class EpicVerificationCallbacks:
    """Callbacks for orchestrator-owned operations during epic verification.

    These callbacks allow the coordinator to trigger orchestrator operations
    without taking dependencies on orchestrator internals.

    Attributes:
        get_parent_epic: Get the parent epic ID for an issue.
        verify_epic: Run epic verification, returns verification result.
        spawn_remediation: Spawn an agent for a remediation issue (issue_id, flow).
        finalize_remediation: Finalize a remediation issue result.
        mark_completed: Mark an issue as completed in the coordinator.
        is_issue_failed: Check if an issue has failed.
        close_eligible_epics: Fallback for mock providers without EpicVerifier.
        on_epic_closed: Emit epic closed event.
        on_warning: Emit warning event.
        has_epic_verifier: Check if an EpicVerifier is available (callable for test patching).
        get_agent_id: Get the agent ID for an issue (for error attribution).
        queue_trigger_validation: Queue a trigger for validation (trigger_type, context).
        get_epic_completion_trigger: Get the epic_completion trigger config (or None).
    """

    get_parent_epic: Callable[[str], Awaitable[str | None]]
    verify_epic: Callable[[str, bool], Awaitable[EpicVerificationResult]]
    spawn_remediation: Callable[[str, str], Awaitable[asyncio.Task[IssueResult] | None]]
    finalize_remediation: Callable[[str, IssueResult, RunMetadata], Awaitable[None]]
    mark_completed: Callable[[str], None]
    is_issue_failed: Callable[[str], bool]
    close_eligible_epics: Callable[[], Awaitable[bool]]
    on_epic_closed: Callable[[str], None]
    on_warning: Callable[[str], None]
    has_epic_verifier: Callable[[], bool]
    get_agent_id: Callable[[str], str]
    queue_trigger_validation: Callable[[TriggerType, dict[str, Any]], None]
    get_epic_completion_trigger: Callable[[], EpicCompletionTriggerConfig | None]


@dataclass
class EpicVerificationCoordinator:
    """Epic verification pipeline stage.

    This class encapsulates the epic closure verification logic that was previously
    inline in MalaOrchestrator._check_epic_closure. It manages:
    - Tracking verified epics to avoid re-verification
    - Re-entrant guard for epics being verified
    - Retry loop with remediation issue execution
    - Fallback for mock providers

    Attributes:
        config: Verification configuration.
        callbacks: Callbacks for orchestrator-owned operations.
        epic_override_ids: Set of epic IDs to force human override.
    """

    config: EpicVerificationConfig
    callbacks: EpicVerificationCallbacks
    epic_override_ids: set[str] = field(default_factory=set)

    # State: Tracked across multiple check_epic_closure calls
    verified_epics: set[str] = field(default_factory=set)
    epics_being_verified: set[str] = field(default_factory=set)

    async def check_epic_closure(
        self,
        issue_id: str,
        run_metadata: RunMetadata,
        *,
        interrupt_event: asyncio.Event | None = None,
    ) -> None:
        """Check if closing this issue should also close its parent epic.

        Implements a retry loop for epic verification:
        1. Run verification
        2. If verification fails and creates remediation issues, execute them
        3. Re-verify the epic
        4. Repeat until verification passes OR max retries reached

        Args:
            issue_id: The issue that was just closed.
            run_metadata: Run metadata for recording remediation issue results.
            interrupt_event: Optional event to signal interrupt. When set, the
                verification loop exits early and remediation tasks are cancelled.
        """
        parent_epic = await self.callbacks.get_parent_epic(issue_id)
        if parent_epic is None or parent_epic in self.verified_epics:
            return

        # Guard against re-entrant verification (e.g., when remediation tasks complete)
        if parent_epic in self.epics_being_verified:
            return

        if self.callbacks.has_epic_verifier():
            # Mark as being verified to prevent parallel verification loops
            self.epics_being_verified.add(parent_epic)
            try:
                await self._verify_epic_with_retries(
                    parent_epic, run_metadata, interrupt_event=interrupt_event
                )
            finally:
                # Always remove from being_verified set when done
                self.epics_being_verified.discard(parent_epic)

        elif await self.callbacks.close_eligible_epics():
            # Fallback for mock providers without EpicVerifier
            self.callbacks.on_epic_closed(issue_id)

    async def _verify_epic_with_retries(
        self,
        epic_id: str,
        run_metadata: RunMetadata,
        *,
        interrupt_event: asyncio.Event | None = None,
    ) -> None:
        """Run epic verification with retry loop.

        Args:
            epic_id: The epic to verify.
            run_metadata: Run metadata for recording remediation issue results.
            interrupt_event: Optional event to signal interrupt.
        """
        from src.infra.sigint_guard import InterruptGuard

        guard = InterruptGuard(interrupt_event)

        # max_retries is the number of retries AFTER the first attempt
        # So total attempts = 1 (initial) + max_retries
        max_retries = self.config.max_retries
        max_attempts = 1 + max_retries
        human_override = epic_id in self.epic_override_ids

        for attempt in range(1, max_attempts + 1):
            # Check for interrupt before each retry iteration
            if guard.is_interrupted():
                return

            # Log attempt if retrying (attempt > 1)
            if attempt > 1:
                self.callbacks.on_warning(
                    f"Epic verification retry {attempt - 1}/{max_retries} for {epic_id}"
                )

            verification_result = await self.callbacks.verify_epic(
                epic_id, human_override
            )

            # If epic wasn't eligible (children still open), don't mark as verified
            # so it can be re-checked when more children close
            if verification_result.verified_count == 0:
                return

            # If epic passed verification, mark as verified and return
            if verification_result.passed_count > 0:
                self.verified_epics.add(epic_id)
                # Queue epic_completion trigger on success
                await self._maybe_queue_epic_completion_trigger(
                    epic_id, verification_passed=True
                )
                return

            # If no remediation issues were created, or max attempts reached,
            # mark as verified (to prevent infinite loops) and return
            if (
                not verification_result.remediation_issues_created
                or attempt >= max_attempts
            ):
                if attempt >= max_attempts and verification_result.failed_count > 0:
                    self.callbacks.on_warning(
                        f"Epic verification failed after {max_retries} retries for {epic_id}"
                    )
                self.verified_epics.add(epic_id)
                # Queue epic_completion trigger on failure (verification completed but didn't pass)
                await self._maybe_queue_epic_completion_trigger(
                    epic_id, verification_passed=False
                )
                return

            # Execute remediation issues before next verification attempt
            await self._execute_remediation_issues(
                verification_result.remediation_issues_created,
                run_metadata,
                interrupt_event=interrupt_event,
            )

    async def _maybe_queue_epic_completion_trigger(
        self,
        epic_id: str,
        *,
        verification_passed: bool,
    ) -> None:
        """Queue epic_completion trigger if filters pass.

        Checks epic_depth and fire_on filters from the trigger config.
        If all filters pass, queues the trigger for validation.

        Args:
            epic_id: The epic that completed verification.
            verification_passed: Whether the epic passed verification.
        """
        from src.domain.validation.config import EpicDepth, FireOn, TriggerType

        trigger_config = self.callbacks.get_epic_completion_trigger()
        if trigger_config is None:
            return

        # Cache parent lookup to avoid double await
        epic_parent = await self.callbacks.get_parent_epic(epic_id)

        # Check epic_depth filter
        if trigger_config.epic_depth == EpicDepth.TOP_LEVEL:
            if epic_parent is not None:
                # Nested epic - don't fire for top_level filter
                return

        # Check fire_on filter: SUCCESS fires only on pass, FAILURE only on fail, BOTH always
        if trigger_config.fire_on == FireOn.SUCCESS and not verification_passed:
            return
        if trigger_config.fire_on == FireOn.FAILURE and verification_passed:
            return

        # Build context and queue the trigger
        context = {
            "epic_id": epic_id,
            "depth": "top_level" if epic_parent is None else "nested",
            "verification_result": "passed" if verification_passed else "failed",
        }
        self.callbacks.queue_trigger_validation(TriggerType.EPIC_COMPLETION, context)

    async def _execute_remediation_issues(
        self,
        issue_ids: list[str],
        run_metadata: RunMetadata,
        *,
        interrupt_event: asyncio.Event | None = None,
    ) -> None:
        """Execute remediation issues and wait for their completion.

        Spawns agents for remediation issues, waits for completion, and finalizes
        results (closes issues, records metadata). This ensures remediation issues
        are properly tracked even though they bypass the main run_loop.

        Args:
            issue_ids: List of remediation issue IDs to execute.
            run_metadata: Run metadata for recording issue results.
            interrupt_event: Optional event to signal interrupt. When set,
                spawning stops and pending tasks are cancelled.
        """
        from src.infra.sigint_guard import InterruptGuard

        guard = InterruptGuard(interrupt_event)

        if not issue_ids:
            return

        # Track (issue_id, task) pairs for finalization
        task_pairs: list[tuple[str, asyncio.Task[IssueResult]]] = []

        for issue_id in issue_ids:
            # Check for interrupt before spawning each remediation task
            if guard.is_interrupted():
                break

            # Skip if already failed (remediation issues are freshly created, so won't be completed)
            if self.callbacks.is_issue_failed(issue_id):
                continue

            # Spawn agent for this issue with flow identifier for logging.
            # The flow="epic_remediation" param propagates via AgentSessionInput.flow
            # to MALA_SDK_FLOW env var, which sdk_transport.py reads to emit:
            # logger.info("sdk_subprocess_spawned pid=%s pgid=%s flow=%s", pid, pgid, flow)
            task = await self.callbacks.spawn_remediation(issue_id, "epic_remediation")
            if task:
                task_pairs.append((issue_id, task))

        # Wait for all remediation tasks to complete
        if not task_pairs:
            return

        tasks = [pair[1] for pair in task_pairs]

        # If already interrupted, cancel all pending tasks
        if guard.is_interrupted():
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for cancellation to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        elif interrupt_event is None:
            # No interrupt event provided - just wait for all tasks normally
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Race between interrupt event and task completion
            async def wait_for_interrupt() -> None:
                await interrupt_event.wait()

            async def gather_tasks() -> list[IssueResult | BaseException]:
                return await asyncio.gather(*tasks, return_exceptions=True)

            interrupt_task = asyncio.create_task(wait_for_interrupt())
            gather_task = asyncio.create_task(gather_tasks())

            done, pending = await asyncio.wait(
                [interrupt_task, gather_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # If interrupt happened first, cancel all pending remediation tasks
            if interrupt_task in done and gather_task in pending:
                # Cancel gather_task first to stop it from holding refs
                gather_task.cancel()
                # Then cancel all underlying remediation tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Wait for all tasks to complete cancellation
                await asyncio.gather(*tasks, return_exceptions=True)
                # Clean up gather_task
                try:
                    await gather_task
                except asyncio.CancelledError:
                    pass

            # Clean up interrupt task if it's still pending
            if interrupt_task in pending:
                interrupt_task.cancel()
                try:
                    await interrupt_task
                except asyncio.CancelledError:
                    pass

        # Finalize each task result (close issue, record metadata, emit events)
        for issue_id, task in task_pairs:
            result = self._extract_task_result(issue_id, task)

            # Finalize (closes issue, records to run_metadata, emits events)
            # Wrap in try/except to ensure all issues are finalized even if one fails
            try:
                await self.callbacks.finalize_remediation(
                    issue_id, result, run_metadata
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.callbacks.on_warning(
                    f"Failed to finalize remediation result for {issue_id} "
                    f"(agent: {result.agent_id}): {e}",
                )

            # Mark as completed in the coordinator
            self.callbacks.mark_completed(issue_id)

    def _extract_task_result(
        self, issue_id: str, task: asyncio.Task[IssueResult]
    ) -> IssueResult:
        """Extract result from a completed task, handling exceptions.

        Args:
            issue_id: The issue ID for error results.
            task: The completed task.

        Returns:
            The task result or an error IssueResult.
        """
        # Import here to avoid circular dependency
        from src.pipeline.issue_result import IssueResult

        try:
            return task.result()
        except asyncio.CancelledError:
            return IssueResult(
                issue_id=issue_id,
                agent_id=self.callbacks.get_agent_id(issue_id),
                success=False,
                summary="Remediation task was cancelled",
            )
        except Exception as e:
            return IssueResult(
                issue_id=issue_id,
                agent_id=self.callbacks.get_agent_id(issue_id),
                success=False,
                summary=str(e),
            )
