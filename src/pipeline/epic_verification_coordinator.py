"""EpicVerificationCoordinator: Epic closure verification pipeline stage.

Extracted from MalaOrchestrator to separate epic verification logic from orchestration.
This module handles:
- Checking if closing an issue should close its parent epic
- Retry loop for epic verification with interrupt support
- Executing remediation issues when verification fails
- Graceful cancellation of remediation tasks on interrupt

Design principles:
- Protocol ports for issue/event/lifecycle operations
- State management: verified_epics, epics_being_verified sets
- Explicit config for retry behavior
- InterruptGuard for consistent interrupt checking
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.core.protocols.issue_lifecycle_port import IssueLifecycleEffect

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.core.models import EpicVerificationResult
    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.issue import IssueProvider
    from src.core.protocols.issue_lifecycle_port import IssueLifecyclePort
    from src.core.protocols.validation import EpicVerifierProtocol
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
class EpicVerificationCoordinator:
    """Epic verification pipeline stage.

    This class encapsulates the epic closure verification logic that was previously
    inline in MalaOrchestrator._check_epic_closure. It manages:
    - Tracking verified epics to avoid re-verification
    - Re-entrant guard for epics being verified
    - Retry loop with remediation issue execution

    Attributes:
        config: Verification configuration.
        issue_provider: Issue tracking port.
        event_sink: Event emission port.
        issue_lifecycle: Issue lifecycle state/effect port.
        epic_verifier_getter: Returns the current epic verifier, if configured.
        spawn_remediation: Spawn an agent for a remediation issue.
        finalize_remediation: Finalize a remediation issue result.
        get_agent_id: Get an agent ID for fallback error attribution.
        queue_trigger_validation: Queue a trigger validation request.
        get_epic_completion_trigger: Return epic_completion trigger config, if any.
        epic_override_ids: Set of epic IDs to force human override.
    """

    config: EpicVerificationConfig
    issue_provider: IssueProvider
    event_sink: MalaEventSink
    issue_lifecycle: IssueLifecyclePort
    epic_verifier_getter: Callable[[], EpicVerifierProtocol | None]
    spawn_remediation: Callable[[str, str], Awaitable[asyncio.Task[IssueResult] | None]]
    finalize_remediation: Callable[[str, IssueResult, RunMetadata], Awaitable[None]]
    get_agent_id: Callable[[str], str]
    queue_trigger_validation: Callable[[TriggerType, dict[str, Any]], None]
    get_epic_completion_trigger: Callable[[], EpicCompletionTriggerConfig | None]
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
        parent_epic = await self.issue_provider.get_parent_epic_async(issue_id)
        if parent_epic is None or parent_epic in self.verified_epics:
            return

        # Guard against re-entrant verification (e.g., when remediation tasks complete)
        if parent_epic in self.epics_being_verified:
            return

        if not self._has_epic_verifier():
            return

        # Mark as being verified to prevent parallel verification loops
        self.epics_being_verified.add(parent_epic)
        try:
            await self._verify_epic_with_retries(
                parent_epic, run_metadata, interrupt_event=interrupt_event
            )
        finally:
            # Always remove from being_verified set when done
            self.epics_being_verified.discard(parent_epic)

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
                self.event_sink.on_warning(
                    f"Epic verification retry {attempt - 1}/{max_retries} for {epic_id}"
                )

            verification_result = await self._verify_epic(epic_id, human_override)

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
                    self.event_sink.on_warning(
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

        trigger_config = self.get_epic_completion_trigger()
        if trigger_config is None:
            return

        # Cache parent lookup to avoid double await
        epic_parent = await self.issue_provider.get_parent_epic_async(epic_id)

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
        self.queue_trigger_validation(TriggerType.EPIC_COMPLETION, context)

    async def _execute_remediation_issues(
        self,
        issue_ids: list[str],
        run_metadata: RunMetadata,
        *,
        interrupt_event: asyncio.Event | None = None,
    ) -> None:
        """Execute remediation issues sequentially, respecting dependency chain.

        Remediation issues are created with sequential dependencies (each depends
        on the previous). This method executes them one at a time, waiting for
        each to complete before starting the next. If an issue fails, the chain
        stops since downstream issues depend on the failed one.

        Args:
            issue_ids: List of remediation issue IDs to execute (in dependency order).
            run_metadata: Run metadata for recording issue results.
            interrupt_event: Optional event to signal interrupt. When set,
                execution stops and the current task is cancelled.
        """
        from src.infra.sigint_guard import InterruptGuard

        guard = InterruptGuard(interrupt_event)

        if not issue_ids:
            return

        for idx, issue_id in enumerate(issue_ids):
            if guard.is_interrupted():
                return

            remaining = issue_ids[idx + 1 :]

            if issue_id in self.issue_lifecycle.failed_issues:
                if remaining:
                    self.event_sink.on_warning(
                        f"Stopping remediation chain: {issue_id} already failed; "
                        f"{len(remaining)} downstream issue(s) remain blocked"
                    )
                return

            try:
                task = await self.spawn_remediation(issue_id, "epic_remediation")
            except Exception as e:
                if remaining:
                    self.event_sink.on_warning(
                        f"Stopping remediation chain: failed to spawn {issue_id}: {e}; "
                        f"{len(remaining)} downstream issue(s) remain blocked"
                    )
                return

            if not task:
                if remaining:
                    self.event_sink.on_warning(
                        f"Stopping remediation chain: no task for {issue_id}; "
                        f"{len(remaining)} downstream issue(s) remain blocked"
                    )
                return

            result = await self._wait_for_task(task, issue_id, interrupt_event)

            try:
                await self.finalize_remediation(issue_id, result, run_metadata)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.event_sink.on_warning(
                    f"Failed to finalize remediation result for {issue_id} "
                    f"(agent: {result.agent_id}): {e}",
                )

            self.issue_lifecycle.apply_effect(
                IssueLifecycleEffect(kind="mark_completed", issue_id=issue_id)
            )

            if not result.success:
                if remaining:
                    self.event_sink.on_warning(
                        f"Stopping remediation chain: {issue_id} failed; "
                        f"{len(remaining)} downstream issue(s) remain blocked"
                    )
                return

    async def _wait_for_task(
        self,
        task: asyncio.Task[IssueResult],
        issue_id: str,
        interrupt_event: asyncio.Event | None,
    ) -> IssueResult:
        """Wait for a single task with interrupt support.

        Args:
            task: The task to wait for.
            issue_id: The issue ID for error results.
            interrupt_event: Optional event to signal interrupt.

        Returns:
            The task result or an error IssueResult if cancelled/failed.
        """
        if interrupt_event is None:
            await asyncio.wait([task])
        else:
            interrupt_task = asyncio.create_task(interrupt_event.wait())
            done, pending = await asyncio.wait(
                [task, interrupt_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if interrupt_task in done and task in pending:
                task.cancel()
                await asyncio.wait([task])

            if interrupt_task in pending:
                interrupt_task.cancel()
                try:
                    await interrupt_task
                except asyncio.CancelledError:
                    pass

        return self._extract_task_result(issue_id, task)

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
                agent_id=self.get_agent_id(issue_id),
                success=False,
                summary="Remediation task was cancelled",
            )
        except Exception as e:
            return IssueResult(
                issue_id=issue_id,
                agent_id=self.get_agent_id(issue_id),
                success=False,
                summary=str(e),
            )

    def _has_epic_verifier(self) -> bool:
        """Return whether an epic verifier is configured."""
        return self.epic_verifier_getter() is not None

    async def _verify_epic(
        self, epic_id: str, human_override: bool
    ) -> EpicVerificationResult:
        """Run epic verification through the configured verifier."""
        epic_verifier = self.epic_verifier_getter()
        if epic_verifier is None:
            raise RuntimeError("Epic verifier is not available")
        return await epic_verifier.verify_and_close_epic(
            epic_id,
            human_override=human_override,
        )
