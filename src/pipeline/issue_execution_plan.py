"""Unit-testable execution plan for issue coordination.

This module keeps MalaOrchestrator out of the business of assembling the
callbacks passed to IssueExecutionCoordinator. The builder is pure wiring:
it captures already-constructed dependencies and returns the callbacks the
coordinator needs for a run.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import TYPE_CHECKING, Protocol, cast

from src.pipeline.issue_result import IssueResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from src.core.protocols.issue_lifecycle_port import IssueLifecycleState
    from src.core.models import PeriodicValidationConfig, RunResult, WatchConfig
    from src.infra.io.log_output.run_metadata import RunMetadata
    from src.pipeline.issue_finalizer import IssueFinalizeOutput
    from src.pipeline.issue_execution_coordinator import (
        AbortCallback,
        AbortResult,
        FinalizeCallback,
        IssueExecutionCoordinator,
        SpawnCallback,
    )
    from src.pipeline.run_coordinator import TriggerValidationResult

logger = logging.getLogger(__name__)


class _ExecutionState(Protocol):
    """State fields needed while mapping task outcomes to issue results."""

    agent_ids: dict[str, str]
    active_session_log_paths: dict[str, Path]
    deadlock_victim_issues: set[str]


class _IssueLifecycle(Protocol):
    """Issue coordinator methods used by plan callbacks."""

    abort_reason: str | None

    @property
    def current_state(self) -> IssueLifecycleState: ...

    def release_task(self, issue_id: str) -> None: ...

    def mark_completed(self, issue_id: str) -> None: ...

    def request_abort(self, reason: str) -> None: ...


class _RunTriggerCoordinator(Protocol):
    """Run coordinator operation used after issue finalization."""

    async def run_trigger_validation(self) -> TriggerValidationResult: ...


class _StartupIssueProvider(Protocol):
    async def close_eligible_epics_async(self) -> bool: ...


class _WarningSink(Protocol):
    def on_warning(self, message: str) -> None: ...


class _StartupEpicVerifier(Protocol):
    async def verify_and_close_eligible(
        self, human_override_epic_ids: set[str]
    ) -> AnyEpicVerificationResult: ...


class AnyEpicVerificationResult(Protocol):
    verified_count: int
    remediation_issues_created: list[str]


class _AbortActiveTasks(Protocol):
    async def __call__(
        self,
        run_metadata: RunMetadata,
        *,
        is_interrupt: bool = False,
    ) -> AbortResult: ...


@dataclasses.dataclass(frozen=True)
class IssueExecutionPlan:
    """Callbacks and options required to run issue execution."""

    spawn_callback: SpawnCallback
    finalize_callback: FinalizeCallback
    abort_callback: AbortCallback
    watch_config: WatchConfig | None = None
    validation_config: PeriodicValidationConfig | None = None
    drain_event: asyncio.Event | None = None
    interrupt_event: asyncio.Event | None = None
    validation_callback: Callable[[], Awaitable[bool]] | None = None
    on_validation_failed: Callable[[], None] | None = None
    on_startup_no_ready: Callable[[], Awaitable[bool]] | None = None

    async def run(self, coordinator: IssueExecutionCoordinator) -> RunResult:
        """Run the plan through the issue coordinator."""
        return await coordinator.run_loop(
            spawn_callback=self.spawn_callback,
            finalize_callback=self.finalize_callback,
            abort_callback=self.abort_callback,
            watch_config=self.watch_config,
            validation_config=self.validation_config,
            drain_event=self.drain_event,
            interrupt_event=self.interrupt_event,
            validation_callback=self.validation_callback,
            on_validation_failed=self.on_validation_failed,
            on_startup_no_ready=self.on_startup_no_ready,
        )


def build_issue_execution_plan(
    *,
    run_metadata: RunMetadata,
    state: _ExecutionState,
    issue_coordinator: _IssueLifecycle,
    run_coordinator: _RunTriggerCoordinator,
    beads: _StartupIssueProvider,
    event_sink: _WarningSink,
    epic_verifier: _StartupEpicVerifier | None,
    epic_override_ids: set[str],
    spawn_agent: SpawnCallback,
    abort_active_tasks: _AbortActiveTasks,
    finalize_issue_result: Callable[
        [str, IssueResult, RunMetadata], Awaitable[IssueFinalizeOutput]
    ],
    cleanup_issue_runtime_state: Callable[[str], None],
    check_and_queue_periodic_trigger: Callable[[IssueResult], None],
    mark_validation_failed: Callable[[], None],
    watch_config: WatchConfig | None = None,
    validation_config: PeriodicValidationConfig | None = None,
    drain_event: asyncio.Event | None = None,
    interrupt_event: asyncio.Event | None = None,
    validation_callback: Callable[[], Awaitable[bool]] | None = None,
    await_background_repo_work: Callable[[], Awaitable[None]] | None = None,
) -> IssueExecutionPlan:
    """Build the issue execution plan for a run without touching IO."""

    async def finalize_callback(issue_id: str, task: asyncio.Task[object]) -> bool:
        result = _task_issue_result(
            issue_id,
            task,
            state=state,
            issue_coordinator=issue_coordinator,
        )
        was_deadlock_victim = issue_id in state.deadlock_victim_issues
        if was_deadlock_victim and not result.success:
            issue_coordinator.release_task(issue_id)
            cleanup_issue_runtime_state(issue_id)
            state.deadlock_victim_issues.discard(issue_id)
            return False

        state.deadlock_victim_issues.discard(issue_id)
        finalize_output = await finalize_issue_result(issue_id, result, run_metadata)
        issue_coordinator.mark_completed(issue_id)

        check_and_queue_periodic_trigger(result)
        if not issue_coordinator.current_state.active_issue_ids:
            if await_background_repo_work is not None:
                await await_background_repo_work()
            trigger_result = await run_coordinator.run_trigger_validation()
            if trigger_result.status == "aborted":
                issue_coordinator.request_abort(reason="Trigger validation aborted")
        return finalize_output.created_follow_up_work

    async def abort_callback(*, is_interrupt: bool = False) -> AbortResult:
        return await abort_active_tasks(run_metadata, is_interrupt=is_interrupt)

    async def verify_startup_eligible_epics() -> bool:
        try:
            if epic_verifier is not None:
                result = await epic_verifier.verify_and_close_eligible(
                    epic_override_ids
                )
                return (
                    result.verified_count > 0
                    or len(result.remediation_issues_created) > 0
                )

            return await beads.close_eligible_epics_async()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Startup epic verification failed")
            event_sink.on_warning(
                "Startup epic verification failed; continuing without startup repoll"
            )
            return False

    return IssueExecutionPlan(
        spawn_callback=spawn_agent,
        finalize_callback=finalize_callback,
        abort_callback=abort_callback,
        watch_config=watch_config,
        validation_config=validation_config,
        drain_event=drain_event,
        interrupt_event=interrupt_event,
        validation_callback=validation_callback,
        on_validation_failed=mark_validation_failed,
        on_startup_no_ready=verify_startup_eligible_epics,
    )


def _task_issue_result(
    issue_id: str,
    task: asyncio.Task[object],
    *,
    state: _ExecutionState,
    issue_coordinator: _IssueLifecycle,
) -> IssueResult:
    try:
        return cast("IssueResult", task.result())
    except asyncio.CancelledError:
        if issue_id in state.deadlock_victim_issues:
            summary = "Killed as deadlock victim (will retry after blocker completes)"
        elif issue_coordinator.abort_reason:
            summary = f"Aborted: {issue_coordinator.abort_reason}"
        else:
            summary = "Aborted due to task cancellation"
    except Exception as exc:
        summary = str(exc)

    return IssueResult(
        issue_id=issue_id,
        agent_id=state.agent_ids.get(issue_id, "unknown"),
        success=False,
        summary=summary,
        session_log_path=state.active_session_log_paths.get(issue_id),
    )
