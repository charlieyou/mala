"""Pure scheduling decisions for issue execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.pipeline.work_queue import WorkQueueSnapshot

ExitReason = Literal[
    "abort",
    "interrupted",
    "drained",
    "limit_reached",
    "success",
    "poll_failed",
]


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


def spawn_capacity(snapshot: WorkQueueSnapshot) -> int:
    """Return how many ready issues may be spawned from this snapshot."""

    if (
        snapshot.abort_requested
        or snapshot.interrupt_requested
        or snapshot.is_draining
        or snapshot.issue_limit_reached
        or snapshot.spawn_limit_reached
    ):
        return 0

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
    return max(0, agent_capacity)


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
