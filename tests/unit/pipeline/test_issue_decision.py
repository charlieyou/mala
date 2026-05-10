from __future__ import annotations

import pytest

from src.pipeline.issue_decision import (
    ExitDecision,
    InterruptDrainDecision,
    PeriodicValidationDecision,
    exit_reason,
    interrupt_drain,
    next_validation_threshold,
    periodic_validation,
    spawn_capacity,
)
from src.pipeline.work_queue import WorkQueueSnapshot


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        (WorkQueueSnapshot(ready_issue_ids=("a", "b"), max_agents=3), 2),
        (
            WorkQueueSnapshot(
                ready_issue_ids=("a", "b", "c"),
                active_issue_ids=frozenset({"active"}),
                max_agents=2,
            ),
            1,
        ),
        (
            WorkQueueSnapshot(
                ready_issue_ids=("active",),
                active_issue_ids=frozenset({"active"}),
                max_agents=2,
            ),
            0,
        ),
        (
            WorkQueueSnapshot(
                ready_issue_ids=("a", "b", "c"),
                completed_count=4,
                max_issues=5,
            ),
            1,
        ),
        (
            WorkQueueSnapshot(
                ready_issue_ids=("a",),
                completed_count=4,
                active_issue_ids=frozenset({"active"}),
                max_issues=5,
            ),
            0,
        ),
        (WorkQueueSnapshot(ready_issue_ids=("a",), is_draining=True), 0),
        (WorkQueueSnapshot(ready_issue_ids=("a",), interrupt_requested=True), 0),
        (WorkQueueSnapshot(ready_issue_ids=("a",), abort_requested=True), 0),
    ],
)
def test_spawn_capacity(snapshot: WorkQueueSnapshot, expected: int) -> None:
    assert spawn_capacity(snapshot) == expected


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        (
            WorkQueueSnapshot(interrupt_requested=True),
            ExitDecision(True, "interrupted", 130),
        ),
        (WorkQueueSnapshot(abort_requested=True), ExitDecision(True, "abort", 3)),
        (
            WorkQueueSnapshot(
                is_draining=True, completed_count=2, last_validation_at=1
            ),
            ExitDecision(True, "drained", 0, run_final_validation=True),
        ),
        (
            WorkQueueSnapshot(consecutive_poll_failures=3),
            ExitDecision(True, "poll_failed", 3),
        ),
        (
            WorkQueueSnapshot(
                consecutive_poll_failures=3,
                active_issue_ids=frozenset({"active"}),
            ),
            ExitDecision(
                True,
                "poll_failed",
                3,
                wait_for_active_work=True,
            ),
        ),
        (
            WorkQueueSnapshot(completed_count=3, max_issues=3, last_validation_at=2),
            ExitDecision(True, "limit_reached", 0, run_final_validation=True),
        ),
        (
            WorkQueueSnapshot(
                completed_count=1,
                last_validation_at=0,
                active_issue_ids=frozenset({"active"}),
            ),
            ExitDecision(False),
        ),
        (WorkQueueSnapshot(ready_issue_ids=("ready",)), ExitDecision(False)),
        (WorkQueueSnapshot(watch_enabled=True), ExitDecision(False)),
        (
            WorkQueueSnapshot(startup_no_ready_check_pending=True),
            ExitDecision(False),
        ),
        (
            WorkQueueSnapshot(completed_count=1, last_validation_at=0),
            ExitDecision(True, "success", 0, run_final_validation=True),
        ),
    ],
)
def test_exit_reason(snapshot: WorkQueueSnapshot, expected: ExitDecision) -> None:
    assert exit_reason(snapshot) == expected


@pytest.mark.parametrize(
    ("snapshot", "validate_every", "expected"),
    [
        (
            WorkQueueSnapshot(completed_count=3, next_validation_threshold=3),
            None,
            False,
        ),
        (WorkQueueSnapshot(completed_count=2, next_validation_threshold=3), 3, False),
        (
            WorkQueueSnapshot(completed_count=3, next_validation_threshold=3),
            3,
            PeriodicValidationDecision(True, threshold=3),
        ),
        (
            WorkQueueSnapshot(completed_count=5),
            5,
            PeriodicValidationDecision(True, threshold=5),
        ),
        (
            WorkQueueSnapshot(
                completed_count=3,
                next_validation_threshold=3,
                active_issue_ids=frozenset({"active"}),
            ),
            3,
            PeriodicValidationDecision(True, block_for_active_work=True, threshold=3),
        ),
    ],
)
def test_periodic_validation(
    snapshot: WorkQueueSnapshot,
    validate_every: int | None,
    expected: bool | PeriodicValidationDecision,
) -> None:
    decision = periodic_validation(snapshot, validate_every)
    if isinstance(expected, bool):
        assert decision.should_validate is expected
    else:
        assert decision == expected


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        (WorkQueueSnapshot(), InterruptDrainDecision(False)),
        (
            WorkQueueSnapshot(
                is_draining=True,
                active_issue_ids=frozenset({"active"}),
                completed_count=1,
                last_validation_at=0,
            ),
            InterruptDrainDecision(True),
        ),
        (
            WorkQueueSnapshot(
                is_draining=True,
                completed_count=2,
                last_validation_at=1,
            ),
            InterruptDrainDecision(
                True, run_final_validation=True, exit_when_validation_passes=True
            ),
        ),
        (
            WorkQueueSnapshot(
                is_draining=True,
                completed_count=2,
                last_validation_at=2,
            ),
            InterruptDrainDecision(True, exit_when_validation_passes=True),
        ),
    ],
)
def test_interrupt_drain(
    snapshot: WorkQueueSnapshot, expected: InterruptDrainDecision
) -> None:
    assert interrupt_drain(snapshot) == expected


@pytest.mark.parametrize(
    ("completed_count", "threshold", "interval", "expected"),
    [(0, 3, 3, 3), (3, 3, 3, 6), (8, 3, 3, 9), (9, 3, 3, 12)],
)
def test_next_validation_threshold(
    completed_count: int, threshold: int, interval: int, expected: int
) -> None:
    assert next_validation_threshold(completed_count, threshold, interval) == expected
