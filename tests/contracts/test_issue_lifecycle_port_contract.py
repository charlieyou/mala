"""Contract tests for IssueLifecyclePort implementations."""

from __future__ import annotations

import asyncio

import pytest

from src.core.protocols.issue_lifecycle_port import (
    IssueLifecycleEffect,
    IssueLifecyclePort,
)
from src.pipeline.issue_execution_coordinator import (
    CoordinatorConfig,
    IssueExecutionCoordinator,
)
from tests.contracts import get_protocol_members
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.issue_lifecycle_port import FakeIssueLifecyclePort
from tests.fakes.issue_provider import FakeIssueProvider


def _make_real_port() -> IssueExecutionCoordinator:
    return IssueExecutionCoordinator(
        beads=FakeIssueProvider(),
        event_sink=FakeEventSink(),
        config=CoordinatorConfig(),
    )


@pytest.fixture(params=["real", "fake"])
def lifecycle_port(request: pytest.FixtureRequest) -> IssueLifecyclePort:
    """Create each IssueLifecyclePort implementation under test."""
    if request.param == "real":
        return _make_real_port()
    return FakeIssueLifecyclePort()


@pytest.mark.unit
def test_issue_lifecycle_port_declares_expected_surface() -> None:
    """The port exposes the lifecycle delegation surface needed by orchestrator."""
    members = get_protocol_members(IssueLifecyclePort)

    assert members == frozenset(
        {
            "current_state",
            "apply_effect",
            "active_tasks",
            "failed_issues",
            "abort_requested",
            "interrupt_event",
            "max_issues",
        }
    )


@pytest.mark.unit
def test_real_and_fake_are_runtime_ports(lifecycle_port: IssueLifecyclePort) -> None:
    """Both implementations satisfy the runtime-checkable protocol."""
    assert isinstance(lifecycle_port, IssueLifecyclePort)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_state_snapshot_reflects_task_failure_abort_and_limit(
    lifecycle_port: IssueLifecyclePort,
) -> None:
    """current_state mirrors the underlying lifecycle properties."""

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(wait_forever())
    lifecycle_port.active_tasks["issue-active"] = task

    try:
        lifecycle_port.apply_effect(
            IssueLifecycleEffect(kind="mark_failed", issue_id="issue-failed")
        )
        lifecycle_port.apply_effect(
            IssueLifecycleEffect(kind="set_max_issues", max_issues=2)
        )
        lifecycle_port.apply_effect(
            IssueLifecycleEffect(kind="request_abort", reason="fatal error")
        )

        state = lifecycle_port.current_state

        assert state.active_issue_ids == frozenset({"issue-active"})
        assert state.failed_issues == frozenset({"issue-failed"})
        assert state.abort_requested is True
        assert state.abort_reason == "fatal error"
        assert state.max_issues == 2
        assert lifecycle_port.abort_requested is True
        assert lifecycle_port.max_issues == 2
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_completion_and_release_effects_remove_active_tasks(
    lifecycle_port: IssueLifecyclePort,
) -> None:
    """Terminal task effects remove the issue from active task tracking."""

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    completed_task = asyncio.create_task(wait_forever())
    released_task = asyncio.create_task(wait_forever())
    lifecycle_port.active_tasks["issue-complete"] = completed_task
    lifecycle_port.active_tasks["issue-release"] = released_task

    try:
        lifecycle_port.apply_effect(
            IssueLifecycleEffect(kind="mark_completed", issue_id="issue-complete")
        )
        lifecycle_port.apply_effect(
            IssueLifecycleEffect(kind="release_task", issue_id="issue-release")
        )

        assert lifecycle_port.current_state.active_issue_ids == frozenset()
    finally:
        completed_task.cancel()
        released_task.cancel()
        await asyncio.gather(completed_task, released_task, return_exceptions=True)


@pytest.mark.unit
def test_interrupt_event_can_be_replaced_and_observed(
    lifecycle_port: IssueLifecyclePort,
) -> None:
    """The interrupt event is replaceable and represented in current_state."""
    interrupt_event = asyncio.Event()

    lifecycle_port.interrupt_event = interrupt_event
    assert lifecycle_port.interrupt_event is interrupt_event
    assert lifecycle_port.current_state.interrupt_requested is False

    interrupt_event.set()
    assert lifecycle_port.current_state.interrupt_requested is True

    replacement = asyncio.Event()
    lifecycle_port.apply_effect(
        IssueLifecycleEffect(kind="set_interrupt_event", interrupt_event=replacement)
    )
    assert lifecycle_port.interrupt_event is replacement
    assert lifecycle_port.current_state.interrupt_requested is False
