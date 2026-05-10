from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.core.models import OrderPreference
from src.pipeline.work_queue import (
    PollStrategy,
    WorkQueue,
    WorkQueueConfig,
)

if TYPE_CHECKING:
    from src.pipeline.work_queue import WorkQueueSnapshot


class FakeIssueProvider:
    def __init__(
        self, ready: list[str] | None = None, error: Exception | None = None
    ) -> None:
        self.ready = ready or []
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def get_ready_async(
        self,
        exclude_ids: set[str] | None = None,
        epic_id: str | None = None,
        only_ids: list[str] | None = None,
        suppress_warn_ids: set[str] | None = None,
        include_wip: bool = False,
        focus: bool = True,
        orphans_only: bool = False,
        order_preference: OrderPreference = OrderPreference.EPIC_PRIORITY,
    ) -> list[str]:
        self.calls.append(
            {
                "exclude_ids": exclude_ids,
                "epic_id": epic_id,
                "only_ids": only_ids,
                "suppress_warn_ids": suppress_warn_ids,
                "include_wip": include_wip,
                "focus": focus,
                "orphans_only": orphans_only,
                "order_preference": order_preference,
            }
        )
        if self.error is not None:
            raise self.error
        return list(self.ready)


class RecordingPollStrategy(PollStrategy):
    def __init__(self) -> None:
        self.failure_waits: list[tuple[WorkQueueSnapshot, Exception]] = []
        self.idle_waits: list[WorkQueueSnapshot] = []

    async def wait_after_poll_failure(
        self, snapshot: WorkQueueSnapshot, error: Exception
    ) -> None:
        self.failure_waits.append((snapshot, error))

    async def wait_until_next_poll(self, snapshot: WorkQueueSnapshot) -> None:
        self.idle_waits.append(snapshot)


def make_queue(
    provider: FakeIssueProvider, strategy: RecordingPollStrategy
) -> WorkQueue:
    return WorkQueue(
        provider,
        WorkQueueConfig(
            max_agents=2,
            max_issues=5,
            epic_id="epic-1",
            only_ids=["issue-1", "issue-2"],
            include_wip=True,
            focus=False,
            orphans_only=True,
            order_preference=OrderPreference.INPUT,
        ),
        strategy,
    )


def test_snapshot_freezes_coordinator_state_and_defaults_completed_count() -> None:
    provider = FakeIssueProvider()
    strategy = RecordingPollStrategy()
    queue = make_queue(provider, strategy)

    snapshot = queue.snapshot(
        active_issue_ids={"active-1"},
        completed_issue_ids={"done-1", "done-2"},
        failed_issue_ids={"failed-1"},
        ready_issue_ids=["ready-1"],
        watch_enabled=True,
        startup_no_ready_check_pending=True,
    )

    assert snapshot.ready_issue_ids == ("ready-1",)
    assert snapshot.active_issue_ids == frozenset({"active-1"})
    assert snapshot.completed_count == 2
    assert snapshot.max_agents == 2
    assert snapshot.max_issues == 5
    assert snapshot.watch_enabled is True
    assert snapshot.startup_no_ready_check_pending is True


@pytest.mark.asyncio
async def test_poll_fetches_ready_issues_and_resets_failures() -> None:
    provider = FakeIssueProvider(ready=["issue-1", "issue-2"])
    strategy = RecordingPollStrategy()
    queue = make_queue(provider, strategy)
    snapshot = queue.snapshot(
        active_issue_ids={"active-1"},
        completed_issue_ids={"done-1"},
        failed_issue_ids={"failed-1"},
        consecutive_poll_failures=2,
    )

    result = await queue.poll(snapshot)

    assert result.succeeded is True
    assert result.ready_issue_ids == ("issue-1", "issue-2")
    assert result.consecutive_poll_failures == 0
    assert provider.calls == [
        {
            "exclude_ids": {"failed-1", "done-1"},
            "epic_id": "epic-1",
            "only_ids": ["issue-1", "issue-2"],
            "suppress_warn_ids": {"active-1", "failed-1", "done-1"},
            "include_wip": True,
            "focus": False,
            "orphans_only": True,
            "order_preference": OrderPreference.INPUT,
        }
    ]
    assert strategy.failure_waits == []
    assert strategy.idle_waits == []


@pytest.mark.asyncio
async def test_poll_suppressed_when_limit_reached_or_draining() -> None:
    provider = FakeIssueProvider(ready=["issue-1"])
    strategy = RecordingPollStrategy()
    queue = make_queue(provider, strategy)

    limited = await queue.poll(queue.snapshot(completed_count=5))
    draining = await queue.poll(queue.snapshot(is_draining=True))

    assert limited.poll_attempted is False
    assert limited.ready_issue_ids == ()
    assert draining.poll_attempted is False
    assert draining.ready_issue_ids == ()
    assert provider.calls == []
    assert strategy.failure_waits == []
    assert strategy.idle_waits == []


@pytest.mark.asyncio
async def test_poll_failure_records_error_and_uses_strategy_wait() -> None:
    error = RuntimeError("provider unavailable")
    provider = FakeIssueProvider(error=error)
    strategy = RecordingPollStrategy()
    queue = make_queue(provider, strategy)

    result = await queue.poll(queue.snapshot(consecutive_poll_failures=1))

    assert result.succeeded is False
    assert result.error is error
    assert result.ready_issue_ids == ()
    assert result.consecutive_poll_failures == 2
    assert strategy.failure_waits == [(result.snapshot, error)]
    assert strategy.idle_waits == []


@pytest.mark.asyncio
async def test_empty_watch_poll_uses_strategy_idle_wait_without_active_work() -> None:
    provider = FakeIssueProvider(ready=[])
    strategy = RecordingPollStrategy()
    queue = make_queue(provider, strategy)
    snapshot = queue.snapshot(watch_enabled=True)

    result = await queue.poll(snapshot)

    assert result.succeeded is True
    assert result.ready_issue_ids == ()
    assert strategy.idle_waits == [result.snapshot]


@pytest.mark.asyncio
async def test_empty_startup_watch_poll_defers_idle_wait_for_startup_check() -> None:
    provider = FakeIssueProvider(ready=[])
    strategy = RecordingPollStrategy()
    queue = make_queue(provider, strategy)
    snapshot = queue.snapshot(watch_enabled=True, startup_no_ready_check_pending=True)

    result = await queue.poll(snapshot)

    assert result.succeeded is True
    assert result.ready_issue_ids == ()
    assert strategy.idle_waits == []


@pytest.mark.asyncio
async def test_empty_poll_with_active_work_does_not_idle_wait() -> None:
    provider = FakeIssueProvider(ready=[])
    strategy = RecordingPollStrategy()
    queue = make_queue(provider, strategy)

    result = await queue.poll(
        queue.snapshot(active_issue_ids={"active-1"}, watch_enabled=True)
    )

    assert result.succeeded is True
    assert strategy.idle_waits == []
