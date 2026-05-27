"""Tests for issue execution plan callback assembly."""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from unittest.mock import MagicMock

import pytest

from src.core.models import EpicVerificationResult, RunResult
from src.core.protocols.issue_lifecycle_port import IssueLifecycleState
from src.pipeline.issue_execution_coordinator import AbortResult
from src.pipeline.issue_execution_plan import (
    IssueExecutionPlan,
    build_issue_execution_plan,
)
from src.pipeline.issue_finalizer import IssueFinalizeOutput
from src.pipeline.issue_result import IssueResult
from src.pipeline.run_coordinator import TriggerValidationResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.pipeline.issue_execution_coordinator import IssueExecutionCoordinator


@dataclasses.dataclass
class FakeState:
    agent_ids: dict[str, str] = dataclasses.field(default_factory=dict)
    active_session_log_paths: dict[str, Path] = dataclasses.field(default_factory=dict)
    deadlock_victim_issues: set[str] = dataclasses.field(default_factory=set)


class FakeIssueCoordinator:
    def __init__(self) -> None:
        self.abort_reason: str | None = None
        self.active_issue_ids: set[str] = set()
        self.released: list[str] = []
        self.completed: list[str] = []
        self.abort_requests: list[str] = []
        self.run_loop_kwargs: dict[str, object] | None = None

    @property
    def current_state(self) -> IssueLifecycleState:
        return IssueLifecycleState(
            active_issue_ids=frozenset(self.active_issue_ids),
            failed_issues=frozenset(),
            abort_requested=False,
            abort_reason=self.abort_reason,
            max_issues=None,
            interrupt_requested=False,
        )

    def release_task(self, issue_id: str) -> None:
        self.active_issue_ids.discard(issue_id)
        self.released.append(issue_id)

    def mark_completed(self, issue_id: str) -> None:
        self.active_issue_ids.discard(issue_id)
        self.completed.append(issue_id)

    def request_abort(self, reason: str) -> None:
        self.abort_requests.append(reason)

    async def run_loop(self, **kwargs: object) -> RunResult:
        self.run_loop_kwargs = kwargs
        return RunResult(issues_spawned=1, exit_code=0, exit_reason="completed")


class FakeRunCoordinator:
    def __init__(
        self, status: Literal["passed", "failed", "aborted"] = "passed"
    ) -> None:
        self.status = status
        self.calls = 0

    async def run_trigger_validation(self) -> TriggerValidationResult:
        self.calls += 1
        return TriggerValidationResult(status=self.status)


class FakeIssueProvider:
    def __init__(self, close_result: bool = False) -> None:
        self.close_result = close_result
        self.close_calls = 0

    async def close_eligible_epics_async(self) -> bool:
        self.close_calls += 1
        return self.close_result


class FakeEpicVerifier:
    def __init__(self, *, should_raise: bool = False) -> None:
        self.should_raise = should_raise
        self.seen_override_ids: set[str] | None = None
        self.seen_scoped_calls: list[tuple[str, set[str] | None]] = []
        self.seen_epic_calls: list[tuple[str, bool]] = []

    async def verify_and_close_epic(
        self,
        epic_id: str,
        human_override: bool = False,
    ) -> EpicVerificationResult:
        self.seen_epic_calls.append((epic_id, human_override))
        if self.should_raise:
            raise RuntimeError("verification unavailable")
        return EpicVerificationResult(
            verified_count=1,
            passed_count=1,
            failed_count=0,
            verdicts={},
            remediation_issues_created=[],
        )

    async def verify_and_close_eligible_within(
        self,
        scope_epic_id: str,
        human_override_epic_ids: set[str] | None = None,
    ) -> EpicVerificationResult:
        self.seen_scoped_calls.append((scope_epic_id, human_override_epic_ids))
        if self.should_raise:
            raise RuntimeError("verification unavailable")
        return EpicVerificationResult(
            verified_count=1,
            passed_count=1,
            failed_count=0,
            verdicts={},
            remediation_issues_created=[],
        )

    async def verify_and_close_eligible(
        self, human_override_epic_ids: set[str]
    ) -> EpicVerificationResult:
        self.seen_override_ids = human_override_epic_ids
        if self.should_raise:
            raise RuntimeError("verification unavailable")
        return EpicVerificationResult(
            verified_count=0,
            passed_count=0,
            failed_count=0,
            verdicts={},
            remediation_issues_created=["remediate-1"],
        )


async def _abort_active_tasks(
    run_metadata: object, *, is_interrupt: bool = False
) -> AbortResult:
    return AbortResult(aborted_count=1 if is_interrupt else 0)


async def _spawn_agent(issue_id: str) -> asyncio.Task[IssueResult] | None:
    async def run() -> IssueResult:
        return IssueResult(
            issue_id=issue_id,
            agent_id="agent-1",
            success=True,
            summary="done",
        )

    return asyncio.create_task(run())


def _build_plan(
    *,
    state: FakeState | None = None,
    issue_coordinator: FakeIssueCoordinator | None = None,
    run_coordinator: FakeRunCoordinator | None = None,
    issue_provider: FakeIssueProvider | None = None,
    epic_verifier: FakeEpicVerifier | None = None,
    finalized: list[IssueResult] | None = None,
    trigger_checked: list[str] | None = None,
    await_background_repo_work: Callable[[], Awaitable[None]] | None = None,
    startup_epic_id: str | None = None,
    allow_global_startup_epic_verification: bool = True,
) -> IssueExecutionPlan:
    finalized = finalized if finalized is not None else []
    trigger_checked = trigger_checked if trigger_checked is not None else []

    async def finalize_issue_result(
        issue_id: str,
        result: IssueResult,
        run_metadata: object,
    ) -> IssueFinalizeOutput:
        finalized.append(result)
        return IssueFinalizeOutput(
            success=True,
            closed=True,
            gate_metadata=MagicMock(),
        )

    return build_issue_execution_plan(
        run_metadata=MagicMock(),
        state=state or FakeState(),
        issue_coordinator=issue_coordinator or FakeIssueCoordinator(),
        run_coordinator=run_coordinator or FakeRunCoordinator(),
        beads=issue_provider or FakeIssueProvider(),
        event_sink=MagicMock(),
        epic_verifier=epic_verifier,
        epic_override_ids={"epic-1"},
        spawn_agent=_spawn_agent,
        abort_active_tasks=_abort_active_tasks,
        finalize_issue_result=finalize_issue_result,
        cleanup_issue_runtime_state=lambda issue_id: trigger_checked.append(
            f"cleanup:{issue_id}"
        ),
        check_and_queue_periodic_trigger=lambda result: trigger_checked.append(
            result.issue_id
        ),
        mark_validation_failed=lambda: trigger_checked.append("validation_failed"),
        startup_epic_id=startup_epic_id,
        allow_global_startup_epic_verification=allow_global_startup_epic_verification,
        await_background_repo_work=await_background_repo_work,
    )


@pytest.mark.unit
def test_issue_execution_plan_and_fakes_are_dataclasses() -> None:
    """Import-time guard for malformed dataclass decorators."""
    assert dataclasses.is_dataclass(IssueExecutionPlan)
    assert IssueExecutionPlan.__dataclass_params__.frozen is True
    assert dataclasses.is_dataclass(FakeState)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finalize_callback_maps_task_exception_to_failed_result() -> None:
    state = FakeState(
        agent_ids={"issue-1": "agent-1"},
        active_session_log_paths={"issue-1": Path("/tmp/session.log")},
    )
    issue_coordinator = FakeIssueCoordinator()
    run_coordinator = FakeRunCoordinator()
    finalized: list[IssueResult] = []
    trigger_checked: list[str] = []
    plan = _build_plan(
        state=state,
        issue_coordinator=issue_coordinator,
        run_coordinator=run_coordinator,
        finalized=finalized,
        trigger_checked=trigger_checked,
    )

    async def fail() -> IssueResult:
        raise RuntimeError("agent failed")

    task = asyncio.create_task(fail())
    await asyncio.sleep(0)

    await plan.finalize_callback("issue-1", task)

    assert issue_coordinator.completed == ["issue-1"]
    assert run_coordinator.calls == 1
    assert trigger_checked == ["issue-1"]
    assert finalized[0].issue_id == "issue-1"
    assert finalized[0].agent_id == "agent-1"
    assert finalized[0].success is False
    assert finalized[0].summary == "agent failed"
    assert finalized[0].session_log_path == Path("/tmp/session.log")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finalize_callback_releases_deadlock_victim_without_completion() -> None:
    state = FakeState(
        agent_ids={"issue-1": "agent-1"},
        deadlock_victim_issues={"issue-1"},
    )
    issue_coordinator = FakeIssueCoordinator()
    finalized: list[IssueResult] = []
    trigger_checked: list[str] = []
    plan = _build_plan(
        state=state,
        issue_coordinator=issue_coordinator,
        finalized=finalized,
        trigger_checked=trigger_checked,
    )

    async def wait_forever() -> IssueResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    task = asyncio.create_task(wait_forever())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await plan.finalize_callback("issue-1", task)

    assert issue_coordinator.released == ["issue-1"]
    assert issue_coordinator.completed == []
    assert finalized == []
    assert trigger_checked == ["cleanup:issue-1"]
    assert state.deadlock_victim_issues == set()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finalize_callback_aborts_when_trigger_validation_aborts() -> None:
    issue_coordinator = FakeIssueCoordinator()
    run_coordinator = FakeRunCoordinator(status="aborted")
    plan = _build_plan(
        issue_coordinator=issue_coordinator,
        run_coordinator=run_coordinator,
    )

    async def succeed() -> IssueResult:
        return IssueResult(
            issue_id="issue-1",
            agent_id="agent-1",
            success=True,
            summary="done",
        )

    task = asyncio.create_task(succeed())
    await asyncio.sleep(0)

    await plan.finalize_callback("issue-1", task)

    assert issue_coordinator.abort_requests == ["Trigger validation aborted"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finalize_callback_defers_trigger_validation_until_agents_drain() -> None:
    issue_coordinator = FakeIssueCoordinator()
    issue_coordinator.active_issue_ids = {"issue-1", "issue-2"}
    run_coordinator = FakeRunCoordinator()
    plan = _build_plan(
        issue_coordinator=issue_coordinator,
        run_coordinator=run_coordinator,
    )

    async def succeed(issue_id: str) -> IssueResult:
        return IssueResult(
            issue_id=issue_id,
            agent_id=f"agent-{issue_id}",
            success=True,
            summary="done",
        )

    task_1 = asyncio.create_task(succeed("issue-1"))
    await asyncio.sleep(0)

    await plan.finalize_callback("issue-1", task_1)

    assert run_coordinator.calls == 0

    task_2 = asyncio.create_task(succeed("issue-2"))
    await asyncio.sleep(0)

    await plan.finalize_callback("issue-2", task_2)

    assert run_coordinator.calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finalize_callback_waits_for_background_repo_work_before_validation() -> None:
    issue_coordinator = FakeIssueCoordinator()
    issue_coordinator.active_issue_ids = {"issue-1"}
    run_coordinator = FakeRunCoordinator()
    drain_started = asyncio.Event()
    drain_can_finish = asyncio.Event()

    async def await_background_repo_work() -> None:
        drain_started.set()
        await drain_can_finish.wait()

    plan = _build_plan(
        issue_coordinator=issue_coordinator,
        run_coordinator=run_coordinator,
        await_background_repo_work=await_background_repo_work,
    )

    async def succeed() -> IssueResult:
        return IssueResult(
            issue_id="issue-1",
            agent_id="agent-1",
            success=True,
            summary="done",
        )

    issue_task = asyncio.create_task(succeed())
    await asyncio.sleep(0)
    finalize_task = asyncio.create_task(
        plan.finalize_callback("issue-1", issue_task)
    )

    await asyncio.wait_for(drain_started.wait(), timeout=1)
    assert run_coordinator.calls == 0

    drain_can_finish.set()
    await finalize_task

    assert run_coordinator.calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_startup_epic_callback_uses_epic_verifier() -> None:
    verifier = FakeEpicVerifier()
    provider = FakeIssueProvider(close_result=False)
    plan = _build_plan(issue_provider=provider, epic_verifier=verifier)

    assert plan.on_startup_no_ready is not None
    assert await plan.on_startup_no_ready() is True
    assert verifier.seen_override_ids == {"epic-1"}
    assert provider.close_calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_startup_epic_callback_honors_epic_scope() -> None:
    verifier = FakeEpicVerifier()
    provider = FakeIssueProvider(close_result=False)
    plan = _build_plan(
        issue_provider=provider,
        epic_verifier=verifier,
        startup_epic_id="epic-1",
        allow_global_startup_epic_verification=False,
    )

    assert plan.on_startup_no_ready is not None
    assert await plan.on_startup_no_ready() is True
    assert verifier.seen_scoped_calls == [("epic-1", {"epic-1"})]
    assert verifier.seen_epic_calls == []
    assert verifier.seen_override_ids is None
    assert provider.close_calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_startup_epic_callback_skips_global_verification_for_restricted_scope() -> None:
    verifier = FakeEpicVerifier()
    provider = FakeIssueProvider(close_result=True)
    plan = build_issue_execution_plan(
        run_metadata=MagicMock(),
        state=FakeState(),
        issue_coordinator=FakeIssueCoordinator(),
        run_coordinator=FakeRunCoordinator(),
        beads=provider,
        event_sink=MagicMock(),
        epic_verifier=verifier,
        epic_override_ids={"other-epic"},
        spawn_agent=_spawn_agent,
        abort_active_tasks=_abort_active_tasks,
        finalize_issue_result=MagicMock(),
        cleanup_issue_runtime_state=MagicMock(),
        check_and_queue_periodic_trigger=MagicMock(),
        mark_validation_failed=MagicMock(),
        allow_global_startup_epic_verification=False,
    )

    assert plan.on_startup_no_ready is not None
    assert await plan.on_startup_no_ready() is False
    assert verifier.seen_epic_calls == []
    assert verifier.seen_override_ids is None
    assert provider.close_calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_startup_epic_callback_skips_legacy_global_close_for_restricted_scope() -> None:
    provider = FakeIssueProvider(close_result=True)
    plan = build_issue_execution_plan(
        run_metadata=MagicMock(),
        state=FakeState(),
        issue_coordinator=FakeIssueCoordinator(),
        run_coordinator=FakeRunCoordinator(),
        beads=provider,
        event_sink=MagicMock(),
        epic_verifier=None,
        epic_override_ids=set(),
        spawn_agent=_spawn_agent,
        abort_active_tasks=_abort_active_tasks,
        finalize_issue_result=MagicMock(),
        cleanup_issue_runtime_state=MagicMock(),
        check_and_queue_periodic_trigger=MagicMock(),
        mark_validation_failed=MagicMock(),
        allow_global_startup_epic_verification=False,
    )

    assert plan.on_startup_no_ready is not None
    assert await plan.on_startup_no_ready() is False
    assert provider.close_calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_startup_epic_callback_warns_and_continues_on_failure() -> None:
    event_sink = MagicMock()
    plan = build_issue_execution_plan(
        run_metadata=MagicMock(),
        state=FakeState(),
        issue_coordinator=FakeIssueCoordinator(),
        run_coordinator=FakeRunCoordinator(),
        beads=FakeIssueProvider(),
        event_sink=event_sink,
        epic_verifier=FakeEpicVerifier(should_raise=True),
        epic_override_ids={"epic-1"},
        spawn_agent=_spawn_agent,
        abort_active_tasks=_abort_active_tasks,
        finalize_issue_result=MagicMock(),
        cleanup_issue_runtime_state=MagicMock(),
        check_and_queue_periodic_trigger=MagicMock(),
        mark_validation_failed=MagicMock(),
    )

    assert plan.on_startup_no_ready is not None
    assert await plan.on_startup_no_ready() is False
    event_sink.on_warning.assert_called_once_with(
        "Startup epic verification failed; continuing without startup repoll"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plan_run_delegates_callbacks_to_issue_coordinator() -> None:
    issue_coordinator = FakeIssueCoordinator()
    plan = _build_plan(issue_coordinator=issue_coordinator)

    result = await plan.run(cast("IssueExecutionCoordinator", issue_coordinator))

    assert result.exit_code == 0
    assert issue_coordinator.run_loop_kwargs is not None
    assert issue_coordinator.run_loop_kwargs["spawn_callback"] is plan.spawn_callback
    assert (
        issue_coordinator.run_loop_kwargs["finalize_callback"] is plan.finalize_callback
    )
    assert issue_coordinator.run_loop_kwargs["abort_callback"] is plan.abort_callback
    assert issue_coordinator.run_loop_kwargs["on_startup_no_ready"] is not None
