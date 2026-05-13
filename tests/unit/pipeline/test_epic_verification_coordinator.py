"""Unit tests for EpicVerificationCoordinator interrupt handling.

Tests verify that epic verification properly handles interrupt events:
- test_epic_verification_checks_interrupt_before_retry: Exits retry loop when interrupted
- test_epic_verification_cancels_remediation_on_interrupt: Cancels remediation tasks
- test_epic_verification_stops_spawning_on_interrupt: No new tasks spawned after interrupt
- test_epic_verification_works_without_interrupt_event: Normal operation without interrupt
- test_spawn_remediation_passes_flow_parameter: Verifies flow="epic_remediation" is passed
"""

import asyncio
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.models import EpicVerificationResult
from src.pipeline.epic_verification_coordinator import (
    EpicVerificationConfig,
    EpicVerificationCoordinator,
)
from src.pipeline.issue_result import IssueResult
from tests.fakes.issue_lifecycle_port import FakeIssueLifecyclePort

if TYPE_CHECKING:
    from src.domain.validation.config_types import (
        EpicCompletionTriggerConfig,
        TriggerType,
    )
    from src.core.protocols.validation import EpicVerifierProtocol


def make_verification_result(
    *,
    verified_count: int = 1,
    passed_count: int = 0,
    failed_count: int = 1,
    remediation_issues: list[str] | None = None,
) -> EpicVerificationResult:
    """Create a verification result for testing."""
    return EpicVerificationResult(
        verified_count=verified_count,
        passed_count=passed_count,
        failed_count=failed_count,
        verdicts={},
        remediation_issues_created=remediation_issues or [],
    )


def make_issue_result(issue_id: str, *, success: bool = True) -> IssueResult:
    """Create an issue result for testing."""
    return IssueResult(
        issue_id=issue_id,
        agent_id=f"agent-{issue_id}",
        success=success,
        summary="Test result",
    )


class EpicVerificationHarness:
    """Mutable test harness for EpicVerificationCoordinator ports."""

    def __init__(self) -> None:
        self.issue_provider = MagicMock()
        self.issue_provider.get_parent_epic_async = AsyncMock(return_value="epic-1")
        self.issue_provider.close_eligible_epics_async = AsyncMock(return_value=False)
        self.event_sink = MagicMock()
        self.issue_lifecycle = FakeIssueLifecyclePort()
        self.verify_epic = AsyncMock(return_value=make_verification_result())
        self.spawn_epic_remediation_mock = AsyncMock(return_value=None)
        self.finalize_epic_remediation_mock = AsyncMock()
        self.get_epic_remediation_agent_id_mock = MagicMock(return_value="agent-1")
        self.has_epic_verifier = MagicMock(return_value=True)
        self.queue_trigger_validation_mock = MagicMock()
        self.get_epic_completion_trigger_mock = MagicMock(return_value=None)

    async def verify_and_close_epic(
        self,
        epic_id: str,
        *,
        human_override: bool,
    ) -> EpicVerificationResult:
        return await self.verify_epic(epic_id, human_override)

    def to_coordinator(self, max_retries: int) -> EpicVerificationCoordinator:
        """Build a coordinator from the current harness attributes."""
        return EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=max_retries),
            issue_provider=self.issue_provider,
            event_sink=self.event_sink,
            issue_lifecycle=self.issue_lifecycle,
            epic_verifier_provider=self,
            remediation_port=self,
            trigger_queue=self,
            epic_completion_trigger_provider=self,
        )

    def get_epic_verifier(self) -> "EpicVerifierProtocol | None":
        return cast("EpicVerifierProtocol", self) if self.has_epic_verifier() else None

    async def spawn_epic_remediation(
        self, issue_id: str, flow: str = "implementer"
    ) -> asyncio.Task[IssueResult] | None:
        return await self.spawn_epic_remediation_mock(issue_id, flow)

    async def finalize_epic_remediation(
        self, issue_id: str, result: IssueResult, run_metadata: object
    ) -> None:
        await self.finalize_epic_remediation_mock(issue_id, result, run_metadata)

    def get_epic_remediation_agent_id(self, issue_id: str) -> str:
        return self.get_epic_remediation_agent_id_mock(issue_id)

    def queue_trigger_validation(
        self, trigger_type: "TriggerType", context: dict[str, object]
    ) -> None:
        self.queue_trigger_validation_mock(trigger_type, context)

    def get_epic_completion_trigger(self) -> "EpicCompletionTriggerConfig | None":
        return self.get_epic_completion_trigger_mock()


class TestEpicVerificationInterruptHandling:
    """Tests for interrupt handling in EpicVerificationCoordinator."""

    @pytest.fixture
    def callbacks(self) -> EpicVerificationHarness:
        """Create mutable coordinator harness."""
        return EpicVerificationHarness()

    @pytest.fixture
    def run_metadata(self) -> MagicMock:
        """Create mock run metadata."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_epic_verification_checks_interrupt_before_retry(
        self,
        callbacks: EpicVerificationHarness,
        run_metadata: MagicMock,
    ) -> None:
        """Verify that interrupt is checked before each retry iteration."""
        # Set up: verification fails with remediation issues, but we interrupt before retry
        verification_call_count = 0

        async def verify_epic(
            epic_id: str, human_override: bool
        ) -> EpicVerificationResult:
            nonlocal verification_call_count
            verification_call_count += 1
            # First call fails with remediation issues
            return make_verification_result(
                failed_count=1,
                remediation_issues=["rem-1"],
            )

        callbacks.verify_epic = AsyncMock(side_effect=verify_epic)

        # Spawn remediation returns a task that completes immediately
        async def spawn_remediation(
            issue_id: str, flow: str = "implementer"
        ) -> asyncio.Task[IssueResult]:
            async def work() -> IssueResult:
                return make_issue_result(issue_id)

            return asyncio.create_task(work())

        callbacks.spawn_epic_remediation_mock = AsyncMock(side_effect=spawn_remediation)

        # Create coordinator with retries allowed
        coordinator = callbacks.to_coordinator(max_retries=2)

        # Create interrupt event that will be set after first verification
        interrupt_event = asyncio.Event()

        # Set interrupt after first verification completes
        original_verify = callbacks.verify_epic

        async def verify_and_interrupt(
            epic_id: str, human_override: bool
        ) -> EpicVerificationResult:
            result = await original_verify(epic_id, human_override)
            # Set interrupt after first verification, before retry
            if verification_call_count == 1:
                interrupt_event.set()
            return result

        callbacks.verify_epic = AsyncMock(side_effect=verify_and_interrupt)

        # Run verification
        await coordinator.check_epic_closure(
            "issue-1", run_metadata, interrupt_event=interrupt_event
        )

        # Should have only called verify_epic once (no retries due to interrupt)
        assert verification_call_count == 1
        # Epic should NOT be marked as verified (interrupted before completing)
        assert "epic-1" not in coordinator.verified_epics

    @pytest.mark.asyncio
    async def test_epic_verification_cancels_remediation_on_interrupt(
        self,
        callbacks: EpicVerificationHarness,
        run_metadata: MagicMock,
    ) -> None:
        """Verify that remediation tasks are cancelled when interrupted."""
        # Track which tasks were cancelled
        cancelled_tasks: list[str] = []
        task_started_event = asyncio.Event()

        async def spawn_remediation(
            issue_id: str, flow: str = "implementer"
        ) -> asyncio.Task[IssueResult]:
            async def work() -> IssueResult:
                try:
                    task_started_event.set()
                    # Simulate long-running work
                    await asyncio.sleep(10.0)
                    return make_issue_result(issue_id)
                except asyncio.CancelledError:
                    cancelled_tasks.append(issue_id)
                    raise

            return asyncio.create_task(work())

        callbacks.spawn_epic_remediation_mock = AsyncMock(side_effect=spawn_remediation)
        callbacks.verify_epic = AsyncMock(
            return_value=make_verification_result(
                failed_count=1,
                remediation_issues=["rem-1"],
            )
        )

        coordinator = callbacks.to_coordinator(max_retries=1)

        interrupt_event = asyncio.Event()

        async def run_verification() -> None:
            await coordinator.check_epic_closure(
                "issue-1", run_metadata, interrupt_event=interrupt_event
            )

        # Start verification in background
        verification_task = asyncio.create_task(run_verification())

        # Wait for remediation task to start
        await task_started_event.wait()

        # Set interrupt while remediation is running
        interrupt_event.set()

        # Wait for verification to complete
        await verification_task

        # Verify the remediation task was cancelled
        assert "rem-1" in cancelled_tasks

    @pytest.mark.asyncio
    async def test_epic_verification_stops_spawning_on_interrupt(
        self,
        callbacks: EpicVerificationHarness,
        run_metadata: MagicMock,
    ) -> None:
        """Verify that no new remediation tasks are spawned after interrupt."""
        spawned_issues: list[str] = []

        async def spawn_remediation(
            issue_id: str, flow: str = "implementer"
        ) -> asyncio.Task[IssueResult]:
            spawned_issues.append(issue_id)

            async def work() -> IssueResult:
                return make_issue_result(issue_id)

            return asyncio.create_task(work())

        callbacks.spawn_epic_remediation_mock = AsyncMock(side_effect=spawn_remediation)
        callbacks.verify_epic = AsyncMock(
            return_value=make_verification_result(
                failed_count=1,
                remediation_issues=["rem-1", "rem-2", "rem-3"],
            )
        )

        coordinator = callbacks.to_coordinator(max_retries=1)

        # Pre-set interrupt before starting
        interrupt_event = asyncio.Event()
        interrupt_event.set()

        await coordinator.check_epic_closure(
            "issue-1", run_metadata, interrupt_event=interrupt_event
        )

        # Should have called verify_epic once, but no remediation tasks spawned
        # because interrupt was already set before spawning loop
        assert len(spawned_issues) == 0

    @pytest.mark.asyncio
    async def test_epic_verification_works_without_interrupt_event(
        self,
        callbacks: EpicVerificationHarness,
        run_metadata: MagicMock,
    ) -> None:
        """Verify that verification works normally when no interrupt event is provided."""
        callbacks.verify_epic = AsyncMock(
            return_value=make_verification_result(
                passed_count=1,
                failed_count=0,
            )
        )

        coordinator = callbacks.to_coordinator(max_retries=1)

        # Call without interrupt_event (default None)
        await coordinator.check_epic_closure("issue-1", run_metadata)

        # Verification should complete normally
        assert "epic-1" in coordinator.verified_epics
        callbacks.verify_epic.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_same_epic_closure_rechecks_after_current_pass(
        self,
        callbacks: EpicVerificationHarness,
        run_metadata: MagicMock,
    ) -> None:
        """A sibling closure during verification schedules a follow-up recheck."""
        first_verification_started = asyncio.Event()
        release_first_verification = asyncio.Event()
        verification_call_count = 0

        async def verify_epic(
            epic_id: str, human_override: bool
        ) -> EpicVerificationResult:
            nonlocal verification_call_count
            del epic_id, human_override
            verification_call_count += 1
            if verification_call_count == 1:
                first_verification_started.set()
                await release_first_verification.wait()
                return make_verification_result(
                    verified_count=0,
                    passed_count=0,
                    failed_count=0,
                )
            return make_verification_result(
                verified_count=1,
                passed_count=1,
                failed_count=0,
            )

        callbacks.verify_epic = AsyncMock(side_effect=verify_epic)
        coordinator = callbacks.to_coordinator(max_retries=0)

        first_check = asyncio.create_task(
            coordinator.check_epic_closure("issue-1", run_metadata)
        )
        await asyncio.wait_for(first_verification_started.wait(), timeout=0.2)

        second_check = asyncio.create_task(
            coordinator.check_epic_closure("issue-2", run_metadata)
        )
        await asyncio.wait_for(second_check, timeout=0.2)

        release_first_verification.set()
        await asyncio.wait_for(first_check, timeout=0.2)

        assert verification_call_count == 2
        assert "epic-1" in coordinator.verified_epics
        assert "epic-1" not in coordinator.epics_being_verified
        assert "epic-1" not in coordinator.epics_pending_recheck

    @pytest.mark.asyncio
    async def test_interrupt_unresponsive_remediation_returns_bounded_failure(
        self,
        callbacks: EpicVerificationHarness,
        run_metadata: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Interrupted remediation that ignores cancellation does not hang forever."""
        import src.pipeline.epic_verification_coordinator as epic_verification_module

        monkeypatch.setattr(
            epic_verification_module, "REMEDIATION_CANCEL_GRACE_SECONDS", 0.01
        )
        finish_task = asyncio.Event()
        spawned_task: asyncio.Task[IssueResult] | None = None

        async def spawn_remediation(
            issue_id: str, flow: str = "implementer"
        ) -> asyncio.Task[IssueResult]:
            del flow

            async def work() -> IssueResult:
                try:
                    await asyncio.sleep(10.0)
                except asyncio.CancelledError:
                    await finish_task.wait()
                return make_issue_result(issue_id)

            nonlocal spawned_task
            spawned_task = asyncio.create_task(work())
            return spawned_task

        callbacks.spawn_epic_remediation_mock = AsyncMock(side_effect=spawn_remediation)
        callbacks.verify_epic = AsyncMock(
            return_value=make_verification_result(
                failed_count=1,
                remediation_issues=["rem-1"],
            )
        )
        coordinator = callbacks.to_coordinator(max_retries=1)
        interrupt_event = asyncio.Event()

        async def interrupt_after_spawn() -> None:
            while spawned_task is None:
                await asyncio.sleep(0)
            interrupt_event.set()

        interrupter = asyncio.create_task(interrupt_after_spawn())
        await asyncio.wait_for(
            coordinator.check_epic_closure(
                "issue-1", run_metadata, interrupt_event=interrupt_event
            ),
            timeout=0.2,
        )
        await interrupter

        callbacks.finalize_epic_remediation_mock.assert_awaited_once()
        finalized_result = callbacks.finalize_epic_remediation_mock.await_args.args[1]
        assert finalized_result.success is False
        assert finalized_result.summary == "Remediation task did not stop after interrupt"

        finish_task.set()
        if spawned_task is not None:
            await asyncio.wait_for(spawned_task, timeout=0.2)

    @pytest.mark.asyncio
    async def test_spawn_remediation_passes_flow_parameter(
        self,
        callbacks: EpicVerificationHarness,
        run_metadata: MagicMock,
    ) -> None:
        """Verify that spawn_remediation is called with flow='epic_remediation'."""
        captured_flows: list[str] = []

        async def spawn_remediation(
            issue_id: str, flow: str = "implementer"
        ) -> asyncio.Task[IssueResult]:
            captured_flows.append(flow)

            async def work() -> IssueResult:
                return make_issue_result(issue_id)

            return asyncio.create_task(work())

        callbacks.spawn_epic_remediation_mock = AsyncMock(side_effect=spawn_remediation)
        callbacks.verify_epic = AsyncMock(
            return_value=make_verification_result(
                failed_count=1,
                remediation_issues=["rem-1"],
            )
        )

        coordinator = callbacks.to_coordinator(max_retries=1)

        await coordinator.check_epic_closure("issue-1", run_metadata)

        # Verify flow parameter was passed as "epic_remediation"
        assert captured_flows == ["epic_remediation"]
