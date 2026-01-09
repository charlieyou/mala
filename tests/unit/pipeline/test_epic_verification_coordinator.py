"""Unit tests for EpicVerificationCoordinator interrupt handling.

Tests verify that epic verification properly handles interrupt events:
- test_epic_verification_checks_interrupt_before_retry: Exits retry loop when interrupted
- test_epic_verification_cancels_remediation_on_interrupt: Cancels remediation tasks
- test_epic_verification_stops_spawning_on_interrupt: No new tasks spawned after interrupt
- test_epic_verification_works_without_interrupt_event: Normal operation without interrupt
- test_spawn_remediation_passes_flow_parameter: Verifies flow="epic_remediation" is passed
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.models import EpicVerificationResult
from src.pipeline.epic_verification_coordinator import (
    EpicVerificationCallbacks,
    EpicVerificationConfig,
    EpicVerificationCoordinator,
)
from src.pipeline.issue_result import IssueResult


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


class TestEpicVerificationInterruptHandling:
    """Tests for interrupt handling in EpicVerificationCoordinator."""

    @pytest.fixture
    def callbacks(self) -> EpicVerificationCallbacks:
        """Create mock callbacks."""
        return EpicVerificationCallbacks(
            get_parent_epic=AsyncMock(return_value="epic-1"),
            verify_epic=AsyncMock(return_value=make_verification_result()),
            spawn_remediation=AsyncMock(return_value=None),
            finalize_remediation=AsyncMock(),
            mark_completed=MagicMock(),
            is_issue_failed=MagicMock(return_value=False),
            close_eligible_epics=AsyncMock(return_value=False),
            on_epic_closed=MagicMock(),
            on_warning=MagicMock(),
            has_epic_verifier=MagicMock(return_value=True),
            get_agent_id=MagicMock(return_value="agent-1"),
            queue_trigger_validation=MagicMock(),
            get_epic_completion_trigger=MagicMock(return_value=None),
        )

    @pytest.fixture
    def run_metadata(self) -> MagicMock:
        """Create mock run metadata."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_epic_verification_checks_interrupt_before_retry(
        self,
        callbacks: EpicVerificationCallbacks,
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

        callbacks.spawn_remediation = AsyncMock(side_effect=spawn_remediation)

        # Create coordinator with retries allowed
        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=2),
            callbacks=callbacks,
        )

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
        callbacks: EpicVerificationCallbacks,
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

        callbacks.spawn_remediation = AsyncMock(side_effect=spawn_remediation)
        callbacks.verify_epic = AsyncMock(
            return_value=make_verification_result(
                failed_count=1,
                remediation_issues=["rem-1"],
            )
        )

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=1),
            callbacks=callbacks,
        )

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
        callbacks: EpicVerificationCallbacks,
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

        callbacks.spawn_remediation = AsyncMock(side_effect=spawn_remediation)
        callbacks.verify_epic = AsyncMock(
            return_value=make_verification_result(
                failed_count=1,
                remediation_issues=["rem-1", "rem-2", "rem-3"],
            )
        )

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=1),
            callbacks=callbacks,
        )

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
        callbacks: EpicVerificationCallbacks,
        run_metadata: MagicMock,
    ) -> None:
        """Verify that verification works normally when no interrupt event is provided."""
        callbacks.verify_epic = AsyncMock(
            return_value=make_verification_result(
                passed_count=1,
                failed_count=0,
            )
        )

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=1),
            callbacks=callbacks,
        )

        # Call without interrupt_event (default None)
        await coordinator.check_epic_closure("issue-1", run_metadata)

        # Verification should complete normally
        assert "epic-1" in coordinator.verified_epics
        callbacks.verify_epic.assert_called_once()

    @pytest.mark.asyncio
    async def test_spawn_remediation_passes_flow_parameter(
        self,
        callbacks: EpicVerificationCallbacks,
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

        callbacks.spawn_remediation = AsyncMock(side_effect=spawn_remediation)
        callbacks.verify_epic = AsyncMock(
            return_value=make_verification_result(
                failed_count=1,
                remediation_issues=["rem-1"],
            )
        )

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=1),
            callbacks=callbacks,
        )

        await coordinator.check_epic_closure("issue-1", run_metadata)

        # Verify flow parameter was passed as "epic_remediation"
        assert captured_flows == ["epic_remediation"]
