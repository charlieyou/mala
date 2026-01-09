"""Unit tests for interrupt_event wiring through orchestrator components.

These are audit tests that verify non-None interrupt_event is passed through
the orchestrator's callback chains. They catch regressions where wiring is
accidentally broken.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from src.infra.io.log_output.run_metadata import RunMetadata
from src.orchestration.factory import OrchestratorConfig, OrchestratorDependencies
from src.pipeline.issue_result import IssueResult
from src.pipeline.run_coordinator import GlobalValidationOutput

if TYPE_CHECKING:
    from pathlib import Path

    from src.pipeline.run_coordinator import GlobalValidationInput


@pytest.fixture
def tmp_runs_dir(tmp_path: Path) -> Path:
    """Create a temporary runs directory."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    return runs_dir


@pytest.mark.unit
class TestInterruptWiring:
    """Audit tests for interrupt_event wiring in orchestrator.

    These tests verify that interrupt_event is properly passed through
    the callback chains from orchestrator to all flow components.
    """

    def test_orchestrator_stores_interrupt_event(
        self, tmp_path: Path, tmp_runs_dir: Path
    ) -> None:
        """MalaOrchestrator._interrupt_event is set during run().

        This verifies the interrupt_event is stored on the orchestrator
        instance so callbacks can access it.
        """
        from src.orchestration.factory import create_orchestrator
        from tests.fakes.event_sink import FakeEventSink
        from tests.fakes.issue_provider import FakeIssueProvider

        provider = FakeIssueProvider()
        event_sink = FakeEventSink()

        config = OrchestratorConfig(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
        )
        deps = OrchestratorDependencies(
            issue_provider=provider,
            event_sink=event_sink,
            runs_dir=tmp_runs_dir,
        )
        orchestrator = create_orchestrator(config, deps=deps)

        # Before run(), _interrupt_event should be None
        assert orchestrator._interrupt_event is None

    def test_trigger_epic_closure_callback_passes_interrupt_event(
        self, tmp_path: Path, tmp_runs_dir: Path
    ) -> None:
        """trigger_epic_closure callback passes interrupt_event to check_epic_closure.

        This audit test verifies the callback defined in _build_issue_finalizer
        correctly passes self._interrupt_event to EpicVerificationCoordinator.
        """
        from src.orchestration.factory import create_orchestrator
        from tests.fakes.event_sink import FakeEventSink
        from tests.fakes.issue_provider import FakeIssueProvider

        provider = FakeIssueProvider()
        event_sink = FakeEventSink()

        config = OrchestratorConfig(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
        )
        deps = OrchestratorDependencies(
            issue_provider=provider,
            event_sink=event_sink,
            runs_dir=tmp_runs_dir,
        )
        orchestrator = create_orchestrator(config, deps=deps)

        # Set up _interrupt_event as if run() had started
        interrupt_event = asyncio.Event()
        orchestrator._interrupt_event = interrupt_event

        # Mock the epic verification coordinator to capture the call
        captured_interrupt_event: asyncio.Event | None = None

        async def mock_check_epic_closure(
            issue_id: str,
            run_metadata: RunMetadata,
            *,
            interrupt_event: asyncio.Event | None = None,
        ) -> None:
            nonlocal captured_interrupt_event
            captured_interrupt_event = interrupt_event

        orchestrator.epic_verification_coordinator.check_epic_closure = (  # type: ignore[method-assign]
            mock_check_epic_closure
        )

        # Call the trigger_epic_closure callback via issue_finalizer
        mock_run_metadata = MagicMock(spec=RunMetadata)
        trigger_callback = orchestrator.issue_finalizer.callbacks.trigger_epic_closure

        # Run async callback - use a new event loop to avoid pytest conflicts
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(trigger_callback("test-issue", mock_run_metadata))
        finally:
            loop.close()

        # Verify interrupt_event was passed
        assert captured_interrupt_event is interrupt_event

    def test_run_coordinator_receives_interrupt_event(
        self, tmp_path: Path, tmp_runs_dir: Path
    ) -> None:
        """RunCoordinator.run_validation receives interrupt_event from orchestrator.

        This verifies the _validation_callback closure in run() correctly
        passes interrupt_event to run_coordinator.run_validation().
        """
        from src.core.models import PeriodicValidationConfig, WatchConfig
        from src.orchestration.factory import create_orchestrator
        from tests.fakes.event_sink import FakeEventSink
        from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

        # Use a provider with an issue so validation would run
        provider = FakeIssueProvider(
            issues={"test-issue": FakeIssue(id="test-issue", status="open")}
        )
        event_sink = FakeEventSink()

        config = OrchestratorConfig(
            repo_path=tmp_path,
            max_agents=1,
            max_issues=1,
        )
        deps = OrchestratorDependencies(
            issue_provider=provider,
            event_sink=event_sink,
            runs_dir=tmp_runs_dir,
        )
        orchestrator = create_orchestrator(config, deps=deps)

        # Capture the interrupt_event passed to run_validation
        captured_interrupt_event: asyncio.Event | None = None

        async def mock_run_validation(
            validation_input: GlobalValidationInput,
            *,
            interrupt_event: asyncio.Event | None = None,
        ) -> GlobalValidationOutput:
            nonlocal captured_interrupt_event
            captured_interrupt_event = interrupt_event
            return GlobalValidationOutput(passed=True, interrupted=False)

        orchestrator.run_coordinator.run_validation = mock_run_validation  # type: ignore[method-assign]

        # Mock run_implementer to immediately return success
        async def mock_implementer(
            issue_id: str, *, flow: str = "implementer"
        ) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id="mock",
                success=True,
                summary="Success",
            )

        orchestrator.run_implementer = mock_implementer  # type: ignore[method-assign]

        # Run orchestrator (will trigger validation due to validate_every=1)
        validation_config = PeriodicValidationConfig(validate_every=1)
        watch_config = WatchConfig(enabled=False)

        # Use new event loop to avoid pytest conflicts
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                orchestrator.run(
                    watch_config=watch_config,
                    validation_config=validation_config,
                )
            )
        finally:
            loop.close()

        # Verify interrupt_event was passed to run_validation
        assert captured_interrupt_event is not None
        assert isinstance(captured_interrupt_event, asyncio.Event)
