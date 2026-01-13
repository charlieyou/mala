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

if TYPE_CHECKING:
    from pathlib import Path


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

    def test_interrupt_event_exists_before_run(
        self, tmp_path: Path, tmp_runs_dir: Path
    ) -> None:
        """MalaOrchestrator._interrupt_event exists before run() is called.

        LifecycleController creates events on init. They're reset fresh per-run
        in _reset_sigint_state(), ensuring callbacks always get fresh events.
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

        # LifecycleController creates events on init, not set initially
        assert orchestrator._interrupt_event is not None
        assert not orchestrator._interrupt_event.is_set()

    async def test_trigger_epic_closure_callback_passes_interrupt_event(
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

        await trigger_callback("test-issue", mock_run_metadata)

        # Verify interrupt_event was passed
        assert captured_interrupt_event is interrupt_event

    async def test_issue_coordinator_receives_interrupt_event(
        self, tmp_path: Path, tmp_runs_dir: Path
    ) -> None:
        """IssueExecutionCoordinator.run_loop receives interrupt_event from orchestrator.

        This audit test verifies interrupt_event is passed from orchestrator.run()
        to issue_coordinator.run_loop().
        """
        from src.core.models import WatchConfig
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

        # Capture the interrupt_event passed to run_loop
        captured_interrupt_event: asyncio.Event | None = None

        async def mock_run_loop(  # noqa: ANN202
            *,
            spawn_callback,  # noqa: ANN001
            finalize_callback,  # noqa: ANN001
            abort_callback,  # noqa: ANN001
            watch_config,  # noqa: ANN001
            validation_config=None,  # noqa: ANN001
            drain_event=None,  # noqa: ANN001
            interrupt_event=None,  # noqa: ANN001
            validation_callback=None,  # noqa: ANN001
            on_validation_failed=None,  # noqa: ANN001
            sleep_fn=asyncio.sleep,  # noqa: ANN001
        ):
            nonlocal captured_interrupt_event
            captured_interrupt_event = interrupt_event
            # Return empty result immediately
            from src.core.models import RunResult

            return RunResult(issues_spawned=0, exit_code=0, exit_reason="completed")

        orchestrator.issue_coordinator.run_loop = mock_run_loop  # type: ignore[method-assign]

        watch_config = WatchConfig(enabled=False)

        await orchestrator.run(watch_config=watch_config)

        # Verify interrupt_event was passed to run_loop
        assert captured_interrupt_event is not None
        assert isinstance(captured_interrupt_event, asyncio.Event)

    async def test_run_implementer_passes_interrupt_event_to_session_runner(
        self, tmp_path: Path, tmp_runs_dir: Path
    ) -> None:
        """run_implementer passes interrupt_event to AgentSessionRunner.run_session.

        This audit test verifies that MalaOrchestrator.run_implementer correctly
        wires self._interrupt_event into the session runner call.
        """
        from unittest.mock import AsyncMock, patch

        from src.orchestration.factory import create_orchestrator
        from src.pipeline.agent_session_runner import AgentSessionOutput
        from tests.fakes.event_sink import FakeEventSink
        from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

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

        # Set up _interrupt_event as if run() had started
        interrupt_event = asyncio.Event()
        orchestrator._interrupt_event = interrupt_event

        # Capture the interrupt_event passed to run_session
        captured_interrupt_event: asyncio.Event | None = None

        async def mock_run_session(
            input,  # noqa: ANN001
            tracer=None,  # noqa: ANN001
            interrupt_event: asyncio.Event | None = None,
        ) -> AgentSessionOutput:
            nonlocal captured_interrupt_event
            captured_interrupt_event = interrupt_event
            return AgentSessionOutput(
                success=True,
                summary="Success",
                agent_id="mock-agent",
                session_id="mock-session",
            )

        # Patch AgentSessionRunner to use our mock
        with patch("src.orchestration.orchestrator.AgentSessionRunner") as MockRunner:
            mock_runner_instance = AsyncMock()
            mock_runner_instance.run_session = mock_run_session
            MockRunner.return_value = mock_runner_instance

            await orchestrator.run_implementer("test-issue")

        # Verify interrupt_event was passed to run_session
        assert captured_interrupt_event is interrupt_event

    async def test_gate_callback_passes_interrupt_event(
        self, tmp_path: Path, tmp_runs_dir: Path
    ) -> None:
        """Gate callback from SessionCallbackFactory passes interrupt_event.

        This audit test verifies that the on_gate_check callback correctly
        passes the interrupt_event to AsyncGateRunner.run_gate_async.
        """
        from src.orchestration.factory import create_orchestrator
        from tests.fakes.event_sink import FakeEventSink
        from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

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

        # Set up _interrupt_event as if run() had started
        interrupt_event = asyncio.Event()
        orchestrator._interrupt_event = interrupt_event

        # Capture the interrupt_event passed to run_gate_async
        captured_interrupt_event: asyncio.Event | None = None

        from src.domain.evidence_check import GateResult
        from src.domain.lifecycle import RetryState

        async def mock_run_gate_async(
            issue_id: str,
            log_path: object,
            retry_state: RetryState,
            interrupt_event: asyncio.Event | None = None,
        ) -> tuple[GateResult, int]:
            nonlocal captured_interrupt_event
            captured_interrupt_event = interrupt_event
            return GateResult(passed=True, failure_reasons=[], commit_hash=None), 0

        orchestrator.async_gate_runner.run_gate_async = mock_run_gate_async  # type: ignore[method-assign]

        # Build adapters and invoke the gate check via adapter
        adapters = orchestrator.session_callback_factory.build_adapters("test-issue")

        # Call the gate check via adapter
        await adapters.gate_runner.run_gate_check(
            "test-issue", tmp_path / "test.log", RetryState()
        )

        # Verify interrupt_event was passed
        assert captured_interrupt_event is interrupt_event

    async def test_review_callback_passes_interrupt_event(
        self, tmp_path: Path, tmp_runs_dir: Path
    ) -> None:
        """Review callback from SessionCallbackFactory passes interrupt_event.

        This audit test verifies that the on_review_check callback correctly
        passes the interrupt_event to ReviewRunner.run_review.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.orchestration.factory import create_orchestrator
        from tests.fakes.event_sink import FakeEventSink
        from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

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

        # Set up _interrupt_event as if run() had started
        interrupt_event = asyncio.Event()
        orchestrator._interrupt_event = interrupt_event

        # Capture the interrupt_event passed to run_review
        captured_interrupt_event: asyncio.Event | None = None

        from src.domain.lifecycle import RetryState
        from src.pipeline.review_runner import ReviewOutput

        async def mock_run_review(
            input: object,
            interrupt_event: asyncio.Event | None = None,
        ) -> ReviewOutput:
            nonlocal captured_interrupt_event
            captured_interrupt_event = interrupt_event
            assert getattr(input, "author_context", None) == "author context"
            # Return a mock result
            mock_result = MagicMock()
            mock_result.passed = True
            mock_result.issues = []
            mock_result.parse_error = None
            mock_result.fatal_error = False
            mock_result.review_log_path = None
            return ReviewOutput(result=mock_result)

        orchestrator.review_runner.run_review = mock_run_review  # type: ignore[method-assign]

        # Build adapters and invoke the review check via adapter
        adapters = orchestrator.session_callback_factory.build_adapters("test-issue")

        with patch(
            "src.infra.git_utils.get_issue_commits_async",
            new=AsyncMock(return_value=["abc123"]),
        ):
            await adapters.review_runner.run_review(
                "test-issue",
                "test description",
                "session-123",
                RetryState(),
                "author context",
                None,  # previous_findings
                None,  # session_end_result
            )

        # Verify interrupt_event was passed
        assert captured_interrupt_event is interrupt_event
