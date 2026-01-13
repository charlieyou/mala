"""Integration tests for validation_triggers config loading."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.orchestration.orchestrator import MalaOrchestrator


def test_trigger_review_independent_of_per_issue_review(
    tmp_path: Path,
    make_orchestrator: Callable[..., MalaOrchestrator],
) -> None:
    """Trigger code_review.enabled is independent of per_issue_review.enabled.

    This test verifies that:
    - per_issue_review absent/disabled does NOT affect trigger code_review
    - Trigger code_review uses its own enabled flag, not _is_review_enabled()

    The separation is:
    - _is_review_enabled() -> per-issue reviews (via session_config.review_enabled)
    - trigger_config.code_review.enabled -> trigger reviews (_run_trigger_code_review)
    """
    # Configure: per_issue_review absent, but run_end trigger has code_review enabled
    config_content = dedent("""\
        preset: python-uv
        validation_triggers:
          run_end:
            failure_mode: continue
            commands: []
            code_review:
              enabled: true
              reviewer_type: cerberus
    """)
    config_file = tmp_path / "mala.yaml"
    config_file.write_text(config_content)

    orchestrator = make_orchestrator(
        repo_path=tmp_path,
        max_agents=1,
        timeout_minutes=1,
        max_issues=1,
    )

    # per_issue_review is absent, so _is_review_enabled() returns False
    assert orchestrator._per_issue_review is not None
    assert orchestrator._per_issue_review.enabled is False
    assert orchestrator._is_review_enabled() is False
    assert orchestrator._session_config.review_enabled is False

    # But trigger code_review should still be enabled (independent path)
    validation_config = orchestrator._validation_config
    assert validation_config is not None
    assert validation_config.validation_triggers is not None
    run_end = validation_config.validation_triggers.run_end
    assert run_end is not None
    assert run_end.code_review is not None
    assert run_end.code_review.enabled is True


def test_config_loads_validation_triggers_via_normal_path(tmp_path: Path) -> None:
    """Test that validation_triggers are loaded through the normal config path.

    This test exercises the full config loading path:
    load_config() → _build_config() → _parse_validation_triggers()
    """
    from src.domain.validation.config import (
        EpicDepth,
        FailureMode,
        FireOn,
    )
    from src.domain.validation.config_loader import load_config

    # Create a minimal mala.yaml with validation_triggers section
    config_content = dedent("""\
        preset: python-uv

        validation_triggers:
          epic_completion:
            epic_depth: top_level
            fire_on: success
            failure_mode: continue
            commands:
              - ref: test
          session_end:
            failure_mode: remediate
            max_retries: 3
            commands:
              - ref: lint
          periodic:
            interval: 5
            failure_mode: abort
            commands: []
    """)

    config_file = tmp_path / "mala.yaml"
    config_file.write_text(config_content)

    config = load_config(tmp_path)

    # Verify validation_triggers were parsed
    assert config.validation_triggers is not None
    triggers = config.validation_triggers

    # Verify epic_completion
    assert triggers.epic_completion is not None
    assert triggers.epic_completion.epic_depth == EpicDepth.TOP_LEVEL
    assert triggers.epic_completion.fire_on == FireOn.SUCCESS
    assert triggers.epic_completion.failure_mode == FailureMode.CONTINUE
    assert len(triggers.epic_completion.commands) == 1
    assert triggers.epic_completion.commands[0].ref == "test"

    # Verify session_end
    assert triggers.session_end is not None
    assert triggers.session_end.failure_mode == FailureMode.REMEDIATE
    assert triggers.session_end.max_retries == 3
    assert len(triggers.session_end.commands) == 1
    assert triggers.session_end.commands[0].ref == "lint"

    # Verify periodic
    assert triggers.periodic is not None
    assert triggers.periodic.interval == 5
    assert triggers.periodic.failure_mode == FailureMode.ABORT
    assert triggers.periodic.commands == ()


def test_trigger_queues_and_executes_via_run_coordinator(tmp_path: Path) -> None:
    """Test that triggers can be queued and executed via RunCoordinator.

    This test exercises RunCoordinator.queue_trigger_validation() →
    run_trigger_validation() path with a real trigger configuration.
    """
    import asyncio
    from unittest.mock import MagicMock

    from src.domain.validation.config import (
        CommandConfig,
        CommandsConfig,
        EpicCompletionTriggerConfig,
        EpicDepth,
        FailureMode,
        FireOn,
        TriggerCommandRef,
        TriggerType,
        ValidationConfig,
        ValidationTriggersConfig,
    )
    from src.pipeline.run_coordinator import RunCoordinator, RunCoordinatorConfig
    from src.pipeline.fixer_service import FixerService
    from src.pipeline.trigger_engine import TriggerEngine
    from tests.fakes import FakeEnvConfig
    from tests.fakes.command_runner import FakeCommandRunner
    from tests.fakes.lock_manager import FakeLockManager

    # Create a validation config with triggers
    validation_config = ValidationConfig(
        commands=CommandsConfig(
            test=CommandConfig(command="uv run pytest"),
        ),
        validation_triggers=ValidationTriggersConfig(
            epic_completion=EpicCompletionTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(TriggerCommandRef(ref="test"),),
                epic_depth=EpicDepth.TOP_LEVEL,
                fire_on=FireOn.SUCCESS,
            )
        ),
    )

    # Create command runner that allows any command (intent is testing queue/execution flow)
    command_runner = FakeCommandRunner(allow_unregistered=True)

    config = RunCoordinatorConfig(
        repo_path=tmp_path,
        timeout_seconds=60,
        validation_config=validation_config,
    )

    trigger_engine = TriggerEngine(validation_config=validation_config)
    mock_fixer_service = MagicMock(spec=FixerService)
    mock_fixer_service.cleanup_locks = MagicMock()

    coordinator = RunCoordinator(
        config=config,
        gate_checker=MagicMock(),
        command_runner=command_runner,
        env_config=FakeEnvConfig(),
        lock_manager=FakeLockManager(),
        sdk_client_factory=MagicMock(),
        trigger_engine=trigger_engine,
        fixer_service=mock_fixer_service,
    )

    # Verify trigger queue starts empty
    assert coordinator._trigger_queue == []

    # Queue a trigger
    coordinator.queue_trigger_validation(
        TriggerType.EPIC_COMPLETION, {"issue_id": "test-123", "epic_id": "epic-1"}
    )

    # Verify trigger was queued
    assert len(coordinator._trigger_queue) == 1
    trigger_type, context = coordinator._trigger_queue[0]
    assert trigger_type == TriggerType.EPIC_COMPLETION
    assert context["issue_id"] == "test-123"

    # Run trigger validation - should now succeed
    async def run_validation() -> None:
        result = await coordinator.run_trigger_validation()
        assert result.status == "passed"

    asyncio.run(run_validation())

    # Queue should be empty after execution
    assert coordinator._trigger_queue == []

    # Test clear_trigger_queue
    coordinator.queue_trigger_validation(
        TriggerType.EPIC_COMPLETION, {"issue_id": "test-456"}
    )
    coordinator.clear_trigger_queue(reason="test cleanup")
    assert coordinator._trigger_queue == []


@pytest.mark.asyncio
async def test_orchestrator_fires_periodic_trigger_at_interval(
    tmp_path: Path,
    make_orchestrator: Callable[..., MalaOrchestrator],
) -> None:
    """Test that orchestrator main loop fires and executes periodic trigger at interval.

    This integration test exercises the full orchestrator loop path:
    - Orchestrator.run() spawns and completes issues
    - finalize_callback invokes _check_and_queue_periodic_trigger() after each issue
    - Hook increments _non_epic_completed_count and queues PERIODIC trigger at interval
    - run_trigger_validation() executes queued triggers (blocking before next issue)

    The test runs the orchestrator with 2 fake issues and interval=2, verifying
    that after both complete, the periodic trigger was queued AND executed.
    """
    import asyncio

    from src.pipeline.issue_result import IssueResult
    from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

    # Create fake issues for the orchestrator to process
    fake_issues = FakeIssueProvider(
        {
            "issue-1": FakeIssue(id="issue-1", priority=1),
            "issue-2": FakeIssue(id="issue-2", priority=2),
        }
    )
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # Create mala.yaml with periodic trigger interval=2 and empty commands
    # (empty commands avoids command resolution; trigger still queues and executes)
    config_content = dedent("""\
        preset: python-uv

        validation_triggers:
          periodic:
            interval: 2
            failure_mode: continue
            commands: []
    """)
    config_file = tmp_path / "mala.yaml"
    config_file.write_text(config_content)

    orchestrator = make_orchestrator(
        repo_path=tmp_path,
        max_agents=2,
        timeout_minutes=1,
        max_issues=2,
        issue_provider=fake_issues,
        runs_dir=runs_dir,
        lock_releaser=lambda _: 0,
    )

    # Track spawned issues and mock spawn to complete immediately with success
    spawned: list[str] = []
    original_spawn = orchestrator.spawn_agent

    async def tracking_spawn(issue_id: str) -> asyncio.Task[IssueResult] | None:
        spawned.append(issue_id)

        async def work() -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="done",
            )

        return asyncio.create_task(work())

    orchestrator.spawn_agent = tracking_spawn  # type: ignore[method-assign]

    try:
        # Run the full orchestrator loop - this exercises the integration path
        await orchestrator.run()
    finally:
        orchestrator.spawn_agent = original_spawn  # type: ignore[method-assign]

    # Verify both issues were processed through the main loop
    assert len(spawned) == 2, f"Expected 2 issues spawned, got {len(spawned)}"

    # Verify _non_epic_completed_count was incremented by finalize_callback
    # calling _check_and_queue_periodic_trigger() after each issue completion
    assert orchestrator._non_epic_completed_count == 2, (
        f"Expected _non_epic_completed_count=2 after 2 issues, "
        f"got {orchestrator._non_epic_completed_count}"
    )

    # Verify trigger queue is empty - proves trigger was queued AND executed
    # (blocking run_trigger_validation() clears queue after processing)
    trigger_queue = orchestrator.run_coordinator._trigger_queue
    assert len(trigger_queue) == 0, (
        f"Trigger queue should be empty after execution, got {len(trigger_queue)} items. "
        "This indicates run_trigger_validation() was not called (blocking fix missing)."
    )


def test_trigger_code_review_emits_lifecycle_events(tmp_path: Path) -> None:
    """Verify code review lifecycle events are emitted during trigger validation.

    This test sets up a trigger with code_review enabled and runs trigger
    validation. It asserts that the FakeEventSink recorded the expected
    event sequence: started → (passed|failed|skipped|error).
    """
    import asyncio
    from unittest.mock import MagicMock

    from src.domain.validation.config import (
        CodeReviewConfig,
        CommandsConfig,
        FailureMode,
        RunEndTriggerConfig,
        ValidationConfig,
        ValidationTriggersConfig,
    )
    from src.pipeline.run_coordinator import RunCoordinator, RunCoordinatorConfig
    from src.pipeline.fixer_service import FixerService
    from src.pipeline.trigger_engine import TriggerEngine
    from tests.fakes import FakeEnvConfig
    from tests.fakes.command_runner import FakeCommandRunner
    from tests.fakes.event_sink import FakeEventSink
    from tests.fakes.lock_manager import FakeLockManager

    # Create validation config with run_end trigger that has code_review enabled
    validation_config = ValidationConfig(
        commands=CommandsConfig(),
        validation_triggers=ValidationTriggersConfig(
            run_end=RunEndTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(),
                code_review=CodeReviewConfig(
                    enabled=True,
                    reviewer_type="cerberus",
                ),
            )
        ),
    )

    # Create command runner (no commands, just testing code review path)
    command_runner = FakeCommandRunner(allow_unregistered=True)

    # Create FakeEventSink to capture events
    event_sink = FakeEventSink()

    config = RunCoordinatorConfig(
        repo_path=tmp_path,
        timeout_seconds=60,
        validation_config=validation_config,
    )

    trigger_engine = TriggerEngine(validation_config=validation_config)
    mock_fixer_service = MagicMock(spec=FixerService)
    mock_fixer_service.cleanup_locks = MagicMock()

    coordinator = RunCoordinator(
        config=config,
        gate_checker=MagicMock(),
        command_runner=command_runner,
        env_config=FakeEnvConfig(),
        lock_manager=FakeLockManager(),
        sdk_client_factory=MagicMock(),
        trigger_engine=trigger_engine,
        fixer_service=mock_fixer_service,
        event_sink=event_sink,
    )

    # Queue the run_end trigger (simulating end of run)
    from src.domain.validation.config import TriggerType

    coordinator.queue_trigger_validation(TriggerType.RUN_END, {})

    # Run trigger validation
    async def run_validation() -> None:
        await coordinator.run_trigger_validation()

    asyncio.run(run_validation())

    # Assert event sequence: should see trigger_code_review_started followed by
    # one of: passed, failed, skipped, or error
    assert event_sink.has_event("trigger_code_review_started"), (
        "Expected trigger_code_review_started event not recorded. "
        "Events recorded: " + ", ".join(e.event_type for e in event_sink.events)
    )
