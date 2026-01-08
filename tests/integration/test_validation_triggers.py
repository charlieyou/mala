"""Integration tests for validation_triggers config loading."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.orchestration.orchestrator import MalaOrchestrator


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

    coordinator = RunCoordinator(
        config=config,
        gate_checker=MagicMock(),
        command_runner=command_runner,
        env_config=FakeEnvConfig(),
        lock_manager=FakeLockManager(),
        sdk_client_factory=MagicMock(),
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


def test_orchestrator_fires_periodic_trigger_at_interval(
    tmp_path: Path,
    make_orchestrator: Callable[..., MalaOrchestrator],
) -> None:
    """Test that orchestrator's periodic trigger hook queues triggers correctly.

    This integration test exercises the periodic trigger integration path:
    - Orchestrator has _check_and_queue_periodic_trigger() hook
    - Hook is invoked from finalize_callback after issue completion
    - Hook increments _non_epic_completed_count and queues trigger at interval

    The test directly invokes the hook to verify it exists and is wired up,
    then asserts on the expected behavior (counter increment, trigger queued).
    """
    from src.domain.validation.config import TriggerType
    from src.pipeline.issue_result import IssueResult

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # Create mala.yaml with periodic trigger interval=2
    config_content = dedent("""\
        preset: python-uv

        validation_triggers:
          periodic:
            interval: 2
            failure_mode: continue
            commands:
              - ref: lint
    """)
    config_file = tmp_path / "mala.yaml"
    config_file.write_text(config_content)

    orchestrator = make_orchestrator(
        repo_path=tmp_path,
        max_agents=1,
        timeout_minutes=1,
        max_issues=2,
        runs_dir=runs_dir,
        lock_releaser=lambda _: 0,
    )

    # Create a mock issue result for testing
    mock_result = IssueResult(
        issue_id="test-issue-1",
        agent_id="test-agent",
        success=True,
        summary="done",
    )

    # Verify the hook method exists and can be called
    assert hasattr(orchestrator, "_check_and_queue_periodic_trigger"), (
        "Orchestrator missing _check_and_queue_periodic_trigger hook"
    )
    assert hasattr(orchestrator, "_non_epic_completed_count"), (
        "Orchestrator missing _non_epic_completed_count state"
    )

    # Directly invoke the hook to test the integration path
    # This simulates what finalize_callback does after issue completion
    initial_count = orchestrator._non_epic_completed_count
    orchestrator._check_and_queue_periodic_trigger(mock_result)

    # T011 should implement: increment counter for non-epic issues
    # Until then, this assertion FAILS because stub is a no-op (pass)
    assert orchestrator._non_epic_completed_count == initial_count + 1, (
        "T011 not implemented: _check_and_queue_periodic_trigger should increment "
        "_non_epic_completed_count for non-epic issues"
    )

    # Call again to reach interval=2
    orchestrator._check_and_queue_periodic_trigger(mock_result)

    # Verify trigger was queued via the public queue_trigger_validation method
    # The run_coordinator is a public attribute (no underscore prefix)
    assert hasattr(orchestrator, "run_coordinator"), (
        "Orchestrator missing run_coordinator attribute"
    )

    # T011 should implement: queue PERIODIC trigger when interval reached
    trigger_queue = orchestrator.run_coordinator._trigger_queue
    periodic_triggers = [
        (t, ctx) for t, ctx in trigger_queue if t == TriggerType.PERIODIC
    ]
    assert len(periodic_triggers) >= 1, (
        "T011 not implemented: PERIODIC trigger should be queued when "
        "_non_epic_completed_count reaches configured interval (2)"
    )
