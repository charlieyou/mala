"""Integration tests for validation_triggers config loading."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

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


@pytest.mark.asyncio
async def test_orchestrator_fires_periodic_trigger_at_interval(
    tmp_path: Path,
    make_orchestrator: Callable[..., MalaOrchestrator],
) -> None:
    """Test that orchestrator queues periodic trigger after interval issues complete.

    This integration test exercises:
    - Orchestrator main loop with periodic trigger configuration
    - _non_epic_completed_count tracking
    - _check_and_queue_periodic_trigger() being called on issue completion
    - Trigger queued at RunCoordinator when interval is reached

    Expected to FAIL until T011 implements the actual trigger integration.
    """
    import asyncio

    from src.domain.validation.config import TriggerType
    from src.pipeline.issue_result import IssueResult
    from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

    # Create issues for testing
    fake_issues = FakeIssueProvider(
        {
            "issue-1": FakeIssue(id="issue-1", priority=1),
            "issue-2": FakeIssue(id="issue-2", priority=2),
        }
    )
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
        max_agents=2,
        timeout_minutes=1,
        max_issues=2,
        issue_provider=fake_issues,
        runs_dir=runs_dir,
        lock_releaser=lambda _: 0,
    )

    # Track spawned issues and mock spawn to complete immediately
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
        await orchestrator.run()
    finally:
        orchestrator.spawn_agent = original_spawn  # type: ignore[method-assign]

    # Verify both issues completed
    assert len(spawned) == 2

    # Verify _non_epic_completed_count was incremented
    # This should be 2 after both issues complete
    assert orchestrator._non_epic_completed_count == 2

    # Verify periodic trigger was queued when interval (2) was reached
    # Check the run_coordinator's trigger queue
    trigger_queue = orchestrator.run_coordinator._trigger_queue
    assert len(trigger_queue) >= 1, "Periodic trigger should be queued at interval"

    # Find the periodic trigger in the queue
    periodic_triggers = [
        (t, ctx) for t, ctx in trigger_queue if t == TriggerType.PERIODIC
    ]
    assert len(periodic_triggers) >= 1, "At least one PERIODIC trigger should be queued"
