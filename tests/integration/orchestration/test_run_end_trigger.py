"""Integration test for run_end trigger.

This test verifies the orchestrator run_end trigger implementation:
- _capture_run_start_commit() captures HEAD at run start
- _fire_run_end_trigger() queues trigger based on fire_on setting
"""

from __future__ import annotations

import asyncio
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from src.pipeline.issue_result import IssueResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.orchestration.orchestrator import MalaOrchestrator


@pytest.mark.asyncio
@pytest.mark.integration
async def test_run_end_trigger_queued_on_success(
    tmp_path: Path,
    make_orchestrator: Callable[..., MalaOrchestrator],
) -> None:
    """Integration: run_end trigger is queued when run completes with success.

    When run_end trigger is configured in mala.yaml with fire_on=success (default),
    and all issues succeed, the trigger should be queued.
    """
    from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

    # Create mala.yaml with run_end trigger configured
    config_content = dedent("""\
        preset: python-uv

        validation_triggers:
          run_end:
            failure_mode: continue
            commands: []
    """)
    config_file = tmp_path / "mala.yaml"
    config_file.write_text(config_content)

    # Create fake issues for the orchestrator
    fake_issues = FakeIssueProvider({"issue-1": FakeIssue(id="issue-1", priority=1)})
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    orchestrator = make_orchestrator(
        repo_path=tmp_path,
        max_agents=1,
        timeout_minutes=1,
        max_issues=1,
        issue_provider=fake_issues,
        runs_dir=runs_dir,
        lock_releaser=lambda _: 0,
        disable_validations={"global-validate"},
    )

    # Track trigger queued events
    trigger_queued_events: list[tuple[str, str]] = []
    original_on_trigger = orchestrator.event_sink.on_trigger_validation_queued

    def track_trigger(trigger_type: str, context: str) -> None:
        trigger_queued_events.append((trigger_type, context))
        original_on_trigger(trigger_type, context)

    orchestrator.event_sink.on_trigger_validation_queued = track_trigger

    # Mock spawn_agent to complete immediately with success
    async def mock_spawn(issue_id: str) -> asyncio.Task[IssueResult] | None:
        async def work() -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="done",
            )

        return asyncio.create_task(work())

    orchestrator.spawn_agent = mock_spawn  # type: ignore[method-assign]

    # Run should complete without error
    success_count, total_count = await orchestrator.run()

    assert success_count == 1
    assert total_count == 1

    # Verify run_end trigger was queued
    run_end_triggers = [t for t in trigger_queued_events if t[0] == "run_end"]
    assert len(run_end_triggers) == 1
    assert "success_count=1" in run_end_triggers[0][1]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_no_error_when_run_end_trigger_not_configured(
    tmp_path: Path,
    make_orchestrator: Callable[..., MalaOrchestrator],
) -> None:
    """Integration: No NotImplementedError when run_end trigger is not configured.

    Without run_end trigger in config, skeleton methods should be no-ops
    and orchestrator.run() should complete normally.
    """
    import asyncio

    from src.pipeline.issue_result import IssueResult
    from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

    # Create mala.yaml WITHOUT run_end trigger
    config_content = dedent("""\
        preset: python-uv
    """)
    config_file = tmp_path / "mala.yaml"
    config_file.write_text(config_content)

    # Create fake issues for the orchestrator
    fake_issues = FakeIssueProvider({"issue-1": FakeIssue(id="issue-1", priority=1)})
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    orchestrator = make_orchestrator(
        repo_path=tmp_path,
        max_agents=1,
        timeout_minutes=1,
        max_issues=1,
        issue_provider=fake_issues,
        runs_dir=runs_dir,
        lock_releaser=lambda _: 0,
        disable_validations={"global-validate"},
    )

    # Mock spawn_agent to complete immediately with success
    async def mock_spawn(issue_id: str) -> asyncio.Task[IssueResult] | None:
        async def work() -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="done",
            )

        return asyncio.create_task(work())

    orchestrator.spawn_agent = mock_spawn  # type: ignore[method-assign]

    # Should complete without NotImplementedError
    success_count, total_count = await orchestrator.run()

    assert success_count == 1
    assert total_count == 1
