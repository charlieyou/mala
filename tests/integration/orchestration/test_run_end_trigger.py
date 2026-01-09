"""Integration test for run_end trigger skeleton.

This test verifies the orchestrator skeleton methods for run_end trigger:
- _capture_run_start_commit() raises NotImplementedError when run_end configured
- _fire_run_end_trigger() raises NotImplementedError when run_end configured

Per acceptance criteria: Tests must fail with NotImplementedError (not import error).
"""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.orchestration.orchestrator import MalaOrchestrator


@pytest.mark.asyncio
@pytest.mark.integration
async def test_orchestrator_raises_not_implemented_for_run_start_commit(
    tmp_path: Path,
    make_orchestrator: Callable[..., MalaOrchestrator],
) -> None:
    """Integration: _capture_run_start_commit raises NotImplementedError when configured.

    When run_end trigger is configured in mala.yaml, orchestrator.run()
    should fail early with NotImplementedError from _capture_run_start_commit.
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
    )

    # Run should raise NotImplementedError from _capture_run_start_commit
    with pytest.raises(NotImplementedError) as exc_info:
        await orchestrator.run()

    # Verify the error message confirms it's from _capture_run_start_commit
    assert "run_start_commit capture not implemented" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_orchestrator_raises_not_implemented_for_run_end_trigger(
    tmp_path: Path,
    make_orchestrator: Callable[..., MalaOrchestrator],
) -> None:
    """Integration: _fire_run_end_trigger raises NotImplementedError when configured.

    When run_end trigger is configured and issues complete successfully,
    orchestrator.run() should fail with NotImplementedError from _fire_run_end_trigger.

    To reach _fire_run_end_trigger, we bypass _capture_run_start_commit by patching it.
    """
    import asyncio
    from unittest.mock import AsyncMock

    from src.pipeline.issue_result import IssueResult
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

    # Bypass _capture_run_start_commit to reach _fire_run_end_trigger
    orchestrator._capture_run_start_commit = AsyncMock()  # type: ignore[method-assign]

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

    # Run should raise NotImplementedError from _fire_run_end_trigger
    with pytest.raises(NotImplementedError) as exc_info:
        await orchestrator.run()

    # Verify the error message confirms it's from _fire_run_end_trigger
    assert "run_end trigger not implemented" in str(exc_info.value)

    # Verify _capture_run_start_commit was called (bypassed)
    orchestrator._capture_run_start_commit.assert_called_once()


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
