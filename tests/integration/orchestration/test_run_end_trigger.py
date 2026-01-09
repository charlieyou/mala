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
async def test_orchestrator_raises_not_implemented_when_run_end_configured(
    tmp_path: Path,
    make_orchestrator: Callable[..., MalaOrchestrator],
) -> None:
    """Integration: Orchestrator fails with NotImplementedError when run_end configured.

    When run_end trigger is configured in mala.yaml, orchestrator.run()
    should fail with NotImplementedError from one of the skeleton methods.
    This proves the methods exist and are wired into the run() flow.
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

    # Run should raise NotImplementedError from skeleton method
    with pytest.raises(NotImplementedError) as exc_info:
        await orchestrator.run()

    # Verify the error is from one of the run_end skeleton methods
    error_msg = str(exc_info.value)
    assert (
        "run_start_commit capture not implemented" in error_msg
        or "run_end trigger not implemented" in error_msg
    )


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
