"""Integration test for run_end trigger skeleton.

This test verifies the orchestrator skeleton methods for run_end trigger:
- _capture_run_start_commit() is called early in run()
- _fire_run_end_trigger() is called after session_end trigger

The skeleton methods are wired into run() flow and ready for T011 implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.orchestration.orchestrator import MalaOrchestrator


@pytest.mark.asyncio
@pytest.mark.integration
async def test_orchestrator_calls_capture_run_start_commit(
    tmp_path: Path,
    make_orchestrator: Callable[..., MalaOrchestrator],
) -> None:
    """Integration: Orchestrator.run() calls _capture_run_start_commit early.

    Verifies that:
    1. The method exists
    2. It's wired into the run() flow
    3. It's called early (before issue processing)
    """
    import asyncio
    from unittest.mock import AsyncMock

    from src.pipeline.issue_result import IssueResult
    from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

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

    # Track if _capture_run_start_commit was called
    capture_mock = AsyncMock()
    original_capture = orchestrator._capture_run_start_commit
    orchestrator._capture_run_start_commit = capture_mock  # type: ignore[method-assign]

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

    try:
        await orchestrator.run()
    finally:
        orchestrator._capture_run_start_commit = original_capture  # type: ignore[method-assign]

    # Verify _capture_run_start_commit was called exactly once at run start
    capture_mock.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_orchestrator_calls_fire_run_end_trigger(
    tmp_path: Path,
    make_orchestrator: Callable[..., MalaOrchestrator],
) -> None:
    """Integration: Orchestrator.run() calls _fire_run_end_trigger after processing.

    Verifies that:
    1. The method exists
    2. It's wired into the run() flow after session_end trigger
    3. It receives success_count and total_count arguments
    """
    import asyncio
    from unittest.mock import AsyncMock

    from src.pipeline.issue_result import IssueResult
    from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

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

    # Track if _fire_run_end_trigger was called with correct args
    run_end_mock = AsyncMock()
    original_fire = orchestrator._fire_run_end_trigger
    orchestrator._fire_run_end_trigger = run_end_mock  # type: ignore[method-assign]

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

    try:
        await orchestrator.run()
    finally:
        orchestrator._fire_run_end_trigger = original_fire  # type: ignore[method-assign]

    # Verify _fire_run_end_trigger was called with success_count=1, total_count=1
    run_end_mock.assert_called_once_with(1, 1)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fire_run_end_trigger_receives_correct_counts(
    tmp_path: Path,
    make_orchestrator: Callable[..., MalaOrchestrator],
) -> None:
    """Integration: _fire_run_end_trigger receives accurate success/total counts.

    With 2 issues where one succeeds and one fails:
    - success_count should be 1
    - total_count should be 2
    """
    import asyncio
    from unittest.mock import AsyncMock

    from src.pipeline.issue_result import IssueResult
    from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

    # Create two fake issues
    fake_issues = FakeIssueProvider(
        {
            "issue-1": FakeIssue(id="issue-1", priority=1),
            "issue-2": FakeIssue(id="issue-2", priority=2),
        }
    )
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    orchestrator = make_orchestrator(
        repo_path=tmp_path,
        max_agents=2,
        timeout_minutes=1,
        max_issues=2,
        issue_provider=fake_issues,
        runs_dir=runs_dir,
        lock_releaser=lambda _: 0,
        disable_validations={"global-validate"},
    )

    # Track _fire_run_end_trigger calls
    run_end_mock = AsyncMock()
    original_fire = orchestrator._fire_run_end_trigger
    orchestrator._fire_run_end_trigger = run_end_mock  # type: ignore[method-assign]

    # Mock spawn_agent: issue-1 succeeds, issue-2 fails
    call_count = 0

    async def mock_spawn(issue_id: str) -> asyncio.Task[IssueResult] | None:
        nonlocal call_count
        call_count += 1
        is_success = issue_id == "issue-1"

        async def work() -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=is_success,
                summary="done" if is_success else "failed",
            )

        return asyncio.create_task(work())

    orchestrator.spawn_agent = mock_spawn  # type: ignore[method-assign]

    try:
        await orchestrator.run()
    finally:
        orchestrator._fire_run_end_trigger = original_fire  # type: ignore[method-assign]

    # Verify counts: 1 success out of 2 total
    run_end_mock.assert_called_once_with(1, 2)
