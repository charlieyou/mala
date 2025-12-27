"""Unit tests for --wip flag prioritization logic."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.beads_client import BeadsClient
from src.orchestrator import MalaOrchestrator


def make_subprocess_result(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> object:
    """Create a mock subprocess result."""
    from subprocess import CompletedProcess

    return CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


@pytest.fixture
def orchestrator(tmp_path: Path) -> MalaOrchestrator:
    """Create an orchestrator with a temporary repo path."""
    return MalaOrchestrator(
        repo_path=tmp_path,
        max_agents=2,
        timeout_minutes=1,
        max_issues=5,
    )


class TestPrioritizeWipFlag:
    """Test --wip flag prioritization in get_ready_async."""

    @pytest.mark.asyncio
    async def test_prioritize_wip_sorts_in_progress_first(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """When prioritize_wip=True, in_progress issues should come before open."""
        issues_json = json.dumps(
            [
                {"id": "open-1", "priority": 1, "status": "open", "issue_type": "task"},
                {
                    "id": "wip-1",
                    "priority": 2,
                    "status": "in_progress",
                    "issue_type": "task",
                },
                {"id": "open-2", "priority": 3, "status": "open", "issue_type": "task"},
            ]
        )
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            new_callable=AsyncMock,
            return_value=make_subprocess_result(stdout=issues_json),
        ):
            result = await orchestrator.beads.get_ready_async(prioritize_wip=True)

        # in_progress should come first, then open sorted by priority
        assert result == ["wip-1", "open-1", "open-2"]

    @pytest.mark.asyncio
    async def test_prioritize_wip_false_uses_priority_only(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """When prioritize_wip=False (default), sort by priority only."""
        issues_json = json.dumps(
            [
                {
                    "id": "wip-1",
                    "priority": 3,
                    "status": "in_progress",
                    "issue_type": "task",
                },
                {"id": "open-1", "priority": 1, "status": "open", "issue_type": "task"},
                {"id": "open-2", "priority": 2, "status": "open", "issue_type": "task"},
            ]
        )
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            new_callable=AsyncMock,
            return_value=make_subprocess_result(stdout=issues_json),
        ):
            result = await orchestrator.beads.get_ready_async(prioritize_wip=False)

        # Should sort by priority only: 1, 2, 3
        assert result == ["open-1", "open-2", "wip-1"]

    @pytest.mark.asyncio
    async def test_prioritize_wip_respects_priority_within_status(
        self, orchestrator: MalaOrchestrator
    ) -> None:
        """Within each status group, issues should still be sorted by priority."""
        issues_json = json.dumps(
            [
                {
                    "id": "wip-high",
                    "priority": 1,
                    "status": "in_progress",
                    "issue_type": "task",
                },
                {
                    "id": "wip-low",
                    "priority": 3,
                    "status": "in_progress",
                    "issue_type": "task",
                },
                {
                    "id": "open-high",
                    "priority": 1,
                    "status": "open",
                    "issue_type": "task",
                },
                {
                    "id": "open-low",
                    "priority": 2,
                    "status": "open",
                    "issue_type": "task",
                },
            ]
        )
        with patch.object(
            orchestrator.beads,
            "_run_subprocess_async",
            new_callable=AsyncMock,
            return_value=make_subprocess_result(stdout=issues_json),
        ):
            result = await orchestrator.beads.get_ready_async(prioritize_wip=True)

        # in_progress first (sorted by priority), then open (sorted by priority)
        assert result == ["wip-high", "wip-low", "open-high", "open-low"]

    @pytest.mark.asyncio
    async def test_prioritize_wip_default_is_false(self, tmp_path: Path) -> None:
        """Default behavior should not prioritize in_progress issues."""
        beads = BeadsClient(tmp_path)
        issues_json = json.dumps(
            [
                {
                    "id": "wip-1",
                    "priority": 3,
                    "status": "in_progress",
                    "issue_type": "task",
                },
                {"id": "open-1", "priority": 1, "status": "open", "issue_type": "task"},
            ]
        )
        with patch.object(
            beads,
            "_run_subprocess_async",
            new_callable=AsyncMock,
            return_value=make_subprocess_result(stdout=issues_json),
        ):
            # Don't pass prioritize_wip - should default to False
            result = await beads.get_ready_async()

        # Should sort by priority only (default behavior)
        assert result == ["open-1", "wip-1"]


class TestOrchestratorPrioritizeWip:
    """Test orchestrator stores and uses prioritize_wip flag."""

    def test_orchestrator_stores_prioritize_wip(self, tmp_path: Path) -> None:
        """Orchestrator should store prioritize_wip parameter."""
        orch = MalaOrchestrator(repo_path=tmp_path, prioritize_wip=True)
        assert orch.prioritize_wip is True

    def test_orchestrator_default_prioritize_wip_is_false(self, tmp_path: Path) -> None:
        """Default prioritize_wip should be False."""
        orch = MalaOrchestrator(repo_path=tmp_path)
        assert orch.prioritize_wip is False
