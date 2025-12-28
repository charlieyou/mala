"""Unit tests for BeadsClient methods.

Tests for parent epic lookup functionality including
get_parent_epic_async and get_parent_epics_async.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.beads_client import BeadsClient, SubprocessResult


def make_subprocess_result(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> SubprocessResult:
    """Create a mock subprocess result."""
    return SubprocessResult(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class TestGetParentEpicAsync:
    """Test get_parent_epic_async method."""

    @pytest.mark.asyncio
    async def test_returns_parent_epic_id(self, tmp_path: Path) -> None:
        """Should return the parent epic ID when task has a parent epic."""
        beads = BeadsClient(tmp_path)
        tree_json = json.dumps(
            [
                {
                    "id": "task-1",
                    "issue_type": "task",
                    "depth": 0,
                },
                {
                    "id": "epic-1",
                    "issue_type": "epic",
                    "depth": 1,
                },
            ]
        )
        with pytest.MonkeyPatch.context() as mp:
            mock_run = AsyncMock(return_value=make_subprocess_result(stdout=tree_json))
            mp.setattr(beads, "_run_subprocess_async", mock_run)
            result = await beads.get_parent_epic_async("task-1")

        assert result == "epic-1"
        mock_run.assert_called_once_with(
            ["bd", "dep", "tree", "task-1", "--direction=down", "--json"]
        )

    @pytest.mark.asyncio
    async def test_returns_none_for_orphan_task(self, tmp_path: Path) -> None:
        """Should return None when task has no parent epic."""
        beads = BeadsClient(tmp_path)
        tree_json = json.dumps(
            [
                {
                    "id": "orphan-task",
                    "issue_type": "task",
                    "depth": 0,
                },
            ]
        )
        with pytest.MonkeyPatch.context() as mp:
            mock_run = AsyncMock(return_value=make_subprocess_result(stdout=tree_json))
            mp.setattr(beads, "_run_subprocess_async", mock_run)
            result = await beads.get_parent_epic_async("orphan-task")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_bd_failure(self, tmp_path: Path) -> None:
        """Should return None when bd dep tree fails."""
        warnings: list[str] = []
        beads = BeadsClient(tmp_path, log_warning=warnings.append)
        with pytest.MonkeyPatch.context() as mp:
            mock_run = AsyncMock(
                return_value=make_subprocess_result(returncode=1, stderr="bd error")
            )
            mp.setattr(beads, "_run_subprocess_async", mock_run)
            result = await beads.get_parent_epic_async("task-1")

        assert result is None
        assert len(warnings) == 1
        assert "bd dep tree failed" in warnings[0]

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        """Should return None when bd returns invalid JSON."""
        beads = BeadsClient(tmp_path)
        with pytest.MonkeyPatch.context() as mp:
            mock_run = AsyncMock(
                return_value=make_subprocess_result(stdout="not valid json")
            )
            mp.setattr(beads, "_run_subprocess_async", mock_run)
            result = await beads.get_parent_epic_async("task-1")

        assert result is None

    @pytest.mark.asyncio
    async def test_skips_non_epic_ancestors(self, tmp_path: Path) -> None:
        """Should skip non-epic ancestors and return the first epic."""
        beads = BeadsClient(tmp_path)
        tree_json = json.dumps(
            [
                {
                    "id": "task-1",
                    "issue_type": "task",
                    "depth": 0,
                },
                {
                    "id": "task-parent",
                    "issue_type": "task",
                    "depth": 1,
                },
                {
                    "id": "epic-1",
                    "issue_type": "epic",
                    "depth": 2,
                },
            ]
        )
        with pytest.MonkeyPatch.context() as mp:
            mock_run = AsyncMock(return_value=make_subprocess_result(stdout=tree_json))
            mp.setattr(beads, "_run_subprocess_async", mock_run)
            result = await beads.get_parent_epic_async("task-1")

        assert result == "epic-1"


class TestGetParentEpicsAsync:
    """Test get_parent_epics_async batch method."""

    @pytest.mark.asyncio
    async def test_returns_dict_mapping_issues_to_epics(self, tmp_path: Path) -> None:
        """Should return dict mapping each issue to its parent epic."""
        beads = BeadsClient(tmp_path)

        async def mock_get_parent_epic(issue_id: str) -> str | None:
            return {
                "task-1": "epic-a",
                "task-2": "epic-a",
                "task-3": "epic-b",
                "orphan": None,
            }.get(issue_id)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "get_parent_epic_async", mock_get_parent_epic)
            result = await beads.get_parent_epics_async(
                ["task-1", "task-2", "task-3", "orphan"]
            )

        assert result == {
            "task-1": "epic-a",
            "task-2": "epic-a",
            "task-3": "epic-b",
            "orphan": None,
        }

    @pytest.mark.asyncio
    async def test_returns_empty_dict_for_empty_input(self, tmp_path: Path) -> None:
        """Should return empty dict for empty input list."""
        beads = BeadsClient(tmp_path)
        result = await beads.get_parent_epics_async([])
        assert result == {}


class TestParentEpicCaching:
    """Test caching behavior for parent epic lookups."""

    @pytest.mark.asyncio
    async def test_caches_parent_epic_result(self, tmp_path: Path) -> None:
        """Should cache parent epic result and not call subprocess again."""
        beads = BeadsClient(tmp_path)
        tree_json = json.dumps(
            [
                {"id": "task-1", "issue_type": "task", "depth": 0},
                {"id": "epic-1", "issue_type": "epic", "depth": 1},
            ]
        )
        with pytest.MonkeyPatch.context() as mp:
            mock_run = AsyncMock(return_value=make_subprocess_result(stdout=tree_json))
            mp.setattr(beads, "_run_subprocess_async", mock_run)

            # First call should invoke subprocess
            result1 = await beads.get_parent_epic_async("task-1")
            assert result1 == "epic-1"
            assert mock_run.call_count == 1

            # Second call should return from cache
            result2 = await beads.get_parent_epic_async("task-1")
            assert result2 == "epic-1"
            assert mock_run.call_count == 1  # Still 1, not 2

    @pytest.mark.asyncio
    async def test_caches_orphan_result(self, tmp_path: Path) -> None:
        """Should cache None result for orphan tasks."""
        beads = BeadsClient(tmp_path)
        tree_json = json.dumps([{"id": "orphan", "issue_type": "task", "depth": 0}])
        with pytest.MonkeyPatch.context() as mp:
            mock_run = AsyncMock(return_value=make_subprocess_result(stdout=tree_json))
            mp.setattr(beads, "_run_subprocess_async", mock_run)

            # First call
            result1 = await beads.get_parent_epic_async("orphan")
            assert result1 is None
            assert mock_run.call_count == 1

            # Second call should use cache
            result2 = await beads.get_parent_epic_async("orphan")
            assert result2 is None
            assert mock_run.call_count == 1

    @pytest.mark.asyncio
    async def test_batch_uses_cache_for_repeated_issues(self, tmp_path: Path) -> None:
        """Batch method should benefit from caching for repeated issue IDs."""
        beads = BeadsClient(tmp_path)
        tree_json = json.dumps(
            [
                {"id": "task-1", "issue_type": "task", "depth": 0},
                {"id": "epic-1", "issue_type": "epic", "depth": 1},
            ]
        )
        with pytest.MonkeyPatch.context() as mp:
            mock_run = AsyncMock(return_value=make_subprocess_result(stdout=tree_json))
            mp.setattr(beads, "_run_subprocess_async", mock_run)

            # Pre-populate cache
            await beads.get_parent_epic_async("task-1")
            assert mock_run.call_count == 1

            # Batch call with same issue should use cache
            result = await beads.get_parent_epics_async(["task-1", "task-1", "task-1"])
            assert result == {"task-1": "epic-1"}
            assert mock_run.call_count == 1  # No additional calls

    @pytest.mark.asyncio
    async def test_batch_makes_one_call_per_unique_uncached_issue(
        self, tmp_path: Path
    ) -> None:
        """Batch method should make one subprocess call per unique uncached issue."""
        beads = BeadsClient(tmp_path)

        def make_tree_response(issue_id: str) -> str:
            epic_id = "epic-a" if issue_id in ("task-1", "task-2") else "epic-b"
            return json.dumps(
                [
                    {"id": issue_id, "issue_type": "task", "depth": 0},
                    {"id": epic_id, "issue_type": "epic", "depth": 1},
                ]
            )

        call_args: list[list[str]] = []

        async def mock_run(cmd: list[str]) -> SubprocessResult:
            call_args.append(cmd)
            # Extract issue_id from command: bd dep tree <issue_id> --direction=down --json
            issue_id = cmd[3]
            return make_subprocess_result(stdout=make_tree_response(issue_id))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", mock_run)

            result = await beads.get_parent_epics_async(["task-1", "task-2", "task-3"])

        assert result == {
            "task-1": "epic-a",
            "task-2": "epic-a",
            "task-3": "epic-b",
        }
        # Should have made 3 calls (one per unique issue)
        assert len(call_args) == 3

    @pytest.mark.asyncio
    async def test_batch_dedupes_issue_ids(self, tmp_path: Path) -> None:
        """Batch method should dedupe issue_ids, making one call per unique issue."""
        beads = BeadsClient(tmp_path)
        tree_json = json.dumps(
            [
                {"id": "task-1", "issue_type": "task", "depth": 0},
                {"id": "epic-1", "issue_type": "epic", "depth": 1},
            ]
        )
        call_count = 0

        async def mock_run(cmd: list[str]) -> SubprocessResult:
            nonlocal call_count
            call_count += 1
            return make_subprocess_result(stdout=tree_json)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", mock_run)

            # Same issue repeated 3 times - should only make 1 subprocess call
            result = await beads.get_parent_epics_async(["task-1", "task-1", "task-1"])

        assert result == {"task-1": "epic-1"}
        assert call_count == 1  # Only one call despite 3 input entries
