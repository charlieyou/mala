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
        task_tree = json.dumps(
            [
                {"id": "task-1", "issue_type": "task", "depth": 0},
                {"id": "epic-1", "issue_type": "epic", "depth": 1},
            ]
        )
        epic_children = json.dumps(
            [
                {"id": "epic-1", "issue_type": "epic", "depth": 0},
                {"id": "task-1", "issue_type": "task", "depth": 1},
            ]
        )

        async def mock_run(cmd: list[str]) -> SubprocessResult:
            if cmd[4] == "--direction=down":
                return make_subprocess_result(stdout=task_tree)
            else:  # --direction=up (get_epic_children)
                return make_subprocess_result(stdout=epic_children)

        with pytest.MonkeyPatch.context() as mp:
            mock_run_async = AsyncMock(side_effect=mock_run)
            mp.setattr(beads, "_run_subprocess_async", mock_run_async)
            result = await beads.get_parent_epic_async("task-1")

        assert result == "epic-1"
        # Now expects 2 calls: task tree + epic children
        assert mock_run_async.call_count == 2

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
        task_tree = json.dumps(
            [
                {"id": "task-1", "issue_type": "task", "depth": 0},
                {"id": "epic-1", "issue_type": "epic", "depth": 1},
            ]
        )
        epic_children = json.dumps(
            [
                {"id": "epic-1", "issue_type": "epic", "depth": 0},
                {"id": "task-1", "issue_type": "task", "depth": 1},
            ]
        )

        async def mock_run(cmd: list[str]) -> SubprocessResult:
            if cmd[4] == "--direction=down":
                return make_subprocess_result(stdout=task_tree)
            else:
                return make_subprocess_result(stdout=epic_children)

        with pytest.MonkeyPatch.context() as mp:
            mock_run_async = AsyncMock(side_effect=mock_run)
            mp.setattr(beads, "_run_subprocess_async", mock_run_async)

            # First call should invoke subprocess (2 calls: task tree + epic children)
            result1 = await beads.get_parent_epic_async("task-1")
            assert result1 == "epic-1"
            assert mock_run_async.call_count == 2

            # Second call should return from cache
            result2 = await beads.get_parent_epic_async("task-1")
            assert result2 == "epic-1"
            assert mock_run_async.call_count == 2  # Still 2, not 4

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
        task_tree = json.dumps(
            [
                {"id": "task-1", "issue_type": "task", "depth": 0},
                {"id": "epic-1", "issue_type": "epic", "depth": 1},
            ]
        )
        epic_children = json.dumps(
            [
                {"id": "epic-1", "issue_type": "epic", "depth": 0},
                {"id": "task-1", "issue_type": "task", "depth": 1},
            ]
        )

        async def mock_run(cmd: list[str]) -> SubprocessResult:
            if cmd[4] == "--direction=down":
                return make_subprocess_result(stdout=task_tree)
            else:
                return make_subprocess_result(stdout=epic_children)

        with pytest.MonkeyPatch.context() as mp:
            mock_run_async = AsyncMock(side_effect=mock_run)
            mp.setattr(beads, "_run_subprocess_async", mock_run_async)

            # Pre-populate cache (2 calls: task tree + epic children)
            await beads.get_parent_epic_async("task-1")
            assert mock_run_async.call_count == 2

            # Batch call with same issue should use cache
            result = await beads.get_parent_epics_async(["task-1", "task-1", "task-1"])
            assert result == {"task-1": "epic-1"}
            assert mock_run_async.call_count == 2  # No additional calls

    @pytest.mark.asyncio
    async def test_batch_makes_one_call_per_unique_uncached_issue(
        self, tmp_path: Path
    ) -> None:
        """Batch method should make calls proportional to unique epics, not tasks."""
        beads = BeadsClient(tmp_path)

        def make_tree_response(issue_id: str) -> str:
            epic_id = "epic-a" if issue_id in ("task-1", "task-2") else "epic-b"
            return json.dumps(
                [
                    {"id": issue_id, "issue_type": "task", "depth": 0},
                    {"id": epic_id, "issue_type": "epic", "depth": 1},
                ]
            )

        def make_children_response(epic_id: str) -> str:
            if epic_id == "epic-a":
                return json.dumps(
                    [
                        {"id": "epic-a", "issue_type": "epic", "depth": 0},
                        {"id": "task-1", "issue_type": "task", "depth": 1},
                        {"id": "task-2", "issue_type": "task", "depth": 1},
                    ]
                )
            else:
                return json.dumps(
                    [
                        {"id": "epic-b", "issue_type": "epic", "depth": 0},
                        {"id": "task-3", "issue_type": "task", "depth": 1},
                    ]
                )

        call_args: list[list[str]] = []

        async def mock_run(cmd: list[str]) -> SubprocessResult:
            call_args.append(cmd)
            issue_id = cmd[3]
            direction = cmd[4]
            if direction == "--direction=down":
                return make_subprocess_result(stdout=make_tree_response(issue_id))
            else:  # --direction=up
                return make_subprocess_result(stdout=make_children_response(issue_id))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", mock_run)

            result = await beads.get_parent_epics_async(["task-1", "task-2", "task-3"])

        assert result == {
            "task-1": "epic-a",
            "task-2": "epic-a",
            "task-3": "epic-b",
        }
        # With epic-level caching:
        # - task-1: 2 calls (tree + epic-a children) -> caches task-1, task-2
        # - task-2: 0 calls (already cached from task-1's epic children)
        # - task-3: 2 calls (tree + epic-b children) -> caches task-3
        # Total: 4 calls (2 per unique epic)
        assert len(call_args) == 4

    @pytest.mark.asyncio
    async def test_batch_dedupes_issue_ids(self, tmp_path: Path) -> None:
        """Batch method should dedupe issue_ids, making one call per unique issue."""
        beads = BeadsClient(tmp_path)
        task_tree = json.dumps(
            [
                {"id": "task-1", "issue_type": "task", "depth": 0},
                {"id": "epic-1", "issue_type": "epic", "depth": 1},
            ]
        )
        epic_children = json.dumps(
            [
                {"id": "epic-1", "issue_type": "epic", "depth": 0},
                {"id": "task-1", "issue_type": "task", "depth": 1},
            ]
        )

        async def mock_run(cmd: list[str]) -> SubprocessResult:
            if cmd[4] == "--direction=down":
                return make_subprocess_result(stdout=task_tree)
            else:
                return make_subprocess_result(stdout=epic_children)

        call_count = 0

        async def counting_mock_run(cmd: list[str]) -> SubprocessResult:
            nonlocal call_count
            call_count += 1
            return await mock_run(cmd)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", counting_mock_run)

            # Same issue repeated 3 times - should only make 2 subprocess calls
            # (1 for task tree + 1 for epic children)
            result = await beads.get_parent_epics_async(["task-1", "task-1", "task-1"])

        assert result == {"task-1": "epic-1"}
        assert call_count == 2  # task tree + epic children

    @pytest.mark.asyncio
    async def test_epic_level_caching_caches_all_siblings(self, tmp_path: Path) -> None:
        """When discovering a parent epic, all sibling tasks should be cached.

        This tests the key optimization: O(unique epics) calls, not O(unique tasks).
        """
        beads = BeadsClient(tmp_path)

        # Mock responses for bd dep tree commands
        task1_tree = json.dumps(
            [
                {"id": "task-1", "issue_type": "task", "depth": 0},
                {"id": "epic-a", "issue_type": "epic", "depth": 1},
            ]
        )
        epic_children = json.dumps(
            [
                {"id": "epic-a", "issue_type": "epic", "depth": 0},
                {"id": "task-1", "issue_type": "task", "depth": 1},
                {"id": "task-2", "issue_type": "task", "depth": 1},
                {"id": "task-3", "issue_type": "task", "depth": 1},
            ]
        )

        call_log: list[str] = []

        async def mock_run(cmd: list[str]) -> SubprocessResult:
            # Log which issue/epic we're querying
            issue_id = cmd[3]
            direction = cmd[4]  # --direction=down or --direction=up
            call_log.append(f"{issue_id}:{direction}")

            if issue_id == "task-1" and direction == "--direction=down":
                return make_subprocess_result(stdout=task1_tree)
            elif issue_id == "epic-a" and direction == "--direction=up":
                return make_subprocess_result(stdout=epic_children)
            return make_subprocess_result(returncode=1, stderr="not found")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", mock_run)

            # Look up 3 tasks that are all under epic-a
            result = await beads.get_parent_epics_async(["task-1", "task-2", "task-3"])

        # All should map to epic-a
        assert result == {
            "task-1": "epic-a",
            "task-2": "epic-a",
            "task-3": "epic-a",
        }

        # Key assertion: we should only make 2 calls total:
        # 1. bd dep tree task-1 --direction=down (to find epic-a)
        # 2. bd dep tree epic-a --direction=up (to find all children)
        # task-2 and task-3 should be cached from step 2
        assert len(call_log) == 2
        assert "task-1:--direction=down" in call_log
        assert "epic-a:--direction=up" in call_log


class TestGetReadyAsyncBlockedEpicFiltering:
    """Test that get_ready_async excludes tasks with blocked parent epics."""

    @pytest.mark.asyncio
    async def test_excludes_tasks_with_blocked_parent_epic(
        self, tmp_path: Path
    ) -> None:
        """Tasks whose parent epic is blocked should be excluded from ready list."""
        beads = BeadsClient(tmp_path)

        # Mock bd ready returning two tasks
        ready_response = json.dumps(
            [
                {"id": "task-1", "issue_type": "task", "priority": 1},
                {"id": "task-2", "issue_type": "task", "priority": 1},
            ]
        )

        # Mock parent epic lookup - both under epic-a
        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {issue_id: "epic-a" for issue_id in issue_ids}

        # Mock epic-a as blocked
        async def mock_is_epic_blocked(epic_id: str) -> bool:
            return epic_id == "epic-a"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_subprocess_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)
            mp.setattr(beads, "_is_epic_blocked_async", mock_is_epic_blocked)

            result = await beads.get_ready_async()

        # Both tasks should be excluded since epic-a is blocked
        assert result == []

    @pytest.mark.asyncio
    async def test_orphan_tasks_appear_when_no_parent_epic(
        self, tmp_path: Path
    ) -> None:
        """Orphan tasks (no parent epic) should still appear in ready list."""
        beads = BeadsClient(tmp_path)

        ready_response = json.dumps(
            [
                {"id": "orphan-task", "issue_type": "task", "priority": 1},
            ]
        )

        # Mock parent epic lookup - orphan has no parent
        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {issue_id: None for issue_id in issue_ids}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_subprocess_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)

            result = await beads.get_ready_async()

        # Orphan task should still appear
        assert result == ["orphan-task"]

    @pytest.mark.asyncio
    async def test_task_appears_when_parent_epic_unblocked(
        self, tmp_path: Path
    ) -> None:
        """Tasks should appear when their parent epic is not blocked."""
        beads = BeadsClient(tmp_path)

        ready_response = json.dumps(
            [
                {"id": "task-1", "issue_type": "task", "priority": 1},
            ]
        )

        # Mock parent epic lookup
        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {issue_id: "epic-a" for issue_id in issue_ids}

        # Mock epic-a as NOT blocked
        async def mock_is_epic_blocked(epic_id: str) -> bool:
            return False

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_subprocess_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)
            mp.setattr(beads, "_is_epic_blocked_async", mock_is_epic_blocked)

            result = await beads.get_ready_async()

        # Task should appear since epic-a is not blocked
        assert result == ["task-1"]

    @pytest.mark.asyncio
    async def test_mixed_blocked_and_unblocked_epics(self, tmp_path: Path) -> None:
        """Tasks under blocked epics excluded, tasks under unblocked epics included."""
        beads = BeadsClient(tmp_path)

        ready_response = json.dumps(
            [
                {"id": "blocked-task", "issue_type": "task", "priority": 1},
                {"id": "ready-task", "issue_type": "task", "priority": 1},
                {"id": "orphan-task", "issue_type": "task", "priority": 1},
            ]
        )

        # Mock parent epic lookup
        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            mapping = {
                "blocked-task": "blocked-epic",
                "ready-task": "ready-epic",
                "orphan-task": None,
            }
            return {issue_id: mapping.get(issue_id) for issue_id in issue_ids}

        # Mock epic blocked status
        async def mock_is_epic_blocked(epic_id: str) -> bool:
            return epic_id == "blocked-epic"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_subprocess_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)
            mp.setattr(beads, "_is_epic_blocked_async", mock_is_epic_blocked)

            result = await beads.get_ready_async()

        # blocked-task excluded, ready-task and orphan-task included
        assert "blocked-task" not in result
        assert "ready-task" in result
        assert "orphan-task" in result


class TestIsEpicBlockedAsync:
    """Test _is_epic_blocked_async method."""

    @pytest.mark.asyncio
    async def test_returns_true_for_blocked_status(self, tmp_path: Path) -> None:
        """Should return True when epic has status=blocked."""
        beads = BeadsClient(tmp_path)
        epic_json = json.dumps({"id": "epic-1", "status": "blocked"})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_subprocess_result(stdout=epic_json)),
            )
            result = await beads._is_epic_blocked_async("epic-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_for_blocked_by_dependency(self, tmp_path: Path) -> None:
        """Should return True when epic has blocked_by field."""
        beads = BeadsClient(tmp_path)
        epic_json = json.dumps(
            {
                "id": "epic-1",
                "status": "open",
                "blocked_by": "other-epic",
            }
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_subprocess_result(stdout=epic_json)),
            )
            result = await beads._is_epic_blocked_async("epic-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_for_unblocked_epic(self, tmp_path: Path) -> None:
        """Should return False when epic is not blocked."""
        beads = BeadsClient(tmp_path)
        epic_json = json.dumps(
            {
                "id": "epic-1",
                "status": "open",
                "blocked_by": None,
            }
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_subprocess_result(stdout=epic_json)),
            )
            result = await beads._is_epic_blocked_async("epic-1")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_bd_failure(self, tmp_path: Path) -> None:
        """Should return False when bd show fails (avoid hiding tasks)."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(
                    return_value=make_subprocess_result(returncode=1, stderr="error")
                ),
            )
            result = await beads._is_epic_blocked_async("epic-1")

        assert result is False

    @pytest.mark.asyncio
    async def test_caches_blocked_status(self, tmp_path: Path) -> None:
        """Should cache blocked status and not call subprocess again."""
        beads = BeadsClient(tmp_path)
        epic_json = json.dumps({"id": "epic-1", "status": "blocked"})

        with pytest.MonkeyPatch.context() as mp:
            mock_run = AsyncMock(return_value=make_subprocess_result(stdout=epic_json))
            mp.setattr(beads, "_run_subprocess_async", mock_run)

            # First call
            result1 = await beads._is_epic_blocked_async("epic-1")
            assert result1 is True
            assert mock_run.call_count == 1

            # Second call should use cache
            result2 = await beads._is_epic_blocked_async("epic-1")
            assert result2 is True
            assert mock_run.call_count == 1  # Still 1

    @pytest.mark.asyncio
    async def test_handles_list_response(self, tmp_path: Path) -> None:
        """Should handle bd show returning a list (single item)."""
        beads = BeadsClient(tmp_path)
        epic_json = json.dumps([{"id": "epic-1", "status": "blocked"}])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_subprocess_result(stdout=epic_json)),
            )
            result = await beads._is_epic_blocked_async("epic-1")

        assert result is True
