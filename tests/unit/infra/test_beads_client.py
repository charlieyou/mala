"""Unit tests for BeadsClient methods.

Tests for parent epic lookup functionality including
get_parent_epic_async and get_parent_epics_async.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.infra.clients.beads_client import BeadsClient
from src.infra.tools.command_runner import CommandResult


def make_command_result(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> CommandResult:
    """Create a mock command result."""
    return CommandResult(
        command=[],
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

        with pytest.MonkeyPatch.context() as mp:
            mock_run_async = AsyncMock(
                return_value=make_command_result(stdout=task_tree)
            )
            mp.setattr(beads, "_run_subprocess_async", mock_run_async)
            result = await beads.get_parent_epic_async("task-1")

        assert result == "epic-1"

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
            mock_run = AsyncMock(return_value=make_command_result(stdout=tree_json))
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
                return_value=make_command_result(returncode=1, stderr="bd error")
            )
            mp.setattr(beads, "_run_subprocess_async", mock_run)
            result = await beads.get_parent_epic_async("task-1")

        assert result is None
        assert len(warnings) == 1
        assert "br dep tree failed" in warnings[0]

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        """Should return None when bd returns invalid JSON."""
        beads = BeadsClient(tmp_path)
        with pytest.MonkeyPatch.context() as mp:
            mock_run = AsyncMock(
                return_value=make_command_result(stdout="not valid json")
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
            mock_run = AsyncMock(return_value=make_command_result(stdout=tree_json))
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

        with pytest.MonkeyPatch.context() as mp:
            mock_run_async = AsyncMock(
                return_value=make_command_result(stdout=task_tree)
            )
            mp.setattr(beads, "_run_subprocess_async", mock_run_async)

            # First call should invoke subprocess (1 call for task tree)
            result1 = await beads.get_parent_epic_async("task-1")
            assert result1 == "epic-1"
            assert mock_run_async.call_count == 1

            # Second call should return from cache
            result2 = await beads.get_parent_epic_async("task-1")
            assert result2 == "epic-1"
            assert mock_run_async.call_count == 1  # Still 1, not 2

    @pytest.mark.asyncio
    async def test_caches_orphan_result(self, tmp_path: Path) -> None:
        """Should cache None result for orphan tasks."""
        beads = BeadsClient(tmp_path)
        tree_json = json.dumps([{"id": "orphan", "issue_type": "task", "depth": 0}])
        with pytest.MonkeyPatch.context() as mp:
            mock_run = AsyncMock(return_value=make_command_result(stdout=tree_json))
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

        with pytest.MonkeyPatch.context() as mp:
            mock_run_async = AsyncMock(
                return_value=make_command_result(stdout=task_tree)
            )
            mp.setattr(beads, "_run_subprocess_async", mock_run_async)

            # Pre-populate cache (1 call for task tree)
            await beads.get_parent_epic_async("task-1")
            assert mock_run_async.call_count == 1

            # Batch call with same issue should use cache
            result = await beads.get_parent_epics_async(["task-1", "task-1", "task-1"])
            assert result == {"task-1": "epic-1"}
            assert mock_run_async.call_count == 1  # No additional calls

    @pytest.mark.asyncio
    async def test_batch_makes_one_call_per_unique_issue(self, tmp_path: Path) -> None:
        """Batch method should make one call per unique uncached issue."""
        beads = BeadsClient(tmp_path)

        def make_tree_response(issue_id: str) -> str:
            epic_id = "epic-a" if issue_id in ("task-1", "task-2") else "epic-b"
            return json.dumps(
                [
                    {"id": issue_id, "issue_type": "task", "depth": 0},
                    {"id": epic_id, "issue_type": "epic", "depth": 1},
                ]
            )

        call_count = 0

        async def mock_run(cmd: list[str]) -> CommandResult:
            nonlocal call_count
            call_count += 1
            issue_id = cmd[3]
            return make_command_result(stdout=make_tree_response(issue_id))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", mock_run)

            result = await beads.get_parent_epics_async(["task-1", "task-2", "task-3"])

        assert result == {
            "task-1": "epic-a",
            "task-2": "epic-a",
            "task-3": "epic-b",
        }
        # One call per unique task (no sibling caching)
        assert call_count == 3

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

        call_count = 0

        async def counting_mock_run(cmd: list[str]) -> CommandResult:
            nonlocal call_count
            call_count += 1
            return make_command_result(stdout=task_tree)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", counting_mock_run)

            # Same issue repeated 3 times - should only make 1 subprocess call
            result = await beads.get_parent_epics_async(["task-1", "task-1", "task-1"])

        assert result == {"task-1": "epic-1"}
        assert call_count == 1  # Only 1 call due to deduplication

    @pytest.mark.asyncio
    async def test_nested_epics_preserve_immediate_parent(self, tmp_path: Path) -> None:
        """Tasks under nested epics should have their immediate parent epic, not the top-level.

        Structure:
            top-epic (epic)
              └── child-epic (epic)
                    └── task-1 (task)

        task-1's parent epic should be child-epic, NOT top-epic.
        """
        beads = BeadsClient(tmp_path)

        # task-1's ancestor tree: task-1 -> child-epic -> top-epic
        task1_tree = json.dumps(
            [
                {"id": "task-1", "issue_type": "task", "depth": 0},
                {"id": "child-epic", "issue_type": "epic", "depth": 1},
                {"id": "top-epic", "issue_type": "epic", "depth": 2},
            ]
        )

        # child-epic's ancestor tree: child-epic -> top-epic
        child_epic_tree = json.dumps(
            [
                {"id": "child-epic", "issue_type": "epic", "depth": 0},
                {"id": "top-epic", "issue_type": "epic", "depth": 1},
            ]
        )

        async def mock_run(cmd: list[str]) -> CommandResult:
            issue_id = cmd[3]
            if issue_id == "task-1":
                return make_command_result(stdout=task1_tree)
            elif issue_id == "child-epic":
                return make_command_result(stdout=child_epic_tree)
            return make_command_result(returncode=1, stderr="not found")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", mock_run)

            # Look up task-1's parent epic
            result = await beads.get_parent_epic_async("task-1")

        # task-1's immediate parent epic should be child-epic, not top-epic
        assert result == "child-epic"

    @pytest.mark.asyncio
    async def test_nested_epics_batch_correctness(self, tmp_path: Path) -> None:
        """Batch lookup with nested epics should return correct immediate parents.

        Structure:
            top-epic (epic)
              ├── task-a (task, direct child of top-epic)
              └── child-epic (epic)
                    └── task-b (task, child of child-epic)

        task-a's parent should be top-epic
        task-b's parent should be child-epic
        """
        beads = BeadsClient(tmp_path)

        # task-a is directly under top-epic
        task_a_tree = json.dumps(
            [
                {"id": "task-a", "issue_type": "task", "depth": 0},
                {"id": "top-epic", "issue_type": "epic", "depth": 1},
            ]
        )

        # task-b is under child-epic which is under top-epic
        task_b_tree = json.dumps(
            [
                {"id": "task-b", "issue_type": "task", "depth": 0},
                {"id": "child-epic", "issue_type": "epic", "depth": 1},
                {"id": "top-epic", "issue_type": "epic", "depth": 2},
            ]
        )

        async def mock_run(cmd: list[str]) -> CommandResult:
            issue_id = cmd[3]
            if issue_id == "task-a":
                return make_command_result(stdout=task_a_tree)
            elif issue_id == "task-b":
                return make_command_result(stdout=task_b_tree)
            return make_command_result(returncode=1, stderr="not found")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", mock_run)

            result = await beads.get_parent_epics_async(["task-a", "task-b"])

        # Each task should have its immediate parent epic
        assert result == {
            "task-a": "top-epic",
            "task-b": "child-epic",  # NOT top-epic
        }


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
                AsyncMock(return_value=make_command_result(stdout=ready_response)),
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
                AsyncMock(return_value=make_command_result(stdout=ready_response)),
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
                AsyncMock(return_value=make_command_result(stdout=ready_response)),
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
                AsyncMock(return_value=make_command_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)
            mp.setattr(beads, "_is_epic_blocked_async", mock_is_epic_blocked)

            result = await beads.get_ready_async()

        # blocked-task excluded, ready-task and orphan-task included
        assert "blocked-task" not in result
        assert "ready-task" in result
        assert "orphan-task" in result

    @pytest.mark.asyncio
    async def test_nested_epics_blocked_filtering_uses_immediate_parent(
        self, tmp_path: Path
    ) -> None:
        """Blocked-epic filtering should use immediate parent epic, not top-level.

        Structure:
            top-epic (epic, NOT blocked)
              └── child-epic (epic, BLOCKED)
                    └── task-1 (task)

        task-1 should be excluded because its immediate parent (child-epic) is blocked,
        even though top-epic is not blocked.
        """
        beads = BeadsClient(tmp_path)

        ready_response = json.dumps(
            [{"id": "task-1", "issue_type": "task", "priority": 1}]
        )

        # task-1's immediate parent is child-epic (not top-epic)
        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {"task-1": "child-epic"}

        # child-epic is blocked, top-epic is not
        async def mock_is_epic_blocked(epic_id: str) -> bool:
            return epic_id == "child-epic"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)
            mp.setattr(beads, "_is_epic_blocked_async", mock_is_epic_blocked)

            result = await beads.get_ready_async()

        # task-1 should be excluded because child-epic is blocked
        assert result == []

    @pytest.mark.asyncio
    async def test_nested_epics_not_blocked_by_ancestor(self, tmp_path: Path) -> None:
        """Tasks should not be blocked by ancestor epics, only immediate parent.

        Structure:
            top-epic (epic, BLOCKED)
              └── child-epic (epic, NOT blocked)
                    └── task-1 (task)

        task-1's immediate parent is child-epic (not blocked), so task-1 should appear
        even though top-epic is blocked.
        """
        beads = BeadsClient(tmp_path)

        ready_response = json.dumps(
            [{"id": "task-1", "issue_type": "task", "priority": 1}]
        )

        # task-1's immediate parent is child-epic
        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {"task-1": "child-epic"}

        # top-epic is blocked, but child-epic is not
        async def mock_is_epic_blocked(epic_id: str) -> bool:
            return epic_id == "top-epic"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)
            mp.setattr(beads, "_is_epic_blocked_async", mock_is_epic_blocked)

            result = await beads.get_ready_async()

        # task-1 should appear because its immediate parent (child-epic) is not blocked
        assert result == ["task-1"]


class TestNestedEpicsFocusGrouping:
    """Test focus mode grouping with nested epics."""

    @pytest.mark.asyncio
    async def test_nested_epics_group_by_immediate_parent(self, tmp_path: Path) -> None:
        """Focus grouping should group tasks by their immediate parent epic.

        Structure:
            top-epic (epic)
              ├── task-a (task, direct child)
              └── child-epic (epic)
                    ├── task-b (task)
                    └── task-c (task)

        task-b and task-c should be grouped under child-epic.
        task-a should be grouped under top-epic.
        """
        beads = BeadsClient(tmp_path)

        ready_response = json.dumps(
            [
                {
                    "id": "task-a",
                    "issue_type": "task",
                    "priority": 1,
                    "updated_at": "2025-01-01T10:00:00Z",
                },
                {
                    "id": "task-b",
                    "issue_type": "task",
                    "priority": 2,
                    "updated_at": "2025-01-01T11:00:00Z",
                },
                {
                    "id": "task-c",
                    "issue_type": "task",
                    "priority": 1,
                    "updated_at": "2025-01-01T09:00:00Z",
                },
            ]
        )

        # Immediate parent mappings
        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {
                "task-a": "top-epic",
                "task-b": "child-epic",
                "task-c": "child-epic",
            }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)

            result = await beads.get_ready_async(focus=True)

        # With focus=True, tasks should be grouped by immediate parent epic
        # task-a is alone in top-epic group
        # task-b and task-c are in child-epic group
        # Groups ordered by min priority, tasks within by (priority, updated DESC)
        assert len(result) == 3
        # Both groups have min priority 1, so order depends on max_updated
        # top-epic max_updated = 2025-01-01T10:00:00Z
        # child-epic max_updated = 2025-01-01T11:00:00Z (task-b)
        # child-epic group should come first (more recent)
        # Within child-epic: task-c (P1) before task-b (P2)
        assert result[0] == "task-c"
        assert result[1] == "task-b"
        assert result[2] == "task-a"

    @pytest.mark.asyncio
    async def test_nested_epics_focus_grouping_preserves_sibling_order(
        self, tmp_path: Path
    ) -> None:
        """Tasks in same nested epic should stay grouped together.

        Structure:
            top-epic (epic)
              └── child-epic (epic)
                    ├── task-1 (P1)
                    └── task-2 (P2)

        Both tasks are under child-epic and should be grouped together,
        not incorrectly grouped under top-epic.
        """
        beads = BeadsClient(tmp_path)

        ready_response = json.dumps(
            [
                {
                    "id": "task-1",
                    "issue_type": "task",
                    "priority": 1,
                    "updated_at": "2025-01-01T10:00:00Z",
                },
                {
                    "id": "task-2",
                    "issue_type": "task",
                    "priority": 2,
                    "updated_at": "2025-01-01T09:00:00Z",
                },
            ]
        )

        # Both tasks have child-epic as immediate parent
        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {
                "task-1": "child-epic",
                "task-2": "child-epic",
            }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)

            result = await beads.get_ready_async(focus=True)

        # Both tasks grouped under child-epic, ordered by priority
        assert result == ["task-1", "task-2"]


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
                AsyncMock(return_value=make_command_result(stdout=epic_json)),
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
                AsyncMock(return_value=make_command_result(stdout=epic_json)),
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
                AsyncMock(return_value=make_command_result(stdout=epic_json)),
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
                    return_value=make_command_result(returncode=1, stderr="error")
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
            mock_run = AsyncMock(return_value=make_command_result(stdout=epic_json))
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
                AsyncMock(return_value=make_command_result(stdout=epic_json)),
            )
            result = await beads._is_epic_blocked_async("epic-1")

        assert result is True


class TestGetReadyMethodsConsistentOrdering:
    """Test that get_ready_async and get_ready_issues_async return consistent ordering."""

    @pytest.mark.asyncio
    async def test_both_methods_return_same_order(self, tmp_path: Path) -> None:
        """get_ready_async and get_ready_issues_async should return IDs in same order."""
        beads = BeadsClient(tmp_path)

        ready_response = json.dumps(
            [
                {
                    "id": "task-1",
                    "issue_type": "task",
                    "priority": 2,
                    "updated_at": "2025-01-01T10:00:00Z",
                },
                {
                    "id": "task-2",
                    "issue_type": "task",
                    "priority": 1,
                    "updated_at": "2025-01-01T11:00:00Z",
                },
                {
                    "id": "task-3",
                    "issue_type": "task",
                    "priority": 1,
                    "updated_at": "2025-01-01T09:00:00Z",
                },
            ]
        )

        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {
                "task-1": "epic-a",
                "task-2": "epic-a",
                "task-3": "epic-b",
            }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)

            ids_result = await beads.get_ready_async(focus=True)
            issues_result = await beads.get_ready_issues_async(focus=True)

        # Both methods should return the same order
        issues_ids = [str(i["id"]) for i in issues_result]
        assert ids_result == issues_ids

    @pytest.mark.asyncio
    async def test_both_methods_consistent_with_include_wip(
        self, tmp_path: Path
    ) -> None:
        """Both methods should have consistent ordering with include_wip=True."""
        beads = BeadsClient(tmp_path)

        ready_response = json.dumps(
            [
                {
                    "id": "task-1",
                    "issue_type": "task",
                    "priority": 1,
                    "status": "ready",
                    "updated_at": "2025-01-01T10:00:00Z",
                },
            ]
        )

        wip_response = json.dumps(
            [
                {
                    "id": "task-2",
                    "issue_type": "task",
                    "priority": 2,
                    "status": "in_progress",
                    "updated_at": "2025-01-01T09:00:00Z",
                },
            ]
        )

        async def mock_run(cmd: list[str]) -> CommandResult:
            if "--status" in cmd and "in_progress" in cmd:
                return make_command_result(stdout=wip_response)
            return make_command_result(stdout=ready_response)

        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {issue_id: None for issue_id in issue_ids}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", mock_run)
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)

            ids_result = await beads.get_ready_async(include_wip=True, focus=False)
            issues_result = await beads.get_ready_issues_async(
                include_wip=True, focus=False
            )

        # Both methods should return consistent ordering
        issues_ids = [str(i["id"]) for i in issues_result]
        assert ids_result == issues_ids
        assert ids_result == ["task-1", "task-2"]

    @pytest.mark.asyncio
    async def test_both_methods_consistent_with_focus_false(
        self, tmp_path: Path
    ) -> None:
        """Both methods should have consistent ordering with focus=False (priority only)."""
        beads = BeadsClient(tmp_path)

        ready_response = json.dumps(
            [
                {
                    "id": "task-1",
                    "issue_type": "task",
                    "priority": 3,
                    "updated_at": "2025-01-01T10:00:00Z",
                },
                {
                    "id": "task-2",
                    "issue_type": "task",
                    "priority": 1,
                    "updated_at": "2025-01-01T09:00:00Z",
                },
                {
                    "id": "task-3",
                    "issue_type": "task",
                    "priority": 2,
                    "updated_at": "2025-01-01T11:00:00Z",
                },
            ]
        )

        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {issue_id: None for issue_id in issue_ids}

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)

            ids_result = await beads.get_ready_async(focus=False)
            issues_result = await beads.get_ready_issues_async(focus=False)

        # Both methods should return the same order (sorted by priority)
        issues_ids = [str(i["id"]) for i in issues_result]
        assert ids_result == issues_ids
        # Verify priority ordering
        assert ids_result == ["task-2", "task-3", "task-1"]


class TestPipelineSteps:
    """Unit tests for pipeline step functions with fixture data."""

    def test_merge_wip_issues_adds_missing(self) -> None:
        """_merge_wip_issues adds WIP issues not in base list."""
        base = [{"id": "a"}, {"id": "b"}]
        wip = [{"id": "b"}, {"id": "c"}]
        result = BeadsClient._merge_wip_issues(base, wip)
        assert len(result) == 3
        assert [r["id"] for r in result] == ["a", "b", "c"]

    def test_merge_wip_issues_empty_base(self) -> None:
        """_merge_wip_issues with empty base returns all WIP."""
        wip = [{"id": "x"}, {"id": "y"}]
        result = BeadsClient._merge_wip_issues([], wip)
        assert [r["id"] for r in result] == ["x", "y"]

    def test_merge_wip_issues_empty_wip(self) -> None:
        """_merge_wip_issues with empty WIP returns base unchanged."""
        base = [{"id": "a"}]
        result = BeadsClient._merge_wip_issues(base, [])
        assert result == base

    def test_apply_filters_excludes_ids(self) -> None:
        """_apply_filters excludes specified IDs."""
        issues = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        result = BeadsClient._apply_filters(issues, {"b"}, None, None)
        assert [r["id"] for r in result] == ["a", "c"]

    def test_apply_filters_excludes_epics(self) -> None:
        """_apply_filters excludes issue_type=epic."""
        issues = [{"id": "a"}, {"id": "b", "issue_type": "epic"}]
        result = BeadsClient._apply_filters(issues, set(), None, None)
        assert [r["id"] for r in result] == ["a"]

    def test_apply_filters_by_epic_children(self) -> None:
        """_apply_filters includes only epic children when specified."""
        issues = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        result = BeadsClient._apply_filters(issues, set(), {"a", "c"}, None)
        assert [r["id"] for r in result] == ["a", "c"]

    def test_apply_filters_by_only_ids(self) -> None:
        """_apply_filters includes only specified IDs."""
        issues = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        result = BeadsClient._apply_filters(issues, set(), None, ["b"])
        assert [r["id"] for r in result] == ["b"]

    def test_sort_issues_by_priority(self, tmp_path: Path) -> None:
        """_sort_issues sorts by priority when focus=False."""
        beads = BeadsClient(tmp_path)
        issues = [
            {"id": "a", "priority": 3},
            {"id": "b", "priority": 1},
            {"id": "c", "priority": 2},
        ]
        result = beads._sort_issues(issues, focus=False, include_wip=False)
        assert [r["id"] for r in result] == ["b", "c", "a"]

    def test_sort_issues_include_wip(self, tmp_path: Path) -> None:
        """_sort_issues does not reorder based on include_wip."""
        beads = BeadsClient(tmp_path)
        issues = [
            {"id": "a", "priority": 1, "status": "open"},
            {"id": "b", "priority": 2, "status": "in_progress"},
            {"id": "c", "priority": 3, "status": "open"},
        ]
        result = beads._sort_issues(issues, focus=False, include_wip=True)
        assert [r["id"] for r in result] == ["a", "b", "c"]

    def test_sort_issues_by_epic_groups(self, tmp_path: Path) -> None:
        """_sort_issues groups by parent_epic when focus=True."""
        beads = BeadsClient(tmp_path)
        issues = [
            {
                "id": "a",
                "priority": 2,
                "parent_epic": "epic-1",
                "updated_at": "2025-01-01",
            },
            {
                "id": "b",
                "priority": 1,
                "parent_epic": "epic-2",
                "updated_at": "2025-01-01",
            },
            {
                "id": "c",
                "priority": 3,
                "parent_epic": "epic-1",
                "updated_at": "2025-01-01",
            },
        ]
        result = beads._sort_issues(issues, focus=True, include_wip=False)
        # epic-2 has lower min priority (1), so it comes first
        assert result[0]["id"] == "b"
        # epic-1 issues grouped together
        epic1_issues = [r for r in result if r["parent_epic"] == "epic-1"]
        assert len(epic1_issues) == 2


class TestWipFallbackOnReadyFailure:
    """Test that WIP issues are still fetched when bd ready fails."""

    @pytest.mark.asyncio
    async def test_returns_wip_issues_when_ready_fails(self, tmp_path: Path) -> None:
        """When bd ready fails but include_wip=True, should still return WIP issues."""
        beads = BeadsClient(tmp_path)

        async def mock_run_async(cmd: list[str]) -> CommandResult:
            if cmd == ["br", "ready", "--json", "-t", "task", "--limit", "0"]:
                # Simulate bd ready failure
                return make_command_result(returncode=1, stderr="bd ready failed")
            if cmd == [
                "br",
                "list",
                "--status",
                "in_progress",
                "--json",
                "-t",
                "task",
                "--limit",
                "0",
            ]:
                # Return WIP issues
                return make_command_result(
                    stdout=json.dumps([{"id": "wip-1", "priority": 1}])
                )
            # Return empty for other commands (like dep tree)
            return make_command_result(stdout="[]")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads, "_run_subprocess_async", AsyncMock(side_effect=mock_run_async)
            )
            result = await beads.get_ready_async(include_wip=True)

        assert result == ["wip-1"]

    @pytest.mark.asyncio
    async def test_returns_empty_when_ready_fails_and_not_wip(
        self, tmp_path: Path
    ) -> None:
        """When bd ready fails and include_wip=False, should return empty list."""
        beads = BeadsClient(tmp_path)

        async def mock_run_async(cmd: list[str]) -> CommandResult:
            if cmd == ["br", "ready", "--json", "-t", "task", "--limit", "0"]:
                # Simulate bd ready failure
                return make_command_result(returncode=1, stderr="bd ready failed")
            return make_command_result(stdout="[]")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads, "_run_subprocess_async", AsyncMock(side_effect=mock_run_async)
            )
            result = await beads.get_ready_async(include_wip=False)

        assert result == []


class TestFilterWipFromReadyOutput:
    """Test that in_progress issues from bd ready are filtered when include_wip=False."""

    @pytest.mark.asyncio
    async def test_filters_in_progress_when_include_wip_false(
        self, tmp_path: Path
    ) -> None:
        """When include_wip=False, in_progress issues from bd ready are filtered out."""
        beads = BeadsClient(tmp_path)

        async def mock_run_async(cmd: list[str]) -> CommandResult:
            if cmd == ["br", "ready", "--json", "-t", "task", "--limit", "0"]:
                # bd ready returns both open and in_progress issues
                return make_command_result(
                    stdout=json.dumps(
                        [
                            {"id": "open-1", "priority": 1, "status": "open"},
                            {"id": "wip-1", "priority": 2, "status": "in_progress"},
                            {"id": "open-2", "priority": 3, "status": "open"},
                        ]
                    )
                )
            # Return empty for dep tree lookup
            return make_command_result(stdout="[]")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads, "_run_subprocess_async", AsyncMock(side_effect=mock_run_async)
            )
            result = await beads.get_ready_async(include_wip=False)

        # in_progress issue should be filtered out
        assert "wip-1" not in result
        assert result == ["open-1", "open-2"]

    @pytest.mark.asyncio
    async def test_keeps_in_progress_when_include_wip_true(
        self, tmp_path: Path
    ) -> None:
        """When include_wip=True, in_progress issues from bd ready are kept."""
        beads = BeadsClient(tmp_path)

        async def mock_run_async(cmd: list[str]) -> CommandResult:
            if cmd == ["br", "ready", "--json", "-t", "task", "--limit", "0"]:
                # bd ready returns both open and in_progress issues
                return make_command_result(
                    stdout=json.dumps(
                        [
                            {"id": "open-1", "priority": 1, "status": "open"},
                            {"id": "wip-1", "priority": 2, "status": "in_progress"},
                        ]
                    )
                )
            if cmd == [
                "br",
                "list",
                "--status",
                "in_progress",
                "--json",
                "-t",
                "task",
                "--limit",
                "0",
            ]:
                # WIP fetch returns the same in_progress issue (will be deduplicated)
                return make_command_result(
                    stdout=json.dumps(
                        [
                            {"id": "wip-1", "priority": 2, "status": "in_progress"},
                        ]
                    )
                )
            # Return empty for dep tree lookup
            return make_command_result(stdout="[]")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads, "_run_subprocess_async", AsyncMock(side_effect=mock_run_async)
            )
            result = await beads.get_ready_async(include_wip=True)

        # in_progress issue should be kept
        assert "wip-1" in result
        assert result == ["open-1", "wip-1"]


class TestFetchBaseIssues:
    """Unit tests for _fetch_base_issues pipeline step."""

    @pytest.mark.asyncio
    async def test_returns_issues_on_success(self, tmp_path: Path) -> None:
        """_fetch_base_issues returns issues and True on success."""
        beads = BeadsClient(tmp_path)
        issues_json = json.dumps(
            [{"id": "a", "priority": 1}, {"id": "b", "priority": 2}]
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=issues_json)),
            )
            issues, ok = await beads._fetch_base_issues()

        assert ok is True
        assert len(issues) == 2
        assert issues[0]["id"] == "a"

    @pytest.mark.asyncio
    async def test_returns_empty_on_failure(self, tmp_path: Path) -> None:
        """_fetch_base_issues returns empty list and False on bd failure."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(
                    return_value=make_command_result(returncode=1, stderr="error")
                ),
            )
            issues, ok = await beads._fetch_base_issues()

        assert ok is False
        assert issues == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_invalid_json(self, tmp_path: Path) -> None:
        """_fetch_base_issues returns empty list and False on invalid JSON."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout="not json")),
            )
            issues, ok = await beads._fetch_base_issues()

        assert ok is False
        assert issues == []


class TestEnrichWithEpics:
    """Unit tests for _enrich_with_epics pipeline step."""

    @pytest.mark.asyncio
    async def test_adds_parent_epic_info(self, tmp_path: Path) -> None:
        """_enrich_with_epics adds parent_epic field to issues."""
        beads = BeadsClient(tmp_path)
        issues: list[dict[str, object]] = [{"id": "a"}, {"id": "b"}]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "get_parent_epics_async",
                AsyncMock(return_value={"a": "epic-1", "b": None}),
            )
            mp.setattr(beads, "_get_blocked_epics_async", AsyncMock(return_value=set()))
            result = await beads._enrich_with_epics(issues)

        assert result[0]["parent_epic"] == "epic-1"
        assert result[1]["parent_epic"] is None

    @pytest.mark.asyncio
    async def test_filters_blocked_epics(self, tmp_path: Path) -> None:
        """_enrich_with_epics filters issues whose parent epic is blocked."""
        beads = BeadsClient(tmp_path)
        issues: list[dict[str, object]] = [{"id": "a"}, {"id": "b"}]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "get_parent_epics_async",
                AsyncMock(return_value={"a": "epic-1", "b": "epic-2"}),
            )
            mp.setattr(
                beads, "_get_blocked_epics_async", AsyncMock(return_value={"epic-2"})
            )
            result = await beads._enrich_with_epics(issues)

        assert len(result) == 1
        assert result[0]["id"] == "a"

    @pytest.mark.asyncio
    async def test_returns_empty_list_unchanged(self, tmp_path: Path) -> None:
        """_enrich_with_epics returns empty list as-is."""
        beads = BeadsClient(tmp_path)
        result = await beads._enrich_with_epics([])
        assert result == []


class TestWarnMissingIds:
    """Unit tests for _warn_missing_ids helper."""

    def test_logs_warning_for_missing_ids(self, tmp_path: Path) -> None:
        """_warn_missing_ids logs warning when only_ids not found in issues."""
        beads = BeadsClient(tmp_path)
        warnings: list[str] = []
        beads._log_warning = lambda msg: warnings.append(msg)  # type: ignore[method-assign]

        issues: list[dict[str, object]] = [{"id": "a"}, {"id": "b"}]
        beads._warn_missing_ids(["a", "c", "d"], issues, set())

        assert len(warnings) == 1
        assert "c" in warnings[0] and "d" in warnings[0]

    def test_respects_suppress_ids(self, tmp_path: Path) -> None:
        """_warn_missing_ids does not warn for suppressed IDs."""
        beads = BeadsClient(tmp_path)
        warnings: list[str] = []
        beads._log_warning = lambda msg: warnings.append(msg)  # type: ignore[method-assign]

        issues: list[dict[str, object]] = [{"id": "a"}]
        beads._warn_missing_ids(["a", "b", "c"], issues, {"b"})

        assert len(warnings) == 1
        assert "b" not in warnings[0]
        assert "c" in warnings[0]

    def test_no_warning_when_all_ids_found(self, tmp_path: Path) -> None:
        """_warn_missing_ids does not log when all IDs are found."""
        beads = BeadsClient(tmp_path)
        warnings: list[str] = []
        beads._log_warning = lambda msg: warnings.append(msg)  # type: ignore[method-assign]

        issues: list[dict[str, object]] = [{"id": "a"}, {"id": "b"}]
        beads._warn_missing_ids(["a", "b"], issues, set())

        assert warnings == []

    def test_no_warning_when_only_ids_none(self, tmp_path: Path) -> None:
        """_warn_missing_ids does nothing when only_ids is None."""
        beads = BeadsClient(tmp_path)
        warnings: list[str] = []
        beads._log_warning = lambda msg: warnings.append(msg)  # type: ignore[method-assign]

        beads._warn_missing_ids(None, [], set())

        assert warnings == []


class TestResolveEpicChildren:
    """Unit tests for _resolve_epic_children helper."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_epic_id(self, tmp_path: Path) -> None:
        """_resolve_epic_children returns None when epic_id is None."""
        beads = BeadsClient(tmp_path)
        result = await beads._resolve_epic_children(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_children_when_found(self, tmp_path: Path) -> None:
        """_resolve_epic_children returns children set when epic has children."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "get_epic_children_async",
                AsyncMock(return_value={"child-1", "child-2"}),
            )
            result = await beads._resolve_epic_children("epic-1")

        assert result == {"child-1", "child-2"}

    @pytest.mark.asyncio
    async def test_logs_warning_and_returns_empty_when_no_children(
        self, tmp_path: Path
    ) -> None:
        """_resolve_epic_children logs warning when epic has no children."""
        beads = BeadsClient(tmp_path)
        warnings: list[str] = []
        beads._log_warning = lambda msg: warnings.append(msg)  # type: ignore[method-assign]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "get_epic_children_async", AsyncMock(return_value=set()))
            result = await beads._resolve_epic_children("epic-1")

        assert result == set()
        assert len(warnings) == 1
        assert "epic-1" in warnings[0]


class TestOrphansOnlyIntegration:
    """Integration tests for orphans_only filter in BeadsClient.

    Verifies that orphans_only=True is properly passed through and applied
    in get_ready_async() and get_ready_issues_async().
    """

    @pytest.mark.asyncio
    async def test_get_ready_async_filters_to_orphans_only(
        self, tmp_path: Path
    ) -> None:
        """get_ready_async with orphans_only=True should return only orphan tasks."""
        beads = BeadsClient(tmp_path)

        # Ready response with mixed orphan and non-orphan tasks
        ready_response = json.dumps(
            [
                {"id": "orphan-1", "issue_type": "task", "priority": 1},
                {"id": "parented-1", "issue_type": "task", "priority": 2},
                {"id": "orphan-2", "issue_type": "task", "priority": 3},
                {"id": "parented-2", "issue_type": "task", "priority": 1},
            ]
        )

        # Parent epic mappings - orphans have None, parented have epic
        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {
                "orphan-1": None,
                "parented-1": "epic-a",
                "orphan-2": None,
                "parented-2": "epic-b",
            }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)

            result = await beads.get_ready_async(orphans_only=True)

        # Should only include orphan tasks
        assert "orphan-1" in result
        assert "orphan-2" in result
        assert "parented-1" not in result
        assert "parented-2" not in result
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_ready_issues_async_filters_to_orphans_only(
        self, tmp_path: Path
    ) -> None:
        """get_ready_issues_async with orphans_only=True should return only orphan tasks."""
        beads = BeadsClient(tmp_path)

        ready_response = json.dumps(
            [
                {"id": "orphan-1", "issue_type": "task", "priority": 1},
                {"id": "parented-1", "issue_type": "task", "priority": 2},
                {"id": "orphan-2", "issue_type": "task", "priority": 3},
            ]
        )

        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {
                "orphan-1": None,
                "parented-1": "epic-a",
                "orphan-2": None,
            }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)

            result = await beads.get_ready_issues_async(orphans_only=True)

        # Should only include orphan tasks with full issue dicts
        result_ids = [str(i["id"]) for i in result]
        assert "orphan-1" in result_ids
        assert "orphan-2" in result_ids
        assert "parented-1" not in result_ids
        assert len(result) == 2
        # Verify issues have parent_epic field populated (as None for orphans)
        for issue in result:
            assert issue.get("parent_epic") is None

    @pytest.mark.asyncio
    async def test_orphans_only_false_returns_all_tasks(self, tmp_path: Path) -> None:
        """get_ready_async with orphans_only=False should return all tasks."""
        beads = BeadsClient(tmp_path)

        ready_response = json.dumps(
            [
                {"id": "orphan-1", "issue_type": "task", "priority": 1},
                {"id": "parented-1", "issue_type": "task", "priority": 2},
            ]
        )

        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {
                "orphan-1": None,
                "parented-1": "epic-a",
            }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)

            result = await beads.get_ready_async(orphans_only=False)

        # Should include both orphan and parented tasks
        assert "orphan-1" in result
        assert "parented-1" in result
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_orphans_only_returns_empty_when_all_have_parents(
        self, tmp_path: Path
    ) -> None:
        """get_ready_async with orphans_only=True should return empty if all have parents."""
        beads = BeadsClient(tmp_path)

        ready_response = json.dumps(
            [
                {"id": "parented-1", "issue_type": "task", "priority": 1},
                {"id": "parented-2", "issue_type": "task", "priority": 2},
            ]
        )

        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {
                "parented-1": "epic-a",
                "parented-2": "epic-b",
            }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)

            result = await beads.get_ready_async(orphans_only=True)

        assert result == []

    @pytest.mark.asyncio
    async def test_orphans_only_combined_with_include_wip(self, tmp_path: Path) -> None:
        """orphans_only should work correctly with include_wip=True."""
        beads = BeadsClient(tmp_path)

        ready_response = json.dumps(
            [
                {
                    "id": "orphan-1",
                    "issue_type": "task",
                    "priority": 2,
                    "status": "open",
                },
                {
                    "id": "parented-1",
                    "issue_type": "task",
                    "priority": 1,
                    "status": "open",
                },
            ]
        )

        wip_response = json.dumps(
            [
                {
                    "id": "wip-orphan",
                    "issue_type": "task",
                    "priority": 3,
                    "status": "in_progress",
                },
                {
                    "id": "wip-parented",
                    "issue_type": "task",
                    "priority": 1,
                    "status": "in_progress",
                },
            ]
        )

        async def mock_run(cmd: list[str]) -> CommandResult:
            if "--status" in cmd and "in_progress" in cmd:
                return make_command_result(stdout=wip_response)
            return make_command_result(stdout=ready_response)

        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {
                "orphan-1": None,
                "parented-1": "epic-a",
                "wip-orphan": None,
                "wip-parented": "epic-b",
            }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", mock_run)
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)

            result = await beads.get_ready_async(
                orphans_only=True, include_wip=True, focus=False
            )

        # Should only include orphan tasks
        assert "wip-orphan" in result
        assert "orphan-1" in result
        assert "parented-1" not in result
        assert "wip-parented" not in result
        assert len(result) == 2
        assert result == ["orphan-1", "wip-orphan"]

    @pytest.mark.asyncio
    async def test_orphans_only_treats_empty_string_parent_as_orphan(
        self, tmp_path: Path
    ) -> None:
        """Tasks with empty string parent_epic should be treated as orphans."""
        beads = BeadsClient(tmp_path)

        ready_response = json.dumps(
            [
                {"id": "empty-parent", "issue_type": "task", "priority": 1},
                {"id": "real-parent", "issue_type": "task", "priority": 2},
            ]
        )

        async def mock_get_parent_epics(
            issue_ids: list[str],
        ) -> dict[str, str | None]:
            return {
                "empty-parent": "",  # Empty string treated as no parent
                "real-parent": "epic-a",
            }

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=ready_response)),
            )
            mp.setattr(beads, "get_parent_epics_async", mock_get_parent_epics)

            result = await beads.get_ready_async(orphans_only=True)

        # Empty string parent should be treated as orphan
        assert "empty-parent" in result
        assert "real-parent" not in result
        assert len(result) == 1


class TestGetBlockedCountAsync:
    """Test get_blocked_count_async method."""

    @pytest.mark.asyncio
    async def test_returns_count_of_blocked_issues(self, tmp_path: Path) -> None:
        """Should return the count of blocked issues."""
        beads = BeadsClient(tmp_path)
        blocked_response = json.dumps(
            [
                {"id": "blocked-1", "status": "blocked"},
                {"id": "blocked-2", "status": "blocked"},
                {"id": "blocked-3", "status": "blocked"},
            ]
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=blocked_response)),
            )
            result = await beads.get_blocked_count_async()

        assert result == 3

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_blocked_issues(self, tmp_path: Path) -> None:
        """Should return 0 when no issues are blocked."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout="[]")),
            )
            result = await beads.get_blocked_count_async()

        assert result == 0

    @pytest.mark.asyncio
    async def test_returns_none_on_bd_failure(self, tmp_path: Path) -> None:
        """Should return None when bd list fails."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(
                    return_value=make_command_result(returncode=1, stderr="bd error")
                ),
            )
            result = await beads.get_blocked_count_async()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        """Should return None when bd returns invalid JSON."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout="not valid json")),
            )
            result = await beads.get_blocked_count_async()

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_non_list_response(self, tmp_path: Path) -> None:
        """Should return None when bd returns non-list JSON."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout='{"error": "bad"}')),
            )
            result = await beads.get_blocked_count_async()

        assert result is None

    @pytest.mark.asyncio
    async def test_calls_bd_list_with_correct_args(self, tmp_path: Path) -> None:
        """Should call bd list with --status blocked --json -t task."""
        beads = BeadsClient(tmp_path)
        captured_cmds: list[list[str]] = []

        async def capturing_run(cmd: list[str]) -> CommandResult:
            captured_cmds.append(cmd)
            return make_command_result(stdout="[]")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", capturing_run)
            await beads.get_blocked_count_async()

        assert len(captured_cmds) == 1
        assert captured_cmds[0] == [
            "br",
            "list",
            "--status",
            "blocked",
            "--json",
            "-t",
            "task",
        ]


class TestClaimAsync:
    """Test claim_async method."""

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self, tmp_path: Path) -> None:
        """Should return True when claim succeeds."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(returncode=0)),
            )
            result = await beads.claim_async("issue-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(self, tmp_path: Path) -> None:
        """Should return False when claim fails."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(returncode=1)),
            )
            result = await beads.claim_async("issue-1")

        assert result is False


class TestResetAsync:
    """Test reset_async method."""

    @pytest.mark.asyncio
    async def test_resets_issue_to_ready(self, tmp_path: Path) -> None:
        """Should call bd update with ready status."""
        beads = BeadsClient(tmp_path)
        captured_cmds: list[list[str]] = []

        async def capturing_run(cmd: list[str]) -> CommandResult:
            captured_cmds.append(cmd)
            return make_command_result()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", capturing_run)
            await beads.reset_async("issue-1")

        assert captured_cmds[0][0:4] == ["br", "update", "issue-1", "--status"]
        assert "ready" in captured_cmds[0]

    @pytest.mark.asyncio
    async def test_includes_error_and_log_path_in_notes(self, tmp_path: Path) -> None:
        """Should include error and log path in notes."""
        beads = BeadsClient(tmp_path)
        captured_cmds: list[list[str]] = []
        log_path = tmp_path / "agent.log"

        async def capturing_run(cmd: list[str]) -> CommandResult:
            captured_cmds.append(cmd)
            return make_command_result()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", capturing_run)
            await beads.reset_async("issue-1", log_path=log_path, error="Test error")

        cmd = captured_cmds[0]
        assert "--notes" in cmd
        notes_idx = cmd.index("--notes")
        notes = cmd[notes_idx + 1]
        assert "Failed: Test error" in notes
        assert str(log_path) in notes


class TestGetIssueStatusAsync:
    """Test get_issue_status_async method."""

    @pytest.mark.asyncio
    async def test_returns_status_from_dict(self, tmp_path: Path) -> None:
        """Should return status when bd show returns dict."""
        beads = BeadsClient(tmp_path)
        issue_json = json.dumps({"id": "issue-1", "status": "in_progress"})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=issue_json)),
            )
            result = await beads.get_issue_status_async("issue-1")

        assert result == "in_progress"

    @pytest.mark.asyncio
    async def test_returns_status_from_list(self, tmp_path: Path) -> None:
        """Should return status when bd show returns list."""
        beads = BeadsClient(tmp_path)
        issue_json = json.dumps([{"id": "issue-1", "status": "open"}])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=issue_json)),
            )
            result = await beads.get_issue_status_async("issue-1")

        assert result == "open"

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self, tmp_path: Path) -> None:
        """Should return None when bd show fails."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(returncode=1)),
            )
            result = await beads.get_issue_status_async("issue-1")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        """Should return None when bd returns invalid JSON."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout="not json")),
            )
            result = await beads.get_issue_status_async("issue-1")

        assert result is None


class TestGetIssueDescriptionAsync:
    """Test get_issue_description_async method."""

    @pytest.mark.asyncio
    async def test_returns_description_with_title(self, tmp_path: Path) -> None:
        """Should return description including title."""
        beads = BeadsClient(tmp_path)
        issue_json = json.dumps(
            {"id": "issue-1", "title": "Fix bug", "description": "Details here"}
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=issue_json)),
            )
            result = await beads.get_issue_description_async("issue-1")

        assert result is not None
        assert "Title: Fix bug" in result
        assert "Details here" in result

    @pytest.mark.asyncio
    async def test_includes_acceptance_criteria(self, tmp_path: Path) -> None:
        """Should include acceptance criteria in description."""
        beads = BeadsClient(tmp_path)
        issue_json = json.dumps(
            {
                "id": "issue-1",
                "title": "Task",
                "acceptance": "- Tests pass\n- Code reviewed",
            }
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=issue_json)),
            )
            result = await beads.get_issue_description_async("issue-1")

        assert result is not None
        assert "Acceptance Criteria:" in result
        assert "Tests pass" in result

    @pytest.mark.asyncio
    async def test_returns_none_when_no_fields(self, tmp_path: Path) -> None:
        """Should return None when issue has no description fields."""
        beads = BeadsClient(tmp_path)
        issue_json = json.dumps({"id": "issue-1"})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=issue_json)),
            )
            result = await beads.get_issue_description_async("issue-1")

        assert result is None


class TestCloseAsync:
    """Test close_async method."""

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self, tmp_path: Path) -> None:
        """Should return True when close succeeds."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(returncode=0)),
            )
            result = await beads.close_async("issue-1")

        assert result is True


class TestReopenIssueAsync:
    """Test reopen_issue_async method."""

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self, tmp_path: Path) -> None:
        """Should return True when reopen succeeds."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(returncode=0)),
            )
            result = await beads.reopen_issue_async("issue-1")

        assert result is True


class TestAddDependencyAsync:
    """Test add_dependency_async method."""

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self, tmp_path: Path) -> None:
        """Should return True when adding dependency succeeds."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(returncode=0)),
            )
            result = await beads.add_dependency_async("issue-1", "blocker-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(self, tmp_path: Path) -> None:
        """Should return False when adding dependency fails."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(returncode=1)),
            )
            result = await beads.add_dependency_async("issue-1", "blocker-1")

        assert result is False


class TestCreateIssueAsync:
    """Test create_issue_async method."""

    @pytest.mark.asyncio
    async def test_returns_issue_id_from_stdout(self, tmp_path: Path) -> None:
        """Should return issue ID parsed from stdout."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(
                    return_value=make_command_result(stdout="Created issue: abc-123")
                ),
            )
            result = await beads.create_issue_async(
                title="Test", description="Desc", priority="P2"
            )

        assert result == "abc-123"

    @pytest.mark.asyncio
    async def test_returns_issue_id_from_json(self, tmp_path: Path) -> None:
        """Should return issue ID from JSON response."""
        beads = BeadsClient(tmp_path)
        response = json.dumps({"id": "xyz-456"})

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=response)),
            )
            result = await beads.create_issue_async(
                title="Test", description="Desc", priority="P2"
            )

        assert result == "xyz-456"

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self, tmp_path: Path) -> None:
        """Should return None when create fails."""
        beads = BeadsClient(tmp_path)
        warnings: list[str] = []
        beads._log_warning = lambda msg: warnings.append(msg)  # type: ignore[method-assign]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(
                    return_value=make_command_result(returncode=1, stderr="error")
                ),
            )
            result = await beads.create_issue_async(
                title="Test", description="Desc", priority="P2"
            )

        assert result is None
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_includes_parent_and_tags(self, tmp_path: Path) -> None:
        """Should include parent and tags in command."""
        beads = BeadsClient(tmp_path)
        captured_cmds: list[list[str]] = []

        async def capturing_run(cmd: list[str]) -> CommandResult:
            captured_cmds.append(cmd)
            return make_command_result(stdout="Created issue: new-1")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", capturing_run)
            await beads.create_issue_async(
                title="Test",
                description="Desc",
                priority="P1",
                parent_id="epic-1",
                tags=["bug", "urgent"],
            )

        cmd = captured_cmds[0]
        assert "--parent" in cmd
        assert "epic-1" in cmd
        assert "--labels" in cmd
        assert "bug,urgent" in cmd


class TestFindIssueByTagAsync:
    """Test find_issue_by_tag_async method."""

    @pytest.mark.asyncio
    async def test_returns_first_matching_issue(self, tmp_path: Path) -> None:
        """Should return first issue ID from list."""
        beads = BeadsClient(tmp_path)
        issues = json.dumps([{"id": "issue-1"}, {"id": "issue-2"}])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=issues)),
            )
            result = await beads.find_issue_by_tag_async("my-tag")

        assert result == "issue-1"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_matches(self, tmp_path: Path) -> None:
        """Should return None when no issues match."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout="[]")),
            )
            result = await beads.find_issue_by_tag_async("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self, tmp_path: Path) -> None:
        """Should return None when bd list fails."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(returncode=1)),
            )
            result = await beads.find_issue_by_tag_async("my-tag")

        assert result is None


class TestUpdateIssueDescriptionAsync:
    """Test update_issue_description_async method."""

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self, tmp_path: Path) -> None:
        """Should return True when update succeeds."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(returncode=0)),
            )
            result = await beads.update_issue_description_async(
                "issue-1", "New description"
            )

        assert result is True


class TestUpdateIssueAsync:
    """Test update_issue_async method."""

    @pytest.mark.asyncio
    async def test_returns_true_on_no_changes(self, tmp_path: Path) -> None:
        """Should return True when no changes requested."""
        beads = BeadsClient(tmp_path)

        result = await beads.update_issue_async("issue-1")

        assert result is True

    @pytest.mark.asyncio
    async def test_updates_title_and_priority(self, tmp_path: Path) -> None:
        """Should update title and priority."""
        beads = BeadsClient(tmp_path)
        captured_cmds: list[list[str]] = []

        async def capturing_run(cmd: list[str]) -> CommandResult:
            captured_cmds.append(cmd)
            return make_command_result()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(beads, "_run_subprocess_async", capturing_run)
            await beads.update_issue_async("issue-1", title="New Title", priority="P1")

        cmd = captured_cmds[0]
        assert "--title" in cmd
        assert "New Title" in cmd
        assert "--priority" in cmd
        assert "P1" in cmd


class TestRunSubprocessAsyncTimeout:
    """Test _run_subprocess_async timeout handling."""

    @pytest.mark.asyncio
    async def test_logs_warning_on_timeout(self, tmp_path: Path) -> None:
        """Should log warning when command times out."""
        from unittest.mock import MagicMock

        from src.infra.tools.command_runner import CommandRunner

        warnings: list[str] = []
        beads = BeadsClient(tmp_path, log_warning=warnings.append)

        # Create a mock runner that returns a timed out result
        mock_runner = MagicMock(spec=CommandRunner)
        mock_runner.run_async = AsyncMock(
            return_value=CommandResult(
                command=["br", "test"],
                returncode=1,
                stdout="",
                stderr="",
                timed_out=True,
                duration_seconds=30.0,
            )
        )
        beads._runner = mock_runner

        result = await beads._run_subprocess_async(["br", "test"], timeout=30.0)

        assert result.timed_out is True
        assert len(warnings) == 1
        assert "timed out" in warnings[0]


class TestGetEpicChildrenAsync:
    """Test get_epic_children_async method."""

    @pytest.mark.asyncio
    async def test_returns_child_ids(self, tmp_path: Path) -> None:
        """Should return child IDs from bd list --parent response."""
        beads = BeadsClient(tmp_path)
        # bd list --parent returns only children, not the parent epic itself
        children = json.dumps(
            [
                {"id": "task-1"},
                {"id": "task-2"},
            ]
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=children)),
            )
            result = await beads.get_epic_children_async("epic-1")

        assert result == {"task-1", "task-2"}

    @pytest.mark.asyncio
    async def test_returns_empty_set_on_failure(self, tmp_path: Path) -> None:
        """Should return empty set when bd list --parent fails."""
        beads = BeadsClient(tmp_path)
        warnings: list[str] = []
        beads._log_warning = lambda msg: warnings.append(msg)  # type: ignore[method-assign]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(returncode=1)),
            )
            result = await beads.get_epic_children_async("epic-1")

        assert result == set()
        assert len(warnings) == 1


class TestFetchWipIssuesAsync:
    """Test fetch_wip_issues_async method."""

    @pytest.mark.asyncio
    async def test_returns_issues_on_success(self, tmp_path: Path) -> None:
        """Should return WIP issues list."""
        beads = BeadsClient(tmp_path)
        issues = json.dumps([{"id": "wip-1"}, {"id": "wip-2"}])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(stdout=issues)),
            )
            result = await beads.fetch_wip_issues_async()

        assert len(result) == 2
        assert result[0]["id"] == "wip-1"

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_failure(self, tmp_path: Path) -> None:
        """Should return empty list when bd list fails."""
        beads = BeadsClient(tmp_path)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                beads,
                "_run_subprocess_async",
                AsyncMock(return_value=make_command_result(returncode=1)),
            )
            result = await beads.fetch_wip_issues_async()

        assert result == []
