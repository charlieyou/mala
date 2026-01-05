"""Unit tests for --wip flag prioritization logic and focus mode.

Tests verify IssueManager.sort_issues which is the production logic for
issue prioritization. FakeIssueProvider uses IssueManager internally,
so testing IssueManager directly ensures production code is verified.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.orchestration.orchestrator import MalaOrchestrator

from src.infra.issue_manager import IssueManager


class TestPrioritizeWipFlag:
    """Test --wip flag prioritization via IssueManager.sort_issues.

    These tests verify the production sort_issues logic directly, ensuring
    that prioritize_wip correctly puts in_progress issues before open ones.
    """

    def test_prioritize_wip_sorts_in_progress_first(self) -> None:
        """When prioritize_wip=True, in_progress issues should come before open."""
        issues = [
            {"id": "open-1", "priority": 1, "status": "open"},
            {"id": "wip-1", "priority": 2, "status": "in_progress"},
            {"id": "open-2", "priority": 3, "status": "open"},
        ]

        result = IssueManager.sort_issues(issues, focus=False, prioritize_wip=True)

        # in_progress should come first, then open sorted by priority
        result_ids = [r["id"] for r in result]
        assert result_ids == ["wip-1", "open-1", "open-2"]

    def test_prioritize_wip_false_uses_priority_only(self) -> None:
        """When prioritize_wip=False, sort by priority only (WIP issues still sorted)."""
        issues = [
            {"id": "wip-1", "priority": 3, "status": "in_progress"},
            {"id": "open-1", "priority": 1, "status": "open"},
            {"id": "open-2", "priority": 2, "status": "open"},
        ]

        result = IssueManager.sort_issues(issues, focus=False, prioritize_wip=False)

        # Without prioritize_wip, sort purely by priority (all issues remain)
        result_ids = [r["id"] for r in result]
        assert result_ids == ["open-1", "open-2", "wip-1"]

    def test_prioritize_wip_respects_priority_within_status(self) -> None:
        """Within each status group, issues should still be sorted by priority."""
        issues = [
            {"id": "wip-low", "priority": 3, "status": "in_progress"},
            {"id": "wip-high", "priority": 1, "status": "in_progress"},
            {"id": "open-low", "priority": 2, "status": "open"},
            {"id": "open-high", "priority": 1, "status": "open"},
        ]

        result = IssueManager.sort_issues(issues, focus=False, prioritize_wip=True)

        # in_progress first (sorted by priority), then open (sorted by priority)
        result_ids = [r["id"] for r in result]
        assert result_ids == ["wip-high", "wip-low", "open-high", "open-low"]

    def test_prioritize_wip_false_preserves_all_issues(self) -> None:
        """Without prioritize_wip, all issues are sorted by priority regardless of status."""
        issues = [
            {"id": "wip-1", "priority": 2, "status": "in_progress"},
            {"id": "open-1", "priority": 1, "status": "open"},
        ]

        result = IssueManager.sort_issues(issues, focus=False, prioritize_wip=False)

        # Without prioritize_wip, just sort by priority - all issues included
        result_ids = [r["id"] for r in result]
        assert result_ids == ["open-1", "wip-1"]


class TestOrchestratorPrioritizeWip:
    """Test orchestrator stores and uses prioritize_wip flag."""

    def test_orchestrator_stores_prioritize_wip(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Orchestrator should store prioritize_wip parameter."""
        orch = make_orchestrator(repo_path=tmp_path, prioritize_wip=True)
        assert orch.prioritize_wip is True

    def test_orchestrator_default_prioritize_wip_is_false(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """Default prioritize_wip should be False."""
        orch = make_orchestrator(repo_path=tmp_path)
        assert orch.prioritize_wip is False


class TestFocusModeEpicGrouping:
    """Test focus mode epic-grouped ordering via IssueManager.

    These tests verify the production sort_by_epic_groups logic directly.
    """

    def test_focus_groups_tasks_by_epic(self) -> None:
        """When focus=True, tasks should be grouped by parent epic."""
        issues = [
            {
                "id": "task-a1",
                "priority": 2,
                "status": "open",
                "parent_epic": "epic-a",
                "updated_at": "2025-01-01T10:00:00Z",
            },
            {
                "id": "task-b1",
                "priority": 1,
                "status": "open",
                "parent_epic": "epic-b",
                "updated_at": "2025-01-01T10:00:00Z",
            },
            {
                "id": "task-a2",
                "priority": 1,
                "status": "open",
                "parent_epic": "epic-a",
                "updated_at": "2025-01-01T10:00:00Z",
            },
        ]

        result = IssueManager.sort_issues(issues, focus=True, prioritize_wip=False)

        # epic-a has min_priority=1 (task-a2), epic-b has min_priority=1 (task-b1)
        # Within epic-a: task-a2 (P1) before task-a1 (P2)
        result_ids = [r["id"] for r in result]
        assert result_ids == ["task-a2", "task-a1", "task-b1"]

    def test_focus_false_uses_priority_only(self) -> None:
        """When focus=False, tasks should be interleaved by priority."""
        issues = [
            {
                "id": "task-a1",
                "priority": 2,
                "status": "open",
                "parent_epic": "epic-a",
                "updated_at": "2025-01-01T10:00:00Z",
            },
            {
                "id": "task-b1",
                "priority": 1,
                "status": "open",
                "parent_epic": "epic-b",
                "updated_at": "2025-01-01T10:00:00Z",
            },
            {
                "id": "task-a2",
                "priority": 3,
                "status": "open",
                "parent_epic": "epic-a",
                "updated_at": "2025-01-01T10:00:00Z",
            },
        ]

        result = IssueManager.sort_issues(issues, focus=False, prioritize_wip=False)

        # Should sort by priority only: P1, P2, P3
        result_ids = [r["id"] for r in result]
        assert result_ids == ["task-b1", "task-a1", "task-a2"]

    def test_focus_orphan_tasks_form_virtual_group(self) -> None:
        """Orphan tasks (no parent epic) should form their own virtual group."""
        issues = [
            {
                "id": "orphan-1",
                "priority": 3,
                "status": "open",
                "parent_epic": None,
                "updated_at": "2025-01-01T10:00:00Z",
            },
            {
                "id": "task-a1",
                "priority": 2,
                "status": "open",
                "parent_epic": "epic-a",
                "updated_at": "2025-01-01T10:00:00Z",
            },
            {
                "id": "orphan-2",
                "priority": 1,
                "status": "open",
                "parent_epic": None,
                "updated_at": "2025-01-01T10:00:00Z",
            },
        ]

        result = IssueManager.sort_issues(issues, focus=True, prioritize_wip=False)

        # Orphan group has min_priority=1 (orphan-2), epic-a has min_priority=2
        # Orphan group comes first, sorted by priority: orphan-2 (P1), orphan-1 (P3)
        # Then epic-a group: task-a1 (P2)
        result_ids = [r["id"] for r in result]
        assert result_ids == ["orphan-2", "orphan-1", "task-a1"]

    def test_focus_tiebreaker_uses_updated_at(self) -> None:
        """Equal-priority groups should be ordered by most recently updated."""
        issues = [
            {
                "id": "task-a1",
                "priority": 1,
                "status": "open",
                "parent_epic": "epic-a",
                "updated_at": "2025-01-01T10:00:00Z",
            },
            {
                "id": "task-b1",
                "priority": 1,
                "status": "open",
                "parent_epic": "epic-b",
                "updated_at": "2025-01-02T10:00:00Z",
            },
        ]

        result = IssueManager.sort_issues(issues, focus=True, prioritize_wip=False)

        # Both epics have P1, but epic-b updated more recently (2025-01-02)
        # So epic-b should come first
        result_ids = [r["id"] for r in result]
        assert result_ids == ["task-b1", "task-a1"]

    def test_focus_with_prioritize_wip(self) -> None:
        """WIP prioritization should still move in_progress to front globally."""
        issues = [
            {
                "id": "task-a1",
                "priority": 1,
                "status": "open",
                "parent_epic": "epic-a",
                "updated_at": "2025-01-01T10:00:00Z",
            },
            {
                "id": "task-b1",
                "priority": 2,
                "status": "open",
                "parent_epic": "epic-b",
                "updated_at": "2025-01-01T10:00:00Z",
            },
            {
                "id": "task-b2",
                "priority": 3,
                "status": "in_progress",
                "parent_epic": "epic-b",
                "updated_at": "2025-01-01T10:00:00Z",
            },
        ]

        result = IssueManager.sort_issues(issues, focus=True, prioritize_wip=True)

        # WIP issue task-b2 should come first globally
        # Then remaining tasks grouped by epic:
        # epic-a (P1) before epic-b (P2 remaining after task-b2)
        result_ids = [r["id"] for r in result]
        assert result_ids[0] == "task-b2"
        # The rest should be epic-grouped: epic-a (task-a1) then epic-b (task-b1)
        assert result_ids == ["task-b2", "task-a1", "task-b1"]

    def test_focus_within_group_sorts_by_priority_then_updated(self) -> None:
        """Within an epic group, tasks should sort by priority then updated DESC."""
        issues = [
            {
                "id": "task-a1",
                "priority": 1,
                "status": "open",
                "parent_epic": "epic-a",
                "updated_at": "2025-01-01T10:00:00Z",
            },
            {
                "id": "task-a2",
                "priority": 1,
                "status": "open",
                "parent_epic": "epic-a",
                "updated_at": "2025-01-02T10:00:00Z",
            },
            {
                "id": "task-a3",
                "priority": 2,
                "status": "open",
                "parent_epic": "epic-a",
                "updated_at": "2025-01-03T10:00:00Z",
            },
        ]

        result = IssueManager.sort_issues(issues, focus=True, prioritize_wip=False)

        # All in same epic, sorted by (priority, updated DESC)
        # P1 tasks first: task-a2 (2025-01-02) before task-a1 (2025-01-01)
        # Then P2: task-a3
        result_ids = [r["id"] for r in result]
        assert result_ids == ["task-a2", "task-a1", "task-a3"]
