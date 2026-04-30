"""Unit tests for --resume scope inclusion and epic priority mode.

Tests verify IssueManager methods which are the production logic for
issue prioritization. FakeIssueProvider uses IssueManager internally,
so testing IssueManager directly ensures production code is verified.

This includes:
- merge_wip_issues: Merging WIP issues fetched separately from bd list
- filter_blocked_wip: Filtering out blocked in_progress issues
- sort_issues: Sorting behavior is unaffected by include_wip
"""

from __future__ import annotations


from src.core.models import OrderPreference
from src.infra.issue_manager import IssueManager


class TestMergeWipIssues:
    """Test IssueManager.merge_wip_issues - merging WIP into base issues.

    This tests the logic used when BeadsClient fetches in_progress issues
    separately via `bd list --status in_progress` and merges them with
    ready issues from `bd ready`.
    """

    def test_merges_wip_issues_into_base(self) -> None:
        """WIP issues should be appended to base list."""
        base = [{"id": "open-1", "status": "open"}]
        wip = [{"id": "wip-1", "status": "in_progress"}]

        result = IssueManager.merge_wip_issues(base, wip)

        assert len(result) == 2
        result_ids = [r["id"] for r in result]
        assert result_ids == ["open-1", "wip-1"]

    def test_avoids_duplicate_wip_issues(self) -> None:
        """WIP issues already in base should not be duplicated."""
        base = [
            {"id": "open-1", "status": "open"},
            {"id": "wip-1", "status": "in_progress"},
        ]
        wip = [{"id": "wip-1", "status": "in_progress"}]

        result = IssueManager.merge_wip_issues(base, wip)

        assert len(result) == 2
        result_ids = [r["id"] for r in result]
        assert result_ids == ["open-1", "wip-1"]

    def test_merges_multiple_wip_issues(self) -> None:
        """Multiple WIP issues should all be merged."""
        base = [{"id": "open-1", "status": "open"}]
        wip = [
            {"id": "wip-1", "status": "in_progress"},
            {"id": "wip-2", "status": "in_progress"},
        ]

        result = IssueManager.merge_wip_issues(base, wip)

        assert len(result) == 3
        result_ids = [r["id"] for r in result]
        assert result_ids == ["open-1", "wip-1", "wip-2"]

    def test_handles_empty_base(self) -> None:
        """WIP issues should work with empty base list."""
        base: list[dict[str, object]] = []
        wip = [{"id": "wip-1", "status": "in_progress"}]

        result = IssueManager.merge_wip_issues(base, wip)

        assert len(result) == 1
        assert result[0]["id"] == "wip-1"

    def test_handles_empty_wip(self) -> None:
        """Empty WIP list should return base unchanged."""
        base = [{"id": "open-1", "status": "open"}]
        wip: list[dict[str, object]] = []

        result = IssueManager.merge_wip_issues(base, wip)

        assert len(result) == 1
        assert result[0]["id"] == "open-1"


class TestFilterBlockedWip:
    """Test IssueManager.filter_blocked_wip - filtering blocked WIP issues.

    This tests the logic used to exclude in_progress issues that are blocked.
    Only in_progress issues with blocked_by are filtered; open issues are kept.
    """

    def test_filters_blocked_wip_issues(self) -> None:
        """Blocked in_progress issues should be filtered out."""
        issues = [
            {"id": "open-1", "status": "open"},
            {"id": "wip-blocked", "status": "in_progress", "blocked_by": "other-issue"},
            {"id": "wip-ready", "status": "in_progress"},
        ]

        result = IssueManager.filter_blocked_wip(issues)

        result_ids = [r["id"] for r in result]
        assert "wip-blocked" not in result_ids
        assert "wip-ready" in result_ids
        assert "open-1" in result_ids

    def test_keeps_open_issues_with_blocked_by(self) -> None:
        """Open issues with blocked_by should NOT be filtered (only WIP)."""
        issues = [
            {"id": "open-blocked", "status": "open", "blocked_by": "other-issue"},
            {"id": "wip-blocked", "status": "in_progress", "blocked_by": "other-issue"},
        ]

        result = IssueManager.filter_blocked_wip(issues)

        result_ids = [r["id"] for r in result]
        # Open issues are kept even if blocked (they're handled by bd ready)
        assert "open-blocked" in result_ids
        # But in_progress issues with blocked_by are filtered
        assert "wip-blocked" not in result_ids

    def test_handles_empty_blocked_by(self) -> None:
        """Issues with empty blocked_by should not be filtered."""
        issues = [
            {"id": "wip-1", "status": "in_progress", "blocked_by": ""},
            {"id": "wip-2", "status": "in_progress", "blocked_by": None},
        ]

        result = IssueManager.filter_blocked_wip(issues)

        # Empty/None blocked_by is falsy, so issues are kept
        assert len(result) == 2


class TestIncludeWipFlag:
    """Test --resume scope behavior via IssueManager.sort_issues.

    These tests verify the production sort_issues logic directly, ensuring
    that include_wip does not affect ordering.
    """

    def test_include_wip_does_not_reorder(self) -> None:
        """When include_wip=True, ordering should still follow priority."""
        issues = [
            {"id": "open-1", "priority": 1, "status": "open"},
            {"id": "wip-1", "priority": 2, "status": "in_progress"},
            {"id": "open-2", "priority": 3, "status": "open"},
        ]

        result = IssueManager.sort_issues(issues, focus=False, include_wip=True)

        # Ordering remains by priority, regardless of status
        result_ids = [r["id"] for r in result]
        assert result_ids == ["open-1", "wip-1", "open-2"]

    def test_include_wip_false_uses_priority_only(self) -> None:
        """When include_wip=False, sort by priority only (WIP issues still sorted)."""
        issues = [
            {"id": "wip-1", "priority": 3, "status": "in_progress"},
            {"id": "open-1", "priority": 1, "status": "open"},
            {"id": "open-2", "priority": 2, "status": "open"},
        ]

        result = IssueManager.sort_issues(issues, focus=False, include_wip=False)

        # Without include_wip, sort purely by priority (all issues remain)
        result_ids = [r["id"] for r in result]
        assert result_ids == ["open-1", "open-2", "wip-1"]

    def test_include_wip_true_matches_false(self) -> None:
        """include_wip should not change order compared to False."""
        issues = [
            {"id": "wip-low", "priority": 3, "status": "in_progress"},
            {"id": "wip-high", "priority": 1, "status": "in_progress"},
            {"id": "open-low", "priority": 2, "status": "open"},
            {"id": "open-high", "priority": 1, "status": "open"},
        ]

        result_true = IssueManager.sort_issues(issues, focus=False, include_wip=True)
        result_false = IssueManager.sort_issues(issues, focus=False, include_wip=False)

        assert [r["id"] for r in result_true] == [r["id"] for r in result_false]

    def test_include_wip_false_preserves_all_issues(self) -> None:
        """Without include_wip, all issues are sorted by priority regardless of status."""
        issues = [
            {"id": "wip-1", "priority": 2, "status": "in_progress"},
            {"id": "open-1", "priority": 1, "status": "open"},
        ]

        result = IssueManager.sort_issues(issues, focus=False, include_wip=False)

        # Without include_wip, just sort by priority - all issues included
        result_ids = [r["id"] for r in result]
        assert result_ids == ["open-1", "wip-1"]


class TestEpicPriorityModeGrouping:
    """Test epic priority mode (group by epic) ordering via IssueManager.

    These tests verify the production sort_by_epic_groups logic directly.
    """

    def test_epic_priority_groups_tasks_by_epic(self) -> None:
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

        result = IssueManager.sort_issues(issues, focus=True, include_wip=False)

        # epic-a has min_priority=1 (task-a2), epic-b has min_priority=1 (task-b1)
        # Within epic-a: task-a2 (P1) before task-a1 (P2)
        result_ids = [r["id"] for r in result]
        assert result_ids == ["task-a2", "task-a1", "task-b1"]

    def test_issue_priority_mode_uses_priority_only(self) -> None:
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

        result = IssueManager.sort_issues(issues, focus=False, include_wip=False)

        # Should sort by priority only: P1, P2, P3
        result_ids = [r["id"] for r in result]
        assert result_ids == ["task-b1", "task-a1", "task-a2"]

    def test_epic_priority_orphan_tasks_form_virtual_group(self) -> None:
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

        result = IssueManager.sort_issues(issues, focus=True, include_wip=False)

        # Orphan group has min_priority=1 (orphan-2), epic-a has min_priority=2
        # Orphan group comes first, sorted by priority: orphan-2 (P1), orphan-1 (P3)
        # Then epic-a group: task-a1 (P2)
        result_ids = [r["id"] for r in result]
        assert result_ids == ["orphan-2", "orphan-1", "task-a1"]

    def test_epic_priority_tiebreaker_uses_updated_at(self) -> None:
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

        result = IssueManager.sort_issues(issues, focus=True, include_wip=False)

        # Both epics have P1, but epic-b updated more recently (2025-01-02)
        # So epic-b should come first
        result_ids = [r["id"] for r in result]
        assert result_ids == ["task-b1", "task-a1"]

    def test_epic_priority_with_include_wip(self) -> None:
        """include_wip should not change epic-priority ordering."""
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

        result = IssueManager.sort_issues(issues, focus=True, include_wip=True)

        result_ids = [r["id"] for r in result]
        assert result_ids == ["task-a1", "task-b1", "task-b2"]

    def test_epic_priority_within_group_sorts_by_priority_then_updated(self) -> None:
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

        result = IssueManager.sort_issues(issues, focus=True, include_wip=False)

        # All in same epic, sorted by (priority, updated DESC)
        # P1 tasks first: task-a2 (2025-01-02) before task-a1 (2025-01-01)
        # Then P2: task-a3
        result_ids = [r["id"] for r in result]
        assert result_ids == ["task-a2", "task-a1", "task-a3"]


class TestFocusModeOrderPreference:
    """Test OrderPreference.FOCUS mode - single epic at a time.

    FOCUS mode returns only issues from the first/top epic group,
    ensuring the orchestrator works on one epic at a time.
    """

    def test_focus_mode_returns_only_first_epic(self) -> None:
        """FOCUS mode should return only issues from the top epic group."""

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

        result = IssueManager.sort_issues(
            issues,
            focus=True,
            include_wip=False,
            order_preference=OrderPreference.FOCUS,
        )

        # epic-a has min_priority=1, should be first group
        # FOCUS returns only epic-a issues
        result_ids = [r["id"] for r in result]
        assert result_ids == ["task-a2", "task-a1"]
        # epic-b's task-b1 is excluded
        assert "task-b1" not in result_ids

    def test_focus_mode_with_orphans_as_first_group(self) -> None:
        """FOCUS mode should work when orphans are the top group."""

        issues = [
            {
                "id": "orphan-1",
                "priority": 0,
                "status": "open",
                "parent_epic": None,
                "updated_at": "2025-01-01T10:00:00Z",
            },
            {
                "id": "task-a1",
                "priority": 1,
                "status": "open",
                "parent_epic": "epic-a",
                "updated_at": "2025-01-01T10:00:00Z",
            },
        ]

        result = IssueManager.sort_issues(
            issues,
            focus=True,
            include_wip=False,
            order_preference=OrderPreference.FOCUS,
        )

        # Orphan group has P0, epic-a has P1 - orphans first
        result_ids = [r["id"] for r in result]
        assert result_ids == ["orphan-1"]

    def test_focus_mode_with_wip_prioritization(self) -> None:
        """FOCUS mode should ignore include_wip for ordering."""

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
                "priority": 2,
                "status": "in_progress",
                "parent_epic": "epic-a",
                "updated_at": "2025-01-01T10:00:00Z",
            },
            {
                "id": "task-b1",
                "priority": 0,
                "status": "open",
                "parent_epic": "epic-b",
                "updated_at": "2025-01-01T10:00:00Z",
            },
        ]

        result = IssueManager.sort_issues(
            issues,
            focus=True,
            include_wip=True,
            order_preference=OrderPreference.FOCUS,
        )

        result_ids = [r["id"] for r in result]
        # epic-b is top group (P0), only task-b1 returned
        assert result_ids == ["task-b1"]


class TestIssePriorityModeOrderPreference:
    """Test OrderPreference.ISSUE_PRIORITY mode - global priority ordering."""

    def test_issue_priority_mode_ignores_epics(self) -> None:
        """ISSUE_PRIORITY should sort globally by priority, ignoring epics."""

        issues = [
            {
                "id": "task-a1",
                "priority": 3,
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
                "priority": 2,
                "status": "open",
                "parent_epic": "epic-a",
                "updated_at": "2025-01-01T10:00:00Z",
            },
        ]

        result = IssueManager.sort_issues(
            issues,
            focus=False,
            include_wip=False,
            order_preference=OrderPreference.ISSUE_PRIORITY,
        )

        # Sorted by priority only: P1, P2, P3
        result_ids = [r["id"] for r in result]
        assert result_ids == ["task-b1", "task-a2", "task-a1"]
