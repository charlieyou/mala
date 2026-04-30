"""Unit tests for IssueManager.

Tests pure domain logic for issue sorting and filtering without subprocess mocking.
All tests use fixture data directly, demonstrating the testability benefits of
separating domain logic from I/O.
"""

from src.infra.issue_manager import IssueManager


class TestMergeWipIssues:
    """Tests for IssueManager.merge_wip_issues."""

    def test_adds_missing_wip_issues(self) -> None:
        """Should add WIP issues not already in base list."""
        base = [{"id": "a"}, {"id": "b"}]
        wip = [{"id": "b"}, {"id": "c"}]
        result = IssueManager.merge_wip_issues(base, wip)
        assert len(result) == 3
        assert [r["id"] for r in result] == ["a", "b", "c"]

    def test_empty_base_returns_all_wip(self) -> None:
        """With empty base, should return all WIP issues."""
        wip = [{"id": "x"}, {"id": "y"}]
        result = IssueManager.merge_wip_issues([], wip)
        assert [r["id"] for r in result] == ["x", "y"]

    def test_empty_wip_returns_base_unchanged(self) -> None:
        """With empty WIP, should return base unchanged."""
        base = [{"id": "a"}]
        result = IssueManager.merge_wip_issues(base, [])
        assert result == base

    def test_preserves_order(self) -> None:
        """Should preserve base order and append new WIP at end."""
        base = [{"id": "c"}, {"id": "a"}]
        wip = [{"id": "b"}, {"id": "d"}]
        result = IssueManager.merge_wip_issues(base, wip)
        assert [r["id"] for r in result] == ["c", "a", "b", "d"]


class TestApplyFilters:
    """Tests for IssueManager.apply_filters."""

    def test_excludes_specified_ids(self) -> None:
        """Should exclude issues in exclude_ids."""
        issues = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        result = IssueManager.apply_filters(issues, {"b"}, None, None)
        assert [r["id"] for r in result] == ["a", "c"]

    def test_excludes_epics(self) -> None:
        """Should exclude issues with issue_type=epic."""
        issues = [{"id": "a"}, {"id": "b", "issue_type": "epic"}]
        result = IssueManager.apply_filters(issues, set(), None, None)
        assert [r["id"] for r in result] == ["a"]

    def test_filters_by_epic_children(self) -> None:
        """Should include only issues in epic_children set."""
        issues = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        result = IssueManager.apply_filters(issues, set(), {"a", "c"}, None)
        assert [r["id"] for r in result] == ["a", "c"]

    def test_filters_by_only_ids(self) -> None:
        """Should include only issues in only_ids list."""
        issues = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        result = IssueManager.apply_filters(issues, set(), None, ["b"])
        assert [r["id"] for r in result] == ["b"]

    def test_combines_all_filters(self) -> None:
        """Should apply all filters together."""
        issues = [
            {"id": "a"},
            {"id": "b", "issue_type": "epic"},
            {"id": "c"},
            {"id": "d"},
        ]
        # Exclude "d", only allow epic children {"a", "c", "d"}, only_ids ["a", "d"]
        result = IssueManager.apply_filters(issues, {"d"}, {"a", "c", "d"}, ["a", "d"])
        # "a" passes all, "b" is epic, "c" not in only_ids, "d" excluded
        assert [r["id"] for r in result] == ["a"]

    def test_empty_issues_returns_empty(self) -> None:
        """Should return empty list for empty input."""
        result = IssueManager.apply_filters([], {"a"}, {"b"}, ["c"])
        assert result == []


class TestFilterBlockedWip:
    """Tests for IssueManager.filter_blocked_wip."""

    def test_keeps_open_issues(self) -> None:
        issues = [
            {"id": "open-1", "status": "open"},
            {"id": "open-2", "status": "open", "blocked_by": ["x"]},
        ]
        result = IssueManager.filter_blocked_wip(issues)
        assert [i["id"] for i in result] == ["open-1", "open-2"]

    def test_excludes_blocked_in_progress(self) -> None:
        issues = [
            {"id": "wip-ok", "status": "in_progress", "blocked_by": None},
            {"id": "wip-blocked", "status": "in_progress", "blocked_by": ["x"]},
        ]
        result = IssueManager.filter_blocked_wip(issues)
        assert [i["id"] for i in result] == ["wip-ok"]


class TestFilterWipIssues:
    """Tests for IssueManager.filter_wip_issues."""

    def test_filters_in_progress_issues(self) -> None:
        """filter_wip_issues removes in_progress issues."""
        issues = [
            {"id": "open-1", "status": "open"},
            {"id": "wip-1", "status": "in_progress"},
            {"id": "open-2", "status": "open"},
        ]
        result = IssueManager.filter_wip_issues(issues)
        assert [i["id"] for i in result] == ["open-1", "open-2"]

    def test_keeps_open_issues(self) -> None:
        """filter_wip_issues keeps open issues."""
        issues = [
            {"id": "open-1", "status": "open"},
            {"id": "open-2", "status": "open"},
        ]
        result = IssueManager.filter_wip_issues(issues)
        assert [i["id"] for i in result] == ["open-1", "open-2"]

    def test_empty_list(self) -> None:
        """filter_wip_issues handles empty list."""
        result = IssueManager.filter_wip_issues([])
        assert result == []

    def test_all_in_progress(self) -> None:
        """filter_wip_issues removes all in_progress issues."""
        issues = [
            {"id": "wip-1", "status": "in_progress"},
            {"id": "wip-2", "status": "in_progress"},
        ]
        result = IssueManager.filter_wip_issues(issues)
        assert result == []


class TestFilterBlockedEpics:
    """Tests for IssueManager.filter_blocked_epics."""

    def test_no_blocked_epics_returns_all(self) -> None:
        issues = [
            {"id": "a", "parent_epic": "epic-1"},
            {"id": "b", "parent_epic": None},
        ]
        result = IssueManager.filter_blocked_epics(issues, set())
        assert result == issues

    def test_excludes_issues_under_blocked_epics(self) -> None:
        issues = [
            {"id": "a", "parent_epic": "epic-1"},
            {"id": "b", "parent_epic": "epic-2"},
            {"id": "c", "parent_epic": None},
        ]
        result = IssueManager.filter_blocked_epics(issues, {"epic-2"})
        assert [i["id"] for i in result] == ["a", "c"]


class TestNegateTimestamp:
    """Tests for IssueManager.negate_timestamp."""

    def test_negates_timestamp_string(self) -> None:
        """Should negate each character for descending sort."""
        ts = "2025-01-15"
        result = IssueManager.negate_timestamp(ts)
        # Each char should be chr(255 - ord(c))
        expected = "".join(chr(255 - ord(c)) for c in ts)
        assert result == expected

    def test_empty_string_returns_xff(self) -> None:
        """Empty string should return \\xff (sorts last)."""
        assert IssueManager.negate_timestamp("") == "\xff"

    def test_none_returns_xff(self) -> None:
        """None should return \\xff (sorts last)."""
        assert IssueManager.negate_timestamp(None) == "\xff"

    def test_non_string_returns_xff(self) -> None:
        """Non-string should return \\xff (sorts last)."""
        assert IssueManager.negate_timestamp(12345) == "\xff"

    def test_later_timestamps_sort_earlier(self) -> None:
        """Later timestamps should sort earlier after negation."""
        ts1 = "2025-01-01T10:00:00Z"
        ts2 = "2025-01-02T10:00:00Z"
        neg1 = IssueManager.negate_timestamp(ts1)
        neg2 = IssueManager.negate_timestamp(ts2)
        # ts2 is later, so neg2 should be "smaller" (sort first)
        assert neg2 < neg1


class TestSortByEpicGroups:
    """Tests for IssueManager.sort_by_epic_groups."""

    def test_empty_list_returns_empty(self) -> None:
        """Should return empty list for empty input."""
        result = IssueManager.sort_by_epic_groups([])
        assert result == []

    def test_groups_by_parent_epic(self) -> None:
        """Should group issues by parent_epic."""
        issues = [
            {
                "id": "a",
                "priority": 1,
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
                "priority": 1,
                "parent_epic": "epic-1",
                "updated_at": "2025-01-01",
            },
        ]
        result = IssueManager.sort_by_epic_groups(issues)
        # Both groups have same priority, so order by max_updated (both same)
        # Epic-1 issues should stay grouped
        epic1_indices = [
            i for i, r in enumerate(result) if r["parent_epic"] == "epic-1"
        ]
        assert epic1_indices == [0, 1] or epic1_indices == [1, 2]

    def test_sorts_groups_by_epic_priority(self) -> None:
        """Groups should be sorted by epic_priority (the epic's own priority)."""
        issues = [
            {
                "id": "a",
                "priority": 1,  # Task has P1
                "parent_epic": "epic-1",
                "epic_priority": 2,  # But epic has P2
                "updated_at": "2025-01-01",
            },
            {
                "id": "b",
                "priority": 3,  # Task has P3
                "parent_epic": "epic-2",
                "epic_priority": 1,  # But epic has P1
                "updated_at": "2025-01-01",
            },
        ]
        result = IssueManager.sort_by_epic_groups(issues)
        # epic-2 has lower epic_priority (1), so comes first even though task has P3
        assert result[0]["id"] == "b"
        assert result[1]["id"] == "a"

    def test_sorts_groups_by_epic_priority_string_format(self) -> None:
        """Groups should handle epic_priority as 'P1' string format."""
        issues = [
            {
                "id": "a",
                "priority": 1,
                "parent_epic": "epic-1",
                "epic_priority": "P2",  # String format with P prefix
                "updated_at": "2025-01-01",
            },
            {
                "id": "b",
                "priority": 3,
                "parent_epic": "epic-2",
                "epic_priority": "P1",  # String format with P prefix
                "updated_at": "2025-01-01",
            },
        ]
        result = IssueManager.sort_by_epic_groups(issues)
        # epic-2 has lower epic_priority (P1 -> 1), so comes first
        assert result[0]["id"] == "b"
        assert result[1]["id"] == "a"

    def test_sorts_groups_by_min_priority_when_no_epic_priority(self) -> None:
        """Groups fall back to min_priority when epic_priority is not set."""
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
        ]
        result = IssueManager.sort_by_epic_groups(issues)
        # epic-2 has lower min priority (1), so comes first
        assert result[0]["id"] == "b"
        assert result[1]["id"] == "a"

    def test_sorts_within_group_by_priority(self) -> None:
        """Issues within group should be sorted by priority."""
        issues = [
            {
                "id": "a",
                "priority": 3,
                "parent_epic": "epic-1",
                "updated_at": "2025-01-01",
            },
            {
                "id": "b",
                "priority": 1,
                "parent_epic": "epic-1",
                "updated_at": "2025-01-01",
            },
            {
                "id": "c",
                "priority": 2,
                "parent_epic": "epic-1",
                "updated_at": "2025-01-01",
            },
        ]
        result = IssueManager.sort_by_epic_groups(issues)
        assert [r["id"] for r in result] == ["b", "c", "a"]

    def test_sorts_within_group_by_updated_desc_for_same_priority(self) -> None:
        """Issues with same priority should be sorted by updated DESC."""
        issues = [
            {
                "id": "a",
                "priority": 1,
                "parent_epic": "epic-1",
                "updated_at": "2025-01-01T10:00:00Z",
            },
            {
                "id": "b",
                "priority": 1,
                "parent_epic": "epic-1",
                "updated_at": "2025-01-02T10:00:00Z",
            },
        ]
        result = IssueManager.sort_by_epic_groups(issues)
        # "b" has later updated_at, so comes first
        assert result[0]["id"] == "b"
        assert result[1]["id"] == "a"

    def test_orphan_tasks_form_virtual_group(self) -> None:
        """Tasks without parent_epic should form a virtual group."""
        issues = [
            {"id": "a", "priority": 1, "parent_epic": None, "updated_at": "2025-01-01"},
            {"id": "b", "priority": 2, "parent_epic": None, "updated_at": "2025-01-01"},
        ]
        result = IssueManager.sort_by_epic_groups(issues)
        # Both in same group, sorted by priority
        assert [r["id"] for r in result] == ["a", "b"]

    def test_handles_missing_updated_at(self) -> None:
        """Should handle issues without updated_at field."""
        issues = [
            {"id": "a", "priority": 1, "parent_epic": "epic-1"},
            {
                "id": "b",
                "priority": 1,
                "parent_epic": "epic-1",
                "updated_at": "2025-01-01",
            },
        ]
        result = IssueManager.sort_by_epic_groups(issues)
        # Both have same priority, so order is by negated updated_at (ascending)
        # "a" has no updated_at -> negate returns "\xff" (chr 255)
        # "b" has "2025-01-01" -> negate returns chr(255-ord(c)) for each char
        # First char '2' -> chr(255-50) = chr(205)
        # Since 255 > 205, "\xff" sorts AFTER negated timestamp
        # So "b" (has updated_at) sorts BEFORE "a" (missing updated_at)
        assert result[0]["id"] == "b"
        assert result[1]["id"] == "a"


class TestSortIssues:
    """Tests for IssueManager.sort_issues."""

    def test_empty_list_returns_empty(self) -> None:
        """Should return empty list for empty input."""
        result = IssueManager.sort_issues([], focus=True, include_wip=True)
        assert result == []

    def test_sorts_by_priority_when_focus_false(self) -> None:
        """With focus=False, should sort by priority only."""
        issues = [
            {"id": "a", "priority": 3},
            {"id": "b", "priority": 1},
            {"id": "c", "priority": 2},
        ]
        result = IssueManager.sort_issues(issues, focus=False, include_wip=False)
        assert [r["id"] for r in result] == ["b", "c", "a"]

    def test_include_wip_when_enabled(self) -> None:
        """With include_wip=True, ordering should remain priority-based."""
        issues = [
            {"id": "a", "priority": 1, "status": "open"},
            {"id": "b", "priority": 2, "status": "in_progress"},
            {"id": "c", "priority": 3, "status": "open"},
        ]
        result = IssueManager.sort_issues(issues, focus=False, include_wip=True)
        assert [r["id"] for r in result] == ["a", "b", "c"]

    def test_uses_epic_groups_when_focus_true(self) -> None:
        """With focus=True, should group by parent epic."""
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
        result = IssueManager.sort_issues(issues, focus=True, include_wip=False)
        # epic-2 has lower min priority, so comes first
        assert result[0]["id"] == "b"

    def test_combines_focus_and_wip_priority(self) -> None:
        """Should apply focus grouping without reordering WIP status."""
        issues = [
            {
                "id": "a",
                "priority": 1,
                "parent_epic": "epic-1",
                "status": "open",
                "updated_at": "2025-01-01",
            },
            {
                "id": "b",
                "priority": 2,
                "parent_epic": "epic-1",
                "status": "in_progress",
                "updated_at": "2025-01-01",
            },
        ]
        result = IssueManager.sort_issues(issues, focus=True, include_wip=True)
        assert [r["id"] for r in result] == ["a", "b"]


class TestFindMissingIds:
    """Tests for IssueManager.find_missing_ids."""

    def test_returns_missing_ids(self) -> None:
        """Should return IDs from only_ids not found in issues."""
        issues = [{"id": "a"}, {"id": "b"}]
        result = IssueManager.find_missing_ids(["a", "c", "d"], issues, set())
        assert result == {"c", "d"}

    def test_respects_suppress_ids(self) -> None:
        """Should not include suppressed IDs in result."""
        issues = [{"id": "a"}]
        result = IssueManager.find_missing_ids(["a", "b", "c"], issues, {"b"})
        assert result == {"c"}

    def test_returns_empty_when_all_found(self) -> None:
        """Should return empty set when all IDs are found."""
        issues = [{"id": "a"}, {"id": "b"}]
        result = IssueManager.find_missing_ids(["a", "b"], issues, set())
        assert result == set()

    def test_returns_empty_when_only_ids_none(self) -> None:
        """Should return empty set when only_ids is None."""
        result = IssueManager.find_missing_ids(None, [], set())
        assert result == set()

    def test_returns_empty_when_only_ids_empty(self) -> None:
        """Should return empty set when only_ids is empty."""
        result = IssueManager.find_missing_ids([], [{"id": "a"}], set())
        assert result == set()


class TestFilterOrphansOnly:
    """Tests for IssueManager.filter_orphans_only."""

    def test_returns_only_orphan_issues(self) -> None:
        """Should return only issues with no parent epic."""
        issues = [
            {"id": "a", "parent_epic": "epic-1"},
            {"id": "b", "parent_epic": None},
            {"id": "c", "parent_epic": "epic-2"},
        ]
        result = IssueManager.filter_orphans_only(issues)
        assert [r["id"] for r in result] == ["b"]

    def test_returns_empty_when_all_have_parent(self) -> None:
        """Should return empty list when all issues have parent epic."""
        issues = [
            {"id": "a", "parent_epic": "epic-1"},
            {"id": "b", "parent_epic": "epic-2"},
        ]
        result = IssueManager.filter_orphans_only(issues)
        assert result == []

    def test_returns_all_when_none_have_parent(self) -> None:
        """Should return all issues when none have parent epic."""
        issues = [
            {"id": "a", "parent_epic": None},
            {"id": "b", "parent_epic": None},
        ]
        result = IssueManager.filter_orphans_only(issues)
        assert [r["id"] for r in result] == ["a", "b"]

    def test_empty_list_returns_empty(self) -> None:
        """Should return empty list for empty input."""
        result = IssueManager.filter_orphans_only([])
        assert result == []

    def test_treats_empty_string_as_no_parent(self) -> None:
        """Should treat empty string parent_epic as no parent."""
        issues = [
            {"id": "a", "parent_epic": ""},
            {"id": "b", "parent_epic": "epic-1"},
        ]
        result = IssueManager.filter_orphans_only(issues)
        assert [r["id"] for r in result] == ["a"]

    def test_handles_missing_parent_epic_field(self) -> None:
        """Should treat missing parent_epic field as no parent."""
        issues = [
            {"id": "a"},
            {"id": "b", "parent_epic": "epic-1"},
        ]
        result = IssueManager.filter_orphans_only(issues)
        assert [r["id"] for r in result] == ["a"]
