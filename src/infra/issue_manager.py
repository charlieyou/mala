"""IssueManager: Pure domain logic for issue sorting and filtering.

This module provides stateless, testable functions for manipulating issue data.
Unlike BeadsClient (which handles I/O with the bd CLI), IssueManager contains
only pure transformations that can be tested without mocking subprocess calls.

Design note: All public methods are static or class methods to emphasize their
pure, functional nature. They operate on issue dicts and return new collections
without mutating inputs.
"""

from __future__ import annotations


class IssueManager:
    """Pure domain logic for issue sorting and filtering.

    All methods are static and operate on issue data without any I/O.
    This separation allows:
    - Direct unit testing with fixture data (no subprocess mocking)
    - Clear boundary between I/O (BeadsClient) and domain logic (IssueManager)
    - Reuse of sorting/filtering logic in different contexts
    """

    @staticmethod
    def merge_wip_issues(
        base_issues: list[dict[str, object]], wip_issues: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Merge WIP issues into base list, avoiding duplicates.

        Args:
            base_issues: List of base issues (typically from bd ready).
            wip_issues: List of in-progress issues to merge.

        Returns:
            Combined list with WIP issues appended if not already in base.
        """
        ready_ids: set[str] = {str(i["id"]) for i in base_issues}
        return base_issues + [w for w in wip_issues if str(w["id"]) not in ready_ids]

    @staticmethod
    def filter_blocked_wip(issues: list[dict[str, object]]) -> list[dict[str, object]]:
        """Filter out blocked in_progress issues.

        A WIP issue is considered blocked when it has a truthy blocked_by value.

        Args:
            issues: List of issue dicts (may include open + in_progress).

        Returns:
            List with blocked in_progress issues removed.
        """
        return [
            issue
            for issue in issues
            if not (issue.get("status") == "in_progress" and issue.get("blocked_by"))
        ]

    @staticmethod
    def filter_blocked_epics(
        issues: list[dict[str, object]], blocked_epics: set[str]
    ) -> list[dict[str, object]]:
        """Filter out issues whose parent epic is blocked.

        Args:
            issues: List of issue dicts with parent_epic field populated.
            blocked_epics: Set of blocked epic IDs.

        Returns:
            List with issues under blocked epics removed.
        """
        if not blocked_epics:
            return issues
        return [
            issue
            for issue in issues
            if str(issue.get("parent_epic")) not in blocked_epics
        ]

    @staticmethod
    def apply_filters(
        issues: list[dict[str, object]],
        exclude_ids: set[str],
        epic_children: set[str] | None,
        only_ids: set[str] | None,
    ) -> list[dict[str, object]]:
        """Apply filtering rules to issues.

        Filters out:
        - Issues in exclude_ids
        - Epics (issue_type == "epic")
        - Issues not in epic_children (if specified)
        - Issues not in only_ids (if specified)

        Args:
            issues: List of issue dicts to filter.
            exclude_ids: Set of issue IDs to exclude.
            epic_children: If set, only include issues in this set.
            only_ids: If set, only include issues in this set.

        Returns:
            Filtered list of issues.
        """
        return [
            i
            for i in issues
            if str(i["id"]) not in exclude_ids
            and i.get("issue_type") != "epic"
            and (epic_children is None or str(i["id"]) in epic_children)
            and (only_ids is None or str(i["id"]) in only_ids)
        ]

    @staticmethod
    def negate_timestamp(timestamp: object) -> str:
        """Negate a timestamp string for descending sort.

        Converts each character to its complement relative to chr(255)
        so that lexicographic sort becomes descending.

        Args:
            timestamp: ISO timestamp string or empty string.

        Returns:
            Negated string for descending sort.
        """
        if not timestamp or not isinstance(timestamp, str):
            # Empty/missing timestamps sort last (after all real timestamps)
            # Use \xff (ordinal 255) since negated timestamp chars are in [198, 207]
            return "\xff"
        # Negate each character: higher chars become lower
        # This works for ISO timestamps like "2025-01-15T10:30:00Z"
        return "".join(chr(255 - ord(c)) for c in timestamp)

    @staticmethod
    def sort_by_epic_groups(
        issues: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Sort issues by epic groups for focus mode.

        Groups issues by parent_epic field, then sorts:
        1. Groups by (min_priority, max_updated DESC)
        2. Within groups by (priority, updated DESC)

        Orphan tasks (no parent epic) form a virtual group with the same rules.

        Note: Issues must have parent_epic field populated before calling.

        Args:
            issues: List of issue dicts with parent_epic field.

        Returns:
            Sorted list of issue dicts.
        """
        if not issues:
            return issues

        # Group issues by parent epic (None for orphans)
        groups: dict[str | None, list[dict[str, object]]] = {}
        for issue in issues:
            key: str | None = (
                str(issue.get("parent_epic")) if issue.get("parent_epic") else None
            )
            groups.setdefault(key, []).append(issue)

        def get_priority(issue: dict[str, object]) -> int:
            """Extract priority as int, defaulting to 0."""
            prio = issue.get("priority")
            if prio is None:
                return 0
            return int(str(prio))

        def get_updated_at(issue: dict[str, object]) -> str:
            """Extract updated_at as string, defaulting to empty."""
            val = issue.get("updated_at")
            return str(val) if val is not None else ""

        # Sort within each group by (priority, updated DESC)
        for group_issues in groups.values():
            group_issues.sort(
                key=lambda i: (
                    get_priority(i),
                    IssueManager.negate_timestamp(get_updated_at(i)),
                )
            )

        # Compute group sort key: (min_priority, -max_updated)
        def group_sort_key(epic: str | None) -> tuple[int, str]:
            group_issues = groups[epic]
            min_priority = min(get_priority(i) for i in group_issues)
            max_updated = max(get_updated_at(i) for i in group_issues)
            return (min_priority, IssueManager.negate_timestamp(max_updated))

        # Sort groups and flatten
        sorted_epics = sorted(groups.keys(), key=group_sort_key)
        return [issue for epic in sorted_epics for issue in groups[epic]]

    @staticmethod
    def sort_issues(
        issues: list[dict[str, object]], focus: bool, prioritize_wip: bool
    ) -> list[dict[str, object]]:
        """Sort issues by focus mode vs priority.

        Args:
            issues: List of issue dicts to sort.
            focus: If True, group by parent epic and sort by epic groups.
            prioritize_wip: If True, put in_progress issues first.

        Returns:
            Sorted list of issue dicts.
        """
        if not issues:
            return issues

        result = (
            IssueManager.sort_by_epic_groups(list(issues))
            if focus
            else sorted(issues, key=lambda i: i.get("priority") or 0)
        )

        if prioritize_wip:
            return sorted(
                result, key=lambda i: 0 if i.get("status") == "in_progress" else 1
            )
        return result

    @staticmethod
    def find_missing_ids(
        only_ids: set[str] | None,
        issues: list[dict[str, object]],
        suppress_ids: set[str],
    ) -> set[str]:
        """Find IDs from only_ids that are not in issues.

        Args:
            only_ids: Set of expected issue IDs (None means no filtering).
            issues: List of issue dicts to check against.
            suppress_ids: IDs to exclude from the missing set.

        Returns:
            Set of IDs that were expected but not found.
        """
        if not only_ids:
            return set()
        return only_ids - {str(i["id"]) for i in issues} - suppress_ids

    @staticmethod
    def filter_orphans_only(
        issues: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Filter to only issues with no parent epic.

        Args:
            issues: List of issue dicts with parent_epic field populated.

        Returns:
            List containing only issues where parent_epic is None/empty.
        """
        return [issue for issue in issues if not issue.get("parent_epic")]
