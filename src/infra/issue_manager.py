"""IssueManager: Pure domain logic for issue sorting and filtering.

This module provides stateless, testable functions for manipulating issue data.
Unlike BeadsClient (which handles I/O with the bd CLI), IssueManager contains
only pure transformations that can be tested without mocking subprocess calls.

Design note: All public methods are static or class methods to emphasize their
pure, functional nature. They operate on issue dicts and return new collections
without mutating inputs.
"""

from __future__ import annotations

from src.core.models import OrderPreference


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
    def filter_wip_issues(issues: list[dict[str, object]]) -> list[dict[str, object]]:
        """Filter out in_progress issues.

        Used when include_wip=False to exclude WIP issues from the result set.
        Since `bd ready` returns both open and in_progress issues, this filter
        ensures only open issues are returned when --resume is not passed.

        Args:
            issues: List of issue dicts (may include open + in_progress).

        Returns:
            List with in_progress issues removed.
        """
        return [issue for issue in issues if issue.get("status") != "in_progress"]

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
        only_ids: list[str] | None,
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
            epic_children: If provided, only include issues in this set.
            only_ids: If provided, only include issues in this list.

        Returns:
            Filtered list of issues.
        """
        # Convert only_ids list to set for O(1) membership lookup
        only_id_set = set(only_ids) if only_ids is not None else None

        return [
            i
            for i in issues
            if str(i["id"]) not in exclude_ids
            and i.get("issue_type") != "epic"
            and (epic_children is None or str(i["id"]) in epic_children)
            and (only_id_set is None or str(i["id"]) in only_id_set)
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
    def parse_priority(prio: object) -> int:
        """Parse priority value to int, handling both int and "P1" string formats.

        Args:
            prio: Priority value (int, string like "1" or "P1", or None).

        Returns:
            Integer priority, defaulting to 0 for unparseable values.
        """
        if prio is None:
            return 0
        try:
            prio_str = str(prio)
            if prio_str.upper().startswith("P"):
                prio_str = prio_str[1:]
            return int(prio_str)
        except (ValueError, IndexError):
            return 0

    @staticmethod
    def sort_by_epic_groups(
        issues: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Sort issues by epic groups for focus mode.

        Groups issues by parent_epic field, then sorts:
        1. Groups by (epic_priority, max_updated DESC)
           - Uses epic_priority field if available (the epic's own priority)
           - Falls back to min_priority of issues in group if epic_priority is None
        2. Within groups by (priority, updated DESC)

        Orphan tasks (no parent epic) form a virtual group using min_priority.

        Note: Issues must have parent_epic field populated before calling.
        Issues should also have epic_priority field for proper epic-based ordering.

        Args:
            issues: List of issue dicts with parent_epic (and optionally epic_priority).

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
            return IssueManager.parse_priority(issue.get("priority"))

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

        def get_epic_priority(issue: dict[str, object]) -> int | None:
            """Extract epic_priority as int if available, or None."""
            prio = issue.get("epic_priority")
            if prio is None:
                return None
            parsed = IssueManager.parse_priority(prio)
            # parse_priority returns 0 for invalid, but we want None for epic_priority
            # to fall back to min_priority. Only return None if original was invalid.
            try:
                prio_str = str(prio)
                if prio_str.upper().startswith("P"):
                    prio_str = prio_str[1:]
                int(prio_str)  # Validate it's parseable
                return parsed
            except (ValueError, IndexError):
                return None

        # Compute group sort key: (epic_priority or min_priority, -max_updated)
        def group_sort_key(epic: str | None) -> tuple[int, str]:
            group_issues = groups[epic]
            # Use epic_priority if available from any issue in group
            # (all issues in same epic should have same epic_priority, but check all
            # in case of mixed enrichment)
            epic_prio: int | None = None
            if epic:
                for issue in group_issues:
                    epic_prio = get_epic_priority(issue)
                    if epic_prio is not None:
                        break
            if epic_prio is not None:
                effective_priority = epic_prio
            else:
                effective_priority = min(get_priority(i) for i in group_issues)
            max_updated = max(get_updated_at(i) for i in group_issues)
            return (effective_priority, IssueManager.negate_timestamp(max_updated))

        # Sort groups and flatten
        sorted_epics = sorted(groups.keys(), key=group_sort_key)
        return [issue for epic in sorted_epics for issue in groups[epic]]

    @staticmethod
    def sort_issues(
        issues: list[dict[str, object]],
        focus: bool,
        include_wip: bool,
        only_ids: list[str] | None = None,
        order_preference: OrderPreference = OrderPreference.EPIC_PRIORITY,
    ) -> list[dict[str, object]]:
        """Sort issues by order_preference (authoritative over focus flag).

        Args:
            issues: List of issue dicts to sort.
            focus: Legacy flag (ignored when order_preference is set explicitly).
            include_wip: Scope-only flag; in_progress issues are included upstream.
                Ordering is unaffected by include_wip.
            only_ids: Optional list of issue IDs for input order preservation.
            order_preference: Issue ordering (focus, epic-priority, issue-priority, or input).
                This is the authoritative source of truth for ordering.

        Returns:
            Sorted list of issue dicts.
        """
        if not issues:
            return issues

        # INPUT order: preserve user-specified order from only_ids
        # Note: include_wip is ignored for ordering to preserve exact user order
        if order_preference == OrderPreference.INPUT and only_ids:
            # Create index map for O(1) lookup
            id_order = {id_: idx for idx, id_ in enumerate(only_ids)}
            return sorted(
                issues,
                key=lambda i: id_order.get(str(i["id"]), len(only_ids)),
            )

        # order_preference is authoritative
        if order_preference == OrderPreference.ISSUE_PRIORITY:
            result = sorted(
                issues, key=lambda i: IssueManager.parse_priority(i.get("priority"))
            )
        elif order_preference == OrderPreference.FOCUS:
            # FOCUS: single-epic mode - return only issues from the top epic group
            sorted_all = IssueManager.sort_by_epic_groups(list(issues))
            if sorted_all:
                first_epic = sorted_all[0].get("parent_epic")
                result = [i for i in sorted_all if i.get("parent_epic") == first_epic]
            else:
                result = sorted_all
        else:
            # EPIC_PRIORITY (default): group by epic, return all
            result = IssueManager.sort_by_epic_groups(list(issues))

        # include_wip intentionally does not alter ordering; it only affects scope
        return result

    @staticmethod
    def find_missing_ids(
        only_ids: list[str] | None,
        issues: list[dict[str, object]],
        suppress_ids: set[str],
    ) -> set[str]:
        """Find IDs from only_ids that are not in issues.

        Args:
            only_ids: List of expected issue IDs (None means no filtering).
            issues: List of issue dicts to check against.
            suppress_ids: IDs to exclude from the missing set.

        Returns:
            Set of IDs that were expected but not found.
        """
        if not only_ids:
            return set()
        return set(only_ids) - {str(i["id"]) for i in issues} - suppress_ids

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
